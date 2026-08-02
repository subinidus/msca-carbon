"""
================================================================================
spatial_grid_dataset.py -- Patch dataset for MS-CA (single-task regression)
================================================================================

Patch-based dataset for the MS-CA spatial tensor ``.npz`` files. Strictly
single-task, pure regression: ``label_cls`` is ignored even when present.

Channel layout (fixed)
----------------------
    Stream A (3 channels): NO2, SO2, CO
    Stream B (4 channels): Nightlight, Urban Fraction, Power Plant, Fossil Capacity

WHAT IS NEW IN THIS VERSION
---------------------------
1. FEATURE STANDARDIZATION (``apply_normalization``). The raw channels span
   about eight orders of magnitude -- NO2 is ~1e-5 mol/m^2 while
   fossil_capacity_mw is ~1e3. Feeding that straight into the patch-embedding
   Conv2d makes the network respond almost exclusively to the largest-scale
   channel. Stats MUST be computed on TRAIN CELLS ONLY and then applied to all
   splits; computing them over the whole dataset leaks val/test information.

2. TARGET TRANSFORM (``apply_target_transform``). Measured label_reg on the
   real data:

       n=617,652  min=0.0  max=94,549.6
       mean=152.33  median=26.77  std=1687.02
       skew=40.03  kurtosis=1723.52  max/median=3531.9
       p99=1205.8  p99.9=1734.8   <- max is 54x the p99.9
       zeros=0.75%

   With skew 40, an L1 objective on the raw scale learns the conditional
   MEDIAN and systematically under-predicts every large emitter -- exactly the
   cells a carbon-hotspot study cares about. ``log1p`` pulls the skew down to
   a workable range and handles the 0.75% zeros without an epsilon hack.
   Metrics are still reported in BOTH spaces by ``evaluation.py``.

3. Invalid (mask == 0) cells are re-zeroed after normalization so that they --
   and the zero padding added by ``_crop_with_padding`` -- both sit at the
   post-standardization channel mean rather than at some arbitrary offset.

4. ``build_splits`` is kept for backwards compatibility but is DEPRECATED: it
   splits per-sample, so the same cell lands in train and test at different
   months. Use ``splits_v2.build_splits_v2`` instead.
================================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split

# ``label_cls`` is deliberately NOT required -- pure regression pipeline.
REQUIRED_ARRAYS = ["stream_a", "stream_b", "label_reg", "mask"]

STREAM_A_CHANNELS = 3  # NO2, SO2, CO
STREAM_B_CHANNELS = 4  # Nightlight, Urban Fraction, Power Plant, Fossil Capacity

VALID_TARGET_TRANSFORMS = ("none", "log1p")

# Ablation modes for the spatial-context debate:
#   'full'        : normal 16x16 patch (default).
#   'center_only' : every cell except the centre is zeroed. Architecture stays
#                   identical, spatial context is removed. THIS is the direct
#                   test of the "cells are independent" hypothesis -- if it
#                   were true, full and center_only would score the same.
#   'shuffle'     : the centre cell stays in place, the other 255 positions
#                   are randomly permuted (same permutation across streams and
#                   mask). Neighbours remain visible as an unordered bag, so
#                   this separates "arrangement matters" from "the mere set of
#                   neighbour values matters". Applied at BOTH train and eval
#                   (a train-only shuffle would measure distribution shift,
#                   not spatial information).
VALID_CONTEXT_MODES = ("full", "center_only", "shuffle")


def load_tensor_index(tensor_index_csv: str | Path) -> list[str]:
    """Read a tensor index CSV and return tensor paths in row order."""
    index_path = Path(tensor_index_csv)
    if not index_path.exists():
        raise FileNotFoundError(f"Tensor index CSV not found: {index_path}")

    df = pd.read_csv(index_path)
    if "tensor_path" not in df.columns:
        raise ValueError(
            f"Tensor index CSV must contain 'tensor_path'. "
            f"Available columns: {list(df.columns)}"
        )
    return df["tensor_path"].astype(str).tolist()


class SpatialGridDataset(Dataset):
    """Patch-based dataset for MS-CA spatial tensor ``.npz`` files (regression)."""

    def __init__(
        self,
        npz_paths: list[str | Path] | None = None,
        tensor_index_csv: str | Path | None = None,
        crop_size: int = 16,
        target: str = "center",
        transform: Any = None,
    ) -> None:
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got: {crop_size}")
        if target != "center":
            raise ValueError("Only target='center' is currently supported.")
        if npz_paths is None and tensor_index_csv is None:
            raise ValueError("Provide either npz_paths or tensor_index_csv.")
        if npz_paths is not None and tensor_index_csv is not None:
            raise ValueError("Provide only one of npz_paths or tensor_index_csv.")

        self.crop_size = crop_size
        self.target = target
        self.transform = transform

        # State flags so the transforms can never be applied twice.
        self.feature_stats: dict[str, list[float]] | None = None
        self.target_transform: str = "none"
        self.context_mode: str = "full"
        self._ctx_rng = np.random.default_rng(0)

        path_values = (
            npz_paths if npz_paths is not None else load_tensor_index(tensor_index_csv)
        )
        self.npz_paths = [Path(path) for path in path_values]
        self.samples: list[tuple[int, int, int]] = []
        self.monthly_tensors: list[dict[str, Any]] = []

        for file_idx, npz_path in enumerate(self.npz_paths):
            tensor_dict = self._load_npz(npz_path)
            self.monthly_tensors.append(tensor_dict)

            valid_rows, valid_cols = np.where(tensor_dict["mask"] == 1)
            for row, col in zip(valid_rows.tolist(), valid_cols.tolist(), strict=True):
                self.samples.append((file_idx, row, col))

        if not self.samples:
            raise ValueError("No valid mask cells found in the provided tensor files.")

    # ------------------------------------------------------------------ #
    # Standard Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, row, col = self.samples[idx]
        tensor_dict = self.monthly_tensors[file_idx]

        stream_a_patch = self._crop_with_padding(tensor_dict["stream_a"], row, col)
        stream_b_patch = self._crop_with_padding(tensor_dict["stream_b"], row, col)
        mask_patch = self._crop_with_padding(tensor_dict["mask"], row, col)

        if self.context_mode != "full":
            stream_a_patch, stream_b_patch, mask_patch = self._apply_context(
                stream_a_patch, stream_b_patch, mask_patch
            )

        sample = {
            "stream_a": torch.as_tensor(stream_a_patch, dtype=torch.float32),
            "stream_b": torch.as_tensor(stream_b_patch, dtype=torch.float32),
            "label_reg": torch.as_tensor(
                tensor_dict["label_reg"][row, col], dtype=torch.float32
            ),
            "mask": torch.as_tensor(mask_patch, dtype=torch.float32),
            "date": tensor_dict["date"],
            "row": row,
            "col": col,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    # ------------------------------------------------------------------ #
    # NEW: target transform
    # ------------------------------------------------------------------ #
    def apply_target_transform(self, name: str) -> None:
        """Transform label_reg in place. Call BEFORE building splits.

        'log1p' is strongly recommended for this data (skew = 40.03).
        """
        if name not in VALID_TARGET_TRANSFORMS:
            raise ValueError(
                f"target_transform must be one of {VALID_TARGET_TRANSFORMS}, "
                f"got {name!r}"
            )
        if self.target_transform != "none":
            raise RuntimeError(
                f"A target transform ({self.target_transform!r}) is already applied."
            )
        if name == "none":
            return

        if name == "log1p":
            for tensor_dict in self.monthly_tensors:
                clipped = np.maximum(tensor_dict["label_reg"], 0.0)
                tensor_dict["label_reg"] = np.log1p(clipped).astype(np.float32)

        self.target_transform = name
        print(f"[Dataset] Applied target transform: {name}")

    # ------------------------------------------------------------------ #
    # NEW: context ablation
    # ------------------------------------------------------------------ #
    def set_context_mode(self, mode: str) -> None:
        """Select the spatial-context ablation applied at read time.

        Because the transform runs inside ``__getitem__``, it affects every
        loader (train, val, test, inference) equally -- which is what makes
        the ablation a coherent condition rather than a train/test mismatch.
        In 'center_only', the zeros equal the post-standardization channel
        mean when normalization is on, i.e. the same value as the padding.
        """
        if mode not in VALID_CONTEXT_MODES:
            raise ValueError(
                f"context_mode must be one of {VALID_CONTEXT_MODES}, got {mode!r}"
            )
        self.context_mode = mode
        if mode != "full":
            print(f"[Dataset] context_mode={mode} (spatial-context ablation)")

    def _apply_context(self, a: np.ndarray, b: np.ndarray, m: np.ndarray):
        ci = self.crop_size // 2
        if self.context_mode == "center_only":
            a2 = np.zeros_like(a)
            b2 = np.zeros_like(b)
            m2 = np.zeros_like(m)
            a2[:, ci, ci] = a[:, ci, ci]
            b2[:, ci, ci] = b[:, ci, ci]
            m2[ci, ci] = m[ci, ci]
            return a2, b2, m2

        # 'shuffle': permute every position EXCEPT the centre, with one shared
        # permutation across both streams and the mask. Keeping the centre
        # fixed is what keeps the task defined -- the label belongs to it.
        n = self.crop_size * self.crop_size
        center = ci * self.crop_size + ci
        others = np.delete(np.arange(n), center)
        perm = np.arange(n)
        perm[others] = self._ctx_rng.permutation(others)
        a2 = a.reshape(a.shape[0], -1)[:, perm].reshape(a.shape)
        b2 = b.reshape(b.shape[0], -1)[:, perm].reshape(b.shape)
        m2 = m.reshape(-1)[perm].reshape(m.shape)
        return a2, b2, m2

    @staticmethod
    def invert_target(values: np.ndarray, name: str) -> np.ndarray:
        """Map predictions back to the original label scale."""
        if name == "none":
            return values
        if name == "log1p":
            return np.expm1(values)
        raise ValueError(f"Unknown target transform: {name!r}")

    # ------------------------------------------------------------------ #
    # NEW: feature standardization
    # ------------------------------------------------------------------ #
    def compute_feature_stats(self, cells: list[tuple[int, int]]) -> dict:
        """Per-channel mean/std over the given cells, across every month.

        Pass TRAIN CELLS ONLY. Using all cells leaks val/test statistics.
        """
        if not cells:
            raise ValueError("compute_feature_stats received an empty cell list.")

        rows = np.array([r for r, _ in cells], dtype=np.int64)
        cols = np.array([c for _, c in cells], dtype=np.int64)

        a_values = np.concatenate(
            [t["stream_a"][:, rows, cols] for t in self.monthly_tensors], axis=1
        )  # [3, n_cells * n_months]
        b_values = np.concatenate(
            [t["stream_b"][:, rows, cols] for t in self.monthly_tensors], axis=1
        )  # [4, n_cells * n_months]

        stats = {
            "a_mean": a_values.mean(axis=1).astype(np.float64).tolist(),
            "a_std": (a_values.std(axis=1) + 1e-8).astype(np.float64).tolist(),
            "b_mean": b_values.mean(axis=1).astype(np.float64).tolist(),
            "b_std": (b_values.std(axis=1) + 1e-8).astype(np.float64).tolist(),
            "n_cells": len(cells),
        }
        return stats

    def apply_normalization(self, stats: dict) -> None:
        """Standardize both streams in place using the supplied stats."""
        if self.feature_stats is not None:
            raise RuntimeError("Normalization has already been applied.")

        a_mean = np.asarray(stats["a_mean"], np.float32)[:, None, None]
        a_std = np.asarray(stats["a_std"], np.float32)[:, None, None]
        b_mean = np.asarray(stats["b_mean"], np.float32)[:, None, None]
        b_std = np.asarray(stats["b_std"], np.float32)[:, None, None]

        for tensor_dict in self.monthly_tensors:
            invalid = tensor_dict["mask"] == 0

            tensor_dict["stream_a"] = (
                (tensor_dict["stream_a"] - a_mean) / a_std
            ).astype(np.float32)
            tensor_dict["stream_b"] = (
                (tensor_dict["stream_b"] - b_mean) / b_std
            ).astype(np.float32)

            # Keep invalid cells (and therefore the zero padding used when a
            # patch runs off the grid) at the post-standardization mean.
            tensor_dict["stream_a"][:, invalid] = 0.0
            tensor_dict["stream_b"][:, invalid] = 0.0

        self.feature_stats = stats
        print(f"[Dataset] Applied feature standardization "
              f"(stats from {stats['n_cells']} train cells).")

    # ------------------------------------------------------------------ #
    # Debug
    # ------------------------------------------------------------------ #
    def print_debug_summary(self) -> None:
        reg_values: list[float] = []
        for tensor_dict in self.monthly_tensors:
            valid_mask = tensor_dict["mask"] == 1
            reg_values.extend(tensor_dict["label_reg"][valid_mask].tolist())

        reg_array = np.asarray(reg_values, dtype=np.float64)
        first_tensor = self.monthly_tensors[0]
        first_sample = self[0]

        print("SpatialGridDataset summary (pure regression)")
        print(f"NPZ files loaded: {len(self.monthly_tensors)}")
        print(f"Total valid samples: {len(self)}")
        print(f"Unique cells: {len({(r, c) for _, r, c in self.samples})}")
        print(f"target_transform: {self.target_transform}")
        print(f"normalized: {self.feature_stats is not None}")
        print(f"stream_a shape: {first_tensor['stream_a'].shape}")
        print(f"stream_b shape: {first_tensor['stream_b'].shape}")
        print("label_reg statistics:")
        print(f"  count : {reg_array.size}")
        print(f"  min   : {reg_array.min():.6f}")
        print(f"  max   : {reg_array.max():.6f}")
        print(f"  mean  : {reg_array.mean():.6f}")
        print(f"  median: {np.median(reg_array):.6f}")
        print(f"  std   : {reg_array.std():.6f}")
        print("First sample shapes:")
        print(f"stream_a: {tuple(first_sample['stream_a'].shape)}")
        print(f"stream_b: {tuple(first_sample['stream_b'].shape)}")
        print(f"mask: {tuple(first_sample['mask'].shape)}")
        print(f"label_reg: {tuple(first_sample['label_reg'].shape)}")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _load_npz(self, npz_path: Path) -> dict[str, Any]:
        if not npz_path.exists():
            raise FileNotFoundError(f"Tensor .npz file not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_ARRAYS if key not in data.files]
            if missing:
                raise ValueError(f"{npz_path} missing required arrays: {missing}")

            tensor_dict = {
                "stream_a": data["stream_a"].astype(np.float32),
                "stream_b": data["stream_b"].astype(np.float32),
                "label_reg": data["label_reg"].astype(np.float32),
                "mask": data["mask"].astype(np.float32),
                "date": _read_npz_date(data, npz_path),
            }

        self._validate_tensor_shapes(npz_path, tensor_dict)
        return tensor_dict

    def _validate_tensor_shapes(self, npz_path: Path, tensor_dict: dict) -> None:
        stream_a = tensor_dict["stream_a"]
        stream_b = tensor_dict["stream_b"]
        label_reg = tensor_dict["label_reg"]
        mask = tensor_dict["mask"]

        if stream_a.ndim != 3 or stream_a.shape[0] != STREAM_A_CHANNELS:
            raise ValueError(
                f"{npz_path} stream_a must have shape [{STREAM_A_CHANNELS}, H, W], "
                f"got {stream_a.shape}"
            )
        if stream_b.ndim != 3 or stream_b.shape[0] != STREAM_B_CHANNELS:
            raise ValueError(
                f"{npz_path} stream_b must have shape [{STREAM_B_CHANNELS}, H, W], "
                f"got {stream_b.shape}"
            )

        spatial_shape = stream_a.shape[1:]
        if stream_b.shape[1:] != spatial_shape:
            raise ValueError(
                f"{npz_path} stream_a and stream_b spatial shapes differ: "
                f"{spatial_shape} vs {stream_b.shape[1:]}"
            )
        for name, array in [("label_reg", label_reg), ("mask", mask)]:
            if array.shape != spatial_shape:
                raise ValueError(
                    f"{npz_path} {name} shape must match H/W {spatial_shape}, "
                    f"got {array.shape}"
                )
        if not np.any(mask == 1):
            raise ValueError(f"{npz_path} contains no valid mask cells.")

    def _crop_with_padding(self, array: np.ndarray, row: int, col: int) -> np.ndarray:
        pad_before = self.crop_size // 2
        pad_after = self.crop_size - pad_before - 1

        row_start = row - pad_before
        row_end = row + pad_after + 1
        col_start = col - pad_before
        col_end = col + pad_after + 1

        if array.ndim == 3:
            _, height, width = array.shape
            output = np.zeros(
                (array.shape[0], self.crop_size, self.crop_size), dtype=array.dtype
            )
            src_row_start = max(row_start, 0)
            src_row_end = min(row_end, height)
            src_col_start = max(col_start, 0)
            src_col_end = min(col_end, width)
            dst_row_start = src_row_start - row_start
            dst_col_start = src_col_start - col_start
            output[
                :,
                dst_row_start : dst_row_start + (src_row_end - src_row_start),
                dst_col_start : dst_col_start + (src_col_end - src_col_start),
            ] = array[:, src_row_start:src_row_end, src_col_start:src_col_end]
            return output

        if array.ndim == 2:
            height, width = array.shape
            output = np.zeros((self.crop_size, self.crop_size), dtype=array.dtype)
            src_row_start = max(row_start, 0)
            src_row_end = min(row_end, height)
            src_col_start = max(col_start, 0)
            src_col_end = min(col_end, width)
            dst_row_start = src_row_start - row_start
            dst_col_start = src_col_start - col_start
            output[
                dst_row_start : dst_row_start + (src_row_end - src_row_start),
                dst_col_start : dst_col_start + (src_col_end - src_col_start),
            ] = array[src_row_start:src_row_end, src_col_start:src_col_end]
            return output

        raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


def _read_npz_date(data: np.lib.npyio.NpzFile, npz_path: Path) -> str:
    if "date" not in data.files:
        return npz_path.stem
    raw_date = data["date"]
    if raw_date.shape == ():
        return str(raw_date.item())
    return str(raw_date.tolist())


# ==============================================================================
# DEPRECATED
# ==============================================================================
def build_splits(
    dataset: Dataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """DEPRECATED per-sample random split. Use splits_v2.build_splits_v2.

    This splits the FLAT SAMPLE LIST, so cell (55, 72) can be in train for
    2022-01 and in test for 2022-02. With ICC = 0.9915 on this data that is a
    severe leak. Kept only so old commands do not crash.
    """
    warnings.warn(
        "build_splits() does not fix cell positions across splits and leaks "
        "badly on this data (ICC = 0.9915). Use splits_v2.build_splits_v2("
        "mode='cell') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must each be in (0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Split leaves no training samples: total={n_total}")

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def debug_spatial_grid_dataset(
    npz_paths: list[str | Path] | None = None,
    tensor_index_csv: str | Path | None = None,
    crop_size: int = 16,
) -> SpatialGridDataset:
    dataset = SpatialGridDataset(
        npz_paths=npz_paths, tensor_index_csv=tensor_index_csv, crop_size=crop_size
    )
    dataset.print_debug_summary()
    return dataset
