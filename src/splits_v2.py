"""
================================================================================
splits_v2.py -- Leak-free split construction for the MS-CA pipeline
================================================================================

WHY THIS FILE EXISTS
--------------------
The original ``build_splits`` in ``spatial_grid_dataset.py`` calls
``random_split`` over the FLAT SAMPLE LIST. Each sample is a
``(file_idx, row, col)`` triple, so cell (55, 72) from 2022-01 can land in
train while cell (55, 72) from 2022-02 lands in test. The cell position is NOT
fixed across splits.

That matters enormously for this dataset. Measured on the real Delhi NCR
tensors (36 months x 17,157 valid cells):

    within-cell variance  =    24,149
    between-cell variance = 2,821,894
    ICC                   = 0.9915
    cell-mean predictor R2 = 0.9915

99.15% of the label variance is explained by "which cell is this". Only 0.85%
is temporal. Combined with the fact that ``power_plant_count`` is 100% static
across all 36 months (it is effectively a location fingerprint), a model under
per-sample random_split can identify the cell and look up the answer from the
~24.5 sibling samples of the same cell that sit in train. It memorises
location instead of learning the proxy -> carbon mapping.

Fixing the cell position across splits (``mode='cell'``, the default here)
removes that shortcut. This is the design specified for this project.

SPLIT MODES
-----------
'cell'   : DEFAULT. Each cell (row, col) is assigned to exactly ONE of
           train/val/test. All 36 months of that cell follow it. Optionally
           stratified by the cell's mean label so the extreme emitters are
           spread evenly across splits.

'block'  : Same as 'cell', but cells are assigned in square blocks and a
           buffer ring around every held-out block is DROPPED from train.
           Use this as a robustness check.

           BUFFER ARITHMETIC. With crop_size=16 the patch of cell r covers
           rows [r-8, r+7]. Two patches are disjoint only when their centres
           are at Chebyshev distance >= 16 -- at distance 8 they still share
           8 of 16 rows. The buffer excludes train cells within `buffer` of a
           held-out cell, so the minimum centre distance is buffer+1 and zero
           pixel overlap needs buffer >= 15. The default is 16.

           Neighbouring cell labels in this dataset correlate at r ~= 0.98
           (lag 1), decaying to 0.54 at lag 16, so reporting 'cell' and
           'block' side by side pre-empts the obvious reviewer question.

'random' : The original per-sample random_split. Reproduced ONLY so it can be
           reported as a leakage control ("this is how inflated the number
           gets"). Do not use it for headline results.

USAGE
-----
    from splits_v2 import build_splits_v2
    train_subset, val_subset, test_subset = build_splits_v2(
        dataset, mode='cell', cell_mean_label=cell_mean, seed=42,
    )
================================================================================
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset, Subset

VALID_MODES = ("cell", "block", "random")


# ==============================================================================
# HELPERS
# ==============================================================================
def unique_cells(dataset: Dataset) -> list[tuple[int, int]]:
    """Sorted list of the distinct (row, col) cells present in the dataset."""
    return sorted({(row, col) for _file_idx, row, col in dataset.samples})


def compute_cell_mean_label(dataset: Dataset) -> np.ndarray:
    """Mean label_reg per cell, averaged over every month, as an [H, W] array.

    Used ONLY to stratify the cell assignment so that the handful of extreme
    emitters (max/median = 3532 on this data) do not all land in one split.
    This is the direct analogue of stratified sampling in classification.
    """
    first = dataset.monthly_tensors[0]["label_reg"]
    accum = np.zeros_like(first, dtype=np.float64)
    for tensor_dict in dataset.monthly_tensors:
        accum += tensor_dict["label_reg"]
    return accum / len(dataset.monthly_tensors)


def _stratum_ids(values: np.ndarray, n_strata: int) -> np.ndarray:
    """Assign each entry to an equal-count stratum by rank (0 .. n_strata-1)."""
    n = values.size
    order = np.argsort(values, kind="stable")
    strata = np.empty(n, dtype=np.int64)
    strata[order] = np.arange(n) * n_strata // n
    return strata


def _assign_cells(
    cells: list[tuple[int, int]],
    cell_mean_label: np.ndarray | None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    n_strata: int,
) -> tuple[set, set]:
    """Pick which cells go to val and test (everything else is train)."""
    n = len(cells)
    rng = np.random.default_rng(seed)

    if cell_mean_label is not None and n_strata > 1:
        values = np.array([cell_mean_label[r, c] for r, c in cells], dtype=np.float64)
        strata = _stratum_ids(values, n_strata)
    else:
        strata = np.zeros(n, dtype=np.int64)

    val_cells: set[tuple[int, int]] = set()
    test_cells: set[tuple[int, int]] = set()

    for stratum in np.unique(strata):
        members = np.where(strata == stratum)[0]
        perm = rng.permutation(members)
        n_val = round(len(perm) * val_ratio)
        n_test = round(len(perm) * test_ratio)
        val_cells.update(cells[i] for i in perm[:n_val])
        test_cells.update(cells[i] for i in perm[n_val : n_val + n_test])

    return val_cells, test_cells


def _assign_blocks(
    cells: list[tuple[int, int]],
    height: int,
    width: int,
    block_size: int,
    buffer: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set, set, set]:
    """Block-wise assignment with a dropped buffer ring around held-out blocks.

    Returns (val_cells, test_cells, train_cells). Cells inside the buffer ring
    belong to none of them and are discarded.
    """
    n_rows = int(np.ceil(height / block_size))
    n_cols = int(np.ceil(width / block_size))
    n_blocks = n_rows * n_cols

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    n_val = max(1, round(n_blocks * val_ratio))
    n_test = max(1, round(n_blocks * test_ratio))
    val_blocks = set(order[:n_val].tolist())
    test_blocks = set(order[n_val : n_val + n_test].tolist())

    # 0 = train, 1 = val, 2 = test
    assign = np.zeros((height, width), dtype=np.int8)
    for bi in range(n_rows):
        for bj in range(n_cols):
            block_id = bi * n_cols + bj
            r0, r1 = bi * block_size, min((bi + 1) * block_size, height)
            c0, c1 = bj * block_size, min((bj + 1) * block_size, width)
            if block_id in val_blocks:
                assign[r0:r1, c0:c1] = 1
            elif block_id in test_blocks:
                assign[r0:r1, c0:c1] = 2

    # Dilate the held-out region by `buffer` cells; train may not touch it.
    held_out = assign > 0
    if buffer > 0:
        dilated = np.zeros_like(held_out)
        for dr in range(-buffer, buffer + 1):
            for dc in range(-buffer, buffer + 1):
                shifted = np.roll(np.roll(held_out, dr, axis=0), dc, axis=1)
                # np.roll wraps around; clear the wrapped edges.
                if dr > 0:
                    shifted[:dr, :] = False
                elif dr < 0:
                    shifted[dr:, :] = False
                if dc > 0:
                    shifted[:, :dc] = False
                elif dc < 0:
                    shifted[:, dc:] = False
                dilated |= shifted
    else:
        dilated = held_out

    val_cells, test_cells, train_cells = set(), set(), set()
    for row, col in cells:
        code = assign[row, col]
        if code == 1:
            val_cells.add((row, col))
        elif code == 2:
            test_cells.add((row, col))
        elif not dilated[row, col]:
            train_cells.add((row, col))
        # else: buffer ring -> dropped
    return val_cells, test_cells, train_cells


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def build_splits_v2(
    dataset: Dataset,
    mode: str = "cell",
    cell_mean_label: np.ndarray | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    n_strata: int = 10,
    block_size: int = 24,
    buffer: int = 16,
    verbose: bool = True,
) -> tuple[Subset, Subset, Subset]:
    """Build train/val/test Subsets under a leak-aware policy.

    Args:
        dataset: a SpatialGridDataset (needs ``.samples`` and ``.monthly_tensors``).
        mode: 'cell' (default), 'block', or 'random'.
        cell_mean_label: [H, W] array of per-cell mean labels for stratification.
            Pass None to disable stratification. Ignored for 'block'/'random'.
        val_ratio / test_ratio: fractions of CELLS (not samples) held out.
        seed: controls both the cell permutation and the block permutation.
        n_strata: number of equal-count label strata (1 disables stratification).
        block_size / buffer: only used when mode='block'.

    Returns:
        (train_subset, val_subset, test_subset)
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must each be in (0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    n_samples = len(dataset)

    # ---------------------------------------------------------------- #
    # 'random' -- the original per-sample split. Leakage control only.
    # ---------------------------------------------------------------- #
    if mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        n_val = int(n_samples * val_ratio)
        n_test = int(n_samples * test_ratio)
        test_idx = perm[:n_test].tolist()
        val_idx = perm[n_test : n_test + n_val].tolist()
        train_idx = perm[n_test + n_val :].tolist()
        if verbose:
            print("[split:random] WARNING -- cell positions are NOT fixed. "
                  "Same cell appears in train and test at different months. "
                  "Use for leakage control only.")
            print(f"[split:random] samples: train={len(train_idx)} "
                  f"val={len(val_idx)} test={len(test_idx)}")
        return (Subset(dataset, train_idx),
                Subset(dataset, val_idx),
                Subset(dataset, test_idx))

    cells = unique_cells(dataset)

    # ---------------------------------------------------------------- #
    # 'cell' -- the project's design. One cell -> one split, all months.
    # ---------------------------------------------------------------- #
    if mode == "cell":
        val_cells, test_cells = _assign_cells(
            cells, cell_mean_label, val_ratio, test_ratio, seed, n_strata
        )
        train_cells = {c for c in cells if c not in val_cells and c not in test_cells}
        dropped_cells = 0

    # ---------------------------------------------------------------- #
    # 'block' -- cell split + spatial blocks + buffer ring. Robustness.
    # ---------------------------------------------------------------- #
    else:
        height = dataset.monthly_tensors[0]["label_reg"].shape[0]
        width = dataset.monthly_tensors[0]["label_reg"].shape[1]
        val_cells, test_cells, train_cells = _assign_blocks(
            cells, height, width, block_size, buffer, val_ratio, test_ratio, seed
        )
        dropped_cells = len(cells) - len(val_cells) - len(test_cells) - len(train_cells)

    # ---------------------------------------------------------------- #
    # Map cell assignment back onto flat sample indices.
    # ---------------------------------------------------------------- #
    train_idx, val_idx, test_idx = [], [], []
    for i, (_file_idx, row, col) in enumerate(dataset.samples):
        key = (row, col)
        if key in test_cells:
            test_idx.append(i)
        elif key in val_cells:
            val_idx.append(i)
        elif key in train_cells:
            train_idx.append(i)

    if verbose:
        print(f"[split:{mode}] cells:   train={len(train_cells)} "
              f"val={len(val_cells)} test={len(test_cells)} "
              f"dropped={dropped_cells}")
        print(f"[split:{mode}] samples: train={len(train_idx)} "
              f"val={len(val_idx)} test={len(test_idx)}")
        if mode == "block":
            print(f"[split:block] buffer={buffer} cells -> min train/holdout "
                  f"centre distance {buffer + 1} "
                  f"(needs >= 16 for zero patch pixel overlap at crop_size=16)")

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if not idx:
            raise ValueError(
                f"Split '{name}' is empty under mode={mode}. Adjust ratios, "
                f"block_size, or buffer."
            )

    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


# ==============================================================================
# VERIFICATION -- run this once and paste the output into your notes
# ==============================================================================
def verify_no_cell_overlap(dataset, train_subset, val_subset, test_subset) -> dict:
    """Assert that no cell appears in more than one split. Returns a summary."""
    def cells_of(subset):
        return {(dataset.samples[i][1], dataset.samples[i][2]) for i in subset.indices}

    train_cells = cells_of(train_subset)
    val_cells = cells_of(val_subset)
    test_cells = cells_of(test_subset)

    overlaps = {
        "train_and_val": len(train_cells & val_cells),
        "train_and_test": len(train_cells & test_cells),
        "val_and_test": len(val_cells & test_cells),
    }
    print("[verify] cell overlap:", overlaps)
    print(f"[verify] cells: train={len(train_cells)} val={len(val_cells)} "
          f"test={len(test_cells)}")

    if any(overlaps.values()):
        raise AssertionError(
            "CELL OVERLAP DETECTED -- the split is leaking. "
            f"{overlaps}"
        )
    print("[verify] OK: every cell belongs to exactly one split.")
    return {
        "n_train_cells": len(train_cells),
        "n_val_cells": len(val_cells),
        "n_test_cells": len(test_cells),
        **overlaps,
    }


def patch_pixel_overlap(dataset, train_subset, test_subset, crop_size: int = 16,
                        sample_limit: int = 3000) -> int:
    """Count pixels shared between train and test PATCH footprints.

    Under mode='cell' this will be large (neighbouring patches overlap by
    93.75% at crop_size=16). Under mode='block' it reaches 0 only when
    buffer >= crop_size - 1 (i.e. >= 15). Reported for transparency, not as a
    pass/fail gate.
    """
    def cells_of(subset):
        return list({(dataset.samples[i][1], dataset.samples[i][2])
                     for i in subset.indices})[:sample_limit]

    half = crop_size // 2

    def footprint(cell_list):
        pixels = set()
        for row, col in cell_list:
            for rr in range(row - half, row + crop_size - half):
                for cc in range(col - half, col + crop_size - half):
                    pixels.add((rr, cc))
        return pixels

    shared = len(footprint(cells_of(train_subset)) & footprint(cells_of(test_subset)))
    print(f"[verify] train/test patch pixel overlap (sampled): {shared}")
    return shared
