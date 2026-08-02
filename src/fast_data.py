"""
================================================================================
fast_data.py -- GPU-resident patch batcher for the MS-CA pipeline
================================================================================

WHY THIS EXISTS
---------------
The standard path (SpatialGridDataset.__getitem__ + DataLoader + a Python
collate) pays, PER SAMPLE: three numpy crops, three torch conversions, and a
Python loop iteration. On Kaggle (4 vCPUs) that is ~2-3 ms per sample. With
~432k training samples per epoch the CPU needs 15-20 minutes per epoch while
the GPU -- running a 16-token ViT -- sits idle. Ten epochs = ~13,000 s,
which is exactly the wall-clock the pipeline showed. The cost is O(N) with a
large Python constant, not an infinite loop; this module removes the constant.

The entire raster stack is tiny (~20 MB for 36 months x 7 channels), so:

  1. Stack all months into two tensors  A:[T,3,H,W]  B:[T,4,H,W]  once.
  2. Zero-pad spatially by (8 top/left, 7 bottom/right) -- the exact geometry
     of SpatialGridDataset._crop_with_padding at crop_size=16.
  3. ``unfold`` twice -> a zero-copy VIEW of every possible 16x16 window:
     A_win[t, :, r, c] == the patch train.py would have cropped for (t, r, c).
  4. Move everything to the GPU (it fits trivially) and materialise each batch
     with ONE advanced-indexing gather:  A_win[t_idx, :, r_idx, c_idx].

One epoch's data work collapses from ~432k Python round-trips to
~N/batch_size tensor gathers on the GPU.

EQUIVALENCE GUARANTEES
----------------------
* Built AFTER apply_target_transform / apply_normalization, it inherits both
  (it reads dataset.monthly_tensors, which those methods mutate in place).
* Patch values are bit-identical to the Dataset path (asserted in tests).
* D4 augmentation reproduces the per-sample (k_rot, flip_h, flip_v) draws of
  d4_augmentation_collate_fn, applied vectorised by grouping the batch by
  transform (16 groups max).
* context_mode 'center_only' is supported natively. 'shuffle' needs a fresh
  per-sample permutation, which defeats vectorisation -- train.py falls back
  to the standard DataLoader for that ablation (it is a one-off run anyway).

USAGE (see train.py)
--------------------
    batcher = FastPatchBatcher(dataset, subset.indices, batch_size=512,
                               shuffle=True, augment=True, device=device,
                               seed=args.seed)
    for batch in batcher:   # dict with the same keys the collate_fns produce
        ...
================================================================================
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class FastPatchBatcher:
    """Iterable over batches of (stream_a, stream_b, label_reg, row, col)."""

    def __init__(self, dataset, indices, batch_size: int, shuffle: bool,
                 augment: bool, device: torch.device, seed: int = 0,
                 drop_last: bool = False):
        if dataset.context_mode == "shuffle":
            raise ValueError(
                "FastPatchBatcher does not support context_mode='shuffle'; "
                "use the standard DataLoader path for that ablation."
            )
        self.crop = dataset.crop_size
        if self.crop != 16:
            raise ValueError("FastPatchBatcher assumes crop_size=16.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.device = device
        self.drop_last = drop_last
        self.center_only = dataset.context_mode == "center_only"
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

        # ---- stack the (already transformed/normalized) monthly tensors ----
        a = torch.from_numpy(np.stack(
            [t["stream_a"] for t in dataset.monthly_tensors])).float()
        b = torch.from_numpy(np.stack(
            [t["stream_b"] for t in dataset.monthly_tensors])).float()
        labels = torch.from_numpy(np.stack(
            [t["label_reg"] for t in dataset.monthly_tensors])).float()

        # ---- pad with the exact _crop_with_padding geometry ----
        pad_before = self.crop // 2            # 8
        pad_after = self.crop - pad_before - 1  # 7
        pad = (pad_before, pad_after, pad_before, pad_after)  # l, r, t, b
        a = F.pad(a, pad)
        b = F.pad(b, pad)

        # ---- every 16x16 window as a zero-copy view ----
        # [T, C, H, W, 16, 16]; window (t, :, r, c) == patch centred at (r, c)
        self.a_win = a.unfold(2, self.crop, 1).unfold(3, self.crop, 1)
        self.b_win = b.unfold(2, self.crop, 1).unfold(3, self.crop, 1)
        self.labels = labels

        samples = [dataset.samples[i] for i in indices]
        self.t_idx = torch.tensor([s[0] for s in samples], dtype=torch.long)
        self.r_idx = torch.tensor([s[1] for s in samples], dtype=torch.long)
        self.c_idx = torch.tensor([s[2] for s in samples], dtype=torch.long)

        # The whole stack is ~tens of MB; park it on the compute device so the
        # per-batch gather never touches the CPU.
        if device.type == "cuda":
            self.a_win = self.a_win.to(device)
            self.b_win = self.b_win.to(device)
            self.labels = self.labels.to(device)
            self.t_idx = self.t_idx.to(device)
            self.r_idx = self.r_idx.to(device)
            self.c_idx = self.c_idx.to(device)

    def __len__(self) -> int:
        n = self.t_idx.numel()
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _d4(self, a: torch.Tensor, b: torch.Tensor) -> tuple:
        """Per-sample D4, vectorised by grouping the batch by transform.

        Reproduces d4_augmentation_collate_fn's draw: k_rot ~ U{0..3},
        flip_h ~ Bern(0.5) on dim -2, flip_v ~ Bern(0.5) on dim -1, the SAME
        transform applied to both streams of a sample.
        """
        n = a.shape[0]
        k_rot = torch.randint(0, 4, (n,), generator=self._gen)
        flip_h = torch.rand(n, generator=self._gen) < 0.5
        flip_v = torch.rand(n, generator=self._gen) < 0.5
        if self.device.type == "cuda":
            k_rot, flip_h, flip_v = (k_rot.to(self.device),
                                     flip_h.to(self.device),
                                     flip_v.to(self.device))
        for k in range(4):
            for fh in (False, True):
                for fv in (False, True):
                    sel = (k_rot == k) & (flip_h == fh) & (flip_v == fv)
                    if not bool(sel.any()):
                        continue
                    ga, gb = a[sel], b[sel]
                    if k:
                        ga = torch.rot90(ga, k=k, dims=(-2, -1))
                        gb = torch.rot90(gb, k=k, dims=(-2, -1))
                    if fh:
                        ga = torch.flip(ga, dims=(-2,))
                        gb = torch.flip(gb, dims=(-2,))
                    if fv:
                        ga = torch.flip(ga, dims=(-1,))
                        gb = torch.flip(gb, dims=(-1,))
                    a[sel], b[sel] = ga, gb
        return a, b

    def __iter__(self):
        n = self.t_idx.numel()
        if self.shuffle:
            order = torch.randperm(n, generator=self._gen)
            if self.device.type == "cuda":
                order = order.to(self.device)
        else:
            order = torch.arange(n, device=self.t_idx.device)

        stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, stop, self.batch_size):
            sel = order[start:start + self.batch_size]
            t, r, c = self.t_idx[sel], self.r_idx[sel], self.c_idx[sel]

            # Advanced indices (t, r, c) separated by the ':' slice -> the
            # gathered dim goes first: result is [B, C, 16, 16]. One kernel.
            a = self.a_win[t, :, r, c].clone()
            b = self.b_win[t, :, r, c].clone()
            y = self.labels[t, r, c]

            if self.center_only:
                ci = self.crop // 2
                a2 = torch.zeros_like(a)
                b2 = torch.zeros_like(b)
                a2[:, :, ci, ci] = a[:, :, ci, ci]
                b2[:, :, ci, ci] = b[:, :, ci, ci]
                a, b = a2, b2

            if self.augment:
                a, b = self._d4(a, b)

            yield {"stream_a": a, "stream_b": b, "label_reg": y,
                   "row": r, "col": c}
