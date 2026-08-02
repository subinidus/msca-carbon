"""
================================================================================
train.py -- Training loop for the MS-CA network (single-task regression)
================================================================================

Trains ``MSCANet`` (model.py) on ``SpatialGridDataset``
(spatial_grid_dataset.py) for a strictly single-task, pure-regression
objective. No classification anywhere.

WHAT CHANGED FROM THE PREVIOUS VERSION
--------------------------------------
1. CELL-FIXED SPLITS (``--split_mode cell``, the default). The old
   ``build_splits`` split the flat sample list, so cell (55, 72) could be in
   train for 2022-01 and in test for 2022-02. Measured ICC on this data is
   0.9915 -- 99.15% of label variance is "which cell is this" -- and
   power_plant_count is 100% static across all 36 months, so it acts as a
   location fingerprint. Under the old split a model could identify the cell
   and look up the answer. Now every cell belongs to exactly one split and all
   36 of its months follow it.

2. STRATIFIED CELL ASSIGNMENT (``--stratify``, on by default). max/median is
   3532, so an unstratified draw can put most extreme emitters in one split
   and make seed-to-seed variance enormous.

3. LOG TARGET (``--target_transform log1p``, the default). skew = 40.03,
   kurtosis = 1723. On the raw scale an L1 objective learns the conditional
   median and gives up on every large emitter.

4. FEATURE STANDARDIZATION (``--normalize``, on by default), computed on
   TRAIN CELLS ONLY. Raw channels span ~8 orders of magnitude.

5. ROBUST MODEL SELECTION. Primary metric configurable (``--select_metric``,
   default r2), smoothed over a 3-epoch window, with a bias guardrail, a
   warmup exclusion, a min-delta, and early stopping.

REPRODUCING THE TEST SPLIT
--------------------------
``evaluation.py`` reads the split settings and the feature stats back out of
the checkpoint, so you do not have to re-specify them. Just point it at
best_model.pth and the same .npz files.

USAGE
-----
    python train.py \\
        --data_path /path/to/*.npz \\
        --output_dir ./checkpoints \\
        --split_mode cell \\
        --epochs 50 --batch_size 128 --use_amp --disable_wandb
================================================================================
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_data import FastPatchBatcher
from metrics import full_metrics
from model import MSCANet
from spatial_grid_dataset import SpatialGridDataset
from splits_v2 import (
    build_splits_v2,
    compute_cell_mean_label,
    verify_no_cell_overlap,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ==============================================================================
# SECTION 0: ARGUMENT PARSING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training script for the MS-CA single-task regression network."
    )

    # --- Data / I/O ---
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_path", type=str, nargs="+", default=None,
                            help="One or more .npz tensor files.")
    data_group.add_argument("--tensor_index_csv", type=str, default=None,
                            help="CSV with a 'tensor_path' column.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)

    # --- Dataset ---
    parser.add_argument("--crop_size", type=int, default=16,
                        help="Must stay 16: MSCANet is built with img_size=(16,16).")
    parser.add_argument("--target_transform", type=str, default="log1p",
                        choices=["none", "log1p"],
                        help="log1p is strongly recommended (label skew = 40.03).")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable train-only feature standardization.")
    parser.add_argument("--context", type=str, default="full",
                        choices=["full", "center_only", "shuffle"],
                        help="Spatial-context ablation. center_only zeroes all "
                             "non-centre cells (architecture constant, context "
                             "removed -- the direct test of cell independence). "
                             "shuffle permutes the 255 non-centre positions "
                             "(bag-of-neighbours kept, arrangement destroyed). "
                             "Applied at train AND eval via the dataset.")

    # --- Split policy ---
    parser.add_argument("--split_mode", type=str, default="cell",
                        choices=["cell", "block", "random"],
                        help="cell: one cell -> one split (default, the project "
                             "design). block: cell split + spatial blocks + "
                             "buffer ring (robustness check). random: the old "
                             "per-sample split, leakage control only.")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--no_stratify", action="store_true",
                        help="Disable label-decile stratification of cells.")
    parser.add_argument("--n_strata", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=24,
                        help="Only used when --split_mode block.")
    parser.add_argument("--buffer", type=int, default=16,
                        help="Buffer ring in cells. Zero patch pixel overlap "
                             "needs buffer >= crop_size - 1 (15 at crop_size=16), "
                             "because a patch reaches 8 cells each way. Only "
                             "used with --split_mode block.")

    # --- Model architecture ---
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # --- Optimization ---
    parser.add_argument("--loss", type=str, default="l1",
                        choices=["l1", "huber", "mse"],
                        help="l1 reproduces the original objective. huber is a "
                             "middle ground if you select on r2.")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_augment", action="store_true")

    # --- Model selection ---
    parser.add_argument("--select_metric", type=str, default="r2",
                        choices=["r2", "mae"],
                        help="r2 pairs with an 'explanatory power' claim; mae "
                             "reproduces the original selection rule.")
    parser.add_argument("--smooth_window", type=int, default=3,
                        help="Moving-average window over the val metric. Picking "
                             "a raw argmax across 50 epochs selects validation "
                             "noise.")
    parser.add_argument("--select_warmup", type=int, default=5,
                        help="Epochs excluded from selection (LR warmup).")
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stop after this many epochs with no gain. "
                             "<=0 disables.")
    parser.add_argument("--bias_guard", type=float, default=0.10,
                        help="Only accept epochs with |val bias| < this * label "
                             "std. <=0 disables. Default relaxed from 0.05: with "
                             "an L1 loss on a right-skewed log target, early-to-"
                             "mid training bias routinely sits at 0.05-0.10 x "
                             "std even when the fit is healthy; 0.05 rejected "
                             "healthy epochs in practice. A fallback checkpoint "
                             "(best_model_unconstrained.pth) is always written "
                             "regardless of this guard.")

    # --- DataLoader / device ---
    parser.add_argument("--data_parallel", action="store_true",
                        help="Wrap the model in nn.DataParallel when >1 GPU is "
                             "visible. OFF by default: for this tiny 16-token "
                             "ViT the per-step replicate + scatter/gather "
                             "overhead exceeds the compute being split, so "
                             "DataParallel makes training SLOWER, not faster.")
    parser.add_argument("--no_fast_loader", action="store_true",
                        help="Use the original per-sample DataLoader instead "
                             "of the GPU-resident FastPatchBatcher. The fast "
                             "path is bit-identical for full/center_only and "
                             "roughly 15-40x faster per epoch on Kaggle; the "
                             "slow path is only needed for --context shuffle "
                             "(where it is selected automatically).")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # --- WandB ---
    parser.add_argument("--wandb_project", type=str, default="ms-ca-regression")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()
    if args.crop_size != 16:
        raise ValueError(
            f"crop_size must be 16 (MSCANet hardcodes img_size=(16,16)), "
            f"got {args.crop_size}."
        )
    return args


# ==============================================================================
# SECTION 1: REPRODUCIBILITY
# ==============================================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# SECTION 2: SYNCHRONIZED D4 AUGMENTATION
# ==============================================================================
def _apply_d4_transform(stream_a, stream_b, k_rot, flip_h, flip_v):
    """One D4 symmetry applied IDENTICALLY to both streams, preserving the
    spatial alignment the cross-attention fusion depends on. The scalar target
    is invariant, so labels are untouched."""
    if k_rot != 0:
        stream_a = torch.rot90(stream_a, k=k_rot, dims=(-2, -1))
        stream_b = torch.rot90(stream_b, k=k_rot, dims=(-2, -1))
    if flip_h:
        stream_a = torch.flip(stream_a, dims=(-2,))
        stream_b = torch.flip(stream_b, dims=(-2,))
    if flip_v:
        stream_a = torch.flip(stream_a, dims=(-1,))
        stream_b = torch.flip(stream_b, dims=(-1,))
    return stream_a, stream_b


def d4_augmentation_collate_fn(batch):
    aug_a, aug_b, regs, rows, cols = [], [], [], [], []
    for item in batch:
        a, b = _apply_d4_transform(
            item["stream_a"], item["stream_b"],
            random.randint(0, 3), random.random() < 0.5, random.random() < 0.5,
        )
        aug_a.append(a)
        aug_b.append(b)
        regs.append(item["label_reg"])
        rows.append(item["row"])
        cols.append(item["col"])
    return {
        "stream_a": torch.stack(aug_a, dim=0),
        "stream_b": torch.stack(aug_b, dim=0),
        "label_reg": torch.stack(regs, dim=0),
        "row": torch.tensor(rows),
        "col": torch.tensor(cols),
    }


def plain_collate_fn(batch):
    return {
        "stream_a": torch.stack([b["stream_a"] for b in batch], dim=0),
        "stream_b": torch.stack([b["stream_b"] for b in batch], dim=0),
        "label_reg": torch.stack([b["label_reg"] for b in batch], dim=0),
        "row": torch.tensor([b["row"] for b in batch]),
        "col": torch.tensor([b["col"] for b in batch]),
    }


# ==============================================================================
# SECTION 3: OPTIMIZER PARAMETER GROUPING
# ==============================================================================
def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Standard AdamW grouping: no decay on bias/LayerNorm/1-D params."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ==============================================================================
# SECTION 4: LINEAR WARM-UP + COSINE ANNEALING
# ==============================================================================
def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, cycles=3):
    warmup_steps = max(1, warmup_steps)
    remaining = max(1, total_steps - warmup_steps)
    cycles = max(1, cycles)
    cycle_length = max(1, remaining // cycles)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float((step - warmup_steps) % cycle_length) / float(cycle_length)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ==============================================================================
# SECTION 5: LOSS
# ==============================================================================
def build_loss(args) -> nn.Module:
    if args.loss == "l1":
        return nn.L1Loss()
    if args.loss == "mse":
        return nn.MSELoss()
    if args.loss == "huber":
        return nn.HuberLoss(delta=args.huber_delta)
    raise ValueError(f"Unknown loss: {args.loss}")


# ==============================================================================
# SECTION 6: CHECKPOINTING
# ==============================================================================
def _unwrap(model: nn.Module) -> nn.Module:
    """Strip DataParallel's .module wrapper so checkpoints load cleanly."""
    return model.module if isinstance(model, nn.DataParallel) else model


def save_best_model(path, model, model_config, epoch, best_score, args,
                    feature_stats, split_info) -> None:
    """The artifact evaluation.py consumes. Carries everything needed to
    rebuild the identical split and normalization."""
    torch.save(
        {
            "model_state_dict": _unwrap(model).state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "best_score": best_score,
            "select_metric": args.select_metric,
            "args": vars(args),
            "feature_stats": feature_stats,
            "target_transform": args.target_transform,
            "context_mode": getattr(args, "context", "full"),
            "split_info": split_info,
        },
        path,
    )


def save_latest_checkpoint(path, model, optimizer, scheduler, model_config,
                           epoch, best_score, args, feature_stats,
                           split_info) -> None:
    torch.save(
        {
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "best_score": best_score,
            "args": vars(args),
            "feature_stats": feature_stats,
            "target_transform": args.target_transform,
            "context_mode": getattr(args, "context", "full"),
            "split_info": split_info,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    _unwrap(model).load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_score = ckpt.get("best_score", -float("inf"))
    print(f"[Resume] Loaded '{path}' -> resuming at epoch {start_epoch}.")
    return start_epoch, best_score


# ==============================================================================
# SECTION 7: TRAIN / VALIDATE ONE EPOCH
# ==============================================================================
def train_one_epoch(model, loss_fn, loader, optimizer, scheduler, device,
                    grad_clip, scaler, use_amp, epoch, wandb_run,
                    log_every=50) -> dict:
    model.train()
    running_loss = 0.0
    preds_all, targets_all = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        stream_a = batch["stream_a"].to(device, non_blocking=True)
        stream_b = batch["stream_b"].to(device, non_blocking=True)
        target = batch["label_reg"].to(device, non_blocking=True).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(stream_a, stream_b)
            loss = loss_fn(out, target)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        preds_all.append(out.detach().float().cpu().view(-1))
        targets_all.append(target.detach().float().cpu().view(-1))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if wandb_run is not None and step % log_every == 0:
            wandb_run.log({
                "train/step_loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })

    preds = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    m = full_metrics(preds, targets)
    m["avg_loss"] = running_loss / max(1, len(loader))
    return m


@torch.no_grad()
def validate_one_epoch(model, loss_fn, loader, device, epoch) -> dict:
    model.eval()
    running_loss = 0.0
    preds_all, targets_all = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False, dynamic_ncols=True)
    for batch in pbar:
        stream_a = batch["stream_a"].to(device, non_blocking=True)
        stream_b = batch["stream_b"].to(device, non_blocking=True)
        target = batch["label_reg"].to(device, non_blocking=True).float().view(-1, 1)

        out = model(stream_a, stream_b)
        loss = loss_fn(out, target)

        running_loss += loss.item()
        preds_all.append(out.float().cpu().view(-1))
        targets_all.append(target.float().cpu().view(-1))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    preds = torch.cat(preds_all).numpy()
    targets = torch.cat(targets_all).numpy()
    m = full_metrics(preds, targets)
    m["avg_loss"] = running_loss / max(1, len(loader))
    return m


# ==============================================================================
# SECTION 8: MAIN
# ==============================================================================
def main():
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Setup] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")

    # ---------------------------------------------------------------- #
    # WandB (degrades to console-only if unavailable or disabled)
    # ---------------------------------------------------------------- #
    wandb_run = None
    if not args.disable_wandb:
        if _WANDB_AVAILABLE:
            wandb_run = wandb.init(project=args.wandb_project,
                                   entity=args.wandb_entity,
                                   name=args.wandb_run_name, config=vars(args))
        else:
            print("[Warning] WandB not installed; console-only logging.")

    # ---------------------------------------------------------------- #
    # Dataset
    # ---------------------------------------------------------------- #
    full_dataset = SpatialGridDataset(
        npz_paths=args.data_path,
        tensor_index_csv=args.tensor_index_csv,
        crop_size=args.crop_size,
    )

    # Per-cell mean label for stratification -- computed on the RAW label,
    # before any transform, so it is comparable across runs.
    cell_mean_label = compute_cell_mean_label(full_dataset) \
        if not args.no_stratify else None

    full_dataset.apply_target_transform(args.target_transform)
    full_dataset.set_context_mode(args.context)

    # ---------------------------------------------------------------- #
    # Split (cell-fixed by default)
    # ---------------------------------------------------------------- #
    train_subset, val_subset, test_subset = build_splits_v2(
        full_dataset,
        mode=args.split_mode,
        cell_mean_label=cell_mean_label,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        n_strata=args.n_strata,
        block_size=args.block_size,
        buffer=args.buffer,
    )
    if args.split_mode in ("cell", "block"):
        verify_no_cell_overlap(full_dataset, train_subset, val_subset, test_subset)

    split_info = {
        "split_mode": args.split_mode,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "stratify": not args.no_stratify,
        "n_strata": args.n_strata,
        "block_size": args.block_size,
        "buffer": args.buffer,
        "crop_size": args.crop_size,
        "npz_files": [str(p) for p in full_dataset.npz_paths],
        "n_train": len(train_subset),
        "n_val": len(val_subset),
        "n_test": len(test_subset),
    }

    # ---------------------------------------------------------------- #
    # Feature standardization -- TRAIN CELLS ONLY
    # ---------------------------------------------------------------- #
    feature_stats = None
    if not args.no_normalize:
        train_cells = sorted({(full_dataset.samples[i][1], full_dataset.samples[i][2])
                              for i in train_subset.indices})
        feature_stats = full_dataset.compute_feature_stats(train_cells)
        full_dataset.apply_normalization(feature_stats)

    # Label std in the TRAINING space, used by the bias guardrail.
    train_targets = np.array(
        [full_dataset.monthly_tensors[f]["label_reg"][r, c]
         for f, r, c in (full_dataset.samples[i] for i in train_subset.indices)],
        dtype=np.float64,
    )
    label_std = float(train_targets.std())
    print(f"[Data] train label std ({args.target_transform} space) = {label_std:.4f}")

    # ---------------------------------------------------------------- #
    # DataLoaders
    # ---------------------------------------------------------------- #
    use_fast = (not args.no_fast_loader) and args.context != "shuffle"
    if use_fast:
        # GPU-resident vectorized batching: the whole raster stack is ~20 MB,
        # so each batch is a single tensor gather instead of ~batch_size
        # Python round-trips. See fast_data.py for the timing arithmetic.
        train_loader = FastPatchBatcher(
            full_dataset, list(train_subset.indices),
            batch_size=args.batch_size, shuffle=True,
            augment=not args.no_augment, device=device,
            seed=args.seed, drop_last=True,
        )
        val_loader = FastPatchBatcher(
            full_dataset, list(val_subset.indices),
            batch_size=args.batch_size, shuffle=False,
            augment=False, device=device, seed=args.seed,
        )
        print(f"[Data] FastPatchBatcher on {device} | "
              f"steps/epoch={len(train_loader)}")
    else:
        if args.context == "shuffle" and not args.no_fast_loader:
            print("[Data] context=shuffle needs per-sample permutations -> "
                  "using the standard DataLoader for this ablation.")
        train_collate = (plain_collate_fn if args.no_augment
                         else d4_augmentation_collate_fn)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            collate_fn=train_collate, persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
            collate_fn=plain_collate_fn, persistent_workers=args.num_workers > 0,
        )
        print(f"[Data] steps/epoch={len(train_loader)}")

    # ---------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------- #
    model_config = {
        "img_size": (16, 16), "patch_size": 4,
        "embed_dim": args.embed_dim, "encoder_depth": args.encoder_depth,
        "num_heads": args.num_heads, "dropout": args.dropout,
        "in_channels_a": 3, "in_channels_b": 4,
    }
    model = MSCANet(**model_config)

    if args.data_parallel and torch.cuda.device_count() > 1:
        print(f"[Setup] Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        print(f"[Setup] {torch.cuda.device_count()} GPUs visible; using ONE. "
              f"This model is a 16-token ViT -- DataParallel's per-step "
              f"replicate/scatter/gather costs more than the compute it "
              f"splits, and was a major part of the 0.38 s/step slowdown. "
              f"Pass --data_parallel to force multi-GPU.")
    model = model.to(device)

    loss_fn = build_loss(args)
    print(f"[Setup] loss={args.loss} | select_metric={args.select_metric} | "
          f"target_transform={args.target_transform}")

    if wandb_run is not None:
        wandb_run.watch(model, log="gradients", log_freq=200)

    # ---------------------------------------------------------------- #
    # Optimizer + scheduler
    # ---------------------------------------------------------------- #
    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999)
    )
    steps_per_epoch = len(train_loader)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=int(steps_per_epoch * args.warmup_epochs),
        total_steps=steps_per_epoch * args.epochs,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)

    # ---------------------------------------------------------------- #
    # Resume
    # ---------------------------------------------------------------- #
    start_epoch = 0
    best_score = -float("inf")
    if args.resume_from is not None and os.path.isfile(args.resume_from):
        start_epoch, best_score = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, device
        )

    # ---------------------------------------------------------------- #
    # Training loop with robust model selection
    # ---------------------------------------------------------------- #
    history: list[float] = []
    best_epoch = -1
    stale_epochs = 0
    fallback_score = -float("inf")
    fallback_epoch = -1
    fallback_ckpt_path = os.path.join(args.output_dir,
                                      "best_model_unconstrained.pth")
    log_rows = []

    def to_score(metrics: dict) -> float:
        """Higher is always better, so one comparison covers both metrics."""
        return metrics["r2"] if args.select_metric == "r2" else -metrics["mae"]

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_m = train_one_epoch(
            model, loss_fn, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, scaler=scaler, use_amp=args.use_amp,
            epoch=epoch, wandb_run=wandb_run,
        )
        val_m = validate_one_epoch(model, loss_fn, val_loader, device, epoch=epoch)
        dt = time.time() - t0

        history.append(to_score(val_m))
        window = history[-args.smooth_window:]
        smoothed = sum(window) / len(window)

        bias_ok = (args.bias_guard <= 0) or \
                  (abs(val_m["bias"]) < args.bias_guard * label_std)
        eligible = (epoch >= args.select_warmup) and bias_ok

        print(
            f"[Epoch {epoch}] {dt:.1f}s | "
            f"train R2={train_m['r2']:.4f} MAE={train_m['mae']:.4f} | "
            f"val R2={val_m['r2']:.4f} MAE={val_m['mae']:.4f} "
            f"bias={val_m['bias']:+.4f} | smoothed={smoothed:.4f}"
            f"{'' if bias_ok else '  [BIAS GUARD]'}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch, "epoch_time_sec": dt,
                "train/r2": train_m["r2"], "train/mae": train_m["mae"],
                "train/rmse": train_m["rmse"],
                "val/r2": val_m["r2"], "val/mae": val_m["mae"],
                "val/rmse": val_m["rmse"], "val/bias": val_m["bias"],
                "val/spearman": val_m["spearman"],
                "val/top10_recall": val_m["top10_recall"],
                "val/smoothed_score": smoothed,
            })

        log_rows.append({
            "epoch": epoch, "train_r2": train_m["r2"], "train_mae": train_m["mae"],
            "val_r2": val_m["r2"], "val_mae": val_m["mae"],
            "val_bias": val_m["bias"], "val_spearman": val_m["spearman"],
            "val_top10_recall": val_m["top10_recall"], "smoothed": smoothed,
            "eligible": bool(eligible),
        })

        save_latest_checkpoint(latest_ckpt_path, model, optimizer, scheduler,
                               model_config, epoch, best_score, args,
                               feature_stats, split_info)

        if eligible and smoothed > best_score + args.min_delta:
            best_score = smoothed
            best_epoch = epoch
            stale_epochs = 0
            save_best_model(best_ckpt_path, model, model_config, epoch,
                            best_score, args, feature_stats, split_info)
            print(f"[Checkpoint] New best (smoothed {args.select_metric}"
                  f"={smoothed:.4f}) -> saved '{best_ckpt_path}'.")
        elif epoch < args.select_warmup:
            # Warmup epochs can never qualify BY DESIGN, so they must not
            # consume the patience budget. The earlier version counted them,
            # which shrank an effective patience of 10 down to 10 - warmup
            # and could kill a healthy run before its first eligible epoch.
            pass
        else:
            stale_epochs += 1

        # Fallback: track the best smoothed score IGNORING warmup and the
        # bias guard, so a run can never end with no checkpoint at all. If
        # the guarded best_model.pth exists, prefer it; the fallback exists
        # for diagnosis and for runs where the guard proves too strict.
        if smoothed > fallback_score + args.min_delta:
            fallback_score = smoothed
            fallback_epoch = epoch
            save_best_model(fallback_ckpt_path, model, model_config, epoch,
                            fallback_score, args, feature_stats, split_info)

        if stale_epochs > 0 and args.patience > 0 and stale_epochs >= args.patience:
            print(f"[EarlyStop] No improvement for {args.patience} epochs. "
                  f"Best epoch={best_epoch}, score={best_score:.4f}.")
            break

    with open(os.path.join(args.output_dir, "train_log.json"), "w") as fh:
        json.dump({"args": vars(args), "split_info": split_info,
                   "label_std": label_std, "history": log_rows}, fh, indent=2)

    if best_epoch < 0:
        print("\n[Done] WARNING: no epoch passed the selection criteria "
              "(warmup + bias guard), so best_model.pth was not written.")
        if fallback_epoch >= 0:
            print(f"[Done] A fallback checkpoint WAS saved: "
                  f"'{fallback_ckpt_path}' (epoch {fallback_epoch}, smoothed "
                  f"{args.select_metric}={fallback_score:.4f}, guard ignored). "
                  f"Inspect its val bias before trusting it; if the bias is "
                  f"acceptable, use it directly or rerun with a looser "
                  f"--bias_guard.")
    else:
        print(f"\n[Done] Best epoch={best_epoch}, smoothed "
              f"{args.select_metric}={best_score:.4f}.")
        print(f"[Next] python evaluation.py --checkpoint {best_ckpt_path} "
              f"--data_path <same .npz files>")


if __name__ == "__main__":
    main()
