"""
================================================================================
inference.py -- Full-grid inference and map export for the MS-CA regressor
================================================================================

Runs the trained model over EVERY valid cell in every month, inverts the
log1p transform, and writes map-ready artefacts.

WHAT IT WRITES
--------------
  predictions_long.csv    one row per (date, row, col): pred, true, split
                          -> the raw table; use for scatter plots, residual
                             analysis, per-month maps
  predictions_cell.csv    one row per (row, col): mean/median pred and true
                          across all 36 months, plus split tag
                          -> the table behind a single static carbon map
  prediction_grids.npz    pred_grid / true_grid as [T, H, W] float arrays,
                          plus mask, dates, rows, cols
                          -> the fastest path to plt.imshow() / rasterio
  inference_summary.json  metrics overall and per split, run configuration

TWO THINGS THIS SCRIPT DOES THAT MATTER FOR THE PAPER
-----------------------------------------------------
1. IT TAGS EVERY CELL WITH ITS SPLIT. Inference covers all ~17k cells, but a
   map drawn over all of them mixes cells the model trained on with cells it
   never saw. Those are not the same claim. The ``split`` column lets you
   colour or filter them, and the summary reports metrics separately. A
   reviewer will ask which cells were held out; answer it in the figure.

2. IT CORRECTS THE BACK-TRANSFORM BIAS (``--smearing``, ON by default here).
   The model minimised error in log space. Since E[expm1(y_hat)] != expm1(
   E[y_hat]) (Jensen's inequality), naively applying expm1 UNDER-predicts on
   the original scale, and the shortfall is worst exactly where emissions are
   largest. If the paper reports absolute carbon units or a regional total,
   that bias goes straight into the headline number. Duan's smearing factor
   is estimated from TRAIN residuals only and reported explicitly. Use
   ``--no_smearing`` to get the uncorrected values.

USAGE
-----
    python inference.py \\
        --checkpoint /kaggle/working/ckpt_cell/best_model.pth \\
        --data_path '/kaggle/input/.../ms_mca_tensor_delhi_ncr_*.npz' \\
        --output_dir /kaggle/working/inference
================================================================================
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import duan_smearing, full_metrics, print_metrics
from model import MSCANet
from spatial_grid_dataset import SpatialGridDataset
from splits_v2 import build_splits_v2, compute_cell_mean_label


# ==============================================================================
# ARGS
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Full-grid inference + map export for the MS-CA regressor."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, nargs="+", required=True,
                   help="The .npz files. A quoted glob also works.")
    p.add_argument("--output_dir", type=str, default="./inference")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--no_smearing", action="store_true",
                   help="Skip Duan's bias correction and use raw expm1 output.")
    p.add_argument("--no_split_tags", action="store_true",
                   help="Skip split reconstruction (use if the file list "
                        "differs from training).")
    p.add_argument("--float_format", type=str, default="%.6g")
    return p.parse_args()


# ==============================================================================
# HELPERS
# ==============================================================================
def expand_paths(patterns):
    """Accept literal paths, shell-expanded lists, or an unexpanded glob."""
    out = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        out.extend(matches if matches else [pattern])
    missing = [p for p in out if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"No such .npz file(s): {missing[:3]}")
    if not out:
        raise FileNotFoundError(f"No .npz matched: {patterns}")
    return sorted(out)


def collate_fn(batch):
    return {
        "stream_a": torch.stack([b["stream_a"] for b in batch], dim=0),
        "stream_b": torch.stack([b["stream_b"] for b in batch], dim=0),
        "label_reg": torch.stack([b["label_reg"] for b in batch], dim=0),
        "row": torch.tensor([b["row"] for b in batch]),
        "col": torch.tensor([b["col"] for b in batch]),
    }


@torch.no_grad()
def predict(model, loader, device, desc=""):
    preds, targets, rows, cols = [], [], [], []
    total = len(loader)
    for i, batch in enumerate(loader):
        stream_a = batch["stream_a"].to(device, non_blocking=True)
        stream_b = batch["stream_b"].to(device, non_blocking=True)
        out = model(stream_a, stream_b).detach().float().cpu().view(-1)
        preds.append(out)
        targets.append(batch["label_reg"].float().view(-1))
        rows.append(batch["row"])
        cols.append(batch["col"])
        if desc and (i % 50 == 0 or i == total - 1):
            print(f"\r  {desc}: {i + 1}/{total} batches", end="", flush=True)
    if desc:
        print()
    return (torch.cat(preds).numpy(), torch.cat(targets).numpy(),
            torch.cat(rows).numpy(), torch.cat(cols).numpy())


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Setup] device={device}")

    # ---------------------------------------------------------------- #
    # Checkpoint
    # ---------------------------------------------------------------- #
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    for key in ("model_config", "model_state_dict"):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing '{key}'. Expected output of train.py.")

    target_transform = ckpt.get("target_transform", "none")
    feature_stats = ckpt.get("feature_stats", None)
    split_info = ckpt.get("split_info", None)

    model = MSCANet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"[Model] epoch={ckpt.get('epoch', '?')} "
          f"best_{ckpt.get('select_metric', '?')}={ckpt.get('best_score', float('nan')):.4f} "
          f"target_transform={target_transform}")

    if target_transform != "log1p":
        print(f"[Warning] target_transform is '{target_transform}', not 'log1p'. "
              f"No inverse transform will be applied.")

    # ---------------------------------------------------------------- #
    # Dataset -- ALL cells, no split filtering
    # ---------------------------------------------------------------- #
    paths = expand_paths(args.data_path)
    crop_size = split_info["crop_size"] if split_info else 16
    dataset = SpatialGridDataset(npz_paths=paths, crop_size=crop_size)

    height, width = dataset.monthly_tensors[0]["label_reg"].shape
    dates = [t["date"] for t in dataset.monthly_tensors]
    n_cells = len({(r, c) for _, r, c in dataset.samples})
    print(f"[Data] {len(paths)} months | grid {height}x{width} | "
          f"{n_cells} valid cells | {len(dataset)} samples")

    # Cell-mean on the RAW label, before any transform (stratification needs it).
    cell_mean_label = None
    if split_info is not None and split_info.get("stratify", True):
        cell_mean_label = compute_cell_mean_label(dataset)

    dataset.apply_target_transform(target_transform)

    # ---------------------------------------------------------------- #
    # Split tags -- inference covers everything, but we record provenance
    # ---------------------------------------------------------------- #
    split_of_cell = {}
    train_subset = None
    if split_info is not None and not args.no_split_tags:
        trained_files = [os.path.basename(p) for p in split_info["npz_files"]]
        current_files = [p.name for p in dataset.npz_paths]
        if trained_files != current_files:
            print("[Warning] .npz list differs from training; split tags will be "
                  "'unknown'. Inference itself is unaffected.")
        elif split_info["split_mode"] == "random":
            print("[Warning] The checkpoint used split_mode='random', which "
                  "assigns individual samples rather than cells, so a cell has "
                  "no single split. Tags are omitted. Note this map cannot "
                  "support a held-out claim: that model saw most cells.")
        else:
            train_subset, val_subset, test_subset = build_splits_v2(
                dataset,
                mode=split_info["split_mode"],
                cell_mean_label=cell_mean_label,
                val_ratio=split_info["val_ratio"],
                test_ratio=split_info["test_ratio"],
                seed=split_info["seed"],
                n_strata=split_info.get("n_strata", 10),
                block_size=split_info.get("block_size", 24),
                buffer=split_info.get("buffer", 16),
                verbose=False,
            )
            for name, sub in [("train", train_subset), ("val", val_subset),
                              ("test", test_subset)]:
                for i in sub.indices:
                    _, r, c = dataset.samples[i]
                    split_of_cell[(r, c)] = name
            counts = {n: sum(1 for v in split_of_cell.values() if v == n)
                      for n in ("train", "val", "test")}
            print(f"[Split] cells tagged: {counts} "
                  f"(dropped/buffer: {n_cells - sum(counts.values())})")

    # ---------------------------------------------------------------- #
    # Normalization -- from the checkpoint, never recomputed
    # ---------------------------------------------------------------- #
    if feature_stats is not None:
        dataset.apply_normalization(feature_stats)
    else:
        print("[Warning] Checkpoint has no feature_stats; inputs left unnormalized.")

    context_mode = ckpt.get("context_mode", "full")
    if context_mode != "full":
        dataset.set_context_mode(context_mode)
        print(f"[Warning] This is an ablation checkpoint (context={context_mode}); "
              f"its map is for the ablation section, not the headline figure.")

    # ---------------------------------------------------------------- #
    # Inference over the full grid
    # ---------------------------------------------------------------- #
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False, collate_fn=collate_fn)
    preds, targets, rows, cols = predict(model, loader, device, desc="inference")

    # Recover the month index for each sample. This is only valid because the
    # loader ran with shuffle=False and drop_last=False, so its output order is
    # exactly dataset.samples order. Assert it rather than trust it -- a silent
    # mismatch here would scatter every prediction to the wrong cell and the
    # map would look plausible while being wrong.
    file_idx = np.array([s[0] for s in dataset.samples], dtype=np.int64)
    expected_rows = np.array([s[1] for s in dataset.samples], dtype=np.int64)
    expected_cols = np.array([s[2] for s in dataset.samples], dtype=np.int64)
    if not (np.array_equal(rows, expected_rows)
            and np.array_equal(cols, expected_cols)):
        raise RuntimeError(
            "DataLoader output order does not match dataset.samples. Cannot "
            "map predictions back to cells. Set --num_workers 0 and retry."
        )
    date_col = np.array([dates[f] for f in file_idx])

    # ---------------------------------------------------------------- #
    # Duan smearing from TRAIN residuals only
    # ---------------------------------------------------------------- #
    smear = 1.0
    smear_rejected = None
    if target_transform == "log1p" and not args.no_smearing:
        if train_subset is None:
            print("[Smearing] Train split unavailable -> skipping correction.")
        else:
            tr_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True, collate_fn=collate_fn)
            tr_pred, tr_true, _, _ = predict(model, tr_loader, device,
                                             desc="train residuals")
            raw_smear = duan_smearing(tr_pred, tr_true)
            print(f"[Smearing] Duan factor = {raw_smear:.4f}")
            if 0.5 < raw_smear < 5.0:
                smear = raw_smear
            else:
                # Duan's estimator averages expm1(residual); when log-space
                # residuals are large it is dominated by a few exponentiated
                # outliers and is meaningless. Multiplying the whole map by it
                # would be worse than not correcting at all, so refuse.
                smear = 1.0
                smear_rejected = raw_smear
                print("[Smearing] REJECTED: a factor this far from 1.0 means "
                      "the log-space residuals are large, so expm1() of them "
                      "explodes and the estimator is dominated by a few "
                      "outliers. Falling back to smearing=1.0 (uncorrected). "
                      "The original-scale bias below is therefore real and "
                      "should be reported, not silently scaled away.")

    # ---------------------------------------------------------------- #
    # Invert to the original carbon scale
    # ---------------------------------------------------------------- #
    if target_transform == "log1p":
        pred_orig = np.expm1(preds.astype(np.float64)) * smear
        true_orig = np.expm1(targets.astype(np.float64))
    else:
        pred_orig = preds.astype(np.float64)
        true_orig = targets.astype(np.float64)
    pred_orig = np.maximum(pred_orig, 0.0)  # emissions cannot be negative

    split_col = np.array([split_of_cell.get((r, c), "unknown")
                          for r, c in zip(rows, cols)])

    # ---------------------------------------------------------------- #
    # 1. Long table
    # ---------------------------------------------------------------- #
    long_df = pd.DataFrame({
        "date": date_col,
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "split": split_col,
        "pred": pred_orig,
        "true": true_orig,
        "residual": pred_orig - true_orig,
        "pred_log1p": preds.astype(np.float64),
        "true_log1p": targets.astype(np.float64),
    })
    long_path = os.path.join(args.output_dir, "predictions_long.csv")
    long_df.to_csv(long_path, index=False, float_format=args.float_format)
    print(f"[Saved] {long_path}  ({len(long_df)} rows)")

    # ---------------------------------------------------------------- #
    # 2. Per-cell table -- the static map
    # ---------------------------------------------------------------- #
    cell_df = (
        long_df.groupby(["row", "col"], as_index=False)
        .agg(split=("split", "first"),
             pred_mean=("pred", "mean"), pred_median=("pred", "median"),
             true_mean=("true", "mean"), true_median=("true", "median"),
             pred_std=("pred", "std"), n_months=("pred", "size"))
    )
    cell_df["residual_mean"] = cell_df["pred_mean"] - cell_df["true_mean"]
    cell_df["abs_pct_error"] = (
        (cell_df["pred_mean"] - cell_df["true_mean"]).abs()
        / cell_df["true_mean"].clip(lower=1e-9)
    )
    # Rank-based hotspot flags -- no arbitrary emission threshold needed.
    cell_df["pred_rank_pct"] = cell_df["pred_mean"].rank(pct=True)
    cell_df["true_rank_pct"] = cell_df["true_mean"].rank(pct=True)
    cell_df["pred_top10"] = cell_df["pred_rank_pct"] >= 0.90
    cell_df["true_top10"] = cell_df["true_rank_pct"] >= 0.90

    cell_path = os.path.join(args.output_dir, "predictions_cell.csv")
    cell_df.to_csv(cell_path, index=False, float_format=args.float_format)
    print(f"[Saved] {cell_path}  ({len(cell_df)} cells)")

    # ---------------------------------------------------------------- #
    # 3. Dense grids -- straight into imshow / rasterio
    # ---------------------------------------------------------------- #
    n_months = len(dates)
    pred_grid = np.full((n_months, height, width), np.nan, np.float32)
    true_grid = np.full((n_months, height, width), np.nan, np.float32)
    pred_grid[file_idx, rows, cols] = pred_orig
    true_grid[file_idx, rows, cols] = true_orig

    mask_grid = np.zeros((height, width), np.uint8)
    mask_grid[rows, cols] = 1

    split_grid = np.zeros((height, width), np.int8)  # 0 unknown/dropped
    code = {"train": 1, "val": 2, "test": 3}
    for (r, c), name in split_of_cell.items():
        split_grid[r, c] = code[name]

    # Cells outside the mask are NaN in every month, so nanmean over them warns
    # about an empty slice. NaN is the correct output there -- it is what keeps
    # invalid cells blank on the map -- so the warning is expected noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice")
        pred_mean_map = np.nanmean(pred_grid, axis=0).astype(np.float32)
        true_mean_map = np.nanmean(true_grid, axis=0).astype(np.float32)

    grid_path = os.path.join(args.output_dir, "prediction_grids.npz")
    np.savez_compressed(
        grid_path,
        pred_grid=pred_grid, true_grid=true_grid,
        pred_mean_map=pred_mean_map,
        true_mean_map=true_mean_map,
        mask=mask_grid, split_grid=split_grid,
        dates=np.array(dates), height=height, width=width,
        smearing=np.array(smear), split_code=np.array(
            ["unknown", "train", "val", "test"]),
    )
    print(f"[Saved] {grid_path}  (pred_grid {pred_grid.shape})")

    # ---------------------------------------------------------------- #
    # 4. Summary
    # ---------------------------------------------------------------- #
    summary = {
        "checkpoint": args.checkpoint,
        "n_files": len(paths), "n_cells": int(n_cells),
        "n_samples": len(dataset),
        "grid": [int(height), int(width)],
        "target_transform": target_transform,
        "smearing": float(smear),
        "smearing_rejected_value": smear_rejected,
        "split_info": split_info,
        "overall_log1p": full_metrics(preds, targets),
        "overall_original": full_metrics(pred_orig, true_orig),
        "by_split": {},
    }

    print("\n" + "=" * 66)
    print("FULL-GRID METRICS -- ORIGINAL CARBON SCALE (all cells)")
    print("=" * 66)
    print_metrics(summary["overall_original"])

    for name in ("train", "val", "test", "unknown"):
        sel = split_col == name
        if not sel.any():
            continue
        m_log = full_metrics(preds[sel], targets[sel])
        m_orig = full_metrics(pred_orig[sel], true_orig[sel])
        summary["by_split"][name] = {"n_samples": int(sel.sum()),
                                     "log1p": m_log, "original": m_orig}
        print(f"\n--- split='{name}'  (n={int(sel.sum())}) ---")
        print(f"  log1p space : R2={m_log['r2']:.4f}  MAE={m_log['mae']:.4f}")
        print(f"  original    : R2={m_orig['r2']:.4f}  MAE={m_orig['mae']:.4f}  "
              f"bias={m_orig['bias']:+.4f}")

    if "test" in summary["by_split"] and "train" in summary["by_split"]:
        gap = (summary["by_split"]["train"]["original"]["r2"]
               - summary["by_split"]["test"]["original"]["r2"])
        summary["train_test_r2_gap_original"] = float(gap)
        print(f"\n[Gap] train R2 - test R2 (original space) = {gap:+.4f}")

    total_pred = float(np.nansum(pred_grid))
    total_true = float(np.nansum(true_grid))
    summary["regional_total_pred"] = total_pred
    summary["regional_total_true"] = total_true
    summary["regional_total_error_pct"] = float(
        (total_pred - total_true) / max(abs(total_true), 1e-9))
    print("\n[Total] regional sum over all cells and months:")
    print(f"  predicted = {total_pred:.6g}")
    print(f"  true      = {total_true:.6g}")
    print(f"  error     = {summary['regional_total_error_pct']:+.2%}")

    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n[Saved] {summary_path}")

    print("\nWhen you draw the map, use the 'split' column: cells tagged")
    print("'train' were seen during fitting and cells tagged 'test' were not.")
    print("A single map over both mixes two different claims.")


if __name__ == "__main__":
    main()
