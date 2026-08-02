"""
================================================================================
evaluation.py -- Standalone test-set evaluation for the MS-CA regressor
================================================================================

Loads a trained ``best_model.pth`` from ``train.py``, rebuilds the IDENTICAL
held-out test split, and reports the full metric suite.

HOW THE SPLIT IS REPRODUCED
---------------------------
Everything needed lives inside the checkpoint: ``split_info`` (mode, ratios,
seed, stratification, block/buffer, crop_size), ``target_transform``, and
``feature_stats``. You do NOT re-specify them on the command line, which
removes the most common way of accidentally evaluating on a different split
than the one that was held out.

The feature statistics are READ FROM THE CHECKPOINT, never recomputed. They
were derived from train cells only; recomputing them here over whatever data
you pass would silently leak test statistics into the inputs.

TWO SPACES
----------
When ``target_transform='log1p'``, metrics are reported both in log space (the
space the model optimised) and in the original label space (the space the
scientific claim is about). Log-space R^2 always looks better; the original
scale is what belongs in the paper.

Back-transforming is biased because E[expm1(y_hat)] != expm1(E[y_hat])
(Jensen's inequality), so a model trained on log targets under-predicts on the
raw scale. ``--smearing`` applies Duan's smearing estimator, computed from the
TRAIN split residuals, to correct it. Check ``bias`` either way.

USAGE
-----
    python evaluation.py \\
        --checkpoint ./checkpoints/best_model.pth \\
        --data_path /path/to/*.npz
================================================================================
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import (
    bootstrap_ci_by_cell,
    duan_smearing,
    full_metrics,
    mae_score,
    print_metrics,
    r2_score,
)
from model import MSCANet
from spatial_grid_dataset import SpatialGridDataset
from splits_v2 import build_splits_v2, compute_cell_mean_label, verify_no_cell_overlap


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MS-CA regressor on the held-out test split."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="best_model.pth saved by train.py.")

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_path", type=str, nargs="+", default=None)
    data_group.add_argument("--tensor_index_csv", type=str, default=None)

    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val", "train"],
                        help="Which split to evaluate. Keep 'test' for the "
                             "final number; open it once.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--smearing", action="store_true",
                        help="Apply Duan's smearing correction (train residuals) "
                             "when back-transforming log1p predictions.")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Cluster-bootstrap iterations over CELLS. 0 disables.")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Optional path to dump all metrics as JSON.")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Optional .npz path for per-sample predictions.")
    return parser.parse_args()


# ==============================================================================
# COLLATE
# ==============================================================================
def eval_collate_fn(batch):
    return {
        "stream_a": torch.stack([b["stream_a"] for b in batch], dim=0),
        "stream_b": torch.stack([b["stream_b"] for b in batch], dim=0),
        "label_reg": torch.stack([b["label_reg"] for b in batch], dim=0),
        "row": torch.tensor([b["row"] for b in batch]),
        "col": torch.tensor([b["col"] for b in batch]),
    }


# ==============================================================================
# MODEL LOADING
# ==============================================================================
def load_checkpoint_bundle(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    for key in ("model_config", "model_state_dict"):
        if key not in ckpt:
            raise KeyError(
                f"Checkpoint is missing '{key}'. Expected a best_model.pth "
                f"produced by this version of train.py."
            )
    if "split_info" not in ckpt:
        raise KeyError(
            "Checkpoint has no 'split_info'. It was produced by an older "
            "train.py whose split cannot be reproduced. Retrain."
        )

    model = MSCANet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[Model] Loaded '{path}' | epoch={ckpt.get('epoch', '?')} | "
          f"best {ckpt.get('select_metric', '?')}={ckpt.get('best_score', float('nan')):.4f}")
    return model, ckpt


# ==============================================================================
# INFERENCE
# ==============================================================================
@torch.no_grad()
def predict(model, loader, device):
    preds, targets, rows, cols = [], [], [], []
    for batch in loader:
        stream_a = batch["stream_a"].to(device, non_blocking=True)
        stream_b = batch["stream_b"].to(device, non_blocking=True)
        out = model(stream_a, stream_b).detach().float().cpu().view(-1)
        preds.append(out)
        targets.append(batch["label_reg"].float().view(-1))
        rows.append(batch["row"])
        cols.append(batch["col"])
    return (torch.cat(preds).numpy(), torch.cat(targets).numpy(),
            torch.cat(rows).numpy(), torch.cat(cols).numpy())


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Setup] Using device: {device}")

    model, ckpt = load_checkpoint_bundle(args.checkpoint, device)
    split_info = ckpt["split_info"]
    target_transform = ckpt.get("target_transform", "none")
    feature_stats = ckpt.get("feature_stats", None)

    print(f"[Setup] split_mode={split_info['split_mode']} "
          f"seed={split_info['seed']} "
          f"val/test={split_info['val_ratio']}/{split_info['test_ratio']} "
          f"stratify={split_info.get('stratify')} "
          f"target_transform={target_transform}")

    # ---------------------------------------------------------------- #
    # Rebuild the dataset exactly as training saw it
    # ---------------------------------------------------------------- #
    dataset = SpatialGridDataset(
        npz_paths=args.data_path,
        tensor_index_csv=args.tensor_index_csv,
        crop_size=split_info["crop_size"],
    )

    trained_files = [os.path.basename(p) for p in split_info["npz_files"]]
    current_files = [p.name for p in dataset.npz_paths]
    if trained_files != current_files:
        raise ValueError(
            "The .npz file list differs from training, so the split cannot be "
            f"reproduced.\n  trained on: {len(trained_files)} files, "
            f"first={trained_files[:1]}\n  given now : {len(current_files)} "
            f"files, first={current_files[:1]}"
        )

    cell_mean_label = compute_cell_mean_label(dataset) \
        if split_info.get("stratify", True) else None

    dataset.apply_target_transform(target_transform)

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
    )
    if split_info["split_mode"] in ("cell", "block"):
        verify_no_cell_overlap(dataset, train_subset, val_subset, test_subset)

    for name, expected, subset in [("train", "n_train", train_subset),
                                   ("val", "n_val", val_subset),
                                   ("test", "n_test", test_subset)]:
        if expected in split_info and len(subset) != split_info[expected]:
            raise ValueError(
                f"Rebuilt {name} split has {len(subset)} samples but training "
                f"recorded {split_info[expected]}. The split did not reproduce."
            )
    print("[Verify] Split reproduced exactly (sample counts match training).")

    # Normalization stats come from the checkpoint -- never recomputed here.
    if feature_stats is not None:
        dataset.apply_normalization(feature_stats)

    # Ablation checkpoints must be evaluated under the same context condition
    # they were trained with, or the numbers are meaningless.
    context_mode = ckpt.get("context_mode", "full")
    if context_mode != "full":
        dataset.set_context_mode(context_mode)

    subset = {"train": train_subset, "val": val_subset, "test": test_subset}[args.split]
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False, collate_fn=eval_collate_fn)

    preds, targets, rows, cols = predict(model, loader, device)
    print(f"[Data] evaluated split='{args.split}' | samples={preds.size} | "
          f"cells={len({(r, c) for r, c in zip(rows, cols)})}")

    # ---------------------------------------------------------------- #
    # Smearing factor from TRAIN residuals (never from test)
    # ---------------------------------------------------------------- #
    smear = None
    if args.smearing and target_transform == "log1p":
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True, collate_fn=eval_collate_fn)
        tr_pred, tr_true, _, _ = predict(model, train_loader, device)
        smear = duan_smearing(tr_pred, tr_true)
        print(f"[Smearing] Duan factor from train residuals = {smear:.4f}")
        if not (0.5 < smear < 5.0):
            print("[Smearing] WARNING: a factor this far from 1.0 means the log-"
                  "space residuals are large, and expm1() of them explodes. The "
                  "correction is unreliable here -- treat the uncorrected "
                  "original-space numbers as primary and fix the underlying fit "
                  "first.")

    # ---------------------------------------------------------------- #
    # Metrics
    # ---------------------------------------------------------------- #
    results = {"split": args.split, "target_transform": target_transform,
               "split_info": split_info}

    train_space = full_metrics(preds, targets)
    space_name = "log1p" if target_transform == "log1p" else "original"
    print("\n" + "=" * 66)
    print(f"TEST-SET METRICS -- TRAINING SPACE ({space_name})")
    print("=" * 66)
    print_metrics(train_space)
    results["train_space"] = train_space

    if target_transform == "log1p":
        preds_orig = np.expm1(preds.astype(np.float64))
        targets_orig = np.expm1(targets.astype(np.float64))
        if smear is not None:
            preds_orig = preds_orig * smear
        orig_space = full_metrics(preds_orig, targets_orig)
        print("\n" + "=" * 66)
        print("TEST-SET METRICS -- ORIGINAL LABEL SPACE  <- report this one")
        print("=" * 66)
        print_metrics(orig_space)
        rel_bias = orig_space["bias"] / max(targets_orig.mean(), 1e-9)
        bias_hint = (
            "   <- >5%, consider --smearing"
            if abs(rel_bias) > 0.05 and smear is None
            else ""
        )
        print(f"  Relative bias     : {rel_bias:+.2%}{bias_hint}")
        results["orig_space"] = orig_space
        results["relative_bias"] = float(rel_bias)
        if smear is not None:
            results["smearing"] = float(smear)

    # ---------------------------------------------------------------- #
    # Cluster bootstrap over CELLS
    # ---------------------------------------------------------------- #
    if args.bootstrap > 0:
        width = dataset.monthly_tensors[0]["label_reg"].shape[1]
        cell_ids = rows.astype(np.int64) * width + cols.astype(np.int64)

        print("\n" + "=" * 66)
        print(f"95% CI -- cluster bootstrap over cells (n_boot={args.bootstrap})")
        print("=" * 66)
        print("Resampling cells, not samples: the 36 months of a cell are near")
        print("replicates (within-cell variance is 0.85% of total), so sample-")
        print("level bootstrap would understate the interval by about 6x.")
        for name, fn in [("R^2", r2_score), ("MAE", mae_score)]:
            point, lo, hi = bootstrap_ci_by_cell(preds, targets, cell_ids, fn,
                                                 n_boot=args.bootstrap)
            print(f"  {name:4s} = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
            results[f"ci_{name.replace('^', '')}"] = {"point": point, "lo": lo, "hi": hi}

    # ---------------------------------------------------------------- #
    # Outputs
    # ---------------------------------------------------------------- #
    if args.save_predictions:
        np.savez_compressed(args.save_predictions, pred=preds, true=targets,
                            row=rows, col=cols)
        print(f"\n[Saved] predictions -> {args.save_predictions}")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"[Saved] metrics -> {args.save_json}")


if __name__ == "__main__":
    main()
