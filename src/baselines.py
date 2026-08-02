"""
================================================================================
baselines.py -- Reference models on the IDENTICAL split as train.py
================================================================================

WHY THIS MATTERS MORE THAN TUNING MS-CA
---------------------------------------
An R^2 of 0.85 means nothing on its own. It only becomes a claim once you know
what a trivial model scores on the same held-out cells.

  B0  Train-mean constant       -> defines R^2 = 0.
  B1  fossil_capacity_mw only   -> power_plant_count is 100% static across all
                                   36 months and fossil_capacity_mw changes in
                                   0.02% of cells. They are effectively
                                   location fingerprints. If a single-variable
                                   regression on fossil capacity already gets
                                   most of the signal, the study is really
                                   "we found the power plants", and the seven
                                   proxies are not doing the work.
  B2  7 centre-cell features, linear
  B3  7 centre-cell features, LightGBM (or sklearn GBT fallback)
                                -> B3 IS THE ONE THAT MATTERS. The only
                                   justification for a ViT + cross-attention
                                   stack is that the 16x16 spatial context
                                   carries information the centre cell's seven
                                   numbers do not. If B3 matches MS-CA, the
                                   architecture is not earning its complexity.

The headline contribution of the paper is (MS-CA R^2) - (B3 R^2) on the same
split. +0.05 or more is a real result. +0.01 is not worth a transformer.

Note that B1' from the earlier discussion (cell-mean predictor) disappears
under cell-fixed splits: a test cell was never seen in training, so it has no
historical mean. Removing that free 0.9915 R^2 is precisely the point of the
cell split.

USAGE
-----
    python baselines.py --data_path /path/to/*.npz --split_mode cell \\
        --target_transform log1p --seed 42
================================================================================
"""

import argparse
import json

import numpy as np

from metrics import full_metrics, print_metrics
from spatial_grid_dataset import SpatialGridDataset
from splits_v2 import build_splits_v2, compute_cell_mean_label

FEATURE_NAMES = ["no2_mean", "so2_mean", "co_mean", "nightlight_mean",
                 "urban_fraction", "power_plant_count", "fossil_capacity_mw"]
FOSSIL_IDX = 6


def parse_args():
    p = argparse.ArgumentParser(description="Baselines on the MS-CA split.")
    p.add_argument("--data_path", type=str, nargs="+", required=True)
    p.add_argument("--crop_size", type=int, default=16)
    p.add_argument("--target_transform", type=str, default="log1p",
                   choices=["none", "log1p"])
    p.add_argument("--split_mode", type=str, default="cell",
                   choices=["cell", "block", "random"])
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_stratify", action="store_true")
    p.add_argument("--n_strata", type=int, default=10)
    p.add_argument("--block_size", type=int, default=24)
    p.add_argument("--buffer", type=int, default=16)
    p.add_argument("--save_json", type=str, default=None)
    return p.parse_args()


def extract_centre_features(dataset, subset):
    """Centre-cell feature matrix [N, 7] and target vector [N] for a Subset."""
    x = np.empty((len(subset), 7), dtype=np.float32)
    y = np.empty(len(subset), dtype=np.float32)
    for j, i in enumerate(subset.indices):
        file_idx, row, col = dataset.samples[i]
        tensors = dataset.monthly_tensors[file_idx]
        x[j, :3] = tensors["stream_a"][:, row, col]
        x[j, 3:] = tensors["stream_b"][:, row, col]
        y[j] = tensors["label_reg"][row, col]
    return x, y


def main():
    args = parse_args()

    dataset = SpatialGridDataset(npz_paths=args.data_path, crop_size=args.crop_size)
    cell_mean_label = None if args.no_stratify else compute_cell_mean_label(dataset)
    dataset.apply_target_transform(args.target_transform)

    train_subset, val_subset, test_subset = build_splits_v2(
        dataset, mode=args.split_mode, cell_mean_label=cell_mean_label,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
        n_strata=args.n_strata, block_size=args.block_size, buffer=args.buffer,
    )

    x_train, y_train = extract_centre_features(dataset, train_subset)
    x_val, y_val = extract_centre_features(dataset, val_subset)
    x_test, y_test = extract_centre_features(dataset, test_subset)
    print(f"[Data] train={x_train.shape} val={x_val.shape} test={x_test.shape}")

    # Standardize features from TRAIN only (matters for the linear models).
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-8
    x_train_s = (x_train - mu) / sd
    x_test_s = (x_test - mu) / sd

    results = {}
    preds_store = {}

    # ---------------------------------------------------------------- #
    # B0: constant train mean
    # ---------------------------------------------------------------- #
    pred = np.full_like(y_test, y_train.mean())
    preds_store["B0_train_mean"] = pred
    results["B0_train_mean"] = full_metrics(pred, y_test)
    print("\n" + "=" * 66)
    print("B0 -- constant (train mean)")
    print("=" * 66)
    print_metrics(results["B0_train_mean"])

    # ---------------------------------------------------------------- #
    # B1: fossil_capacity_mw only
    # ---------------------------------------------------------------- #
    from sklearn.linear_model import LinearRegression

    lr1 = LinearRegression().fit(x_train_s[:, [FOSSIL_IDX]], y_train)
    pred1 = lr1.predict(x_test_s[:, [FOSSIL_IDX]])
    preds_store["B1_fossil_only"] = pred1
    results["B1_fossil_only"] = full_metrics(pred1, y_test)
    print("\n" + "=" * 66)
    print("B1 -- linear on fossil_capacity_mw alone")
    print("=" * 66)
    print_metrics(results["B1_fossil_only"])
    if results["B1_fossil_only"]["r2"] > 0.9:
        print("\n  !! R^2 > 0.9 from one static variable. The task may reduce to")
        print("     locating power plants. Raise this with your advisor before")
        print("     investing more GPU time in the architecture.")

    # ---------------------------------------------------------------- #
    # B2: all 7 centre features, linear
    # ---------------------------------------------------------------- #
    lr2 = LinearRegression().fit(x_train_s, y_train)
    pred2 = lr2.predict(x_test_s)
    preds_store["B2_linear_7"] = pred2
    results["B2_linear_7"] = full_metrics(pred2, y_test)
    print("\n" + "=" * 66)
    print("B2 -- linear on 7 centre-cell features")
    print("=" * 66)
    print_metrics(results["B2_linear_7"])
    print("\n  coefficients (standardized):")
    for name, coef in sorted(zip(FEATURE_NAMES, lr2.coef_),
                             key=lambda kv: -abs(kv[1])):
        print(f"    {name:22s} {coef:+.4f}")

    # ---------------------------------------------------------------- #
    # B3: gradient boosting on the same 7 features  <- the real bar
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 66)
    print("B3 -- gradient boosting on 7 centre-cell features")
    print("=" * 66)
    try:
        import lightgbm as lgb

        gbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05,
                                num_leaves=63, subsample=0.8,
                                colsample_bytree=0.8, random_state=args.seed,
                                verbose=-1)
        gbm.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(50, verbose=False)])
        pred3 = gbm.predict(x_test)
        importances = dict(zip(FEATURE_NAMES,
                               [float(v) for v in gbm.feature_importances_]))
        backend = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        print("  (lightgbm unavailable -- falling back to sklearn "
              "HistGradientBoostingRegressor)")
        gbm = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                            early_stopping=True,
                                            random_state=args.seed)
        gbm.fit(x_train, y_train)
        pred3 = gbm.predict(x_test)
        importances = None
        backend = "sklearn_hgbr"

    preds_store["B3_gbm_7"] = pred3
    results["B3_gbm_7"] = full_metrics(pred3, y_test)
    results["B3_backend"] = backend
    print_metrics(results["B3_gbm_7"])
    if importances:
        print("\n  feature importances:")
        for name, value in sorted(importances.items(), key=lambda kv: -kv[1]):
            print(f"    {name:22s} {value:.0f}")

    # ---------------------------------------------------------------- #
    # Summary
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 66)
    print(f"BASELINE SUMMARY  (split_mode={args.split_mode}, "
          f"transform={args.target_transform}, seed={args.seed})")
    print("=" * 66)
    print(f"{'model':<24}{'R^2':>10}{'MAE':>10}{'Spearman':>12}{'Top10Rec':>10}")
    for key in ["B0_train_mean", "B1_fossil_only", "B2_linear_7", "B3_gbm_7"]:
        m = results[key]
        print(f"{key:<24}{m['r2']:>10.4f}{m['mae']:>10.4f}"
              f"{m['spearman']:>12.4f}{m['top10_recall']:>10.4f}")
    print("\nNow compare MS-CA against B3. That difference is the contribution.")

    # ---------------------------------------------------------------- #
    # Original-space summary -- the numbers for a physical-units table.
    # These invert both predictions and targets with expm1; no smearing is
    # applied here, so report them alongside the same treatment of MS-CA
    # (evaluation.py without --smearing) for a like-for-like comparison.
    # ---------------------------------------------------------------- #
    if args.target_transform == "log1p":
        y_orig = np.expm1(y_test.astype(np.float64))
        print("\n" + "=" * 66)
        print("ORIGINAL-SPACE SUMMARY (expm1, no smearing)")
        print("=" * 66)
        print(f"{'model':<24}{'R^2':>10}{'MAE':>12}{'Spearman':>12}{'Top10Rec':>10}")
        for key in ["B0_train_mean", "B1_fossil_only", "B2_linear_7", "B3_gbm_7"]:
            m = full_metrics(np.expm1(preds_store[key].astype(np.float64)), y_orig)
            results[key + "_original_space"] = m
            print(f"{key:<24}{m['r2']:>10.4f}{m['mae']:>12.4f}"
                  f"{m['spearman']:>12.4f}{m['top10_recall']:>10.4f}")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\n[Saved] {args.save_json}")


if __name__ == "__main__":
    main()
