"""
================================================================================
diagnostics.py -- Data diagnostics that justify the pipeline's design choices
================================================================================

Run this once and keep the output. Every non-obvious decision in this codebase
(cell-fixed splits, log1p target, feature standardization, cluster bootstrap)
traces back to one of these numbers, and reviewers will ask.

WHAT IT REPORTS
---------------
1. Mask consistency across months.
2. Label distribution: skew, kurtosis, quantiles, zero fraction, max/median.
   -> decides whether log1p is needed.
3. Variance decomposition: within-cell vs between-cell, ICC.
   -> decides whether cell positions must be fixed across splits.
4. Cell-mean predictor R^2: the score a model gets for free under a per-sample
   random split, by memorising location.
5. Static-feature audit: which channels never change month to month (they act
   as location fingerprints).
6. Neighbour label correlation vs lag: how much two adjacent cells' labels
   resemble each other, which tells you whether overlapping 16x16 patches
   create near-duplicate samples.
7. Feature scale audit: the per-channel ranges the network sees raw.

USAGE
-----
    python diagnostics.py --data_path /path/to/*.npz --save_json diag.json
================================================================================
"""

import argparse
import json

import numpy as np

try:
    from scipy.stats import kurtosis, skew
    _SCIPY = True
except ImportError:
    _SCIPY = False

FEATURE_NAMES_A = ["no2_mean", "so2_mean", "co_mean"]
FEATURE_NAMES_B = ["nightlight_mean", "urban_fraction",
                   "power_plant_count", "fossil_capacity_mw"]


def parse_args():
    p = argparse.ArgumentParser(description="Diagnostics for the MS-CA tensors.")
    p.add_argument("--data_path", type=str, nargs="+", required=True)
    p.add_argument("--crop_size", type=int, default=16,
                   help="Only used to interpret the lag report.")
    p.add_argument("--save_json", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(args.data_path)
    print(f"[Load] {len(paths)} files")

    labels, masks, streams_a, streams_b = [], [], [], []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        labels.append(data["label_reg"])
        masks.append(data["mask"] == 1)
        streams_a.append(data["stream_a"])
        streams_b.append(data["stream_b"])

    labels = np.stack(labels)          # [T, H, W]
    masks = np.stack(masks)            # [T, H, W]
    streams_a = np.stack(streams_a)    # [T, 3, H, W]
    streams_b = np.stack(streams_b)    # [T, 4, H, W]
    n_months, height, width = labels.shape
    mask0 = masks[0]

    results = {"n_months": n_months, "height": height, "width": width}

    # ---------------------------------------------------------------- #
    # 1. Mask consistency
    # ---------------------------------------------------------------- #
    consistent = bool(all((masks[t] == mask0).all() for t in range(n_months)))
    print(f"\n[1] Mask identical across all months: {consistent}")
    print(f"    valid cells: {int(mask0.sum())} / {height * width}")
    results["mask_consistent"] = consistent
    results["n_valid_cells"] = int(mask0.sum())

    if not consistent:
        print("    !! Masks differ by month. The cell-split logic assumes a")
        print("       stable grid; check your preprocessing before continuing.")

    # ---------------------------------------------------------------- #
    # 2. Label distribution
    # ---------------------------------------------------------------- #
    values = labels[masks]
    print(f"\n[2] label_reg distribution  (n={values.size})")
    print(f"    min={values.min():.4f}  max={values.max():.4f}")
    print(f"    mean={values.mean():.4f}  median={np.median(values):.4f}  "
          f"std={values.std():.4f}")
    if _SCIPY:
        sk, ku = float(skew(values)), float(kurtosis(values))
        print(f"    skew={sk:.2f}  kurtosis={ku:.2f}")
        results["skew"] = sk
        results["kurtosis"] = ku
    ratio = float(values.max() / max(np.median(values), 1e-9))
    print(f"    max/median={ratio:.1f}   zeros={float((values == 0).mean()):.3%}")
    for q in [50, 90, 95, 99, 99.9]:
        print(f"    p{q}: {np.percentile(values, q):.4f}")

    results.update({
        "label_min": float(values.min()), "label_max": float(values.max()),
        "label_mean": float(values.mean()),
        "label_median": float(np.median(values)),
        "label_std": float(values.std()), "max_over_median": ratio,
        "zero_fraction": float((values == 0).mean()),
    })

    log_values = np.log1p(np.maximum(values, 0))
    if _SCIPY:
        print(f"    after log1p -> skew={float(skew(log_values)):.2f}  "
              f"kurtosis={float(kurtosis(log_values)):.2f}  "
              f"std={log_values.std():.3f}")
        results["log_skew"] = float(skew(log_values))

    if _SCIPY and (results.get("skew", 0) > 2 or ratio > 100):
        print("    => log1p transform is required. On the raw scale an L1")
        print("       objective learns the conditional median and abandons")
        print("       every large emitter.")

    # ---------------------------------------------------------------- #
    # 3. Variance decomposition
    # ---------------------------------------------------------------- #
    cells = labels[:, mask0]                    # [T, n_cells]
    within = float(cells.var(axis=0).mean())
    between = float(cells.mean(axis=0).var())
    icc = between / (between + within) if (between + within) > 0 else float("nan")
    print("\n[3] Variance decomposition")
    print(f"    within-cell  var = {within:.4f}   (month-to-month)")
    print(f"    between-cell var = {between:.4f}   (cell-to-cell)")
    print(f"    ICC              = {icc:.4f}")
    results.update({"within_var": within, "between_var": between, "icc": icc})

    if icc > 0.9:
        print("    => Location explains almost all label variance. A per-sample")
        print("       random split lets a model identify the cell and look the")
        print("       answer up from its other months in train. Cell positions")
        print("       MUST be fixed across splits (--split_mode cell).")
        print(f"    => Effective sample size is ~{int(mask0.sum())} cells, not")
        print(f"       {values.size} samples. Bootstrap over cells.")

    # ---------------------------------------------------------------- #
    # 4. Cell-mean predictor
    # ---------------------------------------------------------------- #
    pred = np.repeat(cells.mean(axis=0)[None, :], n_months, axis=0)
    ss_res = float(((cells - pred) ** 2).sum())
    ss_tot = float(((cells - cells.mean()) ** 2).sum())
    r2_cellmean = 1.0 - ss_res / ss_tot
    print(f"\n[4] Cell-mean predictor R^2 = {r2_cellmean:.4f}")
    print("    This is the score a model gets for FREE under a per-sample")
    print("    random split, purely by memorising location. It vanishes under")
    print("    a cell-fixed split, because a held-out cell has no history.")
    results["cell_mean_r2"] = r2_cellmean

    # ---------------------------------------------------------------- #
    # 5. Static-feature audit
    # ---------------------------------------------------------------- #
    print("\n[5] Temporal variability by channel")
    print(f"    {'channel':<22}{'cells that change':>20}")
    static_report = {}
    for stream, names in [(streams_a, FEATURE_NAMES_A), (streams_b, FEATURE_NAMES_B)]:
        for i, name in enumerate(names):
            channel = stream[:, i][:, mask0]           # [T, n_cells]
            frac = float((channel.std(axis=0) > 1e-6).mean())
            static_report[name] = frac
            flag = "  <- static: a location fingerprint" if frac < 0.05 else ""
            print(f"    {name:<22}{frac:>19.2%}{flag}")
    results["temporal_variability"] = static_report

    # ---------------------------------------------------------------- #
    # 6. Neighbour label correlation
    # ---------------------------------------------------------------- #
    print("\n[6] Neighbour correlation of cell-mean label (log space)")
    print("    Patches of two cells at Chebyshev distance d < crop_size share")
    print(f"    pixels; at crop_size={args.crop_size}, d=1 means 93.75% overlap.")
    cell_mean_map = np.log1p(np.maximum(labels.mean(axis=0), 0))
    lag_report = {}
    for lag in [1, 2, 4, 8, 16, 24]:
        if lag >= min(height, width):
            continue
        mh = mask0[:, :-lag] & mask0[:, lag:]
        rh = float(np.corrcoef(cell_mean_map[:, :-lag][mh],
                               cell_mean_map[:, lag:][mh])[0, 1]) if mh.any() else float("nan")
        mv = mask0[:-lag, :] & mask0[lag:, :]
        rv = float(np.corrcoef(cell_mean_map[:-lag, :][mv],
                               cell_mean_map[lag:, :][mv])[0, 1]) if mv.any() else float("nan")
        lag_report[lag] = {"horizontal": rh, "vertical": rv}
        print(f"    lag={lag:2d}   horizontal r={rh:.3f}   vertical r={rv:.3f}")
    results["lag_correlation"] = lag_report

    r1 = lag_report.get(1, {}).get("horizontal", float("nan"))
    r1_is_valid = not np.isnan(r1)
    if r1_is_valid and r1 > 0.7:
        print("    => Adjacent cells are near-identical in label. Under a")
        print("       cell-fixed split their patches still overlap heavily, so")
        print("       train and test contain near-duplicate inputs. Report")
        print("       --split_mode block alongside cell as a robustness check.")
    elif r1_is_valid and r1 < 0.3:
        print("    => Little neighbour structure. A plain cell-fixed split is")
        print("       sufficient; no spatial buffer needed.")

    # ---------------------------------------------------------------- #
    # 7. Feature scales
    # ---------------------------------------------------------------- #
    print("\n[7] Raw feature scales (valid cells, all months)")
    print(f"    {'channel':<22}{'mean':>14}{'std':>14}{'max':>14}")
    scale_report = {}
    for stream, names in [(streams_a, FEATURE_NAMES_A), (streams_b, FEATURE_NAMES_B)]:
        for i, name in enumerate(names):
            channel = stream[:, i][:, mask0].ravel()
            scale_report[name] = {"mean": float(channel.mean()),
                                  "std": float(channel.std()),
                                  "max": float(channel.max())}
            print(f"    {name:<22}{channel.mean():>14.6g}"
                  f"{channel.std():>14.6g}{channel.max():>14.6g}")
    results["feature_scales"] = scale_report

    magnitudes = [abs(v["mean"]) for v in scale_report.values() if v["mean"] != 0]
    if magnitudes and max(magnitudes) / min(magnitudes) > 1e3:
        span = np.log10(max(magnitudes) / min(magnitudes))
        print(f"    => Channel means span ~{span:.0f} orders of magnitude. Without")
        print("       standardization the patch-embedding Conv2d responds almost")
        print("       only to the largest channel. Keep --normalize on.")

    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"\n[Saved] {args.save_json}")


if __name__ == "__main__":
    main()
