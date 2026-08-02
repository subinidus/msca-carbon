"""
================================================================================
make_figures.py -- Paper figures and results tables from inference.py output
================================================================================

Consumes the artefacts written by ``inference.py`` (prediction_grids.npz,
predictions_cell.csv, inference_summary.json) and produces:

  fig_maps.png           True | Predicted | Residual, all valid cells,
                         p99-clipped colour scale
  fig_maps_test_only.png Same three panels restricted to HELD-OUT test cells.
                         This is the figure that supports a generalisation
                         claim; the all-cells map is illustration.
  fig_scatter.png        Per-cell mean predicted vs true on log axes, test
                         cells emphasised, y = x reference
  fig_hotspot.png        Top-10% hotspot agreement map (hit / false alarm /
                         miss), rank-based so no arbitrary threshold
  results_tables.md      Markdown tables ready to transcribe into the paper:
                         per-split metrics in both spaces, baseline comparison
                         (if --baselines_json given), regional totals

WHY P99 CLIPPING
----------------
label max is ~94,550 while p99 is ~1,206 (a 78x gap). On a linear colour
scale one power-plant cell saturates the map and everything else renders
black. Clipping the colour scale at p99 keeps the map readable; the clip
value is printed in the colourbar label so nothing is hidden.

USAGE
-----
    python make_figures.py \\
        --inference_dir /kaggle/working/inference \\
        --baselines_json /kaggle/working/baselines_cell.json \\
        --output_dir /kaggle/working/figures
================================================================================
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def parse_args():
    p = argparse.ArgumentParser(description="Figures + tables for the paper.")
    p.add_argument("--inference_dir", type=str, required=True,
                   help="Directory written by inference.py.")
    p.add_argument("--baselines_json", type=str, default=None,
                   help="Optional baselines.py --save_json output; adds the "
                        "model-comparison table.")
    p.add_argument("--output_dir", type=str, default="./figures")
    p.add_argument("--clip_pct", type=float, default=99.0,
                   help="Colour-scale clip percentile for the value maps.")
    p.add_argument("--dpi", type=int, default=250)
    return p.parse_args()


def _load(inference_dir):
    grids = np.load(os.path.join(inference_dir, "prediction_grids.npz"),
                    allow_pickle=False)
    cells = pd.read_csv(os.path.join(inference_dir, "predictions_cell.csv"))
    with open(os.path.join(inference_dir, "inference_summary.json")) as fh:
        summary = json.load(fh)
    return grids, cells, summary


def _masked(map2d, mask):
    out = map2d.astype(np.float64).copy()
    out[mask == 0] = np.nan
    return out


# ==============================================================================
# FIGURE 1 + 2: value maps
# ==============================================================================
def fig_maps(grids, args, test_only=False):
    true_map = grids["true_mean_map"].astype(np.float64)
    pred_map = grids["pred_mean_map"].astype(np.float64)
    mask = grids["mask"]

    if test_only:
        sel = (grids["split_grid"] == 3).astype(np.uint8)  # 3 == test
        if sel.sum() == 0:
            print("[fig] no test-tagged cells; skipping test-only map")
            return
        true_map, pred_map = _masked(true_map, sel), _masked(pred_map, sel)
        suffix, title_suffix = "_test_only", "  (held-out test cells only)"
    else:
        true_map, pred_map = _masked(true_map, mask), _masked(pred_map, mask)
        suffix, title_suffix = "", "  (all valid cells)"

    vmax = np.nanpercentile(true_map, args.clip_pct)
    resid = pred_map - true_map
    rlim = np.nanpercentile(np.abs(resid), args.clip_pct)

    value_cmap = plt.get_cmap("inferno").copy()
    value_cmap.set_bad("#d9d9d9")
    resid_cmap = plt.get_cmap("RdBu_r").copy()
    resid_cmap.set_bad("#d9d9d9")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    panels = [
        (axes[0], true_map, value_cmap, 0, vmax, "Ground truth"),
        (axes[1], pred_map, value_cmap, 0, vmax, "MS-CA prediction"),
        (axes[2], resid, resid_cmap, -rlim, rlim, "Residual (pred - true)"),
    ]
    for ax, data, cmap, vmin, vmx, title in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmx,
                       interpolation="nearest")
        ax.set_title(title + title_suffix, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        clip_note = (f"clipped at p{args.clip_pct:g} = {vmx:.0f}"
                     if cmap is value_cmap else f"clipped at ±{vmx:.0f}")
        cb.set_label(f"carbon emission ({clip_note})", fontsize=8)

    fig.suptitle("Delhi NCR carbon emission map -- 36-month cell means, "
                 "original scale", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(args.output_dir, f"fig_maps{suffix}.png")
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig] {path}")


# ==============================================================================
# FIGURE 3: scatter
# ==============================================================================
def fig_scatter(cells, summary, args):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    eps = 1.0  # log axes with zeros -> plot value + 1

    other = cells[cells["split"] != "test"]
    test = cells[cells["split"] == "test"]
    ax.scatter(other["true_mean"] + eps, other["pred_mean"] + eps, s=4,
               c="#b9b9b9", alpha=0.35, linewidths=0, label="train / val")
    ax.scatter(test["true_mean"] + eps, test["pred_mean"] + eps, s=7,
               c="#c2402a", alpha=0.6, linewidths=0, label="test (held out)")

    lim_lo = eps * 0.8
    lim_hi = max(cells["true_mean"].max(), cells["pred_mean"].max()) * 1.5 + eps
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("true cell-mean emission + 1 (log scale)")
    ax.set_ylabel("predicted cell-mean emission + 1 (log scale)")

    note = ""
    test_orig = summary.get("by_split", {}).get("test", {}).get("original")
    if test_orig:
        note = (f"test, original space: R$^2$={test_orig['r2']:.3f}, "
                f"MAE={test_orig['mae']:.1f}")
    ax.set_title(f"Predicted vs true per cell\n{note}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "fig_scatter.png")
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig] {path}")


# ==============================================================================
# FIGURE 4: hotspot agreement
# ==============================================================================
def fig_hotspot(grids, cells, args):
    height, width = int(grids["height"]), int(grids["width"])
    agree = np.full((height, width), np.nan)
    # 0 neither, 1 hit (both top-10%), 2 false alarm (pred only), 3 miss (true only)
    for _, r in cells.iterrows():
        pt, tt = bool(r["pred_top10"]), bool(r["true_top10"])
        agree[int(r["row"]), int(r["col"])] = (
            1 if (pt and tt) else 2 if pt else 3 if tt else 0
        )

    cmap = ListedColormap(["#eeeeee", "#1a7a3a", "#e0a13c", "#b03a3a"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.imshow(agree, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    n_hit = int(((cells.pred_top10) & (cells.true_top10)).sum())
    n_true = int(cells.true_top10.sum())
    recall = n_hit / max(n_true, 1)
    ax.set_title(f"Top-10% hotspot agreement (rank-based, no threshold)\n"
                 f"recall = {n_hit}/{n_true} = {recall:.3f}", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ["#eeeeee", "#1a7a3a", "#e0a13c", "#b03a3a"]]
    ax.legend(handles, ["neither", "hit", "false alarm", "miss"],
              loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    path = os.path.join(args.output_dir, "fig_hotspot.png")
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"[fig] {path}")


# ==============================================================================
# TABLES
# ==============================================================================
def _row(name, m):
    return (f"| {name} | {m['r2']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{m['bias']:+.4f} | {m['spearman']:.4f} | {m['top10_recall']:.4f} |")


def write_tables(summary, args):
    lines = ["# Results tables (auto-generated)", ""]

    lines += ["## T1. MS-CA per split -- ORIGINAL space (report these)", "",
              "| split | R2 | MAE | RMSE | bias | Spearman | Top-10% recall |",
              "|---|---|---|---|---|---|---|"]
    for name in ("train", "val", "test"):
        m = summary.get("by_split", {}).get(name, {}).get("original")
        if m:
            lines.append(_row(name, m))
    lines.append("")

    lines += [("## T2. MS-CA per split -- log1p space (for like-for-like with "
              "baselines trained in log space)"), "",
              "| split | R2 | MAE | RMSE | bias | Spearman | Top-10% recall |",
              "|---|---|---|---|---|---|---|"]
    for name in ("train", "val", "test"):
        m = summary.get("by_split", {}).get(name, {}).get("log1p")
        if m:
            lines.append(_row(name, m))
    lines.append("")

    if args.baselines_json and os.path.exists(args.baselines_json):
        with open(args.baselines_json) as fh:
            base = json.load(fh)
        msca_log = summary.get("by_split", {}).get("test", {}).get("log1p")
        msca_orig = summary.get("by_split", {}).get("test", {}).get("original")

        lines += ["## T3. Model comparison on the held-out test split", "",
                  ("| model | R2 (log) | R2 (orig) | MAE (orig) | Spearman | "
                  "Top-10% recall |"), "|---|---|---|---|---|---|"]
        for key, label in [("B0_train_mean", "B0 constant"),
                           ("B1_fossil_only", "B1 fossil-only linear"),
                           ("B2_linear_7", "B2 linear (7 feat)"),
                           ("B3_gbm_7", "B3 GBM (7 feat, 1x1)")]:
            m_log = base.get(key)
            m_or = base.get(key + "_original_space")
            if not m_log:
                continue
            r2o = f"{m_or['r2']:.4f}" if m_or else "--"
            maeo = f"{m_or['mae']:.2f}" if m_or else "--"
            lines.append(f"| {label} | {m_log['r2']:.4f} | {r2o} | {maeo} | "
                         f"{m_log['spearman']:.4f} | "
                         f"{m_log['top10_recall']:.4f} |")
        if msca_log and msca_orig:
            lines.append(f"| **MS-CA (16x16)** | **{msca_log['r2']:.4f}** | "
                         f"**{msca_orig['r2']:.4f}** | **{msca_orig['mae']:.2f}** | "
                         f"**{msca_orig['spearman']:.4f}** | "
                         f"**{msca_orig['top10_recall']:.4f}** |")
            b3 = base.get("B3_gbm_7")
            if b3:
                lines += ["",
                          (f"Headline contribution (log space): MS-CA - B3 = "
                          f"{msca_log['r2'] - b3['r2']:+.4f} R2.")]
        lines.append("")

    lines += ["## T4. Regional totals (all cells x all months)", "",
              f"- predicted total: {summary.get('regional_total_pred'):.6g}",
              f"- true total: {summary.get('regional_total_true'):.6g}",
              f"- error: {summary.get('regional_total_error_pct', 0):+.2%}",
              f"- smearing factor applied: {summary.get('smearing')}",
              ""]

    lines += ["## Still needed before submission",
              "- Multi-seed (3-5) mean +/- cell-bootstrap 95% CI for T3.",
              ("- Same T3 under --split_mode block --buffer 16 (does the "
              "MS-CA - B3 gap survive without patch overlap?)."),
              "- Ablation rows: MS-CA --context center_only / shuffle.",
              ""]

    path = os.path.join(args.output_dir, "results_tables.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[tab] {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    grids, cells, summary = _load(args.inference_dir)
    fig_maps(grids, args, test_only=False)
    fig_maps(grids, args, test_only=True)
    fig_scatter(cells, summary, args)
    fig_hotspot(grids, cells, args)
    write_tables(summary, args)
    print("\nDone. fig_maps_test_only.png is the generalisation figure; "
          "fig_maps.png is illustration.")


if __name__ == "__main__":
    main()
