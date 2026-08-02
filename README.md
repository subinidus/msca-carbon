# MS-CA: Continuous Carbon Emission Prediction Using Satellite Proxy Data

[![Best Poster Award](https://img.shields.io/badge/Award-Best%20Poster-gold)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Multi-Stream Cross-Attention Vision Transformer (**MS-CA**) that predicts
continuous carbon emissions on a 1 km grid from seven freely available
satellite and infrastructure proxies — evaluated under a strict,
leakage-controlled protocol.

> 🏆 **Best Poster Award**, ICCJ (International Conference on Climate Justice).

---

## Abstract

Physical carbon monitoring (flux towers, in-situ sensors) is accurate but
expensive and spatially sparse, leaving most of any metropolitan grid
unmeasured — a gap most acute in the rapidly urbanising Global South.
Conventional proxy models operate on a single 1×1 cell and hit an
informational ceiling because emission is not a locally independent quantity
but a **spatial field**: plumes disperse downwind and traffic follows the road
network, so a cell's emission depends on its neighborhood.

**MS-CA** overcomes this ceiling by reading a 16×16 neighborhood around each
cell and fusing two physically distinct information streams — atmospheric
pollution and socio-infrastructure — via cross-attention. On strictly held-out
cells it raises the log-space R² from **0.757** (a single-cell gradient-boosting
baseline) to **0.960**, a **+0.203** gain equal to removing **83 % of the
variance** left unexplained by single-cell features. The result is, in effect,
a **cost-effective virtual sensor network**: dense emission maps from public
satellite products, with no ground instrumentation.

The gain is rigorously attributed to spatial context, not model capacity,
through a controlled ablation, and it survives a leakage-control experiment
that forces train/test patch overlap to zero.

---

## Repository Structure

```
msca-carbon/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── pyproject.toml             # project metadata + Ruff lint config
├── .gitignore
│
├── data/                      # NOT tracked; see data/README.md for the schema
│   └── README.md
│
└── src/
    ├── model.py               # MSCANet: dual-stream ViT + cross-attention fusion
    ├── spatial_grid_dataset.py# dataset: log1p target, train-only norm, context ablation
    ├── splits_v2.py           # leak-aware splits: cell / block(buffer) / random
    ├── fast_data.py           # GPU-resident patch batcher (bit-identical, ~15-40× faster)
    ├── metrics.py             # both-space metrics, rank stats, Duan smearing, cell bootstrap
    ├── train.py               # training loop with robust, guarded model selection
    ├── evaluation.py          # test-set eval; rebuilds the exact split from the checkpoint
    ├── inference.py           # full-grid inference → CSV / NPZ / JSON for mapping
    ├── baselines.py           # B0–B3 baselines on the identical split
    ├── diagnostics.py         # ICC / skew / lag / static-feature audit
    └── make_figures.py        # publication figures + results tables
```

---

## Key Features & Methodologies

- **Dual-stream cross-attention fusion.** Stream B (socio-infrastructure) forms
  the query and Stream A (pollution) forms the key/value, so the model asks
  *"given each infrastructure patch, which pollution patches are relevant?"* —
  an emission-attribution formulation. Per-head attention weights are retained
  for explainability.
- **Two-layer leakage control.** Location-identity leakage (ICC = 0.9915) is
  removed by a **cell-fixed split** in which all 36 months of a cell stay in one
  split; patch-overlap leakage is removed by a **buffered block split**
  (`buffer = 16`) that forces train/test patch-pixel overlap to zero.
- **Skewness defense.** With target skewness ≈ 40, the raw-scale R² is dominated
  by the top-1 % super-emitters (squared error grows as *(y·δ)²*). The pipeline
  trains on a `log1p` target and evaluates with **rank-based metrics** that are
  immune to this quadratic amplification.
- **Controlled ablation.** A `center_only` mode zeroes the neighborhood while
  holding the architecture constant, isolating the contribution of spatial
  context from that of model capacity.
- **Reproducibility by construction.** Each checkpoint embeds its split
  definition and train-only normalization statistics, so evaluation and
  inference reconstruct the identical held-out set without recomputation.
- **Engineering.** A GPU-resident patch batcher collapses per-epoch data time
  from tens of minutes to seconds while remaining bit-identical to the reference
  data path; model selection uses a smoothed criterion with a bias guardrail,
  warmup exclusion, early stopping, and an always-written fallback checkpoint.

---

## Installation

Requires Python 3.10+ and (recommended) a CUDA-capable GPU.

```bash
git clone https://github.com/<user>/msca-carbon.git
cd msca-carbon

python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

`lightgbm` and `wandb` are optional: without LightGBM the B3 baseline falls back
to scikit-learn's `HistGradientBoostingRegressor`, and training runs without
Weights & Biases via `--disable_wandb`.

Place the 36 monthly `.npz` tensors under `data/` (schema in
[`data/README.md`](data/README.md)).

---

## Usage / Quick Start

All scripts live in `src/`; run them from there so the sibling imports resolve.

```bash
cd src
DATA="../data/ms_mca_tensor_delhi_ncr_*.npz"
```

**1 — Diagnostics** (justifies the transform and split design):

```bash
python diagnostics.py --data_path $DATA --save_json ../outputs/diag.json
```

**2 — Baselines** (run first; B1 checks whether the task reduces to locating plants):

```bash
python baselines.py --data_path $DATA --split_mode cell \
    --target_transform log1p --seed 42 --save_json ../outputs/baselines_cell.json
```

**3 — Train MS-CA** (cell-fixed split, log1p target; fast GPU loader is default):

```bash
python train.py --data_path $DATA \
    --output_dir ../checkpoints/ckpt_cell \
    --split_mode cell --target_transform log1p \
    --epochs 50 --batch_size 512 --use_amp --seed 42 --disable_wandb
```

**4 — Evaluate** (split, normalization, and context are restored from the checkpoint):

```bash
python evaluation.py --checkpoint ../checkpoints/ckpt_cell/best_model.pth \
    --data_path $DATA --smearing --bootstrap 1000 --save_json ../outputs/eval_cell.json
```

**5 — Full-grid inference & figures:**

```bash
python inference.py --checkpoint ../checkpoints/ckpt_cell/best_model.pth \
    --data_path $DATA --output_dir ../outputs/inference
python make_figures.py --inference_dir ../outputs/inference \
    --baselines_json ../outputs/baselines_cell.json --output_dir ../figures
```

**Ablation & robustness check:**

```bash
# center-only: architecture held constant, spatial context removed
python train.py --data_path $DATA --output_dir ../checkpoints/ckpt_center \
    --split_mode cell --context center_only --target_transform log1p \
    --epochs 50 --batch_size 512 --use_amp --seed 42 --disable_wandb

# block split (buffer 16): zero train/test patch overlap, leakage control
python train.py --data_path $DATA --output_dir ../checkpoints/ckpt_block \
    --split_mode block --buffer 16 --target_transform log1p \
    --epochs 50 --batch_size 512 --use_amp --seed 42 --disable_wandb
```

---

## Main Results

All figures are on **strictly held-out test cells** under the cell-fixed split,
in log1p space.

### Ablation — spatial context, not model capacity

| Condition | Information | log R² | Top-10 % recall |
|---|---|:--:|:--:|
| B3 — GBM, 1×1 | center cell only | 0.7567 | 0.820 |
| MS-CA, center-only | center cell only | 0.7450 | 0.824 |
| **MS-CA, full 16×16** | **center + neighborhood** | **0.9595** | **0.952** |

With the neighborhood zeroed, the transformer converges to the tabular ceiling
on **both** metrics (0.745 ≈ 0.757; 0.824 ≈ 0.820). The **+0.203** jump appears
only when the 16×16 context is restored — the advantage is spatial context, not
capacity.

### Rank-based validation (heavy-tail-appropriate)

| Metric | Held-out value | Interpretation |
|---|:--:|---|
| Spearman ρ | **0.978** | Spatial ordering of the emission field is recovered |
| Top-10 % recall | **0.952** | 95 % of true super-emitter hotspots identified (threshold-free) |
| log-space R² | **0.960** | Primary accuracy metric (natural scale for a multiplicative process) |
| original-space R² | 0.067 | Dominated by the top-1 % tail — see skewness defense |
| MedAE (orig.) | 3.2 | Typical-case error; inaccuracy is confined to the extreme tail |

> **On the original-space R².** It is a mathematical property of the skewed
> target, not a model failure: with skewness 40 the raw variance is owned by the
> top 1 % of cells and squared error grows as *(y·δ)²*, so a uniform ±30 %
> multiplicative error collapses raw R² while leaving rank metrics — the correct
> indicators here — unaffected.

---

## Contact / Citation

**Author:** Subin Seo · with Jewon Lee, Rhea Tess Payyapilly, George K. Mathew
**Affiliations:** Kyungpook National University · Christ University (Centre for Digital Innovation)

If you use this work, please cite:

```bibtex
@misc{seo_msca_carbon_2026,
  author = {Seo, Subin and Lee, Jewon and Payyapilly, Rhea Tess and Mathew, George K.},
  title  = {MS-CA: Continuous Carbon Emission Prediction Using Satellite Proxy Data},
  year   = {2026},
  note   = {Best Poster Award,ICCJ (International Conference on Climate Justice)},
}
```

## License

Released under the [MIT License](LICENSE).
