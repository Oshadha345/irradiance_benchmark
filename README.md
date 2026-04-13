# MERCon 2026 Irradiance Baseline Benchmark

[![License: TBD](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

Unified PyTorch benchmark code for a double-blind **MERCon 2026** submission on multi-modal solar irradiance forecasting.

The current repository state implements a strict baseline:
- visual backbone -> 4-stage features -> GAP/project -> `1024`-D visual vector
- temporal history -> `40 x 7` sequence -> `1`-layer `LSTM(128)`
- fusion -> concatenation -> `LayerNorm -> Linear -> GELU -> Dropout(0.3) -> Linear`
- target -> single `10`-minute clear-sky index forecast

> Paper / abstract placeholder: `TBD`

## Benchmark Contract

- All backbones expose the same flat visual descriptor shape: `(B, 1024)`.
- The temporal branch is fixed to a `40`-step, `7`-channel input window.
- `train.py` automatically chooses batch size by scale:
  - `tiny -> 64`
  - `small -> 32`
  - `base -> 16`
  - `large / large2 -> 8`
- Training commands do not need a batch-size flag.
  `train.py` overrides the training dataloader batch size internally.
- The `--batch-size` flag on `scripts/postprocess.py` is only for the efficiency/FPS benchmark.
  Its default is `4`.
- Evaluation is single-horizon only:
  - `RMSE`
  - `nRMSE`
  - `MAE`
  - `MBE`
  - `R2`
  - `FS`

## Chronological Splits

- `Folsom`
  - training/validation years: `2014-2015`
  - test year: `2016`
- `NREL`
  - training/validation years: `2018-2019`
  - test year: `2020`

The NREL split is `2018-2020` rather than `2017-2019` because the enforced `40`-step history window and image-availability filter make `2018` the first valid benchmark year block in the current data path. Train/test boundaries remain chronological, and normalization statistics are recomputed from train/validation years only.

## Directory Tree

```text
irradiance_benchmark/
├── configs/
│   ├── folsom_*.yaml
│   └── nrel_*.yaml
├── datasets/
│   ├── __init__.py
│   ├── folsom_loader.py
│   └── nrel_loader.py
├── docs/
│   └── EXPERIMENT_PLAN.md
├── models/
│   ├── backbones/
│   │   ├── mambavision/
│   │   ├── spatial_mamba/
│   │   └── vmamba/
│   ├── fusion.py
│   ├── model.py
│   └── wrappers.py
├── scripts/
│   ├── postprocess.py
│   └── train.py
├── utils/
│   ├── metrics.py
│   ├── pipeline.py
│   ├── runtime.py
│   └── visualize_erf.py
├── evaluate.py
├── evaluate_efficiency.py
├── requirements.txt
├── setup.sh
└── train.py
```

## Setup

Run from the repo root:

```bash
cd /storage2/CV_Irradiance/Mercon_Mamba/irradiance_benchmark

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export PYTHON_BIN=/userhomes/shehan15/miniconda3/envs/solarmamba_train_1/bin/python

PYTHONNOUSERSITE=1 \
PYTHON_BIN="${PYTHON_BIN}" \
SKIP_PYTHON_DEPS=1 \
bash setup.sh
```

`setup.sh` will:
- refresh vendored backbone source files under `models/backbones/`
- rebuild Spatial-Mamba kernels when needed
- download local checkpoints for VMamba, MambaVision, Spatial-Mamba, and Swin
- leave ConvNeXt on the normal `timm` pretrained path

## Train And Postprocess

The full ordered roadmap is in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).

Set the ERF images once per shell:

```bash
export FOLSOM_ERF_IMAGE="/storage2/CV_Irradiance/datasets/1_Folsom/2014/04/04/20140404_010659.jpg"
export NREL_ERF_IMAGE="/storage2/CV_Irradiance/datasets/4_NREL/2019_09_07/images/UTC-7_2019_09_07-09_50_22_530200.jpg"
```

Example Folsom run:

```bash
"${PYTHON_BIN}" scripts/train.py \
  --config configs/folsom_convnext_tiny.yaml \
  --comment "folsom_convnext_tiny_baseline"

RUN_DIR=$(ls -td results/folsom/convnext_tiny_tiny_folsom_convnext_tiny_baseline_* | head -1)

"${PYTHON_BIN}" scripts/postprocess.py \
  --config configs/folsom_convnext_tiny.yaml \
  --run-dir "${RUN_DIR}" \
  --input-image "${FOLSOM_ERF_IMAGE}"
```

`scripts/postprocess.py` runs:
- checkpoint evaluation on the test split
- efficiency measurement
- ERF generation

## Outputs

Each run creates:

```text
results/[dataset]/[model_slug]_[comment]_[timestamp]/
```

Typical contents:

```text
results/folsom/convnext_tiny_tiny_folsom_convnext_tiny_baseline_20260413_120000/
├── best.ckpt
├── best_val_metrics.json
├── config.json
├── config.yaml
├── efficiency_report.json
├── erf_overlay.png
├── history.json
├── last.ckpt
└── test_metrics.json
```

## Centralized Log

The experiment checklist and centralized result table live in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md#centralized-result-logger).

## BibTeX

```bibtex
@misc{mercon2026_irradiance_baseline_benchmark,
  title   = {MERCon 2026 Irradiance Baseline Benchmark},
  author  = {TBD},
  year    = {2026},
  note    = {Code release for the MERCon 2026 submission}
}
```

## Backbone Sources

Official backbone implementations used in this benchmark:

- `MzeroMiko/VMamba`
- `NVlabs/MambaVision`
- `EdwardChasel/Spatial-Mamba`
- `timm` for `ConvNeXt` and `Swin`
