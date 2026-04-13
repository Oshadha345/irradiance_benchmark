# SolarMamba Benchmarking Suite

[![License: TBD](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

Unified PyTorch benchmarking code for **MERCon 2026** experiments on multi-modal solar irradiance forecasting. The repository standardizes a 4-stage visual pyramid interface over **VMamba**, **MambaVision**, **Spatial-Mamba**, **ConvNeXt**, and **Swin**, then plugs each backbone into the existing **SolarMamba** temporal pathway and **Ladder Fusion** design.

> Paper / abstract placeholder: `TBD`

## What This Repo Does

- Reuses the original **Folsom** and **NREL** dataloaders from SolarMamba.
- Preserves the temporal branch with **PyramidTCN**.
- Preserves the fusion interface with **LadderFusion** and keeps **matrix fusion** available as an ablation.
- Unifies all visual encoders to a standardized pyramid: `[(64, H/4), (128, H/8), (256, H/16), (512, H/32)]`.
- Produces conference-friendly artifacts per run:
  - `config.json`
  - `best.ckpt`
  - `history.json`
  - `best_val_metrics.json`
  - `test_metrics.json`

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
│   ├── heads.py
│   ├── model.py
│   ├── temporal.py
│   └── wrappers.py
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

## Setup Roadmap

1. Activate a Python environment with **PyTorch 2.2+** and CUDA support if you plan to use the Mamba backbones.
2. Run the bootstrap script from the repository root:

```bash
bash setup.sh
```

`setup.sh` will:

- install benchmark dependencies,
- download the official backbone definition files into `models/backbones/`,
- build the required **Spatial-Mamba** custom kernels in-place,
- download local ImageNet-1K pretrained checkpoints for **VMamba**, **MambaVision**, **Spatial-Mamba**, and **Swin-T/S/B** into `weights/`,
- leave **ConvNeXt** on the `timm` code path, so its pretrained weights are resolved through the normal `timm` cache on first use.

If your environment already has the required Python packages and your system `nvcc` does not match `torch.version.cuda`, rerun bootstrap without touching the environment or forcing a failing Spatial-Mamba build:

```bash
SKIP_PYTHON_DEPS=1 SKIP_SPATIAL_MAMBA_BUILD=1 bash setup.sh
```

Example: a `torch==2.5.1+cu121` environment paired with `/usr/bin/nvcc` from CUDA 11.5 cannot compile the Spatial-Mamba extensions. In that case, the rest of the benchmark remains usable, but Spatial-Mamba runs must wait until a CUDA 12.1 toolkit is available and exported through `CUDA_HOME`.

## Experiment Roadmap

The full ordered experiment checklist lives in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md). It includes:

- the exact train/eval/efficiency/ERF commands for every config in this repo,
- a staged roadmap starting from Folsom baselines and moving to the full Folsom/NREL sweep,
- a centralized markdown logger you can keep updating after each finished run.

Minimal example:

```bash
python train.py --config configs/folsom_vmamba_tiny.yaml
RUN_DIR=$(ls -td results/folsom/vmamba_tiny_* | head -1)
python evaluate.py --config configs/folsom_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"
python evaluate_efficiency.py --config configs/folsom_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"
```

ERF example with dataset-specific images:

```bash
export FOLSOM_ERF_IMAGE="/storage2/CV_Irradiance/datasets/1_Folsom/2014/04/04/20140404_010659.jpg"
export NREL_ERF_IMAGE="/storage2/CV_Irradiance/datasets/4_NREL/2019_09_07/images/UTC-7_2019_09_07-09_50_22_530200.jpg"
```

Each run creates:

```text
results/[dataset]/[model_name]_[scale]_[timestamp]/
```

## Metrics

Implemented in [utils/metrics.py](utils/metrics.py):

- `RMSE`
- `nRMSE`
- `MAE`
- `MBE`
- `R^2`
- `FS` relative to a persistence baseline

Both per-horizon and flattened `overall` summaries are emitted in JSON form.

## Notes on the Visual Backbone Interface

- **Swin** and **ConvNeXt** are instantiated through `timm` with `features_only=True`.
- **VMamba**, **MambaVision**, and **Spatial-Mamba** are wrapped from local vendored official code under `models/backbones/`.
- Every encoder is projected to the same channel pyramid: `[64, 128, 256, 512]`.
- This keeps the downstream **Ladder Fusion** branch fixed while swapping only the visual backbone.

## Results Layout

Example:

```text
results/folsom/vmamba_tiny_20260413_120000/
├── best.ckpt
├── best_val_metrics.json
├── config.json
├── config.yaml
├── history.json
├── last.ckpt
└── test_metrics.json
```

## Centralized Result Logger

The maintained research log now lives in [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md#centralized-result-logger). Update that table after each experiment so the run history stays in one place.

## BibTeX

```bibtex
@misc{mercon2026_solarmamba_benchmark,
  title   = {SolarMamba Benchmarking Suite},
  author  = {TBD},
  year    = {2026},
  note    = {Code release for the MERCon 2026 submission}
}
```

## Acknowledgment

This benchmark reuses the SolarMamba temporal/data pipeline and plugs in official visual backbones from:

- `MzeroMiko/VMamba`
- `NVlabs/MambaVision`
- `EdwardChasel/Spatial-Mamba`
