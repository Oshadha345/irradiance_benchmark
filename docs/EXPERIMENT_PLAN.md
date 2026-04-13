# Experiment Plan

Ordered execution sheet for the double-blind MERCon 2026 baseline benchmark.

## Benchmark Rules

- Architecture:
  - visual encoder -> `1024`-D pooled descriptor
  - temporal encoder -> `LSTM(40 x 7 -> 128)`
  - fusion -> concatenation + 2-layer MLP
  - target -> single `10`-minute clear-sky index
- `train.py` enforces the baseline contract at runtime:
  - `sequence_length = 40`
  - `horizons = [10]`
  - batch size is auto-tuned on the active GPU from safe probe candidates, then printed at startup
- Training commands should use `bash scripts/train_exclusive.sh ...` so GPU `1` is serialized and idle before each run.
- `train.py` probes the active GPU, chooses the fastest safe batch size, and prints the resulting batch count.
- `scripts/postprocess.py` has a separate `--batch-size` flag used only for the efficiency/FPS benchmark.
  Its default is `4`, so it is omitted from the roadmap commands below unless you want to override it.
- Chronological dataset splits:
  - `Folsom`: `2014-2015` train/val, `2016` test
  - `NREL`: `2018-2019` train/val, `2020` test
- `scripts/postprocess.py` runs test evaluation, efficiency, and ERF generation in one command.

## Once Per Shell

```bash
cd /storage2/CV_Irradiance/Mercon_Mamba/irradiance_benchmark

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHON_BIN=/userhomes/shehan15/miniconda3/envs/solarmamba_train_1/bin/python

export FOLSOM_ERF_IMAGE="/storage2/CV_Irradiance/datasets/1_Folsom/2014/04/04/20140404_010659.jpg"
export NREL_ERF_IMAGE="/storage2/CV_Irradiance/datasets/4_NREL/2019_09_07/images/UTC-7_2019_09_07-09_50_22_530200.jpg"
```

## Full Orchestration

For unattended execution of the whole roadmap on GPU `1`:

```bash
PYTHON_BIN="${PYTHON_BIN}" \
python scripts/orchestrate_roadmap.py \
  --roadmap docs/EXPERIMENT_PLAN.md \
  --gpu-index 1 \
  --budget-hours 72
```

The orchestrator computes a max epoch-time budget from the remaining wall-clock budget and remaining experiments, searches batch-size and chronological-window candidates, trains one run at a time, postprocesses it, writes plots, and resumes from `results/orchestrator/roadmap_state.json`.

## Stage 0: Setup

- [x] `PYTHONNOUSERSITE=1 PYTHON_BIN="${PYTHON_BIN}" SKIP_PYTHON_DEPS=1 bash setup.sh`
- [x] `ls -lh weights`

Notes:
- Keep `PYTHONNOUSERSITE=1` enabled.
- The first NREL access may build or reuse `_image_index_cache_*.pkl` under `open_solar/srrl_2017_to_2025/`.

## Stage 1: Folsom Baselines

### Folsom ConvNeXt-Tiny

- [x] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_convnext_tiny.yaml --comment "folsom_convnext_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_tiny_tiny_folsom_convnext_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_convnext_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom ConvNeXt-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_convnext_small.yaml --comment "folsom_convnext_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_small_small_folsom_convnext_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_convnext_small.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom ConvNeXt-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_convnext_base.yaml --comment "folsom_convnext_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_base_base_folsom_convnext_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_convnext_base.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom ConvNeXt-Large

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_convnext_large.yaml --comment "folsom_convnext_large_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_large_large_folsom_convnext_large_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_convnext_large.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom Swin-Tiny

- [x] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_swin_tiny.yaml --comment "folsom_swin_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/swin_tiny_patch4_window7_224_tiny_folsom_swin_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_swin_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom Swin-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_swin_small.yaml --comment "folsom_swin_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/swin_small_patch4_window7_224_small_folsom_swin_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_swin_small.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom Swin-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_swin_base.yaml --comment "folsom_swin_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/swin_base_patch4_window7_224_base_folsom_swin_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_swin_base.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

## Stage 2: Folsom Mamba Sweep

### Folsom VMamba-Tiny

- [x] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_vmamba_tiny.yaml --comment "folsom_vmamba_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/vmamba_tiny_folsom_vmamba_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_vmamba_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom VMamba-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_vmamba_small.yaml --comment "folsom_vmamba_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/vmamba_small_folsom_vmamba_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_vmamba_small.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom VMamba-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_vmamba_base.yaml --comment "folsom_vmamba_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/vmamba_base_folsom_vmamba_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_vmamba_base.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom Spatial-Mamba-Tiny

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_spatial_mamba_tiny.yaml --comment "folsom_spatial_mamba_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/spatial_mamba_tiny_folsom_spatial_mamba_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_spatial_mamba_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom Spatial-Mamba-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_spatial_mamba_small.yaml --comment "folsom_spatial_mamba_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/spatial_mamba_small_folsom_spatial_mamba_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_spatial_mamba_small.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom Spatial-Mamba-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_spatial_mamba_base.yaml --comment "folsom_spatial_mamba_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/spatial_mamba_base_folsom_spatial_mamba_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_spatial_mamba_base.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom MambaVision-Tiny

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_mambavision_tiny.yaml --comment "folsom_mambavision_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_tiny_folsom_mambavision_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_mambavision_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom MambaVision-Tiny2

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_mambavision_tiny2.yaml --comment "folsom_mambavision_tiny2_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_tiny2_folsom_mambavision_tiny2_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_mambavision_tiny2.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom MambaVision-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_mambavision_small.yaml --comment "folsom_mambavision_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_small_folsom_mambavision_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_mambavision_small.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom MambaVision-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_mambavision_base.yaml --comment "folsom_mambavision_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_base_folsom_mambavision_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_mambavision_base.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom MambaVision-Large

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_mambavision_large.yaml --comment "folsom_mambavision_large_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_large_folsom_mambavision_large_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_mambavision_large.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

### Folsom MambaVision-Large2

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/folsom_mambavision_large2.yaml --comment "folsom_mambavision_large2_baseline"`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_large2_folsom_mambavision_large2_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/folsom_mambavision_large2.yaml --run-dir "${RUN_DIR}" --input-image "${FOLSOM_ERF_IMAGE}"`

## Stage 3: NREL Baselines

### NREL ConvNeXt-Tiny

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_convnext_tiny.yaml --comment "nrel_convnext_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_tiny_tiny_nrel_convnext_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_convnext_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL ConvNeXt-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_convnext_small.yaml --comment "nrel_convnext_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_small_small_nrel_convnext_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_convnext_small.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL ConvNeXt-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_convnext_base.yaml --comment "nrel_convnext_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_base_base_nrel_convnext_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_convnext_base.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL ConvNeXt-Large

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_convnext_large.yaml --comment "nrel_convnext_large_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_large_large_nrel_convnext_large_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_convnext_large.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL Swin-Tiny

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_swin_tiny.yaml --comment "nrel_swin_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/swin_tiny_patch4_window7_224_tiny_nrel_swin_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_swin_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL Swin-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_swin_small.yaml --comment "nrel_swin_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/swin_small_patch4_window7_224_small_nrel_swin_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_swin_small.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL Swin-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_swin_base.yaml --comment "nrel_swin_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/swin_base_patch4_window7_224_base_nrel_swin_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_swin_base.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

## Stage 4: NREL Mamba Sweep

### NREL VMamba-Tiny

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_vmamba_tiny.yaml --comment "nrel_vmamba_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/vmamba_tiny_nrel_vmamba_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_vmamba_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL VMamba-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_vmamba_small.yaml --comment "nrel_vmamba_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/vmamba_small_nrel_vmamba_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_vmamba_small.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL VMamba-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_vmamba_base.yaml --comment "nrel_vmamba_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/vmamba_base_nrel_vmamba_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_vmamba_base.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL Spatial-Mamba-Tiny

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_spatial_mamba_tiny.yaml --comment "nrel_spatial_mamba_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/spatial_mamba_tiny_nrel_spatial_mamba_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_spatial_mamba_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL Spatial-Mamba-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_spatial_mamba_small.yaml --comment "nrel_spatial_mamba_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/spatial_mamba_small_nrel_spatial_mamba_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_spatial_mamba_small.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL Spatial-Mamba-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_spatial_mamba_base.yaml --comment "nrel_spatial_mamba_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/spatial_mamba_base_nrel_spatial_mamba_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_spatial_mamba_base.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL MambaVision-Tiny

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_mambavision_tiny.yaml --comment "nrel_mambavision_tiny_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_tiny_nrel_mambavision_tiny_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_mambavision_tiny.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL MambaVision-Tiny2

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_mambavision_tiny2.yaml --comment "nrel_mambavision_tiny2_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_tiny2_nrel_mambavision_tiny2_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_mambavision_tiny2.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL MambaVision-Small

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_mambavision_small.yaml --comment "nrel_mambavision_small_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_small_nrel_mambavision_small_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_mambavision_small.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL MambaVision-Base

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_mambavision_base.yaml --comment "nrel_mambavision_base_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_base_nrel_mambavision_base_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_mambavision_base.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL MambaVision-Large

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_mambavision_large.yaml --comment "nrel_mambavision_large_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_large_nrel_mambavision_large_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_mambavision_large.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

### NREL MambaVision-Large2

- [ ] `GPU_INDEX=1 PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_exclusive.sh --config configs/nrel_mambavision_large2.yaml --comment "nrel_mambavision_large2_baseline"`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_large2_nrel_mambavision_large2_baseline_* | head -1)`
- [ ] `"${PYTHON_BIN}" scripts/postprocess.py --config configs/nrel_mambavision_large2.yaml --run-dir "${RUN_DIR}" --input-image "${NREL_ERF_IMAGE}"`

## Optional Validation-Only Check

If you need validation metrics without rerunning efficiency or ERF:

```bash
"${PYTHON_BIN}" scripts/postprocess.py \
  --config <CONFIG_PATH> \
  --run-dir "${RUN_DIR}" \
  --split val \
  --skip-efficiency \
  --skip-erf
```

## Centralized Result Logger

Copy values from each run directory into this table.

| Date | Dataset | Backbone | Scale | Config | Run Dir | Train/Val Years | Test Year | RMSE | nRMSE | MAE | MBE | R2 | FS | Params (M) | FLOPs (G) | FPS | ERF Path | Notes |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| YYYY-MM-DD | Folsom or NREL | ConvNeXt / Swin / VMamba / Spatial-Mamba / MambaVision | Tiny / Small / Base / Large / Tiny2 / Large2 | `configs/...yaml` | `results/...` | `2014-2015` or `2018-2019` | `2016` or `2020` |  |  |  |  |  |  |  |  |  | `results/.../erf_overlay.png` |  |
