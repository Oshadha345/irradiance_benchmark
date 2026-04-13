# Experiment Plan

Central execution log for the MERCon 2026 benchmark sweep. Work through this file from top to bottom and update the logger at the end after each finished run.

## Ground Rules

- Run everything from `/storage2/CV_Irradiance/Mercon_Mamba/irradiance_benchmark`.
- Run each experiment block in the same shell so `RUN_DIR` stays defined.
- `evaluate.py` writes `test_metrics.json` next to `best.ckpt`.
- `evaluate_efficiency.py` is routed into the same run directory with `--output`.
- `ConvNeXt` uses `timm` pretrained weights and does not require a mirrored file in `weights/`.
- ERF needs one representative RGB sky image per dataset. It is only used for visualization, not training.

## Stage 0: Environment Bring-Up

- [x] `cd /storage2/CV_Irradiance/Mercon_Mamba/irradiance_benchmark`
- [x] `bash setup.sh`
- [x] `ls -lh weights`
- [ ] `export FOLSOM_ERF_IMAGE="/storage2/CV_Irradiance/datasets/1_Folsom/2014/04/04/20140404_010659.jpg"`
- [ ] `export NREL_ERF_IMAGE="/storage2/CV_Irradiance/datasets/4_NREL/2019_09_07/images/UTC-7_2019_09_07-09_50_22_530200.jpg"`

If this machine is running a CUDA-mismatched environment such as `torch==2.5.1+cu121` with `/usr/bin/nvcc` from CUDA 11.5, use this instead so the setup does not modify the env and does not fail at Spatial-Mamba kernel compilation:

- [ ] `SKIP_PYTHON_DEPS=1 SKIP_SPATIAL_MAMBA_BUILD=1 bash setup.sh`

Spatial-Mamba experiments should stay unchecked until a matching CUDA toolkit is available through `CUDA_HOME`.

## Stage 1: Folsom Baselines

### Folsom ConvNeXt-Tiny

- [ ] `python train.py --config configs/folsom_convnext_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_tiny_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_convnext_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_convnext_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_convnext_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom ConvNeXt-Small

- [ ] `python train.py --config configs/folsom_convnext_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_small_small_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_convnext_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_convnext_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_convnext_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom ConvNeXt-Base

- [ ] `python train.py --config configs/folsom_convnext_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_base_base_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_convnext_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_convnext_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_convnext_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom ConvNeXt-Large

- [ ] `python train.py --config configs/folsom_convnext_large.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/convnext_large_large_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_convnext_large.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_convnext_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_convnext_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom Swin-T

- [ ] `python train.py --config configs/folsom_swin_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/swin_tiny_patch4_window7_224_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_swin_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_swin_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_swin_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom Swin-S

- [ ] `python train.py --config configs/folsom_swin_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/swin_small_patch4_window7_224_small_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_swin_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_swin_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_swin_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom Swin-B

- [ ] `python train.py --config configs/folsom_swin_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/swin_base_patch4_window7_224_base_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_swin_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_swin_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_swin_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

## Stage 2: Folsom Mamba Sweep

### Folsom VMamba-T

- [ ] `python train.py --config configs/folsom_vmamba_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/vmamba_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom VMamba-S

- [ ] `python train.py --config configs/folsom_vmamba_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/vmamba_small_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_vmamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_vmamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_vmamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom VMamba-B

- [ ] `python train.py --config configs/folsom_vmamba_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/vmamba_base_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_vmamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_vmamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_vmamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom Spatial-Mamba-T

- [ ] `python train.py --config configs/folsom_spatial_mamba_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/spatial_mamba_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_spatial_mamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_spatial_mamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_spatial_mamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom Spatial-Mamba-S

- [ ] `python train.py --config configs/folsom_spatial_mamba_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/spatial_mamba_small_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_spatial_mamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_spatial_mamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_spatial_mamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom Spatial-Mamba-B

- [ ] `python train.py --config configs/folsom_spatial_mamba_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/spatial_mamba_base_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_spatial_mamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_spatial_mamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_spatial_mamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom MambaVision-T

- [ ] `python train.py --config configs/folsom_mambavision_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_mambavision_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_mambavision_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_mambavision_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom MambaVision-T2

- [ ] `python train.py --config configs/folsom_mambavision_tiny2.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_tiny2_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_mambavision_tiny2.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_mambavision_tiny2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_mambavision_tiny2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom MambaVision-S

- [ ] `python train.py --config configs/folsom_mambavision_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_small_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_mambavision_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_mambavision_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_mambavision_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom MambaVision-B

- [ ] `python train.py --config configs/folsom_mambavision_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_base_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_mambavision_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_mambavision_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_mambavision_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom MambaVision-L

- [ ] `python train.py --config configs/folsom_mambavision_large.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_large_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_mambavision_large.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_mambavision_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_mambavision_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### Folsom MambaVision-L2

- [ ] `python train.py --config configs/folsom_mambavision_large2.yaml`
- [ ] `RUN_DIR=$(ls -td results/folsom/mambavision_large2_* | head -1)`
- [ ] `python evaluate.py --config configs/folsom_mambavision_large2.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/folsom_mambavision_large2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/folsom_mambavision_large2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${FOLSOM_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

## Stage 3: NREL Baselines

### NREL ConvNeXt-Tiny

- [ ] `python train.py --config configs/nrel_convnext_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_tiny_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_convnext_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_convnext_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_convnext_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL ConvNeXt-Small

- [ ] `python train.py --config configs/nrel_convnext_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_small_small_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_convnext_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_convnext_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_convnext_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL ConvNeXt-Base

- [ ] `python train.py --config configs/nrel_convnext_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_base_base_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_convnext_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_convnext_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_convnext_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL ConvNeXt-Large

- [ ] `python train.py --config configs/nrel_convnext_large.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/convnext_large_large_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_convnext_large.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_convnext_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_convnext_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL Swin-T

- [ ] `python train.py --config configs/nrel_swin_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/swin_tiny_patch4_window7_224_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_swin_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_swin_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_swin_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL Swin-S

- [ ] `python train.py --config configs/nrel_swin_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/swin_small_patch4_window7_224_small_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_swin_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_swin_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_swin_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL Swin-B

- [ ] `python train.py --config configs/nrel_swin_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/swin_base_patch4_window7_224_base_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_swin_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_swin_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_swin_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

## Stage 4: NREL Mamba Sweep

### NREL VMamba-T

- [ ] `python train.py --config configs/nrel_vmamba_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/vmamba_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_vmamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL VMamba-S

- [ ] `python train.py --config configs/nrel_vmamba_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/vmamba_small_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_vmamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_vmamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_vmamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL VMamba-B

- [ ] `python train.py --config configs/nrel_vmamba_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/vmamba_base_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_vmamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_vmamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_vmamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL Spatial-Mamba-T

- [ ] `python train.py --config configs/nrel_spatial_mamba_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/spatial_mamba_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_spatial_mamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_spatial_mamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_spatial_mamba_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL Spatial-Mamba-S

- [ ] `python train.py --config configs/nrel_spatial_mamba_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/spatial_mamba_small_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_spatial_mamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_spatial_mamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_spatial_mamba_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL Spatial-Mamba-B

- [ ] `python train.py --config configs/nrel_spatial_mamba_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/spatial_mamba_base_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_spatial_mamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_spatial_mamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_spatial_mamba_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL MambaVision-T

- [ ] `python train.py --config configs/nrel_mambavision_tiny.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_tiny_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_mambavision_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_mambavision_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_mambavision_tiny.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL MambaVision-T2

- [ ] `python train.py --config configs/nrel_mambavision_tiny2.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_tiny2_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_mambavision_tiny2.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_mambavision_tiny2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_mambavision_tiny2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL MambaVision-S

- [ ] `python train.py --config configs/nrel_mambavision_small.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_small_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_mambavision_small.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_mambavision_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_mambavision_small.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL MambaVision-B

- [ ] `python train.py --config configs/nrel_mambavision_base.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_base_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_mambavision_base.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_mambavision_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_mambavision_base.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL MambaVision-L

- [ ] `python train.py --config configs/nrel_mambavision_large.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_large_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_mambavision_large.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_mambavision_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_mambavision_large.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

### NREL MambaVision-L2

- [ ] `python train.py --config configs/nrel_mambavision_large2.yaml`
- [ ] `RUN_DIR=$(ls -td results/nrel/mambavision_large2_* | head -1)`
- [ ] `python evaluate.py --config configs/nrel_mambavision_large2.yaml --checkpoint "${RUN_DIR}/best.ckpt"`
- [ ] `python evaluate_efficiency.py --config configs/nrel_mambavision_large2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --batch-size 4 --output "${RUN_DIR}/efficiency_report.json"`
- [ ] `python utils/visualize_erf.py --config configs/nrel_mambavision_large2.yaml --checkpoint "${RUN_DIR}/best.ckpt" --input-image "${NREL_ERF_IMAGE}" --output "${RUN_DIR}/erf_overlay.png"`

## Centralized Result Logger

Copy final numbers from `test_metrics.json` and `efficiency_report.json` into this table after each finished run.

| Dataset | Backbone | Scale | Config | Run Dir | RMSE | nRMSE | MAE | MBE | R2 | FS | Params (M) | FLOPs (G) | FPS | ERF | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Folsom | ConvNeXt | Tiny | `configs/folsom_convnext_tiny.yaml` | `results/folsom/convnext_tiny_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | ConvNeXt | Small | `configs/folsom_convnext_small.yaml` | `results/folsom/convnext_small_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | ConvNeXt | Base | `configs/folsom_convnext_base.yaml` | `results/folsom/convnext_base_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | ConvNeXt | Large | `configs/folsom_convnext_large.yaml` | `results/folsom/convnext_large_large_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | Swin | Tiny | `configs/folsom_swin_tiny.yaml` | `results/folsom/swin_tiny_patch4_window7_224_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | Swin | Small | `configs/folsom_swin_small.yaml` | `results/folsom/swin_small_patch4_window7_224_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | Swin | Base | `configs/folsom_swin_base.yaml` | `results/folsom/swin_base_patch4_window7_224_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | VMamba | Tiny | `configs/folsom_vmamba_tiny.yaml` | `results/folsom/vmamba_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | VMamba | Small | `configs/folsom_vmamba_small.yaml` | `results/folsom/vmamba_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | VMamba | Base | `configs/folsom_vmamba_base.yaml` | `results/folsom/vmamba_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | Spatial-Mamba | Tiny | `configs/folsom_spatial_mamba_tiny.yaml` | `results/folsom/spatial_mamba_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | Spatial-Mamba | Small | `configs/folsom_spatial_mamba_small.yaml` | `results/folsom/spatial_mamba_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | Spatial-Mamba | Base | `configs/folsom_spatial_mamba_base.yaml` | `results/folsom/spatial_mamba_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | MambaVision | Tiny | `configs/folsom_mambavision_tiny.yaml` | `results/folsom/mambavision_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | MambaVision | Tiny2 | `configs/folsom_mambavision_tiny2.yaml` | `results/folsom/mambavision_tiny2_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | MambaVision | Small | `configs/folsom_mambavision_small.yaml` | `results/folsom/mambavision_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | MambaVision | Base | `configs/folsom_mambavision_base.yaml` | `results/folsom/mambavision_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | MambaVision | Large | `configs/folsom_mambavision_large.yaml` | `results/folsom/mambavision_large_*` |  |  |  |  |  |  |  |  |  |  |  |
| Folsom | MambaVision | Large2 | `configs/folsom_mambavision_large2.yaml` | `results/folsom/mambavision_large2_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | ConvNeXt | Tiny | `configs/nrel_convnext_tiny.yaml` | `results/nrel/convnext_tiny_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | ConvNeXt | Small | `configs/nrel_convnext_small.yaml` | `results/nrel/convnext_small_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | ConvNeXt | Base | `configs/nrel_convnext_base.yaml` | `results/nrel/convnext_base_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | ConvNeXt | Large | `configs/nrel_convnext_large.yaml` | `results/nrel/convnext_large_large_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | Swin | Tiny | `configs/nrel_swin_tiny.yaml` | `results/nrel/swin_tiny_patch4_window7_224_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | Swin | Small | `configs/nrel_swin_small.yaml` | `results/nrel/swin_small_patch4_window7_224_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | Swin | Base | `configs/nrel_swin_base.yaml` | `results/nrel/swin_base_patch4_window7_224_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | VMamba | Tiny | `configs/nrel_vmamba_tiny.yaml` | `results/nrel/vmamba_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | VMamba | Small | `configs/nrel_vmamba_small.yaml` | `results/nrel/vmamba_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | VMamba | Base | `configs/nrel_vmamba_base.yaml` | `results/nrel/vmamba_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | Spatial-Mamba | Tiny | `configs/nrel_spatial_mamba_tiny.yaml` | `results/nrel/spatial_mamba_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | Spatial-Mamba | Small | `configs/nrel_spatial_mamba_small.yaml` | `results/nrel/spatial_mamba_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | Spatial-Mamba | Base | `configs/nrel_spatial_mamba_base.yaml` | `results/nrel/spatial_mamba_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | MambaVision | Tiny | `configs/nrel_mambavision_tiny.yaml` | `results/nrel/mambavision_tiny_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | MambaVision | Tiny2 | `configs/nrel_mambavision_tiny2.yaml` | `results/nrel/mambavision_tiny2_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | MambaVision | Small | `configs/nrel_mambavision_small.yaml` | `results/nrel/mambavision_small_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | MambaVision | Base | `configs/nrel_mambavision_base.yaml` | `results/nrel/mambavision_base_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | MambaVision | Large | `configs/nrel_mambavision_large.yaml` | `results/nrel/mambavision_large_*` |  |  |  |  |  |  |  |  |  |  |  |
| NREL | MambaVision | Large2 | `configs/nrel_mambavision_large2.yaml` | `results/nrel/mambavision_large2_*` |  |  |  |  |  |  |  |  |  |  |  |
