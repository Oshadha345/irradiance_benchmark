`cd /storage2/CV_Irradiance/Mercon_Mamba/irradiance_benchmark

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHON_BIN=/userhomes/shehan15/miniconda3/envs/solarmamba_train_1/bin/python
export FOLSOM_ERF_IMAGE="/storage2/CV_Irradiance/datasets/1_Folsom/2014/04/04/20140404_010659.jpg"
export NREL_ERF_IMAGE="/storage2/CV_Irradiance/datasets/4_NREL/2019_09_07/images/UTC-7_2019_09_07-09_50_22_530200.jpg"

"${PYTHON_BIN}" scripts/orchestrate_roadmap.py \
  --roadmap docs/EXPERIMENT_PLAN.md \
  --gpu-index 1 \
  --budget-hours 72
`