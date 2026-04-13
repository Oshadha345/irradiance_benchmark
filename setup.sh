#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
BACKBONE_DIR="${ROOT_DIR}/models/backbones"
WEIGHTS_DIR="${ROOT_DIR}/weights"
TMP_DIR="$(mktemp -d)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

log_warn() {
  echo "[warn] $*" >&2
}

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

run_python() {
  "${PYTHON_BIN}" "$@"
}

require_python_module() {
  local module_name="$1"
  run_python - "$module_name" <<'PY'
import importlib
import sys

module_name = sys.argv[1]
if importlib.util.find_spec(module_name) is None:
    raise SystemExit(f"Missing required Python module: {module_name}")
PY
}

get_torch_cuda_version() {
  run_python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
}

sanitize_cuda_version() {
  echo "$1" | tr -d '.'
}

get_local_cuda_toolkit_root() {
  local cuda_version="$1"
  if [[ -n "${LOCAL_CUDA_TOOLKIT_ROOT:-}" ]]; then
    echo "${LOCAL_CUDA_TOOLKIT_ROOT}"
    return 0
  fi
  echo "${WORKSPACE_ROOT}/.cuda/cu$(sanitize_cuda_version "${cuda_version}")-toolkit"
}

resolve_nvcc_path() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "${CUDA_HOME}/bin/nvcc"
    return 0
  fi
  command -v nvcc 2>/dev/null || true
}

get_nvcc_version() {
  local nvcc_path="$1"
  if [[ -z "${nvcc_path}" || ! -x "${nvcc_path}" ]]; then
    return 0
  fi
  "${nvcc_path}" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}

ensure_local_cuda_toolkit() {
  local cuda_version="$1"
  local toolkit_root conda_bin
  toolkit_root="$(get_local_cuda_toolkit_root "${cuda_version}")"

  if [[ -x "${toolkit_root}/bin/nvcc" ]]; then
    echo "${toolkit_root}"
    return 0
  fi

  if [[ "${AUTO_INSTALL_LOCAL_CUDA_TOOLKIT:-1}" != "1" ]]; then
    return 1
  fi

  conda_bin="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
  if [[ -z "${conda_bin}" ]]; then
    log_warn "No matching nvcc was found and conda is unavailable to provision a local CUDA ${cuda_version} toolkit."
    return 1
  fi

  echo "[setup] installing project-local CUDA ${cuda_version} toolkit into ${toolkit_root}"
  "${conda_bin}" create -p "${toolkit_root}" -y -c nvidia \
    "cuda-nvcc=${cuda_version}" \
    "cuda-cudart=${cuda_version}" \
    "cuda-cudart-dev=${cuda_version}" \
    "cuda-cccl=${cuda_version}"

  echo "${toolkit_root}"
}

apply_spatial_mamba_compat_patches() {
  run_python - "${BACKBONE_DIR}/spatial_mamba" <<'PY'
from pathlib import Path
import sys

spatial_root = Path(sys.argv[1])
dw_setup = spatial_root / "kernels" / "dwconv2d" / "setup.py"
dw_launch = spatial_root / "kernels" / "dwconv2d" / "depthwise_fwd" / "launch.cu"
utils_file = spatial_root / "utils.py"

setup_text = dw_setup.read_text()
setup_text = setup_text.replace("from Cython.Build import cythonize\n", "")
dw_setup.write_text(setup_text)

launch_text = dw_launch.read_text()
include_block = "#include <ATen/ATen.h>\n#include <ATen/core/grad_mode.h>\n"
if include_block not in launch_text:
    launch_text = launch_text.replace("#include <ATen/ATen.h>\n", include_block)
dw_launch.write_text(launch_text)

utils_text = utils_file.read_text()
if "_require_selective_scan_extension" not in utils_text:
    utils_text = utils_text.replace(
        "import selective_scan_cuda_oflex_rh\n",
        """try:\n    import selective_scan_cuda_oflex_rh\nexcept ImportError as exc:\n    selective_scan_cuda_oflex_rh = None\n    _MISSING_SELECTIVE_SCAN_IMPORT_ERROR = exc\n\n\ndef _require_selective_scan_extension():\n    if selective_scan_cuda_oflex_rh is None:\n        raise RuntimeError(\n            \"Spatial-Mamba selective_scan_cuda_oflex_rh is unavailable. Rebuild the kernels with setup.sh using a CUDA toolkit that matches torch.version.cuda.\"\n        ) from _MISSING_SELECTIVE_SCAN_IMPORT_ERROR\n\n""",
    )
    utils_text = utils_text.replace(
        "        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)\n",
        "        _require_selective_scan_extension()\n        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)\n",
    )
    utils_text = utils_text.replace(
        "        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(\n",
        "        _require_selective_scan_extension()\n        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(\n",
    )
    utils_file.write_text(utils_text)
PY
}

build_spatial_mamba_kernels() {
  local torch_cuda nvcc_path nvcc_version local_cuda_home
  if [[ "${SKIP_SPATIAL_MAMBA_BUILD:-0}" == "1" ]]; then
    echo "[setup] skipping Spatial-Mamba custom kernel build (SKIP_SPATIAL_MAMBA_BUILD=1)"
    return 0
  fi

  torch_cuda="$(get_torch_cuda_version)"
  nvcc_path="$(resolve_nvcc_path)"
  nvcc_version="$(get_nvcc_version "${nvcc_path}")"

  if [[ -z "${torch_cuda}" ]]; then
    log_warn "PyTorch was installed without CUDA support. Skipping Spatial-Mamba kernel build."
    return 0
  fi

  if [[ -z "${nvcc_path}" || -z "${nvcc_version}" || "${torch_cuda}" != "${nvcc_version}" ]]; then
    local_cuda_home="$(ensure_local_cuda_toolkit "${torch_cuda}" || true)"
    if [[ -n "${local_cuda_home}" && -x "${local_cuda_home}/bin/nvcc" ]]; then
      export CUDA_HOME="${local_cuda_home}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      nvcc_path="${CUDA_HOME}/bin/nvcc"
      nvcc_version="$(get_nvcc_version "${nvcc_path}")"
    fi
  fi

  if [[ -z "${nvcc_path}" || -z "${nvcc_version}" ]]; then
    log_warn "No usable nvcc compiler was found. Skipping Spatial-Mamba kernel build."
    return 0
  fi

  if [[ "${torch_cuda}" != "${nvcc_version}" ]]; then
    log_warn "Skipping Spatial-Mamba kernel build because PyTorch expects CUDA ${torch_cuda}, but nvcc reports CUDA ${nvcc_version} at ${nvcc_path}."
    log_warn "Set CUDA_HOME to a matching toolkit, or allow setup.sh to provision a project-local toolkit via conda."
    return 0
  fi

  echo "[setup] building Spatial-Mamba custom kernels with nvcc ${nvcc_path} (CUDA ${nvcc_version})"
  (
    cd "${BACKBONE_DIR}/spatial_mamba/kernels/selective_scan"
    run_python setup.py build_ext --inplace
  )
  (
    cd "${BACKBONE_DIR}/spatial_mamba/kernels/dwconv2d"
    run_python setup.py build_ext --inplace
  )
}

download_http() {
  local url="$1"
  local destination="$2"
  run_python - "$url" "$destination" <<'PY'
from pathlib import Path
import requests
import sys

url, destination = sys.argv[1], Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    print(f"[skip] {destination}")
    raise SystemExit(0)

with requests.get(url, stream=True, timeout=120) as response:
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
print(f"[saved] {destination}")
PY
}

download_hf() {
  local repo_id="$1"
  local filename="$2"
  local destination="$3"
  HF_HUB_DISABLE_XET=1 run_python - "$repo_id" "$filename" "$destination" <<'PY'
from pathlib import Path
import shutil
import sys

repo_id, filename, destination = sys.argv[1], sys.argv[2], Path(sys.argv[3])
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    print(f"[skip] {destination}")
    raise SystemExit(0)

from huggingface_hub import hf_hub_download

downloaded = Path(
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=destination.parent,
        local_dir_use_symlinks=False,
    )
)
if downloaded != destination:
    shutil.move(str(downloaded), str(destination))
print(f"[saved] {destination}")
PY
}

download_gdrive() {
  local file_id="$1"
  local destination="$2"
  run_python - "$file_id" "$destination" <<'PY'
from pathlib import Path
import sys

file_id, destination = sys.argv[1], Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    print(f"[skip] {destination}")
    raise SystemExit(0)

import gdown

gdown.download(id=file_id, output=str(destination), quiet=False)
print(f"[saved] {destination}")
PY
}

echo "[setup] verifying Python runtime"
require_python_module torch

if [[ "${SKIP_PYTHON_DEPS:-0}" == "1" ]]; then
  echo "[setup] skipping Python dependency installation (SKIP_PYTHON_DEPS=1)"
else
  echo "[setup] installing Python dependencies"
  if [[ "${UPGRADE_PIP:-0}" == "1" ]]; then
    run_python -m pip install --upgrade pip
  fi
  run_python -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

mkdir -p "${BACKBONE_DIR}" "${WEIGHTS_DIR}"

echo "[setup] cloning official sources"
git clone --depth 1 https://github.com/MzeroMiko/VMamba.git "${TMP_DIR}/VMamba"
git clone --depth 1 https://github.com/NVlabs/MambaVision.git "${TMP_DIR}/MambaVision"
# NOTE: the public paper-linked Spatial-Mamba repository is EdwardChasel/Spatial-Mamba.
git clone --depth 1 "${SPATIAL_MAMBA_REPO:-https://github.com/EdwardChasel/Spatial-Mamba.git}" "${TMP_DIR}/Spatial-Mamba"

echo "[setup] copying VMamba sources"
mkdir -p "${BACKBONE_DIR}/vmamba"
cp "${TMP_DIR}/VMamba/vmamba.py" "${BACKBONE_DIR}/vmamba/vmamba.py"

echo "[setup] copying MambaVision sources"
mkdir -p "${BACKBONE_DIR}/mambavision"
cp "${TMP_DIR}/MambaVision/mambavision/models/__init__.py" "${BACKBONE_DIR}/mambavision/__init__.py"
cp "${TMP_DIR}/MambaVision/mambavision/models/mamba_vision.py" "${BACKBONE_DIR}/mambavision/mamba_vision.py"
cp "${TMP_DIR}/MambaVision/mambavision/models/registry.py" "${BACKBONE_DIR}/mambavision/registry.py"

echo "[setup] copying Spatial-Mamba sources"
mkdir -p "${BACKBONE_DIR}/spatial_mamba"
cp "${TMP_DIR}/Spatial-Mamba/classification/models/__init__.py" "${BACKBONE_DIR}/spatial_mamba/__init__.py"
cp "${TMP_DIR}/Spatial-Mamba/classification/models/spatialmamba.py" "${BACKBONE_DIR}/spatial_mamba/spatialmamba.py"
cp "${TMP_DIR}/Spatial-Mamba/classification/models/utils.py" "${BACKBONE_DIR}/spatial_mamba/utils.py"
rm -rf "${BACKBONE_DIR}/spatial_mamba/kernels"
mkdir -p "${BACKBONE_DIR}/spatial_mamba/kernels"
cp -r "${TMP_DIR}/Spatial-Mamba/kernels/selective_scan" "${BACKBONE_DIR}/spatial_mamba/kernels/selective_scan"
cp -r "${TMP_DIR}/Spatial-Mamba/kernels/dwconv2d" "${BACKBONE_DIR}/spatial_mamba/kernels/dwconv2d"
apply_spatial_mamba_compat_patches

build_spatial_mamba_kernels

echo "[setup] downloading VMamba ImageNet-1K weights"
download_http "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth" "${WEIGHTS_DIR}/vmamba_tiny_1k.pth"
download_http "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth" "${WEIGHTS_DIR}/vmamba_small_1k.pth"
download_http "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth" "${WEIGHTS_DIR}/vmamba_base_1k.pth"

echo "[setup] downloading MambaVision ImageNet-1K weights"
download_hf "nvidia/MambaVision-T-1K" "mambavision_tiny_1k.pth.tar" "${WEIGHTS_DIR}/mambavision_tiny_1k.pth.tar"
download_hf "nvidia/MambaVision-T2-1K" "mambavision_tiny2_1k.pth.tar" "${WEIGHTS_DIR}/mambavision_tiny2_1k.pth.tar"
download_hf "nvidia/MambaVision-S-1K" "mambavision_small_1k.pth.tar" "${WEIGHTS_DIR}/mambavision_small_1k.pth.tar"
download_hf "nvidia/MambaVision-B-1K" "mambavision_base_1k.pth.tar" "${WEIGHTS_DIR}/mambavision_base_1k.pth.tar"
download_hf "nvidia/MambaVision-L-1K" "mambavision_large_1k.pth.tar" "${WEIGHTS_DIR}/mambavision_large_1k.pth.tar"
download_hf "nvidia/MambaVision-L2-1K" "mambavision_large2_1k.pth.tar" "${WEIGHTS_DIR}/mambavision_large2_1k.pth.tar"

echo "[setup] downloading Spatial-Mamba ImageNet-1K weights"
download_gdrive "19kXoqGSTuKKs4AHbdUSrdKZTwTWenLIW" "${WEIGHTS_DIR}/spatial_mamba_tiny_1k.pth"
download_gdrive "1Wb3sYoWLpgmWrmHMYKwdgDwGPZaqM28O" "${WEIGHTS_DIR}/spatial_mamba_small_1k.pth"
download_gdrive "1k8dHp2QRCOqBSgAi36YkhZp_O8LqOPjM" "${WEIGHTS_DIR}/spatial_mamba_base_1k.pth"

echo "[setup] downloading Swin ImageNet-1K weights"
download_hf "timm/swin_tiny_patch4_window7_224.ms_in1k" "model.safetensors" "${WEIGHTS_DIR}/swin_tiny_patch4_window7_224.ms_in1k.safetensors"
download_hf "timm/swin_small_patch4_window7_224.ms_in1k" "model.safetensors" "${WEIGHTS_DIR}/swin_small_patch4_window7_224.ms_in1k.safetensors"
download_hf "timm/swin_base_patch4_window7_224.ms_in1k" "model.safetensors" "${WEIGHTS_DIR}/swin_base_patch4_window7_224.ms_in1k.safetensors"

echo "[setup] completed"
echo "Backbones: ${BACKBONE_DIR}"
echo "Weights:   ${WEIGHTS_DIR}"
