#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="${ROOT_DIR}/results/.locks"

GPU_INDEX="${GPU_INDEX:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOCK_DIR}"
LOCK_FILE="${LOCK_DIR}/gpu${GPU_INDEX}.lock"

gpu_compute_pids() {
  nvidia-smi -i "${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sed '/No running processes found/d'
}

wait_for_gpu_idle() {
  while true; do
    mapfile -t active_pids < <(gpu_compute_pids || true)
    if [[ "${#active_pids[@]}" -eq 0 ]]; then
      return 0
    fi

    echo "[wait] GPU ${GPU_INDEX} busy with compute PIDs: ${active_pids[*]}. Sleeping ${GPU_POLL_SECONDS}s."
    sleep "${GPU_POLL_SECONDS}"
  done
}

exec 9>"${LOCK_FILE}"
flock 9
echo "[lock] acquired ${LOCK_FILE}"
wait_for_gpu_idle

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" scripts/train.py "$@"
