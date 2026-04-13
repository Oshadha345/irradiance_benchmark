from __future__ import annotations

import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gpu_compute_pids(gpu_index: int) -> list[int]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return []

    pids = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or "No running processes found" in stripped:
            continue
        try:
            pids.append(int(stripped))
        except ValueError:
            continue
    return pids


@contextmanager
def exclusive_gpu(gpu_index: int, *, poll_seconds: float = 1.0) -> Iterator[None]:
    lock_dir = PROJECT_ROOT / "results" / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"gpu{gpu_index}.lock"
    wait_frames = [".   ", "..  ", "... ", "...."]

    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        wait_index = 0
        while True:
            active_pids = gpu_compute_pids(gpu_index)
            if not active_pids:
                break
            frame = wait_frames[wait_index % len(wait_frames)]
            print(f"\r[wait] GPU {gpu_index} busy {frame}", end="", flush=True)
            wait_index += 1
            time.sleep(max(0.2, float(poll_seconds)))
        if wait_index:
            print(f"\r[wait] GPU {gpu_index} free. Starting...    ")

        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
