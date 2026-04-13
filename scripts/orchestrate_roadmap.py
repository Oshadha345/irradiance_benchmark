#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_data_loaders, unwrap_dataset
from train import autotune_batch_size, build_optimizer, enable_fast_runtime_paths, select_primary_horizon, synchronize_if_needed
from models import build_model
from utils.gpu_control import exclusive_gpu
from utils.runtime import load_config, model_slug, seed_everything, slugify_comment


DEFAULT_ERF_IMAGES = {
    "folsom": "/storage2/CV_Irradiance/datasets/1_Folsom/2014/04/04/20140404_010659.jpg",
    "nrel": "/storage2/CV_Irradiance/datasets/4_NREL/2019_09_07/images/UTC-7_2019_09_07-09_50_22_530200.jpg",
}

TRAIN_CMD_PATTERN = re.compile(r"--config\s+([^\s`]+)\s+--comment\s+\"([^\"]+)\"")


@dataclass
class ExperimentSpec:
    config_path: Path
    comment: str
    dataset: str
    model: str
    experiment_id: str


@dataclass
class TimeBudget:
    deadline_iso: str
    remaining_experiments: int
    remaining_seconds: float
    per_experiment_train_seconds: float
    max_epoch_seconds: float
    planned_epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full irradiance benchmark roadmap sequentially with adaptive time-budget control.")
    parser.add_argument("--roadmap", type=str, default="docs/EXPERIMENT_PLAN.md", help="Markdown roadmap containing train commands.")
    parser.add_argument("--python-bin", type=str, default=os.environ.get("PYTHON_BIN") or sys.executable, help="Python interpreter used for train/postprocess.")
    parser.add_argument("--gpu-index", type=int, default=1, help="GPU index to reserve exclusively for the roadmap.")
    parser.add_argument("--gpu-poll-seconds", type=float, default=1.0, help="How frequently to poll for GPU idleness while waiting to acquire the target GPU.")
    parser.add_argument("--budget-hours", type=float, default=72.0, help="Total wall-clock budget allocated to all remaining experiments.")
    parser.add_argument("--planned-epochs", type=int, default=8, help="Planned epochs per experiment used when deriving the max epoch-time budget.")
    parser.add_argument("--min-epochs", type=int, default=3, help="Minimum number of epochs to allow before convergence/budget stopping.")
    parser.add_argument("--postprocess-minutes", type=float, default=12.0, help="Reserved minutes per experiment for postprocess/plot generation.")
    parser.add_argument("--safety-factor", type=float, default=1.15, help="Conservative divisor applied to the per-epoch time budget.")
    parser.add_argument("--profile-steps", type=int, default=12, help="Number of training steps used to estimate epoch time for each candidate.")
    parser.add_argument("--profile-warmup-steps", type=int, default=3, help="Warmup steps excluded from timing during candidate profiling.")
    parser.add_argument("--window-fractions", type=str, default="1.0,0.875,0.75,0.625,0.5", help="Descending train-window fractions evaluated against the time budget.")
    parser.add_argument("--train-strides", type=str, default="1,2,3,4,6,8", help="Ascending train subset strides considered for each candidate window.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum RMSE improvement required to reset patience.")
    parser.add_argument("--state-path", type=str, default="results/orchestrator/roadmap_state.json", help="Persistent orchestration state file.")
    parser.add_argument("--generated-config-dir", type=str, default="results/orchestrator/generated_configs", help="Directory for tuned configs emitted by the orchestrator.")
    parser.add_argument("--summary-dir", type=str, default="results/orchestrator", help="Directory for summary CSVs and plots.")
    parser.add_argument("--max-experiments", type=int, default=None, help="Optional cap on how many pending experiments to process in this run.")
    parser.add_argument("--reuse-existing", action="store_true", help="Skip experiments that already have complete outputs on disk.")
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false", help="Ignore existing run folders and rerun everything.")
    parser.set_defaults(reuse_existing=True)
    return parser.parse_args()


def parse_fraction_list(value: str) -> list[float]:
    fractions = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        fraction = float(stripped)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"Invalid window fraction: {fraction}")
        fractions.append(fraction)
    return sorted(set(fractions), reverse=True)


def parse_int_list(value: str) -> list[int]:
    integers = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        parsed = int(stripped)
        if parsed < 1:
            raise ValueError(f"Invalid positive integer: {parsed}")
        integers.append(parsed)
    return sorted(set(integers))


def resolve_cli_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"experiments": {}, "created_at": datetime.now().isoformat()}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_state(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=False)


def parse_roadmap(path: Path) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TRAIN_CMD_PATTERN.search(line)
        if not match:
            continue
        config_path = (PROJECT_ROOT / match.group(1)).resolve()
        config = load_config(config_path)
        dataset = str(config["data"]["dataset_type"]).lower()
        model = model_slug(config)
        experiment_id = f"{dataset}:{model}:{slugify_comment(match.group(2))}"
        if experiment_id in seen:
            continue
        seen.add(experiment_id)
        experiments.append(
            ExperimentSpec(
                config_path=config_path,
                comment=match.group(2),
                dataset=dataset,
                model=model,
                experiment_id=experiment_id,
            )
        )
    if not experiments:
        raise RuntimeError(f"No training commands were parsed from roadmap: {path}")
    return experiments


def expected_run_dir_pattern(config: dict[str, Any], comment: str) -> str:
    return f"{model_slug(config)}_{slugify_comment(comment)}_*"


def latest_matching_run(config: dict[str, Any], comment: str) -> Path | None:
    dataset = str(config["data"]["dataset_type"]).lower()
    root = PROJECT_ROOT / "results" / dataset
    pattern = expected_run_dir_pattern(config, comment)
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def is_run_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "best.ckpt",
        run_dir / "test_metrics.json",
        run_dir / "efficiency_report.json",
        run_dir / "erf_overlay.png",
        run_dir / "status.json",
    ]
    return all(path.exists() for path in required)


def run_has_training_artifacts(run_dir: Path) -> bool:
    return (run_dir / "best.ckpt").is_file() and (run_dir / "status.json").is_file() and any(
        candidate.is_file() for candidate in (run_dir / "config.yaml", run_dir / "config.json")
    )


def resolve_run_config_path(run_dir: Path, fallback: Path | None = None) -> Path:
    for candidate in (run_dir / "config.yaml", run_dir / "config.json"):
        if candidate.is_file():
            return candidate
    if fallback is not None and fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No run config found in {run_dir}")


def resolve_input_image(dataset: str) -> str:
    env_name = f"{dataset.upper()}_ERF_IMAGE"
    return os.environ.get(env_name, DEFAULT_ERF_IMAGES[dataset])


def parse_visible_physical_gpus() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None
    visible: list[int] = []
    for token in tokens:
        if not token.isdigit():
            return None
        visible.append(int(token))
    return visible or None


def resolve_gpu_request(requested_gpu_index: int) -> tuple[int, int]:
    visible_physical = parse_visible_physical_gpus()
    if visible_physical:
        if requested_gpu_index in visible_physical:
            return requested_gpu_index, visible_physical.index(requested_gpu_index)
        if 0 <= requested_gpu_index < len(visible_physical):
            physical_gpu_index = visible_physical[requested_gpu_index]
            print(
                f"[gpu] treating requested gpu_index={requested_gpu_index} as a logical visible index "
                f"within CUDA_VISIBLE_DEVICES={','.join(str(value) for value in visible_physical)}; "
                f"physical_gpu={physical_gpu_index}"
            )
            return physical_gpu_index, requested_gpu_index
        raise RuntimeError(
            f"Requested gpu_index={requested_gpu_index}, but CUDA_VISIBLE_DEVICES exposes "
            f"physical GPU(s) {visible_physical}. Use one of those physical indices or a logical index "
            f"between 0 and {len(visible_physical) - 1}."
        )

    visible_count = torch.cuda.device_count()
    if requested_gpu_index < 0 or requested_gpu_index >= visible_count:
        raise RuntimeError(
            f"Requested gpu_index={requested_gpu_index}, but only {visible_count} CUDA device(s) are visible."
        )
    return requested_gpu_index, requested_gpu_index


def compute_time_budget(
    *,
    deadline: datetime,
    remaining_experiments: int,
    base_config: dict[str, Any],
    args: argparse.Namespace,
) -> TimeBudget:
    remaining_seconds = max((deadline - datetime.now()).total_seconds(), 3600.0)
    reserve_seconds = remaining_experiments * (args.postprocess_minutes * 60.0)
    train_seconds = max(remaining_seconds - reserve_seconds, remaining_seconds * 0.5)
    per_experiment_train_seconds = train_seconds / max(remaining_experiments, 1)
    requested_epochs = int(base_config["training"].get("epochs", 50))
    planned_epochs = max(args.min_epochs, min(requested_epochs, args.planned_epochs))
    max_epoch_seconds = per_experiment_train_seconds / max(planned_epochs, 1)
    max_epoch_seconds /= max(args.safety_factor, 1e-6)
    return TimeBudget(
        deadline_iso=deadline.isoformat(),
        remaining_experiments=remaining_experiments,
        remaining_seconds=remaining_seconds,
        per_experiment_train_seconds=per_experiment_train_seconds,
        max_epoch_seconds=max_epoch_seconds,
        planned_epochs=planned_epochs,
    )


def dataset_timestamps(loader: torch.utils.data.DataLoader) -> list[pd.Timestamp]:
    subset = loader.dataset
    base_dataset = unwrap_dataset(subset)
    indices = list(subset.indices) if hasattr(subset, "indices") else list(range(len(base_dataset)))
    return [pd.Timestamp(base_dataset.samples[index][2]) for index in indices]


def extract_split_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata_config = deepcopy(config)
    metadata_config["data"].pop("split_windows", None)
    metadata_config["data"]["train_stride"] = 1
    metadata_config["data"]["val_stride"] = 1
    metadata_config["data"]["test_stride"] = 1
    train_loader, val_loader, test_loader = get_data_loaders(metadata_config)
    train_times = dataset_timestamps(train_loader)
    val_times = dataset_timestamps(val_loader)
    test_times = dataset_timestamps(test_loader)
    trainval_times = sorted(train_times + val_times)
    if not trainval_times or not test_times:
        raise RuntimeError("Unable to derive benchmark split timestamps for orchestration.")
    return {
        "trainval_times": trainval_times,
        "trainval_start": trainval_times[0],
        "trainval_end": trainval_times[-1],
        "test_start": min(test_times),
        "test_end": max(test_times),
    }


def build_window_candidates(trainval_times: list[pd.Timestamp], fractions: list[float]) -> list[tuple[float, pd.Timestamp]]:
    candidates: list[tuple[float, pd.Timestamp]] = []
    seen: set[pd.Timestamp] = set()
    last_index = len(trainval_times) - 1
    for fraction in fractions:
        index = max(0, min(last_index, math.ceil(len(trainval_times) * fraction) - 1))
        end_time = trainval_times[index]
        if end_time in seen:
            continue
        seen.add(end_time)
        candidates.append((fraction, end_time))
    if trainval_times[-1] not in seen:
        candidates.insert(0, (1.0, trainval_times[-1]))
    return candidates


def dataset_policy_from_config(config: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = {
        "split_windows": deepcopy(config["data"].get("split_windows")),
        "train_stride": int(config["data"].get("train_stride", 1)),
        "val_stride": int(config["data"].get("val_stride", 1)),
        "test_stride": int(config["data"].get("test_stride", 1)),
    }
    if profile:
        policy["window_fraction"] = profile.get("window_fraction")
        policy["trainval_end"] = profile.get("trainval_end")
    return policy


def apply_dataset_policy(config: dict[str, Any], dataset_policy: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(config)
    split_windows = deepcopy(dataset_policy.get("split_windows"))
    if split_windows:
        candidate["data"]["split_windows"] = split_windows
    else:
        candidate["data"].pop("split_windows", None)
    candidate["data"]["train_stride"] = int(dataset_policy.get("train_stride", 1))
    candidate["data"]["val_stride"] = int(dataset_policy.get("val_stride", 1))
    candidate["data"]["test_stride"] = int(dataset_policy.get("test_stride", 1))
    return candidate


def bootstrap_dataset_policies(state: dict[str, Any]) -> None:
    policies = state.setdefault("dataset_policies", {})
    if policies:
        return
    for payload in state.get("experiments", {}).values():
        if payload.get("status") != "completed":
            continue
        dataset = payload.get("dataset")
        if not dataset or dataset in policies:
            continue
        config_candidates: list[Path] = []
        tuned_config_path = payload.get("tuned_config_path")
        if tuned_config_path:
            config_candidates.append(Path(tuned_config_path))
        run_dir_value = payload.get("run_dir")
        if run_dir_value:
            run_dir = Path(run_dir_value)
            config_candidates.extend([run_dir / "config.yaml", run_dir / "config.json"])
        for candidate in config_candidates:
            if candidate.is_file():
                policies[dataset] = dataset_policy_from_config(load_config(candidate), profile=payload.get("profile"))
                break


def profile_candidate(config: dict[str, Any], *, profile_steps: int, profile_warmup_steps: int, device: torch.device) -> dict[str, Any]:
    profile_config = deepcopy(config)
    seed_everything(int(profile_config["training"].get("seed", 42)))
    enable_fast_runtime_paths(device)

    model = build_model(profile_config).to(device)
    requested_batch_size = int(profile_config["data"].get("batch_size", 16))
    batch_size = autotune_batch_size(model, device, profile_config, requested_batch_size)
    profile_config["data"]["batch_size"] = batch_size

    train_loader, val_loader, test_loader = get_data_loaders(profile_config)
    if len(train_loader) == 0:
        raise RuntimeError("Profile candidate produced an empty training loader.")
    train_batches = len(train_loader)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)

    optimizer = build_optimizer(model, profile_config)
    criterion = nn.HuberLoss(delta=float(profile_config["training"].get("huber_delta", 1.0)))
    amp_enabled = device.type == "cuda" and bool(profile_config["training"].get("use_amp", True))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    total_steps = min(int(profile_steps), len(train_loader))
    warmup_steps = min(int(profile_warmup_steps), total_steps - 1) if total_steps > 1 else 0
    timed_seconds = 0.0
    timed_steps = 0

    iterator = iter(train_loader)
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step_index in range(total_steps):
            step_start = time.perf_counter()
            batch = next(iterator)
            batch = tuple(item.to(device, non_blocking=True) if hasattr(item, "to") else item for item in batch)
            images, weather_seq, targets = batch[:3]
            targets = select_primary_horizon(targets)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions, _ = model(images, weather_seq)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(profile_config["training"].get("grad_clip", 5.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            synchronize_if_needed(device)

            if step_index >= warmup_steps:
                timed_seconds += time.perf_counter() - step_start
                timed_steps += 1
    finally:
        del iterator
        del optimizer
        del criterion
        del scaler
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_step_seconds = timed_seconds / max(timed_steps, 1)
    estimated_epoch_seconds = avg_step_seconds * train_batches
    result = {
        "batch_size": batch_size,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "train_batches": train_batches,
        "avg_step_seconds": avg_step_seconds,
        "estimated_epoch_seconds": estimated_epoch_seconds,
    }
    del train_loader
    del val_loader
    del test_loader
    return result


def apply_budget_controls(
    config: dict[str, Any],
    *,
    profile: dict[str, Any],
    budget: TimeBudget,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tuned = deepcopy(config)
    tuned["data"]["batch_size"] = int(profile["batch_size"])
    tuned["training"]["auto_batch_size"] = False
    requested_epochs = int(tuned["training"].get("epochs", 50))
    budget_epochs = max(args.min_epochs, int(budget.per_experiment_train_seconds / max(profile["estimated_epoch_seconds"], 1.0)))
    tuned_epochs = max(args.min_epochs, min(requested_epochs, budget_epochs))
    tuned["training"]["epochs"] = tuned_epochs
    tuned["training"]["min_epochs"] = min(args.min_epochs, tuned_epochs)
    tuned["training"]["early_stop_patience"] = min(
        int(tuned["training"].get("early_stop_patience", 12)),
        max(2, tuned_epochs // 3),
    )
    tuned["training"]["early_stop_min_delta"] = float(args.early_stop_min_delta)
    tuned["training"]["max_total_train_seconds"] = float(budget.per_experiment_train_seconds)
    tuned["training"]["max_epoch_seconds"] = float(budget.max_epoch_seconds)
    return tuned


def tune_experiment(
    spec: ExperimentSpec,
    base_config: dict[str, Any],
    *,
    device: torch.device,
    budget: TimeBudget,
    args: argparse.Namespace,
    dataset_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset_policy is not None:
        candidate = apply_dataset_policy(base_config, dataset_policy)
        candidate["training"]["auto_batch_size"] = True
        profile = profile_candidate(
            candidate,
            profile_steps=args.profile_steps,
            profile_warmup_steps=args.profile_warmup_steps,
            device=device,
        )
        profile["window_fraction"] = dataset_policy.get("window_fraction")
        profile["train_stride"] = int(candidate["data"].get("train_stride", 1))
        profile["trainval_end"] = dataset_policy.get("trainval_end")
        tuned = apply_budget_controls(candidate, profile=profile, budget=budget, args=args)
        return {
            "config": tuned,
            "profile": profile,
            "budget": asdict(budget),
            "selection_reason": "shared_dataset_policy",
            "dataset_policy": dataset_policy_from_config(tuned, profile=profile),
        }

    split_metadata = extract_split_metadata(base_config)
    fractions = parse_fraction_list(args.window_fractions)
    strides = parse_int_list(args.train_strides)
    window_candidates = build_window_candidates(split_metadata["trainval_times"], fractions)

    fallback: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for window_fraction, trainval_end in window_candidates:
        for train_stride in strides:
            candidate = deepcopy(base_config)
            candidate["data"]["split_windows"] = {
                "trainval_start": split_metadata["trainval_start"].isoformat(),
                "trainval_end": trainval_end.isoformat(),
                "test_start": split_metadata["test_start"].isoformat(),
                "test_end": split_metadata["test_end"].isoformat(),
            }
            candidate["data"]["train_stride"] = train_stride
            candidate["data"]["val_stride"] = 1
            candidate["data"]["test_stride"] = 1
            candidate["training"]["auto_batch_size"] = True

            profile = profile_candidate(
                candidate,
                profile_steps=args.profile_steps,
                profile_warmup_steps=args.profile_warmup_steps,
                device=device,
            )
            profile["window_fraction"] = window_fraction
            profile["train_stride"] = train_stride
            profile["trainval_end"] = trainval_end.isoformat()
            tuned = apply_budget_controls(candidate, profile=profile, budget=budget, args=args)

            if fallback is None or profile["estimated_epoch_seconds"] < fallback[1]["estimated_epoch_seconds"]:
                fallback = (tuned, profile, asdict(budget))

            print(
                f"[candidate] {spec.experiment_id} window_fraction={window_fraction:.3f} "
                f"train_stride={train_stride} batch_size={profile['batch_size']} "
                f"train_samples={profile['train_samples']} estimated_epoch_seconds={profile['estimated_epoch_seconds']:.2f}"
            )
            if profile["estimated_epoch_seconds"] <= budget.max_epoch_seconds:
                return {
                    "config": tuned,
                    "profile": profile,
                    "budget": asdict(budget),
                    "selection_reason": "first_candidate_within_epoch_budget",
                    "dataset_policy": dataset_policy_from_config(tuned, profile=profile),
                }

    if fallback is None:
        raise RuntimeError(f"No valid candidate profiles were produced for experiment {spec.experiment_id}.")
    return {
        "config": fallback[0],
        "profile": fallback[1],
        "budget": fallback[2],
        "selection_reason": "fallback_fastest_candidate",
        "dataset_policy": dataset_policy_from_config(fallback[0], profile=fallback[1]),
    }


def write_tuned_config(spec: ExperimentSpec, config: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{spec.experiment_id.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return destination


def run_command(command: list[str], *, env: dict[str, str]) -> None:
    print(f"[exec] {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def locate_fresh_run(config: dict[str, Any], comment: str, started_at: float) -> Path:
    dataset = str(config["data"]["dataset_type"]).lower()
    root = PROJECT_ROOT / "results" / dataset
    pattern = expected_run_dir_pattern(config, comment)
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in matches:
        if path.stat().st_mtime >= started_at - 2.0:
            return path
    raise RuntimeError(f"Unable to locate a fresh run directory for pattern {pattern}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_metric(summary: dict[str, Any], *keys: str) -> float | None:
    current: Any = summary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    return float(current)


def generate_learning_curve(run_dir: Path) -> Path | None:
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return None
    history = load_json(history_path).get("history", [])
    if not history:
        return None

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_rmse = [entry["val_rmse"] for entry in history]

    figure, axis_left = plt.subplots(figsize=(7, 4))
    axis_left.plot(epochs, train_loss, color="#1f77b4", label="Train Loss")
    axis_left.set_xlabel("Epoch")
    axis_left.set_ylabel("Train Loss", color="#1f77b4")
    axis_left.tick_params(axis="y", labelcolor="#1f77b4")

    axis_right = axis_left.twinx()
    axis_right.plot(epochs, val_rmse, color="#d62728", label="Val RMSE")
    axis_right.set_ylabel("Val RMSE", color="#d62728")
    axis_right.tick_params(axis="y", labelcolor="#d62728")

    figure.tight_layout()
    destination = run_dir / "learning_curve.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def build_summary_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for experiment_id, payload in state.get("experiments", {}).items():
        if payload.get("status") != "completed":
            continue
        records.append(
            {
                "experiment_id": experiment_id,
                "dataset": payload.get("dataset"),
                "model": payload.get("model"),
                "comment": payload.get("comment"),
                "run_dir": payload.get("run_dir"),
                "batch_size": payload.get("profile", {}).get("batch_size"),
                "train_stride": payload.get("profile", {}).get("train_stride"),
                "window_fraction": payload.get("profile", {}).get("window_fraction"),
                "train_samples": payload.get("profile", {}).get("train_samples"),
                "train_batches": payload.get("profile", {}).get("train_batches"),
                "estimated_epoch_seconds": payload.get("profile", {}).get("estimated_epoch_seconds"),
                "actual_mean_epoch_seconds": payload.get("actual_mean_epoch_seconds"),
                "epochs_completed": payload.get("status_info", {}).get("epochs_completed"),
                "stop_reason": payload.get("status_info", {}).get("stop_reason"),
                "test_rmse": payload.get("metrics", {}).get("test_rmse"),
                "val_rmse": payload.get("metrics", {}).get("val_rmse"),
                "fps": payload.get("efficiency", {}).get("fps"),
            }
        )
    return records


def write_summary_artifacts(state: dict[str, Any], summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    records = build_summary_records(state)
    if not records:
        return
    frame = pd.DataFrame.from_records(records).sort_values(["dataset", "model"])
    csv_path = summary_dir / "summary.csv"
    frame.to_csv(csv_path, index=False)

    labels = frame["experiment_id"].tolist()

    plt.figure(figsize=(max(10, len(labels) * 0.35), 4))
    plt.bar(labels, frame["test_rmse"].fillna(0.0))
    plt.xticks(rotation=90)
    plt.ylabel("Test RMSE")
    plt.tight_layout()
    plt.savefig(summary_dir / "summary_test_rmse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(max(10, len(labels) * 0.35), 4))
    plt.bar(labels, frame["actual_mean_epoch_seconds"].fillna(frame["estimated_epoch_seconds"]).fillna(0.0))
    plt.xticks(rotation=90)
    plt.ylabel("Epoch Seconds")
    plt.tight_layout()
    plt.savefig(summary_dir / "summary_epoch_seconds.png", dpi=180)
    plt.close()


def summarize_run(run_dir: Path) -> dict[str, Any]:
    history = load_json(run_dir / "history.json").get("history", []) if (run_dir / "history.json").is_file() else []
    status_info = load_json(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
    test_metrics = load_json(run_dir / "test_metrics.json") if (run_dir / "test_metrics.json").is_file() else {}
    val_metrics = load_json(run_dir / "best_val_metrics.json") if (run_dir / "best_val_metrics.json").is_file() else {}
    efficiency = load_json(run_dir / "efficiency_report.json") if (run_dir / "efficiency_report.json").is_file() else {}

    actual_mean_epoch_seconds = None
    if history:
        actual_mean_epoch_seconds = float(sum(entry.get("epoch_seconds", 0.0) for entry in history) / max(len(history), 1))

    return {
        "status_info": status_info,
        "actual_mean_epoch_seconds": actual_mean_epoch_seconds,
        "metrics": {
            "test_rmse": extract_metric(test_metrics, "ghi", "overall", "RMSE"),
            "val_rmse": extract_metric(val_metrics, "ghi", "overall", "RMSE"),
        },
        "efficiency": {
            "fps": efficiency.get("fps"),
            "params_millions": efficiency.get("params_millions"),
        },
    }


def maybe_mark_existing_run(
    spec: ExperimentSpec,
    *,
    base_config: dict[str, Any],
    state: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not args.reuse_existing:
        return False
    existing = latest_matching_run(base_config, spec.comment)
    if existing is None:
        return False
    if not existing.is_dir():
        return False
    if run_has_training_artifacts(existing) and not is_run_complete(existing):
        state.setdefault("dataset_policies", {}).setdefault(
            spec.dataset,
            dataset_policy_from_config(load_config(resolve_run_config_path(existing, fallback=spec.config_path))),
        )
        state["experiments"][spec.experiment_id] = {
            "status": "needs_postprocess",
            "dataset": spec.dataset,
            "model": spec.model,
            "comment": spec.comment,
            "config_path": str(spec.config_path),
            "run_dir": str(existing),
            "selection_reason": "reuse_training_only_run",
        }
        return False
    if not is_run_complete(existing):
        return False

    state.setdefault("dataset_policies", {}).setdefault(
        spec.dataset,
        dataset_policy_from_config(load_config(resolve_run_config_path(existing, fallback=spec.config_path))),
    )
    summary = summarize_run(existing)
    state["experiments"][spec.experiment_id] = {
        "status": "completed",
        "dataset": spec.dataset,
        "model": spec.model,
        "comment": spec.comment,
        "config_path": str(spec.config_path),
        "run_dir": str(existing),
        **summary,
        "completed_at": datetime.now().isoformat(),
        "selection_reason": "reused_existing_run",
    }
    return True


def main() -> None:
    args = parse_args()
    roadmap_path = resolve_cli_path(args.roadmap)
    state_path = resolve_cli_path(args.state_path)
    generated_config_dir = resolve_cli_path(args.generated_config_dir)
    summary_dir = resolve_cli_path(args.summary_dir)

    experiments = parse_roadmap(roadmap_path)
    state = read_state(state_path)
    state.setdefault("experiments", {})
    bootstrap_dataset_policies(state)
    state["roadmap"] = str(roadmap_path)
    state["budget_hours"] = args.budget_hours
    state["gpu_index"] = args.gpu_index

    for spec in experiments:
        if spec.experiment_id in state["experiments"] and state["experiments"][spec.experiment_id].get("status") == "completed":
            continue
        base_config = load_config(spec.config_path)
        if maybe_mark_existing_run(spec, base_config=base_config, state=state, args=args):
            write_state(state, state_path)
            write_summary_artifacts(state, summary_dir)

    pending = [spec for spec in experiments if state["experiments"].get(spec.experiment_id, {}).get("status") != "completed"]
    if args.max_experiments is not None:
        pending = pending[: args.max_experiments]
    if not pending:
        write_state(state, state_path)
        write_summary_artifacts(state, summary_dir)
        print("[done] roadmap already complete.")
        return

    deadline = datetime.now() + timedelta(hours=float(args.budget_hours))
    if not torch.cuda.is_available():
        raise RuntimeError("The roadmap orchestrator expects a CUDA device.")
    physical_gpu_index, visible_gpu_index = resolve_gpu_request(args.gpu_index)
    torch.cuda.set_device(visible_gpu_index)
    device = torch.device(f"cuda:{visible_gpu_index}")
    print(
        f"[gpu] requested={args.gpu_index} physical={physical_gpu_index} visible={visible_gpu_index} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES') or '<all>'}"
    )

    with exclusive_gpu(physical_gpu_index, poll_seconds=args.gpu_poll_seconds):
        os.environ.setdefault("PYTHONNOUSERSITE", "1")
        for index, spec in enumerate(pending, start=1):
            base_config = load_config(spec.config_path)
            remaining = len(pending) - index + 1
            budget = compute_time_budget(deadline=deadline, remaining_experiments=remaining, base_config=base_config, args=args)
            print(
                f"[budget] experiment={spec.experiment_id} remaining_experiments={remaining} "
                f"max_epoch_seconds={budget.max_epoch_seconds:.2f} "
                f"per_experiment_train_seconds={budget.per_experiment_train_seconds:.2f}"
            )

            existing_entry = state["experiments"].get(spec.experiment_id, {})
            if existing_entry.get("status") == "needs_postprocess":
                existing_entry["budget"] = asdict(budget)
                existing_entry["started_at"] = datetime.now().isoformat()
                state["experiments"][spec.experiment_id] = existing_entry
            else:
                state["experiments"][spec.experiment_id] = {
                    "status": "profiling",
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "comment": spec.comment,
                    "config_path": str(spec.config_path),
                    "budget": asdict(budget),
                    "started_at": datetime.now().isoformat(),
                }
            write_state(state, state_path)

            run_dir: Path | None = None
            tuned_config_path: Path | None = None
            tuned: dict[str, Any] | None = None
            profile: dict[str, Any] | None = None
            try:
                existing_entry = state["experiments"].get(spec.experiment_id, {})
                if existing_entry.get("status") == "needs_postprocess" and existing_entry.get("run_dir"):
                    run_dir = Path(existing_entry["run_dir"])
                    if not run_dir.is_dir():
                        raise FileNotFoundError(f"Saved run directory no longer exists: {run_dir}")
                    run_config_path = resolve_run_config_path(run_dir, fallback=spec.config_path)
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
                    env["PYTHONNOUSERSITE"] = env.get("PYTHONNOUSERSITE", "1")
                    input_image = resolve_input_image(spec.dataset)
                    run_command(
                        [
                            args.python_bin,
                            "scripts/postprocess.py",
                            "--config",
                            str(run_config_path),
                            "--run-dir",
                            str(run_dir),
                            "--input-image",
                            input_image,
                            "--device",
                            "cuda",
                        ],
                        env=env,
                    )
                    learning_curve = generate_learning_curve(run_dir)
                    summary = summarize_run(run_dir)
                    state["experiments"][spec.experiment_id] = {
                        "status": "completed",
                        "dataset": spec.dataset,
                        "model": spec.model,
                        "comment": spec.comment,
                        "config_path": str(spec.config_path),
                        "run_dir": str(run_dir),
                        "learning_curve": None if learning_curve is None else str(learning_curve),
                        "selection_reason": "reuse_training_only_run",
                        **summary,
                        "completed_at": datetime.now().isoformat(),
                    }
                    write_state(state, state_path)
                    write_summary_artifacts(state, summary_dir)
                    continue

                shared_policy = state.setdefault("dataset_policies", {}).get(spec.dataset)
                tuned = tune_experiment(
                    spec,
                    base_config,
                    device=device,
                    budget=budget,
                    args=args,
                    dataset_policy=shared_policy,
                )
                tuned_config_path = write_tuned_config(spec, tuned["config"], generated_config_dir)
                profile = tuned["profile"]
                if shared_policy is None:
                    state["dataset_policies"][spec.dataset] = tuned["dataset_policy"]
                    write_state(state, state_path)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
                env["PYTHONNOUSERSITE"] = env.get("PYTHONNOUSERSITE", "1")
                train_started_at = time.time()
                run_command(
                    [
                        args.python_bin,
                        "train.py",
                        "--config",
                        str(tuned_config_path),
                        "--comment",
                        spec.comment,
                        "--device",
                        "cuda",
                    ],
                    env=env,
                )

                run_dir = locate_fresh_run(tuned["config"], spec.comment, train_started_at)
                run_config_path = resolve_run_config_path(run_dir, fallback=tuned_config_path)
                input_image = resolve_input_image(spec.dataset)
                run_command(
                    [
                        args.python_bin,
                        "scripts/postprocess.py",
                        "--config",
                        str(run_config_path),
                        "--run-dir",
                        str(run_dir),
                        "--input-image",
                        input_image,
                        "--device",
                        "cuda",
                    ],
                    env=env,
                )
                learning_curve = generate_learning_curve(run_dir)
                summary = summarize_run(run_dir)

                state["experiments"][spec.experiment_id] = {
                    "status": "completed",
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "comment": spec.comment,
                    "config_path": str(spec.config_path),
                    "tuned_config_path": str(tuned_config_path),
                    "selection_reason": tuned["selection_reason"],
                    "budget": tuned["budget"],
                    "profile": profile,
                    "run_dir": str(run_dir),
                    "learning_curve": None if learning_curve is None else str(learning_curve),
                    **summary,
                    "completed_at": datetime.now().isoformat(),
                }
                write_state(state, state_path)
                write_summary_artifacts(state, summary_dir)
            except Exception as exc:
                resumable_postprocess = run_dir is not None and run_dir.is_dir() and run_has_training_artifacts(run_dir) and not is_run_complete(run_dir)
                if resumable_postprocess:
                    state["experiments"][spec.experiment_id] = {
                        "status": "needs_postprocess",
                        "dataset": spec.dataset,
                        "model": spec.model,
                        "comment": spec.comment,
                        "config_path": str(spec.config_path),
                        "run_dir": str(run_dir),
                        "tuned_config_path": None if tuned_config_path is None else str(tuned_config_path),
                        "budget": None if tuned is None else tuned["budget"],
                        "profile": profile,
                        "selection_reason": "resume_postprocess_after_failure" if tuned is None else tuned["selection_reason"],
                        "last_error": str(exc),
                        "failed_at": datetime.now().isoformat(),
                    }
                else:
                    state["experiments"][spec.experiment_id] = {
                        "status": "failed",
                        "dataset": spec.dataset,
                        "model": spec.model,
                        "comment": spec.comment,
                        "config_path": str(spec.config_path),
                        "run_dir": None if run_dir is None else str(run_dir),
                        "tuned_config_path": None if tuned_config_path is None else str(tuned_config_path),
                        "budget": None if tuned is None else tuned["budget"],
                        "profile": profile,
                        "error": str(exc),
                        "failed_at": datetime.now().isoformat(),
                    }
                write_state(state, state_path)
                write_summary_artifacts(state, summary_dir)
                raise


if __name__ == "__main__":
    main()
