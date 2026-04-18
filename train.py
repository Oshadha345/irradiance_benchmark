from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_data_loaders
from models import build_model
from utils.pipeline import build_metric_report, run_inference
from utils.runtime import config_to_jsonable, create_run_dir, load_checkpoint, load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the irradiance benchmark model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--comment", type=str, default=None, help="Optional tag appended to the run directory name.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional override for the results root.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if configured.")
    parser.add_argument("--auto-batch-size", dest="auto_batch_size", action="store_true", help="Probe the active GPU and pick the fastest safe batch size.")
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false", help="Disable batch-size probing and use the configured batch size directly.")
    parser.set_defaults(auto_batch_size=None)
    return parser.parse_args()


def split_parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    backbone_params, task_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "visual_encoder" in name:
            backbone_params.append(parameter)
        else:
            task_params.append(parameter)
    return backbone_params, task_params


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    backbone_ratio = float(train_cfg.get("backbone_lr_ratio", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    backbone_params, task_params = split_parameter_groups(model)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * backbone_ratio},
            {"params": task_params, "lr": base_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    train_cfg = config["training"]
    epochs = int(train_cfg.get("epochs", 50))
    min_lr = float(train_cfg.get("min_learning_rate", 1e-6))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception as exc:
            print(f"[warn] torch.compile failed, continuing without it: {exc}")
    return model


def canonicalize_device(device: torch.device) -> torch.device:
    if device.type != "cuda" or device.index is not None:
        return device
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def synchronize_if_needed(device: torch.device) -> None:
    device = canonicalize_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def snapshot_module_training_states(model: nn.Module) -> dict[int, bool]:
    return {id(module): module.training for module in model.modules()}


def restore_module_training_states(model: nn.Module, states: dict[int, bool]) -> None:
    for module in model.modules():
        module.train(states.get(id(module), True))


def enable_fast_runtime_paths(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def default_batch_probe_max(config: Dict[str, Any]) -> int:
    family = str(config["visual"].get("family", "timm")).lower()
    scale = str(config["visual"].get("scale", config["visual"].get("name", "tiny"))).lower()
    if any(token in scale for token in ("large2", "large", "_l2", "_l")):
        return 8 if family == "timm" else 4
    if any(token in scale for token in ("base", "_b")):
        return 16 if family != "timm" else 32
    if any(token in scale for token in ("small", "_s")):
        return 32 if family != "timm" else 64
    return 32 if family != "timm" else 64


def _next_power_of_two(value: int) -> int:
    probe = 1
    while probe < max(1, value):
        probe *= 2
    return probe


def memory_aware_batch_probe_max(device: torch.device, config: Dict[str, Any]) -> int:
    device = canonicalize_device(device)
    family = str(config["visual"].get("family", "timm")).lower()
    scale = str(config["visual"].get("scale", config["visual"].get("name", "tiny"))).lower()
    base_max = default_batch_probe_max(config)
    if device.type != "cuda":
        return base_max

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        return base_max

    free_gib = free_bytes / float(1024**3)
    image_size = int(config["data"].get("image_size", 224))

    multiplier = 1
    if free_gib >= 24:
        multiplier = 4
    elif free_gib >= 16:
        multiplier = 2

    if image_size >= 384:
        multiplier = max(1, multiplier // 2)

    if family == "timm":
        absolute_cap = 256
        if any(token in scale for token in ("base", "_b")):
            absolute_cap = 128
        if any(token in scale for token in ("large2", "large", "_l2", "_l")):
            absolute_cap = 32
    else:
        absolute_cap = 128
        if any(token in scale for token in ("base", "_b")):
            absolute_cap = 64
        if any(token in scale for token in ("large2", "large", "_l2", "_l")):
            absolute_cap = 16

    target = min(absolute_cap, max(base_max, _next_power_of_two(base_max * multiplier)))
    print(
        f"[autotune] free_memory_gib={free_gib:.2f} "
        f"base_probe_max={base_max} adjusted_probe_max={target}"
    )
    return target


def build_batch_size_candidates(config: Dict[str, Any], requested_batch_size: int, *, device: torch.device) -> list[int]:
    train_cfg = config["training"]
    explicit_candidates = train_cfg.get("batch_size_candidates")
    if explicit_candidates:
        candidates = [int(value) for value in explicit_candidates]
    else:
        min_candidate = max(1, int(train_cfg.get("batch_size_probe_min", 1)))
        max_candidate = max(
            min_candidate,
            int(train_cfg.get("batch_size_probe_max", memory_aware_batch_probe_max(device, config))),
        )
        candidates = []
        probe = 1
        while probe < max_candidate:
            probe *= 2
        while probe >= min_candidate:
            candidates.append(probe)
            probe //= 2

    candidates.append(int(requested_batch_size))
    candidates.extend([2, 1])
    normalized = sorted({value for value in candidates if value >= 1}, reverse=True)
    return normalized


def required_batch_probe_headroom_bytes(device: torch.device, config: Dict[str, Any]) -> int:
    device = canonicalize_device(device)
    if device.type != "cuda":
        return 0

    train_cfg = config["training"]
    absolute_gib = float(train_cfg.get("batch_size_probe_headroom_gib", 0.5))
    relative_ratio = float(train_cfg.get("batch_size_probe_headroom_ratio", 0.03))
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    return max(int(absolute_gib * (1024**3)), int(total_bytes * relative_ratio))


def autotune_batch_size(
    model: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    requested_batch_size: int,
) -> int:
    device = canonicalize_device(device)
    if device.type != "cuda":
        return requested_batch_size

    train_cfg = config["training"]
    probe_warmup = max(1, int(train_cfg.get("batch_size_probe_warmup", 1)))
    probe_iters = max(1, int(train_cfg.get("batch_size_probe_iters", 2)))
    amp_enabled = bool(train_cfg.get("use_amp", True))
    image_size = int(config["data"].get("image_size", 224))
    sequence_length = int(config["data"]["sequence_length"])
    temporal_channels = int(config["model"].get("temporal_channels", 7))
    delta = float(train_cfg.get("huber_delta", 1.0))
    candidates = build_batch_size_candidates(config, requested_batch_size, device=device)
    required_headroom_bytes = required_batch_probe_headroom_bytes(device, config)

    best_batch_size = requested_batch_size
    best_samples_per_second = -1.0
    found_valid_candidate = False
    original_states = snapshot_module_training_states(model)
    model.train()

    try:
        for batch_size in candidates:
            try:
                torch.cuda.empty_cache()
                synchronize_if_needed(device)
                torch.cuda.reset_peak_memory_stats(device)
                images = torch.randn(batch_size, 3, image_size, image_size, device=device)
                weather = torch.randn(batch_size, sequence_length, temporal_channels, device=device)
                targets = torch.randn(batch_size, 1, device=device)

                for _ in range(probe_warmup):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        predictions, _ = model(images, weather)
                        loss = F.huber_loss(predictions, targets, delta=delta)
                    loss.backward()

                synchronize_if_needed(device)
                start = time.perf_counter()
                for _ in range(probe_iters):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        predictions, _ = model(images, weather)
                        loss = F.huber_loss(predictions, targets, delta=delta)
                    loss.backward()
                synchronize_if_needed(device)

                elapsed = max(time.perf_counter() - start, 1e-8)
                samples_per_second = float((batch_size * probe_iters) / elapsed)
                free_bytes, _ = torch.cuda.mem_get_info(device)
                peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
                free_headroom_gib = free_bytes / float(1024**3)
                peak_reserved_gib = peak_reserved_bytes / float(1024**3)
                if required_headroom_bytes and free_bytes < required_headroom_bytes:
                    required_headroom_gib = required_headroom_bytes / float(1024**3)
                    print(
                        f"[autotune] batch_size={batch_size} rejected: "
                        f"free_headroom_gib={free_headroom_gib:.2f} < required={required_headroom_gib:.2f}"
                    )
                    continue
                print(
                    f"[autotune] batch_size={batch_size} throughput={samples_per_second:.2f} samples/s "
                    f"free_headroom_gib={free_headroom_gib:.2f} peak_reserved_gib={peak_reserved_gib:.2f}"
                )
                found_valid_candidate = True
                if (samples_per_second > best_samples_per_second * 1.01) or (
                    abs(samples_per_second - best_samples_per_second) <= 1e-6 and batch_size > best_batch_size
                ):
                    best_batch_size = batch_size
                    best_samples_per_second = samples_per_second
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                print(f"[autotune] batch_size={batch_size} failed with CUDA OOM")
            finally:
                model.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        restore_module_training_states(model, original_states)

    if not found_valid_candidate:
        raise RuntimeError("Auto batch-size tuning failed: every probe candidate exhausted GPU memory.")

    print(f"[autotune] selected batch_size={best_batch_size}")
    return best_batch_size


def select_primary_horizon(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return targets.unsqueeze(1)
    return targets[:, :1]


def run_training_step(
    *,
    model: nn.Module,
    batch: tuple[Any, ...],
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_accum_steps: int,
    should_step: bool,
    grad_clip: float,
    epoch_index: int,
    step_index: int,
) -> float | None:
    device = canonicalize_device(device)
    batch_size = int(batch[0].shape[0]) if batch and hasattr(batch[0], "shape") else -1
    try:
        images, weather_seq, targets = batch[:3]
        targets = select_primary_horizon(targets)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predictions, _ = model(images, weather_seq)
            loss = criterion(predictions, targets)
            loss = loss / grad_accum_steps

        if not torch.isfinite(loss).all():
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return None

        scaler.scale(loss).backward()
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        return float(loss.detach().item() * grad_accum_steps)
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA OOM at epoch={epoch_index} step={step_index} batch_size={batch_size}"
        ) from exc


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["training"].get("seed", 42)))

    requested_batch_size = int(config["data"].get("batch_size", 16))
    config["model"]["horizons"] = [int(list(config["model"].get("horizons", [10]))[0])]

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = canonicalize_device(torch.device(device_name))
    enable_fast_runtime_paths(device)

    model = build_model(config).to(device)
    auto_batch_enabled = args.auto_batch_size
    if auto_batch_enabled is None:
        auto_batch_enabled = bool(config["training"].get("auto_batch_size", device.type == "cuda"))
    effective_batch_size = requested_batch_size
    if auto_batch_enabled:
        effective_batch_size = autotune_batch_size(model, device, config, requested_batch_size)
    config["data"]["batch_size"] = effective_batch_size

    run_dir = create_run_dir(config, args.output_root, comment=args.comment)
    jsonable_config = config_to_jsonable(config)
    if args.comment:
        jsonable_config["_run"] = {"comment": args.comment}
    save_json(jsonable_config, run_dir / "config.json")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"[run_dir] {run_dir}")
    print(
        f"[batch_size] using batch size {effective_batch_size} "
        f"(requested={requested_batch_size}, auto_batch_size={auto_batch_enabled})"
    )

    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(
        f"[data] train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"test_samples={len(test_loader.dataset)} train_batches={len(train_loader)}"
    )
    compile_enabled = bool(config["training"].get("compile", False)) and not args.no_compile
    model = maybe_compile(model, compile_enabled)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = nn.HuberLoss(delta=float(config["training"].get("huber_delta", 1.0)))
    amp_enabled = device.type == "cuda" and bool(config["training"].get("use_amp", True))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    epochs = int(config["training"].get("epochs", 50))
    grad_accum_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    patience = int(config["training"].get("early_stop_patience", 15))
    min_epochs = int(config["training"].get("min_epochs", 1))
    early_stop_min_delta = float(config["training"].get("early_stop_min_delta", 0.0))
    grad_clip = float(config["training"].get("grad_clip", 5.0))
    max_total_train_seconds = config["training"].get("max_total_train_seconds")
    max_epoch_seconds = config["training"].get("max_epoch_seconds")
    max_total_train_seconds = None if max_total_train_seconds is None else float(max_total_train_seconds)
    max_epoch_seconds = None if max_epoch_seconds is None else float(max_epoch_seconds)
    use_aux_decoder = False
    history = []
    best_rmse = float("inf")
    patience_counter = 0
    start_epoch = 0
    stop_reason = "completed"
    training_start_time = time.perf_counter()

    if args.resume:
        loaded = load_checkpoint(model, args.resume, map_location=device)
        print(f"[resume] missing={len(loaded['missing_keys'])} unexpected={len(loaded['unexpected_keys'])}")
        checkpoint = loaded["checkpoint"]
        if isinstance(checkpoint, dict):
            optimizer_state = checkpoint.get("optimizer_state_dict")
            scheduler_state = checkpoint.get("scheduler_state_dict")
            scaler_state = checkpoint.get("scaler_state_dict")
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            if scaler_state and scaler.is_enabled():
                scaler.load_state_dict(scaler_state)
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_rmse = float(checkpoint.get("best_val_rmse", best_rmse))

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_start_time = time.perf_counter()

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]")
        for step, batch in enumerate(progress, start=1):
            batch = tuple(item.to(device, non_blocking=True) if hasattr(item, "to") else item for item in batch)
            loss_value = run_training_step(
                model=model,
                batch=batch,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_accum_steps=grad_accum_steps,
                should_step=(step % grad_accum_steps == 0) or (step == len(train_loader)),
                grad_clip=grad_clip,
                epoch_index=epoch + 1,
                step_index=step,
            )

            if loss_value is None:
                print(f"[warn] skipping non-finite loss at epoch={epoch + 1} step={step}")
                progress.set_postfix({"loss": "non-finite"})
                continue

            epoch_loss += loss_value
            progress.set_postfix({"loss": f"{loss_value:.4f}"})

        scheduler.step()
        train_loss = epoch_loss / max(len(train_loader), 1)
        epoch_seconds = max(time.perf_counter() - epoch_start_time, 1e-8)
        steps_per_second = float(len(train_loader) / epoch_seconds)
        samples_per_second = float((len(train_loader.dataset)) / epoch_seconds)
        val_outputs = run_inference(
            model,
            val_loader,
            device,
            use_aux_decoder=use_aux_decoder,
            criterion=criterion,
            desc="Validation",
        )
        val_report = build_metric_report(val_outputs, horizons=list(config["model"]["horizons"]))
        val_rmse = float(val_report["ghi"]["overall"]["RMSE"])
        val_loss = float(val_outputs["loss"] or 0.0)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "learning_rate": float(optimizer.param_groups[-1]["lr"]),
                "epoch_seconds": epoch_seconds,
                "steps_per_second": steps_per_second,
                "samples_per_second": samples_per_second,
            }
        )
        save_json({"history": history}, run_dir / "history.json")
        print(
            f"[epoch {epoch + 1:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_rmse={val_rmse:.4f} "
            f"epoch_seconds={epoch_seconds:.2f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "best_val_rmse": best_rmse,
            "config": config,
        }
        torch.save(checkpoint_payload, run_dir / "last.ckpt")

        improvement = best_rmse - val_rmse
        if improvement > early_stop_min_delta:
            best_rmse = val_rmse
            patience_counter = 0
            checkpoint_payload["best_val_rmse"] = best_rmse
            torch.save(checkpoint_payload, run_dir / "best.ckpt")
            save_json(val_report, run_dir / "best_val_metrics.json")
        else:
            patience_counter += 1
            if (epoch + 1) >= min_epochs and patience_counter >= patience:
                stop_reason = f"early_stop_no_improvement_{patience}"
                print(f"[early-stop] no RMSE improvement > {early_stop_min_delta} for {patience} epochs")
                break

        elapsed_train_seconds = time.perf_counter() - training_start_time
        if max_epoch_seconds is not None and (epoch + 1) >= min_epochs and epoch_seconds > max_epoch_seconds:
            stop_reason = f"epoch_time_exceeded_{max_epoch_seconds:.1f}s"
            print(f"[budget-stop] epoch time {epoch_seconds:.2f}s exceeded limit {max_epoch_seconds:.2f}s")
            break
        if max_total_train_seconds is not None and (epoch + 1) >= min_epochs and elapsed_train_seconds > max_total_train_seconds:
            stop_reason = f"train_time_exceeded_{max_total_train_seconds:.1f}s"
            print(f"[budget-stop] train time {elapsed_train_seconds:.2f}s exceeded limit {max_total_train_seconds:.2f}s")
            break

    if not (run_dir / "best.ckpt").is_file():
        stop_reason = "no_best_checkpoint"
        raise RuntimeError("Training finished without producing a best checkpoint.")

    load_checkpoint(model, run_dir / "best.ckpt", map_location=device)
    test_outputs = run_inference(
        model,
        test_loader,
        device,
        use_aux_decoder=use_aux_decoder,
        criterion=criterion,
        desc="Testing",
    )
    test_report = build_metric_report(test_outputs, horizons=list(config["model"]["horizons"]))
    save_json(test_report, run_dir / "test_metrics.json")
    save_json(
        {
            "stop_reason": stop_reason,
            "epochs_completed": len(history),
            "best_val_rmse": best_rmse,
            "requested_epochs": epochs,
            "train_seconds": time.perf_counter() - training_start_time,
        },
        run_dir / "status.json",
    )
    print(f"[done] best checkpoint: {run_dir / 'best.ckpt'}")
    print(f"[done] test metrics: {run_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
