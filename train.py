from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
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


def get_batch_size(model_name: str) -> int:
    normalized = model_name.lower()
    if any(token in normalized for token in ("large2", "large", "_l2", "_l")):
        return 8
    if any(token in normalized for token in ("base", "_b")):
        return 16
    if any(token in normalized for token in ("small", "_s")):
        return 32
    return 64


def select_primary_horizon(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return targets.unsqueeze(1)
    return targets[:, :1]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["training"].get("seed", 42)))

    scale = str(config["visual"].get("scale", config["visual"].get("name", "tiny")))
    effective_batch_size = get_batch_size(scale)
    config["data"]["batch_size"] = effective_batch_size
    config["model"]["horizons"] = [int(list(config["model"].get("horizons", [10]))[0])]

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = create_run_dir(config, args.output_root, comment=args.comment)
    jsonable_config = config_to_jsonable(config)
    if args.comment:
        jsonable_config["_run"] = {"comment": args.comment}
    save_json(jsonable_config, run_dir / "config.json")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"[run_dir] {run_dir}")
    print(f"[batch_size] using dynamic batch size {effective_batch_size}")

    train_loader, val_loader, test_loader = get_data_loaders(config)
    model = build_model(config).to(device)
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
    use_aux_decoder = False
    history = []
    best_rmse = float("inf")
    patience_counter = 0
    start_epoch = 0

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

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]")
        for step, batch in enumerate(progress, start=1):
            batch = tuple(item.to(device, non_blocking=True) if hasattr(item, "to") else item for item in batch)
            images, weather_seq, targets = batch[:3]
            targets = select_primary_horizon(targets)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions, aux_prediction = model(images, weather_seq)
                loss = criterion(predictions, targets)
                loss = loss / grad_accum_steps

            if not torch.isfinite(loss):
                print(f"[warn] skipping non-finite loss at epoch={epoch + 1} step={step}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if (step % grad_accum_steps == 0) or (step == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["training"].get("grad_clip", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item() * grad_accum_steps)
            progress.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        scheduler.step()
        train_loss = epoch_loss / max(len(train_loader), 1)
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
            }
        )
        save_json({"history": history}, run_dir / "history.json")
        print(
            f"[epoch {epoch + 1:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_rmse={val_rmse:.4f}"
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

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            checkpoint_payload["best_val_rmse"] = best_rmse
            torch.save(checkpoint_payload, run_dir / "best.ckpt")
            save_json(val_report, run_dir / "best_val_metrics.json")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early-stop] no RMSE improvement for {patience} epochs")
                break

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
    print(f"[done] best checkpoint: {run_dir / 'best.ckpt'}")
    print(f"[done] test metrics: {run_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
