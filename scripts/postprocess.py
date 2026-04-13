from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from PIL import Image, ImageDraw
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_data_loaders
from models import build_model
from models.wrappers import IMAGENET_MEAN, IMAGENET_STD
from utils.pipeline import build_metric_report, run_inference
from utils.runtime import load_checkpoint, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation, efficiency analysis, and ERF generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing best.ckpt.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--input-image", type=str, default=None, help="Representative sky image for ERF visualization.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the FPS benchmark.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for FPS.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations for FPS.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip metric evaluation.")
    parser.add_argument("--skip-efficiency", action="store_true", help="Skip params/FLOPs/FPS analysis.")
    parser.add_argument("--skip-erf", action="store_true", help="Skip ERF visualization.")
    parser.add_argument("--metrics-output", type=str, default=None, help="Optional metrics JSON path.")
    parser.add_argument("--efficiency-output", type=str, default=None, help="Optional efficiency JSON path.")
    parser.add_argument("--erf-output", type=str, default=None, help="Optional ERF image output path.")
    return parser.parse_args()


def resolve_run_dir_and_checkpoint(args: argparse.Namespace) -> tuple[Path | None, Path]:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
    elif run_dir is not None:
        checkpoint = run_dir / "best.ckpt"
    else:
        raise ValueError("Provide either --run-dir or --checkpoint.")

    if run_dir is None:
        run_dir = checkpoint.parent
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return run_dir, checkpoint


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_image(path: str | Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    mask_radius = max(1, int(round(image_size * 250 / 512)))
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    mask = Image.new("L", (image_size, image_size), 0)
    center = (image_size // 2, image_size // 2)
    ImageDraw.Draw(mask).ellipse(
        (center[0] - mask_radius, center[1] - mask_radius, center[0] + mask_radius, center[1] + mask_radius),
        fill=255,
    )
    image_np = np.array(image, copy=True)
    image_np[np.asarray(mask) == 0] = 0
    image = Image.fromarray(image_np)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(image)
    return tensor.unsqueeze(0), image_np


def save_erf_overlay(model: torch.nn.Module, config: dict, input_image: str | Path, output_path: Path, device: torch.device) -> None:
    image_size = int(config["data"].get("image_size", 224))
    image_tensor, image_np = load_image(input_image, image_size)
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)

    hook_cache: dict[str, torch.Tensor] = {}
    final_stage = model.visual_encoder.final_stage_module

    def forward_hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        hook_cache["activation"] = output

    def backward_hook(_: torch.nn.Module, __, grad_output) -> None:
        hook_cache["gradient"] = grad_output[0]

    forward_handle = final_stage.register_forward_hook(forward_hook)
    backward_handle = final_stage.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    _ = model.visual_encoder(image_tensor)
    activation = hook_cache["activation"]
    center_y = activation.shape[-2] // 2
    center_x = activation.shape[-1] // 2
    target = activation[:, :, center_y, center_x].mean()
    target.backward()

    input_grad = image_tensor.grad.detach().abs().amax(dim=1)[0].cpu().numpy()
    input_grad = input_grad / (input_grad.max() + 1e-8)
    heatmap = plt.get_cmap("jet")(input_grad)[..., :3]
    overlay = np.clip((0.45 * image_np) + (0.55 * heatmap), 0.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    forward_handle.remove()
    backward_handle.remove()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir, checkpoint = resolve_run_dir_and_checkpoint(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(config).to(device).eval()
    loaded = load_checkpoint(model, checkpoint, map_location=device)
    print(f"[checkpoint] missing={len(loaded['missing_keys'])} unexpected={len(loaded['unexpected_keys'])}")

    if not args.skip_eval:
        loaders = get_data_loaders(config)
        dataloader = loaders[1] if args.split == "val" else loaders[2]
        outputs = run_inference(
            model,
            dataloader,
            device,
            use_aux_decoder=bool(config["model"].get("use_aux_decoder", False)),
            desc=f"Evaluating {args.split}",
        )
        metrics_report = build_metric_report(outputs, horizons=list(config["model"]["horizons"]))
        metrics_output = Path(args.metrics_output) if args.metrics_output else run_dir / f"{args.split}_metrics.json"
        save_json(metrics_report, metrics_output)
        print(f"[done] metrics written to {metrics_output}")

    if not args.skip_efficiency:
        image_size = int(config["data"].get("image_size", 224))
        sequence_length = int(config["data"]["sequence_length"])
        temporal_channels = int(config["model"].get("temporal_channels", 7))
        images = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        weather = torch.randn(args.batch_size, sequence_length, temporal_channels, device=device)

        params = int(parameter_count(model)[""])
        flops_giga = None
        try:
            flops = FlopCountAnalysis(model, (images, weather))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            flops_giga = float(flops.total() / 1e9)
        except Exception as exc:
            print(f"[warn] FLOP analysis failed: {exc}")

        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model(images, weather)
            synchronize_if_needed(device)
            start = time.perf_counter()
            for _ in range(args.iters):
                _ = model(images, weather)
            synchronize_if_needed(device)
            elapsed = time.perf_counter() - start

        fps = float((args.batch_size * args.iters) / max(elapsed, 1e-8))
        efficiency_report = {
            "batch_size": args.batch_size,
            "image_size": image_size,
            "params": params,
            "params_millions": params / 1e6,
            "flops_giga": flops_giga,
            "fps": fps,
            "device": str(device),
        }
        efficiency_output = Path(args.efficiency_output) if args.efficiency_output else run_dir / "efficiency_report.json"
        save_json(efficiency_report, efficiency_output)
        print(f"[done] efficiency report written to {efficiency_output}")

    if not args.skip_erf:
        if not args.input_image:
            raise ValueError("--input-image is required unless --skip-erf is set.")
        erf_output = Path(args.erf_output) if args.erf_output else run_dir / "erf_overlay.png"
        save_erf_overlay(model, config, args.input_image, erf_output, device)
        print(f"[done] ERF overlay written to {erf_output}")


if __name__ == "__main__":
    main()
