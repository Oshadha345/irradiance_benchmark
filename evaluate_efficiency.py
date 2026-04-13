from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from utils.runtime import load_checkpoint, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure params, FLOPs, and FPS.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional trained checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    parser.add_argument("--batch-size", type=int, default=4, help="Benchmark batch size.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for FPS.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations for FPS.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location=device)

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
    report = {
        "batch_size": args.batch_size,
        "image_size": image_size,
        "params": params,
        "params_millions": params / 1e6,
        "flops_giga": flops_giga,
        "fps": fps,
        "device": str(device),
    }
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "efficiency_report.json"
    save_json(report, output_path)
    print(report)
    print(f"[done] efficiency report written to {output_path}")


if __name__ == "__main__":
    main()
