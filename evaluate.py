from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import get_data_loaders
from models import build_model
from utils.pipeline import build_metric_report, run_inference
from utils.runtime import load_checkpoint, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders = get_data_loaders(config)
    dataloader = loaders[1] if args.split == "val" else loaders[2]
    model = build_model(config).to(device)
    loaded = load_checkpoint(model, args.checkpoint, map_location=device)
    print(f"[checkpoint] missing={len(loaded['missing_keys'])} unexpected={len(loaded['unexpected_keys'])}")
    outputs = run_inference(
        model,
        dataloader,
        device,
        use_aux_decoder=bool(config["model"].get("use_aux_decoder", False)),
        desc=f"Evaluating {args.split}",
    )
    report = build_metric_report(outputs, horizons=list(config["model"]["horizons"]))
    output_path = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / f"{args.split}_metrics.json"
    save_json(report, output_path)
    print(f"[done] metrics written to {output_path}")


if __name__ == "__main__":
    main()

