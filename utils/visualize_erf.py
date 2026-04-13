from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from models.wrappers import IMAGENET_MEAN, IMAGENET_STD
from utils.runtime import load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Effective Receptive Field overlay.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    parser.add_argument("--input-image", type=str, required=True, help="Sky image used for ERF analysis.")
    parser.add_argument("--output", type=str, required=True, help="Output heatmap path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device override.")
    return parser.parse_args()


def load_image(path: str | Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(image)
    return tensor.unsqueeze(0), image_np


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location=device)

    image_size = int(config["data"].get("image_size", 512))
    image_tensor, image_np = load_image(args.input_image, image_size)
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
    cmap = plt.get_cmap("jet")
    heatmap = cmap(input_grad)[..., :3]
    overlay = np.clip((0.45 * image_np) + (0.55 * heatmap), 0.0, 1.0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    forward_handle.remove()
    backward_handle.remove()
    print(f"[done] ERF overlay written to {output_path}")


if __name__ == "__main__":
    main()
