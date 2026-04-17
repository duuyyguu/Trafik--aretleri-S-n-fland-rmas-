from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.modeling import ModelSpec, build_model
from src.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/latest.pt")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--image-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    model_name = args.model or ckpt.get("model", "resnet18")
    image_size = args.image_size or int(ckpt.get("image_size", 64))
    num_classes = int(ckpt["num_classes"])

    model = build_model(num_classes=num_classes, spec=ModelSpec(model_name))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    img = Image.open(Path(args.image)).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

    print(f"pred_class={idx} confidence={conf:.4f}")


if __name__ == "__main__":
    main()

