from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from kosi_assist.llm_client import GuidanceClient

try:
    import ram
    from ram.models import ram as ram_model_builder
except Exception:
    ram = None
    ram_model_builder = None


_RAM_CACHE: dict[str, object] = {}


class RecognizeEverything:
    def __init__(
        self,
        checkpoint_path: str | None,
        max_tags: int,
    ) -> None:
        self._checkpoint_path = checkpoint_path.strip() if checkpoint_path else ""
        self._max_tags = max(8, max_tags)

    def collect_tags(
        self,
        image_path: Path,
        issue_text: str,
        guidance_client: GuidanceClient,
    ) -> list[str]:
        tags: list[str] = []

        ram_tags = self._ram_tags(image_path)
        if ram_tags:
            tags.extend(ram_tags)

        gpt_tags = guidance_client.suggest_visual_tags(image_path=image_path, issue_text=issue_text)
        if gpt_tags:
            tags.extend(gpt_tags)

        tags.extend(
            [
                "laptop",
                "computer",
                "charger",
                "power cable",
                "usb cable",
                "keyboard",
                "trackpad",
                "screen",
                "power button",
                "indicator light",
                "usb port",
                "battery",
                "adapter",
                "wall socket",
            ]
        )

        unique = _dedupe(tags)
        return unique[: self._max_tags]

    def _ram_tags(self, image_path: Path) -> list[str]:
        if not self._checkpoint_path:
            return []
        checkpoint = Path(self._checkpoint_path)
        if not checkpoint.exists() or ram is None or ram_model_builder is None:
            return []

        model = _get_ram_model(str(checkpoint))
        if model is None:
            return []

        transform = ram.get_transform(image_size=384)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            tensor = transform(rgb).unsqueeze(0)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = tensor.to(device)
        model = model.to(device)

        try:
            tag_text, _ = ram.inference_ram(tensor, model)
        except Exception:
            return []

        if not isinstance(tag_text, str):
            return []

        tags = [p.strip().lower() for p in tag_text.split("|") if p.strip()]
        return _dedupe(tags)


def _get_ram_model(checkpoint_path: str):
    cached = _RAM_CACHE.get(checkpoint_path)
    if cached is not None:
        return cached

    try:
        model = ram_model_builder(
            pretrained=checkpoint_path,
            image_size=384,
            vit="swin_l",
        )
        model.eval()
    except Exception:
        _RAM_CACHE[checkpoint_path] = None
        return None

    _RAM_CACHE[checkpoint_path] = model
    return model


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out
