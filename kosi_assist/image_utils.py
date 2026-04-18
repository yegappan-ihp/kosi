from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from kosi_assist.types import Detection


def crop_detection(
    image_path: Path,
    detection: Detection,
    output_path: Path,
    padding_ratio: float,
) -> Path:
    with Image.open(image_path) as img:
        width, height = img.size
        box_width = detection.x2 - detection.x1
        box_height = detection.y2 - detection.y1
        min_margin = max(20, int(min(width, height) * 0.03))
        pad_x = max(min_margin, int(box_width * max(0.0, padding_ratio)))
        pad_y = max(min_margin, int(box_height * max(0.0, padding_ratio)))

        x1 = max(0, detection.x1 - pad_x)
        y1 = max(0, detection.y1 - pad_y)
        x2 = min(width, detection.x2 + pad_x)
        y2 = min(height, detection.y2 + pad_y)

        min_crop_w = max(220, int(width * 0.18))
        min_crop_h = max(220, int(height * 0.18))

        x1, x2 = _expand_to_min_size(x1, x2, width, min_crop_w)
        y1, y2 = _expand_to_min_size(y1, y2, height, min_crop_h)

        cropped = img.crop((x1, y1, x2, y2))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
    return output_path


def draw_detection_box(
    image_path: Path,
    detection: Detection,
    output_path: Path,
    label_text: str,
) -> Path:
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        stroke = max(3, int(min(canvas.size) * 0.005))

        draw.rectangle(
            [(detection.x1, detection.y1), (detection.x2, detection.y2)],
            outline=(255, 60, 60),
            width=stroke,
        )

        text_x = detection.x1 + 6
        text_y = max(2, detection.y1 - 24)
        draw.rectangle(
            [(text_x - 4, text_y - 2), (text_x + 260, text_y + 20)],
            fill=(255, 60, 60),
        )
        draw.text((text_x, text_y), label_text[:34], fill=(255, 255, 255))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    return output_path


def _expand_to_min_size(start: int, end: int, limit: int, minimum: int) -> tuple[int, int]:
    current = end - start
    if current >= minimum:
        return start, end

    deficit = minimum - current
    add_left = deficit // 2
    add_right = deficit - add_left

    start = max(0, start - add_left)
    end = min(limit, end + add_right)

    current = end - start
    if current >= minimum:
        return start, end

    if start == 0:
        end = min(limit, minimum)
    elif end == limit:
        start = max(0, limit - minimum)

    return start, end
