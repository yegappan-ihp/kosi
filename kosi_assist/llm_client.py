from __future__ import annotations

import base64
import io
import json
import mimetypes
import re
import threading
from pathlib import Path
from typing import Protocol

from openai import OpenAI
from PIL import Image

from kosi_assist.types import Detection, StepTarget


class GuidanceClient(Protocol):
    def get_instructions(self, crop_path: Path, issue_text: str, target_label: str) -> str:
        ...

    def choose_detection(
        self,
        image_path: Path,
        issue_text: str,
        detections: list[Detection],
    ) -> int | None:
        ...

    def extract_step_targets(self, instructions: str) -> list[StepTarget]:
        ...

    def locate_visual_target(self, image_path: Path, visual_target: str) -> Detection | None:
        ...

    def verify_bbox(
        self,
        image_path: Path,
        visual_target: str,
        instruction: str,
        detection: Detection,
    ) -> Detection | None:
        ...

    def suggest_visual_tags(self, image_path: Path, issue_text: str) -> list[str]:
        ...


def build_guidance_client(
    backend: str,
    openai_api_key: str | None,
    openai_model: str,
    openai_timeout_seconds: int,
    openai_max_image_side: int,
    openai_concurrency: int,
    local_model_name: str,
) -> GuidanceClient:
    normalized = backend.lower().strip()
    if normalized == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Set it in your environment or .env")
        return OpenAIGuidanceClient(
            api_key=openai_api_key,
            model_name=openai_model,
            timeout_seconds=openai_timeout_seconds,
            max_image_side=openai_max_image_side,
            max_concurrency=openai_concurrency,
        )
    if normalized == "local":
        return LocalGuidanceClient(model_name=local_model_name)
    raise ValueError(f"Unsupported KOSI_LLM_BACKEND: {backend}")


class OpenAIGuidanceClient:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout_seconds: int,
        max_image_side: int,
        max_concurrency: int,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_image_side = max_image_side
        self._semaphore = threading.Semaphore(max(1, max_concurrency))

    def get_instructions(self, crop_path: Path, issue_text: str, target_label: str) -> str:
        image_data_url, _, _ = _path_to_data_url(
            crop_path,
            max_side=self._max_image_side,
        )
        prompt = (
            "You are helping an elderly user repair or troubleshoot an object. "
            "Use simple words and short steps. "
            f"Object: {target_label}. "
            f"User problem: {issue_text}. "
            "Give 5 to 8 numbered steps. Include safety cautions when needed. "
            "If uncertain, clearly say what to check next instead of guessing."
        )

        response = self._responses_create(
            model=self._model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
        )
        output_text = getattr(response, "output_text", "")
        if output_text:
            return output_text.strip()

        return "I could not generate instructions from the model response."

    def choose_detection(
        self,
        image_path: Path,
        issue_text: str,
        detections: list[Detection],
    ) -> int | None:
        if not detections:
            return None

        image_data_url, _, _ = _path_to_data_url(
            image_path,
            max_side=self._max_image_side,
        )
        candidates_text = "\n".join(
            [
                (
                    f"{idx}: label={det.label}, conf={det.confidence:.2f}, "
                    f"bbox=[{det.x1},{det.y1},{det.x2},{det.y2}]"
                )
                for idx, det in enumerate(detections)
            ]
        )
        prompt = (
            "Pick the single best candidate object index for the user's issue. "
            "Return JSON only with keys: index (integer) and confidence (0-1). "
            "If none fit, set index to -1. "
            f"User issue: {issue_text}.\n"
            f"Candidates:\n{candidates_text}"
        )

        response = self._responses_create(
            model=self._model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            temperature=0,
        )

        output_text = getattr(response, "output_text", "").strip()
        if not output_text:
            return None

        try:
            data = json.loads(output_text)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            raw_index = data.get("index")
            if isinstance(raw_index, int):
                if 0 <= raw_index < len(detections):
                    return raw_index
                return None

        match = re.search(r"-?\d+", output_text)
        if not match:
            return None

        raw_index = int(match.group(0))
        if raw_index < 0 or raw_index >= len(detections):
            return None
        return raw_index

    def extract_step_targets(self, instructions: str) -> list[StepTarget]:
        prompt = (
            "Read the troubleshooting instructions and extract only visual targets that could be boxed in an image. "
            "Return JSON array only with items shaped as: "
            '{"step_number": <int>, "instruction": "<step text>", "visual_target": "<short noun phrase>"}. '
            "Include only targets that are concrete parts or objects (button, port, cable, indicator light, monitor). "
            "If a step has no visual target, skip it.\n\n"
            f"Instructions:\n{instructions}"
        )

        response = self._responses_create(
            model=self._model_name,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            temperature=0,
        )
        output_text = getattr(response, "output_text", "").strip()
        if not output_text:
            return _fallback_step_targets(instructions)

        payload = _extract_json_payload(output_text)
        if not isinstance(payload, list):
            return _fallback_step_targets(instructions)

        parsed: list[StepTarget] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            step_number = item.get("step_number")
            instruction = item.get("instruction")
            visual_target = item.get("visual_target")
            if (
                isinstance(step_number, int)
                and isinstance(instruction, str)
                and isinstance(visual_target, str)
                and visual_target.strip()
            ):
                parsed.append(
                    StepTarget(
                        step_number=step_number,
                        instruction=instruction.strip(),
                        visual_target=visual_target.strip(),
                    )
                )

        return parsed or _fallback_step_targets(instructions)

    def locate_visual_target(self, image_path: Path, visual_target: str) -> Detection | None:
        image_data_url, proc_size, scales = _path_to_data_url(
            image_path,
            max_side=self._max_image_side,
        )
        width, height = proc_size

        prompt = (
            "Locate the visual target in the image and return one tight bounding box in pixel coordinates. "
            "Return JSON only as: "
            '{"found": true|false, "x1": int, "y1": int, "x2": int, "y2": int, "confidence": 0-1}. '
            "If target is not visible, return {\"found\": false}. "
            f"Image size: width={width}, height={height}. "
            f"Visual target: {visual_target}."
        )

        response = self._responses_create(
            model=self._model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            temperature=0,
        )

        output_text = getattr(response, "output_text", "").strip()
        if not output_text:
            return None

        payload = _extract_json_payload(output_text)
        if not isinstance(payload, dict):
            return None

        found = payload.get("found")
        if found is False:
            return None

        x1 = _safe_int(payload.get("x1"))
        y1 = _safe_int(payload.get("y1"))
        x2 = _safe_int(payload.get("x2"))
        y2 = _safe_int(payload.get("y2"))
        if None in (x1, y1, x2, y2):
            return None

        x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, width, height)
        x1, y1, x2, y2 = _rescale_bbox(x1, y1, x2, y2, scales)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return None

        confidence_value = payload.get("confidence")
        confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else 0.55
        confidence = max(0.0, min(1.0, confidence))

        return Detection(
            label=visual_target,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    def verify_bbox(
        self,
        image_path: Path,
        visual_target: str,
        instruction: str,
        detection: Detection,
    ) -> Detection | None:
        image_data_url, proc_size, scales = _path_to_data_url(
            image_path,
            max_side=self._max_image_side,
        )
        width, height = proc_size

        prompt = (
            "You are validating a highlight box in an image. "
            "Check if the current bounding box points to the requested target. "
            "Return JSON only with keys: status, x1, y1, x2, y2, confidence. "
            "status must be one of exact, nearby, wrong. "
            "If exact, you may keep same box or provide tighter corrected box. "
            "If nearby, provide corrected box. If wrong and not visible, set status wrong. "
            f"Image size: width={width}, height={height}. "
            f"Step: {instruction}. "
            f"Target: {visual_target}. "
            f"Current box: x1={detection.x1}, y1={detection.y1}, x2={detection.x2}, y2={detection.y2}."
        )

        response = self._responses_create(
            model=self._model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            temperature=0,
        )

        output_text = getattr(response, "output_text", "").strip()
        payload = _extract_json_payload(output_text)
        if not isinstance(payload, dict):
            return None

        status = str(payload.get("status", "")).strip().lower()
        if status == "wrong":
            return None

        x1 = _safe_int(payload.get("x1"))
        y1 = _safe_int(payload.get("y1"))
        x2 = _safe_int(payload.get("x2"))
        y2 = _safe_int(payload.get("y2"))

        if None in (x1, y1, x2, y2):
            if status == "exact":
                return detection
            return None

        x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, width, height)
        x1, y1, x2, y2 = _rescale_bbox(x1, y1, x2, y2, scales)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return None

        confidence_value = payload.get("confidence")
        confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else detection.confidence
        confidence = max(0.0, min(1.0, confidence))

        return Detection(
            label=visual_target,
            confidence=confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    def suggest_visual_tags(self, image_path: Path, issue_text: str) -> list[str]:
        image_data_url, _, _ = _path_to_data_url(
            image_path,
            max_side=self._max_image_side,
        )
        prompt = (
            "List visible objects and parts in this image for detection. "
            "Return JSON array of 8 to 20 short tags in lowercase English only. "
            "Prefer concrete nouns like laptop, charger, cable, keyboard, button, usb port. "
            f"Issue context: {issue_text}."
        )
        response = self._responses_create(
            model=self._model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
            temperature=0,
        )
        output_text = getattr(response, "output_text", "").strip()
        payload = _extract_json_payload(output_text)
        if not isinstance(payload, list):
            return []
        tags: list[str] = []
        for item in payload:
            if isinstance(item, str):
                normalized = item.strip().lower()
                if normalized:
                    tags.append(normalized)
        return _dedupe_tags(tags)

    def _responses_create(self, **kwargs):
        with self._semaphore:
            return self._client.responses.create(
                timeout=self._timeout_seconds,
                **kwargs,
            )


class LocalGuidanceClient:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def get_instructions(self, crop_path: Path, issue_text: str, target_label: str) -> str:
        raise NotImplementedError(
            "Local backend scaffolding is ready, but local inference is not implemented in this starter. "
            f"Planned model target: {self._model_name}."
        )

    def choose_detection(
        self,
        image_path: Path,
        issue_text: str,
        detections: list[Detection],
    ) -> int | None:
        return None

    def extract_step_targets(self, instructions: str) -> list[StepTarget]:
        return _fallback_step_targets(instructions)

    def locate_visual_target(self, image_path: Path, visual_target: str) -> Detection | None:
        return None

    def verify_bbox(
        self,
        image_path: Path,
        visual_target: str,
        instruction: str,
        detection: Detection,
    ) -> Detection | None:
        return detection

    def suggest_visual_tags(self, image_path: Path, issue_text: str) -> list[str]:
        return []


def _path_to_data_url(
    image_path: Path,
    max_side: int,
) -> tuple[str, tuple[int, int], tuple[float, float]]:
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        original_w, original_h = image.size

        if max(original_w, original_h) > max_side:
            scale = max_side / float(max(original_w, original_h))
            resized_w = max(1, int(round(original_w * scale)))
            resized_h = max(1, int(round(original_h * scale)))
            image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        else:
            resized_w, resized_h = original_w, original_h

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        payload = buffer.getvalue()

    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(payload).decode("utf-8")
    scale_x = original_w / float(resized_w)
    scale_y = original_h / float(resized_h)
    return (
        f"data:{mime_type};base64,{encoded}",
        (resized_w, resized_h),
        (scale_x, scale_y),
    )


def _extract_json_payload(text: str):
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidate = fenced_match.group(1).strip() if fenced_match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _fallback_step_targets(instructions: str) -> list[StepTarget]:
    keywords = [
        "power button",
        "button",
        "light",
        "indicator",
        "charging port",
        "usb port",
        "port",
        "cable",
        "charger",
        "monitor",
        "screen",
        "keyboard",
        "mouse",
        "switch",
    ]
    candidates: list[StepTarget] = []
    for line in instructions.splitlines():
        stripped = line.strip()
        match = re.match(r"^(\d+)[\).:-]\s*(.+)$", stripped)
        if not match:
            continue
        step_number = int(match.group(1))
        step_text = match.group(2).strip()
        lower_step = step_text.lower()
        for word in keywords:
            if word in lower_step:
                candidates.append(
                    StepTarget(
                        step_number=step_number,
                        instruction=step_text,
                        visual_target=word,
                    )
                )
                break
    return candidates


def _safe_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = min(width, max(x1, x2))
    bottom = min(height, max(y1, y2))
    return left, top, right, bottom


def _rescale_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    scales: tuple[float, float],
) -> tuple[int, int, int, int]:
    scale_x, scale_y = scales
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out
