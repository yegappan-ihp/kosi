from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from kosi_assist.matcher import build_query_terms

try:
    from ultralytics import YOLOWorld
except ImportError:
    YOLOWorld = None

from kosi_assist.types import Detection


class YoloDetector:
    def __init__(self, model_name: str, confidence_threshold: float) -> None:
        self._model_name = model_name
        self._is_world_model = "world" in model_name.lower()
        if self._is_world_model and YOLOWorld is not None:
            self._model = YOLOWorld(model_name)
            self._fallback_model = YOLO("yolov8n.pt")
        else:
            self._model = YOLO(model_name)
            self._fallback_model = None
        self._confidence_threshold = confidence_threshold

    def detect(self, image_path: Path, issue_text: str = "") -> list[Detection]:
        classes = _build_world_classes(issue_text)
        return self._predict(image_path=image_path, world_classes=classes, allow_fallback=True)

    def detect_for_terms(self, image_path: Path, terms: list[str]) -> list[Detection]:
        world_classes = _dedupe_terms(terms)
        if not world_classes:
            return self.detect(image_path=image_path, issue_text="")
        return self._predict(image_path=image_path, world_classes=world_classes, allow_fallback=False)

    def _predict(self, image_path: Path, world_classes: list[str], allow_fallback: bool) -> list[Detection]:
        if self._is_world_model and YOLOWorld is not None:
            if world_classes:
                self._model.set_classes(world_classes)

        run_confidence = (
            min(self._confidence_threshold, 0.1)
            if self._is_world_model
            else self._confidence_threshold
        )
        results = self._model.predict(
            source=str(image_path),
            conf=run_confidence,
            verbose=False,
        )
        detections = _results_to_detections(results)
        if detections:
            return detections

        if allow_fallback and self._fallback_model is not None:
            fallback_results = self._fallback_model.predict(
                source=str(image_path),
                conf=self._confidence_threshold,
                verbose=False,
            )
            return _results_to_detections(fallback_results)

        return []


def _results_to_detections(results: list) -> list[Detection]:
    if not results:
        return []

    result = results[0]
    names = result.names
    detections: list[Detection] = []

    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = _resolve_label(names, cls_id)
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


def _resolve_label(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)):
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)
    return str(cls_id)


def _build_world_classes(issue_text: str) -> list[str]:
    classes = build_query_terms(issue_text.strip().lower())
    return _dedupe_terms(classes)


def _dedupe_terms(classes: list[str]) -> list[str]:
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for item in classes:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_unique.append(normalized)

    return ordered_unique[:64]
