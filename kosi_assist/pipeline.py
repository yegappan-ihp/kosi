from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import shutil
import sys
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image
from rapidfuzz import fuzz

from kosi_assist.config import AppConfig
from kosi_assist.detector import YoloDetector
from kosi_assist.image_utils import crop_detection, draw_detection_box
from kosi_assist.llm_client import GuidanceClient, build_guidance_client
from kosi_assist.matcher import is_electronics_label, label_match_score, select_best_detection
from kosi_assist.recognizer import RecognizeEverything
from kosi_assist.types import Detection, PipelineResult, StepTarget


_THREAD_LOCAL = threading.local()


def run_pipeline(
    image_path: Path,
    issue_text: str,
    config: AppConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineResult:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector = YoloDetector(
        model_name=config.yolo_model,
        confidence_threshold=config.yolo_confidence,
    )
    guidance_client = build_guidance_client(
        backend=config.llm_backend,
        openai_api_key=config.openai_api_key,
        openai_model=config.openai_model,
        openai_timeout_seconds=config.openai_timeout_seconds,
        openai_max_image_side=config.openai_max_image_side,
        openai_concurrency=config.openai_concurrency,
        local_model_name=config.local_vlm_model,
    )

    recognition_tags: list[str] = []
    detect_issue_text = issue_text
    if config.detector_mode == "recognize_everything":
        recognizer = RecognizeEverything(
            checkpoint_path=config.ram_checkpoint,
            max_tags=config.everything_max_tags,
        )
        recognition_tags = recognizer.collect_tags(
            image_path=image_path,
            issue_text=issue_text,
            guidance_client=guidance_client,
        )
        if recognition_tags:
            _emit_progress(
                progress_callback,
                f"Recognize-everything tags: {', '.join(recognition_tags[:10])}",
            )
    detections = detector.detect(
        image_path=image_path,
        issue_text=detect_issue_text,
        extra_terms=recognition_tags,
    )
    if not detections:
        raise ValueError(
            "No objects detected. Please retake the photo with better lighting and keep the target device centered."
        )

    selected = select_best_detection(issue_text=issue_text, detections=detections)
    selection_method = "heuristic"

    if _is_ambiguous_selection(selected, detections, issue_text):
        gpt_index = guidance_client.choose_detection(
            image_path=image_path,
            issue_text=issue_text,
            detections=detections,
        )
        if gpt_index is not None:
            selected = detections[gpt_index]
            selection_method = "gpt"
        elif _should_prompt_user_fallback() and len(detections) > 1:
            user_index = _ask_user_for_detection(detections)
            if user_index is not None:
                selected = detections[user_index]
                selection_method = "user"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crop_dir = config.output_dir / "crops"
    gpt_input_dir = config.output_dir / "gpt_inputs"
    report_dir = config.output_dir / "reports"

    crop_dir.mkdir(parents=True, exist_ok=True)
    gpt_input_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    crop_path = crop_dir / f"crop_{timestamp}.jpg"

    crop_detection(
        image_path=image_path,
        detection=selected,
        output_path=crop_path,
        padding_ratio=config.crop_padding_ratio,
    )

    gpt_input_path = gpt_input_dir / f"gpt_input_{timestamp}.jpg"
    shutil.copy2(crop_path, gpt_input_path)

    full_size = _image_size(image_path)
    cropped_size = _image_size(gpt_input_path)
    full_anchor = selected
    cropped_anchor = _detect_cropped_anchor(
        detector=detector,
        cropped_path=gpt_input_path,
        selected_label=selected.label,
        fallback_size=cropped_size,
    )

    instructions = guidance_client.get_instructions(
        crop_path=gpt_input_path,
        issue_text=issue_text,
        target_label=selected.label,
    )
    _emit_progress(progress_callback, "Generated text instructions")

    step_targets = _prune_step_targets(
        guidance_client.extract_step_targets(instructions),
        max_targets=config.max_step_targets,
    )
    _emit_progress(progress_callback, f"Preparing visual highlights for {len(step_targets)} steps")

    reply_dir = config.reply_dir
    _reset_reply_dir(reply_dir)
    reply_images: list[Path] = []
    unfound_targets: list[str] = []
    visual_results: list[dict] = []

    step_results: list[dict] = []
    worker_count = max(1, min(config.step_workers, max(1, len(step_targets))))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _process_step_target,
                step_target=step_target,
                config=config,
                guidance_client=guidance_client,
                cropped_path=gpt_input_path,
                full_image_path=image_path,
                cropped_anchor=cropped_anchor,
                full_anchor=full_anchor,
                cropped_size=cropped_size,
                full_size=full_size,
                reply_dir=reply_dir,
            ): step_target
            for step_target in step_targets
        }

        for future in as_completed(futures):
            result = future.result()
            step_results.append(result)
            if result["found"]:
                _emit_progress(
                    progress_callback,
                    (
                        f"Step {result['step_number']} image ready: "
                        f"{result['reply_image']}"
                    ),
                )
            else:
                _emit_progress(
                    progress_callback,
                    (
                        f"Step {result['step_number']} unfound target: "
                        f"{result['visual_target']}"
                    ),
                )

    step_results.sort(key=lambda item: (item["step_number"], item["visual_target"]))
    for result in step_results:
        if result["found"]:
            reply_images.append(Path(result["reply_image"]))
        else:
            unfound_targets.append(result["visual_target"])
        visual_results.append(result)

    desired_reply_count = 1 if len(step_targets) <= 1 else 2
    if len(reply_images) < desired_reply_count and step_targets:
        missing = desired_reply_count - len(reply_images)
        best_effort = _generate_best_effort_replies(
            missing_count=missing,
            step_targets=step_targets,
            existing_reply_images=reply_images,
            config=config,
            cropped_path=gpt_input_path,
            full_image_path=image_path,
            cropped_anchor=cropped_anchor,
            full_anchor=full_anchor,
            cropped_size=cropped_size,
            full_size=full_size,
            reply_dir=reply_dir,
        )
        for item in best_effort:
            reply_images.append(Path(item["reply_image"]))
            visual_results.append(item)
            target_name = item["visual_target"]
            if target_name in unfound_targets:
                unfound_targets.remove(target_name)
            _emit_progress(
                progress_callback,
                f"Best-effort image ready for step {item['step_number']}: {item['reply_image']}",
            )

    report_path = report_dir / f"report_{timestamp}.json"
    report = {
        "image_path": str(image_path),
        "issue_text": issue_text,
        "selected_detection": asdict(selected),
        "all_detections": [d.to_dict() for d in detections],
        "crop_path": str(crop_path),
        "gpt_input_path": str(gpt_input_path),
        "instructions": instructions,
        "step_targets": [asdict(s) for s in step_targets],
        "reply_dir": str(reply_dir),
        "reply_images": [str(p) for p in reply_images],
        "unfound_targets": unfound_targets,
        "visual_results": visual_results,
        "selection_method": selection_method,
        "detector_mode": config.detector_mode,
        "recognition_tags": recognition_tags,
        "llm_backend": config.llm_backend,
        "openai_model": config.openai_model,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return PipelineResult(
        selected_detection=selected,
        all_detections=detections,
        selection_method=selection_method,
        step_targets=step_targets,
        unfound_targets=unfound_targets,
        reply_images=reply_images,
        reply_dir=reply_dir,
        crop_path=crop_path,
        gpt_input_path=gpt_input_path,
        report_path=report_path,
        instructions=instructions,
    )


def _is_ambiguous_selection(selected, detections, issue_text: str) -> bool:
    generic_labels = {
        "electronic device",
        "device",
        "object",
        "thing",
        "unknown",
    }
    if selected.label.strip().lower() in generic_labels:
        return True

    if label_match_score(issue_text, selected.label) < 55.0:
        return True

    sorted_by_conf = sorted(detections, key=lambda d: d.confidence, reverse=True)
    if len(sorted_by_conf) > 1:
        gap = sorted_by_conf[0].confidence - sorted_by_conf[1].confidence
        if gap < 0.08:
            return True

    return selected.confidence < 0.38


def _should_prompt_user_fallback() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()


def _ask_user_for_detection(detections) -> int | None:
    print("\nI could not confidently pick one object. Please select one option:")
    for idx, det in enumerate(detections[:8]):
        print(f"  {idx}: {det.label} (confidence {det.confidence:.2f})")

    value = input("Enter object index (or press Enter to skip): ").strip()
    if not value:
        return None
    if not value.isdigit():
        return None

    parsed = int(value)
    if 0 <= parsed < len(detections):
        return parsed
    return None


def _reset_reply_dir(reply_dir: Path) -> None:
    reply_dir.mkdir(parents=True, exist_ok=True)
    for entry in reply_dir.iterdir():
        if entry.is_file():
            entry.unlink()


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _prune_step_targets(step_targets: list[StepTarget], max_targets: int) -> list[StepTarget]:
    if max_targets <= 0:
        return []

    ordered = sorted(step_targets, key=lambda s: (s.step_number, len(s.visual_target)))
    pruned: list[StepTarget] = []
    seen: set[tuple[int, str]] = set()
    for step in ordered:
        canonical = _canonical_target(step.visual_target)
        key = (step.step_number, canonical)
        if key in seen:
            continue
        seen.add(key)
        pruned.append(step)
        if len(pruned) >= max_targets:
            break
    return pruned


def _process_step_target(
    step_target: StepTarget,
    config: AppConfig,
    guidance_client: GuidanceClient,
    cropped_path: Path,
    full_image_path: Path,
    cropped_anchor: Detection,
    full_anchor: Detection,
    cropped_size: tuple[int, int],
    full_size: tuple[int, int],
    reply_dir: Path,
) -> dict:
    detector = _get_thread_detector(
        model_name=config.yolo_model,
        confidence_threshold=config.yolo_confidence,
    )
    located = _locate_target_for_step(
        detector=detector,
        guidance_client=guidance_client,
        step_target=step_target,
        cropped_path=cropped_path,
        full_image_path=full_image_path,
        cropped_anchor=cropped_anchor,
        full_anchor=full_anchor,
        cropped_size=cropped_size,
        full_size=full_size,
    )
    if not located:
        return {
            "step_number": step_target.step_number,
            "instruction": step_target.instruction,
            "visual_target": step_target.visual_target,
            "found": False,
        }

    detection, source_key, source_path = located
    reply_name = (
        f"step_{step_target.step_number:02d}_"
        f"{_slugify(step_target.visual_target)}_{source_key}.jpg"
    )
    reply_path = reply_dir / reply_name
    draw_detection_box(
        image_path=source_path,
        detection=detection,
        output_path=reply_path,
        label_text=step_target.visual_target,
    )

    return {
        "step_number": step_target.step_number,
        "instruction": step_target.instruction,
        "visual_target": step_target.visual_target,
        "found": True,
        "source": source_key,
        "reply_image": str(reply_path),
        "detection": detection.to_dict(),
    }


def _generate_best_effort_replies(
    missing_count: int,
    step_targets: list[StepTarget],
    existing_reply_images: list[Path],
    config: AppConfig,
    cropped_path: Path,
    full_image_path: Path,
    cropped_anchor: Detection,
    full_anchor: Detection,
    cropped_size: tuple[int, int],
    full_size: tuple[int, int],
    reply_dir: Path,
) -> list[dict]:
    if missing_count <= 0:
        return []

    detector = _get_thread_detector(
        model_name=config.yolo_model,
        confidence_threshold=config.yolo_confidence,
    )
    produced: list[dict] = []
    used_reply_names = {p.name for p in existing_reply_images}

    for step_target in step_targets:
        if len(produced) >= missing_count:
            break

        candidate = _best_effort_locate_target(
            detector=detector,
            step_target=step_target,
            cropped_path=cropped_path,
            full_image_path=full_image_path,
            cropped_anchor=cropped_anchor,
            full_anchor=full_anchor,
            cropped_size=cropped_size,
            full_size=full_size,
        )
        if candidate is None:
            continue

        detection, source_key, source_path = candidate
        reply_name = (
            f"step_{step_target.step_number:02d}_"
            f"{_slugify(step_target.visual_target)}_{source_key}_best_effort.jpg"
        )
        if reply_name in used_reply_names:
            continue
        used_reply_names.add(reply_name)

        reply_path = reply_dir / reply_name
        draw_detection_box(
            image_path=source_path,
            detection=detection,
            output_path=reply_path,
            label_text=f"{step_target.visual_target} (best effort)",
        )
        produced.append(
            {
                "step_number": step_target.step_number,
                "instruction": step_target.instruction,
                "visual_target": step_target.visual_target,
                "found": True,
                "source": f"{source_key}_best_effort",
                "reply_image": str(reply_path),
                "detection": detection.to_dict(),
            }
        )

    return produced


def _best_effort_locate_target(
    detector: YoloDetector,
    step_target: StepTarget,
    cropped_path: Path,
    full_image_path: Path,
    cropped_anchor: Detection,
    full_anchor: Detection,
    cropped_size: tuple[int, int],
    full_size: tuple[int, int],
) -> tuple[Detection, str, Path] | None:
    canonical_target = _canonical_target(step_target.visual_target)
    terms = _target_terms(canonical_target)
    scope, zone = _target_scope_and_zone(canonical_target)

    full_detections = detector.detect_for_terms(full_image_path, terms)
    best_full = _best_detection_for_target(canonical_target, full_detections)
    if best_full is not None and _validate_best_effort_candidate(
        candidate=best_full,
        target=canonical_target,
        scope=scope,
        anchor=full_anchor,
        image_size=full_size,
    ):
        return _expand_detection(best_full, full_size, ratio=0.08), "full", full_image_path

    cropped_detections = detector.detect_for_terms(cropped_path, terms)
    best_cropped = _best_detection_for_target(canonical_target, cropped_detections)
    if best_cropped is not None and _validate_best_effort_candidate(
        candidate=best_cropped,
        target=canonical_target,
        scope=scope,
        anchor=cropped_anchor,
        image_size=cropped_size,
    ):
        return _expand_detection(best_cropped, cropped_size, ratio=0.08), "cropped", cropped_path

    hint = _anchor_hint_detection(
        target=canonical_target,
        scope=scope,
        zone=zone,
        anchor=full_anchor,
        image_size=full_size,
    )
    if hint is not None:
        return hint, "full_hint", full_image_path

    return None


def _validate_best_effort_candidate(
    candidate: Detection,
    target: str,
    scope: str,
    anchor: Detection,
    image_size: tuple[int, int],
) -> bool:
    image_w, image_h = image_size
    if candidate.x1 < 0 or candidate.y1 < 0 or candidate.x2 > image_w or candidate.y2 > image_h:
        return False

    box_w = candidate.x2 - candidate.x1
    box_h = candidate.y2 - candidate.y1
    if box_w < 12 or box_h < 12:
        return False

    area = box_w * box_h
    image_area = max(1, image_w * image_h)
    if area > image_area * 0.75:
        return False

    target_score = label_match_score(target, candidate.label)
    if target_score < 30:
        return False

    if scope == "local":
        if not _is_near_anchor(candidate, anchor, tolerance_ratio=1.8):
            return False

    if scope == "nearby":
        if not _is_near_anchor(candidate, anchor, tolerance_ratio=2.2):
            return False

    return True


def _anchor_hint_detection(
    target: str,
    scope: str,
    zone: str,
    anchor: Detection,
    image_size: tuple[int, int],
) -> Detection | None:
    image_w, image_h = image_size
    ax1, ay1, ax2, ay2 = anchor.x1, anchor.y1, anchor.x2, anchor.y2
    aw = max(40, ax2 - ax1)
    ah = max(40, ay2 - ay1)

    if scope == "local":
        if zone == "screen_top_left":
            x1 = ax1 + int(0.02 * aw)
            y1 = ay1 + int(0.02 * ah)
            x2 = x1 + int(0.2 * aw)
            y2 = y1 + int(0.12 * ah)
        elif zone == "top_right":
            x1 = ax1 + int(0.72 * aw)
            y1 = ay1 + int(0.08 * ah)
            x2 = x1 + int(0.18 * aw)
            y2 = y1 + int(0.12 * ah)
        elif zone == "keyboard_area":
            x1 = ax1 + int(0.18 * aw)
            y1 = ay1 + int(0.55 * ah)
            x2 = ax1 + int(0.86 * aw)
            y2 = ay1 + int(0.88 * ah)
        elif zone == "screen_area":
            x1 = ax1 + int(0.08 * aw)
            y1 = ay1 + int(0.08 * ah)
            x2 = ax1 + int(0.92 * aw)
            y2 = ay1 + int(0.52 * ah)
        else:
            x1 = ax1 + int(0.1 * aw)
            y1 = ay1 + int(0.1 * ah)
            x2 = ax1 + int(0.9 * aw)
            y2 = ay1 + int(0.9 * ah)
    elif scope == "nearby":
        x1 = max(0, ax1 - int(0.2 * aw))
        y1 = max(0, ay1 + int(0.38 * ah))
        x2 = min(image_w, ax1 + int(0.2 * aw))
        y2 = min(image_h, ay1 + int(0.78 * ah))
    else:
        return None

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(x1 + 12, min(image_w, x2))
    y2 = max(y1 + 12, min(image_h, y2))

    return Detection(
        label=f"{target} (hint)",
        confidence=0.28,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _get_thread_detector(model_name: str, confidence_threshold: float) -> YoloDetector:
    key = (model_name, confidence_threshold)
    cached = getattr(_THREAD_LOCAL, "detector_cache", None)
    if cached is None:
        cached = {}
        _THREAD_LOCAL.detector_cache = cached

    detector = cached.get(key)
    if detector is None:
        detector = YoloDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
        cached[key] = detector
    return detector


def _locate_target_for_step(
    detector: YoloDetector,
    guidance_client: GuidanceClient,
    step_target: StepTarget,
    cropped_path: Path,
    full_image_path: Path,
    cropped_anchor: Detection,
    full_anchor: Detection,
    cropped_size: tuple[int, int],
    full_size: tuple[int, int],
) -> tuple[Detection, str, Path] | None:
    canonical_target = _canonical_target(step_target.visual_target)
    scope, zone = _target_scope_and_zone(canonical_target)
    terms = _target_terms(canonical_target)

    cropped_detections = detector.detect_for_terms(cropped_path, terms)
    best_cropped = _best_detection_for_target(canonical_target, cropped_detections)
    if best_cropped is not None and _validate_candidate(
        candidate=best_cropped,
        target=canonical_target,
        scope=scope,
        zone=zone,
        anchor=cropped_anchor,
        image_size=cropped_size,
        source_key="cropped",
    ):
        finalized = _finalize_candidate(
            guidance_client=guidance_client,
            candidate=best_cropped,
            target=canonical_target,
            instruction=step_target.instruction,
            anchor=cropped_anchor,
            image_size=cropped_size,
            scope=scope,
            zone=zone,
            source_key="cropped",
            source_path=cropped_path,
        )
        if finalized is not None:
            return finalized, "cropped", cropped_path

    full_detections = detector.detect_for_terms(full_image_path, terms)
    best_full = _best_detection_for_target(canonical_target, full_detections)
    if best_full is not None and _validate_candidate(
        candidate=best_full,
        target=canonical_target,
        scope=scope,
        zone=zone,
        anchor=full_anchor,
        image_size=full_size,
        source_key="full",
    ):
        finalized = _finalize_candidate(
            guidance_client=guidance_client,
            candidate=best_full,
            target=canonical_target,
            instruction=step_target.instruction,
            anchor=full_anchor,
            image_size=full_size,
            scope=scope,
            zone=zone,
            source_key="full",
            source_path=full_image_path,
        )
        if finalized is not None:
            return finalized, "full", full_image_path

    if scope == "local":
        gpt_cropped = guidance_client.locate_visual_target(
            image_path=cropped_path,
            visual_target=step_target.visual_target,
        )
        if gpt_cropped is not None and _validate_candidate(
            candidate=gpt_cropped,
            target=canonical_target,
            scope=scope,
            zone=zone,
            anchor=cropped_anchor,
            image_size=cropped_size,
            source_key="cropped_gpt",
        ):
            finalized = _finalize_candidate(
                guidance_client=guidance_client,
                candidate=gpt_cropped,
                target=canonical_target,
                instruction=step_target.instruction,
                anchor=cropped_anchor,
                image_size=cropped_size,
                scope=scope,
                zone=zone,
                source_key="cropped_gpt",
                source_path=cropped_path,
            )
            if finalized is not None:
                return finalized, "cropped_gpt", cropped_path
    else:
        gpt_full = guidance_client.locate_visual_target(
            image_path=full_image_path,
            visual_target=step_target.visual_target,
        )
        if gpt_full is not None and _validate_candidate(
            candidate=gpt_full,
            target=canonical_target,
            scope=scope,
            zone=zone,
            anchor=full_anchor,
            image_size=full_size,
            source_key="full_gpt",
        ):
            finalized = _finalize_candidate(
                guidance_client=guidance_client,
                candidate=gpt_full,
                target=canonical_target,
                instruction=step_target.instruction,
                anchor=full_anchor,
                image_size=full_size,
                scope=scope,
                zone=zone,
                source_key="full_gpt",
                source_path=full_image_path,
            )
            if finalized is not None:
                return finalized, "full_gpt", full_image_path

    return None


def _best_detection_for_target(target: str, detections: list[Detection]) -> Detection | None:
    if not detections:
        return None
    target_lower = target.strip().lower()
    electronics_target = _looks_like_electronics_target(target_lower)

    best: Detection | None = None
    best_score = -1.0
    for det in detections:
        label_lower = det.label.strip().lower()
        text_score = _strict_target_label_score(target_lower, label_lower)
        score = 0.75 * text_score + 25.0 * det.confidence
        if electronics_target:
            if is_electronics_label(label_lower):
                score += 12.0
            else:
                score -= 30.0
        if score > best_score:
            best_score = score
            best = det
    if best is None:
        return None
    best_label = best.label.strip().lower()
    min_text = 62.0 if electronics_target else 55.0
    if _strict_target_label_score(target_lower, best_label) < min_text:
        return None
    if electronics_target and not is_electronics_label(best_label):
        return None
    return best


def _target_terms(target: str) -> list[str]:
    normalized = target.strip().lower()
    terms = [normalized]
    if "button" in normalized:
        terms.extend(["button", "power button", "switch", "key"])
    if "light" in normalized or "indicator" in normalized:
        terms.extend(["indicator light", "led light", "status light"])
    if "port" in normalized:
        terms.extend(["usb port", "charging port", "connector"])
    if "monitor" in normalized or "screen" in normalized:
        terms.extend(["monitor", "screen", "display"])
    if "cable" in normalized or "wire" in normalized:
        terms.extend(["cable", "wire", "charging cable", "usb cable"])
    return _dedupe_preserve_order(terms)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        v = value.strip().lower()
        if not v or v in seen:
            continue
        out.append(v)
        seen.add(v)
    return out


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "target"


def _strict_target_label_score(target: str, label: str) -> float:
    score = max(
        fuzz.partial_ratio(target, label),
        fuzz.token_set_ratio(target, label),
    )
    if label in target or target in label:
        score += 8.0
    return score


def _looks_like_electronics_target(target: str) -> bool:
    hints = [
        "power",
        "button",
        "port",
        "usb",
        "cable",
        "charger",
        "adapter",
        "monitor",
        "screen",
        "keyboard",
        "mouse",
        "laptop",
        "router",
        "wlan",
        "printer",
        "device",
    ]
    return any(h in target for h in hints)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _detect_cropped_anchor(
    detector: YoloDetector,
    cropped_path: Path,
    selected_label: str,
    fallback_size: tuple[int, int],
) -> Detection:
    terms = [selected_label, "laptop", "computer", "electronic device", "device"]
    candidates = detector.detect_for_terms(cropped_path, _dedupe_preserve_order(terms))
    if candidates:
        best = max(candidates, key=lambda d: d.confidence)
        return best

    width, height = fallback_size
    return Detection(
        label=selected_label,
        confidence=1.0,
        x1=0,
        y1=0,
        x2=width,
        y2=height,
    )


def _target_scope_and_zone(target: str) -> tuple[str, str]:
    text = target.strip().lower()
    if any(k in text for k in ["apple menu"]):
        return "local", "screen_top_left"
    if any(k in text for k in ["indicator light", "status light", "light"]):
        return "nearby", "any"
    if any(k in text for k in ["screen", "display", "monitor", "error message", "popup"]):
        return "local", "screen_area"
    if any(k in text for k in ["keyboard", "keys", "key"]):
        return "local", "keyboard_area"
    if "trackpad" in text:
        return "local", "trackpad_area"
    if any(k in text for k in ["button", "power button"]):
        return "local", "top_right"
    if "port" in text:
        return "local", "edge_band"
    if any(k in text for k in ["cable", "wire", "charger", "plug", "battery", "adapter"]):
        return "nearby", "any"
    if any(k in text for k in ["socket", "outlet", "power strip", "wall"]):
        return "scene", "any"
    return "local", "any"


def _canonical_target(target: str) -> str:
    text = target.strip().lower()
    if "apple menu" in text:
        return "apple menu"
    if "brightness" in text and "keyboard" in text:
        return "keyboard"
    if "trackpad" in text:
        return "trackpad"
    if "keyboard" in text or "keys" in text:
        return "keyboard"
    if "power button" in text or "button" in text:
        return "power button"
    if "screen" in text or "display" in text or "monitor" in text:
        return "screen"
    if "light" in text or "indicator" in text:
        return "indicator light"
    if "usb" in text and "cable" in text:
        return "usb cable"
    if "usb" in text and "device" in text:
        return "usb device"
    if "cable" in text or "wire" in text:
        return "cable"
    if "charger" in text:
        return "charger"
    if "battery" in text:
        return "battery"
    return text


def _validate_candidate(
    candidate: Detection,
    target: str,
    scope: str,
    zone: str,
    anchor: Detection,
    image_size: tuple[int, int],
    source_key: str,
) -> bool:
    image_w, image_h = image_size
    if candidate.x1 < 0 or candidate.y1 < 0 or candidate.x2 > image_w or candidate.y2 > image_h:
        return False

    box_w = candidate.x2 - candidate.x1
    box_h = candidate.y2 - candidate.y1
    if box_w < 10 or box_h < 10:
        return False

    anchor_w = max(1, anchor.x2 - anchor.x1)
    anchor_h = max(1, anchor.y2 - anchor.y1)
    anchor_area = anchor_w * anchor_h
    cand_area = box_w * box_h
    image_area = max(1, image_w * image_h)

    cx = (candidate.x1 + candidate.x2) / 2.0
    cy = (candidate.y1 + candidate.y2) / 2.0

    if scope == "local":
        if not _point_in_expanded_anchor(cx, cy, anchor, expand_ratio=0.08):
            return False
        if cand_area > anchor_area * 0.62:
            return False
        if cand_area < anchor_area * 0.0015:
            return False
        if not _passes_zone_rule(candidate, anchor, zone):
            return False

    if scope == "nearby":
        if not _is_near_anchor(candidate, anchor, tolerance_ratio=1.45):
            return False
        if cand_area > image_area * 0.28:
            return False
        if any(k in target for k in ["cable", "charger", "plug", "usb"]):
            if not _intersects_connectivity_ring(candidate, anchor, ring_ratio=0.28):
                return False

    if "light" in target or "button" in target or "menu" in target or "key" in target:
        if cand_area > image_area * 0.08:
            return False

    if source_key.endswith("gpt") and scope == "local":
        border = min(candidate.x1, candidate.y1, image_w - candidate.x2, image_h - candidate.y2)
        if border <= 2 and zone != "screen_top_left":
            return False

    target_lower = target.strip().lower()
    if scope != "scene" and "cable" not in target_lower and "wire" not in target_lower:
        if label_match_score(target_lower, candidate.label.lower()) < 40 and source_key in {"cropped", "full"}:
            return False

    return True


def _finalize_candidate(
    guidance_client: GuidanceClient,
    candidate: Detection,
    target: str,
    instruction: str,
    anchor: Detection,
    image_size: tuple[int, int],
    scope: str,
    zone: str,
    source_key: str,
    source_path: Path,
) -> Detection | None:
    expanded = _expand_detection(candidate, image_size, ratio=0.1)

    if _needs_verification(target, expanded, source_key):
        corrected = guidance_client.verify_bbox(
            image_path=source_path,
            visual_target=target,
            instruction=instruction,
            detection=expanded,
        )
        if corrected is None:
            return None
        expanded = _expand_detection(corrected, image_size, ratio=0.06)

    if not _validate_candidate(
        candidate=expanded,
        target=target,
        scope=scope,
        zone=zone,
        anchor=anchor,
        image_size=image_size,
        source_key=source_key,
    ):
        return None

    return expanded


def _needs_verification(target: str, detection: Detection, source_key: str) -> bool:
    if source_key.endswith("gpt"):
        return True
    if detection.confidence < 0.35:
        return True
    return False


def _expand_detection(detection: Detection, image_size: tuple[int, int], ratio: float) -> Detection:
    width, height = image_size
    box_w = detection.x2 - detection.x1
    box_h = detection.y2 - detection.y1
    pad_x = max(4, int(box_w * max(0.0, ratio)))
    pad_y = max(4, int(box_h * max(0.0, ratio)))
    x1 = max(0, detection.x1 - pad_x)
    y1 = max(0, detection.y1 - pad_y)
    x2 = min(width, detection.x2 + pad_x)
    y2 = min(height, detection.y2 + pad_y)
    return Detection(
        label=detection.label,
        confidence=detection.confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _point_in_expanded_anchor(cx: float, cy: float, anchor: Detection, expand_ratio: float) -> bool:
    w = anchor.x2 - anchor.x1
    h = anchor.y2 - anchor.y1
    ex = w * expand_ratio
    ey = h * expand_ratio
    return (
        anchor.x1 - ex <= cx <= anchor.x2 + ex
        and anchor.y1 - ey <= cy <= anchor.y2 + ey
    )


def _is_near_anchor(candidate: Detection, anchor: Detection, tolerance_ratio: float) -> bool:
    cx = (candidate.x1 + candidate.x2) / 2.0
    cy = (candidate.y1 + candidate.y2) / 2.0
    ax = (anchor.x1 + anchor.x2) / 2.0
    ay = (anchor.y1 + anchor.y2) / 2.0
    dx = abs(cx - ax)
    dy = abs(cy - ay)
    max_dx = (anchor.x2 - anchor.x1) * tolerance_ratio
    max_dy = (anchor.y2 - anchor.y1) * tolerance_ratio
    return dx <= max_dx and dy <= max_dy


def _intersects_connectivity_ring(candidate: Detection, anchor: Detection, ring_ratio: float) -> bool:
    ring_x = int((anchor.x2 - anchor.x1) * ring_ratio)
    ring_y = int((anchor.y2 - anchor.y1) * ring_ratio)
    outer = Detection(
        label="outer",
        confidence=1.0,
        x1=max(0, anchor.x1 - ring_x),
        y1=max(0, anchor.y1 - ring_y),
        x2=anchor.x2 + ring_x,
        y2=anchor.y2 + ring_y,
    )

    intersects_outer = not (
        candidate.x2 < outer.x1
        or candidate.x1 > outer.x2
        or candidate.y2 < outer.y1
        or candidate.y1 > outer.y2
    )
    intersects_anchor = not (
        candidate.x2 < anchor.x1
        or candidate.x1 > anchor.x2
        or candidate.y2 < anchor.y1
        or candidate.y1 > anchor.y2
    )
    return intersects_outer and (intersects_anchor or _touches_anchor_edge(candidate, anchor))


def _touches_anchor_edge(candidate: Detection, anchor: Detection) -> bool:
    near_left = abs(candidate.x2 - anchor.x1) <= 120
    near_right = abs(candidate.x1 - anchor.x2) <= 120
    near_top = abs(candidate.y2 - anchor.y1) <= 120
    near_bottom = abs(candidate.y1 - anchor.y2) <= 120
    return near_left or near_right or near_top or near_bottom


def _passes_zone_rule(candidate: Detection, anchor: Detection, zone: str) -> bool:
    if zone == "any":
        return True

    aw = max(1, anchor.x2 - anchor.x1)
    ah = max(1, anchor.y2 - anchor.y1)
    cx = (candidate.x1 + candidate.x2) / 2.0
    cy = (candidate.y1 + candidate.y2) / 2.0
    nx = (cx - anchor.x1) / aw
    ny = (cy - anchor.y1) / ah

    if zone == "screen_top_left":
        return nx <= 0.45 and ny <= 0.5
    if zone == "screen_area":
        return ny <= 0.7
    if zone == "keyboard_area":
        return ny >= 0.42
    if zone == "trackpad_area":
        return 0.22 <= nx <= 0.78 and ny >= 0.58
    if zone == "top_right":
        return nx >= 0.5 and ny <= 0.55
    if zone == "edge_band":
        return nx <= 0.2 or nx >= 0.8 or ny <= 0.2 or ny >= 0.8
    return True
