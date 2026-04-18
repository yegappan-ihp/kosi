from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from kosi_assist.config import AppConfig
from kosi_assist.detector import YoloDetector
from kosi_assist.llm_client import GuidanceClient, build_guidance_client
from kosi_assist.matcher import label_match_score
from kosi_assist.pipeline import (
    _dedupe_preserve_order,
    _expand_detection,
    _intersects_connectivity_ring,
    _validate_candidate,
)
from kosi_assist.types import Detection


@dataclass(frozen=True)
class ConnectivityScenario:
    name: str
    anchor_terms: list[str]
    peripheral_terms: list[str]
    cable_terms: list[str]
    port_terms: list[str]
    target_port_label: str
    target_cable_label: str


@dataclass(frozen=True)
class FrameValidationResult:
    frame_index: int
    timestamp_seconds: float
    state: str
    confidence: float
    frame_path: str
    annotated_path: str
    anchor: dict | None
    peripheral: dict | None
    cable: dict | None
    port: dict | None
    reason: str


@dataclass(frozen=True)
class VideoValidationResult:
    issue_text: str
    scenario: str
    video_path: str
    sampled_frames: int
    summary: dict[str, int]
    assistant_output: str
    best_frames: list[dict[str, float | str | int]]
    report_path: str
    annotated_dir: str
    frames: list[FrameValidationResult]


def run_video_validation(
    video_path: Path,
    issue_text: str,
    config: AppConfig,
    sample_every_seconds: float = 0.5,
    max_frames: int = 18,
) -> VideoValidationResult:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    scenario = infer_connectivity_scenario(issue_text)
    detector = YoloDetector(
        model_name=config.yolo_model,
        confidence_threshold=config.yolo_confidence,
    )
    guidance_client = _maybe_build_guidance_client(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = config.output_dir / "video_validation" / f"run_{timestamp}"
    frames_dir = base_dir / "frames"
    annotated_dir = base_dir / "annotated"
    report_dir = config.output_dir / "reports"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * max(0.1, sample_every_seconds))))
    sample_indices = _build_sample_indices(total_frames, step, max_frames)

    results: list[FrameValidationResult] = []
    fallback_budget = 4
    try:
        for order, frame_index in enumerate(sample_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            frame_path = frames_dir / f"frame_{order:03d}.jpg"
            annotated_path = annotated_dir / f"frame_{order:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            result, used_budget = _analyze_frame(
                frame_path=frame_path,
                annotated_path=annotated_path,
                issue_text=issue_text,
                scenario=scenario,
                detector=detector,
                guidance_client=guidance_client,
                gpt_fallback_budget=fallback_budget,
                timestamp_seconds=frame_index / max(fps, 1.0),
                frame_index=frame_index,
            )
            fallback_budget = max(0, fallback_budget - used_budget)
            results.append(result)
    finally:
        capture.release()

    summary = {
        "connected": sum(1 for r in results if r.state == "connected"),
        "near": sum(1 for r in results if r.state == "near"),
        "not_connected": sum(1 for r in results if r.state == "not_connected"),
        "inconclusive": sum(1 for r in results if r.state == "inconclusive"),
    }
    assistant_output = _build_assistant_output(issue_text, scenario, results)
    best_frames = _select_best_frames(results)
    report_path = report_dir / f"video_validation_{timestamp}.json"
    payload = {
        "issue_text": issue_text,
        "scenario": asdict(scenario),
        "video_path": str(video_path),
        "sampled_frames": len(results),
        "summary": summary,
        "assistant_output": assistant_output,
        "best_frames": best_frames,
        "frames": [asdict(r) for r in results],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return VideoValidationResult(
        issue_text=issue_text,
        scenario=scenario.name,
        video_path=str(video_path),
        sampled_frames=len(results),
        summary=summary,
        assistant_output=assistant_output,
        best_frames=best_frames,
        report_path=str(report_path),
        annotated_dir=str(annotated_dir),
        frames=results,
    )


def infer_connectivity_scenario(issue_text: str) -> ConnectivityScenario:
    text = issue_text.strip().lower()
    if "wired mouse" in text or ("mouse" in text and "wired" in text):
        return ConnectivityScenario(
            name="wired_mouse_usb",
            anchor_terms=["laptop", "computer", "desktop computer", "monitor"],
            peripheral_terms=["mouse", "wired mouse"],
            cable_terms=["usb cable", "mouse cable", "cable", "wire"],
            port_terms=["usb port", "port", "connector"],
            target_port_label="usb port",
            target_cable_label="mouse cable",
        )
    if "keyboard" in text and "wired" in text:
        return ConnectivityScenario(
            name="wired_keyboard_usb",
            anchor_terms=["laptop", "computer", "desktop computer", "monitor"],
            peripheral_terms=["keyboard", "wired keyboard"],
            cable_terms=["usb cable", "keyboard cable", "cable", "wire"],
            port_terms=["usb port", "port", "connector"],
            target_port_label="usb port",
            target_cable_label="keyboard cable",
        )
    if any(term in text for term in ["charger", "charging", "charge"]):
        return ConnectivityScenario(
            name="charging_connection",
            anchor_terms=["phone", "tablet", "laptop", "computer", "router"],
            peripheral_terms=[],
            cable_terms=["charging cable", "usb cable", "charger", "cable", "wire"],
            port_terms=["charging port", "usb port", "port", "connector"],
            target_port_label="charging port",
            target_cable_label="charging cable",
        )
    return ConnectivityScenario(
        name="generic_usb_connection",
        anchor_terms=["laptop", "computer", "desktop computer", "router", "printer"],
        peripheral_terms=[],
        cable_terms=["usb cable", "cable", "wire", "plug"],
        port_terms=["usb port", "port", "connector"],
        target_port_label="usb port",
        target_cable_label="usb cable",
    )


def _maybe_build_guidance_client(config: AppConfig) -> GuidanceClient | None:
    if config.llm_backend != "openai" or not config.openai_api_key:
        return None
    try:
        return build_guidance_client(
            backend=config.llm_backend,
            openai_api_key=config.openai_api_key,
            openai_model=config.openai_model,
            openai_timeout_seconds=config.openai_timeout_seconds,
            openai_max_image_side=config.openai_max_image_side,
            openai_concurrency=config.openai_concurrency,
            local_model_name=config.local_vlm_model,
        )
    except Exception:
        return None


def _build_sample_indices(total_frames: int, step: int, max_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    indices = list(range(0, total_frames, step))
    if len(indices) <= max_frames:
        return indices
    if max_frames <= 1:
        return [indices[0]]
    last = len(indices) - 1
    selected_positions = {
        round(i * last / (max_frames - 1))
        for i in range(max_frames)
    }
    return [indices[pos] for pos in sorted(selected_positions)]


def _analyze_frame(
    frame_path: Path,
    annotated_path: Path,
    issue_text: str,
    scenario: ConnectivityScenario,
    detector: YoloDetector,
    guidance_client: GuidanceClient | None,
    gpt_fallback_budget: int,
    timestamp_seconds: float,
    frame_index: int,
) -> tuple[FrameValidationResult, int]:
    image_w, image_h = _image_size(frame_path)
    issue_detections = detector.detect(image_path=frame_path, issue_text=issue_text)
    anchor_detections = detector.detect_for_terms(frame_path, _dedupe_preserve_order(scenario.anchor_terms))
    peripheral_detections = (
        detector.detect_for_terms(frame_path, _dedupe_preserve_order(scenario.peripheral_terms))
        if scenario.peripheral_terms
        else []
    )

    anchor = _select_anchor_detection(anchor_detections, scenario.anchor_terms)
    peripheral = _select_best_detection(peripheral_detections, scenario.peripheral_terms) if scenario.peripheral_terms else None
    if anchor is None:
        anchor = _select_best_detection(issue_detections, scenario.anchor_terms)

    if anchor is None:
        _draw_validation_frame(
            frame_path=frame_path,
            annotated_path=annotated_path,
            state="inconclusive",
            anchor=None,
            cable=None,
            port=None,
            peripheral=peripheral,
            note="Host device not found",
        )
        return (
            FrameValidationResult(
                frame_index=frame_index,
                timestamp_seconds=round(timestamp_seconds, 2),
                state="inconclusive",
                confidence=0.0,
                frame_path=str(frame_path),
                annotated_path=str(annotated_path),
                anchor=None,
                peripheral=_det_to_dict(peripheral),
                cable=None,
                port=None,
                reason="Host device not found",
            ),
            0,
        )

    cable_candidates = _filter_valid_candidates(
        detections=detector.detect_for_terms(frame_path, _dedupe_preserve_order(scenario.cable_terms)),
        anchor=anchor,
        image_size=(image_w, image_h),
        target="cable",
        source_key="frame",
    )
    cable = _select_best_by_anchor(cable_candidates, anchor, scenario.target_cable_label)

    port_candidates = _filter_valid_candidates(
        detections=detector.detect_for_terms(frame_path, _dedupe_preserve_order(scenario.port_terms)),
        anchor=anchor,
        image_size=(image_w, image_h),
        target=scenario.target_port_label,
        source_key="frame",
    )
    port = _select_best_port(port_candidates, cable, anchor, scenario.target_port_label)

    budget_used = 0
    if port is None and guidance_client is not None and gpt_fallback_budget > 0 and cable is not None:
        candidate = guidance_client.locate_visual_target(frame_path, scenario.target_port_label)
        budget_used += 1
        if candidate is not None and _is_valid_port_candidate(candidate, anchor, (image_w, image_h), scenario.target_port_label):
            port = _expand_detection(candidate, (image_w, image_h), ratio=0.06)

    if cable is None and guidance_client is not None and gpt_fallback_budget - budget_used > 0:
        candidate = guidance_client.locate_visual_target(frame_path, scenario.target_cable_label)
        budget_used += 1
        if candidate is not None and _is_valid_cable_candidate(candidate, anchor, (image_w, image_h)):
            cable = _expand_detection(candidate, (image_w, image_h), ratio=0.06)

    if port is None:
        port = _infer_port_hint(
            anchor=anchor,
            cable=cable,
            peripheral=peripheral,
            image_size=(image_w, image_h),
            target_label=scenario.target_port_label,
        )

    state, confidence, reason = _score_connection(anchor=anchor, cable=cable, port=port)
    _draw_validation_frame(
        frame_path=frame_path,
        annotated_path=annotated_path,
        state=state,
        anchor=anchor,
        cable=cable,
        port=port,
        peripheral=peripheral,
        note=reason,
    )
    return (
        FrameValidationResult(
            frame_index=frame_index,
            timestamp_seconds=round(timestamp_seconds, 2),
            state=state,
            confidence=confidence,
            frame_path=str(frame_path),
            annotated_path=str(annotated_path),
            anchor=_det_to_dict(anchor),
            peripheral=_det_to_dict(peripheral),
            cable=_det_to_dict(cable),
            port=_det_to_dict(port),
            reason=reason,
        ),
        budget_used,
    )


def _select_anchor_detection(detections: list[Detection], terms: list[str]) -> Detection | None:
    preferred = [
        det for det in detections
        if any(term in det.label.lower() for term in terms)
    ]
    return _select_best_detection(preferred, terms)


def _select_best_detection(detections: list[Detection], terms: list[str]) -> Detection | None:
    if not detections or not terms:
        return None
    best: Detection | None = None
    best_score = -1.0
    joined = " ".join(terms)
    for det in detections:
        label = det.label.lower()
        score = label_match_score(joined, label) + det.confidence * 20.0
        if any(term in label for term in terms):
            score += 12.0
        if score > best_score:
            best = det
            best_score = score
    return best


def _filter_valid_candidates(
    detections: list[Detection],
    anchor: Detection,
    image_size: tuple[int, int],
    target: str,
    source_key: str,
) -> list[Detection]:
    scope = "nearby" if "cable" in target or "wire" in target or "plug" in target else "local"
    zone = "any" if scope == "nearby" else "edge_band"
    valid: list[Detection] = []
    for det in detections:
        if _validate_candidate(
            candidate=det,
            target=target,
            scope=scope,
            zone=zone,
            anchor=anchor,
            image_size=image_size,
            source_key=source_key,
        ):
            valid.append(_expand_detection(det, image_size, ratio=0.04))
    return valid


def _select_best_by_anchor(
    detections: list[Detection],
    anchor: Detection,
    target_label: str,
) -> Detection | None:
    if not detections:
        return None
    best: Detection | None = None
    best_score = -1.0
    ax = (anchor.x1 + anchor.x2) / 2.0
    ay = (anchor.y1 + anchor.y2) / 2.0
    for det in detections:
        cx = (det.x1 + det.x2) / 2.0
        cy = (det.y1 + det.y2) / 2.0
        distance = abs(cx - ax) + abs(cy - ay)
        score = label_match_score(target_label, det.label.lower()) + det.confidence * 25.0 - distance / 40.0
        if score > best_score:
            best = det
            best_score = score
    return best


def _select_best_port(
    detections: list[Detection],
    cable: Detection | None,
    anchor: Detection,
    target_label: str,
) -> Detection | None:
    if not detections:
        return None
    best: Detection | None = None
    best_score = -1.0
    for det in detections:
        score = label_match_score(target_label, det.label.lower()) + det.confidence * 25.0
        if cable is not None:
            score += _intersection_score(det, cable) * 40.0
            score -= _center_distance(det, cable) / 30.0
        else:
            score -= _center_distance(det, anchor) / 35.0
        if score > best_score:
            best = det
            best_score = score
    return best


def _is_valid_port_candidate(
    candidate: Detection,
    anchor: Detection,
    image_size: tuple[int, int],
    target_label: str,
) -> bool:
    return _validate_candidate(
        candidate=candidate,
        target=target_label,
        scope="local",
        zone="edge_band",
        anchor=anchor,
        image_size=image_size,
        source_key="frame_gpt",
    )


def _is_valid_cable_candidate(
    candidate: Detection,
    anchor: Detection,
    image_size: tuple[int, int],
) -> bool:
    return _validate_candidate(
        candidate=candidate,
        target="cable",
        scope="nearby",
        zone="any",
        anchor=anchor,
        image_size=image_size,
        source_key="frame_gpt",
    )


def _score_connection(
    anchor: Detection,
    cable: Detection | None,
    port: Detection | None,
) -> tuple[str, float, str]:
    if cable is None and port is None:
        return "inconclusive", 0.0, "Cable and port were not detected"
    if cable is None:
        return "not_connected", 0.18, "Port found but cable not detected"
    if port is None:
        if _intersects_connectivity_ring(cable, anchor, ring_ratio=0.28):
            return "near", 0.46, "Cable is near the device edge, but the target port was not found"
        return "not_connected", 0.22, "Cable is visible but not near the expected port zone"

    is_hint = "(hint)" in port.label.lower()
    overlap = _intersection_score(cable, port)
    near_anchor = _intersects_connectivity_ring(cable, anchor, ring_ratio=0.28)
    port_on_edge = _port_sits_on_anchor_edge(port, anchor)
    center_gap = _center_distance(cable, port)

    if not is_hint and overlap > 0.04 and near_anchor and port_on_edge:
        confidence = min(0.98, 0.65 + overlap + max(0.0, 0.25 - center_gap / 250.0))
        return "connected", round(confidence, 2), "Cable overlaps the detected port on the device edge"
    if near_anchor and port_on_edge and center_gap < 140:
        confidence = min(0.8, 0.45 + max(0.0, 0.2 - center_gap / 400.0))
        if is_hint:
            return "near", round(confidence, 2), "Cable is close to the expected port area, but the port is only inferred"
        return "near", round(confidence, 2), "Cable is close to the correct port but not clearly inserted"
    return "not_connected", 0.28, "Cable and port were found, but they do not align as a plugged connection"


def _infer_port_hint(
    anchor: Detection,
    cable: Detection | None,
    peripheral: Detection | None,
    image_size: tuple[int, int],
    target_label: str,
) -> Detection | None:
    source = cable or peripheral
    if source is None:
        return None

    side = _nearest_anchor_side(source, anchor)
    aw = max(1, anchor.x2 - anchor.x1)
    ah = max(1, anchor.y2 - anchor.y1)
    width, height = image_size

    if side in {"left", "right"}:
        box_w = max(18, int(aw * 0.06))
        box_h = max(24, int(ah * 0.12))
        cy = int((source.y1 + source.y2) / 2.0)
        y1 = max(anchor.y1, cy - box_h // 2)
        y2 = min(anchor.y2, y1 + box_h)
        if side == "left":
            x1 = max(0, anchor.x1 - box_w // 3)
            x2 = min(width, x1 + box_w)
        else:
            x2 = min(width, anchor.x2 + box_w // 3)
            x1 = max(0, x2 - box_w)
    else:
        box_w = max(24, int(aw * 0.12))
        box_h = max(18, int(ah * 0.06))
        cx = int((source.x1 + source.x2) / 2.0)
        x1 = max(anchor.x1, cx - box_w // 2)
        x2 = min(anchor.x2, x1 + box_w)
        if side == "top":
            y1 = max(0, anchor.y1 - box_h // 3)
            y2 = min(height, y1 + box_h)
        else:
            y2 = min(height, anchor.y2 + box_h // 3)
            y1 = max(0, y2 - box_h)

    x1 = max(0, min(width - 12, x1))
    y1 = max(0, min(height - 12, y1))
    x2 = max(x1 + 12, min(width, x2))
    y2 = max(y1 + 12, min(height, y2))

    return Detection(
        label=f"{target_label} (hint)",
        confidence=0.18,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _nearest_anchor_side(subject: Detection, anchor: Detection) -> str:
    distances = {
        "left": abs(subject.x2 - anchor.x1),
        "right": abs(subject.x1 - anchor.x2),
        "top": abs(subject.y2 - anchor.y1),
        "bottom": abs(subject.y1 - anchor.y2),
    }
    return min(distances, key=distances.get)


def _center_distance(a: Detection, b: Detection) -> float:
    ax = (a.x1 + a.x2) / 2.0
    ay = (a.y1 + a.y2) / 2.0
    bx = (b.x1 + b.x2) / 2.0
    by = (b.y1 + b.y2) / 2.0
    return abs(ax - bx) + abs(ay - by)


def _intersection_score(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    base = max(1, min((a.x2 - a.x1) * (a.y2 - a.y1), (b.x2 - b.x1) * (b.y2 - b.y1)))
    return inter / base


def _port_sits_on_anchor_edge(port: Detection, anchor: Detection) -> bool:
    margin_x = max(20, int((anchor.x2 - anchor.x1) * 0.18))
    margin_y = max(20, int((anchor.y2 - anchor.y1) * 0.18))
    cx = (port.x1 + port.x2) / 2.0
    cy = (port.y1 + port.y2) / 2.0
    return (
        cx <= anchor.x1 + margin_x
        or cx >= anchor.x2 - margin_x
        or cy <= anchor.y1 + margin_y
        or cy >= anchor.y2 - margin_y
    )


def _draw_validation_frame(
    frame_path: Path,
    annotated_path: Path,
    state: str,
    anchor: Detection | None,
    cable: Detection | None,
    port: Detection | None,
    peripheral: Detection | None,
    note: str,
) -> None:
    colors = {
        "connected": (22, 163, 74),
        "near": (245, 158, 11),
        "not_connected": (220, 38, 38),
        "inconclusive": (71, 85, 105),
    }
    status_color = colors[state]

    with Image.open(frame_path) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        if anchor is not None:
            _draw_box(draw, anchor, (37, 99, 235), "host")
        if peripheral is not None:
            _draw_box(draw, peripheral, (168, 85, 247), peripheral.label[:20])
        if port is not None:
            _draw_box(draw, port, status_color if state != "not_connected" else (245, 158, 11), "port")
        if cable is not None:
            _draw_box(draw, cable, status_color, "cable")

        draw.rectangle([(16, 16), (740, 96)], fill=(255, 255, 255))
        draw.text((28, 28), f"state: {state}", fill=status_color)
        draw.text((28, 56), note[:88], fill=(15, 23, 42))
        marker_point = _marker_point(state=state, anchor=anchor, cable=cable, port=port, peripheral=peripheral)
        if state == "connected":
            _draw_tick(draw, x=700, y=56, color=status_color)
            if marker_point is not None:
                _draw_tick(draw, x=marker_point[0], y=marker_point[1], color=status_color)
        elif marker_point is not None:
            _draw_cross(draw, x=marker_point[0], y=marker_point[1], color=(220, 38, 38))

        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(annotated_path)


def _draw_box(draw: ImageDraw.ImageDraw, det: Detection, color: tuple[int, int, int], label: str) -> None:
    draw.rectangle([(det.x1, det.y1), (det.x2, det.y2)], outline=color, width=5)
    text_x = det.x1 + 6
    text_y = max(4, det.y1 - 24)
    draw.rectangle([(text_x - 4, text_y - 2), (text_x + 180, text_y + 18)], fill=color)
    draw.text((text_x, text_y), label[:22], fill=(255, 255, 255))


def _draw_tick(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int]) -> None:
    draw.ellipse([(x - 26, y - 26), (x + 26, y + 26)], fill=(255, 255, 255), outline=color, width=5)
    draw.line([(x - 12, y + 2), (x - 2, y + 14), (x + 16, y - 10)], fill=color, width=6)


def _draw_cross(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int]) -> None:
    draw.ellipse([(x - 26, y - 26), (x + 26, y + 26)], fill=(255, 255, 255), outline=color, width=5)
    draw.line([(x - 12, y - 12), (x + 12, y + 12)], fill=color, width=6)
    draw.line([(x - 12, y + 12), (x + 12, y - 12)], fill=color, width=6)


def _marker_point(
    state: str,
    anchor: Detection | None,
    cable: Detection | None,
    port: Detection | None,
    peripheral: Detection | None,
) -> tuple[int, int] | None:
    if state != "connected" and port is not None:
        return (int((port.x1 + port.x2) / 2.0), int((port.y1 + port.y2) / 2.0))
    if cable is not None and port is not None:
        return (
            int(((cable.x1 + cable.x2) / 2.0 + (port.x1 + port.x2) / 2.0) / 2.0),
            int(((cable.y1 + cable.y2) / 2.0 + (port.y1 + port.y2) / 2.0) / 2.0),
        )
    if port is not None:
        return (int((port.x1 + port.x2) / 2.0), int((port.y1 + port.y2) / 2.0))
    if cable is not None:
        return (int((cable.x1 + cable.x2) / 2.0), int((cable.y1 + cable.y2) / 2.0))
    if peripheral is not None:
        return (int((peripheral.x1 + peripheral.x2) / 2.0), int((peripheral.y1 + peripheral.y2) / 2.0))
    if anchor is not None:
        return (int(anchor.x2 - 40), int((anchor.y1 + anchor.y2) / 2.0))
    return None


def _select_best_frames(results: list[FrameValidationResult], limit: int = 4) -> list[dict[str, float | str | int]]:
    ranked = sorted(
        results,
        key=lambda r: (_state_rank(r.state), r.confidence, -r.timestamp_seconds),
        reverse=True,
    )
    return [
        {
            "frame_index": item.frame_index,
            "timestamp_seconds": item.timestamp_seconds,
            "state": item.state,
            "confidence": item.confidence,
            "annotated_path": item.annotated_path,
        }
        for item in ranked[:limit]
    ]


def _build_assistant_output(
    issue_text: str,
    scenario: ConnectivityScenario,
    results: list[FrameValidationResult],
) -> str:
    if not results:
        return f"I could not read any frames for: {issue_text}"

    connected = [r for r in results if r.state == "connected"]
    near = [r for r in results if r.state == "near"]
    best = max(results, key=lambda r: (_state_rank(r.state), r.confidence))

    if connected:
        strongest = max(connected, key=lambda r: r.confidence)
        return (
            f"For '{issue_text}', I found a valid {scenario.target_cable_label} to {scenario.target_port_label} connection. "
            f"The clearest connected frame is around {strongest.timestamp_seconds:.1f}s."
        )
    if near:
        strongest = max(near, key=lambda r: r.confidence)
        return (
            f"For '{issue_text}', I can see the {scenario.target_cable_label} close to the {scenario.target_port_label}, "
            f"but I cannot confirm it is plugged in correctly. Best evidence is around {strongest.timestamp_seconds:.1f}s."
        )
    return (
        f"For '{issue_text}', I could not confirm a working {scenario.target_cable_label} to {scenario.target_port_label} connection. "
        f"The strongest frame I found was at {best.timestamp_seconds:.1f}s and is still {best.state.replace('_', ' ')}."
    )


def _state_rank(state: str) -> int:
    ranks = {
        "connected": 3,
        "near": 2,
        "not_connected": 1,
        "inconclusive": 0,
    }
    return ranks.get(state, -1)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _det_to_dict(det: Detection | None) -> dict | None:
    if det is None:
        return None
    return {
        "label": det.label,
        "confidence": round(det.confidence, 4),
        "x1": det.x1,
        "y1": det.y1,
        "x2": det.x2,
        "y2": det.y2,
    }
