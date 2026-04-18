from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StepTarget:
    step_number: int
    instruction: str
    visual_target: str


@dataclass(frozen=True)
class PipelineResult:
    selected_detection: Detection
    all_detections: list[Detection]
    selection_method: str
    step_targets: list[StepTarget]
    unfound_targets: list[str]
    reply_images: list[Path]
    reply_dir: Path
    crop_path: Path
    gpt_input_path: Path
    report_path: Path
    instructions: str
