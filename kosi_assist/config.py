from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    llm_backend: str
    openai_api_key: str | None
    openai_model: str
    yolo_model: str
    yolo_confidence: float
    crop_padding_ratio: float
    output_dir: Path
    reply_dir: Path
    local_vlm_model: str

    @classmethod
    def from_env(cls, output_dir: Path | None = None) -> "AppConfig":
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=repo_root / ".env")
        configured_output = os.getenv("KOSI_OUTPUT_DIR", "output_files").strip()
        if configured_output == "outputs":
            configured_output = "output_files"
        resolved_output = output_dir or Path(configured_output).expanduser()
        resolved_reply = Path(os.getenv("KOSI_REPLY_DIR", "reply")).expanduser()
        return cls(
            llm_backend=os.getenv("KOSI_LLM_BACKEND", "openai").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            yolo_model=os.getenv("YOLO_MODEL", "yolov8s-world.pt"),
            yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.25")),
            crop_padding_ratio=float(os.getenv("CROP_PADDING_RATIO", "0.15")),
            output_dir=resolved_output,
            reply_dir=resolved_reply,
            local_vlm_model=os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        )
