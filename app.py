from __future__ import annotations

import argparse
from pathlib import Path

from kosi_assist.config import AppConfig
from kosi_assist.pipeline import run_pipeline
from kosi_assist.video_validation import run_video_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kosi assistant: detect target object and generate guidance"
    )
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--issue", type=str, help="Problem description from user")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=0.5,
        help="Seconds between sampled video frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    issue_text = args.issue or input("Describe the problem: ").strip()

    config = AppConfig.from_env(
        output_dir=Path(args.out_dir).expanduser() if args.out_dir else None
    )

    if args.video:
        result = run_video_validation(
            video_path=Path(args.video).expanduser(),
            issue_text=issue_text,
            config=config,
            sample_every_seconds=args.sample_seconds,
        )
        print("\n--- Video Validation ---")
        print(f"Scenario: {result.scenario}")
        print(f"Sampled frames: {result.sampled_frames}")
        print(f"Summary: {result.summary}")
        print(f"Assistant output: {result.assistant_output}")
        if result.best_frames:
            print("Best frames:")
            for frame in result.best_frames:
                print(
                    "  - "
                    f"t={frame['timestamp_seconds']:.1f}s, "
                    f"state={frame['state']}, "
                    f"confidence={frame['confidence']:.2f}, "
                    f"file={frame['annotated_path']}"
                )
        print(f"Annotated frames: {result.annotated_dir}")
        print(f"Saved JSON: {result.report_path}")
        return

    image_path = args.image or input("Enter image path: ").strip()

    result = run_pipeline(
        image_path=Path(image_path).expanduser(),
        issue_text=issue_text,
        config=config,
        progress_callback=lambda message: print(f"[progress] {message}"),
    )

    print("\n--- Assistant Output ---")
    print(result.instructions)
    print("------------------------")
    detected_labels = ", ".join(
        f"{d.label} ({d.confidence:.2f})" for d in result.all_detections[:8]
    )
    print(f"Detected objects: {detected_labels}")
    print(f"Target object: {result.selected_detection.label}")
    print(f"Selection method: {result.selection_method}")
    print(f"Cropped image: {result.crop_path}")
    print(f"Image sent to GPT: {result.gpt_input_path}")
    print(f"Reply folder: {result.reply_dir}")
    if result.reply_images:
        print("Reply images:")
        for img in result.reply_images:
            print(f"  - {img}")
    else:
        print("Reply images: none")
    if result.unfound_targets:
        print(f"Unfound visual targets: {', '.join(result.unfound_targets)}")
    print(f"Saved run JSON: {result.report_path}")


if __name__ == "__main__":
    main()
