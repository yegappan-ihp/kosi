# kosi

Backend prototype for a hackathon assistant where a user provides:
- an image (for example: a watch on a cluttered table)
- a text issue description

The app detects objects, picks the best candidate from the text, crops only that object, and sends crop + text to an LLM for step-by-step instructions.

If detection is ambiguous, the app asks the LLM to choose the best object from detected candidates. Only if that still fails does it ask the user to pick one.

## Current mode: hybrid

- Local: YOLO-World open-vocabulary detection and image crop
- Cloud: OpenAI vision model for instruction generation

The code is structured so you can switch to local open-source VLM later without breaking hybrid dependencies.

## Project files

- `app.py`: CLI entry point
- `kosi_assist/detector.py`: YOLO object detection
- `kosi_assist/matcher.py`: issue text to detected object matching
- `kosi_assist/image_utils.py`: crop selected object
- `kosi_assist/llm_client.py`: backend abstraction (`openai` now, `local` scaffolded)
- `kosi_assist/pipeline.py`: full flow orchestration and JSON report output

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`:

```bash
OPENAI_API_KEY=...
```

Performance tuning (faster responses):

```bash
OPENAI_TIMEOUT_SECONDS=25
OPENAI_MAX_IMAGE_SIDE=1800
OPENAI_CONCURRENCY=2
KOSI_STEP_WORKERS=3
KOSI_MAX_STEP_TARGETS=4
```

The app also prints live progress as step highlight images finish.

Optional detection mode:

```bash
KOSI_DETECTOR_MODE=recognize_everything
KOSI_EVERYTHING_MAX_TAGS=24
RAM_CHECKPOINT_PATH=/absolute/path/to/ram_swin_large_14m.pth
```

In `recognize_everything` mode, the app enriches detection with Recognize-Anything tags (RAM when checkpoint is available, plus GPT visual tags) while keeping the same YOLO + GPT boxing flow.

## Run

Interactive mode:

```bash
python3 app.py
```

Args mode:

```bash
python3 app.py --image /path/to/photo.jpg --issue "watch button is not working"
```

Outputs are saved in `output_files/`:
- `output_files/crops/`: cropped target image
- `output_files/gpt_inputs/`: exact image sent to the GPT API
- `output_files/reports/`: JSON report with detections and instructions
- `reply/`: step images with rectangle boxes for found visual targets (folder is cleared each run)

Detection notes:
- Default model is `yolov8s-world.pt` to support more than fixed COCO-80 classes.
- The detector builds class prompts from user text plus electronics keywords (power bank, printer, router, watch, etc.).
- Ambiguous selections trigger GPT-based object picking before any user fallback.
- Electronics issue text gets stricter ranking that penalizes non-electronics detections.
- Visual targets are first searched in the cropped GPT image, then in the full original image.
- If YOLO cannot localize a visual target, GPT vision bbox fallback is used on cropped image and then full image.
- If a target cannot be found in either image, it is listed in terminal output as unfound.

## Future local-only mode

When you are ready for fully local open-source inference, install optional deps:

```bash
pip install -r requirements-local-vlm.txt
```

Then switch in `.env`:

```bash
KOSI_LLM_BACKEND=local
```

`local` backend is intentionally scaffolded but not fully implemented yet.
