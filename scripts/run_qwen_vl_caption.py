#!/usr/bin/env python3
"""Caption a SAM2-masked video clip with Qwen3-VL.

Standalone alternative to CAT-V's InternVL-based ``scripts/get_caption.py``,
used by ``run_catv_one_object.py`` when ``--captioner-backend qwen3vl`` is
passed. Must run under an environment whose transformers build supports
``Qwen3VLForConditionalGeneration`` (the InternVL/CAT-V ``test`` env does not;
use the ``sam2hf`` env instead).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image


def _resize_for_vlm(image: Image.Image, max_side: int = 448) -> Image.Image:
    """Bound the longer side so Qwen-VL's dynamic-resolution tiling doesn't
    explode the vision-token count. Without this, native-resolution EgoLife
    frames (e.g. 1408x1408) x many frames can stall prefill for many minutes."""
    width, height = image.size
    scale = max_side / max(width, height)
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))))


def extract_frames(
    video_path: Path, max_frames: int, max_side: int = 448
) -> tuple[list[Image.Image], list[float]]:
    """Return sampled frames plus each one's timestamp (sec, relative to clip start)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frames: list[Image.Image] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_resize_for_vlm(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), max_side))
        index += 1
    cap.release()
    if not frames:
        raise FileNotFoundError(f"No frames decoded from {video_path}")
    kept_indices = list(range(len(frames)))
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        kept_indices = [int(i * step) for i in range(max_frames)]
        frames = [frames[i] for i in kept_indices]
    timestamps = [i / fps for i in kept_indices]
    return frames, timestamps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames-num", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=448, help="Resize each frame's longer side to this many pixels")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.video = args.video.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )

    frames, frame_timestamps_sec = extract_frames(args.video, args.max_frames_num, args.max_side)
    frame_content: list[dict] = []
    for i, frame in enumerate(frames):
        frame_content.append({"type": "text", "text": f"Frame{i + 1}:"})
        frame_content.append({"type": "image", "image": frame})
    messages = [
        {
            "role": "user",
            "content": [*frame_content, {"type": "text", "text": args.question}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(args.device)

    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )[0].strip()
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        text = "<error_processing>"
        print(f"Error encountered: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"model_answer": text, "frame_timestamps_sec": frame_timestamps_sec},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
