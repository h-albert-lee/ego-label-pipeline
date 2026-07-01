#!/usr/bin/env python3
"""Caption SAM-2-masked videos with Qwen3-VL — model loaded once for the whole batch.

Reads a JSONL of captioning jobs, loads Qwen3-VL-8B **once**, and captions every
masked video sequentially. Compared to spawning run_qwen_vl_caption.py per-row,
this eliminates the ~60s model-load overhead that was previously paid for every
object (183 rows × 60s ≈ 3 hours just in model I/O).

Runs in the 'sam2hf' conda env (same env as run_qwen_vl_caption.py).

Job JSONL format (one JSON object per line):
  {
    "job_id": "<record id>",
    "masked_video": "/abs/path/to/sam2_masked.mp4",
    "question": "Describe the highlighted object HO ...",
    "out_json": "/abs/path/to/work_dir/caption.json",
    "max_frames": 16,
    "max_side": 448,
    "max_new_tokens": 512
  }

Output: each job writes a JSON with {"model_answer": "...", "frame_timestamps_sec": [...]}
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import cv2
from PIL import Image


def _resize_for_vlm(image: Image.Image, max_side: int = 448) -> Image.Image:
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return image
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))))


def extract_frames(
    video_path: Path, max_frames: int, max_side: int = 448
) -> tuple[list[Image.Image], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    raw_frames: list[Image.Image] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(_resize_for_vlm(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), max_side))
    cap.release()
    if not raw_frames:
        raise FileNotFoundError(f"No frames decoded from {video_path}")
    kept_indices = list(range(len(raw_frames)))
    if len(raw_frames) > max_frames:
        step = len(raw_frames) / max_frames
        kept_indices = [int(i * step) for i in range(max_frames)]
        raw_frames = [raw_frames[i] for i in kept_indices]
    timestamps = [idx / fps for idx in kept_indices]
    return raw_frames, timestamps


def _caption_one(model, processor, job: dict, *, device: str) -> None:
    import torch

    masked_video = Path(job["masked_video"])
    question = job["question"]
    out_json = Path(job["out_json"])
    max_frames = int(job.get("max_frames", 16))
    max_side = int(job.get("max_side", 448))
    max_new_tokens = int(job.get("max_new_tokens", 512))

    frames, timestamps = extract_frames(masked_video, max_frames, max_side)
    frame_content: list[dict] = []
    for i, frame in enumerate(frames):
        frame_content.append({"type": "text", "text": f"Frame{i + 1}:"})
        frame_content.append({"type": "image", "image": frame})
    messages = [{"role": "user", "content": [*frame_content, {"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    except Exception as exc:
        traceback.print_exc()
        text = "<error_processing>"
        print(f"[batch_qwen_vl_caption] inference error: {exc}", flush=True)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"model_answer": text, "frame_timestamps_sec": timestamps}, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Qwen3-VL captioning (model loaded once)")
    parser.add_argument("--jobs", required=True, type=Path, help="JSONL of captioning jobs")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    print(f"[batch_qwen_vl_caption] loaded {args.model_path}", flush=True)

    jobs = [json.loads(line) for line in args.jobs.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(jobs)
    print(f"[batch_qwen_vl_caption] {total} jobs to caption", flush=True)

    success = 0
    errors = 0
    for i, job in enumerate(jobs):
        job_id = job["job_id"]
        out_json = Path(job["out_json"])
        if out_json.exists() and out_json.stat().st_size > 0:
            print(f"[{i + 1}/{total}] skip (exists): {job_id}", flush=True)
            success += 1
            continue
        print(f"[{i + 1}/{total}] captioning: {job_id}", flush=True)
        try:
            _caption_one(model, processor, job, device=args.device)
            success += 1
        except Exception as exc:
            traceback.print_exc()
            print(f"[batch_qwen_vl_caption] ERROR {job_id}: {exc}", flush=True)
            errors += 1

    print(f"[batch_qwen_vl_caption] finished: {success} ok, {errors} errors / {total} total", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
