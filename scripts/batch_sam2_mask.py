#!/usr/bin/env python3
"""SAM-2 bidirectional tracking for a batch of objects.

Reads a JSONL of job descriptions, loads SAM-2 **once**, and runs bidirectional
mask propagation for every job without reloading the model. Compared to spawning
run_catv_one_object.py per-row, this eliminates the ~30s SAM-2 model-load
overhead that was previously paid for every object.

Runs in the CAT-V 'test' conda env (same env as run_catv_one_object.py).

Job JSONL format (one JSON object per line):
  {
    "job_id": "<record id>",
    "video_path": "/abs/path/to/video.mp4",
    "first_frame_path": "/abs/path/to/ref_frame.jpg",  // optional
    "bbox_str": "x1,y1,x2,y2",                        // absolute pixels
    "fps": 1.0,
    "start_sec": 0.0,
    "whole_video": true,
    "work_dir": "/abs/path/to/per-object/work/dir",
    "out_masked_video": "/abs/path/to/per-object/work/dir/sam2_masked.mp4"
  }

Output: writes the masked video at each job's ``out_masked_video`` path.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from run_catv_one_object import (  # noqa: E402
    make_exact_prompt_frame_sequence,
    normalize_mp4_for_viewing,
    prepare_catv_video,
)

_COLOR = [(255, 0, 0)]


def _determine_model_cfg(model_path: str) -> str:
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError(f"Unknown SAM-2 model size in path: {model_path}")


def _load_frames(video_or_dir: str) -> list:
    if osp.isdir(video_or_dir):
        paths = sorted(osp.join(video_or_dir, f) for f in os.listdir(video_or_dir) if f.endswith(".jpg"))
        frames = [cv2.imread(p) for p in paths]
    else:
        cap = cv2.VideoCapture(video_or_dir)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_or_dir}")
    return frames


def _write_sequence_frames(frames: list, indices: list[int], folder: str) -> str:
    if osp.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for seq_idx, orig_idx in enumerate(indices):
        cv2.imwrite(osp.join(folder, f"{seq_idx:06d}.jpg"), frames[orig_idx])
    return folder


def _masks_to_vis(object_ids, masks) -> tuple[dict, dict]:
    mask_to_vis: dict = {}
    bbox_to_vis: dict = {}
    for obj_id, mask in zip(object_ids, masks):
        mask_arr = mask[0].cpu().numpy() > 0.0
        nz = np.argwhere(mask_arr)
        if len(nz) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = nz.min(axis=0).tolist()
            y_max, x_max = nz.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        bbox_to_vis[obj_id] = bbox
        mask_to_vis[obj_id] = mask_arr
    return mask_to_vis, bbox_to_vis


def _draw_frame(frame, mask_to_vis: dict, bbox_to_vis: dict):
    img = frame.copy()
    h, w = img.shape[:2]
    for obj_id, mask_arr in mask_to_vis.items():
        overlay = np.zeros((h, w, 3), np.uint8)
        overlay[mask_arr] = _COLOR[(obj_id + 1) % len(_COLOR)]
        img = cv2.addWeighted(img, 1, overlay, 0.2, 0)
    for obj_id, bbox in bbox_to_vis.items():
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), _COLOR[obj_id % len(_COLOR)], 2)
    return img


def _process_one_job(predictor, job: dict, *, device: str) -> None:
    work_dir = Path(job["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(job["video_path"])
    fps = float(job.get("fps", 1.0))
    start_sec = float(job.get("start_sec", 0.0))
    whole_video = bool(job.get("whole_video", True))
    duration_sec = float(job.get("duration_sec", 10.0))

    catv_video = prepare_catv_video(
        video_path,
        work_dir,
        start_sec=start_sec,
        duration_sec=duration_sec,
        fps=fps,
        whole_video=whole_video,
    )

    first_frame_path_str = job.get("first_frame_path") or ""
    prompt_frame_idx = int(round(max(0.0, start_sec) * fps)) if whole_video else 0
    mask_input: Path = catv_video
    if first_frame_path_str and Path(first_frame_path_str).exists():
        mask_input, prompt_frame_idx = make_exact_prompt_frame_sequence(
            catv_video,
            Path(first_frame_path_str),
            work_dir,
            prompt_frame_idx=prompt_frame_idx,
        )

    x1, y1, x2, y2 = (int(v) for v in job["bbox_str"].split(","))
    bbox = (x1, y1, x2, y2)

    loaded_frames = _load_frames(str(mask_input))
    height, width = loaded_frames[0].shape[:2]
    prompt_frame_idx = max(0, min(prompt_frame_idx, len(loaded_frames) - 1))
    rendered: dict[int, any] = {}

    seq_root = work_dir / "_sam2_bidirectional_sequences"
    forward_indices = list(range(prompt_frame_idx, len(loaded_frames)))
    backward_indices = list(range(prompt_frame_idx, -1, -1))
    forward_path = _write_sequence_frames(loaded_frames, forward_indices, str(seq_root / "forward"))
    backward_path = _write_sequence_frames(loaded_frames, backward_indices, str(seq_root / "backward"))

    autocast_ctx = (
        torch.autocast(device.split(":")[0], dtype=torch.float16)
        if device.startswith("cuda")
        else __import__("contextlib").nullcontext()
    )
    states = []
    with torch.inference_mode(), autocast_ctx:
        for seq_path, indices in ((backward_path, backward_indices), (forward_path, forward_indices)):
            state = predictor.init_state(seq_path, offload_video_to_cpu=True)
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
            for seq_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0, reverse=False):
                if seq_idx >= len(indices):
                    continue
                orig_idx = indices[seq_idx]
                m2v, b2v = _masks_to_vis(object_ids, masks)
                if 0 <= orig_idx < len(loaded_frames):
                    rendered[orig_idx] = _draw_frame(loaded_frames[orig_idx], m2v, b2v)
            states.append(state)

    out_masked_video = Path(job["out_masked_video"])
    out_masked_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_masked_video), fourcc, fps, (width, height))
    for frame_idx, frame in enumerate(loaded_frames):
        writer.write(rendered.get(frame_idx, frame))
    writer.release()

    shutil.rmtree(seq_root, ignore_errors=True)
    for state in states:
        del state
    gc.collect()
    torch.clear_autocast_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    normalize_mp4_for_viewing(out_masked_video)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch SAM-2 bidirectional tracking (model loaded once)")
    parser.add_argument("--jobs", required=True, type=Path, help="JSONL of SAM-2 masking jobs")
    parser.add_argument("--catv-root", type=Path, default=Path("/home/jhlee/CAT-V"))
    parser.add_argument("--model-path", default=None, help="Path to sam2.1_hiera_*.pt checkpoint")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    catv_root = args.catv_root.resolve()
    sys.path.insert(0, str(catv_root))

    model_path = args.model_path or str(catv_root / "checkpoints" / "sam2.1_hiera_base_plus.pt")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"SAM-2 checkpoint not found: {model_path}")

    from sam2.build_sam import build_sam2_video_predictor

    model_cfg = _determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device=args.device)
    predictor.fill_hole_area = 0
    print(f"[batch_sam2_mask] SAM-2 loaded from {model_path}", flush=True)

    jobs = [json.loads(line) for line in args.jobs.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(jobs)
    print(f"[batch_sam2_mask] {total} jobs to process", flush=True)

    success = 0
    errors = 0
    for i, job in enumerate(jobs):
        job_id = job["job_id"]
        expected_out = Path(job["out_masked_video"])
        if expected_out.exists() and expected_out.stat().st_size > 0:
            print(f"[{i + 1}/{total}] skip (exists): {job_id}", flush=True)
            success += 1
            continue
        print(f"[{i + 1}/{total}] tracking: {job_id}", flush=True)
        try:
            _process_one_job(predictor, job, device=args.device)
            success += 1
            print(f"[{i + 1}/{total}] done: {job_id}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            print(f"[batch_sam2_mask] ERROR {job_id}: {exc}", flush=True)
            errors += 1

    print(f"[batch_sam2_mask] finished: {success} ok, {errors} errors / {total} total", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
