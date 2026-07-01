#!/usr/bin/env python3
"""Run SAM-2 + Qwen3-VL captioning for one object bbox and write one caption JSON.

    --video <mp4> --bbox <x1,y1,x2,y2 txt> --out <json>

Requires: `sam2` pip-installed, and `run_qwen_vl_caption.py` in the same directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--first-frame", type=Path, default=None)
    parser.add_argument("--bbox", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--mask-model-path", required=True, help="Path to sam2.1_hiera_*.pt checkpoint")
    parser.add_argument("--max-frames-num", type=int, default=16)
    parser.add_argument("--catv-device", default="cuda:0")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--whole-video", action="store_true")
    parser.add_argument("--caption", default="")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--qwen-vl-python",
        default=sys.executable,
        help="Python interpreter for run_qwen_vl_caption.py (must have Qwen3-VL / transformers installed).",
    )
    args = parser.parse_args()

    args.video = args.video.expanduser().resolve()
    args.first_frame = args.first_frame.expanduser().resolve() if args.first_frame else None
    args.bbox = args.bbox.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.first_frame is not None and not args.first_frame.exists():
        raise FileNotFoundError(f"First-frame image not found: {args.first_frame}")
    if not args.bbox.exists():
        raise FileNotFoundError(f"BBox file not found: {args.bbox}")

    work_dir = args.work_dir.expanduser().resolve() if args.work_dir else (args.out.parent / f"{args.out.stem}_catv_work").resolve()
    if work_dir.exists() and not args.keep_work_dir:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    bbox_path = work_dir / "bbox.txt"
    shutil.copyfile(args.bbox, bbox_path)

    mask_model_path = Path(args.mask_model_path)
    if not mask_model_path.exists():
        raise FileNotFoundError(f"SAM-2 checkpoint not found: {mask_model_path}")

    catv_video = prepare_catv_video(
        args.video,
        work_dir,
        start_sec=max(0.0, args.start_sec),
        duration_sec=max(0.1, args.duration_sec),
        fps=max(0.1, args.fps),
        whole_video=args.whole_video,
    )
    prompt_frame_idx = int(round(max(0.0, args.start_sec) * max(0.1, args.fps))) if args.whole_video else 0
    mask_input_path = catv_video
    if args.first_frame is not None:
        assert_prompt_resolution_matches(args.first_frame, catv_video)
        mask_input_path, prompt_frame_idx = make_exact_prompt_frame_sequence(
            catv_video,
            args.first_frame,
            work_dir,
            prompt_frame_idx=prompt_frame_idx,
        )
    mask_input_stem = mask_input_path.name if mask_input_path.is_dir() else mask_input_path.stem
    masked_video = work_dir / f"{mask_input_stem}_mask.mp4"
    patched_get_masks = patch_get_masks_script(
        work_dir,
        args.catv_device,
        prompt_frame_idx=prompt_frame_idx,
        fps=args.fps,
    )
    run(
        [
            sys.executable,
            str(patched_get_masks),
            "--video_path",
            str(mask_input_path),
            "--txt_path",
            str(bbox_path),
            "--model_path",
            str(mask_model_path),
            "--video_output_path",
            str(work_dir),
            "--save_to_video",
            "True",
        ],
        cwd=work_dir,
    )
    if not masked_video.exists():
        raise FileNotFoundError(f"SAM-2 masked video was not created: {masked_video}")
    normalize_mp4_for_viewing(masked_video)

    final_json = work_dir / "catv_caption_full.json"
    qwen_vl_script = Path(__file__).resolve().parent / "run_qwen_vl_caption.py"
    run(
        [
            args.qwen_vl_python,
            str(qwen_vl_script),
            "--video",
            str(masked_video),
            "--question",
            _ownership_relevant_question(args.caption),
            "--model-path",
            args.model_path,
            "--device",
            args.catv_device,
            "--max-frames-num",
            str(args.max_frames_num),
            "--out",
            str(final_json),
        ],
        cwd=work_dir,
    )
    if not final_json.exists():
        raise FileNotFoundError(f"Caption JSON was not created: {final_json}")

    catv_output = json.loads(final_json.read_text(encoding="utf-8"))
    caption = extract_caption(catv_output)
    relative_timestamps = catv_output.get("frame_timestamps_sec") or []
    clip_origin_sec = 0.0 if args.whole_video else args.start_sec
    described_frame_timestamps_sec = [clip_origin_sec + t for t in relative_timestamps]

    args.out.write_text(
        json.dumps(
            {
                "object_caption": caption,
                "catv_output": catv_output,
                "masked_video": str(masked_video),
                "catv_input_video": str(catv_video),
                "catv_input_fps": args.fps,
                "catv_input_start_sec": args.start_sec,
                "catv_input_duration_sec": args.duration_sec,
                "catv_input_whole_video": args.whole_video,
                "catv_prompt_frame_idx": prompt_frame_idx,
                "described_frame_timestamps_sec": described_frame_timestamps_sec,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def run(cmd: list[str], *, cwd: Path) -> None:
    print("[run_catv_one_object]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def probe_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = result.stdout.strip()
    if not text:
        raise RuntimeError(f"ffprobe returned no duration for {video_path}")
    return float(text)


def prepare_catv_video(
    video_path: Path,
    work_dir: Path,
    *,
    start_sec: float,
    duration_sec: float,
    fps: float,
    whole_video: bool,
) -> Path:
    """Create a low-fps CAT-V input clip in chronological order."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "catv_input_fps1.mp4"

    if not whole_video:
        source_duration = probe_video_duration(video_path)
        if start_sec >= source_duration:
            raise RuntimeError(
                f"Requested clip start ({start_sec:.3f}s) is at or past the end of "
                f"{video_path} (duration {source_duration:.3f}s) — nothing to extract."
            )
        remaining = source_duration - start_sec
        if duration_sec > remaining:
            print(
                f"[run_catv_one_object] clamping duration {duration_sec:.3f}s -> {remaining:.3f}s "
                f"(start={start_sec:.3f}s exceeds available length of {video_path})",
                flush=True,
            )
            duration_sec = remaining

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if not whole_video:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(
        [
            "-i",
            str(video_path),
        ]
    )
    if not whole_video:
        cmd.extend(["-t", f"{duration_sec:.3f}"])
    cmd.extend(
        [
            "-vf",
            f"fps={fps:g}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(output),
        ]
    )
    print("[run_catv_one_object]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not output.exists() or output.stat().st_size == 0:
        raise FileNotFoundError(f"Failed to create CAT-V fps={fps:g} input clip: {output}")
    try:
        probe_video_resolution(output)
    except (subprocess.CalledProcessError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"CAT-V fps={fps:g} input clip has no decodable video frames: {output} "
            f"(requested start={start_sec:.3f}s, duration={duration_sec:.3f}s from {video_path})"
        ) from exc
    return output


def _ownership_relevant_question(target_hint: str = "") -> str:
    target_text = target_hint.strip()
    if target_text:
        target_text = f" The expected target object name is: {target_text}."
    return (
        "Describe the highlighted object HO in this video for an egocentric implicit-ownership task. "
        "Each video frame is labeled Frame1, Frame2, etc. — use those exact labels to answer question (4)."
        f"{target_text} Answer these four questions about HO, numbered exactly as below: "
        "(1) What is HO? Answer as 'HO: <short name>'. "
        "(2) Does HO's status change during the video, and what is HO's final status? Describe any change as "
        "'from <state> to <state>' (e.g. moved, picked up, placed, opened, given, returned). "
        "(3) Who interacts with HO, and how? Say 'camera wearer/ego hand' for the person wearing the camera, or "
        "'other person' for anyone else, rather than the generic word 'person', and describe the interaction "
        "(e.g. holds, picks up, places, gives, hands, passes, receives, borrows, returns HO). "
        "If the actor identity cannot be distinguished, say 'actor identity is ambiguous'. "
        "If HO's status never changes and nobody interacts with it, say so explicitly in (2)/(3). "
        "(4) Which frame numbers best show: HO's initial state, the key interaction moment, and HO's final "
        "state? Answer on its own line as exactly 'FIRST=<n> KEY=<n> FINAL=<n>' using the Frame numbers shown "
        "(e.g. 'FIRST=1 KEY=5 FINAL=8'). If unsure, give your best estimate rather than omitting it. "
        "Do not describe HO's appearance, surroundings, or unrelated objects beyond what is needed to answer "
        "these four questions."
    )


def _create_low_fps_clip(
    video_path: Path,
    output: Path,
    *,
    fps: float,
    start_sec: float,
    duration_sec: float | None,
) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_sec is not None:
        cmd.extend(["-t", f"{duration_sec:.3f}"])
    cmd.extend(
        [
            "-vf",
            f"fps={fps:g}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(output),
        ]
    )
    print("[run_catv_one_object]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not output.exists() or output.stat().st_size == 0:
        raise FileNotFoundError(f"Failed to create low-fps CAT-V clip: {output}")


def normalize_mp4_for_viewing(path: Path) -> None:
    """Rewrite an MP4 as H.264/yuv420p so VS Code/browser previews can open it."""
    if not path.exists() or path.suffix.lower() != ".mp4":
        return
    tmp = path.with_name(f"{path.stem}.h264{path.suffix}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        "-y",
        str(tmp),
    ]
    print("[run_catv_one_object]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(path)


def assert_prompt_resolution_matches(first_frame_path: Path, catv_video_path: Path) -> None:
    with Image.open(first_frame_path) as image:
        image_width, image_height = image.size
    video_width, video_height = probe_video_resolution(catv_video_path)
    if (image_width, image_height) != (video_width, video_height):
        raise RuntimeError(
            "SAM bbox frame resolution does not match CAT-V prompt video resolution: "
            f"first_frame={image_width}x{image_height}, "
            f"catv_video={video_width}x{video_height}. "
            "BBox coordinates would be applied to the wrong pixel scale."
        )


def make_exact_prompt_frame_sequence(
    catv_video_path: Path,
    first_frame_path: Path,
    work_dir: Path,
    *,
    prompt_frame_idx: int,
) -> tuple[Path, int]:
    """Create a jpg frame directory and replace the prompt frame with exact SAM frame."""
    frames_dir = work_dir / "catv_input_exact_prompt_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(catv_video_path))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    if frame_idx == 0:
        raise RuntimeError(f"No frames could be decoded from CAT-V input video: {catv_video_path}")
    prompt_frame_idx = max(0, min(prompt_frame_idx, frame_idx - 1))

    with Image.open(first_frame_path).convert("RGB") as image:
        image.save(frames_dir / f"{prompt_frame_idx:06d}.jpg", quality=95)
    return frames_dir, prompt_frame_idx


def probe_video_resolution(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = result.stdout.strip().splitlines()[0]
    width, height = text.split("x")
    return int(width), int(height)


def patch_get_masks_script(
    work_dir: Path, device: str, *, prompt_frame_idx: int, fps: float = 1.0
) -> Path:
    """Generate a self-contained SAM-2 bidirectional tracking script in work_dir."""
    autocast_expr = (
        f'torch.autocast("{device.split(":")[0]}", dtype=torch.float16)'
        if device.startswith("cuda")
        else "nullcontext()"
    )
    nullcontext_import = "" if device.startswith("cuda") else "from contextlib import nullcontext\n"
    text = f'''import argparse
import gc
import os
import os.path as osp
import shutil
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

{nullcontext_import}sys.path.insert(0, "./")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]


def load_txt(gt_path):
    with open(gt_path, "r") as f:
        lines = f.readlines()
    prompts = {{}}
    for fid, line in enumerate(lines):
        x_min, y_min, x_max, y_max = line.strip().split(",")
        prompts[fid] = ((int(x_min), int(y_min), int(x_max), int(y_max)), 0)
    return prompts


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError("Unknown model size in path!")


def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def load_frames(video_path):
    if osp.isdir(video_path):
        frame_paths = sorted(osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg"))
        frames = [cv2.imread(frame_path) for frame_path in frame_paths]
    else:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    if not frames:
        raise ValueError("No frames were loaded from the video.")
    return frames


def write_sequence_frames(frames, indices, folder):
    if osp.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    for seq_idx, original_idx in enumerate(indices):
        cv2.imwrite(osp.join(folder, f"{{seq_idx:06d}}.jpg"), frames[original_idx])
    return folder


def masks_to_vis(object_ids, masks):
    mask_to_vis = {{}}
    bbox_to_vis = {{}}
    for obj_id, mask in zip(object_ids, masks):
        mask = mask[0].cpu().numpy() > 0.0
        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        bbox_to_vis[obj_id] = bbox
        mask_to_vis[obj_id] = mask
    return mask_to_vis, bbox_to_vis


def draw_frame(frame, mask_to_vis, bbox_to_vis):
    img = frame.copy()
    height, width = img.shape[:2]
    for obj_id, mask in mask_to_vis.items():
        mask_img = np.zeros((height, width, 3), np.uint8)
        mask_img[mask] = color[(obj_id + 1) % len(color)]
        img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)
    for obj_id, bbox in bbox_to_vis.items():
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)
    return img


def track_sequence(predictor, frames_path, bbox, original_indices, rendered, loaded_frames):
    if not original_indices:
        return None
    state = predictor.init_state(frames_path, offload_video_to_cpu=True)
    predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
    for seq_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0, reverse=False):
        if seq_idx >= len(original_indices):
            continue
        original_idx = original_indices[seq_idx]
        mask_to_vis, bbox_to_vis = masks_to_vis(object_ids, masks)
        if 0 <= original_idx < len(loaded_frames):
            rendered[original_idx] = draw_frame(loaded_frames[original_idx], mask_to_vis, bbox_to_vis)
    return state


def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="{device}")
    predictor.fill_hole_area = 0
    prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)
    print(prompts)

    loaded_frames = load_frames(args.video_path)
    height, width = loaded_frames[0].shape[:2]
    prompt_frame_idx = min({prompt_frame_idx}, max(len(loaded_frames) - 1, 0))
    rendered = {{}}
    bbox, track_label = prompts[0]
    seq_root = osp.join(args.video_output_path, "_sam2_bidirectional_sequences")
    forward_indices = list(range(prompt_frame_idx, len(loaded_frames)))
    backward_indices = list(range(prompt_frame_idx, -1, -1))
    forward_path = write_sequence_frames(loaded_frames, forward_indices, osp.join(seq_root, "forward"))
    backward_path = write_sequence_frames(loaded_frames, backward_indices, osp.join(seq_root, "backward"))
    states = []

    with torch.inference_mode(), {autocast_expr}:
        states.append(track_sequence(predictor, backward_path, bbox, backward_indices, rendered, loaded_frames))
        states.append(track_sequence(predictor, forward_path, bbox, forward_indices, rendered, loaded_frames))

    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = args.video_output_path + f"/{{osp.basename(args.video_path).split('.')[0]}}_mask.mp4"
        # Match the container's declared fps to the real content rate ({fps:g} fps,
        # the rate frames were extracted at), so any later tool that reads this
        # video's fps metadata to compute timestamps gets a correct answer.
        out = cv2.VideoWriter(out_path, fourcc, {fps:g}, (width, height))
        for frame_idx, frame in enumerate(loaded_frames):
            out.write(rendered.get(frame_idx, frame))
        out.release()

    shutil.rmtree(seq_root, ignore_errors=True)
    for state in states:
        del state
    del predictor
    gc.collect()
    torch.clear_autocast_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="./assets/demo.mp4")
    parser.add_argument("--txt_path", default="./assets/demo.txt")
    parser.add_argument("--model_path", default="./checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--video_output_path", default="./results/")
    parser.add_argument("--save_to_video", default=True)
    main(parser.parse_args())
'''
    patched = work_dir / "get_masks_patched.py"
    patched.write_text(text, encoding="utf-8")
    return patched




def extract_caption(data) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        parts = [extract_caption(item) for item in data]
        return "\n".join(part for part in parts if part)
    if isinstance(data, dict):
        for key in ("object_caption", "model_answer", "caption", "answer", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            found = extract_caption(value)
            if found:
                return found
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
