"""Ego4D video lookup, scratch download, and centered subclip extraction."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from egoownership.catv_io import safe_path_part

DEFAULT_CLIP_WINDOW_SEC = 30.0


def default_scratch_root() -> Path:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "egoown"
    return Path("/scratch") / user / "ego4d"


def centered_clip_window(anchor_sec: float, window_sec: float = DEFAULT_CLIP_WINDOW_SEC) -> tuple[float, float, float]:
    """Return ``(window_start, window_end, duration)`` centered on ``anchor_sec``."""
    half = max(0.0, window_sec) / 2.0
    start = max(0.0, float(anchor_sec) - half)
    duration = max(0.0, float(window_sec))
    return start, start + duration, duration


def locate_ego4d_full_video(video_id: str, *roots: Path) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        candidates.append(root / f"{video_id}.mp4")
        candidates.extend(sorted(root.glob(f"{video_id}.mp4*")))
        nested = root / "v2" / "full_scale" / f"{video_id}.mp4"
        candidates.append(nested)
        nested_root = root / "v2" / "full_scale"
        if nested_root.exists():
            candidates.extend(sorted(nested_root.glob(f"{video_id}.mp4*")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def download_ego4d_video(video_uid: str, download_dir: Path) -> Path | None:
    """Download one Ego4D full-scale video via the official ``ego4d`` CLI."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Downloading Ego4D video to scratch: {video_uid} → {download_dir}")
    try:
        subprocess.run(
            [
                "ego4d",
                "--output_directory",
                str(download_dir),
                "--datasets",
                "full_scale",
                "--video_uids",
                video_uid,
                "--yes",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[error] Failed to download {video_uid}. CLI error: {exc.stderr}")
        return None
    except FileNotFoundError:
        print("[error] 'ego4d' command not found. Install with: pip install ego4d")
        return None
    return locate_ego4d_full_video(video_uid, download_dir)


def ensure_ego4d_subclip(
    full_video: Path,
    cache_path: Path,
    *,
    window_start_sec: float,
    window_duration_sec: float,
) -> Path | None:
    """Extract and cache a fixed-duration subclip with ffmpeg."""
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, window_start_sec):.3f}",
        "-i",
        str(full_video),
        "-t",
        f"{max(0.1, window_duration_sec):.3f}",
        "-c",
        "copy",
        "-y",
        str(cache_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return None
    return cache_path if cache_path.exists() and cache_path.stat().st_size > 0 else None


def ego4d_subclip_cache_path(
    scratch_root: Path,
    *,
    video_id: str,
    clip_id: str,
    window_start_sec: float,
) -> Path:
    clip_key = safe_path_part(clip_id or video_id)
    video_key = safe_path_part(video_id)
    return (
        Path(scratch_root)
        / "clips"
        / video_key
        / f"{clip_key}__{max(0.0, window_start_sec):.3f}.mp4"
    )
