"""Sparse frame extraction at ``(t-2, t-1, t)`` using imageio-ffmpeg.

We prefer ``imageio`` over shelling out to ``ffmpeg`` because it surfaces
errors as Python exceptions. For large jobs, a pure-``ffmpeg`` subprocess path
is faster; keep both to let the caller pick.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from egoownership.schema import ClipCandidate

if TYPE_CHECKING:  # imageio is an optional dep; only imported at runtime
    import imageio.v3 as iio  # noqa: F401


def _output_path(out_dir: Path, cand: ClipCandidate, tag: str) -> Path:
    safe_clip = cand.clip_id.replace("/", "_").replace(":", "_")
    return out_dir / cand.dataset / cand.video_id / f"{safe_clip}__{tag}.jpg"


def _frame_times(cand: ClipCandidate) -> list[tuple[str, float]]:
    return [
        ("t-2", cand.t_minus_2_sec),
        ("t-1", cand.t_minus_1_sec),
        ("t", cand.t_sec),
    ]


def extract_with_imageio(
    cand: ClipCandidate, video_path: Path, out_dir: Path
) -> dict[str, Path]:
    """Extract the three sparse frames using imageio.v3."""

    import imageio.v3 as iio  # local import: optional dep

    results: dict[str, Path] = {}
    # Probe fps to translate seconds → frame indices; imageio does this for us
    # via the ``index`` parameter when we ask in seconds.
    meta = iio.immeta(str(video_path))
    fps = float(meta.get("fps") or 0.0) or 30.0
    for tag, t in _frame_times(cand):
        idx = max(0, int(round(t * fps)))
        frame = iio.imread(str(video_path), index=idx, plugin="pyav")
        dest = _output_path(out_dir, cand, tag)
        dest.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(dest), frame)
        results[tag] = dest
    return results


def extract_with_ffmpeg(
    cand: ClipCandidate, video_path: Path, out_dir: Path, *, ffmpeg_bin: str = "ffmpeg"
) -> dict[str, Path]:
    """Extract the three frames by invoking ``ffmpeg`` directly.

    Each frame is seeked with ``-ss`` before ``-i`` for the fast path, then a
    single ``-frames:v 1`` dump. Uses JPEG to keep the frames small.
    """

    results: dict[str, Path] = {}
    for tag, t in _frame_times(cand):
        dest = _output_path(out_dir, cand, tag)
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(dest),
        ]
        subprocess.run(cmd, check=True)
        results[tag] = dest
    return results


def extract_sparse_frames(
    cand: ClipCandidate,
    video_path: Path,
    out_dir: Path,
    *,
    backend: str = "ffmpeg",
) -> dict[str, Path]:
    """Dispatch to the selected backend and return a tag→path mapping."""

    if backend == "imageio":
        return extract_with_imageio(cand, video_path, out_dir)
    if backend == "ffmpeg":
        return extract_with_ffmpeg(cand, video_path, out_dir)
    raise ValueError(f"Unknown frame backend: {backend!r}")


def extract_sparse_frames_for_candidate_inmemory(
    cand: ClipCandidate,
    video_path: Path,
    *,
    backend: str = "ffmpeg",
) -> "dict[str, np.ndarray]":
    """Extract the candidate's sparse ``(t-2, t-1, t)`` frames in memory.

    This is the no-cache companion to :func:`extract_sparse_frames`. It keeps
    VLM prefiltering from creating temporary JPEGs when callers do not request a
    frame cache directory.
    """

    import io
    import numpy as np
    import imageio.v3 as iio

    frames: dict[str, np.ndarray] = {}

    if backend == "imageio":
        meta = iio.immeta(str(video_path), plugin="pyav")
        fps = float(meta.get("fps") or 0.0) or 30.0
        for tag, t in _frame_times(cand):
            idx = max(0, int(round(t * fps)))
            frames[tag] = iio.imread(str(video_path), index=idx, plugin="pyav")
        return frames

    if backend == "ffmpeg":
        for tag, t in _frame_times(cand):
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ]
            result = subprocess.run(cmd, check=True, capture_output=True)
            frames[tag] = iio.imread(io.BytesIO(result.stdout))
        return frames

    raise ValueError(f"Unknown frame backend: {backend!r}")


def save_inmemory_frames(
    frames: "dict[str, np.ndarray]",
    out_dir: Path,
    dataset: str,
    video_id: str,
    clip_id: str,
) -> dict[str, Path]:
    """Save frames from :func:`extract_sparse_frames_inmemory` to JPEG files.

    Follows the same directory layout as :func:`extract_sparse_frames`:
    ``<out_dir>/<dataset>/<video_id>/<safe_clip_id>__<tag>.jpg``

    Returns a tag → path mapping.
    """

    import imageio.v3 as iio

    safe_clip = clip_id.replace("/", "_").replace(":", "_")
    results: dict[str, Path] = {}
    for tag, arr in frames.items():
        dest = out_dir / dataset / video_id / f"{safe_clip}__{tag}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(dest), arr)
        results[tag] = dest
    return results


def extract_sparse_frames_inmemory(
    t_sec: float,
    video_path: Path,
    *,
    n_each: int = 5,
    step_sec: float = 1.0,
    backend: str = "ffmpeg",
) -> "dict[str, np.ndarray]":
    """Extract frames around *t_sec* into memory as numpy arrays (no disk I/O).

    Samples *n_each* steps before and *n_each* steps after *t_sec* (plus the
    frame at *t_sec* itself), with each step spaced *step_sec* seconds apart.
    At the default ``step_sec=1.0`` this yields frames at
    ``t-5s, t-4s, ..., t, ..., t+4s, t+5s``.

    Returns a dict keyed by offset label (e.g. ``"-5"``, ``"-4"``, ...,
    ``"0"``, ..., ``"+5"``).  Offsets that fall before the start of the video
    are clamped to frame 0.
    """

    import io
    import numpy as np
    import imageio.v3 as iio

    # Probe FPS once.
    meta = iio.immeta(str(video_path), plugin="pyav")
    fps = float(meta.get("fps") or 0.0) or 30.0
    center_idx = max(0, int(round(t_sec * fps)))

    offsets = list(range(-n_each, n_each + 1))  # e.g. [-5, -4, ..., 0, ..., +5]
    step_frames = max(1, int(round(step_sec * fps)))
    frame_indices = {off: max(0, center_idx + off * step_frames) for off in offsets}

    frames: dict[str, np.ndarray] = {}

    if backend == "imageio":
        for off, idx in frame_indices.items():
            tag = f"{off:+d}" if off != 0 else "0"
            frames[tag] = iio.imread(str(video_path), index=idx, plugin="pyav")
        return frames

    if backend == "ffmpeg":
        for off, idx in frame_indices.items():
            tag = f"{off:+d}" if off != 0 else "0"
            t = idx / fps
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ]
            result = subprocess.run(cmd, check=True, capture_output=True)
            frames[tag] = iio.imread(io.BytesIO(result.stdout))
        return frames

    raise ValueError(f"Unknown frame backend: {backend!r}")


def extract_frame_by_number(
    video_path: Path,
    frame_number: int,
    *,
    fps: float = 30.0,
    backend: str = "ffmpeg",
) -> "np.ndarray":
    """Extract a single frame by its 0-based frame index.

    Returns an HxWx3 uint8 RGB numpy array.
    """
    import io
    import numpy as np
    import imageio.v3 as iio

    if backend == "imageio":
        return iio.imread(str(video_path), index=frame_number, plugin="pyav")

    t = frame_number / fps
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{t:.6f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True)
    return iio.imread(io.BytesIO(result.stdout))
