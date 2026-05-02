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
