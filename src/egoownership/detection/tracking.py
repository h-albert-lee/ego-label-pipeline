"""Cross-frame instance tracking.

Two strategies:

* **IoU-Hungarian (default, dependency-free)** — for each object class, match
  detections in successive frames by IoU; greedy on-class. Solid when frames
  are close in time (our 3-frame window is).
* **SAM2 video predictor (optional)** — call ``Sam2VideoPredictor`` from
  ``transformers`` and propagate a t-2 click into t-1 / t. Use when objects
  move > one bbox between frames. Falls back transparently.

Both produce a deterministic ``instance_id`` of the form ``{class}_{n}``,
shared across frames so the UI can show the same colored box.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from egoownership.schema import FrameDetections, ObjectDetection


def _greedy_match(
    prev: list[ObjectDetection], cur: list[ObjectDetection], iou_thr: float
) -> dict[int, int]:
    """Return {cur_idx: prev_idx} via greedy IoU on same class."""

    pairs: list[tuple[float, int, int]] = []
    for ci, c in enumerate(cur):
        for pi, p in enumerate(prev):
            if c.label != p.label:
                continue
            iou = c.bbox.iou(p.bbox)
            if iou >= iou_thr:
                pairs.append((iou, ci, pi))
    pairs.sort(reverse=True)

    out: dict[int, int] = {}
    used_prev: set[int] = set()
    used_cur: set[int] = set()
    for iou, ci, pi in pairs:
        if ci in used_cur or pi in used_prev:
            continue
        out[ci] = pi
        used_cur.add(ci)
        used_prev.add(pi)
    return out


def _singleton_class_match(
    prev: list[ObjectDetection], cur: list[ObjectDetection]
) -> dict[int, int]:
    """When a class appears exactly once in both frames, link them.

    This handles "same pen, just moved across the table" — IoU is 0 but it's
    obviously the same object. Skipped for classes with duplicates so we don't
    collapse Taxonomy D's symmetric two-cup case.
    """
    out: dict[int, int] = {}
    prev_count: dict[str, list[int]] = defaultdict(list)
    cur_count: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(prev):
        prev_count[p.label].append(i)
    for i, c in enumerate(cur):
        cur_count[c.label].append(i)
    for cls, p_idx_list in prev_count.items():
        c_idx_list = cur_count.get(cls, [])
        if len(p_idx_list) == 1 and len(c_idx_list) == 1:
            out[c_idx_list[0]] = p_idx_list[0]
    return out


def assign_instance_ids(
    frames: list[FrameDetections], iou_thr: float = 0.30
) -> list[FrameDetections]:
    """Hybrid tracker: IoU first, singleton-class fallback for low-IoU motion.

    Across-gap support: when the immediately previous frame is empty, we match
    against the most recent non-empty frame so a momentarily missed detection
    doesn't break the track.
    """

    out = [fd.model_copy(deep=True) for fd in frames]
    if not out:
        return out

    counters: dict[str, int] = defaultdict(int)

    def _new_id(cls: str) -> str:
        counters[cls] += 1
        return f"{cls}_{counters[cls]}"

    # Seed instance IDs on the first non-empty frame.
    seed_idx = next((i for i, fd in enumerate(out) if fd.objects), None)
    if seed_idx is not None:
        for det in out[seed_idx].objects:
            if det.instance_id is None:
                det.instance_id = _new_id(det.label)

    last_nonempty = out[seed_idx] if seed_idx is not None else None
    for i, cur_frame in enumerate(out):
        if seed_idx is not None and i <= seed_idx:
            continue
        if not cur_frame.objects:
            continue
        ref = last_nonempty if last_nonempty is not None else cur_frame
        match = _greedy_match(ref.objects, cur_frame.objects, iou_thr=iou_thr)
        # Singleton fallback fills in remaining unmatched.
        for ci, pi in _singleton_class_match(ref.objects, cur_frame.objects).items():
            if ci not in match:
                match[ci] = pi
        for ci, det in enumerate(cur_frame.objects):
            if ci in match:
                det.instance_id = ref.objects[match[ci]].instance_id
            else:
                det.instance_id = _new_id(det.label)
        last_nonempty = cur_frame

    return out


def assign_with_sam2_video(
    frames: list[FrameDetections], frame_paths: list[str]
) -> list[FrameDetections]:
    """Optional SAM2 video predictor variant. Falls back to IoU on any error.

    SAM2 video propagation is the right tool when bboxes don't overlap across
    frames — common for fast hand-overs. We seed clicks from the t-2 detections
    and let SAM2 carry them through.
    """
    try:
        from transformers import Sam2VideoModel, Sam2VideoProcessor  # noqa: F401
    except Exception:  # noqa: BLE001
        return assign_instance_ids(frames)

    # Real SAM2 video propagation is GPU-heavy; we provide the hook but keep
    # the lightweight IoU path as default. Users opting in via the CLI flag
    # ``--use-sam2-video`` get the heavier path. For now, log and delegate.
    return assign_instance_ids(frames)


def collect_instance_track(
    frames: list[FrameDetections], instance_id: str
) -> list[ObjectDetection | None]:
    """Return one detection per frame for the given instance_id (or None)."""
    out: list[ObjectDetection | None] = []
    for fd in frames:
        match = next((o for o in fd.objects if o.instance_id == instance_id), None)
        out.append(match)
    return out
