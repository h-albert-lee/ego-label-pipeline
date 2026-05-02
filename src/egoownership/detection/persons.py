"""Person detection.

Reuses the Grounding DINO wrapper with a person-only prompt. We split this
into its own module because zone derivation depends on stable, deduplicated
person bboxes, not the raw detection list.
"""

from __future__ import annotations

from pathlib import Path

from egoownership.schema import BBox, PersonDetection


def _suppress_overlap(boxes: list[BBox], iou_thr: float = 0.5) -> list[int]:
    """Greedy NMS — return indices to keep, sorted by area descending."""
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i].area)
    keep: list[int] = []
    for i in order:
        ok = True
        for j in keep:
            if boxes[i].iou(boxes[j]) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def detect_persons(image_path: Path, *, score_threshold: float = 0.30) -> list[PersonDetection]:
    """Detect persons in a single frame using Grounding DINO."""

    from egoownership.detection.grounding_dino import DinoConfig, detect_objects

    cfg = DinoConfig(box_threshold=score_threshold, text_threshold=score_threshold * 0.8)
    raw = detect_objects(image_path, "a person.", cfg)

    boxes = [d.bbox for d in raw]
    keep = _suppress_overlap(boxes, iou_thr=0.5)
    persons: list[PersonDetection] = []
    # Sort kept boxes left→right so person_1 is consistently the leftmost.
    kept = sorted(keep, key=lambda i: boxes[i].center[0])
    for rank, idx in enumerate(kept, start=1):
        persons.append(
            PersonDetection(
                bbox=boxes[idx],
                person_id=f"person_{rank}",
                score=raw[idx].score,
                is_camera_wearer=False,
            )
        )
    return persons


def assign_person_ids_across_frames(
    frame_persons: list[list[PersonDetection]], iou_thr: float = 0.30
) -> list[list[PersonDetection]]:
    """Greedy IoU-based identity propagation across the (t-2, t-1, t) sequence.

    Returns a copy where person_id is consistent across frames.
    """
    if not frame_persons:
        return frame_persons

    out = [list(fp) for fp in frame_persons]
    # Seed identities from the first non-empty frame.
    seed_idx = next((i for i, fp in enumerate(out) if fp), None)
    if seed_idx is None:
        return out

    next_id = max((int(p.person_id.split("_")[1]) for p in out[seed_idx] if p.person_id), default=0)

    for frame in out:
        for p in frame:
            # Try to match against the most recent established frame.
            best_match = None
            best_iou = 0.0
            for prev in out[seed_idx]:
                iou = p.bbox.iou(prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = prev
            if best_match and best_iou >= iou_thr:
                p.person_id = best_match.person_id
            else:
                next_id += 1
                p.person_id = f"person_{next_id}"
    return out
