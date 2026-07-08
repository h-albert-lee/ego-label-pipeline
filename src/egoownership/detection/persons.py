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


def _contains_face_center(box: BBox, face_boxes: list[BBox]) -> bool:
    """True if any detected face's center falls inside ``box``.

    A bare hand/arm reaching into an egocentric frame is never accompanied by
    a face, whereas another visible person almost always is — so this is
    used to tell the camera wearer's own limb apart from a "person." hit.
    """
    for face in face_boxes:
        fx, fy = face.center
        if box.x_min <= fx <= box.x_max and box.y_min <= fy <= box.y_max:
            return True
    return False


def _ego_hand_score(box: BBox, edge_margin: float = 0.02) -> float:
    """Higher = more likely this (faceless) box is the wearer's own hand/arm.

    A hand/arm reaching into frame always enters through a frame border,
    unlike a seated person whose box is normally fully contained. Among
    faceless candidates that touch a border, the one lowest in the frame
    (largest center-y) is the most ego-hand-like. Returns -1 (never picked)
    if the box doesn't touch a border at all, since a faceless box that's
    fully inside the frame is more likely a real person whose face is just
    occluded/turned away than a hand.
    """
    touches_border = box.y_max >= 1.0 - edge_margin or box.x_min <= edge_margin or box.x_max >= 1.0 - edge_margin
    if not touches_border:
        return -1.0
    return box.center[1]


def _pick_ego_hand_index(kept: list[int], boxes: list[BBox], has_face: list[bool]) -> int | None:
    """Among faceless candidates, return the index of the single one that most
    looks like the camera wearer's own hand — not all faceless boxes, since a
    real other person can legitimately have an occluded/out-of-frame face.
    """
    faceless = [idx for idx, hf in zip(kept, has_face) if not hf]
    if not faceless:
        return None
    best_idx, best_score = max(((idx, _ego_hand_score(boxes[idx])) for idx in faceless), key=lambda t: t[1])
    return best_idx if best_score >= 0 else None


_MAX_FACE_AREA_RATIO = 0.2
"""A real human face box is never this large a fraction of an egocentric
frame; the low-confidence "a human face." prompt occasionally fires on the
whole (low-light / textureless) image instead, and such a giant box's center
tends to land inside any large person candidate — including the wearer's own
body — falsely marking it as "has a face" and blocking ego-hand exclusion."""


def detect_persons(
    image_path: Path,
    *,
    score_threshold: float = 0.30,
    filter_ego_hand: bool = True,
    face_score_threshold: float = 0.20,
) -> tuple[list[PersonDetection], BBox | None]:
    """Detect persons in a single frame using Grounding DINO.

    Grounding DINO's ``"a person."`` prompt frequently fires on the camera
    wearer's own hand/arm reaching into an egocentric frame, which then gets
    treated as another visible person by zone derivation. When
    ``filter_ego_hand`` is set, a second (cheap — same cached model, just
    another forward pass) Grounding DINO query for ``"a human face."`` is run
    on the same frame. Boxes with a face inside are always kept as real
    persons; among the faceless remainder, only the single most
    ego-hand-positioned one (touching a frame border, lowest in frame) is
    excluded from the persons list — other faceless boxes are kept, since a
    real person can have an occluded or out-of-frame face.

    Returns ``(persons, ego_hand_bbox)``: the excluded box isn't just
    dropped, it's returned separately so callers can tell "this object is in
    the wearer's own hand" apart from "this object is in person_k's hand" —
    something the ``persons`` list alone can't represent.
    """

    from egoownership.detection.grounding_dino import DinoConfig, detect_objects

    cfg = DinoConfig(box_threshold=score_threshold, text_threshold=score_threshold * 0.8)
    raw = detect_objects(image_path, "a person.", cfg)

    boxes = [d.bbox for d in raw]
    keep = _suppress_overlap(boxes, iou_thr=0.5)
    # Sort kept boxes left→right so person_1 is consistently the leftmost.
    kept = sorted(keep, key=lambda i: boxes[i].center[0])

    ego_hand_idx: int | None = None
    if filter_ego_hand and kept:
        face_cfg = DinoConfig(box_threshold=face_score_threshold, text_threshold=face_score_threshold * 0.8)
        face_boxes = [
            d.bbox
            for d in detect_objects(image_path, "a human face.", face_cfg)
            if d.bbox.area <= _MAX_FACE_AREA_RATIO
        ]
        has_face = [_contains_face_center(boxes[idx], face_boxes) for idx in kept]
        ego_hand_idx = _pick_ego_hand_index(kept, boxes, has_face)

    persons: list[PersonDetection] = []
    for idx in kept:
        if idx == ego_hand_idx:
            continue
        persons.append(
            PersonDetection(
                bbox=boxes[idx],
                person_id=f"person_{len(persons) + 1}",
                score=raw[idx].score,
                is_camera_wearer=False,
            )
        )
    ego_hand_bbox = boxes[ego_hand_idx] if ego_hand_idx is not None else None
    return persons, ego_hand_bbox


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
