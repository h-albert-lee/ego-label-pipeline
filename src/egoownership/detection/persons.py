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


def detect_hands(image_path: Path, *, score_threshold: float = 0.25) -> list[BBox]:
    """Detect hands in a frame with a dedicated Grounding DINO query.

    Separate from ``detect_persons`` because a hand is what actually grounds a
    possession claim: the person-body box only ever gave an approximate "hand
    zone" (its lower 40%), and the wearer's own hand was inferred from a
    faceless ``"a person."`` hit rather than ever being detected as a hand.

    Returns bare boxes — attributing each one to the wearer or to a specific
    person is the caller's job, since that needs the tracked person ids.
    """
    from egoownership.detection.grounding_dino import DinoConfig, detect_objects

    cfg = DinoConfig(box_threshold=score_threshold, text_threshold=score_threshold * 0.8)
    boxes = [d.bbox for d in detect_objects(image_path, "a hand.", cfg)]
    return [boxes[i] for i in _suppress_overlap(boxes, iou_thr=0.5)]


def attribute_hands(
    hands: list[BBox],
    persons: list[PersonDetection],
    *,
    edge_margin: float = 0.02,
    containment_thr: float = 0.5,
) -> list[tuple[str, BBox]]:
    """Attach each detected hand to its owner: ``person_<n>`` or ``"wearer"``.

    A hand mostly inside a visible person's body box belongs to that person.
    Otherwise, a hand entering through a frame border with no owning body is
    the camera wearer's own — the same reasoning the ego-hand heuristic uses,
    but applied to a real hand detection instead of a person-shaped box.
    Hands matching neither are dropped rather than guessed at.
    """
    owned: list[tuple[str, BBox]] = []
    for hand in hands:
        best_id: str | None = None
        best_containment = containment_thr
        for person in persons:
            if not person.person_id or hand.area <= 0:
                continue
            ix1, iy1 = max(hand.x_min, person.bbox.x_min), max(hand.y_min, person.bbox.y_min)
            ix2, iy2 = min(hand.x_max, person.bbox.x_max), min(hand.y_max, person.bbox.y_max)
            containment = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / hand.area
            if containment > best_containment:
                best_containment = containment
                best_id = person.person_id
        if best_id is not None:
            owned.append((best_id, hand))
            continue
        touches_border = (
            hand.y_max >= 1.0 - edge_margin
            or hand.x_min <= edge_margin
            or hand.x_max >= 1.0 - edge_margin
        )
        if touches_border:
            owned.append(("wearer", hand))
    return owned


def detect_persons(
    image_path: Path,
    *,
    score_threshold: float = 0.30,
    face_score_threshold: float = 0.20,
) -> tuple[list[PersonDetection], BBox | None, list[BBox]]:
    """Detect the other people, the wearer's arm, and the hands in one frame.

    Grounding DINO's ``"a person."`` prompt frequently fires on the camera
    wearer's own hand/arm reaching into an egocentric frame, which would then
    be treated as another visible person by zone derivation. A second (cheap —
    same cached model, just another forward pass) query for ``"a human face."``
    tells them apart: boxes with a face inside are always kept as real persons;
    among the faceless remainder, only the single most ego-hand-positioned one
    (touching a frame border, lowest in frame) is dropped — other faceless
    boxes are kept, since a real person can have an occluded or out-of-frame
    face.

    Returns ``(persons, ego_hand_bbox, hands)``. ``hands`` are real ``"a hand."``
    detections (see ``detect_hands``), left unattributed for the caller.
    ``ego_hand_bbox`` is the dropped wearer-arm box, kept because possession
    falls back to it on frames where the hand query finds nothing — that
    approximation is what produced the currently published labels.
    """

    from egoownership.detection.grounding_dino import DinoConfig, detect_objects

    cfg = DinoConfig(box_threshold=score_threshold, text_threshold=score_threshold * 0.8)
    raw = detect_objects(image_path, "a person.", cfg)

    boxes = [d.bbox for d in raw]
    keep = _suppress_overlap(boxes, iou_thr=0.5)
    # Sort kept boxes left→right so person_1 is consistently the leftmost.
    kept = sorted(keep, key=lambda i: boxes[i].center[0])

    ego_hand_idx: int | None = None
    if kept:
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
    return persons, ego_hand_bbox, detect_hands(image_path)


def assign_person_ids_across_frames(
    frame_persons: list[list[PersonDetection]], iou_thr: float = 0.30
) -> list[list[PersonDetection]]:
    """Greedy IoU identity propagation across the (t-2, t-1, t) sequence.

    ``detect_persons`` numbers people independently per frame (left→right), so
    ``person_1`` at t-2 and ``person_1`` at t are not necessarily the same
    individual — anyone shifting position renumbers everyone. This links each
    frame's detections to the previous frame's, so a given id follows one
    person across the sequence and ``held_by`` edges can be compared over time.

    Matching is one-to-one and greedy by descending IoU: two detections in the
    same frame can never inherit the same id, and a person who leaves and
    re-enters gets a fresh id rather than silently reusing a stale one.

    Returns new ``PersonDetection`` objects; the inputs are left untouched.
    """
    out: list[list[PersonDetection]] = []
    prev_tracks: list[PersonDetection] = []
    next_id = 0

    for persons in frame_persons:
        pairs = sorted(
            (
                (p.bbox.iou(prev.bbox), cur_idx, prev_idx)
                for cur_idx, p in enumerate(persons)
                for prev_idx, prev in enumerate(prev_tracks)
            ),
            key=lambda t: -t[0],
        )
        assigned: dict[int, str] = {}
        used_prev: set[int] = set()
        for iou, cur_idx, prev_idx in pairs:
            if iou < iou_thr:
                break
            if cur_idx in assigned or prev_idx in used_prev:
                continue
            assigned[cur_idx] = prev_tracks[prev_idx].person_id or ""
            used_prev.add(prev_idx)

        tracked: list[PersonDetection] = []
        for cur_idx, person in enumerate(persons):
            person_id = assigned.get(cur_idx)
            if not person_id:
                next_id += 1
                person_id = f"person_{next_id}"
            tracked.append(person.model_copy(update={"person_id": person_id}))

        out.append(tracked)
        # Carry unmatched prior tracks forward so a person briefly occluded at
        # t-1 reclaims their id at t instead of being renumbered.
        prev_tracks = tracked + [
            prev for prev_idx, prev in enumerate(prev_tracks) if prev_idx not in used_prev
        ]

    return out
