"""Scene-graph relation extraction.

We compute three families of relations on the *t* (final) frame:

* **Spatial**: ``next_to`` (centers within 0.10 norm distance), ``on_table``
  (object overlaps the SHARED band horizontally and is below the persons),
  ``in_front_of_wearer`` (object below the wearer y_min).
* **Possession**: ``held_by``: an object whose bbox overlaps a "hand" detection
  (or a person's hand zone — bottom 1/3 of the person bbox).
* **Cross-frame**: ``moved_to`` — same instance ID with center displacement
  > 0.15 norm. Useful evidence for Contextual scenes.

Relations are returned as `Relation(subject_id, predicate, object_id)`.
"""

from __future__ import annotations

from egoownership.schema import (
    BBox,
    FrameDetections,
    ObjectDetection,
    PersonDetection,
    Relation,
)


def _dist_centers(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _hand_zone_for(person: PersonDetection) -> BBox:
    """Approximate hand zone as the lower 40% of the person bbox, padded out."""
    bb = person.bbox
    h = bb.y_max - bb.y_min
    return BBox(
        x_min=max(0.0, bb.x_min - 0.05),
        y_min=max(0.0, bb.y_max - 0.4 * h),
        x_max=min(1.0, bb.x_max + 0.05),
        y_max=min(1.0, bb.y_max + 0.05),
    )


def spatial_relations(frame: FrameDetections) -> list[Relation]:
    rels: list[Relation] = []
    objs = frame.objects
    zones = frame.zones

    for i, a in enumerate(objs):
        if a.instance_id is None:
            continue
        for b in objs[i + 1 :]:
            if b.instance_id is None or b.instance_id == a.instance_id:
                continue
            d = _dist_centers(a.bbox, b.bbox)
            if d <= 0.10:
                rels.append(
                    Relation(
                        subject_id=a.instance_id,
                        object_id=b.instance_id,
                        predicate="next_to",
                        score=1.0 - d / 0.10,
                    )
                )

        cy = a.bbox.center[1]
        if zones is not None and cy >= zones.mine_y_min:
            rels.append(
                Relation(
                    subject_id=a.instance_id,
                    object_id="wearer",
                    predicate="in_front_of",
                )
            )
        if zones is not None and zones.shared_x_min <= a.bbox.center[0] <= zones.shared_x_max:
            rels.append(
                Relation(
                    subject_id=a.instance_id,
                    object_id="table_center",
                    predicate="on_shared_band",
                )
            )
    return rels


def possession_relations(frame: FrameDetections) -> list[Relation]:
    rels: list[Relation] = []

    # Hand-detection-based possession.
    hands = [o for o in frame.objects if "hand" in o.label.lower()]
    objects = [o for o in frame.objects if "hand" not in o.label.lower() and o.instance_id]
    for hand in hands:
        for obj in objects:
            if hand.bbox.iou(obj.bbox) > 0.05:
                rels.append(
                    Relation(
                        subject_id=obj.instance_id,
                        object_id=hand.instance_id or "hand",
                        predicate="held_by",
                        score=hand.bbox.iou(obj.bbox),
                        note="hand-bbox-overlap",
                    )
                )

    # Person-hand-zone possession.
    for person in frame.persons:
        if not person.person_id:
            continue
        zone = _hand_zone_for(person)
        for obj in objects:
            iou = zone.iou(obj.bbox)
            if iou > 0.10:
                rels.append(
                    Relation(
                        subject_id=obj.instance_id,
                        object_id=person.person_id,
                        predicate="held_by",
                        score=iou,
                        note="person-hand-zone",
                    )
                )
    return rels


def cross_frame_relations(frames: list[FrameDetections]) -> list[Relation]:
    """Detect ``moved_to`` events for instances that drift across frames."""
    if len(frames) < 2:
        return []
    first, last = frames[0], frames[-1]
    rels: list[Relation] = []
    last_by_id = {o.instance_id: o for o in last.objects if o.instance_id}
    for first_obj in first.objects:
        last_obj = last_by_id.get(first_obj.instance_id)
        if first_obj.instance_id is None or last_obj is None:
            continue
        d = _dist_centers(first_obj.bbox, last_obj.bbox)
        if d > 0.15:
            rels.append(
                Relation(
                    subject_id=first_obj.instance_id,
                    object_id=first_obj.instance_id,
                    predicate="moved_to",
                    score=d,
                    note=(
                        f"({first_obj.bbox.center[0]:.2f},{first_obj.bbox.center[1]:.2f})"
                        f" → ({last_obj.bbox.center[0]:.2f},{last_obj.bbox.center[1]:.2f})"
                    ),
                )
            )
    return rels


def build_scene_graph(frames: list[FrameDetections]) -> list[FrameDetections]:
    """Return a copy with per-frame ``relations`` populated, plus cross-frame
    ``moved_to`` rels appended to the *last* frame."""

    out = [fd.model_copy(deep=True) for fd in frames]
    for fd in out:
        rels: list[Relation] = []
        rels.extend(spatial_relations(fd))
        rels.extend(possession_relations(fd))
        fd.relations = rels
    if out:
        out[-1].relations.extend(cross_frame_relations(out))
    return out
