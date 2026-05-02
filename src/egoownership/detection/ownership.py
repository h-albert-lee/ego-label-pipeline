"""Rule-based ownership assignment with dynamic zones + instance tracking.

The classification cascade per object on each frame:

1. **held_by relations** — if a Relation says the object is ``held_by`` a
   person_id, label PERSON_k. If held_by the wearer's hand bbox, label MINE.
2. **person_zones** — if the bbox center sits inside a person's influence
   rectangle, label PERSON_k.
3. **wearer near zone** — if cy >= zones.mine_y_min, MINE.
4. **shared band** — central horizontal band, SHARED.
5. **depth refinement** — when ``mean_depth`` is available and < the
   wearer-depth band lower bound, override to MINE.
6. Fallback AMBIGUOUS.

Scene-level label is then derived per object instance via the (t-2, t-1, t)
trajectory: stable → that label, transition → the *final* label.
"""

from __future__ import annotations

from collections import Counter

from egoownership.config import OwnershipZones, TaxonomyConfig, load_config
from egoownership.detection.tracking import collect_instance_track
from egoownership.schema import (
    BBox,
    FrameDetections,
    FrameZones,
    ObjectDetection,
    OwnershipLabel,
    PersonDetection,
    Relation,
    SceneRecord,
)


def _ensure_zones(frame: FrameDetections, yaml_zones: OwnershipZones) -> FrameZones:
    if frame.zones is not None:
        return frame.zones
    return FrameZones(
        mine_y_min=yaml_zones.mine_near_y_min,
        shared_x_min=yaml_zones.shared_x_min,
        shared_x_max=yaml_zones.shared_x_max,
        derivation="static-yaml",
    )


def _classify_with_zones(
    det: ObjectDetection,
    zones: FrameZones,
    persons: list[PersonDetection],
    relations: list[Relation],
    yaml_zones: OwnershipZones,
    wearer_depth_band: tuple[float, float] | None = None,
) -> tuple[OwnershipLabel, list[str]]:
    """Return (label, evidence_list)."""
    evidence: list[str] = []

    if det.bbox.area < yaml_zones.min_bbox_area_ratio:
        return OwnershipLabel.AMBIGUOUS, ["bbox-too-small"]

    # 1. Possession relations.
    for rel in relations:
        if rel.predicate != "held_by" or rel.subject_id != det.instance_id:
            continue
        target = rel.object_id
        if target.startswith("person_"):
            evidence.append(f"held_by:{target}")
            return OwnershipLabel.PERSON_K, evidence
        if target in {"wearer", "hand"} or target.startswith("hand"):
            evidence.append(f"held_by:wearer({target})")
            return OwnershipLabel.MINE, evidence

    # 2. Person influence zones.
    for pid, zone in zones.person_zones.items():
        # Use IoU rather than pure containment so partial overlap still wins.
        iou = zone.iou(det.bbox)
        if iou > 0.05:
            evidence.append(f"in-person-zone:{pid}({iou:.2f})")
            return OwnershipLabel.PERSON_K, evidence

    cx, cy = det.bbox.center

    # 5. Depth shortcut to MINE if available.
    if (
        det.mean_depth is not None
        and wearer_depth_band is not None
        and det.mean_depth >= wearer_depth_band[0]
    ):
        evidence.append(f"depth-near({det.mean_depth:.2f})")
        return OwnershipLabel.MINE, evidence

    # 3. Wearer near zone.
    if cy >= zones.mine_y_min:
        evidence.append(f"y={cy:.2f}>=mine_y_min={zones.mine_y_min:.2f}")
        return OwnershipLabel.MINE, evidence

    in_shared_band = zones.shared_x_min <= cx <= zones.shared_x_max

    # Person-but-no-zone fallback: opposite half of the screen with persons present.
    if not in_shared_band and persons:
        nearest = min(persons, key=lambda p: abs(p.bbox.center[0] - cx))
        if abs(nearest.bbox.center[0] - cx) < 0.20 and nearest.person_id:
            evidence.append(f"nearest-person:{nearest.person_id}")
            return OwnershipLabel.PERSON_K, evidence

    # 4. Shared band.
    if in_shared_band:
        evidence.append(f"x={cx:.2f}∈[{zones.shared_x_min:.2f},{zones.shared_x_max:.2f}]")
        return OwnershipLabel.SHARED, evidence

    # 5. Legacy fallback for the no-person-detector case: an object that's
    # outside the wearer near zone, outside the shared band, and high-ish in
    # the frame is likely on the opposite side of the table → PERSON_k.
    if cy <= yaml_zones.person_far_y_max:
        evidence.append(f"y={cy:.2f}<=person_far_y_max={yaml_zones.person_far_y_max:.2f}")
        return OwnershipLabel.PERSON_K, evidence

    return OwnershipLabel.AMBIGUOUS, ["no-rule-fired"]


def assign_ownership(
    frames: list[FrameDetections],
    cfg: TaxonomyConfig | None = None,
    *,
    wearer_depth_bands: list[tuple[float, float] | None] | None = None,
) -> list[FrameDetections]:
    """Mutate detections with an ``ownership`` label for each object.

    ``wearer_depth_bands`` is one entry per frame (or None). If absent, depth
    is ignored.
    """

    cfg = cfg or load_config()
    out: list[FrameDetections] = []
    for i, fd in enumerate(frames):
        zones = _ensure_zones(fd, cfg.zones)
        depth_band = wearer_depth_bands[i] if wearer_depth_bands else None
        new_objs: list[ObjectDetection] = []
        for obj in fd.objects:
            label, evidence = _classify_with_zones(
                obj,
                zones,
                fd.persons,
                fd.relations,
                cfg.zones,
                wearer_depth_band=depth_band,
            )
            new_objs.append(
                obj.model_copy(update={"ownership": label, "ownership_evidence": evidence})
            )
        out.append(fd.model_copy(update={"objects": new_objs, "zones": zones}))
    return out


def scene_label_for_instance(
    frames: list[FrameDetections], instance_id: str
) -> tuple[OwnershipLabel, str, float]:
    """Derive a scene-level label by watching one instance across frames.

    Returns (label, reasoning_note, confidence_in_[0,1]).
    """

    track = collect_instance_track(frames, instance_id)
    if any(p is None for p in track):
        return (
            OwnershipLabel.AMBIGUOUS,
            f"missing instance {instance_id} in some frames",
            0.10,
        )

    labels = [p.ownership or OwnershipLabel.AMBIGUOUS for p in track]  # type: ignore[union-attr]
    if OwnershipLabel.AMBIGUOUS in labels:
        return (
            OwnershipLabel.AMBIGUOUS,
            "ambiguous per-frame (" + " → ".join(l.value for l in labels) + ")",
            0.20,
        )

    if labels[0] == labels[-1]:
        # Stable → confidence based on how often the label held across all frames.
        conf = labels.count(labels[-1]) / len(labels)
        return labels[-1], f"stable ({labels[-1].value})", conf
    # Transition → confidence 0.6 (not full 1.0 — final frame may be transient).
    return (
        labels[-1],
        f"transition {labels[0].value} → {labels[-1].value}",
        0.65,
    )


def _instance_to_class(track) -> str | None:
    for det in track:
        if det is not None:
            return det.label
    return None


def build_scene_record(clip, frames: list[FrameDetections]) -> SceneRecord:
    """Pick a salient instance and derive a scene-level label.

    Heuristic for picking the *target* instance:
    1. Match the clip's noun against any instance whose class starts with that noun.
    2. Otherwise, pick the highest-score non-hand instance present in the final frame.
    3. If multiple instances of the same class appear (Taxonomy D candidate),
       pick the one whose ownership in the final frame is *not* AMBIGUOUS first.
    """

    if not frames:
        return SceneRecord(
            clip=clip, frames=[], scene_label=OwnershipLabel.AMBIGUOUS, notes="no frames"
        )

    last = frames[-1]
    instance_pool: list[ObjectDetection] = sorted(
        [o for o in last.objects if o.instance_id and "hand" not in o.label.lower()],
        key=lambda o: -(o.score or 0.0),
    )

    # Prefer instances matching a clip noun.
    noun_instances: list[ObjectDetection] = []
    if clip.nouns:
        for noun in clip.nouns:
            for det in instance_pool:
                if noun in det.label.lower():
                    noun_instances.append(det)

    candidates = noun_instances or instance_pool
    if not candidates:
        return SceneRecord(
            clip=clip, frames=frames, scene_label=OwnershipLabel.AMBIGUOUS,
            notes="no target instance"
        )

    # Detect duplicate-class symmetry → Taxonomy D evidence.
    class_counts = Counter(det.label.split("_")[0] for det in last.objects if det.instance_id)
    duplicates = {cls for cls, n in class_counts.items() if n > 1}

    best_label = OwnershipLabel.AMBIGUOUS
    best_note = ""
    best_conf = 0.0
    chosen_instance_id: str | None = None
    for det in candidates:
        label, note, conf = scene_label_for_instance(frames, det.instance_id)  # type: ignore[arg-type]
        if conf > best_conf:
            best_label = label
            best_note = f"{det.instance_id}: {note}"
            best_conf = conf
            chosen_instance_id = det.instance_id

    notes = best_note
    if duplicates and chosen_instance_id and chosen_instance_id.split("_")[0] in duplicates:
        notes += f" | duplicate-{chosen_instance_id.split('_')[0]} present (Taxonomy D candidate)"

    return SceneRecord(
        clip=clip,
        frames=frames,
        scene_label=best_label,
        notes=notes,
        auto_label_confidence=best_conf,
    )
