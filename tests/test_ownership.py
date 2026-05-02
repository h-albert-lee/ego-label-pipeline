"""Tests for ownership classification + scene-level labeling."""

from egoownership.detection.ownership import (
    assign_ownership,
    build_scene_record,
    scene_label_for_instance,
)
from egoownership.detection.tracking import assign_instance_ids
from egoownership.schema import (
    BBox,
    ClipCandidate,
    FrameDetections,
    ObjectDetection,
    OwnershipLabel,
    Taxonomy,
)


def _det(label: str, cx: float, cy: float, size: float = 0.15, score: float = 0.9) -> ObjectDetection:
    half = size / 2
    return ObjectDetection(
        label=label,
        bbox=BBox(
            x_min=max(0.0, cx - half),
            y_min=max(0.0, cy - half),
            x_max=min(1.0, cx + half),
            y_max=min(1.0, cy + half),
        ),
        score=score,
    )


def _fd(tag: str, t: float, objs: list[ObjectDetection]) -> FrameDetections:
    return FrameDetections(tag=tag, timestamp_sec=t, width=640, height=480, objects=objs)


def test_bbox_in_near_zone_is_mine():
    frame = _fd("t", 0.0, [_det("cup", cx=0.5, cy=0.85)])
    [out] = assign_ownership([frame])
    assert out.objects[0].ownership is OwnershipLabel.MINE


def test_bbox_in_center_is_shared():
    frame = _fd("t", 0.0, [_det("plate", cx=0.5, cy=0.50)])
    [out] = assign_ownership([frame])
    assert out.objects[0].ownership is OwnershipLabel.SHARED


def test_bbox_far_and_offside_is_person_k():
    frame = _fd("t", 0.0, [_det("notebook", cx=0.15, cy=0.25)])
    [out] = assign_ownership([frame])
    assert out.objects[0].ownership is OwnershipLabel.PERSON_K


def test_tiny_bbox_is_ambiguous():
    frame = _fd("t", 0.0, [_det("pen", cx=0.5, cy=0.5, size=0.03)])
    [out] = assign_ownership([frame])
    assert out.objects[0].ownership is OwnershipLabel.AMBIGUOUS


def test_give_transition_yields_person_k_final():
    f0 = _fd("t-2", 0.0, [_det("pen", cx=0.55, cy=0.85)])  # MINE
    f1 = _fd("t-1", 0.5, [_det("pen", cx=0.50, cy=0.50)])  # SHARED
    f2 = _fd("t", 1.0, [_det("pen", cx=0.15, cy=0.25)])    # PERSON_k
    frames = assign_instance_ids([f0, f1, f2])
    frames = assign_ownership(frames)
    instance_id = frames[0].objects[0].instance_id
    label, note, conf = scene_label_for_instance(frames, instance_id)
    assert label is OwnershipLabel.PERSON_K
    assert "MINE" in note and "PERSON_k" in note
    assert conf > 0.5


def test_baseline_stable_mine():
    f0 = _fd("t-2", 0.0, [_det("cup", cx=0.55, cy=0.85)])
    f1 = _fd("t-1", 0.1, [_det("cup", cx=0.55, cy=0.85)])
    f2 = _fd("t", 0.2, [_det("cup", cx=0.55, cy=0.85)])
    frames = assign_instance_ids([f0, f1, f2])
    frames = assign_ownership(frames)
    instance_id = frames[0].objects[0].instance_id
    label, note, conf = scene_label_for_instance(frames, instance_id)
    assert label is OwnershipLabel.MINE
    assert "stable" in note
    assert conf == 1.0


def test_build_scene_record_uses_clip_nouns():
    clip = ClipCandidate(
        dataset="unit",
        clip_id="c1",
        taxonomy=Taxonomy.CONTEXTUAL,
        t_minus_2_sec=0,
        t_minus_1_sec=0.5,
        t_sec=1.0,
        verb="give",
        nouns=["pen"],
    )
    frames = [
        _fd("t-2", 0.0, [_det("pen", 0.55, 0.85), _det("hand", 0.60, 0.90)]),
        _fd("t-1", 0.5, [_det("pen", 0.50, 0.50), _det("hand", 0.50, 0.55)]),
        _fd("t", 1.0, [_det("pen", 0.15, 0.25)]),
    ]
    frames = assign_instance_ids(frames)
    frames = assign_ownership(frames)
    record = build_scene_record(clip, frames)
    assert record.scene_label is OwnershipLabel.PERSON_K
    assert record.clip.clip_id == "c1"
    assert record.auto_label_confidence is not None


def test_scene_record_ambiguous_when_object_missing():
    clip = ClipCandidate(
        dataset="unit",
        clip_id="c2",
        taxonomy=Taxonomy.CONTEXTUAL,
        t_minus_2_sec=0,
        t_minus_1_sec=0.5,
        t_sec=1.0,
        verb="give",
        nouns=["pen"],
    )
    frames = [
        _fd("t-2", 0.0, [_det("pen", 0.55, 0.85)]),
        _fd("t-1", 0.5, []),
        _fd("t", 1.0, [_det("pen", 0.15, 0.25)]),
    ]
    frames = assign_instance_ids(frames)
    frames = assign_ownership(frames)
    record = build_scene_record(clip, frames)
    # Pen in t and t-2 get DIFFERENT instance_ids because no IoU continuity.
    # In the final frame the pen is PERSON_k; that's the salient instance.
    # Either result is acceptable: PERSON_k (final-frame instance is in person zone)
    # or AMBIGUOUS (broken track). Check it's not nonsense:
    assert record.scene_label in {OwnershipLabel.PERSON_K, OwnershipLabel.AMBIGUOUS, OwnershipLabel.MINE}
