"""Tests for the dataset-native bbox path (no models required)."""

from pathlib import Path

from egoownership.detection.native_bbox import (
    detections_for_fho,
    detections_for_hd_epic,
    stage_native_detect,
)
from egoownership.datasets import iter_fho_candidates, iter_hd_epic_candidates
from egoownership.detection.ownership import assign_ownership, build_scene_record
from egoownership.detection.tracking import assign_instance_ids
from egoownership.schema import OwnershipLabel

FIXTURES = Path(__file__).parent / "fixtures"


def test_fho_native_extracts_bboxes_for_three_frames():
    out = detections_for_fho(FIXTURES / "fho_with_bboxes.json")
    key = next(iter(out))
    frames = out[key]
    assert len(frames) == 3
    cup_frame = frames[0]
    cup_objects = [o for o in cup_frame.objects if o.label == "cup"]
    hand_objects = [o for o in cup_frame.objects if o.label == "hand"]
    assert len(cup_objects) == 1
    assert len(hand_objects) == 1
    # bbox should be normalized to 0..1
    bb = cup_objects[0].bbox
    assert 0 <= bb.x_min < bb.x_max <= 1
    assert 0 <= bb.y_min < bb.y_max <= 1


def test_hd_epic_native_extracts_bboxes_per_track():
    out = detections_for_hd_epic(FIXTURES / "hd_epic_with_bboxes.json")
    key = next(iter(out))
    frames = out[key]
    assert len(frames) == 3
    notebooks = [o for fd in frames for o in fd.objects if o.label == "notebook"]
    assert len(notebooks) == 3
    # Track moves left in t (final frame), so center_x decreases.
    centers_x = [o.bbox.center[0] for o in notebooks]
    assert centers_x[0] > centers_x[-1]


def test_stage_native_detect_yields_one_record_per_clip():
    candidates = list(iter_fho_candidates(FIXTURES / "fho_with_bboxes.json"))
    records = list(stage_native_detect(candidates, FIXTURES / "fho_with_bboxes.json", "ego4d-fho"))
    assert len(records) == 1
    rec = records[0]
    assert "clip" in rec
    assert len(rec["frames"]) == 3
    assert rec["depth_bands"] == [None, None, None]


def test_native_path_to_scene_label_end_to_end_via_label_pipeline():
    """Native bboxes → ownership rule cascade → SceneRecord works without models."""
    from egoownership.schema import ClipCandidate, FrameDetections

    candidates = list(iter_fho_candidates(FIXTURES / "fho_with_bboxes.json"))
    records = list(stage_native_detect(candidates, FIXTURES / "fho_with_bboxes.json", "ego4d-fho"))
    rec = records[0]

    clip = ClipCandidate.model_validate(rec["clip"])
    frames = [FrameDetections.model_validate(fd) for fd in rec["frames"]]
    frames = assign_instance_ids(frames)
    frames = assign_ownership(frames)
    scene = build_scene_record(clip, frames)

    # The cup starts low (MINE) and ends high (PERSON_k area in default zones)
    # so we should at least get a non-AMBIGUOUS label.
    assert scene.scene_label in {OwnershipLabel.MINE, OwnershipLabel.SHARED, OwnershipLabel.PERSON_K}


def test_hd_epic_native_via_pipeline():
    candidates = list(iter_hd_epic_candidates(FIXTURES / "hd_epic_with_bboxes.json"))
    records = list(
        stage_native_detect(candidates, FIXTURES / "hd_epic_with_bboxes.json", "hd-epic")
    )
    assert len(records) == 1
    assert len(records[0]["frames"]) == 3
