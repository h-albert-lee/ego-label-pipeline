"""labels_row_to_scene_record's optional vlm-crosscheck merge (side-by-side,
never replacing the auto pipeline's own scene_label/auto_key_evidence)."""

from egoownership.catv_pipeline import labels_row_to_scene_record
from egoownership.schema import OwnershipLabel


def _row(**overrides):
    row = {
        "id": "demo:clip#obj0",
        "clip_id": "demo:clip",
        "video_id": "video_A",
        "dataset": "egolife",
        "verb": "pick_up",
        "nouns": ["phone"],
        "dense_caption_en": "test narration",
        "start_sec": 0.0,
        "end_sec": 2.0,
        "reference_frame_sec": 4.0,
        "object": {
            "label": "phone",
            "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
            "score": 0.9,
        },
        "auto_taxonomy": "B",
        "auto_ground_truth": "PERSON_k",
        "evidence": {
            "target_object": "phone",
            "object_type": "personal",
            "target_zone": "other_person_zone",
            "person_count": 1,
            "visible_other_person": True,
            "caption_cues": {},
            "relations": [],
        },
    }
    row.update(overrides)
    return row


def test_no_crosscheck_leaves_vlm_fields_empty():
    scene = labels_row_to_scene_record(_row())
    assert scene.vlm_judgements == {}
    assert scene.vlm_agreement_ratio is None
    assert scene.vlm_majority_label is None


def test_crosscheck_merges_alongside_auto_label_without_replacing_it():
    crosscheck = {
        "id": "demo:clip#obj0",
        "judges": {
            "claude-sonnet-4-6": {
                "label": "MINE",
                "object_type_evidence": "phone is personal",
                "zone_evidence": "near the wearer",
                "relation_graph_evidence": "wearer's hand is on it",
                "context_change_evidence": "unchanged across frames",
                "agrees": False,
            }
        },
        "agreement_ratio": 0.0,
        "majority_label": "MINE",
    }
    scene = labels_row_to_scene_record(_row(), crosscheck=crosscheck)

    # Auto pipeline's own label is untouched by the merge.
    assert scene.scene_label == OwnershipLabel.PERSON_K

    judge = scene.vlm_judgements["claude-sonnet-4-6"]
    assert judge.label == OwnershipLabel.MINE
    assert judge.agrees is False
    assert judge.zone_evidence == "near the wearer"
    assert scene.vlm_agreement_ratio == 0.0
    assert scene.vlm_majority_label == OwnershipLabel.MINE


def test_crosscheck_with_unparseable_judge_label_is_skipped_not_crashed():
    crosscheck = {
        "id": "demo:clip#obj0",
        "judges": {"broken-judge": {"label": "NOT_A_REAL_LABEL", "agrees": None}},
        "agreement_ratio": 0.0,
        "majority_label": "UNKNOWN",
    }
    scene = labels_row_to_scene_record(_row(), crosscheck=crosscheck)
    assert scene.vlm_judgements == {}
    assert scene.vlm_majority_label is None
