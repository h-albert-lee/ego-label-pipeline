"""Re-running `egoown serve --input ...` must regenerate auto fields fresh but
never silently wipe review progress already recorded in scene_records.jsonl.
"""

from pathlib import Path

from egoownership.labeling_pipeline import apply_preserved_review_state, load_preserved_review_state
from egoownership.schema import AnnotationEdit, ClipCandidate, OwnershipLabel, SceneRecord, Taxonomy


def _record(clip_id: str, *, review_status="draft", edits=None, scene_label=None, notes=None) -> SceneRecord:
    clip = ClipCandidate(
        dataset="egolife", clip_id=clip_id, taxonomy=Taxonomy.BASELINE,
        t_minus_2_sec=0.0, t_minus_1_sec=1.0, t_sec=2.0,
    )
    return SceneRecord(
        clip=clip, frames=[], scene_label=scene_label, review_status=review_status,
        notes=notes, edits=edits or [],
    )


def test_load_preserved_review_state_skips_untouched_draft_records(tmp_path: Path):
    scenes = tmp_path / "scene_records.jsonl"
    untouched = _record("a")
    reviewed = _record("b", review_status="verified", edits=[
        AnnotationEdit(annotator="alice", field="review_status", old_value="draft", new_value="verified"),
    ])
    scenes.write_text(untouched.model_dump_json() + "\n" + reviewed.model_dump_json() + "\n")

    preserved = load_preserved_review_state(scenes)
    assert set(preserved) == {"b"}


def test_load_preserved_review_state_keeps_non_draft_even_without_edits(tmp_path: Path):
    # The bug this guards against: the sampling script sets review_status
    # directly (with edit history in practice, but the *status* itself is
    # what must never be silently reset even if edits were somehow empty).
    scenes = tmp_path / "scene_records.jsonl"
    rec = _record("c", review_status="in_review")
    scenes.write_text(rec.model_dump_json() + "\n")
    preserved = load_preserved_review_state(scenes)
    assert set(preserved) == {"c"}


def test_load_preserved_review_state_empty_when_file_missing(tmp_path: Path):
    assert load_preserved_review_state(tmp_path / "does_not_exist.jsonl") == {}


def test_apply_preserved_review_state_overlays_review_and_evidence_fields():
    old = _record("x", review_status="verified", notes="looks right",
                   scene_label=OwnershipLabel.MINE,
                   edits=[AnnotationEdit(annotator="alice", field="review_status", old_value="draft", new_value="verified")])
    old = old.model_copy(update={"auto_key_evidence": {"rationale": "human-edited rationale, matches MINE"}})
    fresh = _record("x", review_status="draft", scene_label=OwnershipLabel.PERSON_K)
    # fresh's clip/vlm fields should win; review fields AND evidence (which
    # the review UI's evidence panel writes into) should come from old --
    # otherwise a bare --input re-conversion silently discards any rationale/
    # selected_evidence/object_type/target_zone edit a human made.
    fresh = fresh.model_copy(update={"auto_key_evidence": {"rationale": "freshly auto-derived, stale after re-conversion"}})

    merged = apply_preserved_review_state(fresh, {"x": old})

    assert merged.review_status == "verified"
    assert merged.notes == "looks right"
    assert merged.scene_label == OwnershipLabel.MINE
    assert merged.edits == old.edits
    assert merged.auto_key_evidence == {"rationale": "human-edited rationale, matches MINE"}


def test_apply_preserved_review_state_passthrough_when_untouched():
    fresh = _record("y")
    merged = apply_preserved_review_state(fresh, {})
    assert merged is fresh
