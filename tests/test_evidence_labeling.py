from egoownership.evidence_labeling import (
    _caption_cues,
    _caption_interaction_evidence,
    _decide_taxonomy_gt,
    _gt_from_past_frame_clue,
    _held_by_wearer_relation,
    _is_ego4d_row,
    _narration_actor_prefix,
    _summarize_context_change,
)
from egoownership.schema import BBox


def _base_evidence(zone: str, temporal: dict) -> dict:
    return {
        "target_object": "plate",
        "object_type": "generic_object",
        "target_zone": zone,
        "caption_cues": {
            "ego_actor": False,
            "other_actor": False,
            "ambiguous": False,
            "shared_use": False,
            "conflict_cue": False,
            "ownership_history": False,
        },
        "relations": [],
        "person_count": 0,
        "temporal": temporal,
    }


_ROW = {"verb": "", "object_caption": "The plate sits somewhere, unclear who it belongs to."}


def test_past_frame_fallback_uses_ego_zone_clue():
    temporal = {
        "frame_snapshots": {
            "t-1": {"target_zone": "ego_zone", "held_by": None, "relations": []},
            "t": {"target_zone": "background_or_ambiguous_zone", "held_by": None, "relations": []},
        },
        "contextual_requires_history": True,
    }
    decision = _decide_taxonomy_gt(_ROW, _base_evidence("background_or_ambiguous_zone", temporal))
    assert decision["ground_truth"] == "MINE"
    assert decision["taxonomy"] == "C"


def test_past_frame_fallback_uses_other_person_zone_clue():
    temporal = {
        "frame_snapshots": {
            "t-1": {"target_zone": "other_person_zone", "held_by": None, "relations": []},
            "t": {"target_zone": "background_or_ambiguous_zone", "held_by": None, "relations": []},
        },
        "contextual_requires_history": True,
    }
    decision = _decide_taxonomy_gt(_ROW, _base_evidence("background_or_ambiguous_zone", temporal))
    assert decision["ground_truth"] == "PERSON_k"
    assert decision["taxonomy"] == "C"


def test_past_frame_fallback_prefers_t_minus_1_over_t_minus_2():
    temporal = {
        "frame_snapshots": {
            "t-2": {"target_zone": "other_person_zone", "held_by": None, "relations": []},
            "t-1": {"target_zone": "ego_zone", "held_by": None, "relations": []},
            "t": {"target_zone": "background_or_ambiguous_zone", "held_by": None, "relations": []},
        },
        "contextual_requires_history": True,
    }
    label, tag = _gt_from_past_frame_clue(temporal)
    assert (label.value, tag) == ("MINE", "t-1")


def test_no_past_frame_clue_stays_ambiguous():
    temporal = {"frame_snapshots": {}, "contextual_requires_history": False}
    decision = _decide_taxonomy_gt(_ROW, _base_evidence("background_or_ambiguous_zone", temporal))
    assert decision["ground_truth"] == "AMBIGUOUS"
    assert decision["taxonomy"] == "D"


def test_current_frame_ego_zone_not_overridden_by_past_clue():
    # Current frame already resolves via zone == ego_zone; past-frame
    # fallback must not run (it only fires when gt would otherwise be
    # AMBIGUOUS), even if the past frame implies a different owner.
    temporal = {
        "frame_snapshots": {
            "t-1": {"target_zone": "other_person_zone", "held_by": None, "relations": []},
            "t": {"target_zone": "ego_zone", "held_by": None, "relations": []},
        },
        "contextual_requires_history": False,
    }
    decision = _decide_taxonomy_gt(_ROW, _base_evidence("ego_zone", temporal))
    assert decision["ground_truth"] == "MINE"


def _row_with_verb(verb: str) -> dict:
    return {"verb": verb, "object_caption": f"HO is {verb}ed between people."}


def test_tier0_shared_type_outranks_ambiguous_caption():
    # A shared-function object stays SHARED even if the caption explicitly
    # says actor identity is ambiguous — its label doesn't depend on who's
    # currently touching it.
    evidence = _base_evidence("background_or_ambiguous_zone", {})
    evidence["object_type"] = "shared"
    evidence["caption_cues"]["ambiguous"] = True
    row = {"verb": "", "object_caption": "The actor identity is ambiguous here."}
    decision = _decide_taxonomy_gt(row, evidence)
    assert decision["ground_truth"] == "SHARED"
    assert decision["taxonomy"] == "A"


def test_transfer_verb_ego_zone_yields_mine():
    evidence = _base_evidence("ego_zone", {})
    evidence["object_type"] = "personal"
    decision = _decide_taxonomy_gt(_row_with_verb("give"), evidence)
    assert decision["ground_truth"] == "MINE"
    assert decision["taxonomy"] == "C"


def test_transfer_verb_other_person_zone_yields_person_k():
    evidence = _base_evidence("other_person_zone", {})
    evidence["object_type"] = "personal"
    decision = _decide_taxonomy_gt(_row_with_verb("return"), evidence)
    assert decision["ground_truth"] == "PERSON_k"
    assert decision["taxonomy"] == "C"


def test_temporary_use_verb_ego_zone_inverts_to_person_k():
    # Wearer currently holds it (ego_zone) but the verb is "borrow" —
    # holding is not owning, so it belongs to someone else.
    evidence = _base_evidence("ego_zone", {})
    evidence["object_type"] = "personal"
    decision = _decide_taxonomy_gt(_row_with_verb("borrow"), evidence)
    assert decision["ground_truth"] == "PERSON_k"
    assert decision["taxonomy"] == "C"


def test_temporary_use_verb_other_person_zone_inverts_to_mine():
    # Another person currently holds it, but the verb is "lend" — it's the
    # wearer's own item, out on loan.
    evidence = _base_evidence("other_person_zone", {})
    evidence["object_type"] = "personal"
    decision = _decide_taxonomy_gt(_row_with_verb("lend"), evidence)
    assert decision["ground_truth"] == "MINE"
    assert decision["taxonomy"] == "C"


def test_history_verb_falls_through_to_relation_when_zone_inconclusive():
    # Verb present but zone gives no signal (shared/background) — should
    # fall through to the relation-graph tier rather than getting stuck.
    evidence = _base_evidence("shared_zone", {})
    evidence["object_type"] = "personal"
    evidence["relations"] = [{"predicate": "held_by", "object_id": "person_1", "subject_id": "target"}]
    decision = _decide_taxonomy_gt(_row_with_verb("give"), evidence)
    assert decision["ground_truth"] == "PERSON_k"
    assert decision["taxonomy"] == "C"  # verb still marks it contextual


def test_relation_graph_checked_before_plain_actor_cue():
    # held_by another person should win even if the caption also names the
    # wearer as ego_actor — Tier 1 (relation) is checked before the plain
    # actor-cue tier when there's no history verb.
    evidence = _base_evidence("shared_zone", {})
    evidence["object_type"] = "personal"
    evidence["caption_cues"]["ego_actor"] = True
    evidence["relations"] = [{"predicate": "held_by", "object_id": "person_2", "subject_id": "target"}]
    decision = _decide_taxonomy_gt({"verb": "", "object_caption": "cue"}, evidence)
    assert decision["ground_truth"] == "PERSON_k"


def test_abandonment_keeps_mine_for_personal_object_moved_to_shared_zone():
    evidence = _base_evidence("shared_zone", {"ownership_trajectory": "ego_to_shared"})
    evidence["object_type"] = "personal"
    decision = _decide_taxonomy_gt({"verb": "", "object_caption": "no actor mentioned"}, evidence)
    assert decision["ground_truth"] == "MINE"
    assert decision["taxonomy"] == "C"


def test_personal_object_in_shared_zone_without_trajectory_stays_ambiguous():
    evidence = _base_evidence("shared_zone", {})
    evidence["object_type"] = "personal"
    decision = _decide_taxonomy_gt({"verb": "", "object_caption": "no actor mentioned"}, evidence)
    assert decision["ground_truth"] == "AMBIGUOUS"
    assert decision["taxonomy"] == "D"


def test_held_by_wearer_relation_fires_on_full_containment():
    target = BBox(x_min=0.40, y_min=0.40, x_max=0.60, y_max=0.60)
    ego_hand = BBox(x_min=0.35, y_min=0.35, x_max=0.65, y_max=0.65)
    rel = _held_by_wearer_relation(target, ego_hand)
    assert rel == {"subject_id": "target", "object_id": "wearer", "predicate": "held_by", "score": rel["score"]}
    assert rel["score"] == 1.0


def test_held_by_wearer_relation_fires_on_small_target_inside_large_loose_hand_box():
    # Regression case: the ego-hand box from "a person." firing on a reaching
    # arm is often much larger than the held object (e.g. spans half the
    # frame), so plain IoU would be tiny even though the object sits fully
    # inside it. Containment should still catch this.
    target = BBox(x_min=0.538, y_min=0.489, x_max=0.619, y_max=0.647)
    ego_hand = BBox(x_min=0.265, y_min=0.495, x_max=0.852, y_max=0.993)
    assert target.iou(ego_hand) < 0.05  # confirms IoU alone would have missed it
    rel = _held_by_wearer_relation(target, ego_hand)
    assert rel is not None
    assert rel["object_id"] == "wearer"


def test_held_by_wearer_relation_none_when_no_overlap():
    target = BBox(x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1)
    ego_hand = BBox(x_min=0.8, y_min=0.8, x_max=0.9, y_max=0.9)
    assert _held_by_wearer_relation(target, ego_hand) is None


def test_held_by_wearer_relation_none_when_no_ego_hand_detected():
    target = BBox(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6)
    assert _held_by_wearer_relation(target, None) is None


def test_tier1_held_by_wearer_yields_mine_not_person_k():
    # Distinct from a plain zone==ego_zone match: this is specifically the
    # relation-graph tier recognizing the dedicated ego-hand box.
    evidence = _base_evidence("shared_zone", {})
    evidence["object_type"] = "personal"
    evidence["relations"] = [{"predicate": "held_by", "object_id": "wearer", "subject_id": "target"}]
    decision = _decide_taxonomy_gt({"verb": "", "object_caption": "cue"}, evidence)
    assert decision["ground_truth"] == "MINE"


def test_tier1_wearer_relation_wins_even_when_listed_after_a_bystander_relation():
    # Regression: the wearer relation must be prioritized regardless of list
    # order, not just "whichever held_by relation comes first" — a nearby
    # bystander's loose hand-zone can incidentally overlap the target too.
    evidence = _base_evidence("shared_zone", {})
    evidence["object_type"] = "personal"
    evidence["relations"] = [
        {"predicate": "held_by", "object_id": "person_4", "subject_id": "target"},
        {"predicate": "held_by", "object_id": "wearer", "subject_id": "target"},
    ]
    decision = _decide_taxonomy_gt({"verb": "", "object_caption": "cue"}, evidence)
    assert decision["ground_truth"] == "MINE"


def test_past_frame_fallback_recognizes_wearer_held_by():
    temporal = {
        "frame_snapshots": {
            "t-1": {"target_zone": "shared_zone", "held_by": "wearer", "relations": []},
            "t": {"target_zone": "background_or_ambiguous_zone", "held_by": None, "relations": []},
        },
        "contextual_requires_history": True,
    }
    label, tag = _gt_from_past_frame_clue(temporal)
    assert (label.value, tag) == ("MINE", "t-1")


def test_key_evidence_is_prose_and_reflects_the_deciding_relation():
    # The bug this fixes: auto_key_evidence used to be a fixed snapshot
    # (object_type/zone/verb/caption_cues/person_count) that never mentioned
    # the held_by relation even when that's what actually decided the GT.
    row = {"verb": "", "object_caption": "(3) The camera wearer holds the target."}
    evidence = _base_evidence("other_person_zone", {})
    evidence["object_type"] = "personal"
    evidence["target_object"] = "phone"
    evidence["relations"] = [{"predicate": "held_by", "object_id": "wearer", "subject_id": "target"}]
    decision = _decide_taxonomy_gt(row, evidence)
    kev = decision["key_evidence"]
    assert decision["ground_truth"] == "MINE"
    assert "wearer's own detected hand" in kev["relation_graph_evidence"]
    assert kev["object_type_evidence"] == "The target object 'phone' is categorized as personal."
    assert kev["rationale"] == decision["rationale"]
    # Raw fields are still present for programmatic re-derivation.
    assert kev["relations"] == evidence["relations"]


def test_summarize_context_change_omits_frames_that_were_never_analyzed():
    # The bug this fixes: when SAM-2 tracking loses the object at t-2/t-1,
    # _build_temporal_evidence's <2-frames guard returns an empty
    # frame_snapshots dict — but `snapshots.get(tag) or {}` turned a missing
    # snapshot into `{}`, which still passed `isinstance(snap, dict)`, so the
    # summary fabricated "At t-2, the target is in unknown_zone and is not
    # held by a detected person" for a frame that was never analyzed at all.
    temporal = {"frames_analyzed": ["t"], "frame_snapshots": {}}
    assert _summarize_context_change(temporal) == ""


def test_summarize_context_change_reports_only_analyzed_frames():
    temporal = {
        "frames_analyzed": ["t-2", "t"],
        "frame_snapshots": {
            "t-2": {"target_zone": "other_person_zone", "held_by": "person_1"},
            "t": {"target_zone": "ego_zone", "held_by": "wearer"},
        },
    }
    summary = _summarize_context_change(temporal)
    assert "t-2" in summary and "other_person_zone" in summary
    assert "the zone changes from other_person_zone to ego_zone" in summary
    assert "unknown_zone" not in summary


# ---------------------------------------------------------------------------
# Ego4D: narration only (no object_caption), #C/#O actor prefix parsing
# ---------------------------------------------------------------------------

def test_narration_actor_prefix_parses_c_and_o_tags():
    assert _narration_actor_prefix("#C picks up the bag") == "ego"
    assert _narration_actor_prefix("#O man E moves bag on the table") == "other"
    assert _narration_actor_prefix("#O2 places the cup down") == "other"
    assert _narration_actor_prefix("#OO takes block from the table") == "other"
    assert _narration_actor_prefix("no hash prefix here") is None
    assert _narration_actor_prefix("") is None
    assert _narration_actor_prefix(None) is None


def test_is_ego4d_row_checks_dataset_and_source_dataset():
    assert _is_ego4d_row({"dataset": "ego4d"})
    assert _is_ego4d_row({"source_dataset": "ego4d_fho"})
    assert not _is_ego4d_row({"dataset": "egolife"})
    assert not _is_ego4d_row({})


def test_caption_cues_object_caption_fills_gaps_for_non_actor_cues_on_ego4d():
    # object_caption is a fallback for the non-actor cues (transfer, serving,
    # etc.) when narration alone doesn't mention the relevant keyword -- it's
    # only excluded from actor-cue (ego_actor/other_actor) detection, since
    # that's the one signal it's known to hallucinate on for this dataset.
    row = {
        "dataset": "ego4d",
        "object_caption": "The other person handed the bag over.",
        "dense_caption_en": "#O man E moves bag on the table",
    }
    cues = _caption_cues(row)
    assert cues["transfer"] is True


def test_caption_cues_disagreement_trusts_neither_for_ego4d():
    # object_caption claims "camera wearer" but narration's #O tag says
    # otherwise -- neither is trusted; both actor cues stay False so the
    # decision falls through to the zone tier instead of picking a side.
    row = {
        "dataset": "ego4d",
        "object_caption": "The camera wearer picks up the bag.",
        "dense_caption_en": "#O man E moves bag on the table",
    }
    cues = _caption_cues(row)
    assert cues["ego_actor"] is False
    assert cues["other_actor"] is False


def test_caption_cues_agreement_sets_actor_for_ego4d():
    # Both narration (#O) and object_caption's own keyword ("other person")
    # agree -- that's high-confidence, use it.
    row = {
        "dataset": "ego4d",
        "object_caption": "The other person picks up the bag.",
        "dense_caption_en": "#O man E moves bag on the table",
    }
    cues = _caption_cues(row)
    assert cues["other_actor"] is True
    assert cues["ego_actor"] is False


def test_caption_cues_object_caption_only_signal_used_for_ego4d():
    # Narration has no parseable #C/#O tag but object_caption does have an
    # actor claim -- nothing to disagree with, so use it.
    row = {
        "dataset": "ego4d",
        "object_caption": "The camera wearer picks up the bag.",
        "dense_caption_en": "no hash prefix here",
    }
    cues = _caption_cues(row)
    assert cues["ego_actor"] is True
    assert cues["other_actor"] is False


def test_caption_cues_uses_object_caption_for_non_ego4d():
    row = {
        "dataset": "egolife",
        "object_caption": "The other person handed the bag over.",
        "dense_caption_en": "unrelated text",
    }
    cues = _caption_cues(row)
    assert cues["transfer"] is True


def test_caption_cues_narration_prefix_sets_other_actor_for_ego4d():
    # Raw Ego4D narration never contains English phrases like "other person"
    # -- only the #O prefix reliably signals the actor here.
    row = {"dataset": "ego4d", "dense_caption_en": "#O man E moves bag on the table"}
    cues = _caption_cues(row)
    assert cues["other_actor"] is True
    assert cues["ego_actor"] is False


def test_caption_cues_narration_prefix_sets_ego_actor_for_ego4d():
    row = {"dataset": "ego4d", "dense_caption_en": "#C picks up the bag from the table"}
    cues = _caption_cues(row)
    assert cues["ego_actor"] is True
    assert cues["other_actor"] is False


def test_caption_interaction_evidence_prefers_object_caption_for_ego4d():
    row = {
        "dataset": "ego4d",
        "object_caption": "(3) camera wearer picks up the plate quickly.",
        "dense_caption_en": "#O man E moves bag on the table",
    }
    evidence = _caption_interaction_evidence(row)
    assert "plate" in evidence
    assert "bag" not in evidence


def test_caption_interaction_evidence_falls_back_to_narration_for_ego4d():
    row = {
        "dataset": "ego4d",
        "object_caption": "",
        "dense_caption_en": "#O man E moves bag on the table",
    }
    evidence = _caption_interaction_evidence(row)
    assert "bag" in evidence


def test_caption_interaction_evidence_uses_object_caption_for_non_ego4d():
    row = {
        "dataset": "egolife",
        "object_caption": "(1) intro\n(3) camera wearer picks up the plate quickly.\n(4) more",
        "dense_caption_en": "unrelated fallback text",
    }
    evidence = _caption_interaction_evidence(row)
    assert "plate" in evidence
