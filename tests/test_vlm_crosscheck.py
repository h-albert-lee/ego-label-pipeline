from egoownership.vlm_crosscheck import _extract_label_from_text, _parse_label_response


def test_parses_person_k_regardless_of_case():
    # The bug this fixes: PERSON_k is the only mixed-case label. Uppercasing
    # an extracted candidate before checking VALID_LABELS membership (or
    # searching for VALID_LABELS members inside an upper-cased text blob)
    # can never match "PERSON_k" since its lowercase "k" can't survive either
    # direction of that case-folding — every case variant fell through to
    # "UNKNOWN" except the one exact string "PERSON_k".
    for variant in ("PERSON_k", "PERSON_K", "person_k", "Person_k"):
        raw = f'{{"label": "{variant}", "zone_evidence": "near the other person"}}'
        assert _parse_label_response(raw)["label"] == "PERSON_k"


def test_parses_other_labels_case_insensitively_too():
    for variant, expected in [("mine", "MINE"), ("Shared", "SHARED"), ("ambiguous", "AMBIGUOUS")]:
        raw = f'{{"label": "{variant}"}}'
        assert _parse_label_response(raw)["label"] == expected


def test_falls_back_to_unknown_for_garbage_label():
    raw = '{"label": "NOT_A_REAL_LABEL"}'
    assert _parse_label_response(raw)["label"] == "UNKNOWN"


def test_extract_label_from_text_finds_person_k_case_insensitively():
    assert _extract_label_from_text("I think this is person_k's item") == "PERSON_k"
    assert _extract_label_from_text("the answer is PERSON_K") == "PERSON_k"
    assert _extract_label_from_text("no label here") == "UNKNOWN"


def test_unparseable_json_falls_back_to_text_search_and_keeps_raw_response():
    raw = "not json at all, but the label should be person_k here"
    result = _parse_label_response(raw)
    assert result["label"] == "PERSON_k"
    assert result["raw_response"] == raw


def test_indexed_person_variants_canonicalize_to_person_k():
    # The bug this fixes: PERSON_k's "_k" reads like a template variable to
    # fill in, and models reasonably answer with an actual index (PERSON_1,
    # PERSON_2, ...) instead of the literal placeholder string — our schema
    # never tracks *which* person, only "not the wearer", so any such
    # indexed variant should still resolve to PERSON_k rather than UNKNOWN.
    for variant in ("PERSON_1", "PERSON_2", "PERSON2", "PERSON 3", "person_1"):
        raw = f'{{"label": "{variant}"}}'
        assert _parse_label_response(raw)["label"] == "PERSON_k"


def test_indexed_person_variant_in_free_text_fallback():
    assert _extract_label_from_text("the object belongs to PERSON_2") == "PERSON_k"
