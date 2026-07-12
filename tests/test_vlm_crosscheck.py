import json
from pathlib import Path
from unittest.mock import patch

from egoownership.vlm_crosscheck import (
    _EVIDENCE_FIELDS,
    _extract_label_from_text,
    _load_disagreed_rows,
    _parse_label_response,
    _reconstruct_frame_paths,
    _resolve_frame_paths,
    merge_crosscheck_jsonl,
    write_crosscheck_jsonl,
)


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


# ---------------------------------------------------------------------------
# Frame reconstruction from a local raw video (metadata-only dataset support)
# ---------------------------------------------------------------------------

def _row(**overrides):
    row = {
        "id": "vid1:clip1:100.0#obj0",
        "video_id": "vid1",
        "source_video_start_sec": 980.821,
        "frame_times_sec": {"t-2": 13.0, "t-1": 14.0, "t": 15.0},
    }
    row.update(overrides)
    return row


def test_reconstruct_returns_none_when_required_fields_missing():
    assert _reconstruct_frame_paths({}, Path("/anywhere"), None) == [None, None, None]
    assert _reconstruct_frame_paths(_row(video_id=None), Path("/anywhere"), None) == [None, None, None]


def test_reconstruct_returns_none_when_video_file_missing(tmp_path: Path):
    # videos_root has no vid1.mp4 in it.
    assert _reconstruct_frame_paths(_row(), tmp_path, None) == [None, None, None]


def test_reconstruct_extracts_at_source_start_plus_local_offset(tmp_path: Path):
    # The formula this guards: absolute timestamp in the raw video is
    # source_video_start_sec + frame_times_sec[tag], not frame_times_sec[tag]
    # alone (that's only valid for our own already-cut per-clip files).
    (tmp_path / "vid1.mp4").touch()
    calls = []

    def fake_extract(video_path, dest, timestamp_sec, force=False):
        calls.append((video_path, dest, round(timestamp_sec, 3)))
        return dest

    with patch("egoownership.vlm_crosscheck._ffmpeg_extract_frame", side_effect=fake_extract):
        result = _reconstruct_frame_paths(_row(), tmp_path, tmp_path / "cache")

    assert len(calls) == 3
    timestamps = {c[2] for c in calls}
    assert timestamps == {993.821, 994.821, 995.821}  # 980.821 + 13/14/15
    assert all(c[0] == tmp_path / "vid1.mp4" for c in calls)
    assert all(r is not None for r in result)


def test_resolve_frame_paths_only_falls_back_when_local_files_missing(tmp_path: Path):
    # If frames_root already has the pre-extracted JPEGs, reconstruction must
    # not be triggered at all (no reason to touch the raw video).
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    (frames_root / "t.jpg").write_bytes(b"fake")
    row = {
        "frame_t_minus_2_path": "missing_t2.jpg",
        "frame_t_minus_1_path": "missing_t1.jpg",
        "frame_t_path": "t.jpg",
        **_row(),
    }
    with patch("egoownership.vlm_crosscheck._reconstruct_frame_paths") as mock_reconstruct:
        mock_reconstruct.return_value = [None, None, None]
        paths = _resolve_frame_paths(row, frames_root, videos_root=tmp_path)
    # t.jpg resolves locally; t-2/t-1 don't, so reconstruction IS attempted
    # (called once), but the pre-resolved t.jpg is kept, not overwritten.
    mock_reconstruct.assert_called_once()
    assert paths[2] == frames_root / "t.jpg"


def test_resolve_frame_paths_skips_reconstruction_when_all_local_files_found(tmp_path: Path):
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    for name in ("t2.jpg", "t1.jpg", "t.jpg"):
        (frames_root / name).write_bytes(b"fake")
    row = {
        "frame_t_minus_2_path": "t2.jpg",
        "frame_t_minus_1_path": "t1.jpg",
        "frame_t_path": "t.jpg",
        **_row(),
    }
    with patch("egoownership.vlm_crosscheck._reconstruct_frame_paths") as mock_reconstruct:
        _resolve_frame_paths(row, frames_root, videos_root=tmp_path)
    mock_reconstruct.assert_not_called()


# ---------------------------------------------------------------------------
# Passing already-disagreed records through unchanged for a second judge
# (--only-agreed-in), instead of re-checking or dropping them
# ---------------------------------------------------------------------------

def test_load_disagreed_rows_keeps_only_majority_agrees_false(tmp_path: Path):
    path = tmp_path / "claude_crosscheck.jsonl"
    agreed_row = {"id": "agreed-1", "majority_agrees": True}
    disagreed_row = {"id": "disagreed-1", "majority_agrees": False, "majority_label": "SHARED"}
    no_data_row = {"id": "no-data-1"}  # majority_agrees absent -> not a confirmed disagree
    path.write_text(
        "\n".join(json.dumps(r) for r in [agreed_row, disagreed_row, no_data_row]),
        encoding="utf-8",
    )
    assert _load_disagreed_rows(path) == {"disagreed-1": disagreed_row}


def test_load_disagreed_rows_returns_empty_dict_when_file_missing(tmp_path: Path):
    assert _load_disagreed_rows(tmp_path / "does_not_exist.jsonl") == {}


class _FakeJudge:
    model_id = "fake-judge"

    def judge(self, frame_paths, record):
        return {"label": record["auto_ground_truth"], **{f: "x" for f in _EVIDENCE_FIELDS}}


def test_write_crosscheck_jsonl_passes_disagreed_records_through_unjudged(tmp_path: Path):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        "\n".join([
            json.dumps({"id": "agreed-1", "auto_ground_truth": "MINE"}),
            json.dumps({"id": "disagreed-1", "auto_ground_truth": "SHARED"}),
            json.dumps({"id": "agreed-2", "auto_ground_truth": "AMBIGUOUS"}),
        ]),
        encoding="utf-8",
    )
    prior_path = tmp_path / "claude_crosscheck.jsonl"
    disagreed_prior_row = {
        "id": "disagreed-1", "majority_agrees": False, "majority_label": "MINE",
        "judges": {"claude-sonnet-4-6": {"label": "MINE"}},
    }
    prior_path.write_text(
        "\n".join([
            json.dumps({"id": "agreed-1", "majority_agrees": True}),
            json.dumps(disagreed_prior_row),
            json.dumps({"id": "agreed-2", "majority_agrees": True}),
        ]),
        encoding="utf-8",
    )
    out_path = tmp_path / "gpt_crosscheck.jsonl"
    fake_judge = _FakeJudge()

    n = write_crosscheck_jsonl(
        labels_path, out_path, [fake_judge],
        only_agreed_in=prior_path, show_progress=False,
    )
    # Same total row count as an unfiltered run -- nothing silently dropped.
    assert n == 3
    written = {json.loads(line)["id"]: json.loads(line) for line in out_path.read_text().splitlines()}
    assert set(written) == {"agreed-1", "disagreed-1", "agreed-2"}
    # The disagreed row is carried through byte-for-byte, not re-judged --
    # it must NOT contain the fake judge's model_id.
    assert written["disagreed-1"] == disagreed_prior_row
    assert fake_judge.model_id not in written["disagreed-1"]["judges"]
    # The agreed rows DID get a fresh judgement from the new judge.
    assert fake_judge.model_id in written["agreed-1"]["judges"]
    assert fake_judge.model_id in written["agreed-2"]["judges"]


# ---------------------------------------------------------------------------
# Merging separate per-judge crosscheck runs (vlm-crosscheck-merge)
# ---------------------------------------------------------------------------

def test_merge_crosscheck_jsonl_unions_judges_per_id(tmp_path: Path):
    claude_path = tmp_path / "claude_crosscheck.jsonl"
    claude_path.write_text(
        "\n".join([
            json.dumps({
                "id": "row-1", "auto_ground_truth": "MINE",
                "judges": {"claude-sonnet-4-6": {"label": "MINE", "agrees": True}},
                "agreement_count": 1, "agreement_ratio": 1.0,
                "majority_label": "MINE", "majority_agrees": True,
            }),
            json.dumps({
                "id": "row-2", "auto_ground_truth": "SHARED",
                "judges": {"claude-sonnet-4-6": {"label": "MINE", "agrees": False}},
                "agreement_count": 0, "agreement_ratio": 0.0,
                "majority_label": "MINE", "majority_agrees": False,
            }),
        ]),
        encoding="utf-8",
    )
    gpt_path = tmp_path / "gpt_crosscheck.jsonl"
    gpt_path.write_text(
        "\n".join([
            json.dumps({
                "id": "row-1", "auto_ground_truth": "MINE",
                "judges": {"gpt-4o": {"label": "MINE", "agrees": True}},
                "agreement_count": 1, "agreement_ratio": 1.0,
                "majority_label": "MINE", "majority_agrees": True,
            }),
            # row-2 was already disagreed for claude, so the gpt run passed
            # it through unchanged with claude's judge, not gpt's.
            json.dumps({
                "id": "row-2", "auto_ground_truth": "SHARED",
                "judges": {"claude-sonnet-4-6": {"label": "MINE", "agrees": False}},
                "agreement_count": 0, "agreement_ratio": 0.0,
                "majority_label": "MINE", "majority_agrees": False,
            }),
        ]),
        encoding="utf-8",
    )
    out_path = tmp_path / "merged.jsonl"

    n = merge_crosscheck_jsonl([claude_path, gpt_path], out_path)
    assert n == 2
    merged = {json.loads(line)["id"]: json.loads(line) for line in out_path.read_text().splitlines()}

    row1 = merged["row-1"]
    assert set(row1["judges"]) == {"claude-sonnet-4-6", "gpt-4o"}
    assert row1["agreement_count"] == 2
    assert row1["agreement_ratio"] == 1.0
    assert row1["majority_agrees"] is True

    # row-2 only ever had claude's judge (gpt's run passed it through) --
    # merging must not fabricate a second judge for it.
    row2 = merged["row-2"]
    assert set(row2["judges"]) == {"claude-sonnet-4-6"}
    assert row2["agreement_count"] == 0
    assert row2["majority_agrees"] is False


def test_merge_crosscheck_jsonl_keeps_id_present_in_only_one_file(tmp_path: Path):
    a_path = tmp_path / "a.jsonl"
    a_path.write_text(
        json.dumps({
            "id": "only-in-a", "auto_ground_truth": "MINE",
            "judges": {"judge-a": {"label": "MINE", "agrees": True}},
        }),
        encoding="utf-8",
    )
    b_path = tmp_path / "b.jsonl"
    b_path.write_text("", encoding="utf-8")
    out_path = tmp_path / "merged.jsonl"

    n = merge_crosscheck_jsonl([a_path, b_path], out_path)
    assert n == 1
    merged = json.loads(out_path.read_text())
    assert merged["id"] == "only-in-a"
    assert set(merged["judges"]) == {"judge-a"}
