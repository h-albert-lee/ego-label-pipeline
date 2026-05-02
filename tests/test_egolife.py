import json
from pathlib import Path

from egoownership.datasets import iter_egolife_candidates
from egoownership.schema import Taxonomy


def test_egolife_parser(tmp_path: Path):
    sample = {
        "clip_id": "day3_dinner_p1",
        "video_id": "day3_dinner",
        "participant": "p1",
        "scenario": "dining",
        "fps": 30,
        "events": [
            {
                "start_sec": 412.0,
                "end_sec": 414.5,
                "verb": "pass",
                "nouns": ["bowl"],
                "narration": "passes the rice bowl to p3",
                "transcript": "could you pass me the rice?",
                "av_caption": "p1 reaches across the table",
            },
            {
                "start_sec": 500.0,
                "end_sec": 502.0,
                "nouns": ["chopstick"],
                "narration": "chopstick rests on table",
            },
        ],
    }
    p = tmp_path / "clip.json"
    p.write_text(json.dumps(sample), encoding="utf-8")

    cands = list(iter_egolife_candidates(p))
    assert len(cands) == 2

    pass_event = cands[0]
    assert pass_event.dataset == "egolife"
    assert pass_event.taxonomy is Taxonomy.CONTEXTUAL
    assert pass_event.t_minus_2_sec == 412.0
    assert pass_event.t_sec == 414.5
    assert "bowl" in pass_event.nouns
    assert "rice bowl" in (pass_event.narration or "")
    assert "could you pass me" in (pass_event.narration or "")

    rest_event = cands[1]
    # No verb → Baseline-leaning since scenario=dining.
    assert rest_event.taxonomy is Taxonomy.BASELINE


def test_egolife_directory_walk(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({
        "clip_id": "a", "events": [{"start_sec": 0.0, "end_sec": 1.0, "verb": "give", "nouns": ["cup"]}]
    }))
    (tmp_path / "b.json").write_text(json.dumps({
        "clip_id": "b", "events": [{"start_sec": 0.0, "end_sec": 1.0, "verb": "put", "nouns": ["plate"]}]
    }))
    cands = list(iter_egolife_candidates(tmp_path))
    assert len(cands) == 2
    assert {c.clip_id.split(":")[0] for c in cands} == {"a", "b"}
