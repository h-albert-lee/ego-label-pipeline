from pathlib import Path

from egoownership.datasets import (
    iter_epic_candidates,
    iter_fho_candidates,
    iter_hd_epic_candidates,
)
from egoownership.schema import Taxonomy

FIXTURES = Path(__file__).parent / "fixtures"


def test_fho_parser_extracts_sparse_frames():
    cands = list(iter_fho_candidates(FIXTURES / "fho_mini.json"))
    assert len(cands) == 4
    put_down = next(c for c in cands if c.verb == "put_down")
    assert put_down.taxonomy is Taxonomy.CONTEXTUAL
    assert put_down.t_minus_2_sec == 10.0
    assert put_down.t_minus_1_sec == 10.4
    assert put_down.t_sec == 10.8
    assert "cup" in put_down.nouns
    assert put_down.video_id == "video_A"


def test_epic_parser_handles_timestamps_and_nouns():
    cands = list(iter_epic_candidates(FIXTURES / "epic_mini.csv"))
    assert len(cands) == 4
    put = next(c for c in cands if c.verb == "put")
    assert put.t_minus_2_sec == 120.0
    assert put.t_sec == 122.0
    # midpoint
    assert abs(put.t_minus_1_sec - 121.0) < 1e-6
    assert "cup" in put.nouns

    # hand-over gets normalized → "hand"
    hand = next(c for c in cands if c.verb == "hand")
    assert "plate" in hand.nouns


def test_hd_epic_parser_converts_frames_to_seconds():
    cands = list(iter_hd_epic_candidates(FIXTURES / "hd_epic_mini.json"))
    assert len(cands) == 2
    transfer = next(c for c in cands if c.verb == "transfer")
    # 13200 / 60 = 220.0, 13800 / 60 = 230.0
    assert transfer.t_minus_2_sec == 220.0
    assert transfer.t_sec == 230.0
    assert "notebook" in transfer.nouns
