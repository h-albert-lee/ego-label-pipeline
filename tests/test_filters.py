from pathlib import Path

from egoownership.datasets import iter_fho_candidates
from egoownership.filters import filter_candidates, matches_taxonomy
from egoownership.config import load_config
from egoownership.schema import Taxonomy, ClipCandidate

FIXTURES = Path(__file__).parent / "fixtures"


def test_contextual_filter_keeps_shared_table_verbs():
    cands = list(iter_fho_candidates(FIXTURES / "fho_mini.json"))
    kept = list(filter_candidates(cands, Taxonomy.CONTEXTUAL))
    verbs = [c.verb for c in kept]
    # put_down (cup) + give (pen) + pass (basket/bread) → 3 kept.
    # turn_on (oven) is not a contextual verb → dropped.
    assert len(kept) == 3
    assert "put_down" in verbs
    assert "give" in verbs
    assert "pass" in verbs
    assert "turn_on" not in verbs
    for c in kept:
        assert c.taxonomy is Taxonomy.CONTEXTUAL


def test_require_shared_noun_can_be_disabled():
    # Craft a candidate with a contextual verb but an off-list noun.
    cand = ClipCandidate(
        dataset="test",
        clip_id="x",
        taxonomy=Taxonomy.CONTEXTUAL,
        t_minus_2_sec=0,
        t_minus_1_sec=1,
        t_sec=2,
        verb="give",
        nouns=["screwdriver"],
    )
    cfg = load_config()
    assert not matches_taxonomy(cand, Taxonomy.CONTEXTUAL, cfg, require_shared_noun=True)
    assert matches_taxonomy(cand, Taxonomy.CONTEXTUAL, cfg, require_shared_noun=False)


def test_limit_enforced():
    cands = list(iter_fho_candidates(FIXTURES / "fho_mini.json"))
    kept = list(filter_candidates(cands, Taxonomy.CONTEXTUAL, limit=2))
    assert len(kept) == 2
