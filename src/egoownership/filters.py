"""Taxonomy-aware filtering of clip candidates.

Rules implement the strategy in §3 of the EDA doc:

* **Taxonomy C (Contextual)** — verb must be in `contextual_verbs`. Noun must
  intersect `shared_table_nouns` when the knob is on. Favors dining / meeting.
* **Taxonomy A (Baseline)** — verb empty OR in `baseline_verbs`; nouns still
  must intersect the shared-table list so we stay on the benchmark surface.
* **Taxonomy D (Ambiguous)** — accepted downstream only (needs detection
  evidence). For filtering purposes we pass everything through with taxonomy=D
  when the caller explicitly asks for D.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from egoownership.config import TaxonomyConfig, load_config
from egoownership.schema import ClipCandidate, Taxonomy


def _has_shared_noun(cand: ClipCandidate, cfg: TaxonomyConfig) -> bool:
    if not cand.nouns:
        return False
    return any(n in cfg.shared_table_nouns for n in cand.nouns)


def matches_taxonomy(
    cand: ClipCandidate,
    target: Taxonomy,
    cfg: TaxonomyConfig,
    require_shared_noun: bool = True,
) -> bool:
    """Return True iff ``cand`` passes the rule set for ``target`` taxonomy."""

    if target is Taxonomy.CONTEXTUAL:
        if cand.verb is None or cand.verb not in cfg.contextual_verbs:
            return False
        if require_shared_noun and not _has_shared_noun(cand, cfg):
            return False
        return True

    if target is Taxonomy.BASELINE:
        if cand.verb is not None and cand.verb not in cfg.baseline_verbs:
            # Mid-action clips are not Baseline.
            return False
        if require_shared_noun and not _has_shared_noun(cand, cfg):
            return False
        return True

    if target is Taxonomy.AMBIGUOUS:
        # Purely structural; we accept any candidate that has any noun,
        # since ambiguity is resolved after detection.
        return bool(cand.nouns)

    # Conflict (B) is defined by post-hoc disagreement between per-frame
    # detections and FHO verb, so filtering layer alone cannot certify it.
    # We pass everything through and let the labeler flag conflicts.
    return True


def filter_candidates(
    cands: Iterable[ClipCandidate],
    target: Taxonomy,
    *,
    config: TaxonomyConfig | None = None,
    require_shared_noun: bool = True,
    limit: int | None = None,
) -> Iterator[ClipCandidate]:
    cfg = config or load_config()
    count = 0
    for cand in cands:
        if limit is not None and count >= limit:
            return
        if matches_taxonomy(cand, target, cfg, require_shared_noun=require_shared_noun):
            # Stamp the taxonomy onto the candidate so downstream stages don't
            # re-derive it. We copy to avoid mutating upstream state.
            stamped = cand.model_copy(update={"taxonomy": target})
            yield stamped
            count += 1
