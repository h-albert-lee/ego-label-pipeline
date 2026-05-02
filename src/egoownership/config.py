"""Configuration loader for taxonomy verb/noun whitelists and zone thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "taxonomy.yaml"


@dataclass(frozen=True)
class OwnershipZones:
    mine_near_y_min: float
    shared_x_min: float
    shared_x_max: float
    person_far_y_max: float
    min_bbox_area_ratio: float


@dataclass(frozen=True)
class TaxonomyConfig:
    contextual_verbs: frozenset[str]
    baseline_verbs: frozenset[str]
    shared_table_nouns: frozenset[str]
    zones: OwnershipZones

    def verbs_for(self, taxonomy: str) -> frozenset[str]:
        if taxonomy == "C":
            return self.contextual_verbs
        if taxonomy == "A":
            return self.baseline_verbs
        # B (Conflict) and D (Ambiguous) use the union — they're defined
        # by post-hoc disagreement / symmetry, not by verb choice.
        return self.contextual_verbs | self.baseline_verbs


def _normalize(tokens: list[str]) -> frozenset[str]:
    """Lowercase + replace spaces/hyphens with underscore to match annotation styles."""
    out: set[str] = set()
    for t in tokens:
        s = t.strip().lower().replace("-", "_").replace(" ", "_")
        if s:
            out.add(s)
    return frozenset(out)


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> TaxonomyConfig:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    zones = OwnershipZones(**raw["ownership_zones"])
    return TaxonomyConfig(
        contextual_verbs=_normalize(raw.get("contextual_verbs", [])),
        baseline_verbs=_normalize(raw.get("baseline_verbs", [])),
        shared_table_nouns=_normalize(raw.get("shared_table_nouns", [])),
        zones=zones,
    )


def normalize_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")
