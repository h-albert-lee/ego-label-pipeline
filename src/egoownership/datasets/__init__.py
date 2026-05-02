"""Per-dataset annotation parsers.

Every parser yields :class:`~egoownership.schema.ClipCandidate` objects, so the
downstream filter / frame / detection / label stages are dataset-agnostic.
"""

from egoownership.datasets.ego4d_fho import iter_fho_candidates
from egoownership.datasets.egolife import iter_egolife_candidates
from egoownership.datasets.epic_kitchens import iter_epic_candidates
from egoownership.datasets.hd_epic import iter_hd_epic_candidates

__all__ = [
    "iter_fho_candidates",
    "iter_egolife_candidates",
    "iter_epic_candidates",
    "iter_hd_epic_candidates",
]
