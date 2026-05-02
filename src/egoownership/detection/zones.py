"""Dynamic zone derivation.

Three strategies, all returning a :class:`FrameZones`:

1. ``static_zones`` — fall back to the YAML thresholds (legacy behavior).
2. ``person_relative_zones`` — each visible person owns the rectangle around
   their bbox; everything below all persons is the wearer's near zone; the
   horizontal band between the leftmost and rightmost person is SHARED.
3. ``depth_aware_zones`` — adds a depth check: if mean depth of an object is
   smaller than the wearer-zone median, it's MINE regardless of y position.

The pipeline can chain these — strategies stamp ``derivation`` so the UI can
show *which* rule fired.
"""

from __future__ import annotations

from egoownership.config import OwnershipZones
from egoownership.schema import BBox, FrameZones, PersonDetection


def static_zones(yaml_zones: OwnershipZones) -> FrameZones:
    return FrameZones(
        mine_y_min=yaml_zones.mine_near_y_min,
        shared_x_min=yaml_zones.shared_x_min,
        shared_x_max=yaml_zones.shared_x_max,
        derivation="static-yaml",
    )


def person_relative_zones(
    persons: list[PersonDetection], yaml_zones: OwnershipZones
) -> FrameZones:
    """Derive zones by treating each visible person as a PERSON_k anchor."""

    if not persons:
        return static_zones(yaml_zones)

    # Wearer's near zone starts just below the lowest person bbox bottom.
    # If the lowest person is high in the frame, we keep a permissive default.
    lowest_y = max(p.bbox.y_max for p in persons)
    mine_y_min = max(0.45, min(0.85, lowest_y + 0.05))

    # SHARED band spans between leftmost and rightmost person centers.
    x_centers = sorted(p.bbox.center[0] for p in persons)
    if len(x_centers) == 1:
        # One opposite person → shared band is the central 20% around the
        # horizontal opposite-of-wearer axis (which is about screen center).
        shared_x_min = max(0.20, yaml_zones.shared_x_min - 0.05)
        shared_x_max = min(0.80, yaml_zones.shared_x_max + 0.05)
    else:
        shared_x_min = max(0.10, x_centers[0] - 0.05)
        shared_x_max = min(0.90, x_centers[-1] + 0.05)

    # Each person gets an *influence* bbox: their own bbox padded laterally and
    # extending up to the top of the frame. Anything inside that rectangle and
    # above the wearer-zone is attributed to that person.
    influence: dict[str, BBox] = {}
    for p in persons:
        if not p.person_id:
            continue
        pad = 0.04
        influence[p.person_id] = BBox(
            x_min=max(0.0, p.bbox.x_min - pad),
            y_min=0.0,
            x_max=min(1.0, p.bbox.x_max + pad),
            y_max=mine_y_min,
        )

    return FrameZones(
        mine_y_min=mine_y_min,
        shared_x_min=shared_x_min,
        shared_x_max=shared_x_max,
        person_zones=influence,
        derivation="person-relative",
    )


def depth_aware_refine(
    zones: FrameZones, *, wearer_depth_band: tuple[float, float] | None
) -> FrameZones:
    """Just stamp the derivation tag — actual depth lookup happens per-object
    in :func:`egoownership.detection.ownership._classify_with_zones`.
    """
    if wearer_depth_band is None:
        return zones
    return zones.model_copy(update={"derivation": zones.derivation + "+depth"})
