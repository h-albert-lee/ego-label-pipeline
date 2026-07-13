"""Core data models for scene records, detections, and the scene graph.

A `SceneRecord` is the final output of the pipeline — one row per clip, with
three sparse frames (`t-2`, `t-1`, `t`), per-frame detections, instance-level
tracking, attributes, scene graph, and a taxonomy + ownership label.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Taxonomy(str, Enum):
    BASELINE = "A"
    CONFLICT = "B"
    CONTEXTUAL = "C"
    AMBIGUOUS = "D"


class OwnershipLabel(str, Enum):
    MINE = "MINE"
    PERSON_K = "PERSON_k"
    SHARED = "SHARED"
    AMBIGUOUS = "AMBIGUOUS"


FrameTag = Literal["t-2", "t-1", "t"]


class BBox(BaseModel):
    """Axis-aligned bbox in *normalized* image coords (0..1)."""

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @field_validator("x_max")
    @classmethod
    def _x_ordered(cls, v: float, info) -> float:
        if v < info.data.get("x_min", 0.0):
            raise ValueError("x_max must be >= x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def _y_ordered(cls, v: float, info) -> float:
        if v < info.data.get("y_min", 0.0):
            raise ValueError("y_max must be >= y_min")
        return v

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x_min, other.x_min), max(self.y_min, other.y_min)
        ix2, iy2 = min(self.x_max, other.x_max), min(self.y_max, other.y_max)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def from_xyxy_abs(cls, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> "BBox":
        return cls(
            x_min=max(0.0, min(1.0, x1 / w)),
            y_min=max(0.0, min(1.0, y1 / h)),
            x_max=max(0.0, min(1.0, x2 / w)),
            y_max=max(0.0, min(1.0, y2 / h)),
        )


class ObjectAttributes(BaseModel):
    """VLM-extracted attributes for one cropped object (Q1 of the ideation)."""

    color: str | None = None
    material: str | None = None
    state: str | None = None  # e.g. "empty", "filled", "open", "closed"
    text_on_object: str | None = None
    fine_grained_label: str | None = None  # e.g. "ceramic mug" instead of just "cup"
    distinctive_marks: str | None = None
    raw_caption: str | None = None
    extras: dict = Field(default_factory=dict)


class ObjectDetection(BaseModel):
    """One detected object in one frame."""

    label: str
    bbox: BBox
    score: float | None = None
    instance_id: str | None = None
    ownership: OwnershipLabel | None = None
    ownership_evidence: list[str] = Field(default_factory=list)
    attributes: ObjectAttributes | None = None
    mean_depth: float | None = None  # 0..1 monocular depth (optional)


class PersonDetection(BaseModel):
    """A visible person other than the camera wearer."""

    bbox: BBox
    person_id: str | None = None  # e.g. "person_1", "person_2"
    score: float | None = None
    is_camera_wearer: bool = False  # the wearer is normally not visible


class Relation(BaseModel):
    """An edge in the scene graph."""

    subject_id: str  # object instance_id or "person_k" / "wearer"
    object_id: str
    predicate: str   # e.g. "next_to", "in_front_of", "held_by", "on_table"
    score: float | None = None
    note: str | None = None


class FrameZones(BaseModel):
    """Per-frame zone definition derived from person bboxes (and/or depth)."""

    mine_y_min: float = 0.55
    person_zones: dict[str, BBox] = Field(default_factory=dict)  # person_id → influence bbox
    shared_x_min: float = 0.30
    shared_x_max: float = 0.70
    derivation: str = "default-static"   # "static-yaml" | "person-relative" | "depth"


class FrameDetections(BaseModel):
    """All detections for a single sparse frame."""

    tag: FrameTag
    frame_path: str | None = None
    timestamp_sec: float
    width: int | None = None
    height: int | None = None
    objects: list[ObjectDetection] = Field(default_factory=list)
    persons: list[PersonDetection] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    zones: FrameZones | None = None
    narration: str | None = None  # frame-level dense narration if available


class ClipCandidate(BaseModel):
    """A clip selected by metadata filtering, before any frame extraction."""

    dataset: str
    clip_id: str
    video_id: str | None = None
    taxonomy: Taxonomy
    t_minus_2_sec: float
    t_minus_1_sec: float
    t_sec: float
    verb: str | None = None
    nouns: list[str] = Field(default_factory=list)
    narration: str | None = None
    source: dict = Field(default_factory=dict)


class AnnotationEdit(BaseModel):
    """One human edit recorded during collaborative review."""

    annotator: str
    when: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    field: str        # "scene_label" | f"object:{instance_id}:ownership" | ...
    old_value: str | None = None
    new_value: str | None = None
    note: str | None = None


class VLMJudgement(BaseModel):
    """A second-opinion ownership label proposed by a remote VLM (Claude / GPT-4o).

    Stored alongside the rule-cascade label, never replacing it. Surfaces in
    the annotator UI so reviewers can compare the two signals.
    """

    provider: str  # "anthropic" | "openai"
    model: str
    label: OwnershipLabel
    confidence: float
    rationale: str | None = None
    target_instance_hint: str | None = None


class SceneRecord(BaseModel):
    """Final record that lands in the benchmark."""

    clip: ClipCandidate
    frames: list[FrameDetections]
    scene_label: OwnershipLabel | None = None
    scene_taxonomy: Taxonomy | None = None
    notes: str | None = None
    auto_label_confidence: float | None = None
    review_status: Literal["draft", "in_review", "verified", "rejected", "auto_accepted"] = "draft"
    edits: list[AnnotationEdit] = Field(default_factory=list)
    vlm_judgement: VLMJudgement | None = None
    # Annotators this scene is assigned to (audit workflow; empty = unassigned).
    assigned_to: list[str] = Field(default_factory=list)
