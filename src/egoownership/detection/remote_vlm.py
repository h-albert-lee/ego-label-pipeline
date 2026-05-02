"""Remote VLM client for ownership-pipeline tasks.

Three operations, all batched and prompt-cache-friendly:

1. ``caption_object`` — Crop one object from a frame, return ``ObjectAttributes``
   (color/material/state/marks) as a structured JSON. Replaces the local BLIP-2
   path in :mod:`egoownership.detection.attributes`.
2. ``tag_frame`` — List visible objects in an entire frame. Replaces the local
   RAM path in :mod:`egoownership.detection.ram` for shops without GPU.
3. ``judge_scene`` — Read all 3 frames + clip metadata, propose a scene-level
   ownership label with rationale. Used as an extra signal alongside the
   rule-cascade label, NOT a replacement.

Backend: Claude Opus 4.7 by default via the Anthropic SDK. Uses adaptive
thinking for ``judge_scene`` (it benefits from reasoning), no thinking for the
attribute / tag tasks (they're tight visual extraction). Prompt caching is set
on the system prompt for every operation so a 100-clip run only pays the system
prompt once per 5-minute window.

For an OpenAI variant, see :mod:`egoownership.detection.remote_vlm_openai`.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from egoownership.schema import (
    BBox,
    ClipCandidate,
    FrameDetections,
    ObjectAttributes,
    ObjectDetection,
    OwnershipLabel,
)

# ---- model ----
# claude-api skill: ALWAYS use claude-opus-4-7 unless the user explicitly
# names a different model. Configurable via constructor / env.
_DEFAULT_MODEL = "claude-opus-4-7"


@dataclass
class RemoteVLMConfig:
    model: str = _DEFAULT_MODEL
    api_key: str | None = None     # falls back to ANTHROPIC_API_KEY
    enable_caching: bool = True
    enable_thinking_for_judge: bool = True
    max_tokens_attrs: int = 800
    max_tokens_tags: int = 600
    max_tokens_judge: int = 4000


# ---------- prompts (kept as constants so prompt-cache stays warm) ----------

_SYS_ATTRS = (
    "You are a vision annotator extracting structured attributes from a single "
    "cropped object image. The crop comes from a first-person (egocentric) "
    "video frame; expect motion blur, partial occlusion, and tight crops. "
    "Return only the JSON requested by the schema. Set a field to null when "
    "the image does not let you identify it confidently. Do not invent details."
)

_SYS_TAGS = (
    "You are a vision tagger that lists every distinct physical object visible "
    "in a single first-person video frame. Output a JSON list of singular "
    "lowercase nouns (e.g. 'cup', 'plate', 'pen', 'hand'). Avoid attributes "
    "(no 'red cup' — just 'cup'). Include 'hand' whenever a hand is visible. "
    "Cap the list at the first 25 objects you would tag for a downstream "
    "object-grounding step."
)

_SYS_JUDGE = (
    "You are an expert annotator for the Egocentric Implicit Ownership "
    "benchmark. Decide who owns the *target object* in this clip based on "
    "three sparse frames (t-2, t-1, t) and the clip metadata (verb, nouns, "
    "narration).\n\n"
    "Label space:\n"
    "  MINE     — owned by the camera wearer\n"
    "  PERSON_k — owned by another person visible in the scene\n"
    "  SHARED   — communal or table-center, not personally owned\n"
    "  AMBIGUOUS — symmetric/occluded/insufficient evidence\n\n"
    "Use the FULL temporal trajectory: an object that starts MINE but is "
    "handed to person_1 by the final frame is PERSON_k. Look for hand "
    "contact, body proximity, and zone consistency. If two equivalent "
    "candidates exist (e.g. two identical cups equally far from any "
    "person), output AMBIGUOUS. Provide a one-sentence rationale citing "
    "specific visual evidence."
)


# ---------- JSON schemas for structured outputs ----------

_ATTRS_SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": ["string", "null"]},
        "material": {"type": ["string", "null"]},
        "state": {"type": ["string", "null"]},
        "text_on_object": {"type": ["string", "null"]},
        "fine_grained_label": {"type": ["string", "null"]},
        "distinctive_marks": {"type": ["string", "null"]},
    },
    "required": [
        "color", "material", "state", "text_on_object",
        "fine_grained_label", "distinctive_marks",
    ],
    "additionalProperties": False,
}

_TAGS_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["tags"],
    "additionalProperties": False,
}

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "target_instance_hint": {"type": ["string", "null"]},
    },
    "required": ["label", "confidence", "rationale", "target_instance_hint"],
    "additionalProperties": False,
}


# ---------- client ----------


@lru_cache(maxsize=1)
def _client(api_key: str | None) -> Any:
    """Lazy-init Anthropic client. Cached so multiple calls share one TCP pool."""
    import anthropic

    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _read_image_b64(path: Path) -> tuple[str, str]:
    """Return (media_type, base64-encoded bytes) for an image file."""
    suffix = path.suffix.lower().lstrip(".")
    media_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(suffix, "image/jpeg")
    return media_type, base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _crop_to_bytes(image_path: Path, bbox: BBox, pad: float = 0.04) -> tuple[str, str]:
    """Crop the bbox region (with padding) and return (media_type, b64)."""
    from io import BytesIO

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1 = max(0, int((bbox.x_min - pad) * w))
    y1 = max(0, int((bbox.y_min - pad) * h))
    x2 = min(w, int((bbox.x_max + pad) * w))
    y2 = min(h, int((bbox.y_max + pad) * h))
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h
    crop = img.crop((x1, y1, x2, y2))
    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _system_blocks(text: str, enable_caching: bool) -> list[dict]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if enable_caching:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _image_block(media_type: str, data: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


# ---------- public API ----------


class RemoteVLM:
    def __init__(self, cfg: RemoteVLMConfig | None = None):
        self.cfg = cfg or RemoteVLMConfig()

    # --- attribute extraction ---

    def caption_object(
        self, frame_path: Path, det: ObjectDetection
    ) -> ObjectAttributes:
        """Run structured attribute extraction on the cropped object."""
        media, data = _crop_to_bytes(frame_path, det.bbox)
        client = _client(self.cfg.api_key)

        # Use messages.create with output_config so the API enforces the schema.
        response = client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens_attrs,
            system=_system_blocks(_SYS_ATTRS, self.cfg.enable_caching),
            messages=[{
                "role": "user",
                "content": [
                    _image_block(media, data),
                    {"type": "text", "text": (
                        f"Object class: {det.label}.\n"
                        "Extract structured attributes. Use null when uncertain."
                    )},
                ],
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _ATTRS_SCHEMA,
                }
            },
        )
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        return ObjectAttributes(
            color=payload.get("color"),
            material=payload.get("material"),
            state=payload.get("state"),
            text_on_object=payload.get("text_on_object"),
            fine_grained_label=payload.get("fine_grained_label"),
            distinctive_marks=payload.get("distinctive_marks"),
            raw_caption=text,
        )

    def annotate_frame(
        self, frame_path: Path, dets: list[ObjectDetection]
    ) -> list[ObjectDetection]:
        """Attach attributes to every detection in a frame."""
        return [
            d.model_copy(update={"attributes": self.caption_object(frame_path, d)})
            for d in dets
        ]

    # --- bottom-up tagging (RAM substitute) ---

    def tag_frame(self, frame_path: Path, *, max_tags: int = 25) -> list[str]:
        media, data = _read_image_b64(frame_path)
        client = _client(self.cfg.api_key)
        response = client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens_tags,
            system=_system_blocks(_SYS_TAGS, self.cfg.enable_caching),
            messages=[{
                "role": "user",
                "content": [
                    _image_block(media, data),
                    {"type": "text", "text": "List the distinct objects visible in this frame."},
                ],
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _TAGS_SCHEMA,
                }
            },
        )
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        tags = payload.get("tags") or []
        return [str(t).lower().strip() for t in tags[:max_tags] if t]

    # --- scene-level VLM judge (extra signal alongside the rule cascade) ---

    def judge_scene(
        self,
        clip: ClipCandidate,
        frame_paths: list[Path],
        *,
        scene_graph: list[FrameDetections] | None = None,
    ) -> dict[str, Any]:
        """Ask the VLM to propose an ownership label for the whole clip.

        Returns a dict with keys ``label``, ``confidence``, ``rationale``,
        ``target_instance_hint``. The rule cascade still produces the primary
        auto-label; this is an extra signal the annotator UI can show.
        """
        client = _client(self.cfg.api_key)

        # Build user content: 3 frames in order + textual context.
        user_content: list[dict] = []
        for tag, p in zip(("t-2", "t-1", "t"), frame_paths):
            media, data = _read_image_b64(p)
            user_content.append({"type": "text", "text": f"Frame {tag}:"})
            user_content.append(_image_block(media, data))

        meta_lines = [
            f"clip_id: {clip.clip_id}",
            f"verb: {clip.verb or '—'}",
            f"nouns: {', '.join(clip.nouns) or '—'}",
            f"narration: {clip.narration or '—'}",
        ]
        if scene_graph:
            instances = sorted({
                obj.instance_id for fd in scene_graph for obj in fd.objects if obj.instance_id
            })
            if instances:
                meta_lines.append(f"detected instance ids: {', '.join(instances)}")
        user_content.append({"type": "text", "text": "Clip metadata:\n" + "\n".join(meta_lines)})
        user_content.append({"type": "text", "text": (
            "Decide the ownership label for the most salient target object "
            "(the one referenced by the verb/nouns). Cite which frame and "
            "what visual evidence drives your choice."
        )})

        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens_judge,
            system=_system_blocks(_SYS_JUDGE, self.cfg.enable_caching),
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )
        if self.cfg.enable_thinking_for_judge:
            kwargs["thinking"] = {"type": "adaptive"}

        response = client.messages.create(**kwargs)
        text = next((b.text for b in response.content if b.type == "text"), "{}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"label": "AMBIGUOUS", "confidence": 0.0, "rationale": text, "target_instance_hint": None}


# ---------- factory ----------


def get_client(provider: str | None = None, **kwargs) -> Any:
    """Pick a backend by name. Default: Anthropic.

    Set ``EGOOWN_VLM_PROVIDER=openai`` (or pass ``provider="openai"``) to use
    the OpenAI variant in :mod:`.remote_vlm_openai`.
    """
    p = (provider or os.environ.get("EGOOWN_VLM_PROVIDER") or "anthropic").lower()
    if p == "anthropic":
        cfg = RemoteVLMConfig(**kwargs) if kwargs else None
        return RemoteVLM(cfg)
    if p == "openai":
        from egoownership.detection.remote_vlm_openai import OpenAIRemoteVLM, OpenAIVLMConfig
        cfg = OpenAIVLMConfig(**kwargs) if kwargs else None
        return OpenAIRemoteVLM(cfg)
    raise ValueError(f"Unknown VLM provider: {p!r} (anthropic | openai)")
