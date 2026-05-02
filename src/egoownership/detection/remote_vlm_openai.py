"""OpenAI variant of the remote VLM client.

Mirror of :mod:`.remote_vlm` for shops that prefer GPT-4o / GPT-4.1 vision over
Anthropic. Same three operations, same return shapes, same cropping path —
only the request layer differs.

Lives in its own module so the Anthropic-default codebase doesn't accidentally
import the ``openai`` SDK, and so prompt caching for Anthropic and OpenAI's
own caching mechanism don't get conflated.
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
)

_DEFAULT_MODEL = os.environ.get("EGOOWN_OPENAI_MODEL", "gpt-4o")


@dataclass
class OpenAIVLMConfig:
    model: str = _DEFAULT_MODEL
    api_key: str | None = None
    max_tokens_attrs: int = 800
    max_tokens_tags: int = 600
    max_tokens_judge: int = 4000


_SYS_ATTRS = (
    "You are a vision annotator extracting structured attributes from a single "
    "cropped object image. Return only the JSON requested. Set fields to null "
    "when uncertain. Do not invent details."
)
_SYS_TAGS = (
    "You are a vision tagger that lists every distinct physical object visible "
    "in a single first-person video frame. Output JSON with a 'tags' array of "
    "lowercase singular nouns. Avoid attributes; cap at 25 entries; include "
    "'hand' when visible."
)
_SYS_JUDGE = (
    "You are an expert annotator for the Egocentric Implicit Ownership "
    "benchmark. Given three sparse frames (t-2, t-1, t) and clip metadata, "
    "decide ownership: MINE / PERSON_k / SHARED / AMBIGUOUS. Use the full "
    "trajectory; cite specific visual evidence in your rationale."
)


@lru_cache(maxsize=1)
def _client(api_key: str | None) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key) if api_key else OpenAI()


def _read_image_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _crop_to_b64(image_path: Path, bbox: BBox, pad: float = 0.04) -> str:
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
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _img_part(b64: str, mime: str = "image/jpeg") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class OpenAIRemoteVLM:
    def __init__(self, cfg: OpenAIVLMConfig | None = None):
        self.cfg = cfg or OpenAIVLMConfig()

    def caption_object(
        self, frame_path: Path, det: ObjectDetection
    ) -> ObjectAttributes:
        client = _client(self.cfg.api_key)
        b64 = _crop_to_b64(frame_path, det.bbox)
        resp = client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens_attrs,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYS_ATTRS},
                {
                    "role": "user",
                    "content": [
                        _img_part(b64),
                        {
                            "type": "text",
                            "text": (
                                f"Object class: {det.label}.\n"
                                "Return JSON with keys color, material, state, "
                                "text_on_object, fine_grained_label, distinctive_marks. "
                                "Set unknown fields to null."
                            ),
                        },
                    ],
                },
            ],
        )
        text = resp.choices[0].message.content or "{}"
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
        return [
            d.model_copy(update={"attributes": self.caption_object(frame_path, d)})
            for d in dets
        ]

    def tag_frame(self, frame_path: Path, *, max_tags: int = 25) -> list[str]:
        client = _client(self.cfg.api_key)
        b64 = _read_image_b64(frame_path)
        resp = client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens_tags,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYS_TAGS},
                {
                    "role": "user",
                    "content": [
                        _img_part(b64),
                        {
                            "type": "text",
                            "text": "Return JSON: {\"tags\": [\"cup\", \"plate\", ...]}",
                        },
                    ],
                },
            ],
        )
        text = resp.choices[0].message.content or "{}"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        tags = payload.get("tags") or []
        return [str(t).lower().strip() for t in tags[:max_tags] if t]

    def judge_scene(
        self,
        clip: ClipCandidate,
        frame_paths: list[Path],
        *,
        scene_graph: list[FrameDetections] | None = None,
    ) -> dict[str, Any]:
        client = _client(self.cfg.api_key)
        user: list[dict] = []
        for tag, p in zip(("t-2", "t-1", "t"), frame_paths):
            user.append({"type": "text", "text": f"Frame {tag}:"})
            user.append(_img_part(_read_image_b64(p)))

        meta = [
            f"clip_id: {clip.clip_id}",
            f"verb: {clip.verb or '—'}",
            f"nouns: {', '.join(clip.nouns) or '—'}",
            f"narration: {clip.narration or '—'}",
        ]
        if scene_graph:
            ids = sorted({
                o.instance_id for fd in scene_graph for o in fd.objects if o.instance_id
            })
            if ids:
                meta.append(f"detected instance ids: {', '.join(ids)}")
        user.append({"type": "text", "text": "Clip metadata:\n" + "\n".join(meta)})
        user.append({
            "type": "text",
            "text": (
                "Return JSON with keys: label "
                "(MINE | PERSON_k | SHARED | AMBIGUOUS), confidence (0..1), "
                "rationale (one sentence citing visual evidence), "
                "target_instance_hint (string or null)."
            ),
        })

        resp = client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens_judge,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYS_JUDGE},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "label": "AMBIGUOUS",
                "confidence": 0.0,
                "rationale": text,
                "target_instance_hint": None,
            }
