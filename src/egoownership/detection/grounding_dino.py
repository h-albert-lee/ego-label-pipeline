"""Grounding DINO wrapper for text-prompted bbox proposals.

Uses the HuggingFace ``transformers`` pipeline so we don't pin a
checkpoint-specific branch. Model is loaded lazily because torch + a ~170M
param checkpoint is a heavy import.

Prompts follow the official Grounding DINO convention: periods separate
concepts, e.g. ``"a cup. a plate. a fork. a hand."``. We always include
``"a hand."`` so the labeler can use hand location as prior evidence for MINE.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from egoownership.schema import BBox, ObjectDetection

if TYPE_CHECKING:  # heavy deps
    from PIL import Image  # noqa: F401

_DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


@dataclass
class DinoConfig:
    model_id: str = _DEFAULT_MODEL
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    device: str | None = None  # None → auto


@lru_cache(maxsize=2)
def _load_model(model_id: str, device: str | None):
    """Cached loader — call sites share one model in memory."""

    import torch  # local import
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(dev)
    model.eval()
    return processor, model, dev


def build_prompt(nouns: list[str]) -> str:
    """Turn a noun list into the ``"a cup. a plate. a hand."`` format."""

    seen: list[str] = []
    for n in [*nouns, "hand"]:
        token = n.strip().lower().replace("_", " ")
        if token and token not in seen:
            seen.append(token)
    return ". ".join(f"a {n}" for n in seen) + "."


def detect_objects(
    image_path: Path,
    prompt: str,
    cfg: DinoConfig | None = None,
) -> list[ObjectDetection]:
    """Run Grounding DINO on one image and return normalized-coord detections."""

    import torch
    from PIL import Image

    cfg = cfg or DinoConfig()
    processor, model, device = _load_model(cfg.model_id, cfg.device)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
        target_sizes=[(h, w)],
    )[0]

    detections: list[ObjectDetection] = []
    for box, score, label in zip(
        results["boxes"].tolist(), results["scores"].tolist(), results["labels"]
    ):
        x1, y1, x2, y2 = box
        detections.append(
            ObjectDetection(
                label=str(label),
                bbox=BBox.from_xyxy_abs(x1, y1, x2, y2, w, h),
                score=float(score),
            )
        )
    return detections
