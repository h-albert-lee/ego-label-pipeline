"""Per-object attribute extraction (Q1).

Strategy:

1. Crop the bbox region with a small padding.
2. Hand the crop to a VLM (BLIP-2 / LLaVA / Qwen-VL — whichever is locally
   available). We default to BLIP-2 because it's small and license-permissive.
3. Parse the freeform caption into the structured ``ObjectAttributes`` slots
   using a simple regex / keyword pass — perfect parsing isn't needed for a
   draft pipeline; the human reviewer fixes the rest in the UI.

If no VLM is available we still emit ``ObjectAttributes`` with ``raw_caption=None``
so downstream code is uniform.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from egoownership.schema import ObjectAttributes, ObjectDetection

_DEFAULT_MODEL = "Salesforce/blip2-opt-2.7b"

_COLOR_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "white", "black", "gray", "grey", "brown", "beige", "silver", "gold",
    "transparent", "clear",
}
_MATERIAL_WORDS = {
    "ceramic", "plastic", "glass", "metal", "wood", "wooden", "paper",
    "cardboard", "leather", "stainless", "porcelain", "stone",
}
_STATE_WORDS = {
    "empty", "full", "filled", "open", "closed", "broken",
    "wet", "dry", "stained", "clean",
}


@lru_cache(maxsize=1)
def _load_vlm(model_id: str = _DEFAULT_MODEL):
    try:
        import torch
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
        return processor, model, device
    except Exception:  # noqa: BLE001
        return None


def _crop(image_path: Path, det: ObjectDetection, pad: float = 0.04):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    bb = det.bbox
    x1 = max(0, int((bb.x_min - pad) * w))
    y1 = max(0, int((bb.y_min - pad) * h))
    x2 = min(w, int((bb.x_max + pad) * w))
    y2 = min(h, int((bb.y_max + pad) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def _caption_with_vlm(crop_image, prompt: str) -> str | None:
    bundle = _load_vlm()
    if bundle is None or crop_image is None:
        return None
    import torch

    processor, model, device = bundle
    inputs = processor(images=crop_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text or None


def _parse_caption(caption: str | None, det: ObjectDetection) -> ObjectAttributes:
    attrs = ObjectAttributes(raw_caption=caption)
    if not caption:
        return attrs
    low = caption.lower()
    for c in _COLOR_WORDS:
        if c in low:
            attrs.color = c
            break
    for m in _MATERIAL_WORDS:
        if m in low:
            attrs.material = m
            break
    for s in _STATE_WORDS:
        if s in low:
            attrs.state = s
            break
    # Distinctive marks: any word that looks like a logo / "with X".
    m = re.search(r"with (a |an |the )?([\w \-]{3,30})", low)
    if m:
        attrs.distinctive_marks = m.group(2).strip()
    # Fine-grained label: prefer two-word adjective + noun pattern.
    m = re.search(r"\b([a-z]+)\s+" + re.escape(det.label.lower()) + r"\b", low)
    if m:
        attrs.fine_grained_label = f"{m.group(1)} {det.label.lower()}"
    return attrs


def annotate_object(image_path: Path, det: ObjectDetection) -> ObjectAttributes:
    """Run the VLM-then-parse pipeline for one detection."""
    crop = _crop(image_path, det)
    caption = _caption_with_vlm(
        crop, prompt=f"Describe the {det.label} including color, material, and state."
    )
    return _parse_caption(caption, det)


def annotate_frame_objects(image_path: Path, dets: list[ObjectDetection]) -> list[ObjectDetection]:
    """In-place attach attributes to each detection in ``dets``. Returns a list."""
    out: list[ObjectDetection] = []
    for det in dets:
        attrs = annotate_object(image_path, det)
        out.append(det.model_copy(update={"attributes": attrs}))
    return out
