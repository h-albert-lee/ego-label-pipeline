"""SAM/SAM-2 mask refinement for boxes proposed by Grounding DINO.

Uses HuggingFace transformers SAM/SAM-2 for simplicity. For EDA we only care about
the *refined bbox* derived from the mask, not the mask itself, so we keep the
surface minimal. If you need the raw mask, extend this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from egoownership.schema import BBox, ObjectDetection

_DEFAULT_MODEL = "facebook/sam-vit-base"


@dataclass
class SamConfig:
    model_id: str = _DEFAULT_MODEL
    device: str | None = None


@lru_cache(maxsize=2)
def _load_model(model_id: str, device: str | None):
    import torch
    from transformers import AutoModel, AutoProcessor

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    return processor, model, dev


def refine_boxes(
    image_path: Path,
    detections: list[ObjectDetection],
    cfg: SamConfig | None = None,
) -> list[ObjectDetection]:
    """Replace each bbox with the tight bbox around its SAM2 mask.

    Drops detections whose mask is empty (typical failure mode on distractors).
    """

    if not detections:
        return []

    import numpy as np
    import torch
    from PIL import Image

    cfg = cfg or SamConfig()
    processor, model, device = _load_model(cfg.model_id, cfg.device)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    # Denormalize boxes to absolute (x1, y1, x2, y2) for SAM input.
    boxes_abs = [
        [d.bbox.x_min * w, d.bbox.y_min * h, d.bbox.x_max * w, d.bbox.y_max * h]
        for d in detections
    ]

    inputs = processor(
        images=image,
        input_boxes=[boxes_abs],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
        reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
    )[0].numpy()  # (N, 1, H, W) → (N, H, W)

    refined: list[ObjectDetection] = []
    for det, mask in zip(detections, masks):
        m = mask[0] if mask.ndim == 3 else mask
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        refined.append(
            det.model_copy(update={"bbox": BBox.from_xyxy_abs(x1, y1, x2, y2, w, h)})
        )
    return refined
