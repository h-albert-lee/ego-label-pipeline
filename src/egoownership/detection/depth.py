"""Monocular depth estimation hook.

Uses Depth Anything v2 (small) when available. Output is a per-pixel relative
depth map normalized to [0, 1]; we only retain the *mean depth* over each
object bbox because that's what zones use. Falls back to a no-op (None depth)
otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from egoownership.schema import ObjectDetection

_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@lru_cache(maxsize=1)
def _load_depth(model_id: str = _DEFAULT_MODEL):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        return proc, model, device
    except Exception:  # noqa: BLE001
        return None


def annotate_object_depth(image_path: Path, dets: list[ObjectDetection]) -> list[ObjectDetection]:
    bundle = _load_depth()
    if bundle is None or not dets:
        return dets

    import numpy as np
    import torch
    from PIL import Image

    proc, model, device = bundle
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    # Normalize 0..1 (closer = higher).
    pmin, pmax = pred.min(), pred.max()
    norm = (pred - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(pred)

    out: list[ObjectDetection] = []
    for det in dets:
        x1 = int(det.bbox.x_min * w)
        y1 = int(det.bbox.y_min * h)
        x2 = int(det.bbox.x_max * w)
        y2 = int(det.bbox.y_max * h)
        if x2 > x1 and y2 > y1:
            mean = float(norm[y1:y2, x1:x2].mean())
        else:
            mean = None
        out.append(det.model_copy(update={"mean_depth": mean}))
    return out


def estimate_wearer_depth_band(dets: list[ObjectDetection]) -> tuple[float, float] | None:
    """Pick a depth threshold that separates "near (wearer)" from "far"."""
    depths = [d.mean_depth for d in dets if d.mean_depth is not None]
    if len(depths) < 3:
        return None
    depths_sorted = sorted(depths)
    # Top-quartile depth is "near"; everything below is far / opponent.
    q3 = depths_sorted[int(0.75 * len(depths_sorted))]
    return (q3, 1.0)
