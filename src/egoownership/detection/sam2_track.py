"""SAM-2 backward bbox tracking for the one-pass-labels stage.

one-pass-labels already has a bbox for frame ``t`` (from SAM-3/CAT-V) and
already extracts the t-2/t-1/t sparse frame images. Rather than re-tracking
the whole clip, this propagates the known frame-t box *backward* directly on
those three already-extracted images, so temporal evidence (zone/held_by
changes over time) has real boxes at t-2/t-1 instead of only the reference
frame.
"""

from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from egoownership.schema import BBox

_HF_MODEL_IDS = {
    "facebook/sam2-hiera-tiny", "facebook/sam2-hiera-small",
    "facebook/sam2-hiera-base-plus", "facebook/sam2-hiera-large",
    "facebook/sam2.1-hiera-tiny", "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus", "facebook/sam2.1-hiera-large",
}


def _is_hf_model_id(model_path: str) -> bool:
    return model_path in _HF_MODEL_IDS or (not Path(model_path).exists() and "/" in model_path)


def _determine_model_cfg(model_path: str) -> str:
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    if "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    raise ValueError(f"Unknown SAM-2 model size in path: {model_path}")


def _silence_sam2_progress_bars() -> None:
    """SAM-2 hardcodes tqdm progress bars ("frame loading (JPEG)",
    "propagate in video") with no argument to disable them. Patch the
    ``tqdm`` name bound in its own modules to a passthrough so
    init_state()/propagate_in_video() run quietly instead.
    """
    passthrough = lambda iterable, *args, **kwargs: iterable  # noqa: E731
    for module_name in (
        "sam2.utils.misc",
        "sam2.sam2_video_predictor",
        "sam2.sam2_video_predictor_legacy",
    ):
        try:
            module = __import__(module_name, fromlist=["tqdm"])
        except ImportError:
            continue
        module.tqdm = passthrough


@lru_cache(maxsize=2)
def _load_predictor(model_id: str, device: str):
    _silence_sam2_progress_bars()
    if _is_hf_model_id(model_id):
        from sam2.build_sam import build_sam2_video_predictor_hf

        predictor = build_sam2_video_predictor_hf(model_id, device=device)
    else:
        if not Path(model_id).exists():
            raise FileNotFoundError(f"SAM-2 checkpoint not found: {model_id}")
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(_determine_model_cfg(model_id), model_id, device=device)
    predictor.fill_hole_area = 0
    return predictor


def track_bbox_backward(
    frame_paths: dict[str, Path],
    reference_bbox: BBox,
    *,
    reference_tag: str = "t",
    model_id: str = "facebook/sam2.1-hiera-base-plus",
    device: str = "cuda",
) -> dict[str, BBox]:
    """Propagate ``reference_bbox`` (known at ``reference_tag``) backward.

    Operates directly on the already-extracted sparse frame images in
    ``frame_paths`` (keys among ``"t-2"``, ``"t-1"``, ``"t"``) — no whole-clip
    re-decode. Returns a dict of tag -> BBox for every tag SAM-2 could still
    find the object at (including ``reference_tag`` itself); a tag is simply
    absent if the propagated mask went empty at that frame.
    """
    chronological = [tag for tag in ("t-2", "t-1", "t") if tag in frame_paths and frame_paths[tag].exists()]
    if reference_tag not in chronological or len(chronological) < 2:
        return {reference_tag: reference_bbox} if reference_tag in chronological else {}

    ref_pos = chronological.index(reference_tag)
    # Reverse the frames up to and including the reference tag so SAM-2 walks
    # backward in time from the one frame we actually have a box for; any
    # frames after the reference tag (shouldn't normally occur here, since
    # one-pass-labels' reference is always "t", the last tag) are appended
    # in forward order.
    order = list(reversed(chronological[: ref_pos + 1])) + chronological[ref_pos + 1 :]

    import numpy as np
    import torch
    from PIL import Image

    with Image.open(frame_paths[reference_tag]) as im:
        width, height = im.size
    x1 = int(reference_bbox.x_min * width)
    y1 = int(reference_bbox.y_min * height)
    x2 = int(reference_bbox.x_max * width)
    y2 = int(reference_bbox.y_max * height)

    predictor = _load_predictor(model_id, device)
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam2_one_pass_"))
    try:
        for i, tag in enumerate(order):
            shutil.copy(frame_paths[tag], tmp_dir / f"{i:06d}.jpg")

        autocast_ctx = (
            torch.autocast(device.split(":")[0], dtype=torch.float16)
            if device.startswith("cuda")
            else __import__("contextlib").nullcontext()
        )
        results: dict[str, BBox] = {reference_tag: reference_bbox}
        with torch.inference_mode(), autocast_ctx:
            state = predictor.init_state(str(tmp_dir), offload_video_to_cpu=True)
            predictor.add_new_points_or_box(state, box=(x1, y1, x2, y2), frame_idx=0, obj_id=0)
            for seq_idx, _object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0, reverse=False):
                if seq_idx == 0 or seq_idx >= len(order):
                    continue
                mask_arr = masks[0][0].cpu().numpy() > 0.0
                nz = np.argwhere(mask_arr)
                if len(nz) == 0:
                    continue
                y_min, x_min = nz.min(axis=0)
                y_max, x_max = nz.max(axis=0)
                results[order[seq_idx]] = BBox(
                    x_min=float(x_min) / width,
                    y_min=float(y_min) / height,
                    x_max=float(x_max) / width,
                    y_max=float(y_max) / height,
                )
            del state
        return results
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
