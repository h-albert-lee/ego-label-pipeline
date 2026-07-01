"""SAM/SAM-2 object candidate extraction for benchmark JSONL files.

This stage is intentionally conservative: it only discovers candidate object
masks/boxes from existing sparse frame images and skips entries where no usable
object-like mask is found. Class labels are left generic because SAM/SAM-2 is a
segmentation model, not an object classifier; a later VLM stage should name and
select the ownership-relevant target object.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRAME_KEYS = {
    "t-2": "frame_t_minus_2",
    "t-1": "frame_t_minus_1",
    "t": "frame_t",
}

DEFAULT_MASK_MODEL_ID = "facebook/sam-vit-base"


@dataclass(frozen=True)
class Sam2ObjectConfig:
    model_id: str = DEFAULT_MASK_MODEL_ID
    backend: str = "transformers"
    device: str | None = None
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.75
    max_objects_per_frame: int = 30
    nms_iou_threshold: float = 0.90


class Sam2ObjectExtractor:
    """Automatic mask generator wrapper.

    The default backend uses Hugging Face ``pipeline("mask-generation")`` with a
    SAM/SAM-2 checkpoint. This avoids requiring the Meta ``sam2`` repo layout in
    normal runs while still using SAM-family automatic masks.
    """

    def __init__(self, cfg: Sam2ObjectConfig | None = None):
        self.cfg = cfg or Sam2ObjectConfig()
        self._pipeline: Any = None

    def extract(self, image_path: Path) -> list[dict[str, Any]]:
        if self.cfg.backend == "sam3":
            return self.extract_with_prompt(image_path, "")
        if self.cfg.backend != "transformers":
            raise ValueError(
                "Unsupported SAM backend. Use backend='transformers' for automatic "
                "SAM masks or backend='sam3' for concept-prompted SAM-3 boxes."
            )
        pipe = self._load_transformers_pipeline()
        output = pipe(str(image_path))
        return masks_to_objects(
            output,
            min_area_ratio=self.cfg.min_area_ratio,
            max_area_ratio=self.cfg.max_area_ratio,
            max_objects=self.cfg.max_objects_per_frame,
            nms_iou_threshold=self.cfg.nms_iou_threshold,
        )

    def extract_with_prompt(self, image_path: Path, prompt: str) -> list[dict[str, Any]]:
        if self.cfg.backend != "sam3":
            return self.extract(image_path)
        return self._extract_sam3(image_path, prompt)

    def _load_transformers_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Transformers is required for SAM-2 object extraction. Install a "
                "compatible eval environment, then rerun this command."
            ) from exc

        device_arg: int | str | None
        if self.cfg.device in (None, "", "auto"):
            device_arg = None
        elif self.cfg.device.startswith("cuda:"):
            device_arg = int(self.cfg.device.split(":", 1)[1])
        else:
            device_arg = self.cfg.device

        try:
            kwargs = {"model": self.cfg.model_id}
            if device_arg is not None:
                kwargs["device"] = device_arg
            self._pipeline = pipeline("mask-generation", **kwargs)
        except Exception as exc:  # noqa: BLE001
            if "sam2_video" in str(exc):
                raise RuntimeError(
                    "The checkpoint uses the `sam2_video` architecture, which this "
                    "Transformers install cannot load. Run this command in a separate "
                    "SAM-2 environment with a newer Transformers build, for example:\n"
                    "  pip install 'torch>=2.5.1' 'torchvision>=0.20.1'\n"
                    "  pip install git+https://github.com/huggingface/transformers.git\n"
                    "Do not upgrade the current EgoGPT/test environment in-place; its "
                    "Torch/Transformers versions conflict with SAM-2."
                ) from exc
            raise RuntimeError(
                "Could not load SAM-2 mask-generation pipeline. If this checkpoint "
                "is unavailable in your Transformers build, install/upgrade the "
                "SAM-2 stack or pass a compatible SAM/SAM-2 mask-generation "
                f"checkpoint. model_id={self.cfg.model_id!r}; original error: {exc}"
            ) from exc
        return self._pipeline

    def _extract_sam3(self, image_path: Path, prompt: str) -> list[dict[str, Any]]:
        prompt = (prompt or "").strip()
        if not prompt:
            return []
        try:
            import torch
            from PIL import Image
            from transformers import Sam3Model, Sam3Processor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "SAM-3 extraction requires a Transformers build with SAM-3 support. "
                "Run from an environment that has sam2/transformers with SAM-3 support installed."
            ) from exc

        if self._pipeline is None:
            try:
                processor = Sam3Processor.from_pretrained(self.cfg.model_id)
                model = Sam3Model.from_pretrained(self.cfg.model_id)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Could not load SAM-3. Use a SAM-3 model id such as "
                    "`facebook/sam3-base` in the sam2hf environment. "
                    f"model_id={self.cfg.model_id!r}; original error: {exc}"
                ) from exc
            device = self.cfg.device
            if device in (None, "", "auto"):
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            self._pipeline = {"processor": processor, "model": model, "device": device}

        bundle = self._pipeline
        processor = bundle["processor"]
        model = bundle["model"]
        device = bundle["device"]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.3,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        width, height = image.size
        objects: list[dict[str, Any]] = []
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
            bbox = {
                "x_min": max(0.0, min(1.0, x1 / width)),
                "y_min": max(0.0, min(1.0, y1 / height)),
                "x_max": max(0.0, min(1.0, x2 / width)),
                "y_max": max(0.0, min(1.0, y2 / height)),
            }
            area = max(0.0, bbox["x_max"] - bbox["x_min"]) * max(0.0, bbox["y_max"] - bbox["y_min"])
            if area <= self.cfg.min_area_ratio or area >= self.cfg.max_area_ratio:
                continue
            objects.append(
                {
                    "label": prompt,
                    "bbox": bbox,
                    "score": float(score.detach().cpu().item()),
                    "area_ratio": area,
                    "source": "sam3",
                    "instance_id": f"sam3_obj_{idx:03d}",
                }
            )
        objects.sort(key=lambda item: (item.get("score") or 0.0, item.get("area_ratio") or 0.0), reverse=True)
        return _nms_objects(objects, iou_threshold=self.cfg.nms_iou_threshold)[: self.cfg.max_objects_per_frame]


def write_sam2_object_jsonl(
    input_path: Path,
    frames_root: Path,
    out_path: Path,
    *,
    extractor: Sam2ObjectExtractor | Callable[[Path], list[dict[str, Any]]],
    video_roots: dict[str, Path] | None = None,
    extracted_frames_dir: Path | None = None,
    skip_source_datasets: set[str] | None = None,
    frame_tags: Iterable[str] = ("t",),
    limit: int | None = None,
    show_progress: bool = True,
    frame_backend: str = "ffmpeg",
) -> int:
    """Read benchmark JSONL, add SAM-2 object candidates, skip no-object rows."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if not frames_root.exists() and not video_roots:
        raise FileNotFoundError(f"Frames root not found: {frames_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = iter_jsonl_records(input_path)
    total = _count_jsonl(input_path) if show_progress else None
    if limit is not None and total is not None:
        total = min(total, limit)
    if show_progress:
        from tqdm.auto import tqdm

        rows = tqdm(rows, total=total, unit="entry", desc="SAM-2 object extraction")

    count = 0
    processed = 0
    tags = tuple(frame_tags)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if limit is not None and processed >= limit:
                break
            processed += 1
            if (row.get("source_dataset") or "") in (skip_source_datasets or set()):
                continue
            updated, object_count = add_sam2_objects_to_entry(
                row,
                frames_root,
                extractor=extractor,
                video_roots=video_roots,
                extracted_frames_dir=extracted_frames_dir,
                frame_tags=tags,
                frame_backend=frame_backend,
            )
            if object_count <= 0:
                continue
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
    return count


def add_sam2_objects_to_entry(
    row: dict[str, Any],
    frames_root: Path,
    *,
    extractor: Sam2ObjectExtractor | Callable[[Path], list[dict[str, Any]]],
    video_roots: dict[str, Path] | None = None,
    extracted_frames_dir: Path | None = None,
    frame_tags: Iterable[str] = ("t",),
    frame_backend: str = "ffmpeg",
) -> tuple[dict[str, Any], int]:
    """Return a copied row with SAM-2 objects inserted into selected frames."""

    updated = dict(row)
    total_objects = 0
    extraction_meta: dict[str, Any] = {}
    for tag in frame_tags:
        frame_key = FRAME_KEYS.get(tag)
        if not frame_key:
            continue
        frame = row.get(frame_key)
        image_path = resolve_frame_image_path(frame, frames_root)
        if image_path is None and video_roots and extracted_frames_dir is not None:
            image_path = extract_frame_for_row(
                row,
                tag,
                video_roots=video_roots,
                out_dir=extracted_frames_dir,
                backend=frame_backend,
            )
        if image_path is None:
            continue
        objects = (
            extractor.extract(image_path)
            if isinstance(extractor, Sam2ObjectExtractor)
            else extractor(image_path)
        )
        if not objects:
            continue
        frame_copy = dict(frame or {})
        frame_copy["objects"] = [
            {
                **obj,
                "instance_id": obj.get("instance_id") or f"{tag}_sam2_obj_{idx:03d}",
            }
            for idx, obj in enumerate(objects)
        ]
        frame_copy["frame_path"] = str(_display_frame_path(image_path, frames_root))
        updated[frame_key] = frame_copy
        extraction_meta[tag] = {
            "frame_path": frame_copy["frame_path"],
            "num_objects": len(objects),
        }
        total_objects += len(objects)

    if total_objects > 0:
        source = dict(updated.get("source") or {})
        source["sam2_object_extraction"] = {
            "num_objects": total_objects,
            "frames": extraction_meta,
            "note": "SAM-2 mask candidates are generic object proposals; use VLM to select/name target.",
        }
        updated["source"] = source
    return updated, total_objects


def extract_frame_for_row(
    row: dict[str, Any],
    tag: str,
    *,
    video_roots: dict[str, Path],
    out_dir: Path,
    backend: str = "ffmpeg",
) -> Path | None:
    source_dataset = str(row.get("source_dataset") or "")
    video_path = resolve_video_path(row, video_roots)
    if video_path is None:
        return None
    timestamp = _timestamp_for_tag(row, tag)
    if timestamp is None:
        return None
    dataset = _safe_path_part(source_dataset or "unknown_dataset")
    video_id = _safe_path_part(str(row.get("video_id") or "unknown_video"))
    clip_id = _safe_path_part(str(row.get("clip_id") or f"{video_id}_{timestamp:.3f}"))
    dest = out_dir / dataset / video_id / f"{clip_id}__{tag}.jpg"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if backend != "ffmpeg":
        raise ValueError("SAM-2 JSONL video-frame extraction currently supports backend='ffmpeg'")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return None
    return dest if dest.exists() else None


def resolve_video_path(row: dict[str, Any], video_roots: dict[str, Path]) -> Path | None:
    source_dataset = str(row.get("source_dataset") or "")
    video_id = str(row.get("video_id") or "")
    root = video_roots.get(source_dataset)
    if root is None or not video_id:
        return None
    candidates: list[Path] = []
    if source_dataset == "egolife":
        parts = video_id.split("_")
        if len(parts) >= 4 and parts[0].startswith("DAY") and parts[1].startswith("A"):
            participant = f"{parts[1]}_{parts[2]}"
            day = parts[0]
            candidates.append(root / participant / day / f"{video_id}.mp4")
        candidates.append(root / f"{video_id}.mp4")
    elif source_dataset == "ego4d_fho":
        candidates.append(root / f"{video_id}.mp4")
        candidates.extend(sorted(root.glob(f"{video_id}.mp4*")))
    else:
        candidates.append(root / f"{video_id}.mp4")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _timestamp_for_tag(row: dict[str, Any], tag: str) -> float | None:
    key = {"t-2": "t_minus_2_sec", "t-1": "t_minus_1_sec", "t": "t_sec"}.get(tag)
    if not key:
        return None
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_frame_image_path(frame: Any, frames_root: Path) -> Path | None:
    if not isinstance(frame, dict):
        return None
    value = frame.get("frame_path") or frame.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [frames_root / path]
    # Some callers pass the dataset root instead of the frames root.
    if not path.is_absolute():
        candidates.append(frames_root / "frames" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def masks_to_objects(
    output: Any,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    max_objects: int,
    nms_iou_threshold: float,
) -> list[dict[str, Any]]:
    masks, scores = _extract_masks_and_scores(output)
    objects: list[dict[str, Any]] = []
    for idx, mask in enumerate(masks):
        obj = _mask_to_object(
            mask,
            score=scores[idx] if idx < len(scores) else None,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        if obj is not None:
            objects.append(obj)
    objects.sort(key=lambda item: (item.get("score") or 0.0, item.get("area_ratio") or 0.0), reverse=True)
    objects = _nms_objects(objects, iou_threshold=nms_iou_threshold)
    return objects[:max_objects]


def _extract_masks_and_scores(output: Any) -> tuple[list[Any], list[float | None]]:
    if isinstance(output, dict):
        masks = _first_present(output, "masks", "segmentation")
        scores = _first_present(output, "scores", "predicted_iou")
        return _as_list(masks), _as_score_list(scores)
    if isinstance(output, list):
        masks = []
        scores = []
        for item in output:
            if isinstance(item, dict):
                mask = _first_present(item, "mask", "segmentation")
                if mask is not None:
                    masks.append(mask)
                    scores.append(_first_present(item, "score", "predicted_iou"))
        return masks, scores
    return [], []


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return [item for item in value.detach().cpu()]
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [item for item in value]
    except Exception:
        pass
    return [value]


def _as_score_list(value: Any) -> list[float | None]:
    scores: list[float | None] = []
    for item in _as_list(value):
        if item is None:
            scores.append(None)
            continue
        try:
            if hasattr(item, "numel") and item.numel() == 1:
                item = item.item()
            scores.append(float(item))
        except Exception:
            scores.append(None)
    return scores


def _mask_to_object(
    mask: Any,
    *,
    score: float | None,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict[str, Any] | None:
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.ndim != 2:
        return None
    binary = arr > 0
    height, width = binary.shape
    area = int(binary.sum())
    image_area = max(1, width * height)
    area_ratio = area / image_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return None
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = {
        "x_min": x1 / width,
        "y_min": y1 / height,
        "x_max": (x2 + 1) / width,
        "y_max": (y2 + 1) / height,
    }
    return {
        "label": "sam2_object",
        "bbox": bbox,
        "score": float(score) if score is not None else None,
        "area_ratio": area_ratio,
        "mask_area": area,
        "source": "sam2",
    }


def _nms_objects(objects: list[dict[str, Any]], *, iou_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for obj in objects:
        bbox = obj.get("bbox")
        if not isinstance(bbox, dict):
            continue
        if all(_bbox_iou(bbox, other.get("bbox") or {}) < iou_threshold for other in kept):
            kept.append(obj)
    return kept


def _bbox_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = a.get("x_min", 0.0), a.get("y_min", 0.0), a.get("x_max", 0.0), a.get("y_max", 0.0)
    bx1, by1, bx2, by2 = b.get("x_min", 0.0), b.get("y_min", 0.0), b.get("x_max", 0.0), b.get("y_max", 0.0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _display_frame_path(path: Path, frames_root: Path) -> Path:
    try:
        return path.relative_to(frames_root)
    except ValueError:
        return path


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "unknown"
