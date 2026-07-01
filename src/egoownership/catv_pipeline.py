"""Multi-dataset table-object caption construction with SAM/SAM-2 + CAT-V.

Pipeline:
1. Iterate dataset-specific caption/candidate records (EgoLife, Ego4D, …).
2. Keep records mentioning a table and at least one concrete table-top noun.
3. Extract a reference frame and run SAM/SAM-3 for object boxes.
4. For each selected object box, call a CAT-V subprocess adapter to create an
   object-centric caption.
5. Optionally build sparse benchmark frames and ownership labels.

CAT-V is a separate research repo with its own environment, so this module
uses a command template instead of importing CAT-V directly.
"""

from __future__ import annotations

import json
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFont

from egoownership.catv_datasets import (
    CatVDatasetAdapter,
    get_catv_dataset_adapter,
    normalize_dataset_id,
    resolve_egolife_video_path,
    resolve_egolife_video_segment,
)
from egoownership.catv_io import (
    count_jsonl,
    iter_jsonl,
    normalize_object_noun,
    safe_path_part,
)
from egoownership.sam2_objects import Sam2ObjectExtractor, _nms_objects


class ObjectCaptioner(Protocol):
    model_id: str

    def caption(
        self,
        *,
        video_path: Path,
        first_frame_path: Path,
        bbox: dict[str, float],
        record: dict[str, Any],
        object_index: int,
    ) -> str:
        """Return an object-centric caption for one object box."""


@dataclass
class SparseFrameSelection:
    frame_paths: dict[str, Path]
    frame_times: dict[str, float]
    target_frame: Path
    target_objects: list[dict[str, Any]]
    visible_sample_count: int
    sampled_frame_count: int
    frame_target_objects: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class CatVCommandCaptioner:
    """Run CAT-V through a shell command template.

    Available placeholders:
    ``{video_path}``, ``{first_frame_path}``, ``{bbox_path}``, ``{output_json}``,
    ``{work_dir}``, ``{object_index}``, ``{caption}``, ``{object_nouns}``,
    ``{start_sec}``, ``{end_sec}``, ``{duration_sec}``.

    The command should write a JSON/JSONL/text file to ``{output_json}``. The
    parser accepts common CAT-V-like fields such as ``model_answer``, ``caption``,
    ``object_caption``, or a list of event dictionaries.
    """

    command_template: str
    model_id: str = "CAT-V"
    work_root: Path = Path("outputs/egolife_catv_work")
    visualization_root: Path = Path("outputs/egolife_catv_visualizations")
    keep_work_dir: bool = False
    suppress_output: bool = True
    save_visualizations: bool = True
    last_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if "/path/to/" in self.command_template:
            raise ValueError(
                "--catv-command-template still contains the placeholder '/path/to/'. "
                "Clone/install CAT-V first, then pass the real wrapper script path."
            )
        self.command_template = _upgrade_catv_wrapper_template(self.command_template)

    def caption(
        self,
        *,
        video_path: Path,
        first_frame_path: Path,
        bbox: dict[str, float],
        record: dict[str, Any],
        object_index: int,
    ) -> str:
        work_dir = self._record_work_dir(record, object_index)
        work_dir.mkdir(parents=True, exist_ok=True)
        if not first_frame_path.exists():
            timestamp_sec = float(
                record.get("reference_frame_sec")
                or record.get("first_frame_sec")
                or record.get("catv_start_sec")
                or 0.0
            )
            first_frame_path = work_dir / "first_frame.jpg"
            if _ffmpeg_extract_frame(video_path, first_frame_path, timestamp_sec, force=True) is None:
                raise FileNotFoundError(
                    f"Could not extract first frame from {video_path} at {timestamp_sec:.3f}s"
                )
        bbox_path = work_dir / "bbox.txt"
        output_json = work_dir / "catv_caption.json"
        _write_absolute_bbox(bbox_path, bbox, first_frame_path)
        target_noun = str((record.get("object") or {}).get("target_noun") or "").strip()
        object_nouns = target_noun or ",".join(record.get("nouns") or [])
        values = {
            "video_path": shlex.quote(str(video_path)),
            "first_frame_path": shlex.quote(str(first_frame_path)),
            "bbox_path": shlex.quote(str(bbox_path)),
            "output_json": shlex.quote(str(output_json)),
            "work_dir": shlex.quote(str(work_dir)),
            "object_index": str(object_index),
            "caption": shlex.quote(object_nouns),
            "object_nouns": shlex.quote(object_nouns),
            "start_sec": f"{float(record.get('catv_start_sec') or record.get('start_sec') or 0.0):.3f}",
            "end_sec": f"{float(record.get('catv_end_sec') or record.get('end_sec') or 0.0):.3f}",
            "duration_sec": f"{float(record.get('catv_duration_sec') or 5.0):.3f}",
        }
        self.last_metadata = {}
        cmd = self.command_template.format(**values)
        try:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=self.suppress_output,
                text=self.suppress_output,
            )
        except subprocess.CalledProcessError as exc:
            if not self.keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
            output_tail = _subprocess_output_tail(exc)
            raise RuntimeError(
                "CAT-V command failed. Check that the conda env exists, the script path "
                f"is real, and CAT-V writes {output_json}. Command: {cmd}{output_tail}"
            ) from exc
        caption = _read_catv_caption(output_json)
        if self.save_visualizations:
            self.last_metadata = self._persist_visualization(output_json, record, object_index)
        if not self.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return caption

    def _record_work_dir(self, record: dict[str, Any], object_index: int) -> Path:
        group_a, group_b = _record_storage_parts(record)
        record_id = safe_path_part(str(record.get("id") or record.get("clip_id") or "caption"))
        return self.work_root / group_a / group_b / f"{record_id}__obj{object_index:03d}"

    def _persist_visualization(
        self,
        output_json: Path,
        record: dict[str, Any],
        object_index: int,
    ) -> dict[str, Any]:
        metadata = _read_json_object(output_json)
        masked_video_value = str(metadata.get("masked_video") or "").strip()
        if not masked_video_value:
            return metadata
        masked_video = Path(masked_video_value)
        if not masked_video.exists() or not masked_video.is_file():
            return metadata
        group_a, group_b = _record_storage_parts(record)
        record_id = safe_path_part(str(record.get("id") or record.get("clip_id") or "caption"))
        dest = self.visualization_root / group_a / group_b / f"{record_id}__obj{object_index:03d}_sam2_mask.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(masked_video, dest)
        metadata["sam2_visualization_path"] = str(dest)
        metadata["catv_masked_video_path"] = str(dest)
        return metadata


_ONE_PASS_DROP_KEYS = {
    "dense_caption",
    "sam2_visualization_path",
    "catv_masked_video_path",
    "catv_start_sec",
    "catv_end_sec",
    "catv_duration_sec",
}


def _caption_describes_target(object_caption: str, target_noun: str) -> bool:
    """Check whether CAT-V's caption is actually about the row's target object.

    CAT-V's caption is generated from a SAM2-masked video and can drift onto
    the wrong object (e.g. target_noun="phone" but the caption describes "a
    piece of paper"). The prompt always asks for an "HO: <description>" first
    line, so compare the target noun against just that line when present.
    """
    target = str(target_noun or "").strip().lower().replace("_", " ")
    if not target or target in {"object", "sam2_object", "sam2 object"}:
        return True
    caption = str(object_caption or "")
    if not caption.strip() or caption.strip() == "<error_processing>":
        return True
    match = re.search(r"HO\s*:\s*(.+)", caption, flags=re.IGNORECASE)
    description = (match.group(1).splitlines()[0] if match else caption).lower()
    if not description.strip():
        return True
    if target in description:
        return True
    return any(part in description for part in target.split() if len(part) > 2)


def write_one_pass_labels(
    descriptions_path: Path,
    out_path: Path,
    *,
    frames_dir: Path = Path("outputs/one_pass_sparse_frames"),
    detect_persons: bool = False,
    decision_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    review_dir: Path | None = None,
    review_max_width: int = 900,
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
    dataset: str | None = None,
    progress_desc: str = "CAT-V one-pass labels",
) -> int:
    return _write_one_pass_labels_impl(
        descriptions_path,
        out_path,
        frames_dir=frames_dir,
        detect_persons=detect_persons,
        decision_fn=decision_fn,
        review_dir=review_dir,
        review_max_width=review_max_width,
        limit=limit,
        resume=resume,
        show_progress=show_progress,
        dataset=dataset,
        progress_desc=progress_desc,
    )


def write_egolife_one_pass_labels(
    descriptions_path: Path,
    out_path: Path,
    *,
    frames_dir: Path = Path("outputs/egolife_one_pass_sparse_frames"),
    detect_persons: bool = False,
    decision_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    review_dir: Path | None = None,
    review_max_width: int = 900,
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> int:
    """Backward-compatible wrapper for EgoLife one-pass labeling."""
    return write_one_pass_labels(
        descriptions_path,
        out_path,
        frames_dir=frames_dir,
        detect_persons=detect_persons,
        decision_fn=decision_fn,
        review_dir=review_dir,
        review_max_width=review_max_width,
        limit=limit,
        resume=resume,
        show_progress=show_progress,
        dataset="egolife",
        progress_desc="EgoLife one-pass labels",
    )


def _write_one_pass_labels_impl(
    descriptions_path: Path,
    out_path: Path,
    *,
    frames_dir: Path,
    detect_persons: bool = False,
    decision_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    review_dir: Path | None = None,
    review_max_width: int = 900,
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
    dataset: str | None = None,
    progress_desc: str = "CAT-V one-pass labels",
) -> int:
    """Attach sparse benchmark frames and labels to existing object descriptions.

    ``descriptions_path`` is the JSONL produced by the bbox/CAT-V stage. This
    function intentionally does not run CAT-V. It treats the existing
    ``object_caption`` as the grounded object-description source. When an
    ``extractor`` is provided, it uses the previous frame-selection criterion:
    sample the caption interval, keep frames where the target object is visible,
    then choose first/middle/last visible frames as t-2/t-1/t.

    When ``review_dir`` is given, a composite review image (t-2/t-1/t stacked
    row-by-row with a caption panel reporting the auto taxonomy/ground-truth
    and evidence) is rendered for each row right after labeling, and its path
    is recorded under ``review_image_path``.
    """
    if not descriptions_path.exists():
        raise FileNotFoundError(f"Object-description JSONL not found: {descriptions_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if review_dir is not None:
        review_dir.mkdir(parents=True, exist_ok=True)

    from egoownership.catv_evidence_label import build_evidence_label

    person_detector = None
    if detect_persons:
        from egoownership.detection.persons import detect_persons as person_detector

    existing_ids: set[str] = set()
    if resume:
        existing_ids = _load_existing_output_ids(out_path, repair=True)

    records = iter_jsonl(descriptions_path, skip_bad=True)
    if show_progress:
        from tqdm.auto import tqdm

        records = tqdm(
            records,
            total=count_jsonl(descriptions_path),
            unit="entry",
            desc=progress_desc,
        )

    count = 0
    processed = 0
    skipped_video = 0
    skipped_frame = 0
    skipped_existing = 0
    skipped_caption_mismatch = 0
    progress_bar = records if show_progress else None
    mode = "a" if resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            if limit is not None and processed >= limit:
                break
            processed += 1
            record_id = _record_output_id(record)
            if record_id in existing_ids:
                skipped_existing += 1
                _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, 0, skipped_existing)
                continue
            video_path = Path(str(record.get("video_path") or ""))
            if not video_path.exists() or not video_path.is_file():
                skipped_video += 1
                _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, 0, skipped_existing)
                continue

            frame_times = _jsonl_sparse_frame_times(record)
            frame_paths = extract_sparse_frames_for_record(
                record,
                video_path,
                frames_dir,
                local_frame_times=frame_times,
                force=False,
            )
            if not frame_paths:
                skipped_frame += 1
                _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, 0, skipped_existing)
                continue

            target_frame_sec = frame_times["t"]
            row = {
                **{k: v for k, v in record.items() if k not in _ONE_PASS_DROP_KEYS},
                "target_frame_tag": "t",
                "target_frame_sec": target_frame_sec,
                "frame_times_sec": frame_times,
                "frame_selection": "jsonl_timestamps",
                "frame_paths": {tag: str(path) for tag, path in frame_paths.items()},
                "frame_t_minus_2_path": str(frame_paths["t-2"]),
                "frame_t_minus_1_path": str(frame_paths["t-1"]),
                "frame_t_path": str(frame_paths["t"]),
                "temporal_target_objects": {},
                "first_frame_path": str(frame_paths["t"]),
                "first_frame_sec": target_frame_sec,
                "object": record.get("object") or {},
                "source": {
                    **(record.get("source") or {}),
                    "object_description_source": str(descriptions_path),
                    "label_builder": "one_pass_from_existing_object_description",
                    "frame_selector": "jsonl_timestamps",
                },
            }
            target_noun = str((row.get("object") or {}).get("target_noun") or (row.get("object") or {}).get("label") or "")
            if not _caption_describes_target(row.get("object_caption"), target_noun):
                skipped_caption_mismatch += 1
                if progress_bar is not None and hasattr(progress_bar, "set_postfix"):
                    progress_bar.set_postfix(
                        written=count,
                        skipped_video=skipped_video,
                        skipped_frame=skipped_frame,
                        skipped_existing=skipped_existing,
                        skipped_caption_mismatch=skipped_caption_mismatch,
                    )
                continue
            labeled_row = build_evidence_label(row, person_detector=person_detector, decision_fn=decision_fn)
            if review_dir is not None:
                group_a, group_b = _record_storage_parts(labeled_row)
                review_dest = review_dir / group_a / group_b / f"{safe_path_part(record_id)}.jpg"
                review_dest.parent.mkdir(parents=True, exist_ok=True)
                render_annotation_review_image(labeled_row, max_width=review_max_width).save(review_dest, quality=90)
                labeled_row["review_image_path"] = str(review_dest)
            handle.write(json.dumps(labeled_row, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            existing_ids.add(record_id)
            _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, 0, skipped_existing)
    return count


def write_catv_captions_batch(
    input_path: Path,
    out_path: Path,
    *,
    catv_root: Path = Path("/home/jhlee/CAT-V"),
    mask_model_path: Path | None = None,
    catv_device: str = "cuda:0",
    fps: float = 1.0,
    whole_video: bool = True,
    max_frames: int = 16,
    max_side: int = 448,
    captioner_backend: str = "qwen3vl",
    caption_model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    qwen_vl_python: str = "/home/jhlee/miniconda3/envs/sam2hf/bin/python",
    catv_python: str | None = None,
    visualization_root: Path | None = None,
    batch_jobs_dir: Path | None = None,
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> int:
    """Batch-mode CAT-V captioning: load SAM-2 once, load the VLM once.

    Instead of spawning run_catv_one_object.py per-row (which reloads both
    models every time), this function:

    1. Prepares all clip/bbox inputs with ffmpeg (fast, no GPU).
    2. Calls batch_sam2_mask.py once — SAM-2 loaded a single time for all rows.
    3. Calls batch_qwen_vl_caption.py once — VLM loaded a single time for all rows.
    4. Collects all outputs and writes the result JSONL.

    With 183 rows the per-row approach wastes ~1.5h (SAM-2) + ~3h (Qwen3-VL) in
    model I/O alone; batch mode reduces that to a one-time ~90s load.
    """
    import subprocess as _sp
    import tempfile

    if not input_path.exists():
        raise FileNotFoundError(f"BBox JSONL not found: {input_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs_dir = (batch_jobs_dir or out_path.parent / "catv_batch_jobs").resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    sam2_jobs_path = jobs_dir / "sam2_jobs.jsonl"
    vl_jobs_path = jobs_dir / "vl_jobs.jsonl"

    with tempfile.TemporaryDirectory(prefix="catv_batch_work_") as _tmpdir:
        work_root = Path(_tmpdir)
        return _write_catv_captions_batch_in_workdir(
            input_path=input_path,
            out_path=out_path,
            work_root=work_root,
            catv_root=catv_root,
            mask_model_path=mask_model_path,
            catv_device=catv_device,
            fps=fps,
            whole_video=whole_video,
            max_frames=max_frames,
            max_side=max_side,
            captioner_backend=captioner_backend,
            caption_model_path=caption_model_path,
            qwen_vl_python=qwen_vl_python,
            catv_python=catv_python,
            visualization_root=visualization_root,
            sam2_jobs_path=sam2_jobs_path,
            vl_jobs_path=vl_jobs_path,
            limit=limit,
            resume=resume,
            show_progress=show_progress,
        )

def _write_catv_captions_batch_in_workdir(
    *,
    input_path: Path,
    out_path: Path,
    work_root: Path,
    catv_root: Path,
    mask_model_path: Path | None,
    catv_device: str,
    fps: float,
    whole_video: bool,
    max_frames: int,
    max_side: int,
    captioner_backend: str,
    caption_model_path: str,
    qwen_vl_python: str,
    catv_python: str | None,
    visualization_root: Path | None,
    sam2_jobs_path: Path,
    vl_jobs_path: Path,
    limit: int | None,
    resume: bool,
    show_progress: bool,
) -> int:
    existing_ids: set[str] = set()
    if resume:
        existing_ids = _load_existing_output_ids(out_path, repair=True)

    records_all = list(iter_jsonl(input_path, skip_bad=True))
    if limit is not None:
        records_all = records_all[:limit]

    if show_progress:
        print(f"[batch_captioning] {len(records_all)} total rows, {len(existing_ids)} already done", flush=True)

    # --- Phase 1: Prepare per-row work dirs and write SAM-2 job JSONL ---
    print("[batch_captioning] Phase 1: preparing clips and bbox files …", flush=True)
    sam2_jobs: list[dict[str, Any]] = []
    skipped_bad = 0
    skipped_existing = 0

    for record in records_all:
        record_id = _record_output_id(record)
        if record_id in existing_ids:
            skipped_existing += 1
            continue

        video_path = Path(str(record.get("video_path") or ""))
        obj = record.get("object") or {}
        bbox = obj.get("bbox") or {}
        if not video_path.exists() or not bbox:
            skipped_bad += 1
            continue

        group_a, group_b = _record_storage_parts(record)
        safe_id = safe_path_part(record_id)
        work_dir = work_root / group_a / group_b / safe_id
        work_dir.mkdir(parents=True, exist_ok=True)

        first_frame_path_str = str(record.get("first_frame_path") or "")
        first_frame_path = Path(first_frame_path_str) if first_frame_path_str else None
        if first_frame_path is None or not first_frame_path.exists():
            timestamp_sec = float(
                record.get("reference_frame_sec") or record.get("first_frame_sec") or record.get("catv_start_sec") or 0.0
            )
            extracted = _ffmpeg_extract_frame(video_path, work_dir / "first_frame.jpg", timestamp_sec, force=False)
            first_frame_path = extracted

        try:
            video_w, video_h = _probe_video_wh(video_path)
        except Exception:
            skipped_bad += 1
            continue

        x1 = int(float(bbox.get("x_min", 0)) * video_w)
        y1 = int(float(bbox.get("y_min", 0)) * video_h)
        x2 = int(float(bbox.get("x_max", 0)) * video_w)
        y2 = int(float(bbox.get("y_max", 0)) * video_h)
        x1, x2 = (min(x1, x2), max(x1, x2))
        y1, y2 = (min(y1, y2), max(y1, y2))
        x1 = max(0, min(video_w - 1, x1))
        y1 = max(0, min(video_h - 1, y1))
        x2 = max(0, min(video_w - 1, x2))
        y2 = max(0, min(video_h - 1, y2))

        start_sec = float(record.get("catv_start_sec") or record.get("start_sec") or 0.0)
        duration_sec = float(record.get("catv_duration_sec") or 5.0)
        out_masked_video = work_dir / "sam2_masked.mp4"
        caption_out_json = work_dir / "caption.json"
        target_noun = str(obj.get("target_noun") or obj.get("label") or "").strip()

        sam2_jobs.append({
            "job_id": record_id,
            "video_path": str(video_path),
            "first_frame_path": str(first_frame_path) if first_frame_path else "",
            "bbox_str": f"{x1},{y1},{x2},{y2}",
            "fps": fps,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "whole_video": whole_video,
            "work_dir": str(work_dir),
            "out_masked_video": str(out_masked_video),
            "caption_out_json": str(caption_out_json),
            "target_noun": target_noun,
            "catv_start_sec": start_sec,
            "whole_video_mode": whole_video,
        })

    if show_progress:
        print(
            f"[batch_captioning] {len(sam2_jobs)} pending jobs "
            f"(skipped: {skipped_existing} existing, {skipped_bad} bad rows)",
            flush=True,
        )

    if not sam2_jobs:
        print("[batch_captioning] nothing to do", flush=True)
        return 0

    sam2_jobs_path.write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in sam2_jobs) + "\n",
        encoding="utf-8",
    )

    # --- Phase 2: Run SAM-2 batch tracker ---
    print(f"[batch_captioning] Phase 2: SAM-2 tracking ({len(sam2_jobs)} jobs) …", flush=True)
    catv_py = catv_python or sys.executable
    mask_model = str(mask_model_path) if mask_model_path else str(catv_root / "checkpoints" / "sam2.1_hiera_base_plus.pt")
    batch_sam2_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "batch_sam2_mask.py"
    _run_batch_subprocess(
        [
            catv_py,
            str(batch_sam2_script),
            "--jobs", str(sam2_jobs_path),
            "--catv-root", str(catv_root),
            "--model-path", mask_model,
            "--device", catv_device,
        ],
        label="batch_sam2_mask",
    )

    # --- Phase 3: Write Qwen3-VL job JSONL for successfully masked rows ---
    print("[batch_captioning] Phase 3: preparing VLM jobs …", flush=True)
    vl_jobs: list[dict[str, Any]] = []
    for job in sam2_jobs:
        out_masked_video = Path(job["out_masked_video"])
        if not out_masked_video.exists() or out_masked_video.stat().st_size == 0:
            continue
        noun_hint = job.get("target_noun", "")
        vl_jobs.append({
            "job_id": job["job_id"],
            "masked_video": job["out_masked_video"],
            "question": _ownership_relevant_question(noun_hint),
            "out_json": job["caption_out_json"],
            "max_frames": max_frames,
            "max_side": max_side,
        })
    print(f"[batch_captioning] {len(vl_jobs)} VLM jobs ({len(sam2_jobs) - len(vl_jobs)} SAM-2 failures)", flush=True)
    vl_jobs_path.write_text(
        "\n".join(json.dumps(j, ensure_ascii=False) for j in vl_jobs) + "\n",
        encoding="utf-8",
    )

    # --- Phase 4: Run Qwen3-VL batch captioner ---
    if captioner_backend == "qwen3vl":
        print(f"[batch_captioning] Phase 4: Qwen3-VL captioning ({len(vl_jobs)} jobs) …", flush=True)
        batch_vl_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "batch_qwen_vl_caption.py"
        _run_batch_subprocess(
            [
                qwen_vl_python,
                str(batch_vl_script),
                "--jobs", str(vl_jobs_path),
                "--model-path", caption_model_path,
                "--device", catv_device,
            ],
            label="batch_qwen_vl_caption",
        )
    else:
        raise NotImplementedError(f"Batch mode not yet implemented for backend: {captioner_backend!r}")

    # --- Phase 5: Collect results and write output JSONL ---
    print("[batch_captioning] Phase 5: collecting results …", flush=True)
    job_by_id = {j["job_id"]: j for j in sam2_jobs}
    record_by_id = {_record_output_id(r): r for r in records_all}

    count = 0
    mode = "a" if resume else "w"
    written_ids = set(existing_ids)
    with out_path.open(mode, encoding="utf-8") as handle:
        for record_id, job in job_by_id.items():
            if record_id in written_ids:
                continue
            record = record_by_id.get(record_id)
            if record is None:
                continue
            caption_json_path = Path(job["caption_out_json"])
            if not caption_json_path.exists():
                continue
            try:
                catv_output = json.loads(caption_json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            caption = _extract_caption_from_json(catv_output)
            if not caption:
                continue

            relative_timestamps = catv_output.get("frame_timestamps_sec") or []
            clip_origin_sec = 0.0 if whole_video else float(record.get("catv_start_sec") or 0.0)
            described_frame_timestamps_sec = [clip_origin_sec + t for t in relative_timestamps]

            masked_video = Path(job["out_masked_video"])
            sam2_vis_path: str | None = None
            if visualization_root is not None and masked_video.exists():
                group_a, group_b = _record_storage_parts(record)
                vis_dest = Path(visualization_root) / group_a / group_b / f"{safe_path_part(record_id)}_sam2_mask.mp4"
                vis_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(masked_video, vis_dest)
                sam2_vis_path = str(vis_dest)

            row = {
                **record,
                "object_caption": caption,
                "described_frame_timestamps_sec": described_frame_timestamps_sec,
                "sam2_visualization_path": sam2_vis_path or record.get("sam2_visualization_path"),
                "catv_masked_video_path": sam2_vis_path or record.get("catv_masked_video_path"),
                "source": {
                    **(record.get("source") or {}),
                    "captioner": caption_model_path,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written_ids.add(record_id)
            count += 1

    if show_progress:
        print(f"[batch_captioning] wrote {count} rows → {out_path}", flush=True)
    return count


def _ownership_relevant_question_for_batch() -> str:
    return _ownership_relevant_question("")


def _ownership_relevant_question(target_noun: str = "") -> str:
    target_text = target_noun.strip()
    if target_text:
        target_text = f" The expected target object name is: {target_text}."
    return (
        "Describe the highlighted object HO in this video for an egocentric implicit-ownership task. "
        "Each video frame is labeled Frame1, Frame2, etc. — use those exact labels to answer question (4)."
        f"{target_text} Answer these four questions about HO, numbered exactly as below: "
        "(1) What is HO? Answer as 'HO: <short name>'. "
        "(2) Does HO's status change during the video, and what is HO's final status? Describe any change as "
        "'from <state> to <state>' (e.g. moved, picked up, placed, opened, given, returned). "
        "(3) Who interacts with HO, and how? Say 'camera wearer/ego hand' for the person wearing the camera, or "
        "'other person' for anyone else, rather than the generic word 'person', and describe the interaction "
        "(e.g. holds, picks up, places, gives, hands, passes, receives, borrows, returns HO). "
        "If the actor identity cannot be distinguished, say 'actor identity is ambiguous'. "
        "If HO's status never changes and nobody interacts with it, say so explicitly in (2)/(3). "
        "(4) Which frame numbers best show: HO's initial state, the key interaction moment, and HO's final "
        "state? Answer on its own line as exactly 'FIRST=<n> KEY=<n> FINAL=<n>' using the Frame numbers shown "
        "(e.g. 'FIRST=1 KEY=5 FINAL=8'). If unsure, give your best estimate rather than omitting it. "
        "Do not describe HO's appearance, surroundings, or unrelated objects beyond what is needed to answer "
        "these four questions."
    )


def _probe_video_wh(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = result.stdout.strip().splitlines()[0]
    w, h = text.split("x")
    return int(w), int(h)


def _run_batch_subprocess(cmd: list[str], *, label: str) -> None:
    print(f"[{label}] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def write_catv_captions_from_bbox_jsonl(
    input_path: Path,
    out_path: Path,
    *,
    captioner: ObjectCaptioner,
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> int:
    """Caption existing bbox rows with CAT-V, appending safely on reruns."""
    if not input_path.exists():
        raise FileNotFoundError(f"EgoLife bbox JSONL not found: {input_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    if resume:
        existing_ids = _load_existing_output_ids(out_path, repair=True)

    records = iter_jsonl(input_path, skip_bad=True)
    if show_progress:
        from tqdm.auto import tqdm

        records = tqdm(
            records,
            total=count_jsonl(input_path),
            unit="bbox",
            desc="CAT-V captioning",
        )

    count = 0
    processed = 0
    skipped_existing = 0
    skipped_bad_row = 0
    skipped_error = 0
    progress_bar = records if show_progress else None
    mode = "a" if resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            if limit is not None and processed >= limit:
                break
            processed += 1
            record_id = _record_output_id(record)
            if record_id in existing_ids:
                skipped_existing += 1
                _set_catv_progress_postfix(progress_bar, count, skipped_existing, skipped_bad_row)
                continue

            video_path = Path(str(record.get("video_path") or ""))
            first_frame_path = Path(str(record.get("first_frame_path") or ""))
            obj = record.get("object") or {}
            bbox = obj.get("bbox") or {}
            if not video_path.exists() or not bbox:
                skipped_bad_row += 1
                _set_catv_progress_postfix(progress_bar, count, skipped_existing, skipped_bad_row)
                continue

            object_index = int(record.get("object_index") or 0)
            try:
                object_caption = captioner.caption(
                    video_path=video_path,
                    first_frame_path=first_frame_path,
                    bbox=bbox,
                    record=record,
                    object_index=object_index,
                )
            except Exception as exc:  # noqa: BLE001
                skipped_error += 1
                print(f"[catv_pipeline] skipping {record_id} after captioner error: {exc}", flush=True)
                _set_catv_progress_postfix(progress_bar, count, skipped_existing, skipped_bad_row)
                continue
            catv_metadata = getattr(captioner, "last_metadata", {}) or {}
            row = {
                **record,
                "sam2_visualization_path": catv_metadata.get("sam2_visualization_path")
                or record.get("sam2_visualization_path"),
                "catv_masked_video_path": catv_metadata.get("catv_masked_video_path")
                or record.get("catv_masked_video_path"),
                "object_caption": object_caption,
                "source": {
                    **(record.get("source") or {}),
                    "captioner": getattr(captioner, "model_id", "CAT-V"),
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            existing_ids.add(record_id)
            count += 1
            _set_catv_progress_postfix(progress_bar, count, skipped_existing, skipped_bad_row)
    return count


def write_caption_bboxes(
    annotations_path: Path,
    videos_root: Path,
    out_path: Path,
    *,
    dataset: str,
    extractor: Sam2ObjectExtractor | Callable[[Path], list[dict[str, Any]]],
    frames_dir: Path = Path("outputs/catv_first_frames_objectlist"),
    object_nouns_path: Path | None = None,
    max_objects_per_record: int = 5,
    reference_frame: str = "midpoint",
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
    ego4d_clip_window_sec: float | None = None,
    ego4d_scratch_root: Path | None = None,
    ego4d_auto_download: bool = True,
    ego4d_require_observer: bool = True,
) -> int:
    """Extract target-object boxes from table-caption records for any supported dataset."""
    adapter_kwargs: dict[str, Any] = {}
    if normalize_dataset_id(dataset) == "ego4d_fho":
        from egoownership.ego4d_video import DEFAULT_CLIP_WINDOW_SEC

        adapter_kwargs = {
            "ego4d_clip_window_sec": ego4d_clip_window_sec
            if ego4d_clip_window_sec is not None
            else DEFAULT_CLIP_WINDOW_SEC,
            "ego4d_scratch_root": ego4d_scratch_root,
            "ego4d_auto_download": ego4d_auto_download,
            "ego4d_require_observer": ego4d_require_observer,
        }
    adapter = get_catv_dataset_adapter(dataset, **adapter_kwargs)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations path not found: {annotations_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    existing_base_ids: set[str] = set()
    if resume:
        existing_ids = _load_existing_output_ids(out_path, repair=True)
        existing_base_ids = {record_id.split("#obj", 1)[0] for record_id in existing_ids}

    object_nouns = load_object_noun_allowlist(object_nouns_path)
    records = adapter.iter_caption_records(annotations_path, object_nouns=object_nouns)
    progress_label = f"{adapter.dataset_id} bbox extraction"
    if show_progress:
        from tqdm.auto import tqdm

        records = tqdm(records, total=None, unit="caption", desc=progress_label)

    count = 0
    processed = 0
    skipped_video = 0
    skipped_frame = 0
    skipped_sam = 0
    skipped_existing = 0
    progress_bar = records if show_progress else None
    mode = "a" if resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            if limit is not None and processed >= limit:
                break
            processed += 1
            if record.get("id") in existing_base_ids:
                skipped_existing += 1
                _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, skipped_sam, skipped_existing)
                continue
            resolved_video = adapter.resolve_video_segment(videos_root, record)
            if resolved_video is None:
                skipped_video += 1
                _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, skipped_sam, skipped_existing)
                continue

            video_path, local_start_sec, video_start_sec = resolved_video
            target_sec = _reference_frame_time(record, local_start_sec, reference_frame)
            target_frame = extract_frame_for_record(
                record=record,
                video_path=video_path,
                frames_dir=frames_dir,
                tag="reference",
                timestamp_sec=target_sec,
                force=False,
            )
            if target_frame is None:
                skipped_frame += 1
                _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, skipped_sam, skipped_existing)
                continue

            objects = _extract_objects_for_caption(extractor, target_frame, record)
            objects = filter_caption_candidate_objects(objects, record, max_objects=max_objects_per_record)
            if not objects:
                skipped_sam += 1
                _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, skipped_sam, skipped_existing)
                continue

            debug_frame = target_frame.with_name(f"{target_frame.stem}__debug_boxes{target_frame.suffix}")
            shutil.copy2(target_frame, debug_frame)
            visualize_object_bboxes_in_place(debug_frame, objects, record=record)
            for object_index, obj in enumerate(objects):
                row = _build_bbox_row(
                    record=record,
                    video_path=video_path,
                    target_frame=target_frame,
                    frame_paths={"t-2": target_frame, "t-1": target_frame, "t": target_frame},
                    obj=obj,
                    object_index=object_index,
                    local_start_sec=local_start_sec,
                    local_frame_times={"t-2": target_sec, "t-1": target_sec, "t": target_sec},
                    video_start_sec=video_start_sec,
                    segmenter=_extractor_model_id(extractor),
                    visible_sample_count=1,
                    sampled_frame_count=1,
                    dataset=adapter.dataset_id,
                )
                row["frame_selection"] = f"single_reference_{reference_frame}"
                _keep_only_reference_frame_fields(row, target_frame=target_frame, target_sec=target_sec)
                record_id = _record_output_id(row)
                if record_id in existing_ids:
                    skipped_existing += 1
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing_ids.add(record_id)
                count += 1
            _set_progress_postfix(progress_bar, count, skipped_video, skipped_frame, skipped_sam, skipped_existing)
    return count


def write_egolife_caption_bboxes(
    annotations_path: Path,
    videos_root: Path,
    out_path: Path,
    *,
    extractor: Sam2ObjectExtractor | Callable[[Path], list[dict[str, Any]]],
    frames_dir: Path = Path("outputs/egolife_catv_first_frames_objectlist"),
    object_nouns_path: Path | None = None,
    max_objects_per_record: int = 5,
    reference_frame: str = "midpoint",
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> int:
    """Backward-compatible wrapper for EgoLife bbox extraction."""
    return write_caption_bboxes(
        annotations_path,
        videos_root,
        out_path,
        dataset="egolife",
        extractor=extractor,
        frames_dir=frames_dir,
        object_nouns_path=object_nouns_path,
        max_objects_per_record=max_objects_per_record,
        reference_frame=reference_frame,
        limit=limit,
        resume=resume,
        show_progress=show_progress,
    )



def _keep_only_reference_frame_fields(row: dict[str, Any], *, target_frame: Path, target_sec: float) -> None:
    for key in (
        "frame_paths",
        "frame_times_sec",
        "frame_t_minus_2_path",
        "frame_t_minus_1_path",
        "frame_t_path",
        "target_frame_tag",
        "target_frame_sec",
        "visible_sample_count",
        "sampled_frame_count",
    ):
        row.pop(key, None)
    row["reference_frame_path"] = str(target_frame)
    row["reference_frame_sec"] = target_sec
    row["first_frame_path"] = str(target_frame)
    row["first_frame_sec"] = target_sec
    obj = row.get("object") or {}
    obj.pop("source_frame_tag", None)
    obj["source_frame_tag"] = "reference"
    obj["source_frame_sec"] = target_sec
    row["object"] = obj


def _description_local_start_sec(record: dict[str, Any]) -> float:
    return float(record.get("catv_start_sec") or record.get("first_frame_sec") or 0.0)


def _reference_frame_time(record: dict[str, Any], local_start_sec: float, reference_frame: str) -> float:
    duration = max(0.0, float(record.get("end_sec") or 0.0) - float(record.get("start_sec") or 0.0))
    mode = reference_frame.lower().strip()
    if mode in {"start", "first"}:
        return round(max(0.0, local_start_sec), 3)
    if mode in {"end", "last"}:
        return round(max(0.0, local_start_sec + duration), 3)
    if mode not in {"mid", "middle", "midpoint"}:
        raise ValueError(f"reference_frame must be start, midpoint, or end; got {reference_frame!r}")
    return round(max(0.0, local_start_sec + duration / 2.0), 3)


def _description_sparse_frame_times(record: dict[str, Any]) -> dict[str, float]:
    start = _description_local_start_sec(record)
    end = record.get("catv_end_sec")
    if end is None:
        duration = float(record.get("catv_duration_sec") or 0.0)
        if duration <= 0:
            duration = max(0.0, float(record.get("end_sec") or 0.0) - float(record.get("start_sec") or 0.0))
        end = start + duration
    end_sec = max(start, float(end))
    midpoint = start + (end_sec - start) / 2.0
    return {
        "t-2": round(max(0.0, start), 3),
        "t-1": round(max(0.0, midpoint), 3),
        "t": round(max(0.0, end_sec), 3),
    }


def _jsonl_sparse_frame_times(record: dict[str, Any]) -> dict[str, float]:
    """Pick t-2/t-1/t from described_frame_timestamps_sec written by Stage 2.

    Falls back to _description_sparse_frame_times when the field is absent or
    empty (e.g. records produced by the old per-row caption-bboxes path).
    """
    timestamps = record.get("described_frame_timestamps_sec")
    if timestamps and len(timestamps) >= 1:
        ts = [float(t) for t in timestamps]
        return {
            "t-2": round(max(0.0, ts[0]), 3),
            "t-1": round(max(0.0, ts[len(ts) // 2]), 3),
            "t": round(max(0.0, ts[-1]), 3),
        }
    return _description_sparse_frame_times(record)


def _merge_description_object_with_visible_object(
    description_obj: dict[str, Any],
    visible_obj: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    merged = {
        **description_obj,
        **visible_obj,
        "bbox": visible_obj.get("bbox") or description_obj.get("bbox"),
        "score": visible_obj.get("score", description_obj.get("score")),
        "source_frame_tag": "t",
    }
    label = str(merged.get("label") or "").strip()
    if not label or label in {"sam2_object", "sam2 object", "object"}:
        merged["label"] = _object_display_label(description_obj or visible_obj, record)
    if description_obj.get("target_noun") and not merged.get("target_noun"):
        merged["target_noun"] = description_obj["target_noun"]
    return merged


def _build_bbox_row(
    *,
    record: dict[str, Any],
    video_path: Path,
    target_frame: Path,
    frame_paths: dict[str, Path],
    obj: dict[str, Any],
    object_index: int,
    local_start_sec: float,
    local_frame_times: dict[str, float],
    video_start_sec: float,
    segmenter: str,
    visible_sample_count: int,
    sampled_frame_count: int,
    dataset: str,
) -> dict[str, Any]:
    target_sec = local_frame_times["t"]
    window_duration = record.get("source_window_duration_sec")
    if window_duration is not None:
        caption_duration = max(0.1, float(window_duration))
        catv_start = 0.0
    else:
        caption_duration = max(1.0, float(record.get("end_sec") or 0.0) - float(record.get("start_sec") or 0.0))
        catv_start = target_sec
    return {
        "id": f"{record['id']}#obj{object_index}",
        "clip_id": record["clip_id"],
        "video_id": record["video_id"],
        "participant": record.get("participant"),
        "day": record.get("day"),
        "start_sec": record["start_sec"],
        "end_sec": record["end_sec"],
        "first_frame_sec": target_sec,
        "target_frame_tag": "t",
        "target_frame_sec": target_sec,
        "frame_times_sec": local_frame_times,
        "frame_selection": "target_visibility_first_mid_last",
        "visible_sample_count": visible_sample_count,
        "sampled_frame_count": sampled_frame_count,
        "frame_paths": {tag: str(path) for tag, path in frame_paths.items()},
        "frame_t_minus_2_path": str(frame_paths["t-2"]),
        "frame_t_minus_1_path": str(frame_paths["t-1"]),
        "frame_t_path": str(frame_paths["t"]),
        "source_video_start_sec": video_start_sec,
        "catv_start_sec": catv_start,
        "catv_end_sec": catv_start + caption_duration,
        "catv_duration_sec": caption_duration,
        "dense_caption": record.get("dense_caption"),
        "dense_caption_en": record.get("dense_caption_en"),
        "transcript": record.get("transcript"),
        "verb": record.get("verb"),
        "nouns": record.get("nouns") or [],
        "video_path": str(video_path),
        "first_frame_path": str(target_frame),
        "object_index": object_index,
        "object": {
            **obj,
            "source_frame_tag": "t",
            "source_frame_sec": target_sec,
            "instance_id": obj.get("instance_id") or f"frame_t_sam_obj_{object_index:03d}",
        },
        "source": {
            **(record.get("source") or {}),
            "segmenter": segmenter,
            "dataset": dataset,
        },
        "source_dataset": record.get("source_dataset") or dataset,
        "dataset": record.get("dataset") or dataset,
    }


def iter_filtered_table_object_caption_records(
    path: Path,
    *,
    object_nouns: set[str] | None = None,
    dataset: str = "egolife",
) -> Iterable[dict[str, Any]]:
    adapter = get_catv_dataset_adapter(dataset)
    yield from adapter.iter_caption_records(path, object_nouns=object_nouns)


def load_object_noun_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Object noun list not found: {path}")
    nouns: set[str] = set()
    for row in iter_jsonl(path):
        noun = normalize_object_noun(str(row.get("noun") or ""))
        if not noun:
            continue
        if row.get("keep_object_noun") is False:
            continue
        nouns.add(noun)
    return nouns


def _record_storage_parts(record: dict[str, Any]) -> tuple[str, str]:
    dataset = str(record.get("source_dataset") or record.get("dataset") or "egolife")
    try:
        adapter = get_catv_dataset_adapter(dataset)
        return adapter.storage_parts(record)
    except ValueError:
        if record.get("participant") and record.get("day"):
            return (
                safe_path_part(str(record.get("participant"))),
                safe_path_part(str(record.get("day"))),
            )
        video_id = safe_path_part(str(record.get("video_id") or "unknown_video"))
        return video_id, "clips"


def _upgrade_catv_wrapper_template(template: str) -> str:
    """Keep older user-provided CAT-V templates aligned with current defaults."""
    if "run_catv_one_object.py" not in template:
        return template
    additions: list[str] = []
    if "--first-frame" not in template:
        additions.extend(["--first-frame", "{first_frame_path}"])
    if "--start-sec" not in template:
        additions.extend(["--start-sec", "{start_sec}"])
    if "--duration-sec" not in template and "--whole-video" not in template:
        additions.extend(["--duration-sec", "{duration_sec}"])
    if "--fps" not in template:
        additions.extend(["--fps", "1"])
    if "--caption" not in template:
        additions.extend(["--caption", "{object_nouns}"])
    if not additions:
        return template
    return template.rstrip() + " " + " ".join(additions)


def extract_first_frame_for_record(
    record: dict[str, Any],
    video_path: Path,
    frames_dir: Path,
    *,
    force: bool = False,
) -> Path | None:
    frame_paths = extract_sparse_frames_for_record(
        record,
        video_path,
        frames_dir,
        local_frame_times={"t-2": float(record.get("start_sec") or 0.0), "t-1": float(record.get("start_sec") or 0.0), "t": float(record.get("start_sec") or 0.0)},
        force=force,
    )
    return frame_paths.get("t")


_FRAME_CITATION_RE = re.compile(
    r"FIRST\s*=\s*(\d+)\D+KEY\s*=\s*(\d+)\D+FINAL\s*=\s*(\d+)", re.IGNORECASE
)


def _parse_frame_citation(object_caption: str) -> dict[str, int] | None:
    """Parse the model's 'FIRST=<n> KEY=<n> FINAL=<n>' frame citation (see
    question (4) in ``_ownership_relevant_question``). Returns 1-indexed frame
    numbers, or None if the caption doesn't contain a well-formed citation.
    """
    match = _FRAME_CITATION_RE.search(str(object_caption or ""))
    if not match:
        return None
    first, key, final = (int(g) for g in match.groups())
    if min(first, key, final) < 1:
        return None
    return {"FIRST": first, "KEY": key, "FINAL": final}


def select_frames_from_citation(
    record: dict[str, Any],
    video_path: Path,
    frames_dir: Path,
    *,
    extractor: Sam2ObjectExtractor | Callable[[Path], list[dict[str, Any]]] | None = None,
    max_objects: int = 5,
    force: bool = False,
) -> SparseFrameSelection | None:
    """Pick t-2/t-1/t directly from the captioner's own frame citation, instead
    of re-running a separate SAM-3 visibility scan. Falls back to None (caller
    should fall back to ``select_visibility_aware_sparse_frames``) if the
    caption has no citation or the row has no recorded frame timestamps.

    When ``extractor`` is given, the object's bbox is re-detected at each of
    the three cited frames (scoped to this row's own target noun) instead of
    reusing the single bbox from the original bbox-extraction reference frame
    for all three. The citation exists precisely because the object's
    position/state changes between FIRST/KEY/FINAL, so a static bbox would
    make every frame's zone identical and silently zero out the temporal
    ownership signals (object_moved, held_by_changed, zone_changed, ...) that
    ``catv_evidence_label._build_temporal_evidence`` derives from them.
    """
    citation = _parse_frame_citation(record.get("object_caption"))
    timestamps = record.get("described_frame_timestamps_sec")
    if citation is None or not timestamps:
        return None
    n_frames = len(timestamps)
    if any(citation[key] > n_frames for key in ("FIRST", "KEY", "FINAL")):
        return None

    fallback_obj = record.get("object") or {}
    target_noun = str(fallback_obj.get("target_noun") or fallback_obj.get("label") or "").strip()
    noun_record = {**record, "nouns": [target_noun]} if target_noun else record

    tag_to_field = {"t-2": "FIRST", "t-1": "KEY", "t": "FINAL"}
    frame_paths: dict[str, Path] = {}
    frame_times: dict[str, float] = {}
    frame_target_objects: dict[str, dict[str, Any]] = {}
    for tag, field_name in tag_to_field.items():
        timestamp = float(timestamps[citation[field_name] - 1])
        dest = _frame_dest_path(record, frames_dir, tag)
        if force or not dest.exists():
            extracted = _ffmpeg_extract_frame(video_path, dest, timestamp, force=True)
            if extracted is None:
                return None
        frame_paths[tag] = dest
        frame_times[tag] = timestamp

        detected_obj = fallback_obj
        if extractor is not None:
            try:
                objects = _extract_objects_for_caption(extractor, dest, noun_record)
                objects = filter_caption_candidate_objects(objects, noun_record, max_objects=max_objects)
            except Exception:  # noqa: BLE001
                objects = []
            if objects:
                detected_obj = _primary_visible_object(objects)
        frame_target_objects[tag] = detected_obj

    return SparseFrameSelection(
        frame_paths=frame_paths,
        frame_times=frame_times,
        target_frame=frame_paths["t"],
        target_objects=[frame_target_objects["t"]],
        visible_sample_count=n_frames,
        sampled_frame_count=n_frames,
        frame_target_objects=frame_target_objects,
    )


def select_visibility_aware_sparse_frames(
    *,
    record: dict[str, Any],
    video_path: Path,
    frames_dir: Path,
    extractor: Sam2ObjectExtractor | Callable[[Path], list[dict[str, Any]]],
    local_start_sec: float,
    sample_fps: float,
    max_objects: int,
    force: bool = False,
) -> SparseFrameSelection | None:
    sample_times = _local_visibility_sample_times(record, local_start_sec, sample_fps)
    target_noun = str((record.get("object") or {}).get("target_noun") or (record.get("object") or {}).get("label") or "").strip()
    # Scope detection/filtering to this row's own object noun. ``record["nouns"]``
    # is the whole caption's noun list (e.g. multiple objects share one caption);
    # reusing it here would let a different object's box win re-selection and
    # silently overwrite this row's bbox/label (see one-pass-labels cross-noun bug).
    noun_record = {**record, "nouns": [target_noun]} if target_noun else record
    visible: list[tuple[int, float, Path, list[dict[str, Any]]]] = []
    with tempfile.TemporaryDirectory(prefix="egolife_visibility_probe_") as scratch_dir:
        scratch = Path(scratch_dir)
        for sample_index, timestamp in enumerate(sample_times):
            sample_frame = _ffmpeg_extract_frame(
                video_path,
                scratch / f"sample_{sample_index:03d}.jpg",
                timestamp,
                force=True,
            )
            if sample_frame is None:
                continue
            objects = _extract_objects_for_caption(extractor, sample_frame, noun_record)
            objects = filter_caption_candidate_objects(objects, noun_record, max_objects=max_objects)
            if objects:
                visible.append((sample_index, timestamp, sample_frame, objects))

        if not visible:
            return None

        first = visible[0]
        middle = visible[len(visible) // 2]
        last = visible[-1]
        frame_paths: dict[str, Path] = {}
        frame_target_objects = {
            "t-2": _primary_visible_object(first[3]),
            "t-1": _primary_visible_object(middle[3]),
            "t": _primary_visible_object(last[3]),
        }
        for tag, chosen in (("t-2", first), ("t-1", middle), ("t", last)):
            dest = _frame_dest_path(record, frames_dir, tag)
            if force or not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(chosen[2], dest)
            frame_paths[tag] = dest

    return SparseFrameSelection(
        frame_paths=frame_paths,
        frame_times={
            "t-2": first[1],
            "t-1": middle[1],
            "t": last[1],
        },
        target_frame=frame_paths["t"],
        target_objects=last[3],
        visible_sample_count=len(visible),
        sampled_frame_count=len(sample_times),
        frame_target_objects=frame_target_objects,
    )


def extract_sparse_frames_for_record(
    record: dict[str, Any],
    video_path: Path,
    frames_dir: Path,
    *,
    local_frame_times: dict[str, float],
    force: bool = False,
) -> dict[str, Path]:
    frame_paths: dict[str, Path] = {}
    for tag in ("t-2", "t-1", "t"):
        timestamp = float(local_frame_times[tag])
        path = extract_frame_for_record(
            record,
            video_path,
            frames_dir,
            tag=tag,
            timestamp_sec=timestamp,
            force=force,
        )
        if path is None:
            return {}
        frame_paths[tag] = path
    return frame_paths


def _frame_dest_path(record: dict[str, Any], frames_dir: Path, tag: str) -> Path:
    group_a, group_b = _record_storage_parts(record)
    record_id = safe_path_part(str(record.get("id") or record.get("clip_id") or "caption"))
    suffix = {"t-2": "t_minus_2", "t-1": "t_minus_1", "t": "t"}.get(tag, safe_path_part(tag))
    return frames_dir / group_a / group_b / f"{record_id}__{suffix}.jpg"


def extract_frame_for_record(
    record: dict[str, Any],
    video_path: Path,
    frames_dir: Path,
    *,
    tag: str,
    timestamp_sec: float,
    force: bool = False,
) -> Path | None:
    dest = _frame_dest_path(record, frames_dir, tag)
    return _ffmpeg_extract_frame(video_path, dest, timestamp_sec, force=force)


def _ffmpeg_extract_frame(
    video_path: Path,
    dest: Path,
    timestamp_sec: float,
    *,
    force: bool = False,
) -> Path | None:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
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


def _primary_visible_object(objects: list[dict[str, Any]]) -> dict[str, Any]:
    if not objects:
        return {}
    return max(objects, key=lambda obj: float(obj.get("score") or 0.0))


def _local_sparse_frame_times(record: dict[str, Any], local_start_sec: float) -> dict[str, float]:
    duration = max(0.0, float(record.get("end_sec") or 0.0) - float(record.get("start_sec") or 0.0))
    return {
        "t-2": local_start_sec,
        "t-1": local_start_sec + duration / 2.0,
        "t": local_start_sec + duration,
    }


def _local_visibility_sample_times(record: dict[str, Any], local_start_sec: float, sample_fps: float) -> list[float]:
    duration = max(0.0, float(record.get("end_sec") or 0.0) - float(record.get("start_sec") or 0.0))
    end = local_start_sec + duration
    if duration <= 0:
        return [local_start_sec]
    step = 1.0 / max(0.1, sample_fps)
    times = [local_start_sec]
    cursor = local_start_sec + step
    while cursor < end:
        times.append(cursor)
        cursor += step
    times.append(end)
    # Include midpoint even for short clips where 1 fps would only produce endpoints.
    midpoint = local_start_sec + duration / 2.0
    times.append(midpoint)
    deduped = sorted({round(max(0.0, t), 3) for t in times})
    return deduped


def visualize_object_bboxes_in_place(
    image_path: Path,
    objects: list[dict[str, Any]],
    *,
    record: dict[str, Any],
) -> bool:
    try:
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            draw = ImageDraw.Draw(image)
            line_width = max(3, round(min(width, height) * 0.006))
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            for object_index, obj in enumerate(objects):
                bbox = obj.get("bbox") or {}
                x1, y1, x2, y2 = _absolute_bbox_xyxy(bbox, width, height)
                color = _bbox_color(object_index)
                for offset in range(line_width):
                    draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
                text = f"{object_index}: {_object_display_label(obj, record)}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                text_x = max(0, min(x1, width - text_w - 8))
                text_y = max(0, y1 - text_h - 8)
                draw.rectangle(
                    (text_x, text_y, text_x + text_w + 8, text_y + text_h + 6),
                    fill=color,
                )
                draw.text((text_x + 4, text_y + 3), text, fill=(255, 255, 255), font=font)
            image.save(image_path, quality=92)
        return True
    except Exception:
        return False


def filter_caption_candidate_objects(
    objects: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    max_objects: int,
) -> list[dict[str, Any]]:
    if not objects:
        return []
    nouns = {str(noun).lower().replace("_", " ") for noun in (record.get("nouns") or [])}
    filtered: list[dict[str, Any]] = []
    for obj in objects:
        bbox = obj.get("bbox") or {}
        area = _bbox_area(bbox)
        if area <= 0.001 or area >= 0.75:
            continue
        label = str(obj.get("label") or "").lower().replace("_", " ")
        if nouns and label and label not in {"sam2_object", "sam2 object", "object"}:
            if label not in nouns and not any(noun in label or label in noun for noun in nouns):
                continue
        filtered.append(obj)
    filtered = _keep_highest_scoring_object_per_name(filtered)
    filtered.sort(key=lambda item: (item.get("score") or 0.0, _bbox_area(item.get("bbox") or {})), reverse=True)
    return filtered[:max_objects]


def _keep_highest_scoring_object_per_name(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for obj in objects:
        key = _object_dedupe_name(obj)
        current = best.get(key)
        if current is None or _object_rank(obj) > _object_rank(current):
            best[key] = obj
    return list(best.values())


def _object_dedupe_name(obj: dict[str, Any]) -> str:
    for field in ("target_noun", "label"):
        value = str(obj.get(field) or "").lower().replace("_", " ").strip()
        if value and value not in {"sam2 object", "sam2_object", "object"}:
            return value
    bbox = obj.get("bbox") or {}
    return json.dumps(bbox, sort_keys=True)


def _object_rank(obj: dict[str, Any]) -> tuple[float, float]:
    return float(obj.get("score") or 0.0), _bbox_area(obj.get("bbox") or {})


def _extract_objects_for_caption(
    extractor: Sam2ObjectExtractor | Callable[[Path], list[dict[str, Any]]],
    first_frame: Path,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(extractor, Sam2ObjectExtractor):
        if getattr(extractor.cfg, "backend", "") == "sam3":
            return _extract_objects_for_each_noun(
                extractor.extract_with_prompt, first_frame, record, nms_iou_threshold=extractor.cfg.nms_iou_threshold
            )
        return extractor.extract(first_frame)
    extract_with_prompt = getattr(extractor, "extract_with_prompt", None)
    if callable(extract_with_prompt):
        return _extract_objects_for_each_noun(extract_with_prompt, first_frame, record)
    return extractor(first_frame)


def _extract_objects_for_each_noun(
    extract_with_prompt: Callable[[Path, str], list[dict[str, Any]]],
    first_frame: Path,
    record: dict[str, Any],
    *,
    nms_iou_threshold: float = 0.90,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    nouns = _caption_object_nouns(record)
    prompts = nouns or [_caption_object_prompt(record)]
    for noun in prompts:
        for obj in extract_with_prompt(first_frame, noun):
            enriched = {**obj}
            enriched.setdefault("label", noun)
            enriched["target_noun"] = noun
            objects.append(enriched)
    # Different nouns (e.g. "box"/"cardboard", "wire"/"cable") can ground to the
    # same physical region; collapse those by geometry, not by label, keeping
    # the highest-scoring noun for each region.
    objects.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return _nms_objects(objects, iou_threshold=nms_iou_threshold)


def _caption_object_prompt(record: dict[str, Any]) -> str:
    nouns = _caption_object_nouns(record)
    return ", ".join(nouns[:3])


def _caption_object_nouns(record: dict[str, Any]) -> list[str]:
    return [str(noun).replace("_", " ").strip() for noun in (record.get("nouns") or []) if str(noun).strip()]


def _object_display_label(obj: dict[str, Any], record: dict[str, Any]) -> str:
    label = str(obj.get("label") or "").lower().replace("_", " ").strip()
    if label and label not in {"sam2 object", "object"}:
        return label
    nouns = [str(noun).replace("_", " ") for noun in (record.get("nouns") or [])]
    return ", ".join(nouns[:3]) if nouns else "object"


def _bbox_color(index: int) -> tuple[int, int, int]:
    colors = (
        (255, 32, 32),
        (32, 144, 255),
        (32, 180, 96),
        (255, 160, 32),
        (180, 64, 255),
    )
    return colors[index % len(colors)]


def _write_absolute_bbox(path: Path, bbox: dict[str, float], image_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    x1, y1, x2, y2 = _absolute_bbox_xyxy(bbox, width, height)
    path.write_text(",".join(map(str, [x1, y1, x2, y2])), encoding="utf-8")


def _absolute_bbox_xyxy(bbox: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(float(bbox.get("x_min", 0.0)) * width)
    y1 = int(float(bbox.get("y_min", 0.0)) * height)
    x2 = int(float(bbox.get("x_max", 0.0)) * width)
    y2 = int(float(bbox.get("y_max", 0.0)) * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _read_catv_caption(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    return _extract_caption_from_json(data)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_caption_from_json(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("object_caption", "caption", "model_answer", "answer", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in data.values():
            found = _extract_caption_from_json(value)
            if found:
                return found
    if isinstance(data, list):
        parts = [_extract_caption_from_json(item) for item in data]
        parts = [part for part in parts if part]
        return "\n".join(parts)
    return ""


def _subprocess_output_tail(exc: subprocess.CalledProcessError, *, max_chars: int = 4000) -> str:
    chunks: list[str] = []
    stdout = getattr(exc, "stdout", None)
    stderr = getattr(exc, "stderr", None)
    if stdout:
        chunks.append("stdout:\n" + str(stdout)[-max_chars:])
    if stderr:
        chunks.append("stderr:\n" + str(stderr)[-max_chars:])
    if not chunks:
        return ""
    return "\n\n" + "\n\n".join(chunks)


def _bbox_area(bbox: dict[str, Any]) -> float:
    try:
        return max(0.0, float(bbox.get("x_max", 0.0)) - float(bbox.get("x_min", 0.0))) * max(
            0.0,
            float(bbox.get("y_max", 0.0)) - float(bbox.get("y_min", 0.0)),
        )
    except Exception:
        return 0.0


def _extractor_model_id(extractor: Any) -> str:
    cfg = getattr(extractor, "cfg", None)
    return str(getattr(cfg, "model_id", "custom"))


def _set_progress_postfix(
    progress_bar: Any,
    written: int,
    skipped_video: int,
    skipped_frame: int,
    skipped_sam: int,
    skipped_existing: int = 0,
) -> None:
    if progress_bar is None or not hasattr(progress_bar, "set_postfix"):
        return
    progress_bar.set_postfix(
        written=written,
        skipped_video=skipped_video,
        skipped_frame=skipped_frame,
        skipped_sam=skipped_sam,
        skipped_existing=skipped_existing,
        refresh=False,
    )


def _set_catv_progress_postfix(
    progress_bar: Any,
    written: int,
    skipped_existing: int,
    skipped_bad_row: int,
) -> None:
    if progress_bar is None or not hasattr(progress_bar, "set_postfix"):
        return
    progress_bar.set_postfix(
        written=written,
        skipped_existing=skipped_existing,
        skipped_bad_row=skipped_bad_row,
        refresh=False,
    )


# Backward-compatible aliases used by tests and older scripts.
_iter_jsonl = iter_jsonl
_count_jsonl = count_jsonl
_safe_path_part = safe_path_part
_normalize_object_noun = normalize_object_noun


def _load_existing_output_ids(path: Path, *, repair: bool) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    valid_lines: list[str] = []
    changed = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                changed = True
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                changed = True
                continue
            ids.add(_record_output_id(row))
            valid_lines.append(json.dumps(row, ensure_ascii=False))
    if changed and repair:
        path.write_text("".join(f"{line}\n" for line in valid_lines), encoding="utf-8")
    return ids


def _record_output_id(record: dict[str, Any]) -> str:
    value = record.get("id")
    if value is not None and str(value).strip():
        return str(value)
    clip_id = record.get("clip_id") or record.get("video_id") or "unknown"
    start = float(record.get("start_sec") or 0.0)
    end = float(record.get("end_sec") or start)
    object_index = record.get("object_index")
    if object_index is None:
        object_index = (record.get("object") or {}).get("instance_id") or 0
    return f"{clip_id}:{start:.3f}-{end:.3f}#obj{object_index}"


_REVIEW_FRAME_TAGS = ("t-2", "t-1", "t")
_REVIEW_FRAME_KEYS = {
    "t-2": "frame_t_minus_2_path",
    "t-1": "frame_t_minus_1_path",
    "t": "frame_t_path",
}
_REVIEW_GT_COLORS = {
    "MINE": (40, 160, 80),
    "PERSON_K": (40, 120, 220),
    "SHARED": (230, 150, 30),
    "AMBIGUOUS": (160, 160, 160),
}


def _review_frame_path(row: dict[str, Any], tag: str) -> Path | None:
    value = row.get(_REVIEW_FRAME_KEYS[tag]) or (row.get("frame_paths") or {}).get(tag)
    return Path(str(value)) if value else None


def _load_review_frame(row: dict[str, Any], tag: str, *, max_width: int) -> Image.Image | None:
    path = _review_frame_path(row, tag)
    if path is None or not path.exists():
        return None
    with Image.open(path) as src:
        image = src.convert("RGB").copy()
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, max(1, round(image.height * ratio))))
    draw = ImageDraw.Draw(image)
    if tag == "t":
        bbox = (row.get("object") or {}).get("bbox") or {}
        if bbox:
            x1, y1, x2, y2 = _absolute_bbox_xyxy(bbox, image.width, image.height)
            color = _bbox_color(0)
            line_width = max(2, round(min(image.size) * 0.006))
            for offset in range(line_width):
                draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
            draw.text((x1 + 4, max(0, y1 - 16)), _object_display_label(row.get("object") or {}, row), fill=color)
    badge_top = image.height - 20
    draw.rectangle((0, badge_top, 40, image.height), fill=(0, 0, 0))
    draw.text((4, badge_top + 3), tag, fill=(255, 255, 255))
    return image


_REVIEW_FONT_PATHS = {
    "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}


def _review_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = _REVIEW_FONT_PATHS["bold" if bold else "regular"]
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except Exception:
            return ImageFont.load_default()


def _wrap_to_pixel_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _arrange_review_frames(row: dict[str, Any], *, max_width: int) -> Image.Image:
    """Lay the t-2/t-1/t frames out side-by-side in a single row."""
    gap = 6
    per_frame_width = max(160, (max_width - gap * (len(_REVIEW_FRAME_TAGS) - 1)) // len(_REVIEW_FRAME_TAGS))
    frames = [
        frame
        for frame in (_load_review_frame(row, tag, max_width=per_frame_width) for tag in _REVIEW_FRAME_TAGS)
        if frame is not None
    ]
    if not frames:
        frames = [Image.new("RGB", (max_width, round(max_width * 0.5625)), (32, 32, 32))]
    height = max(frame.height for frame in frames)
    total_width = sum(frame.width for frame in frames) + gap * (len(frames) - 1)
    canvas = Image.new("RGB", (total_width, height), (0, 0, 0))
    x = 0
    for frame in frames:
        canvas.paste(frame, (x, (height - frame.height) // 2))
        x += frame.width + gap
    return canvas


def render_annotation_review_image(row: dict[str, Any], *, max_width: int = 900) -> Image.Image:
    """Lay the t-2/t-1/t frames out in a single row with a caption panel below.

    The panel reports the auto taxonomy/ground-truth decision, the supporting
    evidence used by ``build_evidence_label``, and the CAT-V object caption in
    clearly separated, legible sections so a human reviewer can sanity-check
    the auto-label without leaving the image viewer.
    """
    frames_canvas = _arrange_review_frames(row, max_width=max_width)
    width = frames_canvas.width
    pad = 14
    text_width = width - 2 * pad

    font_small = _review_font(13)
    font_body = _review_font(15)
    font_header = _review_font(15, bold=True)
    font_title = _review_font(20, bold=True)

    evidence = row.get("evidence") or {}
    cues = evidence.get("caption_cues") or {}
    active_cues = sorted(name for name, value in cues.items() if value)
    confidence = row.get("auto_confidence")
    confidence_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "?"
    gt = str(row.get("auto_ground_truth") or "?")
    accent = _REVIEW_GT_COLORS.get(gt, (90, 90, 90))

    measurer = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    blocks: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int], int]] = []

    def add_line(text: str, font: ImageFont.ImageFont, color: tuple[int, int, int], gap_after: int = 4) -> None:
        for line in _wrap_to_pixel_width(measurer, text, font, text_width):
            blocks.append((line, font, color, gap_after))

    add_line(f"id: {row.get('id', '?')}", font_small, (110, 110, 110), gap_after=8)
    title = f"{row.get('auto_taxonomy', '?')} · {gt}  (confidence {confidence_text})"
    if row.get("needs_review"):
        title += "   ⚠ NEEDS REVIEW"
    add_line(title, font_title, accent, gap_after=8)
    add_line(
        f"object: {evidence.get('target_object', '?')} ({evidence.get('object_type', '?')})"
        f"    zone: {evidence.get('target_zone', '?')}    persons: {evidence.get('person_count', '?')}",
        font_body,
        (20, 20, 20),
        gap_after=4,
    )
    add_line(
        f"evidence cues: {', '.join(active_cues) if active_cues else 'none'}",
        font_body,
        (20, 20, 20),
        gap_after=10,
    )
    add_line("RATIONALE", font_header, (90, 90, 90), gap_after=2)
    add_line(row.get("auto_rationale") or "(none)", font_body, (20, 20, 20), gap_after=10)
    add_line("CAT-V CAPTION", font_header, (90, 90, 90), gap_after=2)
    caption_text = row.get("object_caption") or row.get("dense_caption_en") or "(none)"
    add_line(caption_text, font_body, (20, 20, 20), gap_after=0)

    line_heights = [font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 6 for _, font, _, _ in blocks]
    panel_height = pad + sum(h + gap for h, (*_, gap) in zip(line_heights, blocks)) + pad

    canvas = Image.new("RGB", (width, frames_canvas.height + panel_height), (255, 255, 255))
    canvas.paste(frames_canvas, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, frames_canvas.height, width, frames_canvas.height + 5), fill=accent)
    y = frames_canvas.height + pad
    for (text, font, color, gap_after), line_height in zip(blocks, line_heights):
        draw.text((pad, y), text, fill=color, font=font)
        y += line_height + gap_after
    return canvas


def write_annotation_review_images(
    input_path: Path,
    out_dir: Path,
    *,
    max_width: int = 900,
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> int:
    """Render one review composite per row of a one-pass-labels JSONL."""
    if not input_path.exists():
        raise FileNotFoundError(f"Label JSONL not found: {input_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = iter_jsonl(input_path, skip_bad=True)
    if show_progress:
        from tqdm.auto import tqdm

        records = tqdm(records, total=count_jsonl(input_path), unit="row", desc="CAT-V review images")

    count = 0
    processed = 0
    skipped_existing = 0
    skipped_no_frames = 0
    progress_bar = records if show_progress else None
    for row in records:
        if limit is not None and processed >= limit:
            break
        processed += 1
        record_id = safe_path_part(_record_output_id(row))
        group_a, group_b = _record_storage_parts(row)
        dest = out_dir / group_a / group_b / f"{record_id}.jpg"
        if resume and dest.exists():
            skipped_existing += 1
            if progress_bar is not None:
                progress_bar.set_postfix(
                    written=count, skipped_existing=skipped_existing, skipped_no_frames=skipped_no_frames, refresh=False
                )
            continue
        if not any((_review_frame_path(row, tag) or Path()).exists() for tag in _REVIEW_FRAME_TAGS):
            skipped_no_frames += 1
            if progress_bar is not None:
                progress_bar.set_postfix(
                    written=count, skipped_existing=skipped_existing, skipped_no_frames=skipped_no_frames, refresh=False
                )
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        image = render_annotation_review_image(row, max_width=max_width)
        image.save(dest, quality=90)
        count += 1
        if progress_bar is not None:
            progress_bar.set_postfix(
                written=count, skipped_existing=skipped_existing, skipped_no_frames=skipped_no_frames, refresh=False
            )
    return count


write_egolife_annotation_review_images = write_annotation_review_images
