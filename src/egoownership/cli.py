"""Typer-based CLI. Installed as ``egoown``.

Subcommands mirror the pipeline stages (each step is resumable via JSONL).
The ``serve`` command launches the FastAPI annotator UI for collaborative
review.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from egoownership import pipeline
from egoownership.schema import Taxonomy
try:
    from ego_video_qa.cli import register_eval_command as _register_eval_command
except ModuleNotFoundError:
    _register_eval_command = None

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
download_app = typer.Typer(no_args_is_help=True, help="Dataset download helpers")
app.add_typer(download_app, name="download")

_CONSOLE = Console()

if _register_eval_command is not None:
    _register_eval_command(app, _CONSOLE)


# ---------- downloads ----------


@download_app.command("ego4d")
def download_ego4d(
    out: Path = typer.Option(Path("data/ego4d"), help="Destination directory"),
    videos: bool = typer.Option(False, help="Also emit video download commands"),
):
    """Ego4D is license-gated — prints the commands you need to run yourself."""
    from egoownership.download import ego4d
    ego4d.download(out, videos=videos)


@download_app.command("epic")
def download_epic(
    out: Path = typer.Option(Path("data/epic"), help="Destination directory"),
    videos: bool = typer.Option(False, help="Include video download instructions"),
):
    """Fetch EPIC-KITCHENS-100 annotations from GitHub."""
    from egoownership.download import epic_kitchens
    epic_kitchens.download(out, videos=videos)


@download_app.command("hd-epic")
def download_hd_epic(
    out: Path = typer.Option(Path("data/hd_epic"), help="Destination directory"),
    videos: bool = typer.Option(False, help="Include video download instructions"),
):
    """HD-EPIC annotation download instructions."""
    from egoownership.download import hd_epic
    hd_epic.download(out, videos=videos)


# ---------- ego4d-fetch-clips ----------


@app.command("ego4d-fetch-clips")
def ego4d_fetch_clips_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="BBox JSONL produced by extract-bboxes (ego4d dataset). "
             "Must contain video_id, source_video_start_sec, catv_duration_sec, video_path.",
    ),
    full_video_dir: Path = typer.Option(
        Path("data/ego4d/full_videos"),
        help="Temporary directory for downloading Ego4D full-scale videos. "
             "Each full video is deleted after its clips are extracted.",
    ),
    clips_dir: Path = typer.Option(
        None,
        help="Override clip output directory. Each clip is saved as "
             "<clips-dir>/<video_id>/<original filename>. "
             "If omitted, uses the video_path field from each record as-is.",
    ),
    limit: int = typer.Option(0, help="Hard cap on records processed (0 = no cap)"),
    no_delete: bool = typer.Option(
        False,
        "--no-delete",
        help="Keep full video after extracting clips (default: delete to save disk space)",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bars"),
):
    """Download Ego4D full videos, extract clips, delete full video.

    Groups input records by video_id, downloads each full video once via the
    official ego4d CLI, extracts each 30-second clip with ffmpeg (-c copy, no
    re-encode), then deletes the full video to reclaim disk space.

    After running this, the video_path fields in the JSONL already point to the
    extracted clips, so subsequent extract-bboxes / caption-bboxes runs are fast
    (no network fetch, no large-file seeking).

    Requires:
      - ego4d CLI installed: pip install ego4d
      - ffmpeg on PATH
    """
    from egoownership.ego4d_video import fetch_ego4d_clips

    if not input_jsonl.exists():
        raise typer.BadParameter(f"Input JSONL not found: {input_jsonl}")

    stats = fetch_ego4d_clips(
        input_jsonl,
        full_video_dir,
        clips_dir=clips_dir,
        limit=limit if limit > 0 else None,
        show_progress=progress,
        delete_full_video=not no_delete,
    )

    _CONSOLE.print(
        f"[green]Done.[/green] "
        f"total={stats['total']}  "
        f"extracted={stats['extracted']}  "
        f"skipped={stats['skipped']}  "
        f"failed={stats['failed']}  "
        f"videos_downloaded={stats['videos_downloaded']}  "
        f"videos_deleted={stats['videos_deleted']}"
    )


# ---------- filter ----------


@app.command("filter")
def filter_cmd(
    dataset: str = typer.Argument(..., help="One of: ego4d-fho, epic, hd-epic, egolife"),
    annotations: Path = typer.Option(..., help="Path to annotation file or directory"),
    out: Path = typer.Option(..., help="Output JSONL path for candidates"),
    taxonomy: str = typer.Option("C", help="Target taxonomy A, B, C, or D"),
    require_shared_noun: bool = typer.Option(
        True, help="Drop candidates whose nouns don't intersect the shared-table list"
    ),
    limit: int = typer.Option(0, help="Hard cap on candidates (0 = no cap)"),
):
    """Filter candidates by verb/noun whitelists."""
    try:
        tax = Taxonomy(taxonomy.upper())
    except ValueError:
        raise typer.BadParameter(f"taxonomy must be one of A/B/C/D, got {taxonomy!r}")
    n = pipeline.stage_filter(
        dataset=dataset,
        annotations_path=annotations,
        taxonomy=tax,
        out_path=out,
        require_shared_noun=require_shared_noun,
        limit=limit if limit > 0 else None,
    )
    _CONSOLE.print(f"[green]Wrote {n} candidates[/green] → {out}")


# ---------- egolife-annotate ----------


@app.command("egolife-annotate")
def egolife_annotate_cmd(
    annotations: Path = typer.Option(
        ...,
        help="EgoLife annotation JSON/JSONL file or directory with transcript/dense captions",
    ),
    out: Path = typer.Option(..., help="Output JSONL path for draft ownership annotations"),
    visual_metadata: Path = typer.Option(
        None,
        help="Optional JSON/JSONL keyed by clip_id/event_id with person_count and face_count",
    ),
    min_persons: int = typer.Option(
        2,
        help="Minimum visible non-wearer people required by visual metadata",
    ),
    require_face: bool = typer.Option(
        True,
        "--require-face/--no-require-face",
        help="Reject records with visual metadata indicating no visible face",
    ),
    include_rejected: bool = typer.Option(
        False,
        "--include-rejected",
        help="Write rejected records too, with filter_reasons",
    ),
    require_visual_pass: bool = typer.Option(
        False,
        "--require-visual-pass",
        help="Only keep records whose visual metadata passes person/face filters",
    ),
    limit: int = typer.Option(0, help="Hard cap on output rows (0 = no cap)"),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show a tqdm progress bar while writing draft annotations",
    ),
    output_format: str = typer.Option(
        "candidate",
        help="Output JSONL shape: candidate matches ClipCandidate, draft keeps verbose debugging fields",
    ),
):
    """Draft EgoLife ownership annotations from transcripts/dense captions.

    Captions/transcripts provide a cheap keep/reject/needs_vlm prefilter and a
    taxonomy seed. Use egolife-vlm-filter for frame-based person/face filtering.
    """
    from egoownership.egolife_annotations import (
        write_egolife_annotation_drafts,
    )

    if output_format not in ("candidate", "draft"):
        raise typer.BadParameter("--output-format must be one of: candidate, draft")

    n = write_egolife_annotation_drafts(
        annotations,
        out,
        visual_metadata_path=visual_metadata,
        min_persons=min_persons,
        require_face=require_face,
        include_rejected=include_rejected,
        require_visual_pass=require_visual_pass,
        limit=limit if limit > 0 else None,
        show_progress=progress,
        output_format=output_format,
    )
    _CONSOLE.print(f"[green]Wrote {n} EgoLife {output_format} records[/green] → {out}")


# ---------- egolife-vlm-filter ----------


@app.command("egolife-vlm-filter")
def egolife_vlm_filter_cmd(
    annotations: Path = typer.Option(
        ...,
        help="EgoLifeCap directory or EgoLife annotation JSON/JSONL",
    ),
    videos_root: Path = typer.Option(
        ...,
        help="Root containing EgoLife mp4 files, e.g. data/egolife/raw/data/EgoLife",
    ),
    out: Path = typer.Option(..., help="Output filtered EgoLife candidate JSONL path"),
    frames_dir: Path = typer.Option(
        None,
        help="Optional cache directory for extracted sparse frames",
    ),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        help="Open-source VLM checkpoint for visual person/face filtering",
    ),
    device: str = typer.Option("auto", help="Device for VLM: auto, cuda:0, cpu"),
    dtype: str = typer.Option("auto", help="Dtype: auto, float16, bfloat16, float32"),
    max_new_tokens: int = typer.Option(128, help="Maximum generated tokens per VLM judgement"),
    vlm_image_size: int = typer.Option(
        224,
        help="Resize each sparse frame to this square pixel size before VLM inference; 0 disables resizing",
    ),
    min_visible_people: int = typer.Option(
        2,
        help="Reject if fewer than this many visible people are seen across frames",
    ),
    require_face: bool = typer.Option(
        True,
        "--require-face/--no-require-face",
        help="Reject if the VLM reports no visible face",
    ),
    backend: str = typer.Option("ffmpeg", help="Frame extraction backend: ffmpeg or imageio"),
    limit: int = typer.Option(0, help="Hard cap on candidates (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
    split_by_person_day: bool = typer.Option(
        False,
        "--split-by-person-day/--single-file",
        help="When enabled, treat --out as a directory and save A*_*/DAY*.jsonl files",
    ),
    debug_samples: int = typer.Option(
        0,
        help="Print the first N passing candidate records for debugging",
    ),
):
    """Filter EgoLife annotations with caption hints and an open-source VLM."""
    from egoownership.egolife_visual_filter import (
        QwenVLSceneJudge,
        QwenVLSceneJudgeConfig,
        write_egolife_vlm_filtered_candidates,
    )

    if not annotations.exists():
        raise typer.BadParameter(
            f"EgoLife annotations path does not exist: {annotations}.",
            param_hint="--annotations",
        )

    judge = QwenVLSceneJudge(
        QwenVLSceneJudgeConfig(
            model_id=model_id,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            image_size=vlm_image_size,
        )
    )
    n = write_egolife_vlm_filtered_candidates(
        annotations,
        videos_root,
        out,
        frames_dir=frames_dir,
        judge=judge,
        min_visible_people=min_visible_people,
        require_face=require_face,
        backend=backend,
        limit=limit if limit > 0 else None,
        show_progress=progress,
        split_by_person_day=split_by_person_day,
        debug_samples=debug_samples,
    )
    _CONSOLE.print(f"[green]Wrote {n} EgoLife VLM-filtered candidates[/green] → {out}")


# ---------- egolife-qwen-nouns ----------


@app.command("egolife-qwen-nouns")
def egolife_qwen_nouns_cmd(
    annotations: Path = typer.Option(
        ...,
        help="EgoLifeCap directory containing DenseCaption and Transcript folders",
    ),
    out: Path = typer.Option(
        Path("data/egolife/egolife_table_caption_qwen_translations.jsonl"),
        help="Output JSONL with Qwen translations and spaCy noun/verb candidates",
    ),
    noun_summary_out: Path = typer.Option(
        Path("data/egolife/egolife_table_caption_qwen_noun_summary.jsonl"),
        help="Output JSONL summarizing noun frequency after Qwen translation",
    ),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-3B-Instruct",
        help="Text-only Qwen checkpoint used for Chinese-to-English translation",
    ),
    device: str = typer.Option("auto", help="Device for Qwen: auto, cpu, cuda:0"),
    dtype: str = typer.Option("auto", help="Dtype: auto, float16, bfloat16, float32"),
    max_new_tokens: int = typer.Option(96, help="Maximum generated tokens per translation"),
    local_files_only: bool = typer.Option(
        False,
        "--local-files-only/--allow-download",
        help="Use only locally cached model files",
    ),
    limit: int = typer.Option(0, help="Hard cap on table-caption rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Translate table-related EgoLife captions/transcripts with Qwen, then extract nouns."""
    from egoownership.egolife_qwen_nouns import (
        QwenCaptionTranslator,
        QwenTranslationConfig,
        write_qwen_translated_table_caption_nouns,
    )

    translator = QwenCaptionTranslator(
        QwenTranslationConfig(
            model_id=model_id,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            local_files_only=local_files_only,
        )
    )
    n = write_qwen_translated_table_caption_nouns(
        annotations,
        out,
        translate_fn=translator.translate,
        noun_summary_out=noun_summary_out,
        limit=limit if limit > 0 else None,
        show_progress=progress,
    )
    _CONSOLE.print(
        f"[green]Wrote {n} Qwen-translated table captions[/green] → {out}\n"
        f"[green]Noun summary[/green] → {noun_summary_out}"
    )


# ---------- ego4d-table-object-nouns ----------


@app.command("ego4d-table-object-nouns")
def ego4d_table_object_nouns_cmd(
    narration: Path = typer.Option(
        Path("ego4d_data/v2/annotations/narration.json"),
        help="Ego4D narration.json path",
    ),
    out: Path = typer.Option(
        Path("data/ego4d/ego4d_table_caption_object_nouns.jsonl"),
        help="Output JSONL object noun allowlist",
    ),
    require_observer: bool = typer.Option(
        False,
        "--require-observer/--all-table-narrations",
        help="Only mine nouns from #O observer narrations (default: all table narrations)",
    ),
    batch_size: int = typer.Option(2048, help="spaCy batch size for narration mining"),
    limit: int = typer.Option(0, help="Hard cap on table narrations scanned (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Mine spaCy object nouns from Ego4D table narrations into an allowlist JSONL."""
    from egoownership.catv_nouns import write_ego4d_table_caption_object_nouns

    narrations_with_nouns, noun_count = write_ego4d_table_caption_object_nouns(
        narration,
        out,
        require_observer=require_observer,
        batch_size=batch_size,
        limit=limit if limit > 0 else None,
        show_progress=progress,
    )
    _CONSOLE.print(
        f"[green]Wrote {noun_count} object nouns[/green] from "
        f"{narrations_with_nouns} table narrations → {out}"
    )


# ---------- sam2-extract-objects ----------


@app.command("sam2-extract-objects")
def sam2_extract_objects_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="Input benchmark JSONL, e.g. data/hf/.../jsonl/egolife.jsonl",
    ),
    frames_root: Path = typer.Option(
        Path("data/hf/ego-implicit-ownership-multiperson/frames"),
        help="Root containing frame images, e.g. data/hf/ego-implicit-ownership-multiperson/frames",
    ),
    egolife_videos_root: Path = typer.Option(
        None,
        help="Optional EgoLife video root, e.g. /data/video_datasets/EgoLife",
    ),
    ego4d_videos_root: Path = typer.Option(
        None,
        help="Optional Ego4D full_scale video root, e.g. /data/video_datasets/Ego4D/v2/full_scale",
    ),
    extracted_frames_dir: Path = typer.Option(
        Path("outputs/sam2_extracted_frames"),
        help="Where to cache frames extracted from videos when JSONL frame paths are absent",
    ),
    skip_source_datasets: str = typer.Option(
        "hd_epic",
        help="Comma-separated source_dataset values to skip, e.g. hd_epic",
    ),
    out: Path = typer.Option(..., help="Output JSONL with SAM-2 object candidates"),
    model_id: str = typer.Option(
        "facebook/sam-vit-base",
        help="SAM/SAM-2 mask-generation checkpoint. Default is SAM because current test env cannot load HF sam2_video.",
    ),
    device: str = typer.Option("auto", help="Device for mask generation: auto, cpu, cuda:0"),
    frame_tags: str = typer.Option(
        "t",
        help="Comma-separated sparse frame tags to process: t, t-1, t-2",
    ),
    min_area_ratio: float = typer.Option(
        0.001,
        help="Drop masks smaller than this image-area fraction",
    ),
    max_area_ratio: float = typer.Option(
        0.75,
        help="Drop masks larger than this image-area fraction",
    ),
    max_objects_per_frame: int = typer.Option(
        30,
        help="Keep at most this many SAM-2 object candidates per frame",
    ),
    nms_iou_threshold: float = typer.Option(
        0.90,
        help="Deduplicate highly overlapping mask boxes by this IoU threshold",
    ),
    limit: int = typer.Option(0, help="Hard cap on input rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Use SAM-2 masks to add generic object candidates to benchmark JSONL.

    Entries with no resolvable frame image or no usable object masks are skipped.
    SAM-2 does not provide semantic names; labels are saved as ``sam2_object`` for
    later VLM target selection/naming.
    """
    from egoownership.sam2_objects import (
        FRAME_KEYS,
        Sam2ObjectConfig,
        Sam2ObjectExtractor,
        write_sam2_object_jsonl,
    )

    tags = tuple(tag.strip() for tag in frame_tags.split(",") if tag.strip())
    unknown = [tag for tag in tags if tag not in FRAME_KEYS]
    if unknown:
        raise typer.BadParameter(f"Unknown frame tag(s): {unknown}. Use t,t-1,t-2")

    extractor = Sam2ObjectExtractor(
        Sam2ObjectConfig(
            model_id=model_id,
            device=device,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            max_objects_per_frame=max_objects_per_frame,
            nms_iou_threshold=nms_iou_threshold,
        )
    )
    video_roots = {}
    if egolife_videos_root is not None:
        video_roots["egolife"] = egolife_videos_root
    if ego4d_videos_root is not None:
        video_roots["ego4d"] = ego4d_videos_root
        video_roots["ego4d_fho"] = ego4d_videos_root
    skip_sources = {item.strip() for item in skip_source_datasets.split(",") if item.strip()}
    n = write_sam2_object_jsonl(
        input_jsonl,
        frames_root,
        out,
        extractor=extractor,
        video_roots=video_roots or None,
        extracted_frames_dir=extracted_frames_dir,
        skip_source_datasets=skip_sources,
        frame_tags=tags,
        limit=limit if limit > 0 else None,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} entries with SAM-2 object candidates[/green] → {out}")


@app.command("extract-bboxes")
def extract_bboxes_cmd(
    dataset: str = typer.Option(
        ...,
        help="Dataset adapter: egolife, ego4d, or generic. "
        "'generic' accepts any JSONL whose records already contain video_path, clip_id, "
        "video_id, start_sec, end_sec, and nouns fields. "
        "New adapters can be registered via register_catv_dataset_adapter().",
    ),
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="Input annotation file. "
        "egolife: Qwen-translated JSONL or EgoLifeCap directory. "
        "ego4d: narration.json or candidates JSONL. "
        "generic: any JSONL with the required fields above.",
    ),
    videos_root: Path = typer.Option(
        None,
        help="Root directory containing source videos. "
        "Required for egolife and generic. "
        "For ego4d, defaults to --scratch-root (and videos are auto-downloaded if missing).",
    ),
    out: Path = typer.Option(
        None,
        help="Output bbox JSONL. Defaults to outputs/{dataset}/bbox_objects.jsonl",
    ),
    frames_dir: Path = typer.Option(
        None,
        help="Cache directory for sampled frames and bbox visualizations. Defaults to outputs/{dataset}/catv_first_frames",
    ),
    object_nouns: Path = typer.Option(
        None,
        help="JSONL allowlist of object nouns. Defaults to data/{dataset}/{dataset}_table_caption_object_nouns.jsonl if it exists.",
    ),
    ego4d_clip_sec: float = typer.Option(
        30.0,
        help="[Ego4D] Clip length in seconds, centered on each narration timestamp",
    ),
    auto_download: bool = typer.Option(
        True,
        "--auto-download/--no-auto-download",
        help="[Ego4D] Download missing full-scale videos into --videos-root via ego4d CLI",
    ),
    require_observer: bool = typer.Option(
        True,
        "--require-observer/--all-table-narrations",
        help="[Ego4D narration.json] Keep only #O observer narrations mentioning a table",
    ),
    sam_model_id: str = typer.Option(
        "facebook/sam3",
        help="SAM/SAM-3 checkpoint. Use facebook/sam3 with --sam-backend sam3.",
    ),
    sam_backend: str = typer.Option(
        "sam3",
        help="SAM backend: sam3 for noun-prompted boxes, transformers for generic SAM masks",
    ),
    sam_device: str = typer.Option("auto", help="Device for SAM mask generation: auto, cpu, cuda:0"),
    min_area_ratio: float = typer.Option(0.001, help="Drop masks smaller than this area fraction"),
    max_area_ratio: float = typer.Option(0.75, help="Drop masks larger than this area fraction"),
    max_objects_per_frame: int = typer.Option(30, help="Maximum raw masks per sampled frame"),
    max_objects_per_caption: int = typer.Option(5, help="Maximum target objects retained per caption cue"),
    reference_frame: str = typer.Option(
        "midpoint",
        help="Single caption frame used for bbox extraction: start, midpoint, or end",
    ),
    nms_iou_threshold: float = typer.Option(0.90, help="Mask box NMS IoU threshold"),
    resume: bool = typer.Option(
        True,
        "--resume/--overwrite",
        help="Append to existing bbox output and skip IDs already written; --overwrite starts fresh",
    ),
    limit: int = typer.Option(0, help="Hard cap on filtered caption cues (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Extract target-object boxes before CAT-V captioning (multi-dataset)."""
    from egoownership.catv_datasets import normalize_dataset_id
    from egoownership.catv_pipeline import write_caption_bboxes
    from egoownership.sam2_objects import Sam2ObjectConfig, Sam2ObjectExtractor

    ds = normalize_dataset_id(dataset)
    out_dir = Path("outputs") / ds
    resolved_out = out or (out_dir / "bbox_objects.jsonl")
    resolved_frames_dir = frames_dir or (out_dir / "catv_first_frames")

    _data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    _default_nouns = _data_dir / ds / f"{ds}_table_caption_object_nouns.jsonl"
    resolved_object_nouns = object_nouns or (_default_nouns if _default_nouns.exists() else None)

    if ds in {"ego4d", "ego4d_fho"}:
        resolved_videos_root = videos_root or Path("data/ego4d/videos")
    elif videos_root is None:
        raise typer.BadParameter(f"--videos-root is required for --dataset {dataset}")
    else:
        resolved_videos_root = videos_root

    extractor = Sam2ObjectExtractor(
        Sam2ObjectConfig(
            model_id=sam_model_id,
            backend=sam_backend,
            device=sam_device,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            max_objects_per_frame=max_objects_per_frame,
            nms_iou_threshold=nms_iou_threshold,
        )
    )
    n = write_caption_bboxes(
        input_jsonl,
        resolved_videos_root,
        resolved_out,
        dataset=dataset,
        extractor=extractor,
        frames_dir=resolved_frames_dir,
        object_nouns_path=resolved_object_nouns,
        max_objects_per_record=max_objects_per_caption,
        reference_frame=reference_frame,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
        ego4d_clip_window_sec=ego4d_clip_sec if ds in {"ego4d", "ego4d_fho"} else None,
        ego4d_auto_download=auto_download if ds in {"ego4d", "ego4d_fho"} else False,
        ego4d_require_observer=require_observer if ds in {"ego4d", "ego4d_fho"} else True,
    )
    _CONSOLE.print(f"[green]Wrote {n} {dataset} bbox rows[/green] → {resolved_out}")


@app.command("caption-bboxes")
def caption_bboxes_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="BBox JSONL produced by extract-bboxes",
    ),
    out: Path = typer.Option(..., help="Output JSONL with CAT-V object captions"),
    catv_command_template: str = typer.Option(
        None,
        help="Shell command template for CAT-V captioning. If unset, built automatically from "
        "--captioner-backend/--caption-model-path/--caption-device.",
    ),
    captioner_backend: str = typer.Option(
        "qwen3vl",
        help="qwen3vl (default): caption via Qwen3-VL (scripts/run_catv_one_object.py, runs under the sam2hf env). "
        "Ignored if --catv-command-template is set.",
    ),
    caption_model_path: str = typer.Option(
        None,
        help="Override the captioning model id. Default: Qwen/Qwen3-VL-8B-Instruct.",
    ),
    caption_device: str = typer.Option("cuda:0", help="Device for the captioning model"),
    caption_fps: float = typer.Option(
        1.0,
        help="Frames/sec sampled from each object's whole clip before captioning. With "
        "--whole-video, even fps=1 yields plenty of frames since EgoLife source clips run "
        "~17-30s; total frames are still capped at --caption-max-frames downstream regardless.",
    ),
    caption_max_frames: int = typer.Option(
        16,
        help="Hard cap on frames sent to the captioning model per object, after sampling at "
        "--caption-fps across the whole clip.",
    ),
    catv_work_dir: Path = typer.Option(
        Path("outputs/catv_work"),
        help="Directory for CAT-V temporary work files",
    ),
    catv_visualization_dir: Path = typer.Option(
        Path("outputs/catv_visualizations"),
        help="Directory for persistent SAM2/CAT-V masked visualization videos",
    ),
    keep_catv_work_dir: bool = typer.Option(
        False,
        "--keep-catv-work-dir",
        help="Keep per-object CAT-V intermediate files after successful captions",
    ),
    save_catv_visualizations: bool = typer.Option(
        True,
        "--save-catv-visualizations/--no-save-catv-visualizations",
        help="Copy each object's SAM2 masked video to catv-visualization-dir",
    ),
    show_catv_logs: bool = typer.Option(
        False,
        "--catv-logs",
        help="Stream CAT-V subprocess stdout/stderr instead of suppressing it",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--overwrite",
        help="Append to existing output and skip IDs already written; --overwrite starts fresh",
    ),
    limit: int = typer.Option(0, help="Hard cap on bbox rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Run CAT-V captions for existing bbox rows (dataset-agnostic)."""
    from egoownership.catv_pipeline import (
        CatVCommandCaptioner,
        write_catv_captions_from_bbox_jsonl,
    )

    if catv_command_template is None:
        if captioner_backend != "qwen3vl":
            raise typer.BadParameter("--captioner-backend must be 'qwen3vl'")
        import sys as _sys
        model_path = caption_model_path or "Qwen/Qwen3-VL-8B-Instruct"
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_catv_one_object.py"
        catv_command_template = (
            f"{_sys.executable} {script_path} "
            "--video {video_path} --first-frame {first_frame_path} "
            "--bbox {bbox_path} --out {output_json} "
            f"--catv-device {caption_device} "
            f"--start-sec {{start_sec}} --fps {caption_fps:g} --whole-video "
            f"--max-frames-num {caption_max_frames} "
            f"--captioner-backend qwen3vl "
            f"--model-path {model_path} "
            "--caption {object_nouns}"
        )

    captioner = CatVCommandCaptioner(
        command_template=catv_command_template,
        work_root=catv_work_dir,
        visualization_root=catv_visualization_dir,
        keep_work_dir=keep_catv_work_dir,
        suppress_output=not show_catv_logs,
        save_visualizations=save_catv_visualizations,
    )
    n = write_catv_captions_from_bbox_jsonl(
        input_jsonl,
        out,
        captioner=captioner,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} CAT-V object captions[/green] → {out}")


@app.command("caption-bboxes-batch")
def caption_bboxes_batch_cmd(
    input_jsonl: Path = typer.Option(..., "--input", help="BBox JSONL produced by extract-bboxes"),
    out: Path = typer.Option(
        None,
        help="Output JSONL with object captions. Defaults to outputs/{dataset}/captions.jsonl",
    ),
    dataset: str = typer.Option(
        None,
        help="Dataset name (egolife, ego4d, ...) used to derive default output path",
    ),
    captioner_backend: str = typer.Option("qwen3vl", help="Captioner backend (only qwen3vl is supported)"),
    caption_model_path: str = typer.Option(
        "Qwen/Qwen3-VL-8B-Instruct",
        help="Captioning model id or local path",
    ),
    caption_device: str = typer.Option("cuda:0", help="Device for both SAM-2 and the VLM (used when --devices is not set)"),
    devices: str = typer.Option(
        None,
        help="Comma-separated list of devices for parallel multi-GPU processing, e.g. cuda:0,cuda:1. "
             "Overrides --caption-device.",
    ),
    caption_fps: float = typer.Option(1.0, help="Frames/sec extracted from each clip"),
    caption_max_frames: int = typer.Option(16, help="Max frames sent to the VLM per object"),
    caption_max_side: int = typer.Option(448, help="Resize each frame's longer side to this many pixels"),
    mask_model_path: str = typer.Option(
        "facebook/sam2.1-hiera-base-plus",
        help="SAM-2 model: HuggingFace ID (e.g. facebook/sam2.1-hiera-base-plus) or local .pt path",
    ),
    catv_python: str = typer.Option(
        None,
        help="Python interpreter for the SAM-2 batch step. Defaults to sys.executable (current env).",
    ),
    qwen_vl_python: str = typer.Option(
        None,
        help="Python interpreter for the Qwen3-VL batch step. Defaults to sys.executable (current env).",
    ),
    catv_visualization_dir: Path = typer.Option(
        None,
        help="Copy SAM-2 masked videos here for review. Omit to skip.",
    ),
    batch_jobs_dir: Path = typer.Option(
        None,
        help="Directory for the batch job JSONL files. Defaults to <out-dir>/catv_batch_jobs.",
    ),
    whole_video: bool = typer.Option(True, "--whole-video/--clip-only", help="Track the full source clip"),
    resume: bool = typer.Option(True, "--resume/--overwrite"),
    limit: int = typer.Option(0, help="Hard cap on rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
):
    """Batch CAT-V captioning: load SAM-2 and the VLM once for the whole JSONL.

    This is the fast alternative to caption-bboxes. Instead of reloading both
    models per row (~90s overhead × N rows), it pays the load cost once and
    processes all objects in sequence. Recommended for runs with > 10 rows.
    """
    from egoownership.catv_pipeline import write_catv_captions_batch

    ds = dataset or input_jsonl.parent.name
    resolved_out = out or (Path("outputs") / ds / "captions.jsonl")

    resolved_devices = [d.strip() for d in devices.split(",")] if devices else None

    n = write_catv_captions_batch(
        input_jsonl,
        resolved_out,
        mask_model_path=mask_model_path,
        catv_device=caption_device,
        devices=resolved_devices,
        fps=caption_fps,
        whole_video=whole_video,
        max_frames=caption_max_frames,
        max_side=caption_max_side,
        captioner_backend=captioner_backend,
        caption_model_path=caption_model_path,
        qwen_vl_python=qwen_vl_python,
        catv_python=catv_python,
        visualization_root=catv_visualization_dir,
        batch_jobs_dir=batch_jobs_dir,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} CAT-V object captions (batch)[/green] → {resolved_out}")


@app.command("one-pass-labels")
def one_pass_labels_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="Object-description JSONL from the caption-bboxes stage",
    ),
    out: Path = typer.Option(
        None,
        help="Output JSONL with sparse benchmark frames, evidence, taxonomy, and auto GT labels. "
             "Defaults to outputs/{dataset}/labels.jsonl",
    ),
    dataset: str = typer.Option(
        None,
        help="Dataset name (egolife, ego4d, ...) used to derive default output paths",
    ),
    frames_dir: Path = typer.Option(
        None,
        help="Cache directory for sampled t-2/t-1/t frames. Defaults to outputs/{dataset}/one_pass_sparse_frames",
    ),
    detect_persons: bool = typer.Option(
        False,
        "--detect-persons/--no-detect-persons",
        help="Run person detector on target frame while building evidence labels",
    ),
    sam3_model_id: str = typer.Option(
        None,
        help="SAM-3 HF model ID (e.g. facebook/sam3); if set, re-runs SAM-3 on frame t to refine the object bbox and skips entries with confidence ≤ 0.5",
    ),
    sam2_tracking_model_id: str = typer.Option(
        None,
        "--sam2-track",
        help="SAM-2.1 checkpoint path for backward tracking (e.g. /path/to/sam2.1_hiera_base_plus.pt). "
             "Re-extracts per-frame bbox for t-1/t-2 by propagating the t-frame mask backward. "
             "Automatically enables --detect-persons.",
    ),
    sam2_tracking_device: str = typer.Option(
        "cuda",
        "--sam2-device",
        help="Device for SAM-2 tracking (cuda, cuda:0, cpu)",
    ),
    decision_backend: str = typer.Option(
        "rules",
        help="How to decide taxonomy/ground-truth from evidence: 'rules' or 'llm'",
    ),
    object_type_backend: str = typer.Option(
        "rules",
        help="How to classify object type (personal/shared/generic_object): 'rules' or 'llm'",
    ),
    decision_model_id: str = typer.Option(
        "Qwen/Qwen3-4B",
        help="Text LLM checkpoint used when --decision-backend=llm or --object-type-backend=llm",
    ),
    decision_device: str = typer.Option("auto", help="Device for the LLM decider: auto, cpu, cuda:0"),
    review_dir: Path = typer.Option(
        None,
        help="If set, render a t-2/t-1/t composite review image per row here",
    ),
    review_max_width: int = typer.Option(900, help="Max pixel width of each stacked frame in the review composite"),
    resume: bool = typer.Option(
        True,
        "--resume/--overwrite",
        help="Append to existing output and skip IDs already written; --overwrite starts fresh",
    ),
    limit: int = typer.Option(0, help="Hard cap on object-description rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Build ownership labels from CAT-V object descriptions (multi-dataset)."""
    from egoownership.catv_pipeline import write_one_pass_labels

    decision_fn = None
    if decision_backend == "llm":
        from egoownership.catv_evidence_label import LLMTaxonomyDecider

        decision_fn = LLMTaxonomyDecider(model_id=decision_model_id, device=decision_device)
    elif decision_backend != "rules":
        raise typer.BadParameter("--decision-backend must be 'rules' or 'llm'")

    object_type_fn = None
    if object_type_backend == "llm":
        from egoownership.catv_evidence_label import LLMObjectTypeClassifier

        object_type_fn = LLMObjectTypeClassifier(model_id=decision_model_id, device=decision_device)
    elif object_type_backend != "rules":
        raise typer.BadParameter("--object-type-backend must be 'rules' or 'llm'")

    ds = dataset or input_jsonl.parent.name
    resolved_out = out or (Path("outputs") / ds / "labels.jsonl")
    resolved_frames_dir = frames_dir or (Path("outputs") / ds / "one_pass_sparse_frames")
    progress_desc = f"{ds} one-pass labels"

    n = write_one_pass_labels(
        input_jsonl,
        resolved_out,
        frames_dir=resolved_frames_dir,
        detect_persons=detect_persons,
        decision_fn=decision_fn,
        object_type_fn=object_type_fn,
        sam3_model_id=sam3_model_id or None,
        sam2_tracking_model_id=sam2_tracking_model_id or None,
        sam2_tracking_device=sam2_tracking_device,
        review_dir=review_dir,
        review_max_width=review_max_width,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
        dataset=dataset,
        progress_desc=progress_desc,
    )
    _CONSOLE.print(f"[green]Wrote {n} one-pass labels[/green] → {resolved_out}")


@app.command("visualize-labels")
def visualize_labels_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="One-pass labels JSONL with frame_t_minus_2/1_path, frame_t_path, evidence, auto_* fields",
    ),
    out_dir: Path = typer.Option(..., help="Directory for per-record review composite images"),
    max_width: int = typer.Option(900, help="Max pixel width of each stacked frame"),
    resume: bool = typer.Option(
        True,
        "--resume/--overwrite",
        help="Skip records whose review image already exists; --overwrite re-renders everything",
    ),
    limit: int = typer.Option(0, help="Hard cap on input rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Render per-record review images: t-2/t-1/t frames stacked row-by-row,
    with a caption panel below showing the auto taxonomy/ground-truth and
    supporting evidence."""
    from egoownership.catv_pipeline import write_annotation_review_images

    n = write_annotation_review_images(
        input_jsonl,
        out_dir,
        max_width=max_width,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} annotation review images[/green] → {out_dir}")


@app.command("egolife-review-gradio")
def egolife_review_gradio_cmd(
    labels: Path = typer.Option(
        Path("outputs/egolife/annotations_llm.jsonl"),
        help="One-pass labels JSONL used to generate the review images",
    ),
    review_dir: Path = typer.Option(
        Path("outputs/egolife/annotation_review"),
        help="Directory of review composite JPGs from egolife-visualize-labels",
    ),
    host: str = typer.Option("0.0.0.0", help="Bind address (0.0.0.0 for LAN sharing)"),
    port: int = typer.Option(
        7860,
        help="Gradio port (auto-picks the next free port if busy)",
        envvar="GRADIO_SERVER_PORT",
    ),
    share: bool = typer.Option(
        False,
        "--share/--no-share",
        help="Create a temporary public gradio.live link",
    ),
):
    """Browse and edit EgoLife annotation review composites in a Gradio UI.

    Saves human corrections to review_taxonomy, review_ground_truth, and
    review_evidence in the labels JSONL.
    """
    from egoownership.egolife_review_gradio import launch_egolife_review_gradio, review_stats

    stats = review_stats(labels, review_dir)
    _CONSOLE.print(
        f"[green]Loaded {stats['entries']} review images[/green] "
        f"({stats['needs_review']} flagged needs_review)"
    )
    launch_egolife_review_gradio(
        labels,
        review_dir,
        host=host,
        port=port,
        share=share,
    )


# ---------- vlm-crosscheck ----------


@app.command("vlm-crosscheck")
def vlm_crosscheck_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="labels.jsonl produced by one-pass-labels",
    ),
    out: Path = typer.Option(
        None,
        help="Output JSONL with per-judge predictions and agreement stats. "
             "Defaults to outputs/{dataset}/crosscheck.jsonl",
    ),
    dataset: str = typer.Option(
        None,
        help="Dataset name used to derive default output path (egolife, ego4d, …)",
    ),
    frames_root: Path = typer.Option(
        None,
        help="Root directory for resolving relative frame paths in the JSONL",
    ),
    judges: list[str] = typer.Option(
        [],
        "--judge",
        help="Judge spec: 'qwen:<model_id>', 'anthropic:<model_id>', or 'openai:<model_id>'. "
             "Repeat for multiple judges, e.g. --judge qwen:Qwen/Qwen2.5-VL-7B-Instruct "
             "--judge anthropic:claude-sonnet-4-6",
    ),
    qwen_device: str = typer.Option("auto", help="Device for local Qwen VL judges"),
    qwen_dtype: str = typer.Option("auto", help="Dtype for Qwen VL: auto, float16, bfloat16"),
    qwen_max_tokens: int = typer.Option(256, help="Max new tokens for Qwen VL response"),
    remote_max_tokens: int = typer.Option(512, help="Max tokens for Anthropic/OpenAI response"),
    resume: bool = typer.Option(True, "--resume/--overwrite"),
    limit: int = typer.Option(0, help="Hard cap on records processed (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
):
    """Cross-check ownership labels with multiple VLM judges.

    Each judge independently predicts MINE / PERSON_k / SHARED / AMBIGUOUS from
    the three sparse frames + narration + object description, without seeing the
    pipeline's own prediction.  Agreement with auto_ground_truth is recorded per
    judge, along with a majority-vote label.

    Examples:

    \\b
        # Two judges: local Qwen + Claude
        egoown vlm-crosscheck \\
            --input outputs/egolife/labels.jsonl \\
            --judge qwen:Qwen/Qwen2.5-VL-7B-Instruct \\
            --judge anthropic:claude-sonnet-4-6

        # Three judges
        egoown vlm-crosscheck \\
            --input outputs/ego4d/labels.jsonl \\
            --judge anthropic:claude-sonnet-4-6 \\
            --judge openai:gpt-4o \\
            --judge qwen:Qwen/Qwen2.5-VL-7B-Instruct
    """
    from egoownership.vlm_crosscheck import (
        AnthropicOwnershipJudge,
        AnthropicOwnershipJudgeConfig,
        GeminiOwnershipJudge,
        GeminiOwnershipJudgeConfig,
        OpenAIOwnershipJudge,
        OpenAIOwnershipJudgeConfig,
        QwenOwnershipJudge,
        QwenOwnershipJudgeConfig,
        write_crosscheck_jsonl,
    )

    if not judges:
        raise typer.BadParameter("Specify at least one --judge, e.g. --judge anthropic:claude-sonnet-4-6")

    judge_objs = []
    for spec in judges:
        if ":" in spec:
            backend, model_id = spec.split(":", 1)
        else:
            backend, model_id = spec, spec
        backend = backend.lower()

        if backend == "qwen":
            judge_objs.append(QwenOwnershipJudge(QwenOwnershipJudgeConfig(
                model_id=model_id,
                device=qwen_device,
                dtype=qwen_dtype,
                max_new_tokens=qwen_max_tokens,
            )))
        elif backend == "anthropic":
            judge_objs.append(AnthropicOwnershipJudge(AnthropicOwnershipJudgeConfig(
                model_id=model_id,
                max_tokens=remote_max_tokens,
            )))
        elif backend == "openai":
            judge_objs.append(OpenAIOwnershipJudge(OpenAIOwnershipJudgeConfig(
                model_id=model_id,
                max_tokens=remote_max_tokens,
            )))
        elif backend == "gemini":
            judge_objs.append(GeminiOwnershipJudge(GeminiOwnershipJudgeConfig(
                model_id=model_id,
                max_tokens=remote_max_tokens,
            )))
        else:
            raise typer.BadParameter(
                f"Unknown judge backend {backend!r}. Use qwen, anthropic, openai, or gemini."
            )

    ds = dataset or input_jsonl.parent.name
    resolved_out = out or (Path("outputs") / ds / "crosscheck.jsonl")

    n = write_crosscheck_jsonl(
        input_jsonl,
        resolved_out,
        judge_objs,
        frames_root=frames_root,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} cross-check rows[/green] → {resolved_out}")


# ---------- sam2-vlm-build ----------


@app.command("vlm-ground-objects")
def vlm_ground_objects_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="Input benchmark JSONL, e.g. data/hf/.../jsonl/egolife.jsonl",
    ),
    frames_root: Path = typer.Option(
        Path("data/hf/ego-implicit-ownership-multiperson/frames"),
        help="Root containing frame images",
    ),
    out: Path = typer.Option(..., help="Output JSONL with VLM nouns and Grounding DINO boxes"),
    vlm_model_id: str = typer.Option(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        help="Open-source VLM checkpoint for tabletop noun proposal",
    ),
    vlm_device: str = typer.Option("auto", help="Device for VLM: auto, cpu, cuda:0"),
    vlm_dtype: str = typer.Option("auto", help="VLM dtype: auto, float16, bfloat16, float32"),
    max_new_tokens: int = typer.Option(192, help="Maximum generated tokens per VLM noun proposal"),
    dino_model_id: str = typer.Option(
        "IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO checkpoint for localizing proposed nouns",
    ),
    dino_device: str = typer.Option("auto", help="Device for Grounding DINO: auto, cpu, cuda:0"),
    box_threshold: float = typer.Option(0.25, help="Grounding DINO box threshold"),
    text_threshold: float = typer.Option(0.20, help="Grounding DINO text threshold"),
    max_detections: int = typer.Option(30, help="Maximum grounded detections per frame"),
    frame_tag: str = typer.Option("t", help="Sparse frame tag to process: t, t-1, or t-2"),
    limit: int = typer.Option(0, help="Hard cap on input rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """List tabletop nouns with a VLM, then localize them with Grounding DINO.

    This command writes object nouns and positions only. It strips ownership
    labels/rationales from copied input rows.
    """
    from egoownership.sam2_objects import FRAME_KEYS
    from egoownership.vlm_ground_objects import (
        GroundingDinoObjectGrounder,
        GroundingDinoObjectGrounderConfig,
        QwenNounProposer,
        QwenNounProposerConfig,
        write_vlm_grounded_objects,
    )

    if frame_tag not in FRAME_KEYS:
        raise typer.BadParameter(f"Unknown --frame-tag {frame_tag!r}. Use t,t-1,t-2")

    noun_proposer = QwenNounProposer(
        QwenNounProposerConfig(
            model_id=vlm_model_id,
            device=vlm_device,
            dtype=vlm_dtype,
            max_new_tokens=max_new_tokens,
        )
    )
    grounder = GroundingDinoObjectGrounder(
        GroundingDinoObjectGrounderConfig(
            model_id=dino_model_id,
            device=None if dino_device in ("auto", "") else dino_device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_detections=max_detections,
        )
    )
    n = write_vlm_grounded_objects(
        input_jsonl,
        frames_root,
        out,
        noun_proposer=noun_proposer,
        grounder=grounder,
        frame_tag=frame_tag,
        limit=limit if limit > 0 else None,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} VLM-grounded object entries[/green] → {out}")


@app.command("sam2-vlm-build")
def sam2_vlm_build_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="Input benchmark JSONL, e.g. data/hf/.../jsonl/egolife.jsonl",
    ),
    frames_root: Path = typer.Option(
        Path("data/hf/ego-implicit-ownership-multiperson/frames"),
        help="Root containing frame images, e.g. data/hf/ego-implicit-ownership-multiperson/frames",
    ),
    egolife_videos_root: Path = typer.Option(
        None,
        help="Optional EgoLife video root, e.g. /data/video_datasets/EgoLife",
    ),
    ego4d_videos_root: Path = typer.Option(
        None,
        help="Optional Ego4D full_scale video root, e.g. /data/video_datasets/Ego4D/v2/full_scale",
    ),
    extracted_frames_dir: Path = typer.Option(
        Path("outputs/sam2_extracted_frames"),
        help="Where to cache frames extracted from videos when JSONL frame paths are absent",
    ),
    annotated_frames_dir: Path = typer.Option(
        Path("outputs/sam2_vlm_annotated_frames"),
        help="Where numbered SAM object-box images are saved for VLM inspection",
    ),
    skip_source_datasets: str = typer.Option(
        "hd_epic",
        help="Comma-separated source_dataset values to skip, e.g. hd_epic",
    ),
    out: Path = typer.Option(..., help="Final output JSONL with object nouns and box positions"),
    sam_model_id: str = typer.Option(
        "facebook/sam-vit-base",
        help="SAM/SAM-2 mask-generation checkpoint. Default is SAM because current test env cannot load HF sam2_video.",
    ),
    sam_device: str = typer.Option("auto", help="Device for SAM mask generation: auto, cpu, cuda:0"),
    vlm_model_id: str = typer.Option(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        help="Open-source VLM checkpoint for object noun naming",
    ),
    vlm_device: str = typer.Option("auto", help="Device for VLM: auto, cpu, cuda:0"),
    vlm_dtype: str = typer.Option("auto", help="VLM dtype: auto, float16, bfloat16, float32"),
    max_new_tokens: int = typer.Option(384, help="Maximum generated tokens per VLM judgement"),
    frame_tag: str = typer.Option("t", help="Sparse frame tag to process: t, t-1, or t-2"),
    min_area_ratio: float = typer.Option(
        0.001,
        help="Drop masks smaller than this image-area fraction",
    ),
    max_area_ratio: float = typer.Option(
        0.75,
        help="Drop masks larger than this image-area fraction",
    ),
    max_objects_per_frame: int = typer.Option(
        30,
        help="Keep at most this many SAM/SAM-2 object candidates",
    ),
    nms_iou_threshold: float = typer.Option(
        0.90,
        help="Deduplicate highly overlapping mask boxes by this IoU threshold",
    ),
    limit: int = typer.Option(0, help="Hard cap on input rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Run SAM/SAM-2 object proposals and VLM noun labeling as one JSONL pipeline."""
    from egoownership.sam2_objects import FRAME_KEYS, Sam2ObjectConfig, Sam2ObjectExtractor
    from egoownership.sam2_vlm_label import (
        QwenObjectLabelJudge,
        QwenObjectLabelJudgeConfig,
        write_sam2_vlm_object_labels,
    )

    if frame_tag not in FRAME_KEYS:
        raise typer.BadParameter(f"Unknown --frame-tag {frame_tag!r}. Use t,t-1,t-2")

    extractor = Sam2ObjectExtractor(
        Sam2ObjectConfig(
            model_id=sam_model_id,
            device=sam_device,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            max_objects_per_frame=max_objects_per_frame,
            nms_iou_threshold=nms_iou_threshold,
        )
    )
    judge = QwenObjectLabelJudge(
        QwenObjectLabelJudgeConfig(
            model_id=vlm_model_id,
            device=vlm_device,
            dtype=vlm_dtype,
            max_new_tokens=max_new_tokens,
        )
    )
    video_roots = {}
    if egolife_videos_root is not None:
        video_roots["egolife"] = egolife_videos_root
    if ego4d_videos_root is not None:
        video_roots["ego4d"] = ego4d_videos_root
        video_roots["ego4d_fho"] = ego4d_videos_root
    skip_sources = {item.strip() for item in skip_source_datasets.split(",") if item.strip()}
    n = write_sam2_vlm_object_labels(
        input_jsonl,
        frames_root,
        out,
        extractor=extractor,
        judge=judge,
        frame_tag=frame_tag,
        video_roots=video_roots or None,
        extracted_frames_dir=extracted_frames_dir,
        annotated_frames_dir=annotated_frames_dir,
        skip_source_datasets=skip_sources,
        limit=limit if limit > 0 else None,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} SAM+VLM object noun entries[/green] → {out}")


# ---------- sam2-vlm-label-objects ----------


@app.command("sam2-vlm-label-objects")
def sam2_vlm_label_objects_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="Input JSONL produced by sam2-extract-objects",
    ),
    frames_root: Path = typer.Option(
        Path("data/hf/ego-implicit-ownership-multiperson/frames"),
        help="Root containing benchmark frame images",
    ),
    out: Path = typer.Option(..., help="Output JSONL with VLM object nouns and box positions"),
    annotated_frames_dir: Path = typer.Option(
        Path("outputs/sam2_vlm_annotated_frames"),
        help="Where numbered SAM object-box images are saved for VLM inspection",
    ),
    model_id: str = typer.Option(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        help="Open-source VLM checkpoint for object noun naming",
    ),
    device: str = typer.Option("auto", help="Device for VLM: auto, cpu, cuda:0"),
    dtype: str = typer.Option("auto", help="Dtype: auto, float16, bfloat16, float32"),
    max_new_tokens: int = typer.Option(384, help="Maximum generated tokens per VLM judgement"),
    frame_tag: str = typer.Option("t", help="Sparse frame tag to label: t, t-1, or t-2"),
    limit: int = typer.Option(0, help="Hard cap on input rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Use a VLM to name SAM-2 object proposals.

    SAM/SAM-2 only proposes generic masks. This stage shows numbered boxes to a
    VLM and asks it to write object nouns only, without ownership inference.
    """
    from egoownership.sam2_objects import FRAME_KEYS
    from egoownership.sam2_vlm_label import (
        QwenObjectLabelJudge,
        QwenObjectLabelJudgeConfig,
        write_vlm_object_labels,
    )

    if frame_tag not in FRAME_KEYS:
        raise typer.BadParameter(f"Unknown --frame-tag {frame_tag!r}. Use t,t-1,t-2")

    judge = QwenObjectLabelJudge(
        QwenObjectLabelJudgeConfig(
            model_id=model_id,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
    )
    n = write_vlm_object_labels(
        input_jsonl,
        frames_root,
        out,
        judge=judge,
        frame_tag=frame_tag,
        annotated_frames_dir=annotated_frames_dir,
        limit=limit if limit > 0 else None,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} VLM object noun entries[/green] → {out}")


# ---------- new-filter ----------


@app.command("new-filter")
def new_filter_cmd(
    narration: Path = typer.Option(..., help="Ego4D narration.json path"),
    taxonomy: str = typer.Option(
        "C",
        help="Target taxonomy A/B/C/D, or 'all' for one JSONL with per-clip taxonomy",
    ),
    out: Path = typer.Option(..., help="Output JSONL path for reclassified candidates"),
    require_shared_noun: bool = typer.Option(
        True, help="Drop candidates whose narration has no shared-table noun"
    ),
    limit: int = typer.Option(0, help="Hard cap on output candidates (0 = no cap)"),
    videos_root: Path = typer.Option(None, help="Directory with {video_id}.mp4 files for frame extraction"),
    frame_backend: str = typer.Option("ffmpeg", help="Frame extraction backend: ffmpeg or imageio"),
    frames_out_dir: Path = typer.Option(None, help="Directory to save extracted frames (same layout as extract-frames)"),
    florence_describe: bool = typer.Option(
        False,
        help="After taxonomy pass, run Florence-2 <OD> on sparse frames and merge object labels into nouns",
    ),
    florence_model: str = typer.Option("microsoft/Florence-2-base", help="HuggingFace Florence-2 model id"),
    florence_device: str = typer.Option("", help="Device for Florence-2 (empty = cuda if available else cpu)"),
    auto_download: bool = typer.Option(False, help="Auto-download missing videos via ego4d CLI when --videos-root is set"),
    llm_parse: bool = typer.Option(
        False,
        help="Use spaCy candidates + OpenAI to pick object/verb and contextual vs baseline",
    ),
    openai_model: str = typer.Option(
        "gpt-4o-mini",
        help="OpenAI model for --llm-parse (also EGOOWN_NARRATION_OPENAI_MODEL)",
    ),
    llm_batch_size: int = typer.Option(
        20,
        help="Narrations per OpenAI request when --llm-parse is enabled",
    ),
    no_table_object_prefilter: bool = typer.Option(
        False,
        help="Do not require 'table' and '#O' in narration (default: require both)",
    ),
):
    """Filter candidates from Ego4D narration.json (narration_pass narrations)."""
    if taxonomy.lower() in ("all", "*", "any"):
        tax = None
    else:
        try:
            tax = Taxonomy(taxonomy.upper())
        except ValueError:
            raise typer.BadParameter(
                f"taxonomy must be one of A/B/C/D or 'all', got {taxonomy!r}"
            )

    n = pipeline.stage_new_filter(
        narration_path=narration,
        taxonomy=tax,
        out_path=out,
        require_shared_noun=require_shared_noun,
        limit=limit if limit > 0 else None,
        videos_root=videos_root,
        frame_backend=frame_backend,
        frames_out_dir=frames_out_dir,
        florence_describe=florence_describe,
        florence_model=florence_model,
        florence_device=(florence_device or None),
        auto_download=auto_download,
        use_llm_parse=llm_parse,
        openai_model=openai_model,
        llm_batch_size=llm_batch_size if llm_parse else None,
        require_table_object_markers=not no_table_object_prefilter,
    )
    if tax is None:
        _CONSOLE.print(f"[green]Wrote {n} candidates[/green] (split by taxonomy):")
        for t, path in pipeline.paths_per_taxonomy_out(out).items():
            _CONSOLE.print(f"  {t.value} → {path}")
    else:
        _CONSOLE.print(f"[green]Wrote {n} candidates[/green] → {out}")


# ---------- downsample-candidates ----------


@app.command("downsample-candidates")
def downsample_candidates_cmd(
    inputs: list[Path] = typer.Argument(
        ...,
        help="One or more candidates JSONL paths (e.g. outputs/candidates_narration_A.jsonl)",
    ),
    window_sec: float = typer.Option(
        60.0,
        help="Min seconds between kept clips in the same video_id",
    ),
    out: Path = typer.Option(
        None,
        help="Output path (single input only). Default: <stem>_ds60.jsonl next to input",
    ),
    suffix: str = typer.Option(
        "",
        help="Suffix before .jsonl (default: _ds<window>, e.g. _ds60)",
    ),
    in_place: bool = typer.Option(
        False,
        "--in-place",
        help="Overwrite input files (destructive; not recommended)",
    ),
):
    """Drop nearby clips in the same video, keeping one per minute window."""
    if out is not None and len(inputs) != 1:
        raise typer.BadParameter("--out requires exactly one input JSONL")
    if in_place and out is not None:
        raise typer.BadParameter("Use either --in-place or --out, not both")
    total_before = 0
    total_after = 0
    for path in inputs:
        if not path.exists():
            raise typer.BadParameter(f"Input not found: {path}")
        before, after, dest = pipeline.downsample_candidates_jsonl(
            path,
            window_sec=window_sec,
            out_path=out if len(inputs) == 1 else None,
            in_place=in_place,
            suffix=suffix or None,
        )
        total_before += before
        total_after += after
        _CONSOLE.print(
            f"[green]{path.name}[/green]: {before} → {after} "
            f"({before - after} removed, window={window_sec}s) → {dest}"
        )
    _CONSOLE.print(
        f"[green]Total[/green]: {total_before} → {total_after} "
        f"({total_before - total_after} removed)"
    )


# ---------- dedupe-candidates ----------


@app.command("dedupe-candidates")
def dedupe_candidates_cmd(
    directory: Path = typer.Option(
        Path("outputs"),
        help="Directory with candidates_narration_{A,B,C,D}.jsonl split files",
    ),
    include_ds60: bool = typer.Option(
        False,
        help="Also dedupe candidates_narration_*_ds60.jsonl as a separate group",
    ),
    in_place: bool = typer.Option(
        True,
        "--in-place/--no-in-place",
        help="Rewrite taxonomy JSONLs in place (default: True)",
    ),
    out_dir: Path = typer.Option(
        None,
        help="Write deduped files here instead of in-place",
    ),
    window_sec: float = typer.Option(
        60.0,
        help="Within each taxonomy file, keep one clip per video_id per this many seconds",
    ),
    no_time_downsample: bool = typer.Option(
        False,
        help="Skip temporal thinning (narration dedupe only)",
    ),
):
    """Keep one clip per narration, then one per video per minute, across taxonomy JSONLs."""
    groups = pipeline.discover_taxonomy_candidate_jsonls(
        directory, include_ds60=include_ds60
    )
    if not groups:
        raise typer.BadParameter(
            f"No candidates_narration_{{A,B,C,D}}.jsonl under {directory}"
        )
    grand_before = 0
    grand_after = 0
    for paths in groups:
        label = paths[0].name.rsplit("_", 2)[0] if paths else "candidates"
        results = pipeline.dedupe_taxonomy_candidate_jsonls(
            paths,
            in_place=in_place and out_dir is None,
            out_dir=out_dir,
            window_sec=window_sec,
            apply_time_downsample=not no_time_downsample,
        )
        group_before = sum(b for b, _ in results.values())
        group_after = sum(a for _, a in results.values())
        grand_before += group_before
        grand_after += group_after
        _CONSOLE.print(f"[bold]{label}[/bold] ({paths[0].parent})")
        for dest, (before, after) in sorted(results.items(), key=lambda x: x[0].name):
            _CONSOLE.print(
                f"  [green]{dest.name}[/green]: {before} → {after} "
                f"({before - after} removed)"
            )
    _CONSOLE.print(
        f"[green]Total[/green]: {grand_before} → {grand_after} "
        f"({grand_before - grand_after} removed)"
    )


# ---------- extract-frames ----------


@app.command("extract-frames")
def extract_frames_cmd(
    candidates: Path = typer.Option(..., help="JSONL from the filter stage"),
    videos_root: Path = typer.Option(..., help="Directory with {video_id}.mp4 files"),
    out: Path = typer.Option(..., help="Output directory for frames"),
    backend: str = typer.Option("ffmpeg", help="ffmpeg or imageio"),
):
    """Extract (t-2, t-1, t) frames per candidate."""
    n = pipeline.stage_extract_frames(candidates, videos_root, out, backend=backend)
    _CONSOLE.print(f"[green]Extracted frames for {n} clips[/green] → {out}")


# ---------- detect ----------


@app.command("detect")
def detect_cmd(
    candidates: Path = typer.Option(..., help="JSONL from the filter stage"),
    frames: Path = typer.Option(None, help="Directory produced by extract-frames (model path)"),
    out: Path = typer.Option(..., help="Output JSONL path for detections"),
    source: str = typer.Option(
        "model",
        help="'model' (run DINO/SAM/etc) or 'native' (use dataset-supplied bboxes — no GPU needed)",
    ),
    annotations: Path = typer.Option(
        None,
        help="Annotation file (required when --source=native, e.g. fho_main.json)",
    ),
    dataset: str = typer.Option(
        None,
        help="Dataset name when --source=native (ego4d-fho or hd-epic)",
    ),
    use_sam: bool = typer.Option(False, help="[model] SAM2 mask refinement"),
    use_ram: bool = typer.Option(False, help="[model] Bottom-up RAM tagging → augment DINO prompt"),
    detect_persons: bool = typer.Option(True, help="[model] Person-only DINO pass + dynamic zones"),
    extract_attrs: bool = typer.Option(False, help="[model] VLM attribute extraction"),
    estimate_depth: bool = typer.Option(False, help="[model] Depth Anything v2 → depth-aware zones"),
    use_sam2_video: bool = typer.Option(False, help="[model] SAM2 video predictor for tracking"),
    remote_vlm: str = typer.Option(
        None,
        help="[model] Replace local RAM/BLIP-2 with a remote VLM provider: 'anthropic' or 'openai'",
    ),
):
    """Run DINO (+ optional extras) OR pull dataset-native bboxes — choose with --source."""
    if source == "native":
        if annotations is None or dataset is None:
            raise typer.BadParameter("--annotations and --dataset are required for --source=native")
        n = pipeline.stage_detect_native(
            candidates_path=candidates,
            annotations_path=annotations,
            dataset=dataset,
            out_path=out,
        )
        _CONSOLE.print(f"[green]Native bbox detect: {n} clips[/green] → {out}")
        return

    if source != "model":
        raise typer.BadParameter(f"--source must be 'model' or 'native', got {source!r}")
    if frames is None:
        raise typer.BadParameter("--frames is required when --source=model")

    n = pipeline.stage_detect(
        candidates,
        frames,
        out,
        use_sam=use_sam,
        use_ram=use_ram,
        detect_persons_too=detect_persons,
        extract_attrs=extract_attrs,
        estimate_depth=estimate_depth,
        use_sam2_video=use_sam2_video,
        remote_vlm_provider=remote_vlm,
    )
    _CONSOLE.print(f"[green]Detected on {n} clips[/green] → {out}")


# ---------- label ----------


@app.command("label")
def label_cmd(
    detections: Path = typer.Option(..., help="JSONL from the detect stage"),
    out: Path = typer.Option(..., help="Output JSONL for final scene records"),
    remote_vlm_judge: str = typer.Option(
        None,
        help="Get a second-opinion ownership label from 'anthropic' or 'openai'",
    ),
    frames_root: Path = typer.Option(
        None,
        help="Required when --remote-vlm-judge is set — root dir for frame_path lookups",
    ),
):
    """Apply the rule cascade (+ optional VLM second opinion) and emit SceneRecords."""
    n = pipeline.stage_label(
        detections,
        out,
        remote_vlm_judge=remote_vlm_judge,
        frames_root=frames_root,
    )
    _CONSOLE.print(f"[green]Labeled {n} scenes[/green] → {out}")


# ---------- serve (collaborative annotator) ----------


@app.command("convert-to-server")
def convert_to_server_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="labels.jsonl produced by the one-pass-labels stage",
    ),
    out: Path = typer.Option(
        None,
        help="Output SceneRecord JSONL. Defaults to scene_records.jsonl next to --input",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
):
    """Convert one-pass-labels JSONL to SceneRecord format for the review server.

    Run this once after the one-pass-labels stage, then point 'ego-label serve'
    at the output file.

    Example workflow:

    \b
        ego-label convert-to-server --input outputs/egolife/labels.jsonl
        ego-label serve --scenes outputs/egolife/scene_records.jsonl \\
                        --frames-root outputs/egolife/one_pass_sparse_frames
    """
    from egoownership.catv_io import count_jsonl, iter_jsonl
    from egoownership.catv_pipeline import labels_row_to_scene_record

    if not input_jsonl.exists():
        raise typer.BadParameter(f"File not found: {input_jsonl}")

    resolved_out = out or input_jsonl.with_name("scene_records.jsonl")
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    records = iter_jsonl(input_jsonl, skip_bad=True)
    if progress:
        from tqdm.auto import tqdm
        records = tqdm(records, total=count_jsonl(input_jsonl), unit="row", desc="converting")

    n_ok = n_err = 0
    with resolved_out.open("w", encoding="utf-8") as fh:
        for row in records:
            try:
                scene_record = labels_row_to_scene_record(row)
                fh.write(scene_record.model_dump_json() + "\n")
                n_ok += 1
            except Exception as exc:
                n_err += 1
                msg = f"[warn] skip {row.get('id', '?')}: {exc}"
                if progress:
                    from tqdm import tqdm as _tqdm
                    _tqdm.write(msg)
                else:
                    print(msg, flush=True)

    _CONSOLE.print(f"[green]Wrote {n_ok} SceneRecords[/green] → {resolved_out}")
    if n_err:
        _CONSOLE.print(f"[yellow]Skipped {n_err} rows (conversion errors)[/yellow]")


@app.command("serve")
def serve_cmd(
    scenes: Path = typer.Option(..., help="Path to scene_records.jsonl"),
    frames_root: Path = typer.Option(..., help="Root directory frame paths are relative to"),
    videos_root: Path = typer.Option(
        None, help="Optional: directory with {video_id}.mp4 files for clip playback"
    ),
    host: str = typer.Option("0.0.0.0", help="Bind address (use 0.0.0.0 for LAN sharing)"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)"),
):
    """Launch the FastAPI collaborative annotation server.

    Open http://HOST:PORT in a browser. Multiple annotators can connect at the
    same time; edits are written through to ``scenes`` JSONL with file-locking.
    Pass ``--videos-root`` to enable inline clip playback in the UI.
    """
    import uvicorn
    from egoownership.server import create_app

    fastapi_app = create_app(
        scenes_path=scenes,
        frames_root=frames_root,
        videos_root=videos_root,
    )
    if reload:
        import os
        os.environ["EGOOWN_SCENES_PATH"] = str(scenes)
        os.environ["EGOOWN_FRAMES_ROOT"] = str(frames_root)
        if videos_root is not None:
            os.environ["EGOOWN_VIDEOS_ROOT"] = str(videos_root)
        uvicorn.run("egoownership.server.entry:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()
