"""Typer-based CLI. Installed as ``egoown``.

Pipeline stages (each step is resumable via JSONL):
  extract-bbox → object-caption → auto-label → vlm-crosscheck
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        return None

load_dotenv()

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)

_CONSOLE = Console()


# ---------- extract-bbox ----------


@app.command("extract-bbox")
def extract_bbox_cmd(
    dataset: str = typer.Option(
        ...,
        help="Dataset adapter: egolife, ego4d, or generic. "
        "'generic' accepts any JSONL whose records already contain video_path, clip_id, "
        "video_id, start_sec, end_sec, and nouns fields. "
        "New adapters can be registered via register_dataset_adapter().",
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
        help="Cache directory for sampled frames and bbox visualizations. Defaults to outputs/{dataset}/reference_frames",
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
    sam_score_threshold: float = typer.Option(0.7, help="SAM-3 detection confidence threshold (drop candidate boxes scoring below this)"),
    resume: bool = typer.Option(
        True,
        "--resume/--overwrite",
        help="Append to existing bbox output and skip IDs already written; --overwrite starts fresh",
    ),
    limit: int = typer.Option(0, help="Hard cap on filtered caption cues (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar"),
):
    """Extract target-object boxes before object captioning (multi-dataset)."""
    from egoownership.datasets.adapters import normalize_dataset_id
    from egoownership.labeling_pipeline import write_caption_bboxes
    from egoownership.detection.object_segmentation import Sam2ObjectConfig, Sam2ObjectExtractor

    ds = normalize_dataset_id(dataset)
    out_dir = Path("outputs") / ds
    resolved_out = out or (out_dir / "bbox_objects.jsonl")
    resolved_frames_dir = frames_dir or (out_dir / "reference_frames")

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
            score_threshold=sam_score_threshold,
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


# ---------- object-caption ----------


@app.command("object-caption")
def object_caption_cmd(
    input_jsonl: Path = typer.Option(..., "--input", help="BBox JSONL produced by extract-bbox"),
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
    mask_python: str = typer.Option(
        None,
        "--mask-python",
        help="Python interpreter for the SAM-2 batch step. Defaults to sys.executable (current env).",
    ),
    visualization_dir: Path = typer.Option(
        None,
        "--visualization-dir",
        help="Copy SAM-2 masked videos here for review. Omit to skip.",
    ),
    work_dir: Path = typer.Option(
        None,
        help="Directory for intermediate SAM-2/Qwen files. Defaults to <out-dir>/labeling_work.",
    ),
    whole_video: bool = typer.Option(True, "--whole-video/--clip-only", help="Track the full source clip"),
    resume: bool = typer.Option(True, "--resume/--overwrite"),
    limit: int = typer.Option(0, help="Hard cap on rows (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
):
    """Caption extracted object boxes with SAM-2 masking and one in-process VLM run."""
    from egoownership.labeling_pipeline import write_object_captions

    ds = dataset or input_jsonl.parent.name
    resolved_out = out or (Path("outputs") / ds / "captions.jsonl")

    resolved_devices = [d.strip() for d in devices.split(",")] if devices else None

    n = write_object_captions(
        input_jsonl,
        resolved_out,
        mask_model_path=mask_model_path,
        caption_device=caption_device,
        devices=resolved_devices,
        fps=caption_fps,
        whole_video=whole_video,
        max_frames=caption_max_frames,
        max_side=caption_max_side,
        captioner_backend=captioner_backend,
        caption_model_path=caption_model_path,
        mask_python=mask_python,
        visualization_root=visualization_dir,
        work_dir=work_dir,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} object captions[/green] → {resolved_out}")


# ---------- auto-label ----------


@app.command("auto-label")
def auto_label_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="Object-description JSONL from the object-caption stage",
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
        help="Cache directory for sampled t-2/t-1/t frames. Defaults to outputs/{dataset}/auto_label_sparse_frames",
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
    decision_model_id: str = typer.Option(
        "Qwen/Qwen3-4B",
        help="Text LLM checkpoint used when --decision-backend=llm",
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
    """Build ownership labels from object descriptions (multi-dataset)."""
    from egoownership.labeling_pipeline import write_one_pass_labels

    decision_fn = None
    if decision_backend == "llm":
        from egoownership.evidence_labeling import LLMTaxonomyDecider
        decision_fn = LLMTaxonomyDecider(model_id=decision_model_id, device=decision_device)
    elif decision_backend != "rules":
        raise typer.BadParameter("--decision-backend must be 'rules' or 'llm'")

    ds = dataset or input_jsonl.parent.name
    resolved_out = out or (Path("outputs") / ds / "labels.jsonl")
    resolved_frames_dir = frames_dir or (Path("outputs") / ds / "auto_label_sparse_frames")
    progress_desc = f"{ds} auto-label"
    effective_detect_persons = detect_persons or bool(sam2_tracking_model_id)

    n = write_one_pass_labels(
        input_jsonl,
        resolved_out,
        frames_dir=resolved_frames_dir,
        detect_persons=effective_detect_persons,
        decision_fn=decision_fn,
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
    _CONSOLE.print(f"[green]Wrote {n} auto-label rows[/green] → {resolved_out}")


# ---------- vlm-crosscheck ----------


@app.command("vlm-crosscheck")
def vlm_crosscheck_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="labels.jsonl produced by auto-label",
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
    videos_root: Path = typer.Option(
        None,
        help="Directory with {video_id}.mp4 raw videos, used to reconstruct a record's "
             "sparse frames (via video_id/source_video_start_sec/frame_times_sec, already "
             "in labels.jsonl) whenever frames_root doesn't have the pre-extracted JPEGs — "
             "for a metadata-only dataset shipped without frame crops (e.g. one derived "
             "from a licensed source like Ego4D whose terms don't permit redistributing them).",
    ),
    only_agreed_in: Path = typer.Option(
        None,
        help="Path to an existing crosscheck JSONL (e.g. from a prior judge). Records "
             "already known to disagree there (majority_agrees=false) are carried through "
             "into the output unchanged instead of being re-checked here — a disagreement "
             "already flags the record for review regardless of a second judge's opinion, "
             "so re-checking it just costs more API calls. The output still has one row "
             "per input record, same total as an unfiltered run.",
    ),
    judges: list[str] = typer.Option(
        [],
        "--judge",
        help="Judge spec: 'qwen:<model_id>', 'anthropic:<model_id>', 'openai:<model_id>', or 'gemini:<model_id>'. "
             "Repeat for multiple judges.",
    ),
    qwen_device: str = typer.Option("auto", help="Device for local Qwen VL judges"),
    qwen_dtype: str = typer.Option("auto", help="Dtype for Qwen VL: auto, float16, bfloat16"),
    qwen_max_tokens: int = typer.Option(256, help="Max new tokens for Qwen VL response"),
    remote_max_tokens: int = typer.Option(512, help="Max tokens for Anthropic/OpenAI/Gemini response"),
    resume: bool = typer.Option(True, "--resume/--overwrite"),
    limit: int = typer.Option(0, help="Hard cap on records processed (0 = no cap)"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
):
    """Cross-check ownership labels with multiple VLM judges.

    Each judge independently predicts MINE / PERSON_k / SHARED / AMBIGUOUS from
    the three sparse frames alone (frame t always gets the verified reference
    bbox; t-1/t-2 get their own genuinely-tracked box when --sam2-track
    produced one, otherwise shown unboxed) — no narration,
    no object visual description (that's generated by our own captioning
    stage, so including it would leak our own pipeline's reasoning), and no
    access to the pipeline's own prediction or evidence. Each judge answers
    with a label plus one sentence per remaining evidence aspect (object type,
    zone, relation/contact, temporal context) using the same field names as
    scene_records.jsonl's auto_key_evidence (object_type_evidence,
    zone_evidence, relation_graph_evidence, context_change_evidence), so a
    judge's reasoning can be compared side-by-side with the pipeline's own,
    aspect by aspect.
    Agreement with auto_ground_truth is recorded per judge, along with a
    majority-vote label.

    Examples:

    \\b
        egoown vlm-crosscheck \\
            --input outputs/egolife/labels.jsonl \\
            --judge anthropic:claude-sonnet-4-6 \\
            --judge openai:gpt-4o \\
            --judge gemini:gemini-2.0-flash
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
        backend, model_id = spec.split(":", 1) if ":" in spec else (spec, spec)
        backend = backend.lower()
        if backend == "qwen":
            judge_objs.append(QwenOwnershipJudge(QwenOwnershipJudgeConfig(
                model_id=model_id, device=qwen_device, dtype=qwen_dtype, max_new_tokens=qwen_max_tokens,
            )))
        elif backend == "anthropic":
            judge_objs.append(AnthropicOwnershipJudge(AnthropicOwnershipJudgeConfig(
                model_id=model_id, max_tokens=remote_max_tokens,
            )))
        elif backend == "openai":
            judge_objs.append(OpenAIOwnershipJudge(OpenAIOwnershipJudgeConfig(
                model_id=model_id, max_tokens=remote_max_tokens,
            )))
        elif backend == "gemini":
            judge_objs.append(GeminiOwnershipJudge(GeminiOwnershipJudgeConfig(
                model_id=model_id, max_tokens=remote_max_tokens,
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
        videos_root=videos_root,
        only_agreed_in=only_agreed_in,
        limit=limit if limit > 0 else None,
        resume=resume,
        show_progress=progress,
    )
    _CONSOLE.print(f"[green]Wrote {n} cross-check rows[/green] → {resolved_out}")


@app.command("vlm-crosscheck-merge")
def vlm_crosscheck_merge_cmd(
    input_jsonl: list[str] = typer.Option(
        ...,
        "--input",
        help="Crosscheck JSONL(s) to merge, e.g. from separate per-judge runs "
             "(--only-agreed-in makes it cheap to run additional judges against "
             "just the agreed subset). Comma-separated or repeatable.",
    ),
    out: Path = typer.Option(..., help="Merged output JSONL"),
):
    """Merge two or more crosscheck JSONL files into one, unioning judges per id.

    Each id's judges dicts are combined and agreement_count/ratio/majority_label/
    majority_agrees are recomputed over the full combined judge set. Useful after
    running --only-agreed-in with a second judge: this produces one file with
    every judge's opinion per record, for the review server's --crosscheck.

    \\b
        egoown vlm-crosscheck-merge \\
            --input outputs/egolife/claude_crosscheck.jsonl,outputs/egolife/gpt_crosscheck.jsonl \\
            --out outputs/egolife/merged_crosscheck.jsonl
    """
    from egoownership.vlm_crosscheck import merge_crosscheck_jsonl

    paths = [Path(part.strip()) for raw in input_jsonl for part in raw.split(",") if part.strip()]
    for p in paths:
        if not p.exists():
            raise typer.BadParameter(f"File not found: {p}")
    n = merge_crosscheck_jsonl(paths, out)
    _CONSOLE.print(f"[green]Wrote {n} merged cross-check rows[/green] → {out}")


# ---------- convert-to-server ----------


@app.command("convert-to-server")
def convert_to_server_cmd(
    input_jsonl: Path = typer.Option(
        ...,
        "--input",
        help="labels.jsonl produced by the auto-label stage",
    ),
    out: Path = typer.Option(
        None,
        help="Output SceneRecord JSONL. Defaults to scene_records.jsonl next to --input",
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
):
    """Convert auto-label JSONL to SceneRecord format for the review server."""
    from egoownership.labeling_pipeline import count_jsonl, iter_jsonl, labels_row_to_scene_record

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


# ---------- serve ----------


@app.command("serve")
def serve_cmd(
    input_jsonl: list[str] = typer.Option(
        None,
        "--input",
        help="labels.jsonl from auto-label. Pass multiple paths — comma-separated "
             "(--input a.jsonl,b.jsonl) or by repeating the flag (--input a.jsonl --input b.jsonl) "
             "— to serve multiple datasets in one session: they're merged into one "
             "scene_records.jsonl, filterable by dataset in the UI. Auto-converts, then serves.",
    ),
    scenes: Path = typer.Option(
        None,
        help="Path to scene_records.jsonl (SceneRecord format). Use --input instead to serve labels.jsonl directly.",
    ),
    scenes_out: Path = typer.Option(
        None,
        "--scenes-out",
        help="Where to write the merged scene_records.jsonl. Defaults to alongside the input file "
             "(single --input) or outputs/scene_records.jsonl (multiple --input).",
    ),
    crosscheck: list[str] = typer.Option(
        None,
        "--crosscheck",
        help="vlm-crosscheck output JSONL(s) to merge in as independent judge second opinions, "
             "joined by matching id — surfaces alongside the auto evidence in the UI, never "
             "replacing it. Comma-separated or repeatable, same as --input. Only used with --input.",
    ),
    frames_root: Path = typer.Option(
        None,
        help="Root directory frame paths are relative to. Defaults to CWD when --input is used.",
    ),
    videos_root: Path = typer.Option(
        None, help="Optional: directory with {video_id}.mp4 files for clip playback"
    ),
    host: str = typer.Option("0.0.0.0", help="Bind address (use 0.0.0.0 for LAN sharing)"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev only)"),
):
    """Launch the FastAPI collaborative annotation server.

    Two usage patterns:

    \b
        # Direct from auto-label output (recommended), one or more datasets:
        egoown serve --input outputs/egolife/labels.jsonl
        egoown serve --input outputs/egolife/labels.jsonl,outputs/ego4d/labels.jsonl
        egoown serve --input outputs/egolife/labels.jsonl --input outputs/ego4d/labels.jsonl

        # With vlm-crosscheck judgements merged in for side-by-side comparison:
        egoown serve --input outputs/egolife/labels.jsonl \\
                     --crosscheck outputs/egolife/claude_crosscheck.jsonl

        # From pre-converted SceneRecord JSONL:
        egoown serve --scenes outputs/egolife/scene_records.jsonl \\
                     --frames-root outputs/egolife/auto_label_sparse_frames

    Re-running with --input always regenerates the auto fields (evidence,
    vlm_judgements, frames, ...) fresh from labels.jsonl/crosscheck — but if a
    scene_records.jsonl already exists at the target path, any record with
    edit history or a non-draft review_status keeps its scene_label/taxonomy/
    review_status/notes/edits as-is rather than being reset to a blank draft.
    """
    import uvicorn
    from egoownership.server import create_app

    # Accept both --input a --input b and --input a,b (and a mix of the two).
    inputs: list[Path] = [Path(part.strip()) for raw in (input_jsonl or []) for part in raw.split(",") if part.strip()]
    crosschecks: list[Path] = [Path(part.strip()) for raw in (crosscheck or []) for part in raw.split(",") if part.strip()]

    if not inputs and scenes is None:
        raise typer.BadParameter("Provide either --input labels.jsonl (repeatable/comma-separated) or --scenes scene_records.jsonl")
    if crosschecks and not inputs:
        raise typer.BadParameter("--crosscheck requires --input (it merges in during labels.jsonl → scene_records.jsonl conversion)")

    if inputs:
        for p in inputs:
            if not p.exists():
                raise typer.BadParameter(f"File not found: {p}")
        from egoownership.labeling_pipeline import (
            apply_preserved_review_state,
            count_jsonl,
            iter_jsonl,
            labels_row_to_scene_record,
            load_preserved_review_state,
        )
        from tqdm.auto import tqdm

        crosscheck_by_id: dict[str, dict] = {}
        for p in crosschecks:
            if not p.exists():
                raise typer.BadParameter(f"File not found: {p}")
            for cc_row in iter_jsonl(p, skip_bad=True):
                rid = cc_row.get("id")
                if rid:
                    crosscheck_by_id[rid] = cc_row
        if crosschecks:
            _CONSOLE.print(f"Loaded {len(crosscheck_by_id)} crosscheck judgements from {', '.join(p.name for p in crosschecks)}")

        if scenes_out is not None:
            scene_records_path = scenes_out
        elif len(inputs) == 1:
            scene_records_path = inputs[0].with_name("scene_records.jsonl")
        else:
            import os

            common = Path(os.path.commonpath([str(p.resolve().parent) for p in inputs]))
            scene_records_path = common / "scene_records.jsonl"

        # Re-converting from --input always regenerates auto fields (evidence,
        # vlm_judgements, frames, ...) fresh — that's the point, e.g. to pick up
        # a labels.jsonl bugfix. But it must not silently discard review
        # progress recorded in an existing scene_records.jsonl.
        preserved_by_id = load_preserved_review_state(scene_records_path)
        if preserved_by_id:
            _CONSOLE.print(f"Preserving review state for {len(preserved_by_id)} already-reviewed record(s) in existing {scene_records_path.name}")

        _CONSOLE.print(
            f"Converting {', '.join(p.name for p in inputs)} → {scene_records_path} …"
        )
        n_ok = n_err = n_crosscheck_matched = n_preserved = 0
        with scene_records_path.open("w", encoding="utf-8") as fh:
            for src in inputs:
                for row in tqdm(iter_jsonl(src, skip_bad=True), total=count_jsonl(src), unit="row", desc=src.name):
                    try:
                        cc_row = crosscheck_by_id.get(row.get("id")) if crosscheck_by_id else None
                        if cc_row is not None:
                            n_crosscheck_matched += 1
                        fresh = labels_row_to_scene_record(row, crosscheck=cc_row)
                        merged = apply_preserved_review_state(fresh, preserved_by_id)
                        if merged is not fresh:
                            n_preserved += 1
                        fh.write(merged.model_dump_json() + "\n")
                        n_ok += 1
                    except Exception as exc:
                        n_err += 1
                        _CONSOLE.print(f"[yellow]skip {row.get('id','?')}: {exc}[/yellow]")
        _CONSOLE.print(f"[green]{n_ok} records converted[/green]" + (f", {n_err} skipped" if n_err else ""))
        if n_preserved:
            _CONSOLE.print(f"[cyan]{n_preserved} record(s) kept their prior review_status/notes/edits[/cyan]")
        if crosscheck_by_id:
            unmatched = len(crosscheck_by_id) - n_crosscheck_matched
            _CONSOLE.print(
                f"[cyan]{n_crosscheck_matched}/{n_ok} scene records matched a crosscheck judgement[/cyan]"
                + (f" [yellow]({unmatched} crosscheck rows had no matching id — check the two files agree on id format)[/yellow]" if unmatched > 0 else "")
            )
        scenes = scene_records_path
        if frames_root is None:
            frames_root = Path.cwd()

    if frames_root is None:
        raise typer.BadParameter("--frames-root is required when using --scenes")

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
