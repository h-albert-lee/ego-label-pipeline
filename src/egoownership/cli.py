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

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
download_app = typer.Typer(no_args_is_help=True, help="Dataset download helpers")
app.add_typer(download_app, name="download")

_CONSOLE = Console()


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
