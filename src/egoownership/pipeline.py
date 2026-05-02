"""High-level stage functions glue the dataset / filter / frames / detect
stages together. The CLI is a thin wrapper over these.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from egoownership.datasets import (
    iter_egolife_candidates,
    iter_epic_candidates,
    iter_fho_candidates,
    iter_hd_epic_candidates,
)
from egoownership.filters import filter_candidates
from egoownership.schema import (
    ClipCandidate,
    FrameDetections,
    SceneRecord,
    Taxonomy,
)


_LOADER_MAP = {
    "ego4d-fho": iter_fho_candidates,
    "epic": iter_epic_candidates,
    "hd-epic": iter_hd_epic_candidates,
    "egolife": iter_egolife_candidates,
}


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates(path: Path) -> Iterator[ClipCandidate]:
    for d in read_jsonl(path):
        yield ClipCandidate.model_validate(d)


def stage_filter(
    dataset: str,
    annotations_path: Path,
    taxonomy: Taxonomy,
    out_path: Path,
    *,
    require_shared_noun: bool = True,
    limit: int | None = None,
) -> int:
    if dataset not in _LOADER_MAP:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose one of {list(_LOADER_MAP)}")

    candidates = _LOADER_MAP[dataset](annotations_path)
    filtered = filter_candidates(
        candidates,
        taxonomy,
        require_shared_noun=require_shared_noun,
        limit=limit,
    )
    return write_jsonl(out_path, (c.model_dump(mode="json") for c in filtered))


def stage_extract_frames(
    candidates_path: Path,
    videos_root: Path,
    out_dir: Path,
    *,
    backend: str = "ffmpeg",
) -> int:
    from egoownership.frames import extract_sparse_frames

    count = 0
    for cand in load_candidates(candidates_path):
        if not cand.video_id:
            continue
        video_path = videos_root / f"{cand.video_id}.mp4"
        if not video_path.exists():
            for ext in (".MP4", ".mkv", ".webm"):
                alt = video_path.with_suffix(ext)
                if alt.exists():
                    video_path = alt
                    break
            else:
                continue
        try:
            extract_sparse_frames(cand, video_path, out_dir, backend=backend)
            count += 1
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {cand.clip_id}: frame extraction failed: {e}")
    return count


def _frame_path_for(cand: ClipCandidate, frames_root: Path, tag: str) -> Path:
    safe_clip = cand.clip_id.replace("/", "_").replace(":", "_")
    return frames_root / cand.dataset / (cand.video_id or "_") / f"{safe_clip}__{tag}.jpg"


def stage_detect(
    candidates_path: Path,
    frames_root: Path,
    out_path: Path,
    *,
    use_sam: bool = False,
    use_ram: bool = False,
    detect_persons_too: bool = True,
    extract_attrs: bool = False,
    estimate_depth: bool = False,
    use_sam2_video: bool = False,
) -> int:
    """Run the *visual evidence* stage.

    Order: Grounding DINO (top-down via clip nouns ± RAM) → optional SAM mask
    refinement → person detector → instance tracking → optional depth →
    optional VLM attributes → relation graph build.
    """

    from egoownership.detection.grounding_dino import (
        DinoConfig,
        build_prompt,
        detect_objects,
    )
    from egoownership.detection.persons import (
        assign_person_ids_across_frames,
        detect_persons,
    )
    from egoownership.detection.relations import build_scene_graph
    from egoownership.detection.tracking import assign_instance_ids, assign_with_sam2_video
    from egoownership.detection.zones import person_relative_zones, static_zones
    from egoownership.config import load_config

    cfg = load_config()
    dino_cfg = DinoConfig()

    records: list[dict] = []
    count = 0
    for cand in load_candidates(candidates_path):
        # 1. Build prompt: clip nouns optionally augmented by RAM.
        nouns = list(cand.nouns)
        prompt_nouns = nouns
        times = [
            ("t-2", cand.t_minus_2_sec),
            ("t-1", cand.t_minus_1_sec),
            ("t", cand.t_sec),
        ]

        per_frame_paths: list[Path] = []
        frames: list[FrameDetections] = []
        for tag, t in times:
            fp = _frame_path_for(cand, frames_root, tag)
            if not fp.exists():
                continue
            per_frame_paths.append(fp)
            if use_ram:
                from egoownership.detection.ram import extract_tags, merge_with_clip_nouns
                ram_tags = extract_tags(fp)
                prompt_nouns = merge_with_clip_nouns(nouns, ram_tags)
            prompt = build_prompt(prompt_nouns)
            detections = detect_objects(fp, prompt, dino_cfg)
            if use_sam and detections:
                from egoownership.detection.sam import refine_boxes
                detections = refine_boxes(fp, detections)

            persons = detect_persons(fp) if detect_persons_too else []

            frames.append(
                FrameDetections(
                    tag=tag,
                    frame_path=str(fp.relative_to(frames_root)),
                    timestamp_sec=t,
                    objects=detections,
                    persons=persons,
                )
            )

        if not frames:
            continue

        # 2. Person identity propagation.
        if detect_persons_too:
            propagated = assign_person_ids_across_frames([f.persons for f in frames])
            for fd, plist in zip(frames, propagated):
                fd.persons = plist

        # 3. Object instance tracking.
        if use_sam2_video:
            frames = assign_with_sam2_video(frames, [str(p) for p in per_frame_paths])
        else:
            frames = assign_instance_ids(frames)

        # 4. Optional depth + zones.
        if estimate_depth:
            from egoownership.detection.depth import (
                annotate_object_depth,
                estimate_wearer_depth_band,
            )
            for fd, fp in zip(frames, per_frame_paths):
                fd.objects = annotate_object_depth(fp, fd.objects)
            depth_bands = [estimate_wearer_depth_band(fd.objects) for fd in frames]
        else:
            depth_bands = [None] * len(frames)

        # 5. Zones.
        for fd, db in zip(frames, depth_bands):
            if fd.persons:
                fd.zones = person_relative_zones(fd.persons, cfg.zones)
                if db is not None:
                    fd.zones = fd.zones.model_copy(update={"derivation": fd.zones.derivation + "+depth"})
            else:
                fd.zones = static_zones(cfg.zones)

        # 6. Optional VLM attributes.
        if extract_attrs:
            from egoownership.detection.attributes import annotate_frame_objects
            for fd, fp in zip(frames, per_frame_paths):
                fd.objects = annotate_frame_objects(fp, fd.objects)

        # 7. Scene graph relations.
        frames = build_scene_graph(frames)

        records.append(
            {
                "clip": cand.model_dump(mode="json"),
                "frames": [fd.model_dump(mode="json") for fd in frames],
                "depth_bands": depth_bands,
            }
        )
        count += 1

    write_jsonl(out_path, records)
    return count


def stage_label(detections_path: Path, out_path: Path) -> int:
    from egoownership.detection.ownership import assign_ownership, build_scene_record
    from egoownership.detection.tracking import assign_instance_ids

    records: list[dict] = []
    for d in read_jsonl(detections_path):
        clip = ClipCandidate.model_validate(d["clip"])
        frames = [FrameDetections.model_validate(fd) for fd in d["frames"]]
        # If the upstream stage didn't already track instances, do it now —
        # build_scene_record needs instance_ids to derive a scene label.
        needs_tracking = any(
            o.instance_id is None for fd in frames for o in fd.objects
        )
        if needs_tracking:
            frames = assign_instance_ids(frames)
        depth_bands = d.get("depth_bands")
        frames_with_own = assign_ownership(frames, wearer_depth_bands=depth_bands)
        scene: SceneRecord = build_scene_record(clip, frames_with_own)
        records.append(scene.model_dump(mode="json"))
    return write_jsonl(out_path, records)
