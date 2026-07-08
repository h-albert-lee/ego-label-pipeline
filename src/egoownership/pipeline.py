"""High-level stage functions glue the dataset / filter / frames / detect
stages together. The CLI is a thin wrapper over these.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from egoownership.datasets import (
    iter_egolife_candidates,
    iter_epic_candidates,
    iter_fho_candidates,
    iter_hd_epic_candidates,
)
from egoownership.filters import filter_candidates, new_filter
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


class IncrementalJsonlWriter:
    """Append candidates to JSONL file(s), flushing after each LLM batch.

    In ``--taxonomy all`` mode, A/C/D files keep the LLM-assigned bucket while
  ``*_B.jsonl`` receives every candidate (stamped ``taxonomy=B``) because any
    clip may be annotated as Conflict downstream.
    """

    def __init__(self, out_path: Path, taxonomy: Taxonomy | None) -> None:
        self.taxonomy = taxonomy
        self._mirror_all_to_conflict = taxonomy is None
        if taxonomy is None:
            self._paths = paths_per_taxonomy_out(out_path)
            self._handles = {
                tax: path.open("w", encoding="utf-8")
                for tax, path in self._paths.items()
            }
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._paths = {taxonomy: out_path}
            self._handles = {taxonomy: out_path.open("w", encoding="utf-8")}

    def write_candidates(self, candidates: list[ClipCandidate]) -> int:
        if not candidates:
            return 0
        by_tax: dict[Taxonomy, list[dict]] = {tax: [] for tax in Taxonomy}
        conflict_records: list[dict] = []
        for cand in candidates:
            by_tax[cand.taxonomy].append(cand.model_dump(mode="json"))
            if self._mirror_all_to_conflict:
                conflict_records.append(
                    cand.model_copy(update={"taxonomy": Taxonomy.CONFLICT}).model_dump(
                        mode="json"
                    )
                )
        for tax, records in by_tax.items():
            if self._mirror_all_to_conflict and tax is Taxonomy.CONFLICT:
                records = conflict_records
            elif not records:
                continue
            handle = self._handles[tax]
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
        return len(candidates)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()


def paths_per_taxonomy_out(out_path: Path) -> dict[Taxonomy, Path]:
    """``--taxonomy all`` → ``{stem}_A.jsonl``, ``{stem}_B.jsonl``, …"""
    if out_path.suffix:
        stem, suffix = out_path.stem, out_path.suffix
    else:
        stem, suffix = out_path.name, ".jsonl"
    parent = out_path.parent
    return {tax: parent / f"{stem}_{tax.value}{suffix}" for tax in Taxonomy}


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def video_id_from_record(record: dict) -> str:
    """Resolve grouping key for temporal downsampling."""
    vid = record.get("video_id")
    if isinstance(vid, str) and vid.strip():
        return vid.strip()
    clip_id = record.get("clip_id")
    if isinstance(clip_id, str) and ":" in clip_id:
        return clip_id.split(":", 1)[0]
    return ""


def candidate_timestamp_sec(record: dict, *, time_key: str = "t_sec") -> float:
    raw = record.get(time_key)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def downsample_by_video_time(
    records: list[dict],
    *,
    window_sec: float = 60.0,
    time_key: str = "t_sec",
) -> list[dict]:
    """Keep at most one clip per ``video_id`` when timestamps are within ``window_sec``.

    Clips are sorted by ``time_key`` within each video, then filtered with a greedy
    rule: keep the earliest, drop later clips until ``t - last_kept >= window_sec``.
    Records without a video id are kept unchanged.
    """
    if window_sec <= 0:
        return list(records)

    by_video: dict[str, list[dict]] = {}
    no_video: list[dict] = []
    for rec in records:
        vid = video_id_from_record(rec)
        if vid:
            by_video.setdefault(vid, []).append(rec)
        else:
            no_video.append(rec)

    kept: list[dict] = []
    for vid in sorted(by_video.keys()):
        group = sorted(by_video[vid], key=lambda r: candidate_timestamp_sec(r, time_key=time_key))
        last_kept_t: float | None = None
        for rec in group:
            t = candidate_timestamp_sec(rec, time_key=time_key)
            if last_kept_t is None or (t - last_kept_t) >= window_sec:
                kept.append(rec)
                last_kept_t = t
    kept.extend(no_video)
    return kept


def downsample_output_path(
    path: Path,
    *,
    suffix: str | None = None,
    window_sec: float = 60.0,
) -> Path:
    """Default output path: ``candidates_A.jsonl`` → ``candidates_A_ds60.jsonl``."""
    tag = suffix if suffix is not None else f"_ds{int(window_sec)}"
    if path.suffix:
        return path.with_name(f"{path.stem}{tag}{path.suffix}")
    return path.with_name(f"{path.name}{tag}")


def downsample_candidates_jsonl(
    path: Path,
    *,
    window_sec: float = 60.0,
    out_path: Path | None = None,
    in_place: bool = False,
    suffix: str | None = None,
) -> tuple[int, int, Path]:
    """Downsample one candidates JSONL; returns (count_before, count_after, dest)."""
    records = list(read_jsonl(path))
    before = len(records)
    after_records = downsample_by_video_time(records, window_sec=window_sec)
    after = len(after_records)
    if in_place:
        dest = path
    elif out_path is not None:
        dest = out_path
    else:
        dest = downsample_output_path(path, suffix=suffix, window_sec=window_sec)
    if not in_place and dest.resolve() == path.resolve():
        raise ValueError(
            f"Refusing to overwrite {path}; pass in_place=True or a different --out"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest, after_records)
    return before, after, dest


_TAXONOMY_JSONL_RE = re.compile(r"_([ABCD])(?:_|\.jsonl|$)", re.IGNORECASE)


def narration_dedup_key(record: dict) -> str:
    """Normalized narration text for cross-file deduplication."""
    from egoownership.narration_parse import preprocess_narration

    raw = record.get("narration")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    return preprocess_narration(raw).casefold()


def _record_dedupe_score(record: dict) -> tuple:
    tax = str(record.get("taxonomy", ""))
    llm = str((record.get("source") or {}).get("llm_taxonomy", "") or "")
    matches_llm = bool(llm) and tax == llm
    mirror_b = tax == Taxonomy.CONFLICT.value and llm not in ("", Taxonomy.CONFLICT.value)
    return (
        matches_llm,
        not mirror_b,
        tax != Taxonomy.CONFLICT.value,
        -candidate_timestamp_sec(record),
    )


def pick_record_for_narration(candidates: list[dict]) -> dict:
    """Choose one row when the same narration appears in multiple taxonomy JSONLs."""
    return max(candidates, key=_record_dedupe_score)


def record_output_taxonomy(record: dict) -> Taxonomy:
    """Bucket for rewritten per-taxonomy files (prefer LLM label over stamped B mirror)."""
    from egoownership.narration_parse import parse_llm_taxonomy

    llm = (record.get("source") or {}).get("llm_taxonomy")
    if llm is not None:
        tax = parse_llm_taxonomy(llm)
        if tax is not None:
            return tax
    raw = record.get("taxonomy")
    if raw is None:
        return Taxonomy.AMBIGUOUS
    try:
        return Taxonomy(str(raw))
    except ValueError:
        return Taxonomy.AMBIGUOUS


def dedupe_by_narration(records: list[dict]) -> list[dict]:
    """Within one list, keep a single clip per normalized narration."""
    groups: dict[str, list[dict]] = defaultdict(list)
    no_narration: list[dict] = []
    for rec in records:
        key = narration_dedup_key(rec)
        if key:
            groups[key].append(rec)
        else:
            no_narration.append(rec)
    kept = [pick_record_for_narration(group) for group in groups.values()]
    kept.extend(no_narration)
    return kept


def dedupe_taxonomy_candidate_jsonls(
    paths: list[Path],
    *,
    in_place: bool = True,
    out_dir: Path | None = None,
    window_sec: float = 60.0,
    apply_time_downsample: bool = True,
) -> dict[Path, tuple[int, int]]:
    """Dedupe across taxonomy split JSONLs, then rewrite one file per taxonomy.

    When the same narration appears in A and B (mirror) or twice in C, only the
    best-scoring row is kept and written to the file for its LLM taxonomy.

    When ``apply_time_downsample`` is True, each taxonomy file is further
    thinned to one clip per ``video_id`` per ``window_sec`` interval.
    """
    if not paths:
        return {}

    by_tax_path: dict[Taxonomy, Path] = {}
    for path in paths:
        match = _TAXONOMY_JSONL_RE.search(path.stem)
        if not match:
            raise ValueError(f"Cannot infer taxonomy from filename: {path.name}")
        tax = Taxonomy(match.group(1).upper())
        by_tax_path[tax] = path

    all_records: list[dict] = []
    for path in paths:
        all_records.extend(read_jsonl(path))

    groups: dict[str, list[dict]] = defaultdict(list)
    no_narration: list[dict] = []
    for rec in all_records:
        key = narration_dedup_key(rec)
        if key:
            groups[key].append(rec)
        else:
            no_narration.append(rec)

    buckets: dict[Taxonomy, list[dict]] = {tax: [] for tax in Taxonomy}
    for group in groups.values():
        winner = pick_record_for_narration(group)
        tax = record_output_taxonomy(winner)
        stamped = {**winner, "taxonomy": tax.value}
        buckets[tax].append(stamped)
    buckets[Taxonomy.AMBIGUOUS].extend(no_narration)

    results: dict[Path, tuple[int, int]] = {}
    for tax, src_path in by_tax_path.items():
        before = sum(1 for _ in read_jsonl(src_path))
        records = buckets[tax]
        if apply_time_downsample and window_sec > 0:
            records = downsample_by_video_time(records, window_sec=window_sec)
        records.sort(
            key=lambda r: (
                video_id_from_record(r),
                candidate_timestamp_sec(r),
                str(r.get("clip_id", "")),
            )
        )
        if in_place:
            dest = src_path
        elif out_dir is not None:
            dest = out_dir / src_path.name
        else:
            dest = src_path.with_name(f"{src_path.stem}_dedup{src_path.suffix}")
        write_jsonl(dest, records)
        results[dest] = (before, len(records))

    return results


def discover_taxonomy_candidate_jsonls(
    directory: Path,
    *,
    include_ds60: bool = False,
) -> list[list[Path]]:
    """Return path groups: base A/B/C/D and optional ``*_ds60`` set."""
    groups: list[list[Path]] = []
    base = sorted(
        directory.glob("candidates_narration_[ABCD].jsonl"),
        key=lambda p: p.name,
    )
    if base:
        groups.append(base)
    if include_ds60:
        ds60 = sorted(
            directory.glob("candidates_narration_[ABCD]_ds60.jsonl"),
            key=lambda p: p.name,
        )
        if ds60:
            groups.append(ds60)
    return groups


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


def stage_new_filter(
    narration_path: Path,
    taxonomy: Taxonomy | None,
    out_path: Path,
    *,
    require_shared_noun: bool = True,
    limit: int | None = None,
    videos_root: Path | None = None,
    frame_backend: str = "ffmpeg",
    frames_out_dir: Path | None = None,
    florence_describe: bool = False,
    florence_model: str = "microsoft/Florence-2-base",
    florence_device: str | None = None,
    auto_download: bool = False,
    use_llm_parse: bool = False,
    openai_model: str = "gpt-4o-mini",
    llm_batch_size: int | None = None,
    llm_parser=None,
    require_table_object_markers: bool = True,
) -> int:
    """Filter candidates from Ego4D ``narration.json`` (dense narrations)."""
    resolver = download_ego4d_video if (auto_download and videos_root is not None) else None
    if use_llm_parse and llm_parser is None:
        from egoownership.narration_parse import OpenAINarrationParser, OpenAINarrationParserConfig

        cfg_kwargs: dict = {"model": openai_model}
        if llm_batch_size is not None:
            cfg_kwargs["batch_size"] = llm_batch_size
        llm_parser = OpenAINarrationParser(OpenAINarrationParserConfig(**cfg_kwargs))
    writer = IncrementalJsonlWriter(out_path, taxonomy)
    total = 0

    def _on_llm_batch(batch: list[ClipCandidate]) -> None:
        nonlocal total
        total += writer.write_candidates(batch)

    try:
        for cand in new_filter(
            narration_path,
            taxonomy,
            require_shared_noun=require_shared_noun,
            limit=limit,
            videos_root=videos_root,
            frame_backend=frame_backend,
            frames_out_dir=frames_out_dir,
            florence_describe=florence_describe,
            florence_model=florence_model,
            florence_device=florence_device,
            video_resolver=resolver,
            use_llm_parse=use_llm_parse,
            llm_parser=llm_parser,
            llm_batch_size=llm_batch_size,
            on_llm_batch=_on_llm_batch if use_llm_parse else None,
            require_table_object_markers=require_table_object_markers,
        ):
            if use_llm_parse:
                continue
            total += writer.write_candidates([cand])
    finally:
        writer.close()

    return total


def download_ego4d_video(video_uid: str, download_dir: Path) -> Path | None:
    from egoownership.ego4d_video import download_ego4d_video as _download

    return _download(video_uid, download_dir)


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
                video_path = download_ego4d_video(cand.video_id, videos_root)
                if video_path is None:
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


def stage_detect_native(
    candidates_path: Path,
    annotations_path: Path,
    dataset: str,
    out_path: Path,
) -> int:
    """No-model path: pull bboxes straight from the dataset annotations.

    Use this when you don't have GPU access or when you want a smoke-test run
    on real data before committing to the full model stack. Output schema is
    identical to ``stage_detect``, so ``stage_label`` can consume it as-is.
    """
    from egoownership.detection.native_bbox import stage_native_detect

    candidates = list(load_candidates(candidates_path))
    records = stage_native_detect(candidates, Path(annotations_path), dataset)
    return write_jsonl(out_path, records)


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
    remote_vlm_provider: str | None = None,
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

    remote_vlm = None
    if remote_vlm_provider:
        from egoownership.detection.remote_vlm import get_client
        remote_vlm = get_client(remote_vlm_provider)

    records: list[dict] = []
    count = 0
    for cand in load_candidates(candidates_path):
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
                # Prefer remote VLM tagging when available; fall back to local RAM.
                if remote_vlm is not None:
                    ram_tags = remote_vlm.tag_frame(fp)
                else:
                    from egoownership.detection.ram import extract_tags
                    ram_tags = extract_tags(fp)
                from egoownership.detection.ram import merge_with_clip_nouns
                prompt_nouns = merge_with_clip_nouns(nouns, ram_tags)
            prompt = build_prompt(prompt_nouns)
            detections = detect_objects(fp, prompt, dino_cfg)
            if use_sam and detections:
                from egoownership.detection.sam import refine_boxes
                detections = refine_boxes(fp, detections)

            persons, _ego_hand_bbox = detect_persons(fp) if detect_persons_too else ([], None)

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
            if remote_vlm is not None:
                for fd, fp in zip(frames, per_frame_paths):
                    fd.objects = remote_vlm.annotate_frame(fp, fd.objects)
            else:
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


def stage_label(
    detections_path: Path,
    out_path: Path,
    *,
    remote_vlm_judge: str | None = None,
    frames_root: Path | None = None,
) -> int:
    """Apply the ownership rule cascade to the detect output.

    When ``remote_vlm_judge`` is set (e.g. ``"anthropic"``), each scene also
    gets a second-opinion VLM label stored in ``scene_record.vlm_judgements``.
    Requires ``frames_root`` so the VLM can read the actual frame images.
    """
    from egoownership.detection.ownership import assign_ownership, build_scene_record
    from egoownership.detection.tracking import assign_instance_ids
    from egoownership.schema import OwnershipLabel, VLMJudgement

    judge = None
    if remote_vlm_judge:
        if frames_root is None:
            raise ValueError("--frames-root is required when --remote-vlm-judge is set")
        from egoownership.detection.remote_vlm import get_client
        judge = (remote_vlm_judge, get_client(remote_vlm_judge))

    records: list[dict] = []
    for d in read_jsonl(detections_path):
        clip = ClipCandidate.model_validate(d["clip"])
        frames = [FrameDetections.model_validate(fd) for fd in d["frames"]]
        needs_tracking = any(
            o.instance_id is None for fd in frames for o in fd.objects
        )
        if needs_tracking:
            frames = assign_instance_ids(frames)
        depth_bands = d.get("depth_bands")
        frames_with_own = assign_ownership(frames, wearer_depth_bands=depth_bands)
        scene: SceneRecord = build_scene_record(clip, frames_with_own)

        if judge is not None:
            provider, vlm = judge
            paths = [
                Path(frames_root) / fd.frame_path
                for fd in frames_with_own
                if fd.frame_path
            ]
            if len(paths) == len(frames_with_own):
                try:
                    result = vlm.judge_scene(clip, paths, scene_graph=frames_with_own)
                    label = OwnershipLabel(result.get("label", "AMBIGUOUS"))
                    model_id = f"{provider}:{getattr(vlm.cfg, 'model', 'unknown')}"
                    scene = scene.model_copy(update={
                        "vlm_judgements": {
                            model_id: VLMJudgement(
                                model_id=model_id,
                                label=label,
                                agrees=(scene.scene_label == label) if scene.scene_label else None,
                                rationale=result.get("rationale"),
                            )
                        }
                    })
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] VLM judge failed for {clip.clip_id}: {e}")

        records.append(scene.model_dump(mode="json"))
    return write_jsonl(out_path, records)
