"""Draft annotation generation for EgoLife-style captions and transcripts.

This module intentionally keeps the first pass lightweight: it uses the
available transcript / dense-caption text to propose candidate ownership labels
and to mark clips that need visual filtering. Detector or scene-graph outputs
can be merged by clip id to enforce person-count and visible-face rules.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from egoownership.schema import ClipCandidate, OwnershipLabel, Taxonomy


TEXT_FIELDS = (
    "transcript",
    "transcription",
    "dense_caption",
    "dense_captions",
    "caption",
    "av_caption",
    "narration",
)
ANNOTATION_CONTEXT_FIELDS = (
    "scenario",
    "environment",
    "scene",
    "location",
    "label",
    "day",
    "participant",
    "task",
    "activity",
)

TRANSFER_TERMS = {
    "pass",
    "give",
    "hand",
    "receive",
    "take",
    "borrow",
    "return",
    "serve",
    "offer",
    "passes",
    "gives",
    "hands",
    "receives",
    "takes",
}
TEMPORAL_ACTION_TERMS = TRANSFER_TERMS | {
    "move",
    "moves",
    "moved",
    "place",
    "places",
    "placed",
    "put",
    "puts",
    "pick",
    "picks",
    "picked",
    "bring",
    "brings",
    "brought",
    "lend",
    "lends",
    "lent",
    "slide",
    "slides",
    "slid",
    "push",
    "pushes",
    "pushed",
}
SHARED_TERMS = {
    "share",
    "shared",
    "together",
    "both",
    "everyone",
    "communal",
    "table",
    "dining",
    "meeting",
    "우리",
    "같이",
    "함께",
    "공유",
}
SHARED_OBJECT_TERMS = {
    "tissue",
    "napkin",
    "sauce",
    "condiment",
    "tongs",
    "chopsticks",
    "serving spoon",
    "dish",
    "plate of food",
    "bowl of food",
    "paper towel",
    "salt",
    "pepper",
    "oil",
    "공용",
    "소스",
    "반찬",
    "집게",
    "티슈",
}
MINE_TERMS = {"my", "mine", "i ", "i'm", "i pick", "i take", "내", "내가", "제"}
OTHER_PERSON_TERMS = {
    "person",
    "someone",
    "somebody",
    "he ",
    "she ",
    "his ",
    "her ",
    "another",
    "friend",
    "colleague",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "사람",
    "친구",
    "동료",
    "alice",
    "tasha",
    "lucia",
    "katrina",
    "shure",
    "jake",
}
SINGLE_PERSON_TERMS = {
    "alone",
    "by myself",
    "only me",
    "single person",
    "혼자",
}
AMBIGUOUS_TERMS = {
    "unclear",
    "ambiguous",
    "cannot tell",
    "hard to tell",
    "partially occluded",
    "occluded",
    "between two people",
    "middle of two people",
    "모호",
    "가려",
}
EGO_ZONE_TERMS = {
    "in front of the wearer",
    "in front of me",
    "near the wearer",
    "near me",
    "wearer's side",
    "camera wearer",
    "my side",
    "내 앞",
    "내 쪽",
}
OTHER_ZONE_TERMS = {
    "another person",
    "other person",
    "across the table",
    "opposite person",
    "in front of p",
    "near p",
    "next to p",
    "상대",
    "맞은편",
}
CENTER_ZONE_TERMS = {
    "center of the table",
    "middle of the table",
    "central table",
    "table center",
    "테이블 중앙",
    "중앙",
}
CONFLICT_TERMS = {
    "handle facing",
    "handle points",
    "screen facing",
    "open toward",
    "facing the wearer",
    "facing me",
    "name badge",
    "id card",
    "employee id",
    "keys",
    "key",
    "mirror",
    "reflection",
    "reflected",
    "screen",
    "손잡이",
    "화면",
    "거울",
    "반사",
    "사원증",
    "열쇠",
}
IDENTITY_OBJECT_TERMS = {
    "phone",
    "smartphone",
    "laptop",
    "notebook",
    "name badge",
    "id card",
    "employee id",
    "keys",
    "key",
    "wallet",
    "사원증",
    "핸드폰",
    "노트북",
    "열쇠",
    "지갑",
}
SERVING_TERMS = {
    "server",
    "waiter",
    "waitress",
    "staff",
    "serves",
    "served",
    "places a new",
    "puts a new",
    "직원",
    "서빙",
}
TABLE_CAPTION_TERMS = {
    "table",
    "desk",
    "countertop",
    "counter",
    "桌",
    "桌子",
    "桌上",
    "餐桌",
    "台面",
}
CHINESE_VERB_TERMS = (
    ("拿", "pick_up"),
    ("取", "pick_up"),
    ("拿起", "pick_up"),
    ("放", "place"),
    ("放进", "place"),
    ("放到", "place"),
    ("推", "push"),
    ("递", "pass"),
    ("给", "give"),
    ("洗", "wash"),
    ("找", "search"),
    ("翻", "search"),
    ("看", "look"),
    ("坐", "sit"),
    ("站", "stand"),
    ("走", "walk"),
    ("指", "point"),
)
CHINESE_NOUN_TERMS = (
    ("椅子", "chair"),
    ("袋子", "bag"),
    ("裱花袋", "piping_bag"),
    ("裱花嘴", "piping_tip"),
    ("手机", "phone"),
    ("电脑", "computer"),
    ("杯子", "cup"),
    ("杯", "cup"),
    ("盘子", "plate"),
    ("碗", "bowl"),
    ("勺子", "spoon"),
    ("刀", "knife"),
    ("叉", "fork"),
    ("锅", "pot"),
    ("模具", "mold"),
    ("磨具", "mold"),
    ("东西", "object"),
)
ENGLISH_VERB_TERMS = (
    ("pick up", "pick_up"),
    ("picked up", "pick_up"),
    ("take", "pick_up"),
    ("takes", "pick_up"),
    ("place", "place"),
    ("places", "place"),
    ("placed", "place"),
    ("put", "place"),
    ("puts", "place"),
    ("push", "push"),
    ("pushed", "push"),
    ("pass", "pass"),
    ("passes", "pass"),
    ("give", "give"),
    ("gives", "give"),
    ("wash", "wash"),
    ("search", "search"),
    ("look", "look"),
    ("looking", "look"),
)
ENGLISH_NOUN_TERMS = (
    ("box", "box"),
    ("container", "container"),
    ("chair", "chair"),
    ("bag", "bag"),
    ("piping bag", "piping_bag"),
    ("piping tip", "piping_tip"),
    ("phone", "phone"),
    ("pen", "pen"),
    ("pencil", "pencil"),
    ("notebook", "notebook"),
    ("paper", "paper"),
    ("book", "book"),
    ("remote", "remote"),
    ("remote control", "remote"),
    ("charger", "charger"),
    ("cable", "cable"),
    ("bottle", "bottle"),
    ("computer", "computer"),
    ("laptop", "computer"),
    ("cup", "cup"),
    ("mug", "cup"),
    ("plate", "plate"),
    ("bowl", "bowl"),
    ("spoon", "spoon"),
    ("knife", "knife"),
    ("fork", "fork"),
    ("pot", "pot"),
    ("pan", "pan"),
    ("mold", "mold"),
    ("object", "object"),
)
CHINESE_TRANSLATION_TERMS = (
    ("正在", "is "),
    ("现在", "now "),
    ("我", "I"),
    ("把", ""),
    ("在", "am at"),
    ("厨房", "kitchen"),
    ("桌子上", "on the table"),
    ("桌上", "on the table"),
    ("桌子", "table"),
    ("餐桌", "dining table"),
    ("椅子", "chair"),
    ("袋子", "bag"),
    ("裱花袋", "piping bag"),
    ("裱花嘴", "piping tip"),
    ("手机", "phone"),
    ("电脑", "computer"),
    ("杯子", "cup"),
    ("盘子", "plate"),
    ("勺子", "spoon"),
    ("东西", "object"),
    ("拿起来", "pick up"),
    ("拿起", "pick up"),
    ("拿", "pick up"),
    ("放进去", "put inside"),
    ("放到", "place at"),
    ("放", "place"),
    ("推", "push"),
    ("递", "pass"),
    ("给", "give"),
    ("洗", "wash"),
    ("找", "search for"),
    ("翻", "rummage through"),
    ("看了看", "look at"),
    ("看着", "looking at"),
    ("看", "look"),
    ("坐着", "sitting"),
    ("站着", "standing"),
    ("走", "walk"),
    ("指", "point"),
    ("聊天", "chat"),
    ("帮忙", "help"),
    ("旁边", "nearby"),
)
KNOWN_TRANSCRIPT_SPEAKER_LABELS = (
    "Jake",
    "Alice",
    "Tasha",
    "Lucia",
    "Katrina",
    "Shure",
    "其他人",
    "大家",
    "Others",
    "Other people",
    "Everyone",
)


def iter_egolife_annotation_drafts(
    annotations_path: Path,
    *,
    visual_metadata_path: Path | None = None,
    min_persons: int = 2,
    require_face: bool = True,
    include_rejected: bool = False,
    require_visual_pass: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield draft ownership annotation records from EgoLife metadata.

    ``annotations_path`` accepts one JSON, one JSONL, or a directory containing
    either. The source may be clip-level ``{"events": [...]}`` JSON or flat
    records. ``visual_metadata_path`` is an optional JSONL keyed by
    ``clip_id``/``event_id`` with fields like ``person_count`` and
    ``face_count``.
    """

    visual_by_id = _load_visual_metadata(visual_metadata_path)
    for record in _iter_records(annotations_path):
        for event_record in _flatten_record(record):
            draft = build_egolife_annotation_draft(
                event_record,
                visual_metadata=visual_by_id.get(_record_key(event_record)),
                min_persons=min_persons,
                require_face=require_face,
            )
            if require_visual_pass and draft["visual_filter_status"] != "pass":
                draft = {
                    **draft,
                    "keep_for_annotation": False,
                    "filter_reasons": [
                        *draft["filter_reasons"],
                        "visual_filter_not_passed",
                    ],
                }
            if include_rejected or draft["keep_for_annotation"]:
                yield draft


def build_egolife_annotation_draft(
    record: dict[str, Any],
    *,
    visual_metadata: dict[str, Any] | None = None,
    min_persons: int = 2,
    require_face: bool = True,
) -> dict[str, Any]:
    text_by_field = _extract_text_fields(record)
    annotation_context = _extract_annotation_context(record)
    caption_for_extraction = _primary_caption_text(text_by_field)
    target_objects = _normalize_list(
        record.get("nouns") or record.get("objects") or record.get("target_objects")
    )
    action_verb = record.get("verb") or record.get("action_verb")
    combined_text = " | ".join(text_by_field.values())
    translated_caption_text = _translate_caption_text(caption_for_extraction) or combined_text
    feature_text = _feature_text(record, text_by_field, annotation_context, target_objects)
    text_signals = _text_signals(
        feature_text,
        action_verb=action_verb,
        target_objects=target_objects,
    )
    text_signals["caption_mentions_table"] = _caption_mentions_table(caption_for_extraction)
    transcript_speakers = _extract_transcript_speakers(text_by_field.get("transcript", ""))
    text_signals["transcript_speakers"] = transcript_speakers
    text_signals["transcript_speaker_count"] = len(transcript_speakers)
    text_signals["transcript_single_speaker"] = bool(transcript_speakers) and len(transcript_speakers) == 1
    visual_status, visual_reasons = _visual_filter_status(
        visual_metadata,
        min_persons=min_persons,
        require_face=require_face,
    )
    text_status, text_reasons = _text_filter_status(text_signals)
    reasons = [*text_reasons, *visual_reasons]
    keep = text_status != "reject" and visual_status != "reject"

    return {
        "id": _record_key(record),
        "clip_id": record.get("clip_id") or record.get("id") or record.get("video_id"),
        "video_id": record.get("video_id") or record.get("video"),
        "event_idx": record.get("event_idx"),
        "start_sec": _maybe_float(record.get("start_sec") or record.get("start")),
        "end_sec": _maybe_float(record.get("end_sec") or record.get("end")),
        "taxonomy": Taxonomy.AMBIGUOUS.value,
        "proposed_label": OwnershipLabel.AMBIGUOUS.value,
        "label_source": "pending_post_filter_taxonomy",
        "target_objects": target_objects,
        "action_verb": action_verb,
        "caption_text": combined_text,
        "translated_caption_text": translated_caption_text,
        "text_fields": text_by_field,
        "annotation_context": annotation_context,
        "source": record.get("source", {}),
        "text_signals": text_signals,
        "text_filter_status": text_status,
        "visual_filter_status": visual_status,
        "visual_metadata": visual_metadata or {},
        "filter_reasons": reasons,
        "keep_for_annotation": keep,
        "tracks": {
            "caption_prefilter": {
                "status": "ready" if combined_text else "missing_caption",
                "input": combined_text,
                "suggested_task": "cheap text keep/reject/needs_vlm prefilter",
            },
            "visual_filter": {
                "status": visual_status,
                "suggested_task": "filter person count, visible faces, and person-object relations",
            },
        },
    }


def write_egolife_annotation_drafts(
    annotations_path: Path,
    out_path: Path,
    *,
    visual_metadata_path: Path | None = None,
    min_persons: int = 2,
    require_face: bool = True,
    include_rejected: bool = False,
    require_visual_pass: bool = False,
    limit: int | None = None,
    show_progress: bool = False,
    output_format: str = "draft",
) -> int:
    if output_format not in {"draft", "candidate"}:
        raise ValueError("output_format must be one of: draft, candidate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    drafts = iter_egolife_annotation_drafts(
        annotations_path,
        visual_metadata_path=visual_metadata_path,
        min_persons=min_persons,
        require_face=require_face,
        include_rejected=include_rejected,
        require_visual_pass=require_visual_pass,
    )
    progress = None
    if show_progress:
        from tqdm.auto import tqdm

        estimated_total = _estimated_output_total(
            annotations_path,
            include_rejected=include_rejected,
            require_visual_pass=require_visual_pass,
            limit=limit,
        )
        progress = tqdm(
            total=estimated_total,
            unit="record",
            desc=f"Writing EgoLife {output_format} records",
        )

    with out_path.open("w", encoding="utf-8") as f:
        try:
            for draft in drafts:
                record = (
                    egolife_draft_to_candidate_record(draft)
                    if output_format == "candidate"
                    else draft
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if progress is not None:
                    progress.update(1)
                if limit is not None and count >= limit:
                    break
        finally:
            if progress is not None:
                progress.close()
    return count


def egolife_draft_to_candidate_record(draft: dict[str, Any]) -> dict[str, Any]:
    """Convert a verbose EgoLife draft into the standard ClipCandidate JSONL shape."""

    t_sec = _candidate_t_sec(draft)
    source = {
        "bboxes": [],
        "annotation_parse": "egolife_cap",
        "proposed_label": draft.get("proposed_label"),
        "label_source": draft.get("label_source"),
        "text_filter_status": draft.get("text_filter_status"),
        "visual_filter_status": draft.get("visual_filter_status"),
        "filter_reasons": draft.get("filter_reasons") or [],
        "keep_for_annotation": draft.get("keep_for_annotation"),
        "caption_fields": draft.get("text_fields") or {},
        "original_caption_text": draft.get("caption_text"),
        "annotation_context": draft.get("annotation_context") or {},
        "start_sec": draft.get("start_sec"),
        "end_sec": draft.get("end_sec"),
        "annotation_source": draft.get("source"),
    }
    if draft.get("open_model_judgement"):
        source["open_model_judgement"] = draft["open_model_judgement"]
    if draft.get("text_signals"):
        source["text_signals"] = draft["text_signals"]
    if draft.get("overlap_metadata"):
        source["overlap_metadata"] = draft["overlap_metadata"]

    candidate = ClipCandidate(
        dataset="egolife",
        clip_id=str(draft.get("id") or draft.get("clip_id") or draft.get("video_id")),
        video_id=str(draft.get("video_id") or draft.get("clip_id") or ""),
        taxonomy=Taxonomy(str(draft.get("taxonomy") or Taxonomy.AMBIGUOUS.value)),
        t_minus_2_sec=max(0.0, t_sec - 2.0),
        t_minus_1_sec=max(0.0, t_sec - 1.0),
        t_sec=t_sec,
        verb=draft.get("action_verb"),
        nouns=list(draft.get("target_objects") or []),
        narration=draft.get("translated_caption_text") or draft.get("caption_text"),
        source=source,
    )
    return candidate.model_dump(mode="json")


def _candidate_t_sec(draft: dict[str, Any]) -> float:
    for key in ("end_sec", "t_sec", "start_sec"):
        value = _maybe_float(draft.get(key))
        if value is not None:
            return value
    return 0.0


def _estimated_output_total(
    annotations_path: Path,
    *,
    include_rejected: bool,
    require_visual_pass: bool,
    limit: int | None,
) -> int | None:
    if limit is not None:
        return limit
    if not include_rejected or require_visual_pass:
        return None
    return _count_source_records(annotations_path)


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        handled_egolife_cap = False
        if (p / "DenseCaption").exists() or (p / "EgoLifeCap" / "DenseCaption").exists():
            handled_egolife_cap = True
            yield from _iter_egolife_cap_srt_records(p)
        allowed_suffixes = {".json", ".jsonl"} if handled_egolife_cap else {".json", ".jsonl", ".srt"}
        for child in sorted(p.rglob("*")):
            if child.suffix.lower() in allowed_suffixes:
                yield from _iter_records(child)
        return
    if p.suffix.lower() == ".srt":
        for cue in _parse_srt_file(p):
            participant, day, clip_id = _parse_egolife_path_parts(p)
            yield {
                "id": f"{clip_id}:{cue['start_sec']:.3f}-{cue['end_sec']:.3f}",
                "clip_id": clip_id,
                "video_id": clip_id,
                "participant": participant,
                "day": day,
                "start_sec": cue["start_sec"],
                "end_sec": cue["end_sec"],
                "caption": cue["text"],
                "source": {"annotation_file": str(p)},
            }
        return
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return
    with p.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        yield value


def _count_source_records(path: Path) -> int | None:
    p = Path(path)
    if p.is_dir():
        handled_egolife_cap = False
        total = 0
        if (p / "DenseCaption").exists() or (p / "EgoLifeCap" / "DenseCaption").exists():
            handled_egolife_cap = True
            cap_total = _count_egolife_cap_srt_records(p)
            if cap_total is None:
                return None
            total += cap_total
        allowed_suffixes = {".json", ".jsonl"} if handled_egolife_cap else {".json", ".jsonl", ".srt"}
        for child in sorted(p.rglob("*")):
            if child.suffix.lower() not in allowed_suffixes:
                continue
            child_total = _count_source_records(child)
            if child_total is None:
                return None
            total += child_total
        return total
    if p.suffix.lower() == ".srt":
        return _count_srt_cues(p)
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if p.suffix.lower() != ".json":
        return None
    with p.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if isinstance(value, list):
        return sum(1 for item in value if isinstance(item, dict))
    if isinstance(value, dict):
        events = value.get("events")
        if isinstance(events, list):
            return sum(1 for item in events if isinstance(item, dict))
        return 1
    return None


def _count_egolife_cap_srt_records(path: Path) -> int | None:
    p = Path(path)
    cap_root = p / "EgoLifeCap" if (p / "EgoLifeCap").exists() else p
    dense_root = cap_root / "DenseCaption"
    if not dense_root.exists():
        return None
    return sum(_count_srt_cues(srt_path) for srt_path in dense_root.rglob("*.srt"))


def _count_srt_cues(path: Path) -> int:
    text = Path(path).read_text(encoding="utf-8-sig")
    return sum(
        1
        for block in re.split(r"\n\s*\n", text.strip())
        if "-->" in block and block.strip()
    )


def _iter_egolife_cap_srt_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield timestamp-aligned records from EgoLifeCap dense captions.

    EgoLifeCap stores dense captions and transcripts as mirrored SRT trees:
    ``EgoLifeCap/DenseCaption/<participant>/<day>/<clip>.srt`` and
    ``EgoLifeCap/Transcript/<participant>/<day>/<clip>.srt``. Dense-caption
    cues become candidate annotation events, with overlapping transcript cues
    attached as extra evidence.
    """

    p = Path(path)
    cap_root = p / "EgoLifeCap" if (p / "EgoLifeCap").exists() else p
    dense_root = cap_root / "DenseCaption"
    transcript_root = cap_root / "Transcript"
    if not dense_root.exists():
        return

    for dense_path in sorted(dense_root.rglob("*.srt")):
        rel = dense_path.relative_to(dense_root)
        transcript_path = transcript_root / rel
        transcript_cues = _parse_srt_file(transcript_path) if transcript_path.exists() else []
        participant, day, clip_id = _parse_egolife_path_parts(dense_path)
        for cue in _parse_srt_file(dense_path):
            overlapping_transcript = [
                transcript["text"]
                for transcript in transcript_cues
                if _time_ranges_overlap(
                    cue["start_sec"],
                    cue["end_sec"],
                    transcript["start_sec"],
                    transcript["end_sec"],
                )
            ]
            yield {
                "id": f"{clip_id}:{cue['start_sec']:.3f}-{cue['end_sec']:.3f}",
                "clip_id": clip_id,
                "video_id": clip_id,
                "participant": participant,
                "day": day,
                "start_sec": cue["start_sec"],
                "end_sec": cue["end_sec"],
                "dense_caption": cue["text"],
                "transcript": " ".join(overlapping_transcript),
                "source": {
                    "dataset": "lmms-lab/EgoLife",
                    "dense_caption_file": str(dense_path),
                    "transcript_file": str(transcript_path) if transcript_path.exists() else None,
                },
            }


def _parse_srt_file(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    cues: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_idx = next((idx for idx, line in enumerate(lines) if "-->" in line), None)
        if time_idx is None:
            continue
        time_match = re.match(
            r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[time_idx],
        )
        if time_match is None:
            continue
        cue_text = _normalize_space(" ".join(lines[time_idx + 1 :]))
        if not cue_text:
            continue
        cues.append(
            {
                "start_sec": _srt_time_to_seconds(time_match.group("start")),
                "end_sec": _srt_time_to_seconds(time_match.group("end")),
                "text": cue_text,
            }
        )
    return cues


def _srt_time_to_seconds(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    seconds, millis = rest.split(".")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def _time_ranges_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and start_b < end_a


def _find_retained_time_overlap(
    draft: dict[str, Any],
    retained_windows: dict[tuple[str, str, str], list[tuple[float, float, str]]],
) -> dict[str, Any] | None:
    start = _maybe_float(draft.get("start_sec"))
    end = _maybe_float(draft.get("end_sec"))
    if start is None or end is None or end <= start:
        return None
    key = _draft_time_window_key(draft)
    for prev_start, prev_end, prev_id in retained_windows.get(key, []):
        if _time_ranges_overlap(start, end, prev_start, prev_end):
            return {
                "overlap_with_id": prev_id,
                "overlap_start_sec": prev_start,
                "overlap_end_sec": prev_end,
            }
    return None


def _remember_retained_time_window(
    draft: dict[str, Any],
    retained_windows: dict[tuple[str, str, str], list[tuple[float, float, str]]],
) -> None:
    start = _maybe_float(draft.get("start_sec"))
    end = _maybe_float(draft.get("end_sec"))
    if start is None or end is None or end <= start:
        return
    key = _draft_time_window_key(draft)
    retained_windows.setdefault(key, []).append((start, end, str(draft.get("id") or "")))


def _mark_time_overlap_rejected(draft: dict[str, Any], overlap: dict[str, Any]) -> dict[str, Any]:
    reasons = list(draft.get("filter_reasons") or [])
    if "overlap_with_existing_time_window" not in reasons:
        reasons.append("overlap_with_existing_time_window")
    return {
        **draft,
        "text_filter_status": "reject",
        "filter_reasons": reasons,
        "keep_for_annotation": False,
        "overlap_metadata": overlap,
    }


def _draft_time_window_key(draft: dict[str, Any]) -> tuple[str, str, str]:
    context = draft.get("annotation_context") or {}
    participant = context.get("participant") if isinstance(context, dict) else None
    day = context.get("day") if isinstance(context, dict) else None
    video_id = draft.get("video_id") or draft.get("clip_id") or ""
    return str(video_id), str(participant or ""), str(day or "")


def _parse_egolife_path_parts(path: Path) -> tuple[str | None, str | None, str]:
    parts = Path(path).parts
    participant: str | None = None
    day: str | None = None
    for idx, part in enumerate(parts):
        if participant is None and re.match(r"A\d+_", part):
            participant = part
            if idx + 1 < len(parts) and parts[idx + 1].upper().startswith("DAY"):
                day = parts[idx + 1]
        if day is None and part.upper().startswith("DAY"):
            day = part
    return participant, day, Path(path).stem


def _flatten_record(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    events = record.get("events")
    if not isinstance(events, list):
        yield record
        return
    base = {k: v for k, v in record.items() if k != "events"}
    for idx, event in enumerate(events):
        if isinstance(event, dict):
            yield {**base, **event, "event_idx": idx}


def divide_egolife_taxonomy(
    signals: dict[str, bool],
    *,
    action_verb: Any = None,
    target_objects: list[str] | None = None,
) -> tuple[Taxonomy, OwnershipLabel, str]:
    """Map EgoLife caption/annotation signals to the ownership taxonomy.

    Priority order follows the benchmark definition:
    C contextual overrides need temporal evidence, B conflicts need opposed
    semantic/spatial cues, A is static cue agreement, and D is true ambiguity or
    insufficient cueing.
    """

    normalized_verb = normalize_text_token(action_verb)
    object_text = " ".join(target_objects or []).casefold()

    if signals["mentions_ambiguity"]:
        return Taxonomy.AMBIGUOUS, OwnershipLabel.AMBIGUOUS, "caption_explicit_ambiguous"

    if signals["mentions_temporal_action"] or normalized_verb in TEMPORAL_ACTION_TERMS:
        if signals["mentions_serving_to_ego"]:
            return Taxonomy.CONTEXTUAL, OwnershipLabel.MINE, "context_served_to_ego"
        if signals["mentions_shared_object"] and (
            signals["mentions_center_zone"] or signals["mentions_ego_zone"]
        ):
            return Taxonomy.CONTEXTUAL, OwnershipLabel.SHARED, "context_shared_origin_or_temporary_use"
        if signals["mentions_transfer"] and signals["mentions_other_person"]:
            return Taxonomy.CONTEXTUAL, OwnershipLabel.PERSON_K, "context_transfer_to_person"
        if signals["mentions_mine_context"] and signals["mentions_other_person"]:
            return Taxonomy.CONTEXTUAL, OwnershipLabel.MINE, "context_prior_possession"
        if signals["mentions_other_person"]:
            return Taxonomy.CONTEXTUAL, OwnershipLabel.PERSON_K, "context_other_person_action"
        if signals["mentions_shared_context"]:
            return Taxonomy.CONTEXTUAL, OwnershipLabel.SHARED, "context_shared_interaction"
        return Taxonomy.CONTEXTUAL, OwnershipLabel.AMBIGUOUS, "context_action_needs_vlm"

    if signals["mentions_conflict_cue"]:
        if signals["mentions_reflection"]:
            return Taxonomy.CONFLICT, OwnershipLabel.MINE, "conflict_reflection_exclusion"
        if signals["mentions_shared_object"] and signals["mentions_ego_zone"]:
            return Taxonomy.CONFLICT, OwnershipLabel.SHARED, "conflict_shared_object_near_ego"
        if signals["mentions_identity_object"] and signals["mentions_ego_zone"]:
            return Taxonomy.CONFLICT, OwnershipLabel.PERSON_K, "conflict_identity_object_near_ego"
        if signals["mentions_handle_oriented_to_ego"]:
            return Taxonomy.CONFLICT, OwnershipLabel.MINE, "conflict_affordance_oriented_to_ego"
        if signals["mentions_screen_oriented_to_ego"]:
            return Taxonomy.CONFLICT, OwnershipLabel.SHARED, "conflict_screen_oriented_to_ego"
        return Taxonomy.CONFLICT, OwnershipLabel.AMBIGUOUS, "conflict_needs_visual_judge"

    if (
        signals["mentions_shared_object"]
        or signals["mentions_center_zone"]
        or signals["mentions_shared_context"]
    ):
        return Taxonomy.BASELINE, OwnershipLabel.SHARED, "baseline_shared_object_or_center_zone"
    if signals["mentions_other_zone"] or (
        signals["mentions_identity_object"] and signals["mentions_other_person"]
    ):
        return Taxonomy.BASELINE, OwnershipLabel.PERSON_K, "baseline_other_person_zone"
    if signals["mentions_ego_zone"] or signals["mentions_mine_context"]:
        return Taxonomy.BASELINE, OwnershipLabel.MINE, "baseline_ego_zone"
    if any(term in object_text for term in SHARED_OBJECT_TERMS):
        return Taxonomy.BASELINE, OwnershipLabel.SHARED, "baseline_shared_object_type"

    return Taxonomy.AMBIGUOUS, OwnershipLabel.AMBIGUOUS, "needs_vlm"


def _extract_text_fields(record: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in TEXT_FIELDS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            fields[key] = _normalize_space(value)
        elif isinstance(value, list):
            text = " ".join(str(item) for item in value if item)
            if text.strip():
                fields[key] = _normalize_space(text)
    source = record.get("source")
    if isinstance(source, str) and source.strip():
        fields["source"] = _normalize_space(source)
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        human_parts: list[str] = []
        assistant_parts: list[str] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("value", "")).replace("<speech>", " ").replace("<image>", " ")
            if not text.strip():
                continue
            speaker = str(turn.get("from", "")).lower()
            if speaker in {"gpt", "assistant", "model"}:
                assistant_parts.append(text)
            elif speaker in {"human", "user"}:
                human_parts.append(text)
        if assistant_parts:
            fields["conversation_answer"] = _normalize_space(" ".join(assistant_parts))
        if human_parts:
            fields["conversation_prompt"] = _normalize_space(" ".join(human_parts))
    return fields


def _extract_annotation_context(record: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in ANNOTATION_CONTEXT_FIELDS:
        value = record.get(key)
        if value not in (None, ""):
            fields[key] = _normalize_space(str(value))
    source = record.get("source")
    if isinstance(source, dict):
        for key in ANNOTATION_CONTEXT_FIELDS:
            value = source.get(key)
            if value not in (None, "") and key not in fields:
                fields[key] = _normalize_space(str(value))
    return fields


def _feature_text(
    record: dict[str, Any],
    text_by_field: dict[str, str],
    annotation_context: dict[str, str],
    target_objects: list[str],
) -> str:
    values: list[str] = []
    values.extend(text_by_field.values())
    values.extend(annotation_context.values())
    values.extend(target_objects)
    for key in ("verb", "action_verb", "object", "objects", "target", "target_object"):
        value = record.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return " | ".join(_normalize_space(str(value)) for value in values if str(value).strip())


def _text_signals(
    text: str,
    *,
    action_verb: Any = None,
    target_objects: list[str] | None = None,
) -> dict[str, bool]:
    lowered = f" {text.casefold()} "
    normalized_verb = normalize_text_token(action_verb)
    object_text = f" {' '.join(target_objects or []).casefold()} "
    combined = f"{lowered} {object_text}"
    return {
        "has_caption": bool(text.strip()),
        "mentions_transfer": _contains_any(combined, TRANSFER_TERMS),
        "mentions_temporal_action": _contains_any(combined, TEMPORAL_ACTION_TERMS)
        or normalized_verb in TEMPORAL_ACTION_TERMS,
        "mentions_shared_context": _contains_any(combined, SHARED_TERMS),
        "mentions_shared_object": _contains_any(combined, SHARED_OBJECT_TERMS),
        "mentions_mine_context": _contains_any(combined, MINE_TERMS),
        "mentions_other_person": _contains_any(combined, OTHER_PERSON_TERMS)
        or bool(re.search(r"\bp[1-6]\b", combined)),
        "mentions_ego_zone": _contains_any(combined, EGO_ZONE_TERMS),
        "mentions_other_zone": _contains_any(combined, OTHER_ZONE_TERMS),
        "mentions_center_zone": _contains_any(combined, CENTER_ZONE_TERMS),
        "mentions_conflict_cue": _contains_any(combined, CONFLICT_TERMS),
        "mentions_identity_object": _contains_any(combined, IDENTITY_OBJECT_TERMS),
        "mentions_handle_oriented_to_ego": "handle" in combined and (
            "facing" in combined or "toward" in combined or "points" in combined
        ),
        "mentions_screen_oriented_to_ego": "screen" in combined and (
            "facing" in combined or "open toward" in combined or "toward the wearer" in combined
        ),
        "mentions_reflection": _contains_any(combined, {"mirror", "reflection", "reflected", "거울", "반사"}),
        "mentions_serving_to_ego": _contains_any(combined, SERVING_TERMS)
        and _contains_any(combined, EGO_ZONE_TERMS | {"wearer", "me", "my"}),
        "mentions_ambiguity": _contains_any(combined, AMBIGUOUS_TERMS),
        "suggests_single_person": _contains_any(combined, SINGLE_PERSON_TERMS),
        "requests_or_dialogue": "?" in text or any(
            phrase in lowered
            for phrase in (" could you ", " can you ", " would you ", " please ", " 줘 ", " 주세요 ")
        ),
    }


def _text_filter_status(signals: dict[str, bool]) -> tuple[str, list[str]]:
    if not signals["has_caption"]:
        return "needs_caption", ["missing_caption_or_transcript"]
    if not signals.get("caption_mentions_table"):
        return "reject", ["caption_missing_table"]
    return "keep", []


def _propose_taxonomy_and_label(signals: dict[str, bool]) -> tuple[Taxonomy, OwnershipLabel, str]:
    return divide_egolife_taxonomy(signals)


def _primary_caption_text(text_by_field: dict[str, str]) -> str:
    for key in ("dense_caption", "caption", "av_caption", "narration"):
        value = text_by_field.get(key)
        if value:
            return value
    return ""


def _caption_mentions_table(caption: str) -> bool:
    lowered = caption.casefold()
    return any(term.casefold() in lowered for term in TABLE_CAPTION_TERMS)


def _extract_caption_verb_nouns(caption: str) -> tuple[str | None, list[str]]:
    verb: str | None = None
    for marker, normalized in CHINESE_VERB_TERMS:
        if marker in caption:
            verb = normalized
            break
    lowered = f" {caption.casefold()} "
    if verb is None:
        for marker, normalized in ENGLISH_VERB_TERMS:
            if re.search(rf"(?<![a-z0-9_]){re.escape(marker)}(?![a-z0-9_])", lowered):
                verb = normalized
                break
    nouns: list[str] = []
    seen: set[str] = set()
    for marker, normalized in CHINESE_NOUN_TERMS:
        if marker in caption and normalized not in seen:
            nouns.append(normalized)
            seen.add(normalized)
    for marker, normalized in ENGLISH_NOUN_TERMS:
        if normalized in seen:
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(marker)}s?(?![a-z0-9_])", lowered):
            nouns.append(normalized)
            seen.add(normalized)
    return verb, nouns


def _translate_caption_text(caption: str) -> str:
    caption = _normalize_space(caption)
    if not caption:
        return ""
    if not _contains_cjk(caption):
        return caption

    pattern_translation = _translate_caption_pattern(caption)
    if pattern_translation:
        return pattern_translation

    translated = caption
    for source, target in CHINESE_TRANSLATION_TERMS:
        translated = translated.replace(source, f" {target} ")
    translated = re.sub(r"\s+", " ", translated).strip()
    translated = _cleanup_translated_caption(translated)
    if translated and translated != caption:
        return translated
    return f"[untranslated] {caption}"


def _translate_caption_pattern(caption: str) -> str | None:
    match = re.fullmatch(r"我把(?P<object>.+?)推了过去，现在看着桌子上(?P<table_object>.+)", caption)
    if match:
        obj = _translate_chinese_noun_phrase(match.group("object"))
        table_obj = _translate_chinese_noun_phrase(match.group("table_object"))
        return f"I pushed the {obj} over and am now looking at {table_obj} on the table."

    match = re.fullmatch(r"(?P<person>[A-Za-z]+)在坐着看(?P<object>.+)", caption)
    if match:
        obj = _translate_chinese_noun_phrase(match.group("object"))
        return f"{match.group('person')} is sitting and looking at a {obj}."

    match = re.fullmatch(r"(?P<person>[A-Za-z]+)正在拿(?P<object>.+)", caption)
    if match:
        obj = _translate_chinese_noun_phrase(match.group("object"))
        return f"{match.group('person')} is picking up {obj}."

    match = re.fullmatch(r"我现在在(?P<place>.+)里", caption)
    if match:
        place = _translate_chinese_noun_phrase(match.group("place"))
        return f"I am now in the {place}."

    return None


def _translate_chinese_noun_phrase(text: str) -> str:
    translated = text
    for source, target in CHINESE_TRANSLATION_TERMS:
        translated = translated.replace(source, f" {target} ")
    translated = _cleanup_translated_caption(translated)
    if translated:
        translated = translated[0].lower() + translated[1:]
    return translated or "object"


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _cleanup_translated_caption(text: str) -> str:
    text = text.replace("I am at", "I am in")
    text = text.replace("I is ", "I am ")
    text = text.replace("I now", "I am now")
    text = re.sub(r"\s+了\b", "", text)
    text = re.sub(r"\s+的\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _extract_transcript_speakers(transcript: str) -> list[str]:
    """Return explicit speaker labels found in an EgoLife transcript string.

    EgoLife transcripts are commonly speaker-prefixed and bilingual, e.g.
    ``Jake: ... Jake: ...``. We only use explicit labels as a conservative
    signal; transcript text without labels is left for the VLM.
    """

    if not transcript.strip():
        return []
    speakers: set[str] = set()
    known_pattern = "|".join(re.escape(label) for label in KNOWN_TRANSCRIPT_SPEAKER_LABELS)
    for match in re.finditer(rf"(?<!\S)(?P<label>{known_pattern})\s*[:：]", transcript, flags=re.IGNORECASE):
        label = _normalize_transcript_speaker(match.group("label"))
        if label:
            speakers.add(label)
    if speakers:
        return sorted(speakers)
    for match in re.finditer(r"(?:^|\s)(?P<label>[^:：]{1,32})\s*[:：]", transcript):
        label = _normalize_transcript_speaker(match.group("label"))
        if label:
            speakers.add(label)
    return sorted(speakers)


def _normalize_transcript_speaker(label: str) -> str | None:
    label = _normalize_space(label)
    label = re.sub(r"^[\"'“”‘’\[\](){}]+|[\"'“”‘’\[\](){}]+$", "", label).strip()
    if not label:
        return None
    label = re.sub(r"^(speaker|spk)\s*", "", label, flags=re.IGNORECASE).strip()
    lowered = label.casefold()
    if lowered in {"others", "other", "other people", "everyone"} or label in {"其他人", "大家"}:
        return "other_people"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_. -]{0,24}", label):
        return lowered
    if re.fullmatch(r"[\u4e00-\u9fff]{1,8}", label):
        return label
    return None


def _visual_filter_status(
    metadata: dict[str, Any] | None,
    *,
    min_persons: int,
    require_face: bool,
) -> tuple[str, list[str]]:
    if not metadata:
        return "pending", ["needs_visual_person_face_filter"]
    reasons: list[str] = []
    face_count = _maybe_int(
        _first_present(metadata, "face_count", "faces_count", "num_faces")
    )
    if require_face and face_count is not None and face_count <= 0:
        reasons.append("no_visible_face")
    if require_face and face_count is None:
        reasons.append("face_count_missing")
    if reasons:
        return "reject", reasons
    return "pass", []


def _load_visual_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {_record_key(record): record for record in _iter_records(path)}


def _record_key(record: dict[str, Any]) -> str:
    for key in ("id", "event_id", "clip_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    video = record.get("video_id") or record.get("video") or "unknown"
    event_idx = record.get("event_idx")
    if event_idx is not None:
        return f"{video}:event_{event_idx}"
    start = record.get("start_sec") or record.get("start")
    end = record.get("end_sec") or record.get("end")
    if start is not None or end is not None:
        return f"{video}:{start}-{end}"
    return str(video)


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _contains_any(text: str, terms: set[str]) -> bool:
    for term in terms:
        normalized = term.casefold()
        if re.fullmatch(r"[a-z0-9_]+", normalized):
            if re.search(rf"\b{re.escape(normalized)}\b", text):
                return True
        elif normalized in text:
            return True
    return False


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_token(value: Any) -> str:
    if value in (None, ""):
        return ""
    return _normalize_space(str(value)).casefold().replace("_", " ")


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None
