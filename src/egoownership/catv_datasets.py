"""Dataset adapters for the CAT-V bbox → caption → label pipeline.

Each adapter normalizes caption records, resolves local video paths, and
chooses cache-directory layout for frames / CAT-V work files.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from egoownership.catv_io import iter_jsonl, normalize_object_noun, safe_path_part
from egoownership.catv_nouns import (
    _ego4d_narration_example_id,
    _iter_ego4d_table_narrations,
    extract_caption_object_nouns,
    ego4d_person_tokens,
)
from egoownership.egolife_annotations import (
    _caption_mentions_table,
    _extract_caption_verb_nouns,
    _iter_egolife_cap_srt_records,
    _translate_caption_text,
)
from egoownership.ego4d_video import (
    DEFAULT_CLIP_WINDOW_SEC,
    centered_clip_window,
    default_scratch_root,
    download_ego4d_video,
    ego4d_subclip_cache_path,
    ensure_ego4d_subclip,
    locate_ego4d_full_video,
)


def normalize_dataset_id(name: str) -> str:
    value = name.strip().lower().replace("-", "_")
    aliases = {
        "ego4d": "ego4d_fho",
        "ego4d_fho": "ego4d_fho",
        "egolife": "egolife",
        "ego_life": "egolife",
        "generic": "generic",
        "custom": "generic",
    }
    return aliases.get(value, value)


class CatVDatasetAdapter(Protocol):
    dataset_id: str

    def iter_caption_records(
        self,
        path: Path,
        *,
        object_nouns: set[str] | None = None,
    ) -> Iterable[dict[str, Any]]: ...

    def resolve_video_segment(
        self,
        videos_root: Path,
        record: dict[str, Any],
    ) -> tuple[Path, float, float] | None: ...

    def storage_parts(self, record: dict[str, Any]) -> tuple[str, str]: ...


def _iter_filtered_qwen_caption_records(
    path: Path,
    *,
    object_nouns: set[str] | None,
) -> Iterable[dict[str, Any]]:
    for record in iter_jsonl(path):
        caption = str(record.get("dense_caption") or "")
        translated = str(record.get("dense_caption_en") or record.get("qwen_translation") or "")
        if not _caption_mentions_table(caption) and not _caption_mentions_table(translated):
            continue

        raw_nouns = [
            str(noun).strip()
            for noun in (record.get("noun_candidates") or [])
            if str(noun).strip()
        ]
        if object_nouns is not None:
            nouns = [noun for noun in raw_nouns if normalize_object_noun(noun) in object_nouns]
        else:
            nouns = [
                noun
                for noun in raw_nouns
                if normalize_object_noun(noun) not in {"object", "table", "chair"}
            ]
        if not nouns:
            continue

        raw_verbs = [
            str(verb).strip()
            for verb in (record.get("verb_candidates") or [])
            if str(verb).strip()
        ]
        yield {
            **record,
            "id": record.get("id")
            or f"{record.get('clip_id')}:{float(record.get('start_sec') or 0.0):.3f}-{float(record.get('end_sec') or 0.0):.3f}",
            "clip_id": record.get("clip_id") or record.get("video_id"),
            "video_id": record.get("video_id") or record.get("clip_id"),
            "dense_caption_en": translated,
            "verb": raw_verbs[0] if raw_verbs else None,
            "nouns": nouns,
        }


@dataclass(frozen=True)
class EgoLifeCatVAdapter:
    dataset_id: str = "egolife"

    def iter_caption_records(
        self,
        path: Path,
        *,
        object_nouns: set[str] | None = None,
    ) -> Iterable[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            yield from _iter_filtered_qwen_caption_records(path, object_nouns=object_nouns)
            return

        for record in _iter_egolife_cap_srt_records(path):
            caption = record.get("dense_caption") or ""
            translated = _translate_caption_text(caption)
            verb, nouns = _extract_caption_verb_nouns(translated or caption)
            nouns = [noun for noun in nouns if noun not in {"object", "table", "chair"}]
            if object_nouns is not None:
                nouns = [noun for noun in nouns if normalize_object_noun(noun) in object_nouns]
            if not _caption_mentions_table(caption) and not _caption_mentions_table(translated):
                continue
            if not nouns:
                continue
            yield {
                **record,
                "source_dataset": self.dataset_id,
                "dataset": self.dataset_id,
                "dense_caption_en": translated,
                "verb": verb,
                "nouns": nouns,
            }

    def resolve_video_segment(
        self,
        videos_root: Path,
        record: dict[str, Any],
    ) -> tuple[Path, float, float] | None:
        return resolve_egolife_video_segment(videos_root, record)

    def storage_parts(self, record: dict[str, Any]) -> tuple[str, str]:
        participant = safe_path_part(str(record.get("participant") or "unknown_participant"))
        day = safe_path_part(str(record.get("day") or "unknown_day"))
        return participant, day


@dataclass(frozen=True)
class GenericJsonlAdapter:
    """Pass-through adapter for any custom JSONL where each record already has
    ``video_path``, ``clip_id``/``video_id``, ``start_sec``, ``end_sec``, and
    ``nouns``.  Use ``--dataset generic`` to bypass dataset-specific pre-filtering
    and video-resolution logic — the pipeline trusts records are already normalized.

    Typical use-case: a new dataset (EPIC-Kitchens, Assembly101, …) for which no
    dedicated adapter exists yet, or a custom annotation file assembled manually.

    Required per-record fields
    --------------------------
    video_path  : absolute path to the source video file
    clip_id     : unique clip identifier
    video_id    : video file stem (used as fallback path lookup)
    start_sec   : action/caption start time within the video
    end_sec     : action/caption end time within the video
    nouns       : list[str] of target object nouns
    """

    dataset_id: str = "generic"

    def iter_caption_records(
        self,
        path: Path,
        *,
        object_nouns: set[str] | None = None,
    ) -> Iterable[dict[str, Any]]:
        for record in iter_jsonl(path):
            nouns = [str(n).strip() for n in (record.get("nouns") or []) if str(n).strip()]
            if object_nouns is not None:
                nouns = [n for n in nouns if normalize_object_noun(n) in object_nouns]
            if not nouns:
                continue
            yield {
                **record,
                "nouns": nouns,
                "source_dataset": record.get("source_dataset") or self.dataset_id,
                "dataset": record.get("dataset") or self.dataset_id,
            }

    def resolve_video_segment(
        self,
        videos_root: Path,
        record: dict[str, Any],
    ) -> tuple[Path, float, float] | None:
        # Prefer the absolute video_path already stored in the record.
        video_path_str = str(record.get("video_path") or "")
        if video_path_str:
            video_path = Path(video_path_str)
            if video_path.exists():
                return video_path, float(record.get("start_sec") or 0.0), 0.0
        # Fallback: resolve relative to videos_root by video_id.
        video_id = str(record.get("video_id") or record.get("clip_id") or "")
        if video_id and videos_root.exists():
            for ext in (".mp4", ".MP4", ".mov", ".MOV", ".avi"):
                candidate = videos_root / f"{video_id}{ext}"
                if candidate.exists():
                    return candidate, float(record.get("start_sec") or 0.0), 0.0
        return None

    def storage_parts(self, record: dict[str, Any]) -> tuple[str, str]:
        video_id = safe_path_part(str(record.get("video_id") or record.get("clip_id") or "unknown"))
        return video_id, "clips"


@dataclass
class Ego4DCatVAdapter:
    dataset_id: str = "ego4d_fho"
    clip_window_sec: float = DEFAULT_CLIP_WINDOW_SEC
    auto_download: bool = True
    require_observer: bool = True

    def iter_caption_records(
        self,
        path: Path,
        *,
        object_nouns: set[str] | None = None,
    ) -> Iterable[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Ego4D input path not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            yield from self._iter_jsonl_caption_records(path, object_nouns=object_nouns)
            return
        if suffix == ".json":
            yield from self._iter_narration_json_caption_records(path, object_nouns=object_nouns)
            return

        raise ValueError(
            f"Ego4D CAT-V input must be narration.json or a candidates JSONL file, got: {path}"
        )

    def _iter_jsonl_caption_records(
        self,
        path: Path,
        *,
        object_nouns: set[str] | None,
    ) -> Iterable[dict[str, Any]]:
        for record in iter_jsonl(path):
            narration = str(record.get("narration") or record.get("dense_caption") or "").strip()
            if not narration:
                continue
            if not _caption_mentions_table(narration):
                continue

            try:
                anchor_sec = float(record.get("t_sec") or record.get("end_sec") or 0.0)
            except (TypeError, ValueError):
                continue

            clip_id = str(record.get("clip_id") or record.get("id") or "")
            video_id = str(record.get("video_id") or "")
            if not clip_id or not video_id:
                continue

            built = _build_ego4d_caption_record(
                video_id=video_id,
                clip_id=clip_id,
                anchor_sec=anchor_sec,
                narration=narration,
                object_nouns=object_nouns,
                clip_window_sec=self.clip_window_sec,
                dataset_id=self.dataset_id,
                base_record=record,
                input_kind="jsonl",
            )
            if built is not None:
                yield built

    def _iter_narration_json_caption_records(
        self,
        path: Path,
        *,
        object_nouns: set[str] | None,
    ) -> Iterable[dict[str, Any]]:
        for video_uid, entry, narration in _iter_ego4d_table_narrations(
            path,
            require_observer=self.require_observer,
        ):
            try:
                anchor_sec = float(
                    entry.get("timestamp_sec")
                    or entry.get("_unmapped_timestamp_sec")
                    or 0.0
                )
            except (TypeError, ValueError):
                continue

            clip_id = _ego4d_narration_example_id(video_uid, entry)
            built = _build_ego4d_caption_record(
                video_id=video_uid,
                clip_id=clip_id,
                anchor_sec=anchor_sec,
                narration=narration,
                object_nouns=object_nouns,
                clip_window_sec=self.clip_window_sec,
                dataset_id=self.dataset_id,
                base_record={
                    "annotation_uid": entry.get("annotation_uid"),
                    "timestamp_sec": entry.get("timestamp_sec"),
                },
                input_kind="narration_json",
            )
            if built is not None:
                yield built

    def resolve_video_segment(
        self,
        videos_root: Path,
        record: dict[str, Any],
    ) -> tuple[Path, float, float] | None:
        video_id = str(record.get("video_id") or "")
        if not video_id:
            return None

        videos_root_path = Path(videos_root)
        full_video = locate_ego4d_full_video(video_id, videos_root_path)
        if full_video is None and self.auto_download:
            full_video = download_ego4d_video(video_id, videos_root_path)

        if full_video is None:
            return None

        window_start = float(record.get("source_window_start_sec") or 0.0)
        window_duration = float(record.get("source_window_duration_sec") or self.clip_window_sec)
        if window_duration <= 0:
            local_start = float(record.get("start_sec") or 0.0)
            return full_video, max(0.0, local_start), 0.0

        cache_path = ego4d_subclip_cache_path(
            videos_root_path,
            video_id=video_id,
            clip_id=str(record.get("clip_id") or record.get("id") or video_id),
            window_start_sec=window_start,
        )
        subclip = ensure_ego4d_subclip(
            full_video,
            cache_path,
            window_start_sec=window_start,
            window_duration_sec=window_duration,
        )
        if subclip is None:
            return None
        return subclip, 0.0, window_start

    def storage_parts(self, record: dict[str, Any]) -> tuple[str, str]:
        video_id = safe_path_part(str(record.get("video_id") or "unknown_video"))
        return video_id, "clips"


def _build_ego4d_caption_record(
    *,
    video_id: str,
    clip_id: str,
    anchor_sec: float,
    narration: str,
    object_nouns: set[str] | None,
    clip_window_sec: float,
    dataset_id: str,
    base_record: dict[str, Any] | None = None,
    input_kind: str = "jsonl",
) -> dict[str, Any] | None:
    person_tokens = ego4d_person_tokens(narration)
    verb, nouns, noun_source = extract_caption_object_nouns(
        narration,
        object_nouns_allowlist=object_nouns,
        extra_stopwords=person_tokens,
    )
    if not nouns:
        return None

    window_start, window_end, window_duration = centered_clip_window(
        anchor_sec,
        clip_window_sec,
    )
    local_anchor = max(0.0, anchor_sec - window_start)
    record = dict(base_record or {})

    return {
        **record,
        "id": record.get("id") or clip_id,
        "clip_id": clip_id,
        "video_id": video_id,
        "source_anchor_sec": anchor_sec,
        "source_window_start_sec": window_start,
        "source_window_end_sec": window_end,
        "source_window_duration_sec": window_duration,
        "start_sec": 0.0,
        "end_sec": window_duration,
        "t_minus_2_sec": max(0.0, local_anchor - 2.0),
        "t_minus_1_sec": max(0.0, local_anchor - 1.0),
        "t_sec": local_anchor,
        "dense_caption": narration,
        "dense_caption_en": narration,
        "narration": narration,
        "verb": verb or record.get("verb"),
        "nouns": nouns,
        "noun_candidates": noun_source.get("spacy_noun_candidates") or [],
        "source": {
            **(record.get("source") or {}),
            **noun_source,
            "input_kind": input_kind,
        },
        "source_dataset": record.get("source_dataset") or dataset_id,
        "dataset": record.get("dataset") or record.get("source_dataset") or dataset_id,
    }


def _parse_clock_code_to_seconds(value: str) -> float | None:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 6:
        return None
    hh = int(digits[0:2])
    mm = int(digits[2:4])
    ss = int(digits[4:6])
    frac = float(f"0.{digits[6:]}") if len(digits) > 6 else 0.0
    return hh * 3600 + mm * 60 + ss + frac


def _egolife_timecode_to_seconds(stem: str) -> float | None:
    parts = stem.split("_")
    if not parts:
        return None
    return _parse_clock_code_to_seconds(parts[-1])


def _egolife_caption_file_start_sec(clip_id: str, participant: str, day: str) -> float | None:
    for prefix in (f"{participant}_{day}_", f"{day}_{participant}_"):
        if clip_id.startswith(prefix):
            return _parse_clock_code_to_seconds(clip_id.removeprefix(prefix))
    return None


def _egolife_video_stem_candidates(clip_id: str, participant: str, day: str) -> list[str]:
    stems: list[str] = []
    if clip_id:
        stems.append(clip_id)
    if participant and day and clip_id.startswith(f"{participant}_{day}_"):
        suffix = clip_id.removeprefix(f"{participant}_{day}_")
        stems.append(f"{day}_{participant}_{suffix}")
    if participant and day and clip_id.startswith(f"{day}_{participant}_"):
        suffix = clip_id.removeprefix(f"{day}_{participant}_")
        stems.append(f"{participant}_{day}_{suffix}")
    deduped: list[str] = []
    for stem in stems:
        if stem and stem not in deduped:
            deduped.append(stem)
    return deduped


def _resolve_egolife_video_by_time(
    videos_root: Path,
    clip_id: str,
    participant: str,
    day: str,
    record: dict[str, Any],
) -> tuple[Path, float, float] | None:
    if not participant or not day or not clip_id:
        return None
    caption_start = _egolife_caption_file_start_sec(clip_id, participant, day)
    if caption_start is None:
        return None
    absolute_start = caption_start + float(record.get("start_sec") or 0.0)
    video_dir = videos_root / participant / day
    if not video_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for path in sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.MP4")):
        start = _egolife_timecode_to_seconds(path.stem)
        if start is not None and start <= absolute_start:
            candidates.append((start, path))
    if not candidates:
        return None
    video_start, video_path = max(candidates, key=lambda item: item[0])
    return video_path, max(0.0, absolute_start - video_start), video_start


def resolve_egolife_video_segment(
    videos_root: Path, record: dict[str, Any]
) -> tuple[Path, float, float] | None:
    clip_id = str(record.get("clip_id") or record.get("video_id") or "")
    participant = record.get("participant")
    day = record.get("day")
    segment = _resolve_egolife_video_by_time(
        videos_root, clip_id, str(participant or ""), str(day or ""), record
    )
    if segment is not None:
        return segment
    video_stems = _egolife_video_stem_candidates(
        clip_id, str(participant or ""), str(day or "")
    )
    candidates: list[Path] = []
    for stem in video_stems:
        if participant and day:
            candidates.append(videos_root / str(participant) / str(day) / f"{stem}.mp4")
            candidates.append(videos_root / str(participant) / str(day) / f"{stem}.MP4")
        candidates.append(videos_root / f"{stem}.mp4")
        candidates.append(videos_root / f"{stem}.MP4")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate, float(record.get("start_sec") or 0.0), _egolife_timecode_to_seconds(candidate.stem)
    return None


def resolve_egolife_video_path(videos_root: Path, record: dict[str, Any]) -> Path | None:
    resolved = resolve_egolife_video_segment(videos_root, record)
    return resolved[0] if resolved else None


_DATASET_ADAPTERS: dict[str, CatVDatasetAdapter] = {
    "egolife": EgoLifeCatVAdapter(),
    "generic": GenericJsonlAdapter(),
}

# User-registered adapters (via register_catv_dataset_adapter) take precedence
# over the built-in ones so that callers can override or extend the registry
# without modifying this module.
_REGISTERED_ADAPTERS: dict[str, CatVDatasetAdapter] = {}


def register_catv_dataset_adapter(name: str, adapter: CatVDatasetAdapter) -> None:
    """Register a custom dataset adapter accessible via --dataset <name>.

    Example::

        from egoownership.catv_datasets import register_catv_dataset_adapter

        class MyDatasetAdapter:
            dataset_id = "mydataset"
            def iter_caption_records(self, path, *, object_nouns=None): ...
            def resolve_video_segment(self, videos_root, record): ...
            def storage_parts(self, record): ...

        register_catv_dataset_adapter("mydataset", MyDatasetAdapter())
    """
    _REGISTERED_ADAPTERS[normalize_dataset_id(name)] = adapter


def get_catv_dataset_adapter(
    dataset: str,
    *,
    ego4d_clip_window_sec: float = DEFAULT_CLIP_WINDOW_SEC,
    ego4d_auto_download: bool = True,
    ego4d_require_observer: bool = True,
) -> CatVDatasetAdapter:
    key = normalize_dataset_id(dataset)
    if key == "ego4d_fho":
        return Ego4DCatVAdapter(
            clip_window_sec=ego4d_clip_window_sec,
            auto_download=ego4d_auto_download,
            require_observer=ego4d_require_observer,
        )
    # User-registered adapters override built-ins.
    adapter = _REGISTERED_ADAPTERS.get(key) or _DATASET_ADAPTERS.get(key)
    if adapter is None:
        supported = ", ".join(sorted({*_DATASET_ADAPTERS, *_REGISTERED_ADAPTERS, "ego4d_fho"}))
        raise ValueError(f"Unsupported CAT-V dataset {dataset!r}. Supported: {supported}")
    return adapter
