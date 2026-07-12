"""Ego4D video lookup, scratch download, and centered subclip extraction."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Iterator
import subprocess
from pathlib import Path
from typing import Any

from egoownership.config import safe_path_part
from egoownership.narration_parse import (
    _collect_noun_lemmas,
    _collect_verb_lemmas,
    _dedupe_sorted,
    _load_spacy_model,
    preprocess_narration,
)
from egoownership.tabletop_nouns import (
    caption_mentions_table,
    object_nouns_from_spacy_candidates,
)

DEFAULT_CLIP_WINDOW_SEC = 30.0


def _iter_narration_json_entries(path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return
    for video_uid, payload in data.items():
        if not isinstance(video_uid, str) or not isinstance(payload, dict):
            continue
        pass_keys = sorted(
            k for k in payload if isinstance(k, str) and k.startswith("narration_pass_")
        )
        for pass_key in pass_keys:
            block = payload.get(pass_key) or {}
            if not isinstance(block, dict):
                continue
            narrations = block.get("narrations") or []
            if not isinstance(narrations, list):
                continue
            for item in narrations:
                if isinstance(item, dict):
                    yield video_uid, item


def default_scratch_root() -> Path:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "egoown"
    return Path("/scratch") / user / "ego4d"


def centered_clip_window(anchor_sec: float, window_sec: float = DEFAULT_CLIP_WINDOW_SEC) -> tuple[float, float, float]:
    """Return ``(window_start, window_end, duration)`` centered on ``anchor_sec``."""
    half = max(0.0, window_sec) / 2.0
    start = max(0.0, float(anchor_sec) - half)
    duration = max(0.0, float(window_sec))
    return start, start + duration, duration


def ego4d_narration_example_id(video_uid: str, entry: dict[str, Any]) -> str:
    ann_uid = str(entry.get("annotation_uid") or "na")
    raw_ts = entry.get("timestamp_sec")
    if raw_ts is None:
        raw_ts = entry.get("_unmapped_timestamp_sec") or 0
    return f"{video_uid}:{ann_uid}:{raw_ts}"


def ego4d_person_tokens(narration: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(
        r"\b(?:man|woman|lady|girl|person|boy)\s+([a-z])\b",
        narration,
        flags=re.IGNORECASE,
    ):
        tokens.add(match.group(1).casefold())
    return tokens


def iter_ego4d_table_narrations(
    narration_path: Path,
    *,
    require_observer: bool = False,
    limit: int | None = None,
) -> Iterator[tuple[str, dict[str, Any], str]]:
    count = 0
    for video_uid, entry in _iter_narration_json_entries(narration_path):
        if limit is not None and count >= limit:
            return
        narration = str(entry.get("narration_text") or "").strip()
        if not narration:
            continue
        if not caption_mentions_table(narration):
            continue
        if require_observer and "#O" not in narration:
            continue
        yield video_uid, entry, narration
        count += 1


def _process_ego4d_narration_batch(
    batch: list[tuple[str, dict[str, Any], str]],
    *,
    noun_counts: Counter[str],
    noun_examples: dict[str, dict[str, Any]],
) -> int:
    if not batch:
        return 0

    nlp = _load_spacy_model()
    texts = [preprocess_narration(narration) for _video_uid, _entry, narration in batch]
    processed = 0
    for (video_uid, entry, narration), doc in zip(batch, nlp.pipe(texts, batch_size=256)):
        verb_lemmas = _collect_verb_lemmas(doc)
        noun_candidates = _dedupe_sorted(_collect_noun_lemmas(doc, verb_lemmas))
        nouns = object_nouns_from_spacy_candidates(
            noun_candidates,
            extra_stopwords=ego4d_person_tokens(narration),
        )
        if not nouns:
            continue
        processed += 1
        example = {
            "id": ego4d_narration_example_id(video_uid, entry),
            "video_id": video_uid,
            "narration": narration,
            "timestamp_sec": entry.get("timestamp_sec"),
            "annotation_uid": entry.get("annotation_uid"),
        }
        for noun in nouns:
            noun_counts[noun] += 1
            if noun not in noun_examples:
                noun_examples[noun] = example
    return processed


def write_ego4d_table_caption_object_nouns(
    narration_path: Path,
    out_path: Path,
    *,
    require_observer: bool = False,
    batch_size: int = 2048,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[int, int]:
    """Mine object nouns from Ego4D table narrations and write an allowlist JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    noun_counts: Counter[str] = Counter()
    noun_examples: dict[str, dict[str, Any]] = {}
    narration_iter = iter_ego4d_table_narrations(
        narration_path,
        require_observer=require_observer,
        limit=limit,
    )
    if show_progress:
        from tqdm.auto import tqdm

        narration_iter = tqdm(
            narration_iter,
            unit="narration",
            desc="Ego4D table narration nouns",
        )

    batch: list[tuple[str, dict[str, Any], str]] = []
    narrations_with_nouns = 0
    for row in narration_iter:
        batch.append(row)
        if len(batch) < batch_size:
            continue
        narrations_with_nouns += _process_ego4d_narration_batch(
            batch,
            noun_counts=noun_counts,
            noun_examples=noun_examples,
        )
        batch.clear()
    narrations_with_nouns += _process_ego4d_narration_batch(
        batch,
        noun_counts=noun_counts,
        noun_examples=noun_examples,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for rank, (noun, count) in enumerate(noun_counts.most_common(), start=1):
            row = {
                "rank": rank,
                "noun": noun,
                "count": count,
                "example": noun_examples.get(noun, {}),
                "keep_object_noun": True,
                "category": "object_noun",
                "filter_reason": "allowlist_object",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return narrations_with_nouns, len(noun_counts)


def locate_ego4d_full_video(video_id: str, *roots: Path) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        candidates.append(root / f"{video_id}.mp4")
        candidates.extend(sorted(root.glob(f"{video_id}.mp4*")))
        nested = root / "v2" / "full_scale" / f"{video_id}.mp4"
        candidates.append(nested)
        nested_root = root / "v2" / "full_scale"
        if nested_root.exists():
            candidates.extend(sorted(nested_root.glob(f"{video_id}.mp4*")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def download_ego4d_video(video_uid: str, download_dir: Path) -> Path | None:
    """Download one Ego4D full-scale video via the official ``ego4d`` CLI."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Downloading Ego4D video to scratch: {video_uid} → {download_dir}")
    try:
        subprocess.run(
            [
                "ego4d",
                "--output_directory",
                str(download_dir),
                "--datasets",
                "full_scale",
                "--video_uids",
                video_uid,
                "--yes",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[error] Failed to download {video_uid}. CLI error: {exc.stderr}")
        return None
    except FileNotFoundError:
        print("[error] 'ego4d' command not found. Install with: pip install ego4d")
        return None
    return locate_ego4d_full_video(video_uid, download_dir)


def ensure_ego4d_subclip(
    full_video: Path,
    cache_path: Path,
    *,
    window_start_sec: float,
    window_duration_sec: float,
) -> Path | None:
    """Extract and cache a fixed-duration subclip with ffmpeg."""
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, window_start_sec):.3f}",
        "-i",
        str(full_video),
        "-t",
        f"{max(0.1, window_duration_sec):.3f}",
        "-c",
        "copy",
        "-y",
        str(cache_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return None
    return cache_path if cache_path.exists() and cache_path.stat().st_size > 0 else None


def ego4d_subclip_cache_path(
    scratch_root: Path,
    *,
    video_id: str,
    clip_id: str,
    window_start_sec: float,
) -> Path:
    clip_key = safe_path_part(clip_id or video_id)
    video_key = safe_path_part(video_id)
    return (
        Path(scratch_root)
        / "clips"
        / video_key
        / f"{clip_key}__{max(0.0, window_start_sec):.3f}.mp4"
    )
