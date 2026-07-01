"""Extract tabletop object nouns from caption / narration text."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from egoownership.config import normalize_token
from egoownership.catv_io import normalize_object_noun
from egoownership.egolife_annotations import _caption_mentions_table
from egoownership.filters import _iter_narration_json_entries
from egoownership.narration_parse import (
    _collect_noun_lemmas,
    _collect_verb_lemmas,
    _dedupe_sorted,
    _load_spacy_model,
    extract_spacy_candidates,
    preprocess_narration,
)

_GENERIC_OR_NON_OBJECT_NOUNS = frozenset(
    {
        "all",
        "alice",
        "anybody",
        "anyone",
        "both",
        "camera",
        "dining",
        "disposable",
        "ego",
        "everybody",
        "everyone",
        "girl",
        "hand",
        "hands",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "it",
        "its",
        "jake",
        "katrina",
        "lady",
        "left",
        "lucia",
        "man",
        "me",
        "mine",
        "object",
        "objects",
        "one",
        "people",
        "person",
        "right",
        "scene",
        "she",
        "shure",
        "somebody",
        "someone",
        "table",
        "tablemat",
        "tables",
        "tasha",
        "that",
        "theirs",
        "them",
        "these",
        "they",
        "thing",
        "things",
        "this",
        "those",
        "us",
        "water",
        "we",
        "woman",
        "you",
        "your",
        "yours",
    }
)


def canonical_object_noun(noun: str) -> str:
    normalized = normalize_token(noun)
    if len(normalized) > 3 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if (
        len(normalized) > 3
        and normalized.endswith("s")
        and not normalized.endswith(("ss", "us"))
    ):
        return normalized[:-1]
    return normalized


def object_nouns_from_spacy_candidates(
    noun_candidates: list[str] | tuple[str, ...],
    *,
    object_nouns_allowlist: set[str] | None = None,
    extra_stopwords: set[str] | None = None,
) -> list[str]:
    stopwords = set(_GENERIC_OR_NON_OBJECT_NOUNS)
    if extra_stopwords:
        stopwords.update(extra_stopwords)

    objects: list[str] = []
    seen: set[str] = set()
    for raw in noun_candidates:
        normalized = normalize_token(raw)
        if not normalized or normalized in stopwords:
            continue
        noun = canonical_object_noun(raw)
        if not noun or noun in stopwords:
            continue
        if object_nouns_allowlist is not None:
            key = normalize_object_noun(noun)
            if key not in object_nouns_allowlist:
                continue
        if noun in seen:
            continue
        objects.append(noun)
        seen.add(noun)
    return objects


def extract_caption_object_nouns(
    caption: str,
    *,
    object_nouns_allowlist: set[str] | None = None,
    extra_stopwords: set[str] | None = None,
) -> tuple[str | None, list[str], dict[str, Any]]:
    """Parse a caption and return verb + filtered object nouns from spaCy candidates."""
    candidates = extract_spacy_candidates(caption)
    verbs = list(candidates.verb_candidates)
    nouns = object_nouns_from_spacy_candidates(
        candidates.noun_candidates,
        object_nouns_allowlist=object_nouns_allowlist,
        extra_stopwords=extra_stopwords,
    )
    metadata = {
        "narration_parse": "spacy",
        "spacy_noun_candidates": list(candidates.noun_candidates),
        "spacy_object_noun_candidates": nouns,
        "spacy_verb_candidates": verbs,
    }
    return (verbs[0] if verbs else None), nouns, metadata


def merge_noun_lists(*lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for items in lists:
        for noun in items:
            key = normalize_object_noun(noun)
            if not key or key in seen:
                continue
            merged.append(noun)
            seen.add(key)
    return merged


def ego4d_person_tokens(narration: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(
        r"\b(?:man|woman|lady|girl|person|boy)\s+([a-z])\b",
        narration,
        flags=re.IGNORECASE,
    ):
        tokens.add(match.group(1).casefold())
    return tokens


def _ego4d_narration_example_id(video_uid: str, entry: dict[str, Any]) -> str:
    ann_uid = str(entry.get("annotation_uid") or "na")
    raw_ts = entry.get("timestamp_sec")
    if raw_ts is None:
        raw_ts = entry.get("_unmapped_timestamp_sec") or 0
    return f"{video_uid}:{ann_uid}:{raw_ts}"


def _iter_ego4d_table_narrations(
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
        if not _caption_mentions_table(narration):
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
            "id": _ego4d_narration_example_id(video_uid, entry),
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
    narration_iter = _iter_ego4d_table_narrations(
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
