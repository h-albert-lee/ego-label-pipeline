"""Extract tabletop object nouns from caption / narration text."""

from __future__ import annotations

from typing import Any

from egoownership.config import normalize_object_noun, normalize_token
from egoownership.narration_parse import (
    _dedupe_sorted,
    extract_spacy_candidates,
)

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

_GENERIC_OR_NON_OBJECT_NOUNS = frozenset(
    {
        "all",
        "alice",
        "anybody",
        "anyone",
        "both",
        "camera",
        "chair",
        "chairs",
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


def caption_mentions_table(caption: str) -> bool:
    lowered = caption.casefold()
    return any(term.casefold() in lowered for term in TABLE_CAPTION_TERMS)


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

