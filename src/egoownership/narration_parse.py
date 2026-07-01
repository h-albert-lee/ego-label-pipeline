"""spaCy candidate extraction + OpenAI selection for narration filtering.

Pipeline per narration line:

1. ``extract_spacy_candidates`` — noun chunks and verb lemmas (deterministic).
2. ``OpenAINarrationParser.parse`` — pick primary object/verb and taxonomy A/B/C/D.
3. ``map_to_shared_table_noun`` — map the chosen object onto ``taxonomy.yaml`` nouns.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from egoownership.config import TaxonomyConfig, normalize_token
from egoownership.schema import Taxonomy

# Scene furniture / body parts that narrations mention constantly but are not target objects.
_NOUN_STOPWORDS = frozenset({"hand", "hands", "table", "tables"})

# Auxiliary / copular verbs — not useful as the main action verb for taxonomy.
_VERB_BLOCKLIST = frozenset(
    {
        "be",
        "have",
        "do",
        "is",
        "are",
        "was",
        "were",
        "seem",
        "become",
    }
)

_HASHTAG_RE = re.compile(r"#\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SpacyCandidates:
    noun_candidates: tuple[str, ...]
    verb_candidates: tuple[str, ...]


@dataclass(frozen=True)
class ParsedNarration:
    primary_object: str | None
    primary_verb: str | None
    taxonomy: Taxonomy
    canonical_noun: str | None
    rationale: str | None = None
    spacy_nouns: tuple[str, ...] = ()
    spacy_verbs: tuple[str, ...] = ()


def preprocess_narration(text: str) -> str:
    """Strip Ego4D hashtag markers before NLP."""
    cleaned = _HASHTAG_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def _is_person_marker_token(tok: str) -> bool:
    """Ego4D person ids (B, X, C, O) are usually one character after normalize."""
    return len(tok) <= 1


def _keep_noun_candidate(tok: str) -> bool:
    return bool(tok) and not _is_person_marker_token(tok) and tok not in _NOUN_STOPWORDS


def _token_is_verb_like(tok: Any) -> bool:
    """True when spaCy treats the token as a verb (POS or Penn Treebank tag)."""
    return tok.pos_ == "VERB" or str(tok.tag_).startswith("VB")


def _token_is_noun_like(tok: Any) -> bool:
    """Noun-like token that is not tagged as a verb."""
    if _token_is_verb_like(tok):
        return False
    return tok.pos_ in ("NOUN", "PROPN")


def _misclassified_root_verb(tok: Any) -> bool:
    """Catch ROOT tokens like ``moves`` mistagged as NOUN but governing an object."""
    if tok.dep_ not in ("ROOT", "conj"):
        return False
    if _token_is_verb_like(tok):
        return False
    has_object = any(child.dep_ in ("dobj", "attr", "prep", "prt", "obj") for child in tok.children)
    if not has_object:
        return False
    lemma = normalize_token(tok.lemma_)
    return bool(lemma) and lemma not in _VERB_BLOCKLIST and lemma not in _NOUN_STOPWORDS


def _verb_lemma_from_token(tok: Any) -> str | None:
    if not (_token_is_verb_like(tok) or _misclassified_root_verb(tok)):
        return None
    lemma = normalize_token(tok.lemma_)
    if not lemma or lemma in _VERB_BLOCKLIST:
        return None
    return lemma


def _dedupe_sorted(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        tok = normalize_token(raw)
        if not _keep_noun_candidate(tok) or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return tuple(sorted(out))


@lru_cache(maxsize=2)
def _load_spacy_model():
    import spacy

    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            return spacy.load(name)
        except OSError:
            continue
    raise OSError(
        "No English spaCy model found. Install one, e.g.\n"
        "  python -m spacy download en_core_web_sm"
    )


def _collect_verb_lemmas(doc: Any) -> set[str]:
    """Verb lemmas from POS/tag, ROOT/conj chain, and mis-tagged action roots."""
    lemmas: set[str] = set()
    for tok in doc:
        lemma = _verb_lemma_from_token(tok)
        if lemma:
            lemmas.add(lemma)
    for tok in doc:
        if tok.dep_ != "conj" or tok.head is None:
            continue
        head_lemma = _verb_lemma_from_token(tok.head)
        lemma = _verb_lemma_from_token(tok)
        if head_lemma and lemma:
            lemmas.add(lemma)
    return lemmas


def _collect_noun_lemmas(doc: Any, verb_lemmas: set[str]) -> list[str]:
    """Noun lemmas excluding verb-tagged tokens and lemmas already used as verbs."""
    nouns: list[str] = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        if _token_is_verb_like(root) or _misclassified_root_verb(root):
            continue
        head = normalize_token(root.lemma_)
        if _keep_noun_candidate(head) and head not in verb_lemmas:
            nouns.append(head)
        phrase = normalize_token(chunk.text)
        if phrase and phrase != head:
            parts = [p for p in phrase.split("_") if p]
            tail = parts[-1] if parts else ""
            if _keep_noun_candidate(tail) and tail not in verb_lemmas:
                nouns.append(tail)

    for tok in doc:
        if not _token_is_noun_like(tok):
            continue
        lemma = normalize_token(tok.lemma_)
        if _keep_noun_candidate(lemma) and lemma not in verb_lemmas:
            nouns.append(lemma)
    return nouns


def extract_spacy_candidates(text: str) -> SpacyCandidates:
    """Extract noun/verb candidate lists from narration text using spaCy."""
    nlp = _load_spacy_model()
    doc = nlp(preprocess_narration(text))

    verb_lemmas = _collect_verb_lemmas(doc)
    nouns = _collect_noun_lemmas(doc, verb_lemmas)

    return SpacyCandidates(
        noun_candidates=_dedupe_sorted(nouns),
        verb_candidates=_dedupe_sorted(verb_lemmas),
    )


def map_to_shared_table_noun(object_label: str | None, cfg: TaxonomyConfig) -> str | None:
    """Map a free-text object label to a canonical ``shared_table_nouns`` entry."""
    if not object_label:
        return None
    n = normalize_token(object_label)
    if n in cfg.shared_table_nouns:
        return n
    for sn in sorted(cfg.shared_table_nouns, key=len, reverse=True):
        parts = [p for p in sn.split("_") if p]
        if n == sn or all(p in n.split("_") for p in parts):
            return sn
        if sn in n or n in sn:
            return sn
    return None


_SYS_NARRATION = (
    "You annotate egocentric narrations for an implicit object-ownership benchmark.\n"
    "Given noun and verb candidate lists extracted by spaCy, choose:\n"
    "- primary_object: the physical object whose ownership matters (from noun_candidates, or null)\n"
    "- primary_verb: the main action on that object (from verb_candidates, or null)\n"
    "- taxonomy: exactly one of A, B, C, D (scene category):\n"
    "  A (Baseline): object at rest or static arrangement; single-frame ownership cues suffice.\n"
    "  B (Conflict): narration implies visual contact/location cues would MISLEAD ownership "
    "(e.g. hand on another person's object, ambiguous whose item is touched).\n"
    "  C (Contextual): ownership may change over time — give, pass, put down, take, pick up, "
    "hand over, move to/from person or shared table in an ownership-relevant way.\n"
    "  D (Ambiguous): multiple similar objects, unclear referent, symmetric layout, or "
    "insufficient cues (#unsure, 'both', 'two' cups, etc.).\n"
    "- canonical_noun: optional — if primary_object matches a shared_table_nouns entry, "
    "return that exact string; otherwise null (primary_object is still returned)\n"
    "Prefer manipulated objects over body parts (hand) or furniture surfaces (table) "
    "when other candidates exist.\n"
    "Return JSON only with keys: primary_object, primary_verb, taxonomy, "
    "canonical_noun, rationale."
)

_SYS_NARRATION_BATCH = (
    _SYS_NARRATION
    + "\n\nYou receive a JSON object with shared_table_nouns and an items array. "
    "Each item has id, narration, noun_candidates, verb_candidates. "
    "Return JSON only: {\"results\": [{\"id\": \"...\", \"primary_object\": ..., "
    "\"primary_verb\": ..., \"taxonomy\": \"A|B|C|D\", \"canonical_noun\": ..., "
    "\"rationale\": ...}, ...]} with one result per input id, same order as items."
)


def parse_llm_taxonomy(raw: Any) -> Taxonomy | None:
    """Normalize LLM taxonomy field to ``Taxonomy`` (supports legacy interaction_type)."""
    if raw is None:
        return None
    key = str(raw).strip().upper()
    aliases = {
        "A": Taxonomy.BASELINE,
        "BASELINE": Taxonomy.BASELINE,
        "B": Taxonomy.CONFLICT,
        "CONFLICT": Taxonomy.CONFLICT,
        "C": Taxonomy.CONTEXTUAL,
        "CONTEXTUAL": Taxonomy.CONTEXTUAL,
        "D": Taxonomy.AMBIGUOUS,
        "AMBIGUOUS": Taxonomy.AMBIGUOUS,
        # legacy interaction_type values
        "CONTEXTUAL_VERB": Taxonomy.CONTEXTUAL,
        "BASELINE_VERB": Taxonomy.BASELINE,
    }
    if key in aliases:
        return aliases[key]
    legacy = str(raw).strip().lower()
    if legacy == "contextual":
        return Taxonomy.CONTEXTUAL
    if legacy == "baseline":
        return Taxonomy.BASELINE
    if legacy in ("unknown", "ambiguous"):
        return Taxonomy.AMBIGUOUS
    return None

_DEFAULT_OPENAI_MODEL = os.environ.get("EGOOWN_NARRATION_OPENAI_MODEL", "gpt-4.1-nano")


@dataclass(frozen=True)
class NarrationParseRequest:
    """One narration line ready for batched LLM disambiguation."""

    request_id: str
    narration: str
    candidates: SpacyCandidates


@dataclass
class OpenAINarrationParserConfig:
    model: str = _DEFAULT_OPENAI_MODEL
    api_key: str | None = None
    max_tokens: int = 400
    max_tokens_batch: int = 4096
    temperature: float = 0.0
    batch_size: int = 20


class OpenAINarrationParser:
    """Call OpenAI chat completions to disambiguate spaCy candidates."""

    def __init__(self, cfg: OpenAINarrationParserConfig | None = None):
        self.cfg = cfg or OpenAINarrationParserConfig()

    def _client(self) -> Any:
        from openai import OpenAI

        key = self.cfg.api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAI(api_key=key) if key else OpenAI()

    def _build_parsed(
        self,
        payload: dict[str, Any],
        candidates: SpacyCandidates,
        shared: list[str],
    ) -> ParsedNarration:
        primary_object = _norm_optional(payload.get("primary_object"))
        primary_verb = _norm_optional(payload.get("primary_verb"))
        taxonomy = parse_llm_taxonomy(payload.get("taxonomy"))
        if taxonomy is None:
            taxonomy = parse_llm_taxonomy(payload.get("interaction_type"))
        if taxonomy is None:
            taxonomy = Taxonomy.AMBIGUOUS

        canonical = _norm_optional(payload.get("canonical_noun"))
        if canonical and canonical not in shared:
            canonical = map_to_shared_table_noun(canonical, _cfg_from_shared(shared))
        if canonical is None and primary_object:
            canonical = map_to_shared_table_noun(primary_object, _cfg_from_shared(shared))

        return ParsedNarration(
            primary_object=primary_object,
            primary_verb=primary_verb,
            taxonomy=taxonomy,
            canonical_noun=canonical,
            rationale=_norm_optional(payload.get("rationale")),
            spacy_nouns=candidates.noun_candidates,
            spacy_verbs=candidates.verb_candidates,
        )

    def parse(
        self,
        narration: str,
        candidates: SpacyCandidates,
        shared_table_nouns: frozenset[str] | list[str],
    ) -> ParsedNarration:
        results = self.parse_batch(
            [NarrationParseRequest(request_id="0", narration=narration, candidates=candidates)],
            shared_table_nouns,
        )
        return results["0"]

    def parse_batch(
        self,
        requests: list[NarrationParseRequest],
        shared_table_nouns: frozenset[str] | list[str],
    ) -> dict[str, ParsedNarration]:
        """Parse many narrations in one API call per batch (size ``cfg.batch_size``)."""
        if not requests:
            return {}

        shared = sorted(shared_table_nouns)
        by_id = {r.request_id: r for r in requests}
        out: dict[str, ParsedNarration] = {}

        for start in range(0, len(requests), self.cfg.batch_size):
            chunk = requests[start : start + self.cfg.batch_size]
            if len(chunk) == 1:
                req = chunk[0]
                user_payload = {
                    "narration": req.narration,
                    "noun_candidates": list(req.candidates.noun_candidates),
                    "verb_candidates": list(req.candidates.verb_candidates),
                    "shared_table_nouns": shared,
                }
                system = _SYS_NARRATION
            else:
                user_payload = {
                    "shared_table_nouns": shared,
                    "items": [
                        {
                            "id": req.request_id,
                            "narration": req.narration,
                            "noun_candidates": list(req.candidates.noun_candidates),
                            "verb_candidates": list(req.candidates.verb_candidates),
                        }
                        for req in chunk
                    ],
                }
                system = _SYS_NARRATION_BATCH

            client = self._client()
            max_tokens = self.cfg.max_tokens
            if len(chunk) > 1:
                max_tokens = min(
                    self.cfg.max_tokens_batch,
                    max(self.cfg.max_tokens, self.cfg.max_tokens * len(chunk)),
                )
            resp = client.chat.completions.create(
                model=self.cfg.model,
                max_tokens=max_tokens,
                temperature=self.cfg.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
            )
            text = resp.choices[0].message.content or "{}"
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {}

            if len(chunk) == 1:
                rows = [payload]
                row_ids = [chunk[0].request_id]
            else:
                rows = payload.get("results") or payload.get("items") or []

            parsed_by_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("id", ""))
                if rid:
                    parsed_by_id[rid] = row
            if len(chunk) == 1 and chunk[0].request_id not in parsed_by_id:
                parsed_by_id[chunk[0].request_id] = payload

            for req in chunk:
                row = parsed_by_id.get(req.request_id, {})
                out[req.request_id] = self._build_parsed(row, req.candidates, shared)

        return out


def _norm_optional(value: Any) -> str | None:
    if value is None:
        return None
    s = normalize_token(str(value))
    return s or None


def _cfg_from_shared(shared: list[str]) -> TaxonomyConfig:
    from egoownership.config import OwnershipZones

    return TaxonomyConfig(
        contextual_verbs=frozenset(),
        baseline_verbs=frozenset(),
        shared_table_nouns=frozenset(shared),
        zones=OwnershipZones(
            mine_near_y_min=0.55,
            shared_x_min=0.3,
            shared_x_max=0.7,
            person_far_y_max=0.55,
            min_bbox_area_ratio=0.003,
        ),
    )


def parse_narration_with_llm(
    narration: str,
    cfg: TaxonomyConfig,
    *,
    parser: OpenAINarrationParser | None = None,
) -> ParsedNarration | None:
    """Run spaCy extraction then OpenAI disambiguation. Returns None if spaCy finds nothing."""
    candidates = extract_spacy_candidates(narration)
    if not candidates.noun_candidates and not candidates.verb_candidates:
        return None
    llm = parser or OpenAINarrationParser()
    return llm.parse(narration, candidates, cfg.shared_table_nouns)
