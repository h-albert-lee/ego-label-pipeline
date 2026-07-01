"""Taxonomy-aware filtering of clip candidates.

Rules implement the strategy in §3 of the EDA doc:

* **Taxonomy C (Contextual)** — verb must be in `contextual_verbs`. Noun must
  intersect `shared_table_nouns` when the knob is on. Favors dining / meeting.
* **Taxonomy A (Baseline)** — verb empty OR in `baseline_verbs`; nouns still
  must intersect the shared-table list so we stay on the benchmark surface.
* **Taxonomy D (Ambiguous)** — accepted downstream only (needs detection
  evidence). For filtering purposes we pass everything through with taxonomy=D
  when the caller explicitly asks for D.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from functools import lru_cache
from pathlib import Path

from egoownership.config import TaxonomyConfig, load_config, normalize_token
from egoownership.schema import ClipCandidate, Taxonomy


MomentEntry = tuple[float, float, str]
_AMBIGUOUS_LABEL_TOKENS = frozenset({"another", "both", "pair", "same", "two"})


def _label_vocab(label: str) -> set[str]:
    normalized = normalize_token(label.replace("/", " "))
    parts = [part for part in normalized.split("_") if part]
    vocab = set(parts)
    for part in parts:
        if len(part) > 3 and part.endswith("es"):
            vocab.add(part[:-2])
        elif len(part) > 2 and part.endswith("s"):
            vocab.add(part[:-1])
    vocab.add(normalized)
    return vocab


def _match_label_tokens(label: str, lexicon: frozenset[str]) -> set[str]:
    vocab = _label_vocab(label)
    hits: set[str] = set()
    for token in lexicon:
        token_parts = [part for part in token.split("_") if part]
        if token in vocab or all(part in vocab for part in token_parts):
            hits.add(token)
    return hits


@lru_cache(maxsize=4)
def load_moment_index(path: str | Path) -> dict[str, tuple[MomentEntry, ...]]:
    """Load Ego4D moments annotations into a per-video primary-label index."""

    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    by_video: dict[str, list[MomentEntry]] = {}
    for video in raw.get("videos", []) or []:
        if not isinstance(video, dict):
            continue
        video_uid = video.get("video_uid") or video.get("video_id")
        if not isinstance(video_uid, str) or not video_uid:
            continue
        labels: list[MomentEntry] = []
        for clip in video.get("clips", []) or []:
            if not isinstance(clip, dict):
                continue
            for annotation in clip.get("annotations", []) or []:
                if not isinstance(annotation, dict):
                    continue
                for item in annotation.get("labels", []) or []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("primary") is False:
                        continue
                    start = item.get("video_start_time")
                    end = item.get("video_end_time")
                    label = item.get("label")
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and isinstance(label, str):
                        labels.append((float(start), float(end), label))
        if labels:
            by_video[video_uid] = sorted(labels, key=lambda entry: (entry[0], entry[1], entry[2]))
    return {video_uid: tuple(entries) for video_uid, entries in by_video.items()}


def _iter_narration_json_entries(path: str | Path) -> Iterator[tuple[str, dict]]:

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
        for pk in pass_keys:
            block = payload.get(pk) or {}
            if not isinstance(block, dict):
                continue
            narrations = block.get("narrations") or []
            if not isinstance(narrations, list):
                continue
            for item in narrations:
                if isinstance(item, dict):
                    yield video_uid, item


        
def _match_verb_token(tok: str, verbs: frozenset[str]) -> str | None:
    """Map a narration token to a canonical taxonomy verb (handles inflections).

    Longer lexicon entries are tried first so e.g. ``put_down`` wins over ``put``.
    """

    if tok in verbs:
        return tok
    for v in sorted(verbs, key=lambda x: -len(x)):
        if tok.startswith(v) and len(tok) - len(v) <= 3:
            return v
    return None


def _extract_verb_and_nouns_from_narration(text: str, cfg: TaxonomyConfig) -> tuple[str | None, list[str]]:
    norm = normalize_token(text)
    tokens = [t for t in re.findall(r"[a-z0-9]+", norm) if t]

    verb: str | None = None
    for tok in tokens:
        hit = _match_verb_token(tok, cfg.contextual_verbs)
        if hit is not None:
            verb = hit
            break
        hit = _match_verb_token(tok, cfg.baseline_verbs)
        if hit is not None:
            verb = hit
            break

    token_set = set(tokens)
    nouns = sorted(
        noun
        for noun in cfg.shared_table_nouns
        if all(part in token_set for part in noun.split("_"))
    )
    return verb, nouns



def infer_taxonomy_from_moment_label(
    label: str,
    cfg: TaxonomyConfig,
    *,
    require_shared_noun: bool = True,
) -> Taxonomy | None:
    """Infer taxonomy from an Ego4D moments label string."""

    shared_nouns = _match_label_tokens(label, cfg.shared_table_nouns)
    contextual_verbs = _match_label_tokens(label, cfg.contextual_verbs)
    baseline_verbs = _match_label_tokens(label, cfg.baseline_verbs)
    vocab = _label_vocab(label)

    if require_shared_noun and not shared_nouns:
        return None
    if shared_nouns and len(shared_nouns) > 1 and vocab & _AMBIGUOUS_LABEL_TOKENS:
        return Taxonomy.AMBIGUOUS
    if contextual_verbs:
        return Taxonomy.CONTEXTUAL
    if baseline_verbs or shared_nouns:
        return Taxonomy.BASELINE
    return None


def _has_shared_noun(cand: ClipCandidate, cfg: TaxonomyConfig) -> bool:
    if not cand.nouns:
        return False
    return any(n in cfg.shared_table_nouns for n in cand.nouns)


def _uses_llm_narration_parse(cand: ClipCandidate) -> bool:
    return cand.source.get("narration_parse") == "spacy_openai"


def _passes_noun_requirement(
    cand: ClipCandidate,
    cfg: TaxonomyConfig,
    *,
    require_shared_noun: bool,
) -> bool:
    """LLM path: require a primary object in ``nouns``; rules path: shared-table whitelist."""
    if not require_shared_noun:
        return True
    if not cand.nouns:
        return False
    if _uses_llm_narration_parse(cand):
        return True
    return _has_shared_noun(cand, cfg)


def _infer_taxonomy_from_rules(
    verb: str | None,
    nouns: list[str],
    cfg: TaxonomyConfig,
) -> Taxonomy | None:
    """Heuristic A/C inference from verb+nouns when not using LLM (``all`` mode)."""
    if not nouns:
        return None
    if verb and verb in cfg.contextual_verbs:
        return Taxonomy.CONTEXTUAL
    if verb is None or verb in cfg.baseline_verbs:
        return Taxonomy.BASELINE
    # Other action verbs default to contextual-leaning for ego narrations.
    if verb:
        return Taxonomy.CONTEXTUAL
    return Taxonomy.BASELINE


def _resolve_output_taxonomy(
    target: Taxonomy | None,
    parse_source: dict,
    verb: str | None,
    nouns: list[str],
    cfg: TaxonomyConfig,
    *,
    use_llm_parse: bool,
) -> Taxonomy | None:
    """Taxonomy stamped on each candidate; ``target`` overrides when filtering a single bucket."""
    if target is not None:
        return target
    if use_llm_parse:
        from egoownership.narration_parse import parse_llm_taxonomy

        return parse_llm_taxonomy(parse_source.get("llm_taxonomy"))
    return _infer_taxonomy_from_rules(verb, nouns, cfg)


def _llm_predicted_taxonomy(cand: ClipCandidate) -> Taxonomy | None:
    """Return LLM-predicted taxonomy (A/B/C/D) when narration was parsed with OpenAI."""
    if not _uses_llm_narration_parse(cand):
        return None
    from egoownership.narration_parse import parse_llm_taxonomy

    raw = cand.source.get("llm_taxonomy")
    if raw is None:
        raw = cand.source.get("interaction_type")
    tax = parse_llm_taxonomy(raw)
    return tax


def matches_taxonomy(
    cand: ClipCandidate,
    target: Taxonomy,
    cfg: TaxonomyConfig,
    require_shared_noun: bool = True,
) -> bool:
    """Return True iff ``cand`` passes the rule set for ``target`` taxonomy."""

    # B is not inferred from narration (visual vs context conflict needs detection).
    if target is Taxonomy.CONFLICT:
        return True

    llm_tax = _llm_predicted_taxonomy(cand)
    if llm_tax is not None:
        if llm_tax != target:
            return False
        if target is Taxonomy.AMBIGUOUS:
            return bool(cand.nouns)
        return _passes_noun_requirement(cand, cfg, require_shared_noun=require_shared_noun)

    if target is Taxonomy.CONTEXTUAL:
        if cand.verb is None or cand.verb not in cfg.contextual_verbs:
            return False
        if not _passes_noun_requirement(cand, cfg, require_shared_noun=require_shared_noun):
            return False
        return True

    if target is Taxonomy.BASELINE:
        if cand.verb is not None and cand.verb not in cfg.baseline_verbs:
            # Mid-action clips are not Baseline.
            return False
        if not _passes_noun_requirement(cand, cfg, require_shared_noun=require_shared_noun):
            return False
        return True

    if target is Taxonomy.AMBIGUOUS:
        # Purely structural; we accept any candidate that has any noun,
        # since ambiguity is resolved after detection.
        return bool(cand.nouns)

    return False


def filter_candidates(
    cands: Iterable[ClipCandidate],
    target: Taxonomy,
    *,
    config: TaxonomyConfig | None = None,
    require_shared_noun: bool = True,
    limit: int | None = None,
) -> Iterator[ClipCandidate]:
    cfg = config or load_config()
    count = 0
    for cand in cands:
        if limit is not None and count >= limit:
            return
        if matches_taxonomy(cand, target, cfg, require_shared_noun=require_shared_noun):
            # Stamp the taxonomy onto the candidate so downstream stages don't
            # re-derive it. We copy to avoid mutating upstream state.
            stamped = cand.model_copy(update={"taxonomy": target})
            yield stamped
            count += 1

def _metadata_from_parsed(parsed) -> tuple[str | None, list[str], dict]:
    """Turn ``ParsedNarration`` into (verb, nouns, source) for ``ClipCandidate``."""
    if parsed is None or not parsed.primary_object:
        return None, [], {"bboxes": [], "narration_parse": "spacy_openai_skip"}

    extra: dict = {
        "bboxes": [],
        "narration_parse": "spacy_openai",
        "llm_taxonomy": parsed.taxonomy.value,
        "primary_object": parsed.primary_object,
        "canonical_noun": parsed.canonical_noun,
        "on_shared_table": parsed.canonical_noun is not None,
        "spacy_noun_candidates": list(parsed.spacy_nouns),
        "spacy_verb_candidates": list(parsed.spacy_verbs),
        "llm_rationale": parsed.rationale,
    }
    return parsed.primary_verb, [parsed.primary_object], extra


def _metadata_from_narration(
    narration_text: str,
    entry: dict,
    cfg: TaxonomyConfig,
) -> tuple[str | None, list[str], dict]:
    """Return (verb, nouns, extra source fields) using rule-based extraction only."""
    extra: dict = {"bboxes": [], "narration_parse": "rules"}

    structured_verb = entry.get("structured_verb")
    if isinstance(structured_verb, str) and structured_verb.strip():
        verb = normalize_token(structured_verb)
        _, noun_hits = _extract_verb_and_nouns_from_narration(narration_text, cfg)
    else:
        verb, noun_hits = _extract_verb_and_nouns_from_narration(narration_text, cfg)
    return verb, noun_hits, extra


def new_filter(
    narration_path: str | Path,
    target: Taxonomy | None,
    *,
    config: TaxonomyConfig | None = None,
    require_shared_noun: bool = True,
    limit: int | None = None,
    videos_root: str | Path | None = None,
    frame_backend: str = "ffmpeg",
    frames_out_dir: str | Path | None = None,
    florence_describe: bool = False,
    florence_model: str = "microsoft/Florence-2-base",
    florence_device: str | None = None,
    video_resolver=None,
    use_llm_parse: bool = False,
    llm_parser=None,
    llm_batch_size: int | None = None,
    on_llm_batch: Callable[[list[ClipCandidate]], None] | None = None,
    require_table_object_markers: bool = True,
) -> Iterator[ClipCandidate]:
    cfg = config or load_config()
    count = 0

    from egoownership.narration_parse import (
        NarrationParseRequest,
        OpenAINarrationParser,
        extract_spacy_candidates,
    )

    parser = llm_parser or OpenAINarrationParser()
    _parser_cfg = getattr(parser, "cfg", None)
    batch_size = llm_batch_size or (_parser_cfg.batch_size if _parser_cfg else 20)

    # (video_uid, entry, narration_text, candidates, request_id)
    pending_llm: list[tuple[str, dict, str, Any, str]] = []

    def _emit_rows(rows: list[tuple[str, dict, str, str | None, list[str], dict]]) -> Iterator[ClipCandidate]:
        nonlocal count
        for video_uid, entry, narration_text, verb, noun_hits, parse_source in rows:
            if limit is not None and count >= limit:
                return
            if use_llm_parse and parse_source.get("narration_parse") == "spacy_openai_skip":
                continue

            stamped_tax = _resolve_output_taxonomy(
                target,
                parse_source,
                verb,
                noun_hits,
                cfg,
                use_llm_parse=use_llm_parse,
            )
            if stamped_tax is None:
                continue

            raw_ts = entry.get("timestamp_sec")
            if raw_ts is None:
                raw_ts = entry.get("_unmapped_timestamp_sec")
            try:
                action_ts = float(raw_ts if raw_ts is not None else 0.0)
            except (TypeError, ValueError):
                action_ts = 0.0
            interval_start = max(0.0, action_ts - 2.0)
            interval_mid = max(0.0, action_ts - 1.0)
            interval_end = max(0.0, action_ts)

            ann_uid = entry.get("annotation_uid")
            uid_part = ann_uid if isinstance(ann_uid, str) and ann_uid else "na"
            clip_id = f"{video_uid}:{uid_part}:{action_ts:.4f}"

            cand = ClipCandidate(
                dataset="ego4d_fho",
                clip_id=clip_id,
                video_id=video_uid,
                taxonomy=stamped_tax,
                t_minus_2_sec=interval_start,
                t_minus_1_sec=interval_mid,
                t_sec=interval_end,
                verb=verb,
                nouns=noun_hits,
                narration=narration_text,
                source=parse_source,
            )

            video_path: Path | None = None
            if videos_root is not None:
                _candidate = Path(videos_root) / "v2" / "full_scale" / f"{video_uid}.mp4"
                if not _candidate.exists():
                    for _ext in (".MP4", ".mkv", ".webm"):
                        _alt = _candidate.with_suffix(_ext)
                        if _alt.exists():
                            _candidate = _alt
                            break
                    else:
                        if video_resolver is not None:
                            _resolved = video_resolver(video_uid, Path(videos_root))
                            if _resolved is not None:
                                _candidate = _resolved
                if _candidate.exists():
                    video_path = _candidate

            if video_path is not None and frames_out_dir is not None:
                try:
                    from egoownership.frames import extract_sparse_frames
                    frame_paths = extract_sparse_frames(
                        cand, video_path, Path(frames_out_dir), backend=frame_backend
                    )
                    cand = cand.model_copy(update={
                        "source": {
                            **cand.source,
                            "frame_paths": {k: str(v) for k, v in frame_paths.items()},
                        }
                    })
                except Exception as e:
                    print(f"[warn] {cand.clip_id}: Frame extraction failed: {e}")

            if video_path is not None and florence_describe:
                try:
                    import torch
                    from egoownership.detection import florence2 as fz
                    dev = florence_device or ("cuda" if torch.cuda.is_available() else "cpu")
                    pil_imgs = fz.load_pil_frames_at_times(
                        video_path,
                        (cand.t_minus_2_sec, cand.t_minus_1_sec, cand.t_sec),
                        backend=frame_backend,
                    )
                    fz_nouns = fz.extract_object_nouns_from_images(
                        pil_imgs,
                        model_id=str(florence_model),
                        device=str(dev),
                    )
                    merged = sorted(set(noun_hits) | set(fz_nouns))
                    cand = cand.model_copy(
                        update={
                            "nouns": merged,
                            "source": {
                                **cand.source,
                                "florence_nouns": fz_nouns,
                                "florence_model": str(florence_model),
                            },
                        }
                    )
                except Exception as e:
                    print(f"[warn] {cand.clip_id}: Florence-2 failed: {e}")

            if target is not None:
                if not matches_taxonomy(cand, target, cfg, require_shared_noun=require_shared_noun):
                    continue
            elif not _passes_noun_requirement(cand, cfg, require_shared_noun=require_shared_noun):
                continue

            yield cand
            count += 1

    def _flush_llm_batch() -> Iterator[ClipCandidate]:
        if not pending_llm:
            return
        requests = [
            NarrationParseRequest(
                request_id=req_id,
                narration=narration_text,
                candidates=candidates,
            )
            for _video_uid, _entry, narration_text, candidates, req_id in pending_llm
        ]
        parsed_map = parser.parse_batch(requests, cfg.shared_table_nouns)
        rows: list[tuple[str, dict, str, str | None, list[str], dict]] = []
        for video_uid, entry, narration_text, candidates, req_id in pending_llm:
            verb, noun_hits, parse_source = _metadata_from_parsed(parsed_map.get(req_id))
            rows.append((video_uid, entry, narration_text, verb, noun_hits, parse_source))
        pending_llm.clear()
        batch_cands = list(_emit_rows(rows))
        if on_llm_batch is not None:
            on_llm_batch(batch_cands)
            return
        yield from batch_cands

    for video_uid, entry in _iter_narration_json_entries(narration_path):
        if limit is not None and count >= limit:
            return

        narration_text = entry.get("narration_text", "")
        if not isinstance(narration_text, str) or not narration_text.strip():
            continue

        if require_table_object_markers and (
            "table" not in narration_text.lower() or "#O" not in narration_text
        ):
            continue

        if use_llm_parse:
            candidates = extract_spacy_candidates(narration_text)
            if not candidates.noun_candidates and not candidates.verb_candidates:
                continue
            raw_ts = entry.get("timestamp_sec") or entry.get("_unmapped_timestamp_sec") or 0
            ann_uid = entry.get("annotation_uid") or "na"
            req_id = f"{video_uid}:{ann_uid}:{raw_ts}"
            pending_llm.append((video_uid, entry, narration_text, candidates, req_id))
            if len(pending_llm) >= batch_size:
                print(f"Done {count} candidates")
                yield from _flush_llm_batch()
            continue

        verb, noun_hits, parse_source = _metadata_from_narration(narration_text, entry, cfg)
        yield from _emit_rows([(video_uid, entry, narration_text, verb, noun_hits, parse_source)])

    yield from _flush_llm_batch()
