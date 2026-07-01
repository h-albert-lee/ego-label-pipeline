"""Translate EgoLife table captions with Qwen, then extract English nouns."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from egoownership.config import normalize_token
from egoownership.egolife_annotations import (
    _caption_mentions_table,
    _iter_egolife_cap_srt_records,
)


TranslationFn = Callable[[str, str], str]
_NOUN_STOPWORDS = frozenset(
    {
        "caption",
        "dining",
        "dining_table",
        "egocentric",
        "everyone",
        "frame",
        "front",
        "haha",
        "hahaha",
        "he",
        "hmm",
        "i",
        "it",
        "jake",
        "alice",
        "tasha",
        "lucia",
        "katrina",
        "shure",
        "me",
        "one",
        "other",
        "she",
        "side",
        "something",
        "table",
        "tables",
        "that",
        "thing",
        "things",
        "they",
        "this",
        "transcript",
        "us",
        "we",
        "what",
        "who",
        "you",
    }
)
_VERB_STOPWORDS = frozenset({"be", "do", "have"})


@dataclass
class QwenTranslationConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 96
    trust_remote_code: bool = True
    local_files_only: bool = False


class QwenCaptionTranslator:
    """Small text-only Qwen wrapper for Chinese-to-English caption translation."""

    def __init__(self, cfg: QwenTranslationConfig | None = None):
        self.cfg = cfg or QwenTranslationConfig()
        self._tokenizer: Any = None
        self._model: Any = None

    def translate(self, text: str, field_name: str = "text") -> str:
        text = text.strip()
        if not text:
            return ""
        tokenizer, model = self._load()
        prompt = _translation_prompt(text, field_name)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt")
        device = _model_input_device(model, self.cfg.device)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        generated = output_ids[:, prompt_len:] if output_ids.shape[-1] > prompt_len else output_ids
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return _clean_translation(decoded[0] if decoded else "")

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": self.cfg.trust_remote_code,
            "local_files_only": self.cfg.local_files_only,
        }
        model_kwargs: dict[str, Any] = dict(tokenizer_kwargs)
        dtype = _torch_dtype(torch, self.cfg.dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.cfg.device == "auto":
            model_kwargs["device_map"] = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, **tokenizer_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, **model_kwargs)
        if self.cfg.device not in ("auto", "", None) and hasattr(self._model, "to"):
            self._model = self._model.to(self.cfg.device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self._tokenizer, self._model


def write_qwen_translated_table_caption_nouns(
    annotations_path: Path,
    out_path: Path,
    *,
    translate_fn: TranslationFn,
    noun_summary_out: Path | None = None,
    limit: int | None = None,
    show_progress: bool = True,
) -> int:
    """Translate table-related captions first, then extract nouns from English."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if noun_summary_out is not None:
        noun_summary_out.parent.mkdir(parents=True, exist_ok=True)

    records: Iterable[dict[str, Any]] = _table_caption_records(annotations_path)
    if show_progress:
        from tqdm.auto import tqdm

        records = tqdm(records, unit="caption", desc="Qwen translate table captions")

    noun_counts: Counter[str] = Counter()
    noun_examples: dict[str, dict[str, str]] = {}
    count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for row_idx, record in enumerate(records):
            if limit is not None and count >= limit:
                break
            caption = str(record.get("dense_caption") or "")
            transcript = str(record.get("transcript") or "")
            caption_en = translate_fn(caption, "dense caption")
            transcript_en = translate_fn(transcript, "transcript") if transcript.strip() else ""
            combined_translation = _combine_translations(caption_en, transcript_en)
            nouns, verbs = _extract_english_nouns_verbs(combined_translation)

            for noun in nouns:
                noun_counts[noun] += 1
                noun_examples.setdefault(
                    noun,
                    {
                        "dense_caption": caption,
                        "transcript": transcript,
                        "qwen_translation": combined_translation,
                    },
                )

            row = {
                "row_idx": row_idx,
                "id": record.get("id") or "",
                "clip_id": record.get("clip_id") or "",
                "video_id": record.get("video_id") or "",
                "participant": record.get("participant") or "",
                "day": record.get("day") or "",
                "start_sec": record.get("start_sec"),
                "end_sec": record.get("end_sec"),
                "dense_caption": caption,
                "transcript": transcript,
                "dense_caption_en": caption_en,
                "transcript_en": transcript_en,
                "qwen_translation": combined_translation,
                "noun_candidates": nouns,
                "verb_candidates": verbs,
                "source": record.get("source") or {},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += 1

    if noun_summary_out is not None:
        _write_noun_summary(noun_summary_out, noun_counts, noun_examples)
    return count


def _table_caption_records(path: Path) -> Iterable[dict[str, Any]]:
    for record in _iter_egolife_cap_srt_records(path):
        caption = str(record.get("dense_caption") or "")
        if _caption_mentions_table(caption):
            yield record


def _write_noun_summary(
    out_path: Path,
    counts: Counter[str],
    examples: dict[str, dict[str, str]],
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for rank, (noun, count) in enumerate(counts.most_common(), start=1):
            example = examples.get(noun, {})
            row = {
                "rank": rank,
                "noun": noun,
                "count": count,
                "example_caption_zh": example.get("dense_caption", ""),
                "example_transcript": example.get("transcript", ""),
                "example_translation": example.get("qwen_translation", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _translation_prompt(text: str, field_name: str = "text") -> str:
    return (
        f"Translate the following EgoLife {field_name} into fluent English for object extraction. "
        "Preserve concrete physical object names, speaker names, and action verbs. "
        "Do not explain. Do not output Chinese. Return only the English translation.\n\n"
        f"{field_name}: {text}"
    )


def _combine_translations(caption_en: str, transcript_en: str) -> str:
    parts = []
    if caption_en.strip():
        parts.append(caption_en.strip())
    if transcript_en.strip():
        parts.append(transcript_en.strip())
    return " ".join(parts)


def _clean_translation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|text)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(r"^(?:Translation|English|Answer)\s*:\s*", "", text, flags=re.I).strip()
    text = text.strip('"').strip()
    return re.sub(r"\s+", " ", text)


def _extract_english_nouns_verbs(text: str) -> tuple[list[str], list[str]]:
    """Strict POS-based extraction for already-translated English captions."""

    doc = _load_spacy_model()(text)
    nouns: set[str] = set()
    verbs: set[str] = set()

    chunk_tokens: set[int] = set()
    for chunk in doc.noun_chunks:
        root = normalize_token(chunk.root.lemma_)
        if _keep_translated_noun(root):
            nouns.add(root)
        chunk_tokens.update(tok.i for tok in chunk)

    # Only pick up NOUN/PROPN tokens outside any noun chunk here, so a
    # compound like "paper towel" contributes just its chunk root ("towel")
    # instead of also adding "paper" as a separate, spurious noun candidate.
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and tok.i not in chunk_tokens:
            noun = normalize_token(tok.lemma_)
            if _keep_translated_noun(noun):
                nouns.add(noun)
        elif tok.pos_ == "VERB":
            verb = normalize_token(tok.lemma_)
            if verb and verb not in _VERB_STOPWORDS:
                verbs.add(verb)

    return sorted(nouns), sorted(verbs)


def _keep_translated_noun(noun: str) -> bool:
    return bool(noun) and len(noun) > 1 and noun not in _NOUN_STOPWORDS


@lru_cache(maxsize=1)
def _load_spacy_model() -> Any:
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


def _torch_dtype(torch: Any, dtype: str) -> Any | None:
    normalized = (dtype or "auto").lower()
    if normalized in ("", "auto"):
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError("--dtype must be one of auto, float16, bfloat16, float32")
    return mapping[normalized]


def _model_input_device(model: Any, requested_device: str) -> Any | None:
    if requested_device and requested_device != "auto":
        return requested_device
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        for key in ("model.embed_tokens", "transformer.wte", "model", ""):
            value = device_map.get(key)
            if value not in (None, "cpu", "disk", "meta"):
                return f"cuda:{value}" if isinstance(value, int) else value
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None
