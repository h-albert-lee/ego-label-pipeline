"""VLM cross-check for labels.jsonl ownership judgements.

Multiple VLM judges independently predict the ownership label (MINE / PERSON_k /
SHARED / AMBIGUOUS) from the 3 sparse frames + narration + object description.
Each judge's answer is compared to ``auto_ground_truth`` and recorded.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

def _iter_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

VALID_LABELS = {"MINE", "PERSON_k", "SHARED", "AMBIGUOUS"}

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert annotator for egocentric video understanding. "
    "You decide who OWNS or primarily uses a highlighted target object "
    "visible in first-person (ego-camera) footage."
)

_LABEL_DEFS = """\
Label definitions:
  MINE      – the camera wearer (ego) primarily owns / uses this object
  PERSON_k  – another person in the scene primarily owns / uses this object
  SHARED    – ownership is genuinely shared between multiple people
  AMBIGUOUS – impossible to determine from the available evidence
"""


def _build_prompt(record: dict[str, Any]) -> str:
    narration = str(record.get("dense_caption_en") or record.get("dense_caption") or "").strip()
    caption = str(record.get("object_caption") or "").strip()
    noun = str((record.get("object") or {}).get("label") or record.get("nouns", ["object"])[0])

    lines = [
        f"Target object: {noun}",
        "",
        f"Narration (what is happening): {narration}" if narration else "",
        "",
    ]
    if caption:
        # Keep only sections (1) and (2) — section (3) can hint at the label.
        cap_section = re.split(r"\(\s*3\s*\)", caption, maxsplit=1)[0].strip()
        if cap_section:
            lines += [f"Object visual description:\n{cap_section}", ""]

    lines += [
        _LABEL_DEFS,
        "You are shown three frames from the video:",
        "  • frame t-2  (2 seconds before the action)",
        "  • frame t-1  (1 second before the action)",
        "  • frame t    (the action moment — target object is highlighted with a bounding box)",
        "",
        "Respond ONLY with valid JSON (no markdown fences):",
        '{"label": "<MINE|PERSON_k|SHARED|AMBIGUOUS>", "rationale": "<one concise sentence>"}',
    ]
    return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class OwnershipJudge(Protocol):
    model_id: str

    def judge(
        self,
        frame_paths: list[Path],  # [t-2, t-1, t] — t has bbox drawn
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Return {"label": str, "rationale": str}."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_label_response(raw: str) -> dict[str, Any]:
    """Extract label/rationale from a VLM text response."""
    raw = raw.strip()
    # Strip markdown fences if present.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).strip().upper()
        if label not in VALID_LABELS:
            label = _extract_label_from_text(raw)
        return {"label": label, "rationale": str(data.get("rationale", "")).strip()}
    except (json.JSONDecodeError, ValueError):
        label = _extract_label_from_text(raw)
        return {"label": label, "rationale": raw[:200]}


def _extract_label_from_text(text: str) -> str:
    for candidate in VALID_LABELS:
        if candidate in text.upper():
            return candidate
    return "UNKNOWN"


def _draw_bbox_on_frame(src: Path, dst: Path, bbox: dict[str, Any]) -> None:
    """Copy src to dst and draw normalized bbox in red."""
    try:
        from PIL import Image, ImageDraw
        with Image.open(src).convert("RGB") as img:
            w, h = img.size
            draw = ImageDraw.Draw(img)
            x1 = int(bbox.get("x_min", 0) * w)
            y1 = int(bbox.get("y_min", 0) * h)
            x2 = int(bbox.get("x_max", 1) * w)
            y2 = int(bbox.get("y_max", 1) * h)
            lw = max(3, round(min(w, h) * 0.006))
            for off in range(lw):
                draw.rectangle((x1 - off, y1 - off, x2 + off, y2 + off), outline=(255, 50, 50))
            img.save(dst, quality=92)
    except Exception:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Qwen VL judge
# ---------------------------------------------------------------------------

@dataclass
class QwenOwnershipJudgeConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    trust_remote_code: bool = True


class QwenOwnershipJudge:
    def __init__(self, cfg: QwenOwnershipJudgeConfig | None = None):
        self.cfg = cfg or QwenOwnershipJudgeConfig()
        self.model_id = self.cfg.model_id
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dtype = dtype_map.get(self.cfg.dtype, "auto") if self.cfg.dtype != "auto" else "auto"
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.cfg.model_id,
                torch_dtype=dtype,
                device_map=self.cfg.device,
                trust_remote_code=self.cfg.trust_remote_code,
            )
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(
                self.cfg.model_id,
                trust_remote_code=self.cfg.trust_remote_code,
            )
        return self._processor, self._model

    def judge(self, frame_paths: list[Path], record: dict[str, Any]) -> dict[str, Any]:
        processor, model = self._load()

        content: list[dict] = []
        for fp in frame_paths:
            if fp.exists():
                content.append({"type": "image", "image": str(fp)})
        content.append({"type": "text", "text": _build_prompt(record)})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")

        try:
            device = next(model.parameters()).device
            inputs = inputs.to(device)
        except Exception:
            pass

        import torch
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return _parse_label_response(raw)


# ---------------------------------------------------------------------------
# Anthropic (Claude) judge
# ---------------------------------------------------------------------------

@dataclass
class AnthropicOwnershipJudgeConfig:
    model_id: str = "claude-sonnet-4-6"
    max_tokens: int = 512
    api_key: str | None = None


class AnthropicOwnershipJudge:
    def __init__(self, cfg: AnthropicOwnershipJudgeConfig | None = None):
        self.cfg = cfg or AnthropicOwnershipJudgeConfig()
        self.model_id = self.cfg.model_id
        self._client: Any = None

    def _load(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                **({"api_key": self.cfg.api_key} if self.cfg.api_key else {})
            )
        return self._client

    def judge(self, frame_paths: list[Path], record: dict[str, Any]) -> dict[str, Any]:
        import base64
        client = self._load()

        content: list[dict] = []
        for fp in frame_paths:
            if fp.exists():
                data = base64.standard_b64encode(fp.read_bytes()).decode()
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
                })
        content.append({"type": "text", "text": _build_prompt(record)})

        response = client.messages.create(
            model=self.cfg.model_id,
            max_tokens=self.cfg.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text
        return _parse_label_response(raw)


# ---------------------------------------------------------------------------
# OpenAI judge
# ---------------------------------------------------------------------------

@dataclass
class OpenAIOwnershipJudgeConfig:
    model_id: str = "gpt-4o"
    max_tokens: int = 512
    api_key: str | None = None


class OpenAIOwnershipJudge:
    def __init__(self, cfg: OpenAIOwnershipJudgeConfig | None = None):
        self.cfg = cfg or OpenAIOwnershipJudgeConfig()
        self.model_id = self.cfg.model_id
        self._client: Any = None

    def _load(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                **({"api_key": self.cfg.api_key} if self.cfg.api_key else {})
            )
        return self._client

    def judge(self, frame_paths: list[Path], record: dict[str, Any]) -> dict[str, Any]:
        import base64
        client = self._load()

        content: list[dict] = []
        for fp in frame_paths:
            if fp.exists():
                data = base64.standard_b64encode(fp.read_bytes()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{data}", "detail": "high"},
                })
        content.append({"type": "text", "text": _build_prompt(record)})

        response = client.chat.completions.create(
            model=self.cfg.model_id,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": content},
            ],
        )
        raw = response.choices[0].message.content or ""
        return _parse_label_response(raw)


# ---------------------------------------------------------------------------
# Gemini judge
# ---------------------------------------------------------------------------

@dataclass
class GeminiOwnershipJudgeConfig:
    model_id: str = "gemini-2.0-flash"
    max_tokens: int = 512
    api_key: str | None = None


class GeminiOwnershipJudge:
    def __init__(self, cfg: GeminiOwnershipJudgeConfig | None = None):
        self.cfg = cfg or GeminiOwnershipJudgeConfig()
        self.model_id = self.cfg.model_id
        self._client: Any = None

    def _load(self) -> Any:
        if self._client is None:
            import google.generativeai as genai
            if self.cfg.api_key:
                genai.configure(api_key=self.cfg.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.cfg.model_id,
                system_instruction=_SYSTEM,
                generation_config={"max_output_tokens": self.cfg.max_tokens},
            )
        return self._client

    def judge(self, frame_paths: list[Path], record: dict[str, Any]) -> dict[str, Any]:
        import google.generativeai as genai
        client = self._load()

        parts: list[Any] = []
        for fp in frame_paths:
            if fp and Path(fp).exists():
                parts.append({"mime_type": "image/jpeg", "data": Path(fp).read_bytes()})
        parts.append(_build_prompt(record))

        response = client.generate_content(parts)
        raw = response.text or ""
        return _parse_label_response(raw)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _resolve_frame_paths(record: dict[str, Any], frames_root: Path | None) -> list[Path]:
    """Return [t-2, t-1, t] paths, resolving against frames_root or CWD."""
    keys = ["frame_t_minus_2_path", "frame_t_minus_1_path", "frame_t_path"]
    paths = []
    for k in keys:
        raw = record.get(k) or ""
        if not raw:
            paths.append(None)
            continue
        p = Path(raw)
        candidates = []
        if frames_root:
            candidates.append(frames_root / p)
        candidates.append(p)                          # relative to CWD
        candidates.append(Path.cwd() / p)            # explicit CWD join
        found = next((c for c in candidates if c.exists()), None)
        paths.append(found)
    return paths


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def load_records_from_hf(
    dataset_id: str,
    split: str = "train",
    frames_cache: Path | None = None,
) -> tuple[list[dict], Path]:
    """Load records from a HuggingFace dataset, saving embedded images to disk.

    Returns (records, frames_cache_dir). The frame path keys in each record
    are updated to point to the saved JPEG files so existing judge code works
    without modification.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)

    if frames_cache is None:
        frames_cache = Path(tempfile.mkdtemp(prefix="vlm_crosscheck_hf_"))
    else:
        frames_cache = Path(frames_cache)
        frames_cache.mkdir(parents=True, exist_ok=True)

    _JSON_COLS = {"object", "nouns", "frame_times_sec", "frame_paths",
                  "temporal_target_objects", "evidence", "described_frame_timestamps_sec"}
    _IMAGE_KEYS = ("frame_t_minus_2_path", "frame_t_minus_1_path", "frame_t_path")

    records = []
    for i, row in enumerate(ds):
        rec = dict(row)

        for col in _JSON_COLS:
            val = rec.get(col)
            if isinstance(val, str) and val:
                try:
                    rec[col] = json.loads(val)
                except json.JSONDecodeError:
                    pass

        for key in _IMAGE_KEYS:
            img = rec.get(key)
            if img is not None:
                try:
                    fpath = frames_cache / f"{i}_{key}.jpg"
                    if not fpath.exists():
                        img.save(fpath, quality=92)
                    rec[key] = str(fpath)
                except Exception:
                    rec[key] = None

        records.append(rec)

    return records, frames_cache


def write_crosscheck_jsonl(
    labels_path: Path | None,
    out_path: Path,
    judges: list[Any],
    *,
    records: list[dict] | None = None,
    frames_root: Path | None = None,
    limit: int | None = None,
    resume: bool = True,
    show_progress: bool = True,
) -> int:
    """Run all judges on each record and write agreement stats.

    Records can be supplied directly via ``records`` (e.g. loaded from HF) or
    read from ``labels_path``.  At least one must be provided.

    Output schema per row:
      id, auto_ground_truth, judges: {model_id: {label, rationale, agrees}},
      agreement_count, agreement_ratio, majority_label
    """
    if records is None:
        if labels_path is None:
            raise ValueError("Provide either labels_path or records.")
        records = list(_iter_jsonl(labels_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _load_existing_ids(out_path) if resume else set()

    if limit:
        records = records[:limit]

    todo = [r for r in records if r.get("id") not in existing_ids]

    if show_progress:
        try:
            from tqdm.auto import tqdm
            todo = tqdm(todo, total=len(todo), unit="record", desc="vlm-crosscheck")
        except ImportError:
            pass

    n_written = 0
    mode = "a" if resume else "w"
    with out_path.open(mode, encoding="utf-8") as fh:
        for record in todo:
            rid = record.get("id", "")
            auto_gt = record.get("auto_ground_truth") or record.get("auto_label") or ""
            bbox = (record.get("object") or {}).get("bbox") or {}

            # Prepare frames: draw bbox on a temp copy of frame_t.
            raw_paths = _resolve_frame_paths(record, frames_root)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                frame_t_src = raw_paths[2]
                frame_t_annotated = tmp_path / "frame_t_bbox.jpg"
                if frame_t_src and frame_t_src.exists() and bbox:
                    _draw_bbox_on_frame(frame_t_src, frame_t_annotated, bbox)
                else:
                    frame_t_annotated = frame_t_src  # fallback: no bbox drawn

                frame_paths = [
                    p for p in [raw_paths[0], raw_paths[1], frame_t_annotated]
                    if p and Path(p).exists()
                ]

                judge_results: dict[str, dict] = {}
                for judge in judges:
                    try:
                        result = judge.judge(frame_paths, record)
                    except Exception as exc:
                        result = {"label": "ERROR", "rationale": str(exc)[:200]}
                    result["agrees"] = (result.get("label") == auto_gt)
                    judge_results[judge.model_id] = result

            labels_predicted = [v["label"] for v in judge_results.values() if v["label"] in VALID_LABELS]
            agreement_count = sum(1 for v in judge_results.values() if v.get("agrees"))
            majority_label = Counter(labels_predicted).most_common(1)[0][0] if labels_predicted else "UNKNOWN"

            row = {
                "id": rid,
                "auto_ground_truth": auto_gt,
                "judges": judge_results,
                "agreement_count": agreement_count,
                "agreement_ratio": round(agreement_count / max(1, len(judge_results)), 4),
                "majority_label": majority_label,
                "majority_agrees": majority_label == auto_gt,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_written += 1

    return n_written
