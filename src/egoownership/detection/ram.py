"""RAM (Recognize Anything Model) wrapper for bottom-up tag extraction.

We use ``xinyu1205/recognize-anything-plus-model`` via ``transformers`` when
available. The function returns a list of nouns suitable for feeding back
into Grounding DINO. If RAM weights aren't installed, the wrapper falls back
to a *static frequent-noun whitelist* sourced from the EDA's shared-table
list — degraded but still functional.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from egoownership.config import load_config

_FALLBACK_TAGS = (
    "cup mug bottle glass bowl plate dish fork spoon knife chopstick "
    "pen pencil notebook notepad laptop phone document paper folder "
    "bread basket salt pepper sauce napkin chair table"
).split()


@lru_cache(maxsize=2)
def _load_ram(model_id: str = "xinyu1205/recognize-anything-plus-model"):
    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return processor, model, device
    except Exception:  # noqa: BLE001
        return None


def extract_tags(image_path: Path, *, max_tags: int = 30) -> list[str]:
    """Run RAM on one frame; degrade gracefully to the EDA noun whitelist."""

    bundle = _load_ram()
    if bundle is not None:
        try:
            from PIL import Image
            import torch

            processor, model, device = bundle
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                tags = model.generate_tag(**inputs)
            if isinstance(tags, list):
                return [t.lower().strip() for t in tags[:max_tags] if t]
        except Exception:  # noqa: BLE001
            pass

    # Fallback: take the EDA shared-table nouns. Order is preserved for
    # determinism so prompts hash the same across runs.
    cfg = load_config()
    tags = list(cfg.shared_table_nouns)
    # Round out with the static fallback so we don't miss "table" / "chair".
    for t in _FALLBACK_TAGS:
        if t not in tags:
            tags.append(t)
    return tags[:max_tags]


def merge_with_clip_nouns(clip_nouns: list[str], ram_tags: list[str], *, cap: int = 25) -> list[str]:
    """Combine clip-provided nouns (high-precision) with RAM tags (high-recall).

    Clip nouns go first so they always make the prompt. Duplicate detection
    uses normalized (lowercase, underscore) tokens.
    """
    seen: set[str] = set()
    out: list[str] = []
    for source in (clip_nouns, ram_tags):
        for n in source:
            key = n.lower().replace("-", "_").replace(" ", "_").strip()
            if key and key not in seen:
                seen.add(key)
                out.append(n)
                if len(out) >= cap:
                    return out
    return out
