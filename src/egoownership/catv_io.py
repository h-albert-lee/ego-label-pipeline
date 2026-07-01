"""Small shared helpers for the CAT-V pipeline JSONL I/O."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path, *, skip_bad: bool = False) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if not skip_bad:
                        raise
                    continue


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def normalize_object_noun(noun: str) -> str:
    return noun.lower().replace("_", " ").strip()


def safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "unknown"
