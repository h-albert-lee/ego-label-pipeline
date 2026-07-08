#!/usr/bin/env python3
"""Standalone CLI for VLM cross-check of ownership labels.

Usage examples:
    python run_vlm_crosscheck.py \\
        --input outputs/egolife/labels.jsonl \\
        --judge anthropic:claude-sonnet-4-6 \\
        --judge openai:gpt-4o \\
        --judge gemini:gemini-2.0-flash

    python run_vlm_crosscheck.py \\
        --input outputs/ego4d/labels.jsonl \\
        --frames-root /home/user/ego-label-pipeline \\
        --out outputs/ego4d/crosscheck.jsonl \\
        --judge anthropic:claude-opus-4-8 \\
        --judge openai:gpt-4o \\
        --judge gemini:gemini-2.0-flash \\
        --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from egoownership.vlm_crosscheck import (
    AnthropicOwnershipJudge,
    AnthropicOwnershipJudgeConfig,
    GeminiOwnershipJudge,
    GeminiOwnershipJudgeConfig,
    OpenAIOwnershipJudge,
    OpenAIOwnershipJudgeConfig,
    QwenOwnershipJudge,
    QwenOwnershipJudgeConfig,
    write_crosscheck_jsonl,
)


def _parse_judge(spec: str):
    if ":" in spec:
        backend, model_id = spec.split(":", 1)
    else:
        backend = model_id = spec
    backend = backend.lower()

    if backend == "anthropic":
        return AnthropicOwnershipJudge(AnthropicOwnershipJudgeConfig(model_id=model_id))
    if backend == "openai":
        return OpenAIOwnershipJudge(OpenAIOwnershipJudgeConfig(model_id=model_id))
    if backend == "gemini":
        return GeminiOwnershipJudge(GeminiOwnershipJudgeConfig(model_id=model_id))
    if backend == "qwen":
        return QwenOwnershipJudge(QwenOwnershipJudgeConfig(model_id=model_id))
    raise ValueError(f"Unknown judge backend {backend!r}. Use anthropic, openai, gemini, or qwen.")


def main():
    parser = argparse.ArgumentParser(description="VLM cross-check for ownership labels.jsonl")
    parser.add_argument("--input", required=True, type=Path, help="labels.jsonl from one-pass-labels")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL (default: next to --input as crosscheck.jsonl)")
    parser.add_argument("--frames-root", type=Path, default=None, help="Root dir for resolving relative frame paths")
    parser.add_argument("--judge", action="append", dest="judges", default=[],
                        metavar="BACKEND:MODEL_ID",
                        help="Judge spec, e.g. anthropic:claude-sonnet-4-6. Repeat for multiple judges.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to process (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    if not args.judges:
        parser.error("Specify at least one --judge, e.g. --judge anthropic:claude-sonnet-4-6")

    judge_objs = [_parse_judge(s) for s in args.judges]
    resolved_out = args.out or args.input.with_name("crosscheck.jsonl")

    print(f"Input:  {args.input}")
    print(f"Output: {resolved_out}")
    print(f"Judges: {[j.model_id for j in judge_objs]}")

    n = write_crosscheck_jsonl(
        args.input,
        resolved_out,
        judge_objs,
        frames_root=args.frames_root,
        limit=args.limit if args.limit > 0 else None,
        resume=not args.overwrite,
        show_progress=not args.no_progress,
    )
    print(f"Done. Wrote {n} rows → {resolved_out}")


if __name__ == "__main__":
    main()
