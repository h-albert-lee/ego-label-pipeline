#!/usr/bin/env python3
"""Standalone CLI for VLM cross-check of ownership labels.

Supports two input modes:
  --input      local labels.jsonl  (frames resolved via --frames-root)
  --hf-dataset HuggingFace dataset (images embedded; no --frames-root needed)
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
    load_records_from_hf,
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
    parser = argparse.ArgumentParser(
        description="VLM cross-check for ownership labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From local JSONL
  python run_vlm_crosscheck.py \\
      --input outputs/egolife/labels.jsonl \\
      --judge anthropic:claude-sonnet-4-6 --judge openai:gpt-4o

  # From HuggingFace dataset
  python run_vlm_crosscheck.py \\
      --hf-dataset leejangha1257/ego-ownership-egolife \\
      --judge anthropic:claude-sonnet-4-6 --judge gemini:gemini-2.0-flash \\
      --out outputs/egolife/crosscheck.jsonl
""",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="labels.jsonl from one-pass-labels")
    src.add_argument("--hf-dataset", metavar="DATASET_ID",
                     help="HuggingFace dataset ID, e.g. leejangha1257/ego-ownership-egolife")

    parser.add_argument("--hf-split", default="train", help="HF split to load (default: train)")
    parser.add_argument("--hf-cache-dir", type=Path, default=None,
                        help="Directory to cache frames extracted from HF images (default: temp dir)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSONL (default: crosscheck.jsonl next to --input, or required with --hf-dataset)")
    parser.add_argument("--frames-root", type=Path, default=None,
                        help="Root dir for resolving relative frame paths (local JSONL only)")
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

    records = None
    frames_root = args.frames_root

    if args.hf_dataset:
        if args.out is None:
            parser.error("--out is required when using --hf-dataset")
        print(f"Loading from HuggingFace: {args.hf_dataset} (split={args.hf_split})")
        records, hf_frames_root = load_records_from_hf(
            args.hf_dataset, split=args.hf_split, frames_cache=args.hf_cache_dir
        )
        frames_root = hf_frames_root
        print(f"Loaded {len(records)} records, frames cached in {hf_frames_root}")
        resolved_out = args.out
        input_path = None
    else:
        resolved_out = args.out or args.input.with_name("crosscheck.jsonl")
        input_path = args.input
        print(f"Input:  {input_path}")

    print(f"Output: {resolved_out}")
    print(f"Judges: {[j.model_id for j in judge_objs]}")

    n = write_crosscheck_jsonl(
        input_path,
        resolved_out,
        judge_objs,
        records=records,
        frames_root=frames_root,
        limit=args.limit if args.limit > 0 else None,
        resume=not args.overwrite,
        show_progress=not args.no_progress,
    )
    print(f"Done. Wrote {n} rows → {resolved_out}")


if __name__ == "__main__":
    main()
