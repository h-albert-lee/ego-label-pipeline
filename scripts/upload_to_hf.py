#!/usr/bin/env python3
"""Upload labels.jsonl + sparse frames to a HuggingFace Dataset.

Usage:
    python scripts/upload_to_hf.py \\
        --input outputs/egolife/labels.jsonl \\
        --repo your-hf-username/ego-ownership-egolife \\
        --frames-root .

Requirements:
    pip install datasets huggingface_hub Pillow
    huggingface-cli login   # run once to authenticate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload labels + frames to HuggingFace")
    parser.add_argument("--input", required=True, type=Path, help="labels.jsonl path")
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/dataset-name")
    parser.add_argument("--frames-root", type=Path, default=Path("."),
                        help="Root directory for resolving relative frame paths (default: CWD)")
    parser.add_argument("--private", action="store_true", help="Create as private dataset")
    parser.add_argument("--split", default="train", help="Dataset split name (default: train)")
    parser.add_argument("--limit", type=int, default=0, help="Upload only first N records (0 = all)")
    args = parser.parse_args()

    try:
        from datasets import Dataset, Features, Image, Value
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    # Load records
    records = []
    with args.input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if args.limit > 0:
        records = records[:args.limit]

    print(f"Loaded {len(records)} records from {args.input}")

    # Resolve frame paths to absolute paths; drop records with missing frames.
    frame_keys = ["frame_t_minus_2_path", "frame_t_minus_1_path", "frame_t_path"]
    resolved = []
    n_missing = 0
    for rec in records:
        ok = True
        for k in frame_keys:
            raw = rec.get(k) or ""
            if not raw:
                ok = False
                break
            p = Path(raw)
            if not p.is_absolute():
                p = args.frames_root / p
            if not p.exists():
                ok = False
                break
            rec[k] = str(p)
        if ok:
            resolved.append(rec)
        else:
            n_missing += 1

    if n_missing:
        print(f"  Skipped {n_missing} records with missing frame files")
    print(f"  Uploading {len(resolved)} records with all 3 frames present")

    # Flatten nested dicts to strings so HuggingFace schema is simple.
    for rec in resolved:
        for k, v in list(rec.items()):
            if isinstance(v, (dict, list)):
                rec[k] = json.dumps(v, ensure_ascii=False)

    ds = Dataset.from_list(resolved)

    # Cast frame columns to Image so HF stores + previews them properly.
    for k in frame_keys:
        if k in ds.column_names:
            ds = ds.cast_column(k, Image())

    print(f"Pushing to hub: {args.repo} (split={args.split}, private={args.private})")
    ds.push_to_hub(
        args.repo,
        split=args.split,
        private=args.private,
    )
    print(f"Done → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
