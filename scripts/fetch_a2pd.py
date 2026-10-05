#!/usr/bin/env python3
"""Download selected categories from an A2PD download-manifest JSON.

The manifest lists every file of every sequence with a signed CDN URL, so the
whole release is 148 GB. This pulls only the categories asked for — the ASR
transcripts alone are 31 KB, enough to decide whether the dataset can feed the
ownership pipeline at all before committing to the video.

    python3 scripts/fetch_a2pd.py <manifest.json> --categories video_main_rgb --out data/gen2pilot/a2pd

Note `mps_artifacts` is a zip bundle of the individual `mps_slam_*` /
`mps_hand_tracking` files; taking both downloads 41 GB twice.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--categories", nargs="+", default=["diarization"])
    ap.add_argument("--sequences", nargs="+", default=None, help="default: all")
    ap.add_argument("--out", type=Path, default=Path("a2pd"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    sequences = manifest["sequences"]
    wanted_seqs = args.sequences or sorted(sequences)

    jobs = []
    for seq in wanted_seqs:
        if seq not in sequences:
            print(f"!! unknown sequence {seq}", file=sys.stderr)
            continue
        for cat in args.categories:
            entry = sequences[seq].get(cat)
            if entry is None:
                print(f"   {seq}/{cat}: not in manifest, skipping")
                continue
            jobs.append((seq, cat, entry))

    total = sum(e["file_size_bytes"] for _, _, e in jobs)
    print(f"{len(jobs)} files, {human(total)} total")
    if args.dry_run:
        for seq, cat, e in jobs:
            print(f"   {seq:10} {cat:26} {human(e['file_size_bytes'])}")
        return 0

    for seq, cat, entry in jobs:
        dest_dir = args.out / seq
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / entry["filename"]
        if dest.exists() and dest.stat().st_size == entry["file_size_bytes"]:
            print(f"   {seq}/{cat}: already have it")
        else:
            print(f"   {seq}/{cat}: {human(entry['file_size_bytes'])} -> {dest}")
            urllib.request.urlretrieve(entry["download_url"], dest)
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(dest_dir)

    print(f"\ndone -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
