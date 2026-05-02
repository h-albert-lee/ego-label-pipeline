"""EPIC-KITCHENS-100 annotation download helper.

Annotations live on the public epic-kitchens/epic-kitchens-100-annotations
GitHub repo. Videos are distributed via the University of Bristol; we only
fetch annotations by default.
"""

from __future__ import annotations

from pathlib import Path

import requests
from rich.console import Console
from tqdm import tqdm

_CONSOLE = Console()

_ANNOT_BASE = (
    "https://raw.githubusercontent.com/epic-kitchens/"
    "epic-kitchens-100-annotations/master"
)

_ANNOT_FILES = [
    "EPIC_100_train.csv",
    "EPIC_100_validation.csv",
    "EPIC_100_verb_classes.csv",
    "EPIC_100_noun_classes.csv",
]


def _fetch(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download(out_dir: Path, videos: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _CONSOLE.print("[bold]Fetching EPIC-KITCHENS-100 annotations…[/bold]")
    for name in _ANNOT_FILES:
        dest = out_dir / name
        if dest.exists():
            _CONSOLE.print(f"  skip {name} (already present)")
            continue
        _fetch(f"{_ANNOT_BASE}/{name}", dest)

    if videos:
        _CONSOLE.print(
            "\n[yellow]Video distribution is gated by the University of Bristol.[/yellow]"
        )
        _CONSOLE.print(
            "See https://github.com/epic-kitchens/epic-kitchens-download-scripts"
            " for the official downloader."
        )
