"""HD-EPIC annotation download helper.

HD-EPIC (CVPR 2025) publishes annotations — object tracks, bboxes, masks — on
the project site. As of April 2026 the canonical location is the GitHub
release at https://github.com/hd-epic/hd-epic-annotations .

We print the commands rather than pinning a specific URL that may rotate.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

_CONSOLE = Console()


def download(out_dir: Path, videos: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _CONSOLE.print("[bold]HD-EPIC annotations[/bold]")
    _CONSOLE.print(
        "1. Visit the project site: https://hd-epic.github.io/ "
        "and grab the latest annotations release."
    )
    _CONSOLE.print(
        "2. Clone or download the annotation bundle. Typical layout:"
    )
    _CONSOLE.print(
        f"   [cyan]git clone https://github.com/hd-epic/hd-epic-annotations "
        f"{out_dir}/annotations[/cyan]"
    )
    _CONSOLE.print(
        "   You want the files matching [cyan]movement_tracks/*.json[/cyan] — these hold"
        " the per-object pickup→place tracks this pipeline consumes."
    )
    if videos:
        _CONSOLE.print(
            "\n[yellow]Videos: HD-EPIC videos are gated; follow the project-site"
            " instructions.[/yellow]"
        )
