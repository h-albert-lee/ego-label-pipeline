"""Ego4D download helper.

Ego4D is license-gated. Rather than scraping URLs, we emit the exact commands
that the official ``ego4d`` CLI uses once the user has signed the license
agreement at https://ego4d-data.org/ .
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

_CONSOLE = Console()


def download(out_dir: Path, videos: bool = False) -> None:
    """Print the commands needed to fetch Ego4D annotations (and optionally videos).

    Only *instructions* are printed — Ego4D requires an accepted license and an
    AWS access key assigned to the user's account.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    _CONSOLE.print("[bold]Ego4D download is license-gated.[/bold]")
    _CONSOLE.print(
        "1. Sign the license at https://ego4d-data.org/ and receive your AWS keys."
    )
    _CONSOLE.print("2. Install the official CLI:")
    _CONSOLE.print("   [cyan]pip install ego4d[/cyan]")
    _CONSOLE.print("3. Download FHO annotations (needed by this pipeline):")
    _CONSOLE.print(
        f"   [cyan]ego4d --output_directory {out_dir} --datasets annotations"
        f" --version v2 --metadata[/cyan]"
    )
    _CONSOLE.print(
        "   The FHO main file lands at: "
        f"[cyan]{out_dir}/v2/annotations/fho_main.json[/cyan]"
    )
    if videos:
        _CONSOLE.print("4. (Optional) Download full videos — this is ~several TB:")
        _CONSOLE.print(
            f"   [cyan]ego4d --output_directory {out_dir} --datasets full_scale"
            " --version v2[/cyan]"
        )
    _CONSOLE.print(
        "\nOnce annotations land, run:\n"
        f"   [cyan]egoown filter ego4d-fho "
        f"--annotations {out_dir}/v2/annotations/fho_main.json "
        f"--out outputs/candidates_fho.jsonl[/cyan]"
    )
