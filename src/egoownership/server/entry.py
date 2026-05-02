"""Reload-friendly entry point.

Reads ``EGOOWN_SCENES_PATH`` and ``EGOOWN_FRAMES_ROOT`` from the environment so
``uvicorn --reload`` can re-import the app without re-running the CLI.
"""

from __future__ import annotations

import os
from pathlib import Path

from egoownership.server.app import create_app

_scenes = os.environ.get("EGOOWN_SCENES_PATH", "outputs/scene_records.jsonl")
_frames = os.environ.get("EGOOWN_FRAMES_ROOT", "frames")
_videos = os.environ.get("EGOOWN_VIDEOS_ROOT")
app = create_app(
    Path(_scenes),
    Path(_frames),
    videos_root=Path(_videos) if _videos else None,
)
