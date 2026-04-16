"""Segmentation backends.

Every backend implements `segment(input_video, prompt, work_dir, **opts) -> Path`
and returns the path to a B/W mask-only mp4 that matches the input's frame
count and framerate. The caller (`cli.py`) then composites the mask against
the original via `video_io.compose_rgba_mov` — shared across all backends.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    name: str

    def segment(
        self,
        input_video: Path,
        prompt: str,
        *,
        work_dir: Path,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> Path:
        """Produce a B/W mask-only mp4. Return its path."""
        ...


def available() -> list[str]:
    return ["replicate", "fal", "local"]


def resolve(name: str | None = None) -> Backend:
    """Pick a backend.

    Explicit name wins. Otherwise auto-detect by env var:
      REPLICATE_API_TOKEN -> replicate
      FAL_KEY             -> fal
      (else)              -> error asking the user to pick
    """
    if name is None:
        if os.environ.get("FAL_KEY"):
            name = "fal"
        elif os.environ.get("REPLICATE_API_TOKEN"):
            name = "replicate"
        else:
            sys.exit(
                "[samify] No backend selected and no cloud API key found.\n"
                "  Set FAL_KEY (recommended, SAM 3.1) or REPLICATE_API_TOKEN,\n"
                "  or pass --backend local (requires GPU + `pip install -e .[local]`)."
            )

    if name == "replicate":
        from samify.backends.replicate_cloud import ReplicateBackend

        return ReplicateBackend()
    if name == "fal":
        from samify.backends.fal_cloud import FalBackend

        return FalBackend()
    if name == "local":
        from samify.backends.local import LocalBackend

        return LocalBackend()
    raise ValueError(
        f"Unknown backend {name!r}. Pick one of: {available()}"
    )
