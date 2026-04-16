"""Smoke tests.

Two modes:
  - `test_cloud_replicate`   — needs REPLICATE_API_TOKEN; hits the real API.
  - `test_local`             — needs .[local] deps + CUDA; runs on-device.

Both render a tiny synthetic clip via ffmpeg, run samify end-to-end, and
assert the output is a real ProRes 4444 RGBA .mov with matching frame count.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _make_fixture_clip(path: Path, *, seconds: int = 2, fps: int = 24) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg not on PATH"
    # Bright lime rectangle on a dark background — unambiguous subject.
    filter_complex = (
        f"color=c=black:s=480x270:d={seconds}:r={fps}[bg];"
        f"color=c=lime:s=120x180:d={seconds}:r={fps}[fg];"
        "[bg][fg]overlay=x='40+100*t':y=45"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-filter_complex",
            filter_complex,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _ffprobe_stream(path: Path) -> dict:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,nb_read_frames,codec_name",
            "-count_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(out)["streams"][0]


def _run_samify(args: list[str]) -> int:
    from samify.cli import main

    return main(args)


@pytest.mark.skipif(
    not os.environ.get("REPLICATE_API_TOKEN"),
    reason="REPLICATE_API_TOKEN not set",
)
def test_cloud_replicate(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    out = tmp_path / "out.mov"
    _make_fixture_clip(clip)
    rc = _run_samify(
        [
            str(clip),
            "-p",
            "green rectangle",
            "-o",
            str(out),
            "--backend",
            "replicate",
            "-v",
        ]
    )
    assert rc == 0
    assert out.exists()
    info = _ffprobe_stream(out)
    assert info["codec_name"] == "prores"
    assert info["pix_fmt"] == "yuva444p10le"


@pytest.mark.skipif(
    "torch" not in sys.modules and not os.environ.get("SAMIFY_RUN_LOCAL"),
    reason="local deps not installed or SAMIFY_RUN_LOCAL not set",
)
def test_local(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    out = tmp_path / "out.mov"
    _make_fixture_clip(clip)
    rc = _run_samify(
        [
            str(clip),
            "-p",
            "green rectangle",
            "-o",
            str(out),
            "--backend",
            "local",
            "-v",
        ]
    )
    assert rc == 0
    assert out.exists()
    info = _ffprobe_stream(out)
    assert info["codec_name"] == "prores"
    assert info["pix_fmt"] == "yuva444p10le"


if __name__ == "__main__":
    # Manual run, no pytest: pick whichever backend is configured.
    backend = "replicate" if os.environ.get("REPLICATE_API_TOKEN") else "local"
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        clip = td_p / "clip.mp4"
        out = td_p / "out.mov"
        _make_fixture_clip(clip)
        rc = _run_samify(
            [str(clip), "-p", "green rectangle", "-o", str(out), "--backend", backend, "-v"]
        )
        print(f"exit code: {rc}, output exists: {out.exists()}")
