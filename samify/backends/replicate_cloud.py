"""Replicate backend: `lucataco/sam3-video` with mask_only=True.

Docs + source: https://github.com/lucataco/cog-sam3-video
This is the simplest cloud path — one API call, returns a B/W mask mp4.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

# Pinned version (as of April 2026). Kept as a module constant so it's easy
# to bump and audit. Override per-invocation with --model-version.
DEFAULT_VERSION = (
    "lucataco/sam3-video:"
    "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269"
)


class ReplicateBackend:
    name = "replicate"

    def segment(
        self,
        input_video: Path,
        prompt: str,
        *,
        work_dir: Path,
        negative_prompt: str | None = None,
        model_version: str | None = None,
        poll_interval: float = 2.0,
        **_ignored,
    ) -> Path:
        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise RuntimeError(
                "REPLICATE_API_TOKEN not set. Get a token from "
                "https://replicate.com/account/api-tokens and export it."
            )

        # Import lazily so `samify --help` doesn't need `replicate` installed.
        import replicate

        version = model_version or DEFAULT_VERSION
        size_mb = input_video.stat().st_size / (1024 * 1024)
        _log(
            f"Uploading {input_video.name} ({size_mb:.1f} MB) to Replicate and "
            f"running {version.split(':')[0]}…"
        )
        if size_mb > 256:
            _log(
                "Warning: file is >256 MB; Replicate's inline upload may reject it. "
                "Consider trimming or re-encoding."
            )

        t0 = time.time()
        # Use predictions.create so we can stream status updates; replicate.run
        # blocks silently otherwise.
        with open(input_video, "rb") as f:
            prediction = replicate.predictions.create(
                version=version.split(":", 1)[1],
                input={
                    "video": f,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "mask_only": True,  # B/W mask-only mp4 — exactly what we want
                    "return_zip": False,
                },
            )

        last_status = None
        while prediction.status not in ("succeeded", "failed", "canceled"):
            if prediction.status != last_status:
                _log(f"  status: {prediction.status}")
                last_status = prediction.status
            time.sleep(poll_interval)
            prediction.reload()

        if prediction.status != "succeeded":
            raise RuntimeError(
                f"Replicate prediction {prediction.status}: "
                f"{prediction.error or '(no error message)'}"
            )

        # Output is a single URL string (or a FileOutput in newer client
        # versions — both str() to the URL).
        mask_url = str(prediction.output)
        _log(f"  finished in {time.time() - t0:.1f}s — downloading mask…")

        mask_path = work_dir / "mask.mp4"
        _download(mask_url, mask_path)
        return mask_path


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _log(msg: str) -> None:
    print(f"[samify:replicate] {msg}", file=sys.stderr, flush=True)
