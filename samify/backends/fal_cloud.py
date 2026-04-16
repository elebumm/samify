"""fal.ai backend: `fal-ai/sam-3-1/video-rle` — SAM 3.1 with per-frame masks.

Response shape (discovered empirically — their docs don't spell this out):
    {
      "rle":       [str, ...]      # one per frame, space-separated uncompressed
                                   # "start length start length ..." pairs over a
                                   # COLUMN-MAJOR (Fortran) flat H*W buffer.
      "metadata":  [{index, score, box}, ...]   # per-frame (box = normalized xyxy)
      "boxes":     [...]           # alt per-frame boxes
      "scores":    None | [...]
      "boundingbox_frames_zip": None | File
    }

Multi-object: prompts are COMMA-SEPARATED, e.g. "person, laptop".
Pricing: $0.01 per 16 frames (≈$0.019/sec at 30fps).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "fal-ai/sam-3-1/video-rle"


class FalBackend:
    name = "fal"

    def segment(
        self,
        input_video: Path,
        prompt: str,
        *,
        work_dir: Path,
        negative_prompt: str | None = None,
        detection_threshold: float = 0.5,
        model: str | None = None,
        **_ignored,
    ) -> Path:
        if not os.environ.get("FAL_KEY"):
            raise RuntimeError(
                "FAL_KEY not set. Get a key from https://fal.ai/dashboard/keys."
            )
        if negative_prompt:
            _log(
                "Note: fal.ai SAM 3.1 has no negative_prompt. Ignoring "
                f"(value was {negative_prompt!r})."
            )

        import fal_client
        import numpy as np
        from PIL import Image

        model_id = model or DEFAULT_MODEL
        width, height = _probe_dims(input_video)
        fps = _probe_fps(input_video)
        _log(f"Uploading {input_video.name}…")
        t0 = time.time()
        video_url = fal_client.upload_file(str(input_video))

        _log(f"Running {model_id} with prompt={prompt!r}…")
        handler = fal_client.submit(
            model_id,
            arguments={
                "video_url": video_url,
                "prompt": prompt,
                "apply_mask": False,
                "detection_threshold": detection_threshold,
            },
        )
        last_status = None
        for event in handler.iter_events(with_logs=False):
            status = type(event).__name__
            if status != last_status:
                _log(f"  status: {status}")
                last_status = status
        result = handler.get()
        _log(f"  finished in {time.time() - t0:.1f}s")

        rle_list = result.get("rle") or []
        if not isinstance(rle_list, list):
            rle_list = [rle_list]
        _log(f"  got {len(rle_list)} frame(s) of RLE; decoding to mask video…")

        # Decode every frame.  Empty / None RLE -> black frame (subject missing).
        masks_dir = work_dir / "mask_frames"
        masks_dir.mkdir(parents=True, exist_ok=True)
        empty_frames = 0
        for i, frame_rle in enumerate(rle_list):
            if not frame_rle:
                mask = np.zeros((height, width), dtype=np.uint8)
                empty_frames += 1
            else:
                mask = _decode_uncompressed_rle(frame_rle, width, height)
            Image.fromarray(mask, mode="L").save(masks_dir / f"{i:08d}.png")
        if empty_frames:
            _log(f"  {empty_frames} frames had no detection (black masks)")

        mask_video = work_dir / "mask.mp4"
        _encode_mask_video(masks_dir, mask_video, fps)
        return mask_video


def _decode_uncompressed_rle(rle_str: str, width: int, height: int) -> "any":
    """Decode SAM 3.1's 'start length start length …' RLE into an HxW uint8 mask.

    Pixels are addressed in COLUMN-MAJOR order (Fortran) on a flat H*W buffer
    — the SAM/COCO convention. Out-of-bounds runs are clipped defensively.
    """
    import numpy as np

    tokens = rle_str.split()
    if len(tokens) % 2 != 0:
        # Malformed — return black frame rather than crash the pipeline.
        return np.zeros((height, width), dtype=np.uint8)
    nums = np.fromiter((int(t) for t in tokens), dtype=np.int64, count=len(tokens))
    starts = nums[0::2]
    lengths = nums[1::2]

    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        if s >= total:
            continue
        end = min(s + l, total)
        flat[s:end] = 255

    # Row-major: flat[i] == pixel at (y=i//W, x=i%W).
    return flat.reshape((height, width))


def _probe_dims(video: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", str(video),
        ]
    ).decode().strip()
    w, h = out.split("x")
    return int(w), int(h)


def _probe_fps(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=nokey=1:noprint_wrappers=1", str(video),
        ]
    ).decode().strip()
    num, den = (int(x) for x in out.split("/"))
    return num / den if den else float(num)


def _encode_mask_video(frames_dir: Path, out_path: Path, fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{fps:.6f}",
            "-start_number", "0",
            "-i", str(frames_dir / "%08d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        check=True,
    )


def _log(msg: str) -> None:
    print(f"[samify:fal] {msg}", file=sys.stderr, flush=True)
