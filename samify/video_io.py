"""ffmpeg helpers.

All backends produce a B/W mask mp4 with the same timebase as the original.
`compose_rgba_mov` takes (original, mask) and writes a ProRes 4444 .mov with
the mask as the alpha channel, optionally muxing the original audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `choco install ffmpeg` "
            "or https://ffmpeg.org/)."
        )
    return ffmpeg


def _require_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe not found on PATH (usually ships with ffmpeg).")
    return ffprobe


def probe(input_path: Path) -> dict:
    """Return {width, height, fps, n_frames, has_audio}."""
    ffprobe = _require_ffprobe()
    data = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(input_path),
            ]
        )
    )
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise RuntimeError(f"No video stream found in {input_path}")
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    fps = num / den if den else float(num)

    nb = v.get("nb_frames")
    if nb and nb.isdigit():
        n_frames = int(nb)
    else:
        dur = float(v.get("duration") or data["format"].get("duration") or 0.0)
        n_frames = int(round(dur * fps)) if dur else 0

    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "n_frames": n_frames,
        "has_audio": a is not None,
    }


def compose_rgba_mov(
    original: Path,
    mask_video: Path,
    output: Path,
    *,
    keep_audio: bool = True,
    feather: int = 0,
    fps: float | None = None,
) -> None:
    """Composite original (RGB) + mask (B/W) into ProRes 4444 RGBA .mov.

    Uses ffmpeg's alphamerge: the luma of the mask stream becomes the alpha
    channel of the first stream. Optional `feather` (pixels) softens the
    alpha edge with a gaussian blur applied only to the mask.
    """
    ffmpeg = _require_ffmpeg()

    # Some SAM providers (Replicate's cog wrapper in particular) re-encode
    # the mask mp4 with imageio/libx264, which can pad dimensions to an even
    # multiple (480x270 -> 480x272). Force the mask stream to match the
    # reference video's dimensions before alphamerge. Nearest-neighbor
    # preserves mask sharpness; the optional feather re-softens the edge.
    feather_step = (
        f",gblur=sigma={feather / 2:.3f}" if feather > 0 else ""
    )
    # setpts=PTS-STARTPTS on both streams so alphamerge doesn't drop frames
    # when the mask's container has a different start-time than the source.
    # scale2ref handles the Replicate mod-16 padding (e.g. 480x270 -> 480x272).
    filter_complex = (
        f"[0:v]setpts=PTS-STARTPTS[vref];"
        f"[1:v]setpts=PTS-STARTPTS[mref];"
        f"[mref][vref]scale2ref=flags=neighbor[m0][v];"
        f"[m0]format=gray{feather_step}[m];"
        f"[v][m]alphamerge,format=yuva444p10le[out]"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(original),
        "-i",
        str(mask_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
    ]

    # Audio from the original, if present and requested.
    if keep_audio:
        cmd += ["-map", "0:a?", "-c:a", "copy"]

    # Lock output framerate to the source; without this, ProRes encoder
    # silently falls back to 25 fps when setpts strips FPS metadata.
    if fps is None:
        fps = probe(original)["fps"]

    cmd += [
        "-r",
        f"{fps:.6f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4444",
        "-pix_fmt",
        "yuva444p10le",
        "-qscale:v",
        "9",
        "-vendor",
        "apl0",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def extract_mask_frames(mask_video: Path, out_dir: Path) -> int:
    """Decode a mask mp4 to PNG-seq for `--format png-seq` output.

    Returns frame count.
    """
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mask_video),
            "-start_number",
            "0",
            str(out_dir / "mask_%08d.png"),
        ],
        check=True,
    )
    return len(list(out_dir.glob("mask_*.png")))


def compose_rgba_png_sequence(
    original: Path,
    mask_video: Path,
    out_dir: Path,
    *,
    feather: int = 0,
) -> int:
    """Write RGBA PNGs (rgba_00000000.png …) by extracting frames from the
    alphamerge'd output. Returns frame count.
    """
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)

    feather_step = (
        f",gblur=sigma={feather / 2:.3f}" if feather > 0 else ""
    )
    filter_complex = (
        f"[0:v]setpts=PTS-STARTPTS[vref];"
        f"[1:v]setpts=PTS-STARTPTS[mref];"
        f"[mref][vref]scale2ref=flags=neighbor[m0][v];"
        f"[m0]format=gray{feather_step}[m];"
        f"[v][m]alphamerge,format=rgba[out]"
    )

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(original),
            "-i",
            str(mask_video),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-start_number",
            "0",
            str(out_dir / "rgba_%08d.png"),
        ],
        check=True,
    )
    return len(list(out_dir.glob("rgba_*.png")))
