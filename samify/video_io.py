"""ffmpeg helpers.

All backends produce a B/W mask mp4 with the same timebase as the original.
`compose_rgba_mov` takes (original, mask) and writes a ProRes 4444 .mov with
the mask as the alpha channel, optionally muxing the original audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
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


def _build_mask_filter(
    feather: int = 0,
    dilate: int = 0,
    erode: int = 0,
    threshold: int = 0,
    invert: bool = False,
) -> str:
    """Build the mask processing filter chain inserted between scale2ref and alphamerge."""
    steps = ["format=gray"]

    # Morphological operations (applied to mask before feathering).
    if erode > 0:
        for _ in range(erode):
            steps.append("erosion")
    if dilate > 0:
        for _ in range(dilate):
            steps.append("dilation")

    # Threshold: binarize at a specific value.
    if threshold > 0:
        steps.append(f"lutrgb=r='if(val,if(gt(val,{threshold}),255,0),0)'"
                      f":g='if(val,if(gt(val,{threshold}),255,0),0)'"
                      f":b='if(val,if(gt(val,{threshold}),255,0),0)'")

    # Feather (gaussian blur on mask edges).
    if feather > 0:
        steps.append(f"gblur=sigma={feather / 2:.3f}")

    # Invert mask.
    if invert:
        steps.append("negate")

    return ",".join(steps)


def compose_rgba_mov(
    original: Path,
    mask_video: Path,
    output: Path,
    *,
    keep_audio: bool = True,
    feather: int = 0,
    fps: float | None = None,
    dilate: int = 0,
    erode: int = 0,
    threshold: int = 0,
    invert: bool = False,
) -> None:
    """Composite original (RGB) + mask (B/W) into ProRes 4444 RGBA .mov.

    Uses ffmpeg's alphamerge: the luma of the mask stream becomes the alpha
    channel of the first stream. Optional `feather` (pixels) softens the
    alpha edge with a gaussian blur applied only to the mask.
    """
    ffmpeg = _require_ffmpeg()

    mask_filter = _build_mask_filter(feather, dilate, erode, threshold, invert)

    # setpts=PTS-STARTPTS on both streams so alphamerge doesn't drop frames
    # when the mask's container has a different start-time than the source.
    # scale2ref handles the Replicate mod-16 padding (e.g. 480x270 -> 480x272).
    filter_complex = (
        f"[0:v]setpts=PTS-STARTPTS[vref];"
        f"[1:v]setpts=PTS-STARTPTS[mref];"
        f"[mref][vref]scale2ref=flags=neighbor[m0][v];"
        f"[m0]{mask_filter}[m];"
        f"[v][m]alphamerge,format=yuva444p10le[out]"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-stats",
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


def compose_rgba_webm(
    original: Path,
    mask_video: Path,
    output: Path,
    *,
    keep_audio: bool = True,
    feather: int = 0,
    fps: float | None = None,
    dilate: int = 0,
    erode: int = 0,
    threshold: int = 0,
    invert: bool = False,
) -> None:
    """Composite original (RGB) + mask (B/W) into VP9+alpha WebM."""
    ffmpeg = _require_ffmpeg()

    mask_filter = _build_mask_filter(feather, dilate, erode, threshold, invert)

    filter_complex = (
        f"[0:v]setpts=PTS-STARTPTS[vref];"
        f"[1:v]setpts=PTS-STARTPTS[mref];"
        f"[mref][vref]scale2ref=flags=neighbor[m0][v];"
        f"[m0]{mask_filter}[m];"
        f"[v][m]alphamerge,format=yuva420p[out]"
    )

    if fps is None:
        fps = probe(original)["fps"]

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-i",
        str(original),
        "-i",
        str(mask_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
    ]

    if keep_audio:
        cmd += ["-map", "0:a?", "-c:a", "libopus"]

    cmd += [
        "-r",
        f"{fps:.6f}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-b:v",
        "2M",
        "-auto-alt-ref",
        "0",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def compose_preview(
    original: Path,
    mask_video: Path,
    output: Path,
    *,
    fps: float | None = None,
) -> None:
    """Generate a quick side-by-side preview: original left, red mask overlay right.

    Uses fast H.264 encoding at reduced quality for quick iteration.
    """
    ffmpeg = _require_ffmpeg()

    if fps is None:
        fps = probe(original)["fps"]

    # Overlay mask as semi-transparent red on original, stack horizontally.
    filter_complex = (
        "[0:v]setpts=PTS-STARTPTS[v];"
        "[1:v]setpts=PTS-STARTPTS[mraw];"
        "[mraw][v]scale2ref=flags=neighbor[m][vref];"
        # Create red overlay from mask.
        "[m]format=gray,colorize=hue=0:saturation=1:lightness=0.5[red];"
        "[vref][red]overlay=format=auto:alpha=premultiplied[overlaid];"
        "[v][overlaid]hstack[out]"
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
        "-r",
        f"{fps:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
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
    dilate: int = 0,
    erode: int = 0,
    threshold: int = 0,
    invert: bool = False,
) -> int:
    """Write RGBA PNGs (rgba_00000000.png …) by extracting frames from the
    alphamerge'd output. Returns frame count.
    """
    ffmpeg = _require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_filter = _build_mask_filter(feather, dilate, erode, threshold, invert)

    filter_complex = (
        f"[0:v]setpts=PTS-STARTPTS[vref];"
        f"[1:v]setpts=PTS-STARTPTS[mref];"
        f"[mref][vref]scale2ref=flags=neighbor[m0][v];"
        f"[m0]{mask_filter}[m];"
        f"[v][m]alphamerge,format=rgba[out]"
    )

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-stats",
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
