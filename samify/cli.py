"""Samify CLI.

Typical use:
    samify clip.mp4 -p "person"                    # -> clip_cutout.mov (via Replicate)
    samify clip.mp4 -p "coffee mug" --backend fal
    samify clip.mp4 -p "person" --backend local    # requires .[local] + GPU
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="samify",
        description="Text-prompted video segmentation (SAM 3). "
        "Defaults to a hosted API — add --backend local to run on your GPU.",
    )
    p.add_argument("input", type=Path, help="Input video file.")
    p.add_argument(
        "-p",
        "--prompt",
        required=True,
        help='Text describing what to segment, e.g. "person", "coffee mug", "hand".',
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <input_stem>_cutout.mov next to the input.",
    )
    p.add_argument(
        "--backend",
        choices=["replicate", "fal", "local"],
        default=None,
        help="Which SAM 3 backend to use. Default: auto — picks 'fal' (SAM 3.1) if "
        "FAL_KEY is set, else 'replicate' if REPLICATE_API_TOKEN is set.",
    )
    p.add_argument(
        "--format",
        choices=["prores4444", "png-seq"],
        default="prores4444",
        help="Output format.",
    )
    p.add_argument(
        "--negative-prompt",
        default=None,
        help='Optional text describing what to EXCLUDE, e.g. "shadow".',
    )
    p.add_argument(
        "--feather",
        type=int,
        default=1,
        help="Alpha edge feather radius in pixels (0 = hard edge).",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop the original audio track from the output .mov.",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Leave the temp working dir in place (for debugging / inspecting the raw mask).",
    )

    # Backend-specific tunables (all optional).
    p.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="[fal only] Confidence threshold (0.01-1.0).",
    )
    p.add_argument(
        "--model-version",
        default=None,
        help="[replicate only] Override the pinned model version.",
    )
    p.add_argument(
        "--hf-model",
        default="facebook/sam3",
        help="[local only] HuggingFace model ID for SAM 3 weights.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="[local only] Torch device (default: cuda if available).",
    )

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _default_output(input_path: Path, fmt: str) -> Path:
    if fmt == "prores4444":
        return input_path.with_name(f"{input_path.stem}_cutout.mov")
    return input_path.with_name(f"{input_path.stem}_cutout")


def _log(msg: str) -> None:
    print(f"[samify] {msg}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.input.exists():
        _log(f"Input not found: {args.input}")
        return 2

    output_path = args.output or _default_output(args.input, args.format)
    if args.format == "prores4444" and output_path.suffix.lower() != ".mov":
        _log(
            f"Warning: ProRes output usually ends in .mov (got {output_path.suffix})"
        )

    # Lazy imports so --help is snappy.
    from samify import backends, video_io

    backend = backends.resolve(args.backend)
    _log(f"Backend: {backend.name}")

    info = video_io.probe(args.input)
    _log(
        f"Input: {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
        f"~{info['n_frames']} frames, audio={'yes' if info['has_audio'] else 'no'}"
    )

    t0 = time.time()
    tmp_root = Path(tempfile.mkdtemp(prefix="samify_"))
    try:
        # 1. Backend produces a B/W mask video.
        mask_video = backend.segment(
            args.input,
            args.prompt,
            work_dir=tmp_root,
            negative_prompt=args.negative_prompt,
            detection_threshold=args.detection_threshold,
            model_version=args.model_version,
            hf_model=args.hf_model,
            device=args.device,
        )
        _log(f"Mask video: {mask_video}")

        # 2. Composite original + mask -> RGBA output.
        if args.format == "prores4444":
            _log(f"Encoding ProRes 4444 -> {output_path}")
            video_io.compose_rgba_mov(
                args.input,
                mask_video,
                output_path,
                keep_audio=info["has_audio"] and not args.no_audio,
                feather=args.feather,
                fps=info["fps"],
            )
        else:
            _log(f"Writing PNG sequence -> {output_path}")
            n = video_io.compose_rgba_png_sequence(
                args.input,
                mask_video,
                output_path,
                feather=args.feather,
            )
            _log(f"  wrote {n} PNGs")

    finally:
        if args.keep_temp:
            _log(f"Temp kept at {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    _log(f"Done in {time.time() - t0:.1f}s — {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
