"""Samify CLI.

Typical use:
    samify clip.mp4 -p "person"                    # -> clip_cutout.mov (via fal.ai)
    samify clip.mp4 -p "coffee mug" --backend fal
    samify clip.mp4 -p "person" --backend local    # requires .[local] + GPU
    samify clip.mp4 -p "person" --dry-run          # estimate cost without running
    samify clip.mp4 -p "person" --preview          # quick preview before full encode
    samify batch clips/ -p "person"                # process a whole folder
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="samify",
        description="Text-prompted video segmentation (SAM 3 / 3.1). "
        "Defaults to a hosted API — add --backend local to run on your GPU.",
    )
    p.add_argument("input", type=Path, help="Input video file or directory (with --batch).")
    p.add_argument(
        "--batch",
        action="store_true",
        help="Treat input as a directory and process all videos in it.",
    )
    p.add_argument(
        "--ext",
        default="mp4,mov,mkv,avi,webm",
        help="[batch only] Comma-separated extensions to process (default: mp4,mov,mkv,avi,webm).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="[batch only] Skip videos whose output file already exists.",
    )
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
        help="Which SAM backend to use. Default: auto — picks 'fal' (SAM 3.1) if "
        "FAL_KEY is set, else 'replicate' if REPLICATE_API_TOKEN is set.",
    )
    p.add_argument(
        "--format",
        choices=["prores4444", "png-seq", "webm"],
        default="prores4444",
        help="Output format (default: prores4444).",
    )

    # Mask refinement.
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
        "--dilate",
        type=int,
        default=0,
        help="Expand the mask by N pixels (morphological dilation).",
    )
    p.add_argument(
        "--erode",
        type=int,
        default=0,
        help="Shrink the mask by N pixels (morphological erosion).",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Binarize the mask at this value (0-255). 0 = no threshold.",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert the mask (keep background, remove subject).",
    )

    # Workflow options.
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop the original audio track from the output .mov.",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Generate a quick side-by-side preview instead of full encode.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cost/time estimate without running segmentation.",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Leave the temp working dir in place (for debugging / inspecting the raw mask).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache masks to this directory. Re-runs with the same input+prompt reuse cached masks.",
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
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="[local only] Process in chunks of N frames (0 = auto-detect from VRAM).",
    )

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _default_output(input_path: Path, fmt: str) -> Path:
    if fmt == "prores4444":
        return input_path.with_name(f"{input_path.stem}_cutout.mov")
    if fmt == "webm":
        return input_path.with_name(f"{input_path.stem}_cutout.webm")
    return input_path.with_name(f"{input_path.stem}_cutout")


def _cache_key(input_path: Path, prompt: str, backend_name: str) -> str:
    """Deterministic hash for mask caching."""
    h = hashlib.sha256()
    h.update(str(input_path.resolve()).encode())
    h.update(str(input_path.stat().st_size).encode())
    h.update(str(input_path.stat().st_mtime_ns).encode())
    h.update(prompt.encode())
    h.update(backend_name.encode())
    return h.hexdigest()[:16]


def _report_quality(result, fps: float) -> None:
    """Print mask quality summary with detection stats."""
    from samify.backends import SegmentResult

    if result.total_frames == 0:
        return

    pct = result.detected_frames / result.total_frames * 100
    empty = result.total_frames - result.detected_frames

    if pct >= 90:
        color = "\033[92m"  # Green
    elif pct >= 60:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red
    reset = "\033[0m"

    _log(
        f"Mask quality: {color}{result.detected_frames}/{result.total_frames} "
        f"frames ({pct:.0f}%){reset}"
    )

    if empty > 0 and result.empty_ranges:
        gaps = []
        for start, end in result.empty_ranges[:5]:  # Show up to 5 gaps.
            t_start = start / fps if fps else start
            t_end = end / fps if fps else end
            if start == end:
                gaps.append(f"{t_start:.1f}s")
            else:
                gaps.append(f"{t_start:.1f}–{t_end:.1f}s")
        gap_str = ", ".join(gaps)
        if len(result.empty_ranges) > 5:
            gap_str += f" (+{len(result.empty_ranges) - 5} more)"
        _log(f"  {empty} empty frames at: {gap_str}")

    if pct < 50:
        _log(
            "  Warning: low detection rate. Try a different prompt, "
            "lower --detection-threshold, or check that the subject is visible."
        )


def _log(msg: str) -> None:
    print(f"[samify] {msg}", file=sys.stderr, flush=True)


def _process_single(args: argparse.Namespace, input_path: Path) -> int:
    """Process a single video file. Returns exit code."""
    output_path = args.output or _default_output(input_path, args.format)
    if args.format == "prores4444" and output_path.suffix.lower() != ".mov":
        _log(
            f"Warning: ProRes output usually ends in .mov (got {output_path.suffix})"
        )
    if args.format == "webm" and output_path.suffix.lower() != ".webm":
        _log(
            f"Warning: WebM output usually ends in .webm (got {output_path.suffix})"
        )

    # Lazy imports so --help is snappy.
    from samify import backends, video_io

    backend = backends.resolve(args.backend)
    _log(f"Backend: {backend.name}")

    info = video_io.probe(input_path)
    _log(
        f"Input: {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
        f"~{info['n_frames']} frames, audio={'yes' if info['has_audio'] else 'no'}"
    )

    # --dry-run: show cost estimate and exit.
    if args.dry_run:
        est = backend.estimate_cost(info["n_frames"], info["fps"])
        if est:
            _log(f"Backend: {est['backend']}")
            _log(f"Estimated cost: {est['cost_str']}")
            _log(f"Estimated time: {est['time_str']}")
        else:
            _log("Cost estimation not available for this backend.")
        return 0

    t0 = time.time()

    # Check mask cache.
    cached_mask = None
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(input_path, args.prompt, backend.name)
        cached_mask = args.cache_dir / f"{key}.mp4"
        if cached_mask.exists():
            _log(f"Using cached mask: {cached_mask}")

    tmp_root = Path(tempfile.mkdtemp(prefix="samify_"))
    try:
        # 1. Backend produces a B/W mask video (or use cache).
        if cached_mask and cached_mask.exists():
            mask_video = cached_mask
            # No SegmentResult stats from cache.
            from samify.backends import SegmentResult
            result = SegmentResult(mask_video=mask_video)
        else:
            result = backend.segment(
                input_path,
                args.prompt,
                work_dir=tmp_root,
                negative_prompt=args.negative_prompt,
                detection_threshold=args.detection_threshold,
                model_version=args.model_version,
                hf_model=args.hf_model,
                device=args.device,
                chunk_size=args.chunk_size,
            )
            mask_video = result.mask_video
            _log(f"Mask video: {mask_video}")

            # Report quality.
            _report_quality(result, info["fps"])

            # Save to cache.
            if cached_mask and not cached_mask.exists():
                shutil.copy2(mask_video, cached_mask)
                _log(f"Mask cached: {cached_mask}")

        # 2. Preview mode: quick side-by-side and exit.
        if args.preview:
            preview_path = tmp_root / "preview.mp4"
            _log("Generating preview…")
            video_io.compose_preview(input_path, mask_video, preview_path, fps=info["fps"])
            # Copy preview next to input.
            final_preview = input_path.with_name(f"{input_path.stem}_preview.mp4")
            shutil.copy2(preview_path, final_preview)
            _log(f"Preview saved: {final_preview}")
            # Try to open with system default player.
            _open_file(final_preview)
            return 0

        # 3. Composite original + mask -> final output.
        mask_opts = dict(
            feather=args.feather,
            dilate=args.dilate,
            erode=args.erode,
            threshold=args.threshold,
            invert=args.invert,
        )

        if args.format == "prores4444":
            _log(f"Encoding ProRes 4444 -> {output_path}")
            video_io.compose_rgba_mov(
                input_path,
                mask_video,
                output_path,
                keep_audio=info["has_audio"] and not args.no_audio,
                fps=info["fps"],
                **mask_opts,
            )
        elif args.format == "webm":
            _log(f"Encoding VP9+alpha WebM -> {output_path}")
            video_io.compose_rgba_webm(
                input_path,
                mask_video,
                output_path,
                keep_audio=info["has_audio"] and not args.no_audio,
                fps=info["fps"],
                **mask_opts,
            )
        else:
            _log(f"Writing PNG sequence -> {output_path}")
            n = video_io.compose_rgba_png_sequence(
                input_path,
                mask_video,
                output_path,
                **mask_opts,
            )
            _log(f"  wrote {n} PNGs")

    finally:
        if args.keep_temp:
            _log(f"Temp kept at {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    _log(f"Done in {time.time() - t0:.1f}s — {output_path}")
    return 0


def _open_file(path: Path) -> None:
    """Open a file with the system default application (best-effort)."""
    import platform
    import subprocess

    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass  # Non-critical; user can open manually.


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        _log(f"Input not found: {args.input}")
        return 2

    # Batch mode.
    if args.batch:
        return _batch_main(args)

    return _process_single(args, args.input)


def _batch_main(args: argparse.Namespace) -> int:
    """Process all videos in a directory."""
    input_dir = args.input
    if not input_dir.is_dir():
        _log(f"Not a directory: {input_dir}")
        return 2

    extensions = [e.strip().lower() for e in args.ext.split(",")]
    videos = []
    for ext in extensions:
        videos.extend(input_dir.glob(f"*.{ext}"))
    videos.sort()

    if not videos:
        _log(f"No video files found in {input_dir} (extensions: {args.ext})")
        return 1

    _log(f"Found {len(videos)} video(s) to process")

    results = []
    for i, video in enumerate(videos, 1):
        _log(f"\n{'='*60}")
        _log(f"[{i}/{len(videos)}] {video.name}")
        _log(f"{'='*60}")

        output = _default_output(video, args.format)
        if args.skip_existing and output.exists():
            _log(f"  Skipping (output exists): {output}")
            results.append((video.name, "skipped", 0))
            continue

        # Override output for this file.
        args.output = None  # Let _process_single use default naming.
        t0 = time.time()
        try:
            code = _process_single(args, video)
            elapsed = time.time() - t0
            results.append((video.name, "ok" if code == 0 else f"exit {code}", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            _log(f"  Error: {e}")
            results.append((video.name, f"error: {e}", elapsed))

    # Summary.
    _log(f"\n{'='*60}")
    _log("Batch summary:")
    for name, status, elapsed in results:
        _log(f"  {name}: {status} ({elapsed:.1f}s)")
    ok = sum(1 for _, s, _ in results if s == "ok")
    _log(f"  {ok}/{len(results)} succeeded")

    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
