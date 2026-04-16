"""Local backend: SAM 3 via HuggingFace `transformers`.

Requires `pip install -e .[local]` plus a CUDA-capable PyTorch
(on RTX 5090: install with `--index-url https://download.pytorch.org/whl/cu128`).

Produces a B/W mask mp4 matching the input's frame count and fps so the
rest of the pipeline can treat cloud and local identically.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from samify.backends import SegmentResult


class LocalBackend:
    name = "local"

    def segment(
        self,
        input_video: Path,
        prompt: str,
        *,
        work_dir: Path,
        negative_prompt: str | None = None,
        device: str | None = None,
        hf_model: str = "facebook/sam3",
        chunk_size: int = 0,
        **_ignored,
    ) -> SegmentResult:
        # Lazy imports — keep cloud users from needing torch.
        import cv2  # type: ignore
        import imageio  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from tqdm import tqdm
        from transformers import Sam3VideoModel, Sam3VideoProcessor  # type: ignore

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float16
            if device.startswith("cuda")
            else torch.float32
        )

        _log(f"Loading SAM 3 ({hf_model}) on {device} ({dtype})…")
        t0 = time.time()
        model = Sam3VideoModel.from_pretrained(hf_model).to(device, dtype=dtype).eval()
        processor = Sam3VideoProcessor.from_pretrained(hf_model)
        _log(f"  loaded in {time.time() - t0:.1f}s")

        # 1. Load frames via OpenCV (matches the official cog predict.py flow).
        cap = cv2.VideoCapture(str(input_video))
        frames: list[Image.Image] = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from {input_video}")
        height, width = np.array(frames[0]).shape[:2]
        _log(f"Loaded {len(frames)} frames @ {fps:.3f} fps ({width}x{height})")

        total_frames = len(frames)

        # Auto-chunk if requested or auto-detect based on VRAM.
        if chunk_size <= 0 and device.startswith("cuda"):
            chunk_size = _auto_chunk_size(width, height, device)
        if chunk_size > 0 and total_frames > chunk_size:
            return self._segment_chunked(
                model, processor, frames, prompt, negative_prompt,
                device, dtype, width, height, fps, chunk_size, work_dir,
            )

        # 2. Session + text prompt.
        session = processor.init_video_session(
            video=frames,
            inference_device=device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=dtype,
        )
        session = processor.add_text_prompt(inference_session=session, text=prompt)
        if negative_prompt:
            # Best-effort: transformers' API name varies; try the obvious one.
            try:
                session = processor.add_text_prompt(
                    inference_session=session, text=negative_prompt, negative=True
                )
            except TypeError:
                _log(
                    "Warning: this transformers version doesn't accept a negative "
                    "text prompt; ignoring."
                )

        # 3. Propagate.
        _log("Propagating masks across frames…")
        t0 = time.time()
        mask_frames, empty_frames, empty_ranges = _propagate_masks(
            model, processor, session, frames, width, height, cv2,
        )
        _log(f"  propagation done in {time.time() - t0:.1f}s")

        # 4. Write a B/W mask mp4 matching the source.
        mask_path = work_dir / "mask.mp4"
        _write_mask_video(mask_frames, total_frames, height, width, fps, mask_path)

        return SegmentResult(
            mask_video=mask_path,
            total_frames=total_frames,
            detected_frames=total_frames - empty_frames,
            empty_ranges=empty_ranges,
        )

    def _segment_chunked(
        self, model, processor, frames, prompt, negative_prompt,
        device, dtype, width, height, fps, chunk_size, work_dir,
    ):
        """Process video in chunks to stay within VRAM limits."""
        import cv2  # type: ignore
        import imageio  # type: ignore
        import numpy as np  # type: ignore
        from tqdm import tqdm

        total_frames = len(frames)
        overlap = min(30, chunk_size // 10)
        _log(
            f"Chunked processing: {total_frames} frames in chunks of "
            f"{chunk_size} (overlap {overlap})"
        )

        all_masks: dict[int, np.ndarray] = {}
        total_empty = 0
        all_empty_ranges: list[tuple[int, int]] = []

        start = 0
        chunk_idx = 0
        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunk_frames = frames[start:end]
            _log(f"  chunk {chunk_idx}: frames {start}–{end - 1}")

            session = processor.init_video_session(
                video=chunk_frames,
                inference_device=device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=dtype,
            )
            session = processor.add_text_prompt(
                inference_session=session, text=prompt
            )
            if negative_prompt:
                try:
                    session = processor.add_text_prompt(
                        inference_session=session, text=negative_prompt, negative=True
                    )
                except TypeError:
                    pass

            chunk_masks, chunk_empty, chunk_ranges = _propagate_masks(
                model, processor, session, chunk_frames, width, height, cv2,
            )

            # Map chunk-local frame indices to global.
            for local_idx, mask in chunk_masks.items():
                global_idx = start + local_idx
                if global_idx in all_masks and local_idx < overlap:
                    # Blend overlap region: average the two masks.
                    blend_weight = local_idx / overlap
                    blended = (
                        all_masks[global_idx].astype(np.float32) * (1 - blend_weight)
                        + mask.astype(np.float32) * blend_weight
                    )
                    all_masks[global_idx] = blended.astype(np.uint8)
                else:
                    all_masks[global_idx] = mask

            total_empty += chunk_empty
            for rs, re in chunk_ranges:
                all_empty_ranges.append((start + rs, start + re))

            start = end - overlap if end < total_frames else total_frames
            chunk_idx += 1

        mask_path = work_dir / "mask.mp4"
        _write_mask_video(all_masks, total_frames, height, width, fps, mask_path)

        return SegmentResult(
            mask_video=mask_path,
            total_frames=total_frames,
            detected_frames=total_frames - total_empty,
            empty_ranges=all_empty_ranges,
        )

    @staticmethod
    def estimate_cost(n_frames: int, fps: float) -> dict[str, str | float]:
        duration = n_frames / fps if fps else 0
        return {
            "backend": "Local GPU (SAM 3)",
            "cost": 0.0,
            "cost_str": "Free (your GPU)",
            "time_str": f"~{duration:.0f}s (realtime on RTX 5090)",
        }


def _propagate_masks(model, processor, session, frames, width, height, cv2):
    """Run mask propagation and return (mask_dict, empty_count, empty_ranges)."""
    import numpy as np
    import torch
    from tqdm import tqdm

    mask_frames: dict[int, np.ndarray] = {}
    empty_frames = 0
    empty_ranges: list[tuple[int, int]] = []
    in_empty_run = False
    empty_run_start = 0

    with torch.inference_mode():
        pbar = tqdm(
            total=len(frames), desc="Propagating", unit="frame", file=sys.stderr
        )
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(frames),
        ):
            processed = processor.postprocess_outputs(session, model_outputs)
            masks = processed.get("masks")
            idx = model_outputs.frame_idx

            if masks is None or len(masks) == 0:
                mask_frames[idx] = np.zeros((height, width), dtype=np.uint8)
                empty_frames += 1
                if not in_empty_run:
                    in_empty_run = True
                    empty_run_start = idx
            else:
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                union = np.zeros((height, width), dtype=bool)
                for m in masks:
                    m = np.squeeze(m)
                    if m.shape != (height, width):
                        m = cv2.resize(
                            m.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    union |= m > 0.0
                mask_frames[idx] = (union * 255).astype(np.uint8)
                if in_empty_run:
                    empty_ranges.append((empty_run_start, idx - 1))
                    in_empty_run = False

            pbar.update(1)
        pbar.close()

    if in_empty_run:
        empty_ranges.append((empty_run_start, len(frames) - 1))

    return mask_frames, empty_frames, empty_ranges


def _write_mask_video(mask_frames, total_frames, height, width, fps, mask_path):
    """Write dict of mask frames to a B/W mp4."""
    import imageio  # type: ignore
    import numpy as np  # type: ignore
    from tqdm import tqdm

    writer = imageio.get_writer(
        str(mask_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=None,
    )
    black = np.zeros((height, width), dtype=np.uint8)
    for i in tqdm(
        range(total_frames), desc="Writing mask", unit="frame", file=sys.stderr
    ):
        m = mask_frames.get(i, black)
        # imageio wants RGB; replicate the grayscale to 3 channels.
        writer.append_data(np.stack([m, m, m], axis=-1))
    writer.close()


def _auto_chunk_size(width: int, height: int, device: str) -> int:
    """Estimate a safe chunk size from available VRAM."""
    import torch

    try:
        free, total = torch.cuda.mem_get_info(
            torch.device(device) if device != "cuda" else 0
        )
    except Exception:
        return 0  # Can't detect — don't chunk.

    # Rough heuristic: SAM 3 uses ~50 MB per 1080p frame in its memory bank.
    # Scale by resolution ratio.
    pixels = width * height
    ref_pixels = 1920 * 1080
    mb_per_frame = 50 * (pixels / ref_pixels)
    # Reserve 4 GB for model weights + overhead.
    available_mb = (free / (1024 * 1024)) - 4096
    if available_mb <= 0:
        return 300  # Conservative fallback.
    chunk = int(available_mb / mb_per_frame)
    return max(100, min(chunk, 10000))  # Clamp to [100, 10000].


def _log(msg: str) -> None:
    print(f"[samify:local] {msg}", file=sys.stderr, flush=True)
