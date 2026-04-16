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
        **_ignored,
    ) -> Path:
        # Lazy imports — keep cloud users from needing torch.
        import cv2  # type: ignore
        import imageio  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
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
        mask_frames: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for model_outputs in model.propagate_in_video_iterator(
                inference_session=session,
                max_frame_num_to_track=len(frames),
            ):
                processed = processor.postprocess_outputs(session, model_outputs)
                masks = processed.get("masks")
                if masks is None:
                    continue
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                if len(masks) == 0:
                    mask_frames[model_outputs.frame_idx] = np.zeros(
                        (height, width), dtype=np.uint8
                    )
                    continue
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
                mask_frames[model_outputs.frame_idx] = (union * 255).astype(np.uint8)
        _log(f"  propagation done in {time.time() - t0:.1f}s")

        # 4. Write a B/W mask mp4 matching the source.
        mask_path = work_dir / "mask.mp4"
        writer = imageio.get_writer(
            str(mask_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=None,
        )
        black = np.zeros((height, width), dtype=np.uint8)
        for i in range(len(frames)):
            m = mask_frames.get(i, black)
            # imageio wants RGB; replicate the grayscale to 3 channels.
            writer.append_data(np.stack([m, m, m], axis=-1))
        writer.close()

        return mask_path


def _log(msg: str) -> None:
    print(f"[samify:local] {msg}", file=sys.stderr, flush=True)
