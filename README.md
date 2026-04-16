# Samify

Text-prompted video segmentation. Give it a clip and a phrase ("person", "coffee mug", "hand"); get back a ProRes 4444 `.mov` with a clean alpha channel that drops straight into Resolve / Premiere / After Effects / FCP.

Built on **SAM 3.1** (Meta, March 2026) — text prompts are native, no grounding model needed.

**Defaults to a hosted cloud API (fal.ai).** Your GPU sits idle; a 10-second 1080p clip costs ~$0.19 and runs in ~30 seconds. Fall back to Replicate (`--backend replicate`) for cheaper batch processing, or go fully offline with `--backend local`.

## Install

```bash
cd C:\Users\lewis\samify
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e .[fal]
```

### API key setup

**fal.ai (recommended — SAM 3.1, fastest)**

```bash
# https://fal.ai/dashboard/keys — $10 free starter credit
export FAL_KEY=your-key-here
```

**Replicate (cheaper fallback — SAM 3)**

```bash
# https://replicate.com/account/api-tokens
export REPLICATE_API_TOKEN=r8_...
```

When both are set, fal.ai wins. Override with `--backend replicate` or `--backend local`.

### Local GPU (optional, no API needed)

```bash
# RTX 5090 / Blackwell needs CUDA 12.8 wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .[local]
```

First run downloads `facebook/sam3` (~1.5 GB) to `~/.cache/huggingface/`.

## Quick start

```bash
# Cut yourself out of a talking-head clip
samify talking-head.mp4 -p "person"
# -> talking-head_cutout.mov (ProRes 4444 RGBA)

# Isolate a specific object
samify b-roll.mp4 -p "coffee mug" -o mug.mov

# Multiple subjects — comma-separated for SAM 3.1
samify interview.mp4 -p "person, laptop"

# Exclude something from the mask
samify clip.mp4 -p "person" --negative-prompt "hat"

# Softer edges
samify clip.mp4 -p "person" --feather 3

# Quick preview before committing to full encode
samify clip.mp4 -p "person" --preview

# Check cost before running
samify clip.mp4 -p "person" --dry-run

# WebM output (VP9+alpha, for web)
samify clip.mp4 -p "person" --format webm

# PNG sequence for After Effects
samify clip.mp4 -p "person" --format png-seq -o cutout_frames/

# Invert the mask (keep background, remove subject)
samify clip.mp4 -p "person" --invert

# Refine mask edges
samify clip.mp4 -p "person" --dilate 2 --feather 3

# Cache masks for iterating on encoding settings
samify clip.mp4 -p "person" --cache-dir .mask_cache
samify clip.mp4 -p "person" --cache-dir .mask_cache --feather 5  # reuses cached mask

# Process a whole folder
samify clips/ -p "person" --batch --skip-existing

# Cheaper batch processing via Replicate
samify clip.mp4 -p "person" --backend replicate

# Fully offline on your GPU
samify clip.mp4 -p "person" --backend local
```

## All flags

```
samify INPUT -p PROMPT
    [-o OUTPUT]
    [--backend fal|replicate|local]
    [--format prores4444|png-seq|webm]
    [--negative-prompt TEXT]
    [--feather PX]
    [--dilate PX]                      # expand mask
    [--erode PX]                       # shrink mask
    [--threshold 0-255]                # binarize mask
    [--invert]                         # flip mask
    [--no-audio]
    [--preview]                        # quick side-by-side check
    [--dry-run]                        # cost estimate only
    [--cache-dir DIR]                  # reuse masks across runs
    [--batch]                          # process all videos in dir
    [--ext mp4,mov,...]                # batch: file extensions
    [--skip-existing]                  # batch: skip done files
    [--detection-threshold F]          # fal only
    [--model-version STR]              # replicate only
    [--hf-model STR] [--device D]      # local only
    [--chunk-size N]                   # local only: VRAM chunking
    [--keep-temp] [-v]
```

## How it works

```
INPUT VIDEO
  ├─ fal.ai (default)   → SAM 3.1 returns per-frame RLE masks → decoded to mask.mp4
  ├─ Replicate           → SAM 3 returns mask.mp4 directly (mask_only=true)
  └─ Local               → SAM 3 via HuggingFace transformers → mask.mp4

         ↓ (shared step)

ffmpeg alphamerge(original, mask.mp4) → ProRes 4444 RGBA .mov + original audio
```

1. The chosen backend produces a **B/W mask video** matching the input's frame count and fps.
2. ffmpeg's `alphamerge` filter merges the original RGB with the mask as alpha, with optional edge feather via `gblur`.
3. Mask refinement filters (dilate, erode, threshold, invert) are applied before alphamerge.
4. Output is encoded as ProRes 4444 (`yuva444p10le`), VP9+alpha WebM, or RGBA PNG sequence.
5. Progress bars and detection quality reports are shown throughout the pipeline.

## Backend comparison

| | fal.ai (default) | Replicate | Local |
|---|---|---|---|
| SAM version | **3.1** | 3 | 3 |
| Cost (10s 720p30) | ~$0.19 | ~$0.06 | Free (your GPU) |
| Speed (10s 720p30) | ~33s | ~67s | ~realtime on 5090 |
| Hosting | First-party | Community wrapper | Your machine |
| Upload limit | URL-based (large) | ~256 MB inline | N/A |
| Multi-object | Comma-separated prompts | Single prompt (unions all) | Single prompt |

## Tips

- **Multiple subjects.** On fal.ai (SAM 3.1), use comma-separated prompts: `-p "person, laptop"`. On Replicate, a single `"person"` prompt unions all detected instances automatically.
- **Subject must appear on screen.** SAM 3.1 tracks across the whole clip but needs to lock on first. If the subject enters mid-clip, trim the clip so the subject is visible in the first frame.
- **Prompt phrasing.** Keep prompts to nouns. Adjectives help distinguish ("red cup" vs "cup") but aren't always reliable. "person", "hand", "coffee mug", "laptop screen", "dog" all work well.
- **Preview first.** Use `--preview` to quickly check mask quality before committing to the full ProRes encode. This generates a fast H.264 side-by-side comparison.
- **Check cost first.** Use `--dry-run` to see estimated cost and processing time before submitting to cloud APIs.
- **Cache masks.** Use `--cache-dir .mask_cache` to avoid re-running expensive segmentation when iterating on encoding settings (feather, dilate, format).
- **Lower-third graphics.** If text overlays cross the subject's body, SAM can get confused and produce fragmented masks. Remove graphics first or use a frame without them as the prompt frame.
- **Long clips on local.** SAM 3's memory bank grows with frame count. The local backend auto-detects available VRAM and processes in chunks when needed. Use `--chunk-size N` to override.
- **Big files on Replicate.** Inline uploads cap around 256 MB. For longer clips, use `--backend fal` (URL-based upload) or trim first.
- **Mask quality report.** After segmentation, samify shows detection coverage (e.g., `93% frames detected`) with timestamps of gaps. If coverage is low, try a different prompt or lower `--detection-threshold`.

## Project structure

```
samify/
├── samify/
│   ├── cli.py              # argparse + orchestration
│   ├── video_io.py         # ffmpeg probe, alphamerge, ProRes/WebM encode
│   └── backends/
│       ├── __init__.py     # SegmentResult, auto-resolve backend from env vars
│       ├── fal_cloud.py    # fal-ai/sam-3-1/video-rle (SAM 3.1)
│       ├── replicate_cloud.py  # lucataco/sam3-video (SAM 3)
│       └── local.py        # HuggingFace transformers (SAM 3)
├── tests/
│   ├── test_smoke.py       # end-to-end integration tests
│   └── test_unit.py        # unit tests for core logic (39 tests)
├── pyproject.toml
└── README.md
```

## License

Samify is MIT-licensed. SAM 3 / 3.1 is Apache 2.0 (Meta).
