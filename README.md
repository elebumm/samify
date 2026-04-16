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

# PNG sequence for After Effects
samify clip.mp4 -p "person" --format png-seq -o cutout_frames/

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
    [--format prores4444|png-seq]
    [--negative-prompt TEXT]
    [--feather PX]
    [--no-audio]
    [--detection-threshold F]      # fal only
    [--model-version STR]          # replicate only
    [--hf-model STR] [--device D]  # local only
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
3. Output is encoded as ProRes 4444 (`yuva444p10le`) with the original audio track copied through unchanged.
4. Every backend produces the same mask format, so compositing is identical everywhere.

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
- **Subject must appear on screen.** SAM 3.1 tracks across the whole clip but needs to lock on first. If the subject enters mid-clip, use `--prompt-frame N` (local backend only) or trim the clip.
- **Prompt phrasing.** Keep prompts to nouns. Adjectives help distinguish ("red cup" vs "cup") but aren't always reliable. "person", "hand", "coffee mug", "laptop screen", "dog" all work well.
- **Lower-third graphics.** If text overlays cross the subject's body, SAM can get confused and produce fragmented masks. Remove graphics first or use a frame without them as the prompt frame.
- **Long clips on local.** SAM 3's memory bank grows with frame count; 32 GB VRAM handles a few minutes of 1080p. For 10+ minute clips, prefer a cloud backend.
- **Big files on Replicate.** Inline uploads cap around 256 MB. For longer clips, use `--backend fal` (URL-based upload) or trim first.

## Project structure

```
samify/
├── samify/
│   ├── cli.py              # argparse + orchestration
│   ├── video_io.py         # ffmpeg probe, alphamerge, ProRes encode
│   └── backends/
│       ├── __init__.py     # auto-resolve backend from env vars
│       ├── fal_cloud.py    # fal-ai/sam-3-1/video-rle (SAM 3.1)
│       ├── replicate_cloud.py  # lucataco/sam3-video (SAM 3)
│       └── local.py        # HuggingFace transformers (SAM 3)
├── tests/
│   └── test_smoke.py
├── pyproject.toml
└── README.md
```

## License

Samify is MIT-licensed. SAM 3 / 3.1 is Apache 2.0 (Meta).
