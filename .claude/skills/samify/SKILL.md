---
name: samify
description: Text-prompted video segmentation using SAM 3.1. Use when the user wants to cut out a person or object from a video, create a video cutout, remove a background from video, isolate a subject, rotoscope, create a matte, segment video, or put text behind a person. Triggers on phrases like "cut me out", "remove the background", "isolate the person", "segment the video", "create a cutout", "rotoscope", "make a matte", "put text behind me".
---

# Samify — Text-Prompted Video Segmentation

## What this does

`samify` is a CLI that takes a video + text prompt and produces a ProRes 4444 `.mov` with a transparent alpha channel. The subject stays; the background becomes transparent. Drop the output into any NLE (Resolve, Premiere, After Effects, FCP) and composite text/graphics behind the subject.

## Prerequisites

- Python 3.10+ with samify installed (`pip install -e .[fal]` from the repo root).
- `FAL_KEY` env var must be set (fal.ai, SAM 3.1 — the default and recommended backend).
  - Alternative: `REPLICATE_API_TOKEN` for Replicate backend, or `--backend local` with a CUDA GPU.
- `ffmpeg` and `ffprobe` must be on PATH.

## How to use

### Translate natural language to CLI

Map the user's request to the `samify` command. The key parameters are the input file and the `-p` prompt.

```bash
# "Cut me out of this clip"
samify clip.mp4 -p "person"

# "Isolate the laptop from the b-roll"
samify broll.mp4 -p "laptop" -o laptop_cutout.mov

# "Remove the background from this interview"
samify interview.mp4 -p "person"

# "Segment both people and the table"
samify clip.mp4 -p "person, table"

# "Make a matte of just my hands"
samify clip.mp4 -p "hand"
```

### Resolve the prompt

Map natural language to a noun prompt:
- "cut me out" / "remove my background" / "isolate myself" → `-p "person"`
- "isolate the [object]" → `-p "[object]"`
- "segment everything" → not supported; samify needs a specific subject
- Multiple subjects → comma-separated: `-p "person, laptop, coffee mug"`

### Output naming

Default output is `<input_stem>_cutout.mov` next to the source. If the user specifies a name or path, use `-o`.

## All flags

| Flag | When to use |
|---|---|
| `-p "prompt"` | Always required. The thing to cut out. |
| `-o path.mov` | Custom output path. |
| `--feather 3` | User wants softer edges (default is 1). |
| `--dilate 2` | Expand the mask outward by N pixels (makes cutout bigger). |
| `--erode 2` | Shrink the mask inward by N pixels (trims edges). |
| `--threshold 128` | Binarize soft mask edges at a specific value (0-255). |
| `--invert` | Invert the mask — keep background, remove subject. |
| `--format webm` | VP9+alpha WebM output for web use (Chrome, social). |
| `--format png-seq` | PNG frames instead of .mov (for After Effects). |
| `--preview` | Quick side-by-side H.264 preview before full encode. Great for checking mask quality fast. |
| `--dry-run` | Show estimated cost and processing time without running. |
| `--cache-dir .mask_cache` | Cache masks so re-runs with different encode settings skip segmentation. |
| `--batch` | Treat input as a directory; process all videos in it. |
| `--ext mp4,mov` | (batch only) File extensions to include. |
| `--skip-existing` | (batch only) Skip files whose output already exists. |
| `--no-audio` | Drop audio from the output. |
| `--negative-prompt "shadow"` | Exclude something from the mask (Replicate/local only). |
| `--backend replicate` | Cheaper (~3x) but older SAM 3 model. |
| `--backend local` | Offline mode, runs on the user's GPU. |
| `--chunk-size 300` | (local only) Process in chunks of N frames for VRAM-limited GPUs. 0 = auto-detect. |
| `--detection-threshold 0.3` | (fal only) Lower = more sensitive detection. |
| `-v` | Verbose — shows detection details and timing. |

## Feature guide — when to suggest what

### User wants to check quality before committing
Use `--preview`. It generates a fast H.264 side-by-side (original + red mask overlay) and opens it in the system player. Much faster than waiting for ProRes encode.

```bash
samify clip.mp4 -p "person" --preview
# If it looks good, run again without --preview for full encode.
```

### User wants to know cost before running
Use `--dry-run`. Shows estimated cost and time without running segmentation.

```bash
samify clip.mp4 -p "person" --dry-run
# Output: Estimated cost: ~$0.56 (57 chunks @ $0.01), Estimated time: ~90s
```

### User wants to iterate on mask settings
Use `--cache-dir` to cache the expensive segmentation step, then re-run with different feather/dilate/format settings without paying for segmentation again.

```bash
samify clip.mp4 -p "person" --cache-dir .mask_cache
samify clip.mp4 -p "person" --cache-dir .mask_cache --feather 5  # instant, reuses mask
samify clip.mp4 -p "person" --cache-dir .mask_cache --format webm  # instant, reuses mask
```

### User wants to refine edges
Combine `--dilate`, `--erode`, `--feather`, and `--threshold` for precise mask control:

```bash
# Expand mask slightly to catch hair/fringe
samify clip.mp4 -p "person" --dilate 2 --feather 3

# Tight crop with hard edges
samify clip.mp4 -p "person" --erode 1 --feather 0

# Clean up noisy edges
samify clip.mp4 -p "person" --threshold 128 --feather 2
```

### User wants to invert (keep background, remove subject)
```bash
samify clip.mp4 -p "person" --invert
```

### User wants web-friendly output
Use `--format webm` for VP9+alpha (works in Chrome, social media):
```bash
samify clip.mp4 -p "person" --format webm
```

### User wants to process multiple clips
Use `--batch` with a directory:
```bash
samify clips/ -p "person" --batch --skip-existing
```

### Long clips on local GPU
The local backend auto-chunks based on available VRAM. Override with `--chunk-size` if needed:
```bash
samify long_clip.mp4 -p "person" --backend local --chunk-size 300
```

## What the output looks like

- **ProRes 4444 `.mov`** with RGBA (transparent background). ~10 MB/sec at 720p.
- **WebM** with VP9+alpha. Much smaller files, web-native.
- **PNG sequence** for frame-by-frame import (After Effects).
- Audio from the source is preserved by default.
- After segmentation, samify reports mask quality: detection percentage, timestamps of gaps, and warnings if coverage is low.

## Troubleshooting

- **"No backend selected"** → No API key set. Set `FAL_KEY` (recommended), `REPLICATE_API_TOKEN`, or use `--backend local`.
- **Bad mask quality** → Use `--preview` to check quickly. Try a different prompt, lower `--detection-threshold`, or use `--dilate` to expand the mask.
- **Low detection rate warning** → samify shows this after segmentation (e.g., "47% frames detected"). Suggests the prompt isn't matching well — rephrase or lower threshold.
- **"ffmpeg not found"** → Install ffmpeg and ensure it's on PATH.
- **Prompt finds nothing** → Rephrase. "person" is more reliable than "man" or "woman". Nouns work better than descriptions.
- **Multiple subjects merging** → On fal.ai, use comma-separated prompts: `-p "person on left, person on right"`.
- **VRAM OOM on local** → Use `--chunk-size 200` or lower, or switch to a cloud backend.
- **Expensive re-runs** → Use `--cache-dir` to avoid re-running segmentation when tweaking encode settings.

## Limitations

- No negative prompting on fal.ai (SAM 3.1) — `--negative-prompt` only works on Replicate/local.
- Cloud backends need internet + API key.
- Very long clips (10+ minutes) should use cloud or enable chunking on local.
- WebM encoding is slower than ProRes.
