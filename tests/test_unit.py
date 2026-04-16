"""Unit tests for samify core logic.

These tests don't require any API keys, GPU, or heavy dependencies.
They test the pure-logic functions: RLE decoding, output path defaults,
backend resolution, mask filter construction, cache key generation,
and the SegmentResult dataclass.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# RLE decoding (fal backend)
# ---------------------------------------------------------------------------

class TestDecodeUncompressedRLE:
    """Test _decode_uncompressed_rle from the fal backend."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from samify.backends.fal_cloud import _decode_uncompressed_rle
        self.decode = _decode_uncompressed_rle

    def test_simple_square(self):
        """A 4x4 image with the top-left 2x2 block filled."""
        import numpy as np
        # Column-major: fill positions 0,1 (col 0 rows 0-1) and 4,5 (col 1 rows 0-1)
        rle = "0 2 4 2"
        mask = self.decode(rle, width=4, height=4)
        assert mask.shape == (4, 4)
        assert mask.dtype == np.uint8
        # Filled positions in the flat buffer.
        flat = mask.flatten()
        assert flat[0] == 255
        assert flat[1] == 255
        assert flat[4] == 255
        assert flat[5] == 255
        # Rest should be 0.
        assert flat[2] == 0
        assert flat[3] == 0

    def test_empty_rle(self):
        """Empty string should produce all-black mask."""
        import numpy as np
        # Empty RLE has odd token count (0), but we handle it.
        mask = self.decode("", width=4, height=4)
        assert mask.shape == (4, 4)
        assert mask.max() == 0

    def test_malformed_odd_tokens(self):
        """Odd number of tokens should return black frame."""
        import numpy as np
        mask = self.decode("0 2 4", width=4, height=4)
        assert mask.shape == (4, 4)
        assert mask.max() == 0

    def test_full_frame(self):
        """RLE covering the entire frame."""
        import numpy as np
        mask = self.decode("0 16", width=4, height=4)
        assert mask.shape == (4, 4)
        assert mask.min() == 255

    def test_out_of_bounds_clipped(self):
        """Runs extending past the frame should be clipped, not crash."""
        import numpy as np
        # Start at position 14, length 10 — but only 16 pixels total.
        mask = self.decode("14 10", width=4, height=4)
        flat = mask.flatten()
        assert flat[14] == 255
        assert flat[15] == 255
        # Didn't crash.

    def test_start_beyond_frame(self):
        """Start position beyond frame size should produce no output."""
        import numpy as np
        mask = self.decode("100 5", width=4, height=4)
        assert mask.max() == 0


# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------

class TestDefaultOutput:
    @pytest.fixture(autouse=True)
    def _import(self):
        from samify.cli import _default_output
        self.default_output = _default_output

    def test_prores(self):
        p = self.default_output(Path("video.mp4"), "prores4444")
        assert p == Path("video_cutout.mov")

    def test_webm(self):
        p = self.default_output(Path("clip.mp4"), "webm")
        assert p == Path("clip_cutout.webm")

    def test_png_seq(self):
        p = self.default_output(Path("clip.mp4"), "png-seq")
        assert p == Path("clip_cutout")

    def test_nested_path(self):
        p = self.default_output(Path("/a/b/clip.mov"), "prores4444")
        assert p == Path("/a/b/clip_cutout.mov")


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

class TestBackendResolve:
    def test_explicit_fal(self):
        from samify.backends import resolve
        b = resolve("fal")
        assert b.name == "fal"

    def test_explicit_replicate(self):
        from samify.backends import resolve
        b = resolve("replicate")
        assert b.name == "replicate"

    def test_explicit_local(self):
        from samify.backends import resolve
        b = resolve("local")
        assert b.name == "local"

    def test_unknown_backend(self):
        from samify.backends import resolve
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve("nonexistent")

    def test_auto_fal_priority(self):
        """FAL_KEY should take priority over REPLICATE_API_TOKEN."""
        from samify.backends import resolve
        with patch.dict(os.environ, {"FAL_KEY": "test", "REPLICATE_API_TOKEN": "test"}):
            b = resolve()
            assert b.name == "fal"

    def test_auto_replicate_fallback(self):
        from samify.backends import resolve
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test"}, clear=False):
            env = os.environ.copy()
            env.pop("FAL_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                b = resolve()
                assert b.name == "replicate"


# ---------------------------------------------------------------------------
# Mask filter construction
# ---------------------------------------------------------------------------

class TestBuildMaskFilter:
    @pytest.fixture(autouse=True)
    def _import(self):
        from samify.video_io import _build_mask_filter
        self.build = _build_mask_filter

    def test_defaults(self):
        f = self.build()
        assert f == "format=gray"

    def test_feather_only(self):
        f = self.build(feather=4)
        assert "gblur=sigma=2.000" in f

    def test_erode(self):
        f = self.build(erode=2)
        assert f.count("erosion") == 2

    def test_dilate(self):
        f = self.build(dilate=3)
        assert f.count("dilation") == 3

    def test_invert(self):
        f = self.build(invert=True)
        assert "negate" in f

    def test_threshold(self):
        f = self.build(threshold=128)
        assert "128" in f

    def test_combined(self):
        f = self.build(feather=2, erode=1, dilate=1, invert=True)
        parts = f.split(",")
        # Order should be: format, erosion, dilation, feather, negate
        assert parts[0] == "format=gray"
        assert "erosion" in parts[1]
        assert "dilation" in parts[2]
        assert "gblur" in parts[3]
        assert "negate" in parts[4]


# ---------------------------------------------------------------------------
# SegmentResult
# ---------------------------------------------------------------------------

class TestSegmentResult:
    def test_defaults(self):
        from samify.backends import SegmentResult
        r = SegmentResult(mask_video=Path("mask.mp4"))
        assert r.total_frames == 0
        assert r.detected_frames == 0
        assert r.empty_ranges == []

    def test_with_stats(self):
        from samify.backends import SegmentResult
        r = SegmentResult(
            mask_video=Path("mask.mp4"),
            total_frames=100,
            detected_frames=90,
            empty_ranges=[(5, 10), (50, 54)],
        )
        assert r.total_frames == 100
        assert r.detected_frames == 90
        assert len(r.empty_ranges) == 2


# ---------------------------------------------------------------------------
# Cache key determinism
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_same_inputs_same_key(self, tmp_path):
        from samify.cli import _cache_key
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fake video data")
        k1 = _cache_key(f, "person", "fal")
        k2 = _cache_key(f, "person", "fal")
        assert k1 == k2

    def test_different_prompt_different_key(self, tmp_path):
        from samify.cli import _cache_key
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fake video data")
        k1 = _cache_key(f, "person", "fal")
        k2 = _cache_key(f, "dog", "fal")
        assert k1 != k2

    def test_different_backend_different_key(self, tmp_path):
        from samify.cli import _cache_key
        f = tmp_path / "test.mp4"
        f.write_bytes(b"fake video data")
        k1 = _cache_key(f, "person", "fal")
        k2 = _cache_key(f, "person", "replicate")
        assert k1 != k2


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:
    def test_fal_cost(self):
        from samify.backends.fal_cloud import FalBackend
        est = FalBackend.estimate_cost(300, 30.0)
        assert est is not None
        assert est["cost"] > 0
        assert "$" in est["cost_str"]

    def test_replicate_cost(self):
        from samify.backends.replicate_cloud import ReplicateBackend
        est = ReplicateBackend.estimate_cost(300, 30.0)
        assert est is not None
        assert est["cost"] > 0

    def test_local_free(self):
        from samify.backends.local import LocalBackend
        est = LocalBackend.estimate_cost(300, 30.0)
        assert est is not None
        assert est["cost"] == 0.0
        assert "Free" in est["cost_str"]


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------

class TestArgParsing:
    def _parse(self, argv):
        from samify.cli import _build_parser
        return _build_parser().parse_args(argv)

    def test_basic_parse(self):
        args = self._parse(["video.mp4", "-p", "person"])
        assert args.input == Path("video.mp4")
        assert args.prompt == "person"
        assert args.format == "prores4444"
        assert args.batch is False

    def test_all_mask_opts(self):
        args = self._parse([
            "v.mp4", "-p", "person",
            "--feather", "3",
            "--dilate", "2",
            "--erode", "1",
            "--threshold", "128",
            "--invert",
        ])
        assert args.feather == 3
        assert args.dilate == 2
        assert args.erode == 1
        assert args.threshold == 128
        assert args.invert is True

    def test_webm_format(self):
        args = self._parse(["v.mp4", "-p", "x", "--format", "webm"])
        assert args.format == "webm"

    def test_dry_run(self):
        args = self._parse(["v.mp4", "-p", "x", "--dry-run"])
        assert args.dry_run is True

    def test_preview(self):
        args = self._parse(["v.mp4", "-p", "x", "--preview"])
        assert args.preview is True

    def test_cache_dir(self):
        args = self._parse(["v.mp4", "-p", "x", "--cache-dir", "/tmp/cache"])
        assert args.cache_dir == Path("/tmp/cache")

    def test_chunk_size(self):
        args = self._parse(["v.mp4", "-p", "x", "--chunk-size", "500"])
        assert args.chunk_size == 500

    def test_batch_flag(self):
        args = self._parse(["clips/", "-p", "person", "--batch"])
        assert args.batch is True
        assert args.input == Path("clips/")
