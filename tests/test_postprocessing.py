"""
Unit tests for postprocessing.py
======================================

Run with coverage from the repo root:

    pytest tests/test_postprocessing.py -v --cov=cv4.postprocessing --cov-report=term-missing

Module structure under test
----------------------------
In-memory primitives (what pipeline.py calls):
    apply_double_threshold  – probability map → binary mask
    unpad                   – strip inference padding
    micro_close             – morphological closing
    skeleton_endpoints      – find degree-1 skeleton pixels  (public)
    bridge_mask_gaps        – reconnect split root segments  (public, kwargs)
    postprocess_mask        – full chain: threshold → unpad → close → bridge

Batch file wrapper (offline / notebook use):
    apply_micro_close_and_bridge – reads from disk, calls in-memory
                                   primitives in a loop, writes back to disk

Design notes
------------
- No real images or model weights needed.  Every test builds its own
  minimal NumPy arrays.
- File I/O tests use pytest's tmp_path fixture — nothing written to the
  source tree and cleanup is automatic.
- All tests are deterministic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv4.postprocessing import (
    # Config dataclasses
    BridgeConfig,
    MicroCloseConfig,
    # Internal helper tested directly
    _lane_bucket,
    # In-memory primitives
    apply_double_threshold,
    # Batch wrapper
    apply_micro_close_and_bridge,
    bridge_mask_gaps,
    micro_close,
    postprocess_mask,
    skeleton_endpoints,
    unpad,
)

# =============================================================================
# Shared helpers
# =============================================================================


def _vertical_stripe(H: int = 100, W: int = 50, col: int = 25, thickness: int = 3) -> np.ndarray:
    """Binary mask with a single vertical stripe — a minimal synthetic root."""
    m = np.zeros((H, W), dtype=np.uint8)
    m[:, col - thickness // 2 : col + thickness // 2 + 1] = 1
    return m


def _five_stripes(H: int = 200, W: int = 500) -> np.ndarray:
    """Binary mask with 5 evenly-spaced vertical stripes (one per root lane)."""
    m = np.zeros((H, W), dtype=np.uint8)
    for c in [50, 150, 250, 350, 450]:
        m[:, c - 2 : c + 3] = 1
    return m


def _write_png(path: Path, arr: np.ndarray) -> None:
    """Write a NumPy array to disk as a PNG file."""
    cv2.imwrite(str(path), arr)


# =============================================================================
# 1. apply_double_threshold
# =============================================================================


class TestApplyDoubleThreshold:
    """Tests for apply_double_threshold — hysteresis thresholding on a prob map."""

    def test_high_confidence_pixels_kept(self):
        """Pixels above the high threshold must always be in the output."""
        prob = np.zeros((10, 10), dtype=np.float32)
        prob[5, 5] = 0.9  # well above any sensible high threshold
        result = apply_double_threshold(prob, low=0.05, high=0.25)
        assert result[5, 5] == 1

    def test_low_confidence_isolated_pixels_removed(self):
        """A pixel below the high threshold with no high-confidence neighbour is dropped."""
        prob = np.zeros((10, 10), dtype=np.float32)
        prob[5, 5] = 0.10  # between low and high but isolated
        result = apply_double_threshold(prob, low=0.05, high=0.25)
        assert result[5, 5] == 0

    def test_connected_weak_pixel_kept(self):
        """A weak pixel connected to a strong pixel must be retained."""
        prob = np.zeros((10, 10), dtype=np.float32)
        prob[5, 5] = 0.9  # strong anchor
        prob[5, 6] = 0.10  # weak but adjacent
        result = apply_double_threshold(prob, low=0.05, high=0.25)
        assert result[5, 6] == 1

    def test_output_dtype_is_uint8(self):
        """Output must be uint8 with values 0 or 1."""
        prob = np.random.rand(20, 20).astype(np.float32)
        result = apply_double_threshold(prob, low=0.05, high=0.25)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1})

    def test_all_zeros_prob_map_gives_all_zeros(self):
        """A completely blank probability map should produce a blank mask."""
        prob = np.zeros((10, 10), dtype=np.float32)
        assert apply_double_threshold(prob, low=0.05, high=0.25).sum() == 0

    def test_raises_for_non_2d_input(self):
        """A 3-D input should raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            apply_double_threshold(np.zeros((10, 10, 3), dtype=np.float32))

    def test_raises_when_low_greater_than_high(self):
        """low > high is a configuration error and must raise ValueError."""
        with pytest.raises(ValueError, match="low threshold"):
            apply_double_threshold(np.zeros((10, 10), dtype=np.float32), low=0.8, high=0.2)


# =============================================================================
# 2. unpad
# =============================================================================


class TestUnpad:
    """Tests for unpad — strips inference padding from a mask or prob map."""

    def test_symmetric_removal(self):
        """Removing equal padding from all sides should halve the padded area."""
        arr = np.ones((12, 12), dtype=np.uint8)
        assert unpad(arr, (2, 2), (3, 3)).shape == (8, 6)

    def test_asymmetric_removal(self):
        """Different top/bottom and left/right values should each be applied correctly."""
        arr = np.arange(100).reshape(10, 10).astype(np.uint8)
        result = unpad(arr, (1, 3), (2, 0))
        assert result.shape == (6, 8)
        assert result[0, 0] == arr[1, 2]

    def test_zero_padding_returns_original(self):
        """Zero padding on all sides should return the array unchanged."""
        arr = np.eye(5, dtype=np.uint8)
        np.testing.assert_array_equal(unpad(arr, (0, 0), (0, 0)), arr)

    def test_content_preserved_after_unpad(self):
        """Values inside the padded region should survive the strip."""
        inner = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        padded = cv2.copyMakeBorder(inner, 2, 2, 3, 3, cv2.BORDER_CONSTANT, value=0)
        np.testing.assert_array_equal(unpad(padded, (2, 2), (3, 3)), inner)

    def test_raises_when_padding_exceeds_dimensions(self):
        """Padding larger than the array dimensions must raise ValueError."""
        arr = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="too large"):
            unpad(arr, (3, 3), (0, 0))  # 6 rows removed from a 5-row array


# =============================================================================
# 3. micro_close
# =============================================================================


class TestMicroClose:
    """Tests for micro_close — morphological closing on a binary mask."""

    def test_fills_small_gap_in_vertical_stripe(self):
        """A one-pixel vertical gap inside a stripe should be closed."""
        mask = _vertical_stripe(H=20, W=20, col=10, thickness=2)
        mask[10, 9:12] = 0  # punch a hole
        result = micro_close(mask, kernel_size=4, iterations=1)
        assert result[10, 10] == 1

    def test_output_dtype_is_uint8(self):
        """Output must always be uint8."""
        assert micro_close(_vertical_stripe()).dtype == np.uint8

    def test_output_values_are_binary(self):
        """Output must contain only 0 and 1."""
        result = micro_close(_vertical_stripe())
        assert set(np.unique(result)).issubset({0, 1})

    def test_accepts_255_scaled_mask(self):
        """Input values of 0/255 (common PNG format) must be handled correctly."""
        mask = (_vertical_stripe() * 255).astype(np.uint8)
        result = micro_close(mask)
        assert result.dtype == np.uint8

    def test_raises_for_invalid_kernel_size(self):
        """kernel_size < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="kernel_size"):
            micro_close(_vertical_stripe(), kernel_size=0)

    def test_raises_for_invalid_iterations(self):
        """iterations < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="iterations"):
            micro_close(_vertical_stripe(), iterations=0)


# =============================================================================
# 4. skeleton_endpoints
# =============================================================================


class TestSkeletonEndpoints:
    """Tests for skeleton_endpoints — finds degree-1 pixels in a skeleton."""

    def test_straight_line_has_two_endpoints(self):
        """A 10-pixel vertical line has exactly one endpoint at each end."""
        skel = np.zeros((20, 20), dtype=np.uint8)
        skel[5:15, 10] = 1
        assert skeleton_endpoints(skel).shape[0] == 2

    def test_empty_skeleton_returns_empty(self):
        """An all-zero image has no pixels so no endpoints can exist."""
        assert skeleton_endpoints(np.zeros((10, 10), dtype=np.uint8)).shape == (0, 2)

    def test_result_is_two_column_array(self):
        """Returned array must have shape (N, 2) — columns are y and x."""
        skel = np.zeros((20, 20), dtype=np.uint8)
        skel[5:15, 10] = 1
        eps = skeleton_endpoints(skel)
        assert eps.ndim == 2 and eps.shape[1] == 2

    def test_single_pixel_has_no_endpoints(self):
        """A lone pixel has 0 neighbours so it is not degree-1."""
        skel = np.zeros((10, 10), dtype=np.uint8)
        skel[5, 5] = 1
        assert skeleton_endpoints(skel).shape[0] == 0


# =============================================================================
# 5. bridge_mask_gaps
# =============================================================================


class TestBridgeMaskGaps:
    """Tests for bridge_mask_gaps — connects split root segments via endpoints."""

    def _two_stripes(self, gap: int = 5) -> np.ndarray:
        """Two vertical stripes separated by a vertical gap."""
        m = np.zeros((60, 60), dtype=np.uint8)
        m[5:30, 25:28] = 1  # top stripe
        m[30 + gap : 55, 25:28] = 1  # bottom stripe
        return m

    def test_bridgeable_gap_adds_pixels(self):
        """Two stripes with a small gap should be connected."""
        m = self._two_stripes(gap=5)
        result = bridge_mask_gaps(m, max_dist=30, max_dx=4, min_dy=3, thickness=1, enforce_same_lane=False)
        assert result.sum() > m.sum()

    def test_gap_too_large_not_bridged(self):
        """Gap larger than max_dist must not be bridged."""
        m = self._two_stripes(gap=40)
        result = bridge_mask_gaps(m, max_dist=10, max_dx=4, min_dy=3, thickness=1, enforce_same_lane=False)
        assert result.sum() == m.sum()

    def test_single_component_pixel_count_nondecreasing(self):
        """Bridging an already-connected mask should not remove pixels."""
        m = _vertical_stripe(H=50, W=50)
        assert bridge_mask_gaps(m, max_dist=200, max_dx=5, min_dy=2, thickness=1).sum() >= m.sum()

    def test_output_dtype_is_uint8(self):
        """Output must always be uint8."""
        assert bridge_mask_gaps(_vertical_stripe()).dtype == np.uint8

    def test_fewer_than_two_endpoints_returns_unchanged(self):
        """A single isolated pixel has no endpoints — mask returned as-is."""
        m = np.zeros((20, 20), dtype=np.uint8)
        m[10, 10] = 1
        np.testing.assert_array_equal(bridge_mask_gaps(m), m)

    def test_enforce_same_lane_prevents_cross_lane_bridge(self):
        """Endpoints in different lane buckets must not be bridged."""
        m = np.zeros((60, 100), dtype=np.uint8)
        m[5:30, 10:13] = 1  # left lane
        m[35:55, 80:83] = 1  # right lane — different bucket
        result = bridge_mask_gaps(
            m,
            max_dist=200,
            max_dx=200,
            min_dy=3,
            thickness=1,
            enforce_same_lane=True,
            expected_roots=2,
        )
        assert result.sum() == m.sum()


# =============================================================================
# 6. postprocess_mask
# =============================================================================


class TestPostprocessMask:
    """Tests for postprocess_mask — the full in-memory chain pipeline.py calls."""

    def _prob_map_with_root(self, H: int = 50, W: int = 50) -> np.ndarray:
        """Probability map with a strong vertical root stripe down the centre."""
        prob = np.zeros((H, W), dtype=np.float32)
        prob[:, W // 2 - 2 : W // 2 + 3] = 0.9
        return prob

    def test_returns_binary_uint8(self):
        """Output must be a uint8 mask with only values 0 and 1."""
        prob = self._prob_map_with_root()
        result = postprocess_mask(prob, pad_h=(2, 2), pad_w=(2, 2))
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1})

    def test_output_shape_matches_unpadded_size(self):
        """Shape after postprocessing must equal prob_map minus the padding."""
        prob = self._prob_map_with_root(H=60, W=60)
        result = postprocess_mask(prob, pad_h=(5, 5), pad_w=(5, 5))
        assert result.shape == (50, 50)

    def test_strong_root_pixels_survive_pipeline(self):
        """High-confidence root pixels should still be present in the output."""
        prob = self._prob_map_with_root(H=50, W=50)
        result = postprocess_mask(prob, pad_h=(2, 2), pad_w=(2, 2))
        assert result.sum() > 0

    def test_blank_prob_map_gives_blank_output(self):
        """A probability map of all zeros should produce an all-zero mask."""
        prob = np.zeros((50, 50), dtype=np.float32)
        result = postprocess_mask(prob, pad_h=(2, 2), pad_w=(2, 2))
        assert result.sum() == 0

    def test_propagates_value_error_from_unpad(self):
        """Invalid padding that exceeds array size must propagate as ValueError."""
        prob = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            postprocess_mask(prob, pad_h=(6, 6), pad_w=(0, 0))

    def test_logs_info_at_start(self, caplog):
        """An INFO message should be emitted at the start of postprocess_mask."""
        prob = self._prob_map_with_root()
        with caplog.at_level(logging.INFO, logger="cv4.postprocessing"):
            postprocess_mask(prob, pad_h=(2, 2), pad_w=(2, 2))
        assert any("postprocess_mask" in r.message for r in caplog.records)


# =============================================================================
# 7. _lane_bucket
# =============================================================================


class TestLaneBucket:
    """Tests for _lane_bucket — maps an x-coordinate to a lane index."""

    def test_leftmost_pixel_is_lane_0(self):
        """x=0 should always map to the first lane."""
        assert _lane_bucket(0, 500, 5) == 0

    def test_rightmost_pixel_is_last_lane(self):
        """x just below the image width should map to the last lane."""
        assert _lane_bucket(499, 500, 5) == 4

    def test_midpoint_lands_in_correct_lane(self):
        """With 5 equal lanes across 500px (100px each), x=150 falls in lane 1."""
        assert _lane_bucket(150, 500, 5) == 1

    def test_clamped_below_zero(self):
        """Negative x values must be clamped to lane 0, not raise an error."""
        assert _lane_bucket(-99, 500, 5) == 0

    def test_clamped_above_max(self):
        """x values beyond the image width must be clamped to the last lane."""
        assert _lane_bucket(9999, 500, 5) == 4


# =============================================================================
# 8. apply_micro_close_and_bridge  (batch disk I/O wrapper)
# =============================================================================


class TestApplyMicroCloseAndBridge:
    """Tests for apply_micro_close_and_bridge — the batch file-based wrapper."""

    def test_produces_one_output_per_input(self, tmp_path):
        """Each input mask should produce exactly one output file."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_png(in_dir / "img_prediction.png", (_vertical_stripe() * 255).astype(np.uint8))
        apply_micro_close_and_bridge(str(in_dir), str(tmp_path / "out"))
        assert len(list((tmp_path / "out").glob("*.png"))) == 1

    def test_output_name_contains_microclose_bridge(self, tmp_path):
        """Output filename must include the pipeline stage suffix."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_png(in_dir / "test_image_01_prediction.png", np.zeros((50, 50), dtype=np.uint8))
        apply_micro_close_and_bridge(str(in_dir), str(tmp_path / "out"))
        names = [p.name for p in (tmp_path / "out").glob("*.png")]
        assert any("microclose_bridge" in n for n in names)

    def test_stem_carries_through_correctly(self, tmp_path):
        """Output should be {original_stem}_prediction_microclose_bridge.png."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_png(in_dir / "test_image_03_prediction.png", np.zeros((50, 50), dtype=np.uint8))
        apply_micro_close_and_bridge(str(in_dir), str(tmp_path / "out"))
        assert (tmp_path / "out" / "test_image_03_prediction_microclose_bridge.png").exists()

    def test_multiple_inputs_all_processed(self, tmp_path):
        """All masks in the input directory should be processed."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for i in range(3):
            _write_png(in_dir / f"img{i:02d}_prediction.png", (_vertical_stripe() * 255).astype(np.uint8))
        apply_micro_close_and_bridge(str(in_dir), str(tmp_path / "out"))
        assert len(list((tmp_path / "out").glob("*.png"))) == 3

    def test_output_is_binary_0_or_255(self, tmp_path):
        """Saved PNGs must contain only pixel values 0 and 255."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_png(in_dir / "img_prediction.png", (_vertical_stripe() * 255).astype(np.uint8))
        apply_micro_close_and_bridge(str(in_dir), str(tmp_path / "out"))
        out_img = cv2.imread(str(next((tmp_path / "out").glob("*.png"))), cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(out_img)).issubset({0, 255})

    def test_returns_out_dir_path(self, tmp_path):
        """Return value should be the output directory path for call chaining."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_png(in_dir / "img_prediction.png", np.zeros((50, 50), dtype=np.uint8))
        out_dir = str(tmp_path / "out")
        assert apply_micro_close_and_bridge(str(in_dir), out_dir) == out_dir

    def test_creates_out_dir_if_absent(self, tmp_path):
        """The output directory should be created automatically if it does not exist."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_png(in_dir / "img_prediction.png", np.zeros((50, 50), dtype=np.uint8))
        out_dir = tmp_path / "does_not_exist_yet"
        apply_micro_close_and_bridge(str(in_dir), str(out_dir))
        assert out_dir.exists()

    def test_raises_if_no_prediction_masks_found(self, tmp_path):
        """A FileNotFoundError should be raised when in_dir has no *_prediction.png files."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No \\*_prediction.png"):
            apply_micro_close_and_bridge(str(in_dir), str(tmp_path / "out"))

    def test_skips_unreadable_mask_and_logs_warning(self, tmp_path, caplog):
        """An unreadable file should be skipped with a warning, not crash the run."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "bad_prediction.png").write_bytes(b"not an image")
        _write_png(in_dir / "good_prediction.png", (_vertical_stripe() * 255).astype(np.uint8))
        with caplog.at_level(logging.WARNING, logger="cv4.postprocessing"):
            apply_micro_close_and_bridge(str(in_dir), str(tmp_path / "out"))
        assert any("skipping" in r.message.lower() for r in caplog.records)


# =============================================================================
# 9. Dataclass defaults mirror config
# =============================================================================


class TestDataclassDefaults:
    """Verify that config dataclass defaults match the values in cv4/config.py."""

    def test_micro_close_defaults(self):
        """MicroCloseConfig defaults must reflect the config.py constants."""
        c = MicroCloseConfig()
        assert c.kernel_size == 4
        assert c.iterations == 1

    def test_bridge_defaults(self):
        """BridgeConfig defaults must reflect the config.py constants."""
        c = BridgeConfig()
        assert c.max_dist == 60
        assert c.max_dx == 4
        assert c.min_dy == 8
        assert c.thickness == 3
        assert c.enforce_same_lane is True
        assert c.expected_roots == 5

    def test_overriding_one_field_does_not_affect_others(self):
        """Changing one field at construction time must leave all others at their defaults."""
        c = BridgeConfig(max_dist=999)
        assert c.max_dist == 999
        assert c.max_dx == 4  # unchanged
