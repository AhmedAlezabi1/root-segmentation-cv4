"""
Unit tests for cv4.segmentation.

Tests use synthetic numpy masks built in code rather than real plate
images. Lane boundaries are computed from those synthetic masks so
tests don't depend on real lane_edges.npy.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage import measure

from cv4 import segmentation

# Fixtures


def _make_lane_mask(
    height: int = 400,
    width: int = 500,
    trunk_positions: tuple[int, ...] = (50, 150, 250, 350, 450),
    trunk_height_range: tuple[int, int] = (20, 380),
    trunk_thickness: int = 5,
) -> np.ndarray:
    """Build a binary mask with one vertical root per lane."""
    mask = np.zeros((height, width), dtype=np.uint8)
    y_start, y_end = trunk_height_range
    half = trunk_thickness // 2
    for x in trunk_positions:
        mask[y_start:y_end, x - half : x + half + 1] = 1
    return mask


@pytest.fixture
def five_root_mask() -> np.ndarray:
    """Standard 5-root plate: vertical trunks at x = 50, 150, 250, 350, 450."""
    return _make_lane_mask()


@pytest.fixture
def five_root_mask_list(five_root_mask) -> list[np.ndarray]:
    """Three copies of the five-root mask, for compute_lane_edges input."""
    return [five_root_mask.copy() for _ in range(3)]


@pytest.fixture
def lane_edges(five_root_mask_list) -> np.ndarray:
    """Lane edges computed from the synthetic masks."""
    return segmentation.compute_lane_edges(five_root_mask_list, expected_roots=5)


@pytest.fixture
def three_root_mask() -> np.ndarray:
    """Plate with only 3 roots: trunks at x = 50, 250, 450 (lanes 1 and 3 empty)."""
    return _make_lane_mask(trunk_positions=(50, 250, 450))


@pytest.fixture
def empty_mask() -> np.ndarray:
    """All-zero mask: no roots at all."""
    return np.zeros((400, 500), dtype=np.uint8)


# x_crown_from_region


class TestXCrownFromRegion:
    def test_returns_float(self):
        """Output is a float, not a numpy scalar."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 20:25] = 1
        region = measure.regionprops(measure.label(mask))[0]
        result = segmentation.x_crown_from_region(region)
        assert isinstance(result, float)

    def test_simple_vertical_column(self):
        """For a vertical column, crown x equals column center."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 20:25] = 1
        region = measure.regionprops(measure.label(mask))[0]
        result = segmentation.x_crown_from_region(region)
        assert result == pytest.approx(22.0, abs=0.5)


# lane_from_x


class TestLaneFromX:
    def test_basic_mapping(self):
        """Given uniform edges, x values map to the correct lane index."""
        edges = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
        assert segmentation.lane_from_x(50, edges) == 0
        assert segmentation.lane_from_x(150, edges) == 1
        assert segmentation.lane_from_x(250, edges) == 2
        assert segmentation.lane_from_x(350, edges) == 3
        assert segmentation.lane_from_x(450, edges) == 4

    def test_clips_below_zero(self):
        """Negative x clips to lane 0."""
        edges = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
        assert segmentation.lane_from_x(-10, edges) == 0

    def test_clips_above_width(self):
        """x past the last edge clips to the final lane."""
        edges = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
        assert segmentation.lane_from_x(10000, edges) == 4


# crop_to_root


class TestCropToRoot:
    def test_returns_none_for_empty_mask(self):
        empty = np.zeros((100, 100), dtype=np.uint8)
        assert segmentation.crop_to_root(empty) is None

    def test_crops_tight_around_content(self):
        """Output excludes zero-padding beyond the margin."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[80:120, 80:120] = 1
        cropped = segmentation.crop_to_root(mask, margin=10)
        assert cropped is not None
        assert cropped.shape[0] < mask.shape[0]
        assert cropped.shape[1] < mask.shape[1]

    def test_respects_custom_margin(self):
        """Larger margin produces a larger crop."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[80:120, 80:120] = 1
        small = segmentation.crop_to_root(mask, margin=5)
        big = segmentation.crop_to_root(mask, margin=30)
        assert big.shape[0] > small.shape[0]
        assert big.shape[1] > small.shape[1]

    def test_margin_clips_to_image_bounds(self):
        """Margin doesn't extend beyond the original mask edges."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1  # top-left corner
        cropped = segmentation.crop_to_root(mask, margin=50)
        assert cropped.shape[0] <= 100
        assert cropped.shape[1] <= 100


# compute_lane_edges


class TestComputeLaneEdges:
    def test_returns_six_edges_for_five_lanes(self, five_root_mask_list):
        """5 expected roots produces 6 edges: [0, e1, e2, e3, e4, W]."""
        edges = segmentation.compute_lane_edges(five_root_mask_list, expected_roots=5)
        assert len(edges) == 6

    def test_first_edge_is_zero(self, five_root_mask_list):
        edges = segmentation.compute_lane_edges(five_root_mask_list, expected_roots=5)
        assert edges[0] == 0.0

    def test_last_edge_is_image_width(self, five_root_mask_list):
        """Final edge equals the mask width."""
        expected_width = five_root_mask_list[0].shape[1]
        edges = segmentation.compute_lane_edges(five_root_mask_list, expected_roots=5)
        assert edges[-1] == expected_width

    def test_edges_are_monotonically_increasing(self, five_root_mask_list):
        """Lane boundaries must be sorted left to right."""
        edges = segmentation.compute_lane_edges(five_root_mask_list, expected_roots=5)
        for i in range(len(edges) - 1):
            assert edges[i] < edges[i + 1], f"Edge {i} = {edges[i]} >= edge {i + 1} = {edges[i + 1]}"

    def test_edges_land_between_trunks(self, five_root_mask_list):
        """Inner edges sit between adjacent trunk positions (50, 150, 250, 350, 450)."""
        edges = segmentation.compute_lane_edges(five_root_mask_list, expected_roots=5)
        trunk_midpoints = [100.0, 200.0, 300.0, 400.0]
        for midpoint, actual_edge in zip(trunk_midpoints, edges[1:5]):
            assert abs(actual_edge - midpoint) <= 5.0, f"Edge {actual_edge} too far from expected midpoint {midpoint}"

    def test_empty_list_raises(self):
        """Empty input is a caller bug, not a data issue."""
        with pytest.raises(ValueError, match="No masks provided"):
            segmentation.compute_lane_edges([], expected_roots=5)

    def test_all_none_masks_raises(self):
        """All-None masks can't produce edges; should raise."""
        with pytest.raises(ValueError, match="No usable masks"):
            segmentation.compute_lane_edges([None, None, None], expected_roots=5)

    def test_mask_with_too_few_components_raises(self, empty_mask):
        """A mask with no viable components raises."""
        with pytest.raises(ValueError, match="No usable masks"):
            segmentation.compute_lane_edges([empty_mask, empty_mask], expected_roots=5)

    def test_skips_masks_with_wrong_width(self, five_root_mask, five_root_mask_list):
        """Masks of differing widths are skipped, not used."""
        wrong_width_mask = np.zeros((400, 700), dtype=np.uint8)
        wrong_width_mask[20:380, 100:105] = 1
        masks = five_root_mask_list + [wrong_width_mask]
        # Still works because the correct-width masks are enough.
        edges = segmentation.compute_lane_edges(masks, expected_roots=5)
        assert len(edges) == 6


# lane_clean_and_instances (core instance-segmentation function)


class TestLaneCleanAndInstances:
    def test_returns_three_elements(self, five_root_mask, lane_edges):
        """Function returns (clean_union, lane_masks, debug_info)."""
        result = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        assert len(result) == 3

    def test_returns_exactly_five_lane_masks(self, five_root_mask, lane_edges):
        """Acceptance criterion: always returns expected number of plant instances."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        assert len(lane_masks) == 5

    def test_full_plate_populates_all_lanes(self, five_root_mask, lane_edges):
        """A 5-root mask should produce 5 non-empty lane masks."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        for lane_idx, mask in enumerate(lane_masks):
            assert mask.sum() > 0, f"Lane {lane_idx} unexpectedly empty"

    def test_empty_lanes_return_empty_masks(self, three_root_mask, lane_edges):
        """Acceptance criterion: lanes with no root get empty masks, no crash."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(three_root_mask, lane_edges)
        assert len(lane_masks) == 5
        # Trunks at x=50, 250, 450 -> lanes 0, 2, 4 populated, 1 and 3 empty.
        assert lane_masks[0].sum() > 0
        assert lane_masks[1].sum() == 0
        assert lane_masks[2].sum() > 0
        assert lane_masks[3].sum() == 0
        assert lane_masks[4].sum() > 0

    def test_all_empty_mask_does_not_crash(self, empty_mask, lane_edges):
        """Acceptance criterion: completely empty input returns 5 empty masks."""
        clean_union, lane_masks, _ = segmentation.lane_clean_and_instances(empty_mask, lane_edges)
        assert len(lane_masks) == 5
        for mask in lane_masks:
            assert mask.sum() == 0
        assert clean_union.sum() == 0

    def test_lane_masks_are_binary(self, five_root_mask, lane_edges):
        """Output masks contain only 0s and 1s."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        for mask in lane_masks:
            assert set(np.unique(mask).tolist()).issubset({0, 1})

    def test_lane_masks_match_input_shape(self, five_root_mask, lane_edges):
        """All lane masks have the same shape as the input mask."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        for mask in lane_masks:
            assert mask.shape == five_root_mask.shape

    def test_clean_union_is_union_of_lanes(self, five_root_mask, lane_edges):
        """The returned union mask is the OR of all lane masks."""
        clean_union, lane_masks, _ = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        expected_union = np.zeros_like(clean_union)
        for mask in lane_masks:
            expected_union |= mask
        assert np.array_equal(clean_union, expected_union)

    def test_debug_info_has_expected_keys(self, five_root_mask, lane_edges):
        """debug_info dict contains diagnostic fields used by pipeline logging."""
        _, _, debug_info = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        expected_keys = {"top_band_px", "y_cut_px", "chosen_labels", "chosen_scores"}
        assert expected_keys.issubset(set(debug_info.keys()))

    def test_debug_info_chosen_labels_length(self, five_root_mask, lane_edges):
        """chosen_labels has exactly K entries (one per lane)."""
        _, _, debug_info = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        assert len(debug_info["chosen_labels"]) == 5

    def test_speck_removal(self, lane_edges):
        """Tiny isolated specks (< speck_area) are filtered out."""
        mask = np.zeros((400, 500), dtype=np.uint8)
        # Add one real root
        mask[20:380, 150:155] = 1
        # Sprinkle single-pixel specks around
        mask[100, 30] = 1
        mask[200, 480] = 1
        mask[300, 10] = 1

        _, lane_masks, _ = segmentation.lane_clean_and_instances(mask, lane_edges)
        # The speck pixels shouldn't appear in any lane mask.
        for mask_out in lane_masks:
            assert mask_out[100, 30] == 0
            assert mask_out[200, 480] == 0
            assert mask_out[300, 10] == 0

    def test_masks_are_disjoint(self, five_root_mask, lane_edges):
        """No pixel appears in more than one lane mask."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        total = sum(mask.sum() for mask in lane_masks)
        union_sum = int(np.logical_or.reduce(lane_masks).sum())
        assert total == union_sum, "Lane masks overlap — a pixel appears in multiple lanes"


# Acceptance criteria (Task 2.3.5)
# Explicit tests mapping to the task description so the PR-to-spec
# mapping is obvious to reviewers.


class TestAcceptanceCriteria:
    def test_returns_five_plant_instances_for_standard_plate(self, five_root_mask, lane_edges):
        """Criterion: returns the expected number of plant instances (5 for standard plate)."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(five_root_mask, lane_edges)
        assert len(lane_masks) == 5

    def test_empty_lane_returns_empty_mask_without_crashing(self, three_root_mask, lane_edges):
        """Criterion: lanes with no root return empty masks, no crash."""
        _, lane_masks, _ = segmentation.lane_clean_and_instances(three_root_mask, lane_edges)
        assert len(lane_masks) == 5
        empty_lanes = [i for i, m in enumerate(lane_masks) if m.sum() == 0]
        assert len(empty_lanes) == 2, f"Expected 2 empty lanes (indices 1 and 3), got {len(empty_lanes)}: {empty_lanes}"

    def test_all_empty_input_returns_five_empty_masks_without_crashing(self, empty_mask, lane_edges, caplog):
        """Criterion edge case: all-empty input still returns 5 masks, doesn't raise."""
        import logging

        with caplog.at_level(logging.WARNING, logger="cv4.segmentation"):
            clean_union, lane_masks, _ = segmentation.lane_clean_and_instances(empty_mask, lane_edges)

        assert len(lane_masks) == 5
        assert all(m.sum() == 0 for m in lane_masks)
        assert clean_union.sum() == 0

        # Verify the warning was actually logged (regression guard).
        assert any("No plants detected" in record.message for record in caplog.records), "Expected a warning about empty plates, none was logged"
