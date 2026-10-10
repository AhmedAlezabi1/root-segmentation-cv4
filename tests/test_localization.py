"""
Unit tests for cv4.localization.

Tests are grouped by function, starting with the pure output functions
and building up to the pathfinding strategies. Inputs are synthetic
numpy arrays so tests stay fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from cv4 import localization

# Fixtures


@pytest.fixture
def small_plant_mask() -> np.ndarray:
    """Tiny plant (~300 px): triggers the small-plant strategy."""
    mask = np.zeros((150, 100), dtype=np.uint8)
    trunk_x = 50
    mask[20:120, trunk_x - 1 : trunk_x + 2] = 1
    return mask


@pytest.fixture
def standard_plant_mask() -> np.ndarray:
    """
    Medium plant (SMALL_PLANT_THRESH < area < HUGE_PLANT_THRESH):
    triggers the standard-plant strategy with thickness-ranked start
    selection and bottleneck check.
    """
    mask = np.zeros((500, 400), dtype=np.uint8)
    trunk_x = 200
    # Thick trunk: 12 px wide, 460 px tall = ~5500 px
    mask[20:480, trunk_x - 6 : trunk_x + 7] = 1
    # One lateral branching off to the right
    mask[200:210, trunk_x : trunk_x + 80] = 1
    return mask


@pytest.fixture
def prob_map_high_confidence() -> np.ndarray:
    """Probability map with a 200x100 high-confidence region in the center."""
    prob_map = np.full((400, 300), 0.05, dtype=np.float32)
    prob_map[100:300, 100:200] = 0.9
    return prob_map


# pixel_to_mm


class TestPixelToMm:
    def test_basic_conversion(self):
        y_mm, x_mm = localization.pixel_to_mm((100, 50), plate_size_px=200)
        assert y_mm == 75.0
        assert x_mm == 37.5

    def test_origin_coordinates(self):
        y_mm, x_mm = localization.pixel_to_mm((0, 0), plate_size_px=200)
        assert y_mm == 0.0
        assert x_mm == 0.0

    def test_custom_plate_size_mm(self):
        """Explicit plate_size_mm overrides the config default."""
        y_mm, x_mm = localization.pixel_to_mm((100, 50), plate_size_px=200, plate_size_mm=100.0)
        assert y_mm == 50.0
        assert x_mm == 25.0

    def test_zero_plate_size_raises(self):
        with pytest.raises(ValueError, match="plate_size_px must be positive"):
            localization.pixel_to_mm((100, 50), plate_size_px=0)

    def test_negative_plate_size_raises(self):
        with pytest.raises(ValueError, match="plate_size_px must be positive"):
            localization.pixel_to_mm((100, 50), plate_size_px=-10)


# get_root_tip


class TestGetRootTip:
    def test_normal_path(self):
        """Tip is the last coordinate in the path."""
        path = [(10, 50), (20, 50), (30, 51), (40, 52)]
        assert localization.get_root_tip(path) == (40, 52)

    def test_single_point_path(self):
        assert localization.get_root_tip([(100, 200)]) == (100, 200)

    def test_none_path_returns_none(self):
        assert localization.get_root_tip(None) is None

    def test_empty_path_returns_none(self):
        assert localization.get_root_tip([]) is None


# get_confidence_score


class TestGetConfidenceScore:
    """
    Tests for get_confidence_score.

    Confidence is averaged over the last N points of the root path
    rather than a window around the tip, so test fixtures are paths
    (lists of (y, x) tuples) along with a matching probability map.
    """

    def _high_prob_map(self, h: int = 100, w: int = 100, value: float = 0.95) -> np.ndarray:
        """Build a uniform high-probability map for happy-path tests."""
        return np.full((h, w), value, dtype=np.float32)

    def _zero_prob_map(self, h: int = 100, w: int = 100) -> np.ndarray:
        return np.zeros((h, w), dtype=np.float32)

    def test_high_probability_path_returns_high_score(self):
        """A path entirely on high-prob pixels should score near 1.0."""
        prob_map = self._high_prob_map(value=0.95)
        path = [(50, x) for x in range(20, 80)]
        score = localization.get_confidence_score(prob_map, path)
        assert score == pytest.approx(0.95, abs=0.01)

    def test_zero_probability_path_returns_zero(self):
        prob_map = self._zero_prob_map()
        path = [(50, x) for x in range(20, 80)]
        score = localization.get_confidence_score(prob_map, path)
        assert score == 0.0

    def test_only_last_n_samples_used(self):
        """
        The function samples the trailing portion of the path. Earlier
        path points should not influence the score.
        """
        # Build a map where the early path is on zeros and the tail is
        # on high probabilities. With n_samples=20 only the tail matters.
        prob_map = np.zeros((100, 100), dtype=np.float32)
        prob_map[50, 60:90] = 0.9

        path = [(50, x) for x in range(0, 90)]  # 90 points total
        score = localization.get_confidence_score(prob_map, path, n_samples=20)
        assert score == pytest.approx(0.9, abs=0.01)

    def test_path_shorter_than_n_samples_is_ok(self):
        """If the path is shorter than n_samples, all of it is used."""
        prob_map = self._high_prob_map(value=0.8)
        path = [(50, x) for x in range(40, 45)]  # 5 points only
        score = localization.get_confidence_score(prob_map, path, n_samples=20)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_empty_path_returns_zero(self):
        prob_map = self._high_prob_map()
        score = localization.get_confidence_score(prob_map, [])
        assert score == 0.0

    def test_all_out_of_bounds_returns_zero(self):
        """If every sampled point is outside the map, the score is 0."""
        prob_map = self._high_prob_map(h=50, w=50)
        path = [(200, 200), (201, 201), (202, 202)]
        score = localization.get_confidence_score(prob_map, path)
        assert score == 0.0

    def test_some_out_of_bounds_skipped(self):
        """Out-of-bounds points are skipped; in-bounds ones still count."""
        prob_map = np.zeros((50, 50), dtype=np.float32)
        prob_map[25, 25] = 0.9
        # First two points out of bounds, third one in bounds.
        path = [(100, 100), (101, 101), (25, 25)]
        score = localization.get_confidence_score(prob_map, path)
        assert score == pytest.approx(0.9, abs=0.01)

    def test_score_is_clipped_to_unit_range(self):
        """A pathological prob map with values above 1 still returns <= 1."""
        prob_map = np.full((50, 50), 1.5, dtype=np.float32)
        path = [(25, 25), (25, 26)]
        score = localization.get_confidence_score(prob_map, path)
        assert score == 1.0

    def test_non_2d_prob_map_raises(self):
        prob_map = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="must be 2D"):
            localization.get_confidence_score(prob_map, [(5, 5)])

    def test_zero_n_samples_raises(self):
        prob_map = self._high_prob_map()
        with pytest.raises(ValueError, match="must be positive"):
            localization.get_confidence_score(prob_map, [(5, 5)], n_samples=0)

    def test_negative_n_samples_raises(self):
        prob_map = self._high_prob_map()
        with pytest.raises(ValueError, match="must be positive"):
            localization.get_confidence_score(prob_map, [(5, 5)], n_samples=-1)


# analyze_components


class TestAnalyzeComponents:
    def test_returns_winner_and_dt_for_valid_mask(self, standard_plant_mask):
        """A valid mask returns a non-None skeleton and distance transform."""
        winner, dt = localization.analyze_components(standard_plant_mask)
        assert winner is not None
        assert dt is not None
        assert winner.shape == standard_plant_mask.shape
        assert dt.shape == standard_plant_mask.shape

    def test_winner_is_binary(self, standard_plant_mask):
        """Winner mask only contains 0s and 1s."""
        winner, _ = localization.analyze_components(standard_plant_mask)
        assert set(np.unique(winner).tolist()).issubset({0, 1})

    def test_dt_is_non_negative(self, standard_plant_mask):
        """Distance transform values are always >= 0."""
        _, dt = localization.analyze_components(standard_plant_mask)
        assert dt.min() >= 0

    def test_empty_mask_returns_none(self):
        empty = np.zeros((100, 100), dtype=np.uint8)
        winner, dt = localization.analyze_components(empty)
        assert winner is None
        assert dt is None

    def test_zero_size_mask_returns_none(self):
        """0-element mask is treated as degenerate input, not an error."""
        zero = np.zeros((0, 0), dtype=np.uint8)
        winner, dt = localization.analyze_components(zero)
        assert winner is None
        assert dt is None

    def test_skeleton_too_small_returns_none(self):
        """A mask with only a few pixels can't produce a viable skeleton."""
        tiny = np.zeros((50, 50), dtype=np.uint8)
        tiny[10, 10] = 1
        tiny[10, 11] = 1
        winner, dt = localization.analyze_components(tiny)
        assert winner is None
        assert dt is None

    def test_picks_taller_of_two_components(self):
        """When given two components, the scoring picks the taller one."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Short fat component on the left
        mask[50:70, 20:30] = 1
        # Tall thin component on the right — should win on height
        mask[20:180, 100:103] = 1

        winner, _ = localization.analyze_components(mask)
        assert winner is not None

        ys, xs = np.where(winner > 0)
        # The winner's x-coordinates should be in the right component's range.
        assert xs.mean() > 50, f"Expected winner to be the right (taller) component, got xs.mean()={xs.mean()}"

    def test_wider_than_tall_component_penalized(self):
        """A horizontal strip loses to a vertical one even with more pixels."""
        mask = np.zeros((200, 300), dtype=np.uint8)
        # Wide horizontal strip on the left — larger area, but wider than tall.
        mask[95:100, 20:180] = 1
        # Narrow vertical strip on the right, non-overlapping in x and y.
        mask[20:180, 250:254] = 1

        winner, _ = localization.analyze_components(mask)
        assert winner is not None

        ys, xs = np.where(winner > 0)
        y_range = ys.max() - ys.min()
        x_range = xs.max() - xs.min()
        assert y_range > x_range, f"Expected vertical strip to win, got y_range={y_range}, x_range={x_range}"


# Fixture for a huge plant


@pytest.fixture
def huge_plant_mask() -> np.ndarray:
    """
    Large plant (>35k px): triggers the huge-plant strategy with central
    corridor filtering and path quality scoring.
    """
    mask = np.zeros((700, 600), dtype=np.uint8)
    trunk_x = 300
    # Thick main trunk, 40 px wide, 680 px tall = ~27200 px
    mask[10:690, trunk_x - 20 : trunk_x + 21] = 1
    # Thick laterals on both sides to bulk up the pixel count
    mask[100:120, trunk_x : trunk_x + 250] = 1  # ~5000 px
    mask[250:270, trunk_x - 250 : trunk_x] = 1  # ~5000 px
    mask[450:470, trunk_x : trunk_x + 270] = 1  # ~5400 px
    mask[580:600, trunk_x - 280 : trunk_x] = 1  # ~5600 px
    return mask


# get_path_from_component


class TestGetPathFromComponent:
    def _run(self, mask):
        """Helper: run the full chain analyze -> path. Returns (path, plant_area)."""
        plant_area = int(np.sum(mask > 0))
        winner, dt = localization.analyze_components(mask)
        assert winner is not None, "fixture didn't produce a viable skeleton"
        path = localization.get_path_from_component(winner, dt, plant_area)
        return path, plant_area

    def test_small_plant_returns_valid_path(self, small_plant_mask):
        """Small-plant strategy returns a non-empty path of (y, x) tuples."""
        path, _ = self._run(small_plant_mask)
        assert path is not None
        assert len(path) > 0
        for point in path:
            assert len(point) == 2
            assert isinstance(point[0], int)
            assert isinstance(point[1], int)

    def test_standard_plant_returns_valid_path(self, standard_plant_mask):
        path, _ = self._run(standard_plant_mask)
        assert path is not None
        assert len(path) > 0

    def test_huge_plant_returns_valid_path(self, huge_plant_mask):
        """Huge-plant strategy (area > 35k) runs without crashing."""
        path, area = self._run(huge_plant_mask)
        assert area > 35000, f"fixture area {area} didn't exceed the huge threshold"
        assert path is not None
        assert len(path) > 0

    def test_path_goes_top_to_bottom(self, standard_plant_mask):
        """First point's y is near the top, last point's y is near the bottom."""
        path, _ = self._run(standard_plant_mask)
        first_y = path[0][0]
        last_y = path[-1][0]
        assert last_y > first_y, f"Path should descend: first y={first_y}, last y={last_y}"

    def test_path_points_inside_mask(self, standard_plant_mask):
        """Every path point sits on a mask pixel."""
        path, _ = self._run(standard_plant_mask)
        for y, x in path:
            assert standard_plant_mask[y, x] > 0, f"Path point ({y}, {x}) is outside the mask"

    def test_tip_stays_on_trunk_not_lateral(self, standard_plant_mask):
        """The root tip should be on the main trunk (x=200), not on a lateral branch."""
        path, _ = self._run(standard_plant_mask)
        tip_y, tip_x = path[-1]
        assert abs(tip_x - 200) <= 6, f"Tip x={tip_x} drifted too far from trunk x=200"


# find_main_root


class TestFindMainRoot:
    def test_small_plant(self, small_plant_mask):
        """End-to-end wrapper works on a small plant."""
        path = localization.find_main_root(small_plant_mask)
        assert path is not None
        assert len(path) > 0

    def test_standard_plant(self, standard_plant_mask):
        path = localization.find_main_root(standard_plant_mask)
        assert path is not None
        assert len(path) > 0

    def test_empty_mask_returns_none(self):
        """Wrapper short-circuits on empty input without calling downstream."""
        empty = np.zeros((100, 100), dtype=np.uint8)
        assert localization.find_main_root(empty) is None

    def test_tiny_skeleton_returns_none(self):
        """Wrapper returns None when analyze_components can't find a skeleton."""
        tiny = np.zeros((50, 50), dtype=np.uint8)
        tiny[10, 10] = 1
        assert localization.find_main_root(tiny) is None


# Acceptance criteria (Task 2.3.5)
# Explicit tests mapping to the task description, even where other tests
# cover the behavior implicitly. Keeps the PR-to-spec mapping obvious.


class TestAcceptanceCriteria:
    def test_root_tip_within_plate_boundaries(self, standard_plant_mask):
        """Root tip coordinates stay within the plate image bounds."""
        path = localization.find_main_root(standard_plant_mask)
        tip = localization.get_root_tip(path)
        assert tip is not None
        tip_y, tip_x = tip
        h, w = standard_plant_mask.shape
        assert 0 <= tip_y < h
        assert 0 <= tip_x < w

    def test_pixel_to_mm_within_plate_range(self):
        """Any in-plate pixel converts to a value in [0, plate_size_mm]."""
        plate_size_px = 2734
        plate_size_mm = 150.0
        # Sample a few positions across the plate.
        for y, x in [(0, 0), (1000, 1000), (2733, 2733), (1367, 2000)]:
            y_mm, x_mm = localization.pixel_to_mm((y, x), plate_size_px=plate_size_px, plate_size_mm=plate_size_mm)
            assert 0 <= y_mm <= plate_size_mm
            assert 0 <= x_mm <= plate_size_mm

    def test_confidence_always_in_valid_range(self, prob_map_high_confidence):
        """Confidence is always a float in [0.0, 1.0] regardless of input."""
        # Path on high-confidence region (centered around y=200, x=150).
        high_conf_path = [(y, 150) for y in range(180, 220)]

        # Path on low-confidence region (centered around y=50, x=50).
        low_conf_path = [(y, 50) for y in range(30, 70)]

        # Path that runs off the map (entirely out of bounds).
        out_of_bounds_path = [(10000, 10000), (10001, 10001)]

        # Empty path edge case.
        empty_path = []

        for path in [high_conf_path, low_conf_path, out_of_bounds_path, empty_path]:
            conf = localization.get_confidence_score(prob_map_high_confidence, path)
            assert isinstance(conf, float), f"confidence is not a float for path of length {len(path)}"
            assert 0.0 <= conf <= 1.0, f"confidence {conf} outside [0,1]"

    def test_skeleton_too_small_returns_none_end_to_end(self):
        """find_main_root handles too-small skeletons gracefully, no crash."""
        tiny = np.zeros((50, 50), dtype=np.uint8)
        tiny[10, 10] = 1
        tiny[10, 11] = 1
        assert localization.find_main_root(tiny) is None
