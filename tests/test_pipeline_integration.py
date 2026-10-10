"""
Integration tests for cv4.pipeline.

Unlike the per-module unit tests (which use synthetic numpy arrays),
these tests run the full pipeline on real petri dish images stored in
tests/data/. They're slow because they load the real model and run
inference, so they're separated from the fast unit-test suite.

Run only this file:
    uv run pytest tests/test_pipeline_integration.py -v

Skip these in fast iterations:
    uv run pytest -v --ignore=tests/test_pipeline_integration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cv4 import pipeline

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "data"
_VALID_IMAGE = _FIXTURE_DIR / "valid_dish.png"
_CORRUPT_IMAGE = _FIXTURE_DIR / "corrupted.png"


# Skip the whole file if the model weights aren't on disk. Lets the
# rest of the test suite run on machines that haven't downloaded them.
pytestmark = pytest.mark.skipif(
    not (_REPO_ROOT / "weights" / "unet_256px.h5").exists(),
    reason="Model weights not available",
)


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the pipeline once for the whole test module.

    Module-scoped because each pipeline run takes ~30-60s. All tests
    that inspect the result share this single execution.
    """
    return pipeline.run_pipeline(str(_VALID_IMAGE))


# Output structure


class TestOutputStructure:
    def test_returns_dict(self, pipeline_result):
        assert isinstance(pipeline_result, dict)

    def test_has_all_required_top_level_keys(self, pipeline_result):
        expected = {
            "image_path",
            "binary_mask",
            "plant_masks",
            "plants",
            "processing_time_seconds",
            "plate_bbox",
        }
        assert expected.issubset(pipeline_result.keys())

    def test_image_path_echoed(self, pipeline_result):
        assert pipeline_result["image_path"] == str(_VALID_IMAGE)

    def test_returns_five_plants(self, pipeline_result):
        """Acceptance criterion: pipeline always returns 5 plant entries."""
        assert len(pipeline_result["plants"]) == 5
        assert len(pipeline_result["plant_masks"]) == 5

    def test_plate_bbox_is_three_tuple(self, pipeline_result):
        bbox = pipeline_result["plate_bbox"]
        assert len(bbox) == 3
        x, y, side = bbox
        assert side > 0

    def test_processing_time_is_positive(self, pipeline_result):
        assert pipeline_result["processing_time_seconds"] > 0


# Mask outputs


class TestMaskOutputs:
    def test_binary_mask_is_2d_array(self, pipeline_result):
        mask = pipeline_result["binary_mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2

    def test_binary_mask_is_actually_binary(self, pipeline_result):
        mask = pipeline_result["binary_mask"]
        unique_values = set(np.unique(mask).tolist())
        assert unique_values.issubset({0, 1})

    def test_plant_masks_share_binary_shape(self, pipeline_result):
        binary = pipeline_result["binary_mask"]
        for plant_mask in pipeline_result["plant_masks"]:
            assert plant_mask.shape == binary.shape

    def test_plant_masks_are_disjoint(self, pipeline_result):
        """No pixel may appear in more than one plant mask."""
        masks = pipeline_result["plant_masks"]
        total = sum(int(m.sum()) for m in masks)
        union = int(np.logical_or.reduce(masks).sum())
        assert total == union, "Plant masks overlap"


# Per-plant entries


class TestPerPlantEntries:
    def test_each_plant_has_required_keys(self, pipeline_result):
        required = {"lane", "has_root", "main_root_path"}
        for plant in pipeline_result["plants"]:
            assert required.issubset(plant.keys())

    def test_lane_indices_are_zero_to_four(self, pipeline_result):
        lanes = [p["lane"] for p in pipeline_result["plants"]]
        assert lanes == [0, 1, 2, 3, 4]

    def test_plants_with_root_have_valid_coordinates(self, pipeline_result):
        """Detected roots produce coordinates within the unpadded mask bounds."""
        h, w = pipeline_result["binary_mask"].shape
        for plant in pipeline_result["plants"]:
            if plant["has_root"]:
                tip_px = plant["root_tip_px"]
                assert 0 <= tip_px["x"] < w
                assert 0 <= tip_px["y"] < h

    def test_plants_with_root_have_mm_coordinates_in_plate_range(self, pipeline_result):
        """mm coordinates fall within the physical plate (150mm)."""
        for plant in pipeline_result["plants"]:
            if plant["has_root"]:
                tip_mm = plant["root_tip_mm"]
                assert 0.0 <= tip_mm["x"] <= 150.0
                assert 0.0 <= tip_mm["y"] <= 150.0

    def test_plants_with_root_have_confidence_in_unit_range(self, pipeline_result):
        for plant in pipeline_result["plants"]:
            if plant["has_root"]:
                conf = plant["confidence"]
                assert isinstance(conf, float)
                assert 0.0 <= conf <= 1.0

    def test_empty_lanes_have_none_coordinates(self, pipeline_result):
        """Acceptance criterion: empty lanes return None, not (0, 0)."""
        for plant in pipeline_result["plants"]:
            if not plant["has_root"]:
                assert plant["root_tip_px"] is None
                assert plant["root_tip_mm"] is None
                assert plant["confidence"] is None

    def test_at_least_one_root_detected(self, pipeline_result):
        """The fixture image was chosen to have visible roots."""
        with_root = sum(1 for p in pipeline_result["plants"] if p["has_root"])
        assert with_root >= 1, "No roots detected on the fixture, the fixture may be unsuitable or the pipeline regressed"


# Error handling


class TestErrorHandling:
    def test_missing_image_raises(self):
        """Acceptance criterion: bad inputs raise clean errors."""
        with pytest.raises(FileNotFoundError):
            pipeline.run_pipeline(str(_FIXTURE_DIR / "nonexistent.png"))

    def test_corrupt_image_raises(self):
        """A file with invalid bytes should raise ValueError, not silently produce garbage.

        Verifies the fixture exists first so a missing fixture is reported
        as a setup error rather than masquerading as a passing test.
        """
        assert _CORRUPT_IMAGE.exists(), f"Test fixture missing: {_CORRUPT_IMAGE}. Restore from tests/data/ before running."
        with pytest.raises(ValueError, match="decode|corrupt"):
            pipeline.run_pipeline(str(_CORRUPT_IMAGE))

    def test_missing_lane_edges_raises(self, tmp_path):
        """Pipeline must raise when lane edges are missing and no fallback exists."""
        with pytest.raises(FileNotFoundError, match="Lane edges not found"):
            pipeline.run_pipeline(
                str(_VALID_IMAGE),
                lane_edges_path=str(tmp_path / "does_not_exist.npy"),
            )
