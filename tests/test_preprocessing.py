"""
Unit tests for cv4.preprocessing.

All inputs are synthetic numpy arrays or tmp files so tests stay fast,
offline, and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cv4 import preprocessing

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_gray(h: int = 512, w: int = 512, fill: int = 128) -> np.ndarray:
    """Return a solid-colour grayscale image."""
    return np.full((h, w), fill, dtype=np.uint8)


def _make_dish_image(h: int = 600, w: int = 800) -> np.ndarray:
    """Grayscale image with a bright circular disc near the centre."""
    img = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = min(h, w) // 3
    cv2.circle(img, (cx, cy), r, 200, -1)
    return img


@pytest.fixture()
def dish_image() -> np.ndarray:
    return _make_dish_image()


@pytest.fixture()
def tmp_png(tmp_path: Path) -> Path:
    """Write a real PNG to a temp file and return its path."""
    img = _make_dish_image()
    p = tmp_path / "test.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture()
def tmp_corrupt(tmp_path: Path) -> Path:
    """A file that exists but is not a valid image."""
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"not an image")
    return p


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_loads_valid_png(self, tmp_png: Path):
        img = preprocessing.load_image(str(tmp_png))
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2

    def test_returns_uint8(self, tmp_png: Path):
        img = preprocessing.load_image(str(tmp_png))
        assert img.dtype == np.uint8

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            preprocessing.load_image(str(tmp_path / "nonexistent.png"))

    def test_corrupt_file_raises_value_error(self, tmp_corrupt: Path):
        with pytest.raises(ValueError, match="Could not decode"):
            preprocessing.load_image(str(tmp_corrupt))


# ---------------------------------------------------------------------------
# detect_petri_bbox
# ---------------------------------------------------------------------------


class TestDetectPetriBbox:
    def test_returns_three_ints(self, dish_image: np.ndarray):
        result = preprocessing.detect_petri_bbox(dish_image)
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_side_positive(self, dish_image: np.ndarray):
        _, _, side = preprocessing.detect_petri_bbox(dish_image)
        assert side > 0

    def test_bbox_within_image_bounds(self, dish_image: np.ndarray):
        h, w = dish_image.shape
        x, y, side = preprocessing.detect_petri_bbox(dish_image)
        assert 0 <= x
        assert 0 <= y
        assert x + side <= w
        assert y + side <= h

    def test_fallback_on_blank_image(self):
        """Blank (uniform) image triggers the fallback centred square."""
        blank = np.zeros((400, 600), dtype=np.uint8)
        x, y, side = preprocessing.detect_petri_bbox(blank)
        assert side == min(400, 600)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            preprocessing.detect_petri_bbox(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_detects_dish_near_centre(self, dish_image: np.ndarray):
        h, w = dish_image.shape
        x, y, side = preprocessing.detect_petri_bbox(dish_image)
        cx = x + side / 2
        cy = y + side / 2
        assert abs(cx - w / 2) < w * 0.25, f"Dish centre x={cx} too far from image centre"
        assert abs(cy - h / 2) < h * 0.25, f"Dish centre y={cy} too far from image centre"


# ---------------------------------------------------------------------------
# refine_to_inner_plate
# ---------------------------------------------------------------------------


class TestRefineToInnerPlate:
    def test_returns_three_ints(self, dish_image: np.ndarray):
        bbox = preprocessing.detect_petri_bbox(dish_image)
        result = preprocessing.refine_to_inner_plate(dish_image, bbox)
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_side_positive(self, dish_image: np.ndarray):
        bbox = preprocessing.detect_petri_bbox(dish_image)
        _, _, side = preprocessing.refine_to_inner_plate(dish_image, bbox)
        assert side > 0

    def test_refined_within_image_bounds(self, dish_image: np.ndarray):
        h, w = dish_image.shape
        bbox = preprocessing.detect_petri_bbox(dish_image)
        x, y, side = preprocessing.refine_to_inner_plate(dish_image, bbox)
        assert x >= 0
        assert y >= 0
        assert x + side <= w
        assert y + side <= h

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            preprocessing.refine_to_inner_plate(np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 10))

    def test_pad_zero_gives_tight_bbox(self, dish_image: np.ndarray):
        """pad=0 should produce a smaller or equal side than pad=20."""
        bbox = preprocessing.detect_petri_bbox(dish_image)
        _, _, side_tight = preprocessing.refine_to_inner_plate(dish_image, bbox, pad=0)
        _, _, side_padded = preprocessing.refine_to_inner_plate(dish_image, bbox, pad=20)
        assert side_tight <= side_padded


# ---------------------------------------------------------------------------
# crop_square
# ---------------------------------------------------------------------------


class TestCropSquare:
    def test_output_shape_matches_side(self):
        img = _make_gray(300, 400)
        cropped = preprocessing.crop_square(img, (50, 30, 100))
        assert cropped.shape == (100, 100)

    def test_top_left_crop(self):
        img = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        cropped = preprocessing.crop_square(img, (0, 0, 10))
        assert np.array_equal(cropped, img[:10, :10])

    def test_offset_crop(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50:80, 60:90] = 255
        cropped = preprocessing.crop_square(img, (60, 50, 30))
        assert cropped.shape == (30, 30)
        assert cropped.max() == 255


# ---------------------------------------------------------------------------
# pad_to_patch_size
# ---------------------------------------------------------------------------


class TestPadToPatchSize:
    def test_output_divisible_by_patch_size(self):
        img = _make_gray(300, 500)
        padded, _, _ = preprocessing.pad_to_patch_size(img, patch_size=256)
        assert padded.shape[0] % 256 == 0
        assert padded.shape[1] % 256 == 0

    def test_already_aligned_image_unchanged(self):
        img = _make_gray(256, 512)
        padded, pad_h, pad_w = preprocessing.pad_to_patch_size(img, patch_size=256)
        assert padded.shape == img.shape
        assert pad_h == (0, 0)
        assert pad_w == (0, 0)

    def test_returns_three_elements(self):
        img = _make_gray(300, 300)
        result = preprocessing.pad_to_patch_size(img)
        assert len(result) == 3

    def test_padding_tuples_are_pairs(self):
        img = _make_gray(300, 300)
        _, pad_h, pad_w = preprocessing.pad_to_patch_size(img)
        assert len(pad_h) == 2
        assert len(pad_w) == 2

    def test_padding_is_symmetric_when_even(self):
        """When the required pad is even, top==bottom and left==right."""
        img = _make_gray(254, 510)  # needs +2 in h, +2 in w
        _, pad_h, pad_w = preprocessing.pad_to_patch_size(img, patch_size=256)
        assert pad_h[0] == pad_h[1]
        assert pad_w[0] == pad_w[1]

    def test_fill_value_respected(self):
        img = _make_gray(100, 100, fill=50)
        padded, (top, _), (left, _) = preprocessing.pad_to_patch_size(img, patch_size=256, value=99)
        if top > 0:
            assert padded[0, 0] == 99
        if left > 0:
            assert padded[top, 0] == 99

    def test_padded_size_correct(self):
        img = _make_gray(300, 700)
        padded, _, _ = preprocessing.pad_to_patch_size(img, patch_size=256)
        assert padded.shape[0] == 512  # next multiple of 256 above 300
        assert padded.shape[1] == 768  # next multiple of 256 above 700


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def test_returns_dict_with_required_keys(self, tmp_png: Path):
        result = preprocessing.preprocess_image(str(tmp_png))
        required = {"original_image", "original_shape", "bbox", "cropped_dish", "padded_dish", "pad_h", "pad_w"}
        assert required.issubset(result.keys())

    def test_padded_dish_divisible_by_patch_size(self, tmp_png: Path):
        result = preprocessing.preprocess_image(str(tmp_png))
        padded = result["padded_dish"]
        assert padded.shape[0] % 256 == 0
        assert padded.shape[1] % 256 == 0

    def test_cropped_dish_is_square(self, tmp_png: Path):
        result = preprocessing.preprocess_image(str(tmp_png))
        crop = result["cropped_dish"]
        assert crop.shape[0] == crop.shape[1]

    def test_original_shape_matches_loaded_image(self, tmp_png: Path):
        result = preprocessing.preprocess_image(str(tmp_png))
        assert result["original_shape"] == result["original_image"].shape

    def test_file_not_found_propagates(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            preprocessing.preprocess_image(str(tmp_path / "missing.png"))

    def test_corrupt_file_propagates(self, tmp_corrupt: Path):
        with pytest.raises(ValueError):
            preprocessing.preprocess_image(str(tmp_corrupt))

    def test_custom_patch_size(self, tmp_png: Path):
        result = preprocessing.preprocess_image(str(tmp_png), patch_size=128)
        padded = result["padded_dish"]
        assert padded.shape[0] % 128 == 0
        assert padded.shape[1] % 128 == 0


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    def test_functions_callable_independently(self, dish_image, tmp_png):
        """Each function can be called on its own without the pipeline wrapper."""
        img = preprocessing.load_image(str(tmp_png))
        bbox = preprocessing.detect_petri_bbox(img)
        bbox = preprocessing.refine_to_inner_plate(img, bbox)
        crop = preprocessing.crop_square(img, bbox)
        padded, pad_h, pad_w = preprocessing.pad_to_patch_size(crop)
        assert padded.ndim == 2

    def test_no_raw_traceback_on_missing_file(self, tmp_path):
        """Missing file raises a clean exception, not a raw OpenCV traceback."""
        with pytest.raises(FileNotFoundError) as exc_info:
            preprocessing.preprocess_image(str(tmp_path / "ghost.png"))
        assert "ghost.png" in str(exc_info.value)

    def test_no_raw_traceback_on_corrupt_file(self, tmp_corrupt):
        with pytest.raises(ValueError) as exc_info:
            preprocessing.load_image(str(tmp_corrupt))
        assert "corrupt" in str(exc_info.value).lower() or "decode" in str(exc_info.value).lower()

    def test_logging_emitted_on_load(self, tmp_png, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="cv4.preprocessing"):
            preprocessing.load_image(str(tmp_png))
        assert any("Loaded image" in r.message for r in caplog.records)

    def test_logging_emitted_on_detect(self, dish_image, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="cv4.preprocessing"):
            preprocessing.detect_petri_bbox(dish_image)
        assert any("bounding box" in r.message or "Detected" in r.message for r in caplog.records)
