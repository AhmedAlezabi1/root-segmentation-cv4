"""
Preprocessing module for the cv4 plant root segmentation pipeline.

Handles everything that happens to a raw grayscale image before it reaches
the model: loading, petri dish detection, inner-plate refinement, cropping,
and patch-aligned padding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load and validate a grayscale image from disk.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        Grayscale image as a 2-D uint8 numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as a valid image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image (unsupported format or corrupt file): {path}")

    logger.info("Loaded image with shape %s from %s", img.shape, path)
    return img


def detect_petri_bbox(gray: np.ndarray) -> tuple[int, int, int]:
    """Find the petri dish bounding box in a grayscale image.

    Uses adaptive thresholding and morphological operations to detect candidate
    contours, then selects the contour that is large and close to the image
    centre.  Falls back to a centred square covering the shorter image
    dimension when no contour is found.

    Args:
        gray: 2-D uint8 grayscale image.

    Returns:
        Tuple ``(x, y, side)`` — top-left corner and side length of the square
        bounding box around the dish (in pixels).

    Raises:
        ValueError: If ``gray`` is not a 2-D array.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {gray.shape}")

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10)
    if np.mean(thr) < 127:
        thr = 255 - thr

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape

    if not cnts:
        side = min(h, w)
        bbox = ((w - side) // 2, (h - side) // 2, side)
        logger.warning("No contours found — falling back to centred square bbox %s", bbox)
        return bbox

    cx, cy = w / 2.0, h / 2.0

    def _score(c):
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        mx, my = bx + bw / 2.0, by + bh / 2.0
        dist = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
        return area - 0.1 * dist

    best = max(cnts, key=_score)
    bx, by, bw, bh = cv2.boundingRect(best)

    side = max(bw, bh)
    side = min(side, h, w)
    mx, my = bx + bw // 2, by + bh // 2

    xs = int(mx - side / 2)
    ys = int(my - side / 2)
    xs = max(0, min(xs, w - side))
    ys = max(0, min(ys, h - side))

    bbox = (xs, ys, side)
    logger.info("Detected dish bounding box at x=%d y=%d side=%d", xs, ys, side)
    return bbox


def refine_to_inner_plate(
    gray: np.ndarray,
    bbox: tuple[int, int, int],
    pad: int = 20,
) -> tuple[int, int, int]:
    """Tighten an outer dish bbox to the inner plate region.

    Computes row and column intensity profiles inside the initial crop,
    applies Otsu thresholding on each profile, finds the longest contiguous
    foreground run, and centres a square bounding box on that region.

    Args:
        gray: 2-D uint8 grayscale image (full resolution).
        bbox: Outer bounding box ``(x, y, side)`` from :func:`detect_petri_bbox`.
        pad: Extra pixels to add around the detected foreground region before
            computing the final square.  Defaults to 20.

    Returns:
        Refined ``(x, y, side)`` bounding box in full-image coordinates.

    Raises:
        ValueError: If ``gray`` is not a 2-D array.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {gray.shape}")

    x, y, s = bbox
    crop = gray[y : y + s, x : x + s]

    rmean = crop.mean(axis=1)
    cmean = crop.mean(axis=0)

    _, rbin = cv2.threshold(rmean.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, cbin = cv2.threshold(cmean.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rbin = rbin > 0
    cbin = cbin > 0

    def _longest_run(b: np.ndarray) -> tuple[int, int]:
        best = (0, -1)
        i = 0
        while i < len(b):
            if b[i]:
                j = i
                while j < len(b) and b[j]:
                    j += 1
                if (j - 1) - i > best[1] - best[0]:
                    best = (i, j - 1)
                i = j
            else:
                i += 1
        return best

    r0, r1 = _longest_run(rbin)
    c0, c1 = _longest_run(cbin)

    r0 = max(0, r0 - pad)
    r1 = min(s - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(s - 1, c1 + pad)

    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    side = min(hh, ww)

    cy_inner = (r0 + r1) // 2
    cx_inner = (c0 + c1) // 2

    r0_new = max(0, cy_inner - side // 2)
    c0_new = max(0, cx_inner - side // 2)

    refined = (x + c0_new, y + r0_new, side)
    logger.info("Refined inner-plate bbox to x=%d y=%d side=%d", *refined)
    return refined


def crop_square(img: np.ndarray, bbox: tuple[int, int, int]) -> np.ndarray:
    """Crop a square region from *img* using a ``(x, y, side)`` bounding box.

    Args:
        img: 2-D numpy array (grayscale image).
        bbox: ``(x, y, side)`` tuple — top-left column, top-left row, side length.

    Returns:
        Cropped 2-D numpy array of shape ``(side, side)``.
    """
    x, y, s = bbox
    cropped = img[y : y + s, x : x + s]
    logger.info("Cropped image to shape %s using bbox %s", cropped.shape, bbox)
    return cropped


def pad_to_patch_size(
    img: np.ndarray,
    patch_size: int = 256,
    value: int = 0,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Symmetrically pad a 2-D image so its dimensions are divisible by *patch_size*.

    Args:
        img: 2-D uint8 grayscale image.
        patch_size: Target divisor for both height and width.  Defaults to 256.
        value: Constant fill value for the border pixels.  Defaults to 0 (black).

    Returns:
        A tuple ``(padded_image, (top, bottom), (left, right))`` where the
        second and third elements describe the padding applied to each side.
    """
    h, w = img.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
    logger.info(
        "Padded image from %s to %s (top=%d bottom=%d left=%d right=%d)",
        img.shape,
        padded.shape,
        top,
        bottom,
        left,
        right,
    )
    return padded, (top, bottom), (left, right)


def preprocess_image(path: str, patch_size: int = 256) -> dict:
    """Run the full preprocessing pipeline on a single image.

    Chains :func:`load_image`, :func:`detect_petri_bbox`,
    :func:`refine_to_inner_plate`, :func:`crop_square`, and
    :func:`pad_to_patch_size` into a single convenience call.

    Args:
        path: Path to the input image file.
        patch_size: Patch size used for padding alignment.  Defaults to 256.

    Returns:
        Dictionary with keys:

        - ``original_image``: Full-resolution grayscale array.
        - ``original_shape``: ``(H, W)`` of the original image.
        - ``bbox``: ``(x, y, side)`` of the refined inner-plate crop.
        - ``cropped_dish``: Square crop of the inner plate.
        - ``padded_dish``: Patch-aligned padded crop ready for inference.
        - ``pad_h``: ``(top, bottom)`` padding applied vertically.
        - ``pad_w``: ``(left, right)`` padding applied horizontally.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file cannot be decoded as a valid image, or if no
            petri dish is detected and the fallback bbox produces an empty crop.
    """
    logger.info("Starting preprocessing for %s", path)

    img = load_image(path)
    original_shape = img.shape

    bbox = detect_petri_bbox(img)
    bbox = refine_to_inner_plate(img, bbox, pad=20)
    cropped = crop_square(img, bbox)

    if cropped.size == 0:
        raise ValueError(f"Dish not detected — crop is empty for image: {path}")

    padded, pad_h, pad_w = pad_to_patch_size(cropped, patch_size=patch_size)

    logger.info(
        "Preprocessing complete: original=%s bbox=%s crop=%s padded=%s",
        original_shape,
        bbox,
        cropped.shape,
        padded.shape,
    )

    return {
        "original_image": img,
        "original_shape": original_shape,
        "bbox": bbox,
        "cropped_dish": cropped,
        "padded_dish": padded,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }
