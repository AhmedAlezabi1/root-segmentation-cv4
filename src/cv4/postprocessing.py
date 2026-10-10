"""
postprocessing.py
=================
In-memory post-processing primitives for the root segmentation pipeline.

Covers notebook cells 3, 4, and 7:

    Cell 3  → apply_double_threshold   – hysteresis thresholding on a
                                         probability map → binary mask.
    Cell 4  → unpad                    – remove padding added by inference
                                         preprocessing.
    Cell 7  → micro_close              – morphological closing on a mask.
              skeleton_endpoints       – find degree-1 pixels in a skeleton.
              bridge_mask_gaps         – connect split root segments.
              postprocess_mask         – single-array wrapper that pipeline.py
                                         calls: threshold → unpad → close →
                                         bridge, all in memory.

The batch file-based wrapper ``apply_micro_close_and_bridge`` is also kept
for offline / notebook use.  It reads files from disk, calls the in-memory
primitives in a loop, and writes results back to disk.

``pipeline.py`` should call ``postprocess_mask``; notebook or script users
can call the individual primitives or the batch wrapper as needed.

All tuneable constants come from ``cv4.config``.  Every public primitive
accepts plain keyword arguments rather than a config dataclass, so
``pipeline.py`` can call them without importing an extra type.  The
dataclasses (``MicroCloseConfig``, ``BridgeConfig``) exist for the batch
wrapper and for users who prefer named configuration objects.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage import measure, morphology
from skimage.filters import apply_hysteresis_threshold

from cv4 import config

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration dataclasses  (used by the batch wrapper)
# =============================================================================


@dataclass
class MicroCloseConfig:
    """Parameters for the morphological closing step (notebook cell 7).

    Attributes
    ----------
    kernel_size:
        Side length of the elliptical structuring element in pixels.
    iterations:
        How many times to apply the closing operation.
    """

    kernel_size: int = config.MICRO_CLOSE_KERNEL_SIZE
    iterations: int = config.MICRO_CLOSE_ITERATIONS


@dataclass
class BridgeConfig:
    """Parameters for skeleton gap bridging (notebook cell 7).

    Attributes
    ----------
    max_dist:
        Maximum Euclidean distance between two endpoints that may be bridged.
    max_dx:
        Maximum horizontal delta — keeps bridges roughly vertical so they
        do not accidentally cross into a neighbouring lane.
    min_dy:
        Minimum vertical delta — prevents a single endpoint from bridging
        to its immediate neighbour (self-loop avoidance).
    thickness:
        Width of the line drawn when connecting two endpoints.
    enforce_same_lane:
        When ``True``, only bridge endpoints that fall in the same
        evenly-spaced lane bucket.
    expected_roots:
        Number of root lanes — used only when ``enforce_same_lane=True``.
    """

    max_dist: float = config.BRIDGE_MAX_DIST
    max_dx: int = config.BRIDGE_MAX_DX
    min_dy: int = config.BRIDGE_MIN_DY
    thickness: int = config.BRIDGE_THICKNESS
    enforce_same_lane: bool = config.BRIDGE_ENFORCE_SAME_LANE
    expected_roots: int = config.EXPECTED_ROOTS


# =============================================================================
# Cell 3 — Hysteresis thresholding
# =============================================================================


def apply_double_threshold(
    prob_map: np.ndarray,
    low: float = config.HYSTERESIS_LOW,
    high: float = config.HYSTERESIS_HIGH,
) -> np.ndarray:
    """Convert a probability map to a binary mask using hysteresis thresholding.

    Pixels at or above *high* are definite roots.  Pixels between *low* and
    *high* are kept only if they are connected to a definite pixel.  This
    two-level approach is more robust than a single threshold because it
    avoids breaking faint but connected root segments.

    Parameters
    ----------
    prob_map:
        Float array of shape ``(H, W)`` with values in ``[0, 1]``, as
        returned by ``inference.predict_smooth_windowing``.
    low:
        Lower hysteresis threshold.  Connected pixels above this value are
        retained.
    high:
        Upper hysteresis threshold.  Pixels above this value are always kept.

    Returns
    -------
    np.ndarray
        Binary uint8 mask of shape ``(H, W)`` with values 0 or 1.

    Raises
    ------
    ValueError
        If *low* is greater than *high*, or if *prob_map* is not 2-D.
    """
    if prob_map.ndim != 2:
        raise ValueError(f"prob_map must be 2-D, got shape {prob_map.shape}")
    if low > high:
        raise ValueError(f"low threshold ({low}) must be <= high threshold ({high})")

    logger.debug(
        "apply_double_threshold: low=%.3f high=%.3f map shape=%s",
        low,
        high,
        prob_map.shape,
    )

    binary = apply_hysteresis_threshold(prob_map, low, high)
    return binary.astype(np.uint8)


# =============================================================================
# Cell 4 — Remove inference padding
# =============================================================================


def unpad(
    mask: np.ndarray,
    pad_h: tuple[int, int],
    pad_w: tuple[int, int],
) -> np.ndarray:
    """Remove padding that was added by the inference preprocessing step.

    Parameters
    ----------
    mask:
        Padded array of any dtype, shape ``(H, W)`` or ``(H, W, C)``.
    pad_h:
        ``(top, bottom)`` number of rows to strip.
    pad_w:
        ``(left, right)`` number of columns to strip.

    Returns
    -------
    np.ndarray
        Array with padding removed.  Same dtype as *mask*.

    Raises
    ------
    ValueError
        If the padding values are larger than the array dimensions, which
        would result in an empty or negative-size array.
    """
    top, bottom = pad_h
    left, right = pad_w

    if top + bottom >= mask.shape[0] or left + right >= mask.shape[1]:
        raise ValueError(f"Padding {pad_h}, {pad_w} is too large for mask shape {mask.shape}")

    logger.debug(
        "unpad: removing pad_h=%s pad_w=%s from shape %s",
        pad_h,
        pad_w,
        mask.shape,
    )

    return mask[top : mask.shape[0] - bottom, left : mask.shape[1] - right]


# =============================================================================
# Cell 7 — In-memory closing, endpoint detection, bridging
# =============================================================================


def micro_close(
    mask: np.ndarray,
    kernel_size: int = config.MICRO_CLOSE_KERNEL_SIZE,
    iterations: int = config.MICRO_CLOSE_ITERATIONS,
) -> np.ndarray:
    """Apply morphological closing to seal small gaps in a binary mask.

    Uses an elliptical structuring element, which is more isotropic than a
    rectangular one and better suited to thin curved root shapes.

    Parameters
    ----------
    mask:
        Binary mask (0/1 or 0/255, any dtype).
    kernel_size:
        Side length of the elliptical structuring element in pixels.
    iterations:
        How many times to apply the closing operation.

    Returns
    -------
    np.ndarray
        Binary uint8 mask after closing.

    Raises
    ------
    ValueError
        If *kernel_size* is less than 1 or *iterations* is less than 1.
    """
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    logger.debug("micro_close: kernel=%dpx iterations=%d", kernel_size, iterations)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(
        (mask > 0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
    )
    return closed


def skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return (y, x) coordinates of skeleton pixels with exactly one neighbour.

    A pixel is an endpoint when the 3×3 neighbourhood sum (excluding the
    pixel itself) equals 1 — i.e. it has exactly one connected neighbour in
    the skeleton.

    Parameters
    ----------
    skel:
        Binary skeleton image (0/1, dtype uint8 or bool).

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` with columns ``[y, x]``.  Returns shape ``(0, 2)``
        when no endpoints exist.
    """
    skel = skel.astype(np.uint8)
    neighbour_count = cv2.filter2D(skel, ddepth=-1, kernel=np.ones((3, 3), np.uint8)) - skel
    ys, xs = np.where((skel == 1) & (neighbour_count == 1))
    return np.stack([ys, xs], axis=1) if len(xs) else np.zeros((0, 2), dtype=np.int32)


def bridge_mask_gaps(
    mask: np.ndarray,
    max_dist: float = config.BRIDGE_MAX_DIST,
    max_dx: int = config.BRIDGE_MAX_DX,
    min_dy: int = config.BRIDGE_MIN_DY,
    thickness: int = config.BRIDGE_THICKNESS,
    enforce_same_lane: bool = config.BRIDGE_ENFORCE_SAME_LANE,
    expected_roots: int = config.EXPECTED_ROOTS,
) -> np.ndarray:
    """Connect nearby skeleton endpoints across separate components.

    Skeletonises *mask*, finds all degree-1 endpoint pixels, then for each
    unpaired endpoint searches for the nearest compatible partner in a
    different component.  A straight line is drawn between the pair if all
    constraints are satisfied.  Each endpoint is used in at most one bridge.

    Parameters
    ----------
    mask:
        Binary mask (0/1 or 0/255, any dtype).
    max_dist:
        Maximum Euclidean distance between two endpoints that may be bridged.
    max_dx:
        Maximum horizontal delta — keeps bridges roughly vertical.
    min_dy:
        Minimum vertical delta — prevents self-loop bridges.
    thickness:
        Width of the bridge line in pixels.
    enforce_same_lane:
        When ``True``, only bridge endpoints in the same lane bucket.
    expected_roots:
        Number of root lanes — used only when ``enforce_same_lane=True``.

    Returns
    -------
    np.ndarray
        Binary uint8 mask with bridge lines drawn in.
    """
    mask = (mask > 0).astype(np.uint8)
    H, W = mask.shape

    skel = morphology.skeletonize(mask.astype(bool)).astype(np.uint8)
    component_labels = measure.label(mask, connectivity=2)
    endpoints = skeleton_endpoints(skel)

    logger.debug(
        "bridge_mask_gaps: %d endpoints found in mask of shape %s",
        endpoints.shape[0],
        mask.shape,
    )

    if endpoints.shape[0] < 2:
        return mask

    out = mask.copy()
    used = np.zeros(endpoints.shape[0], dtype=bool)
    bridges_drawn = 0

    for i, (y1, x1) in enumerate(endpoints):
        if used[i]:
            continue
        comp1 = component_labels[y1, x1]
        if comp1 == 0:
            continue

        lane1 = _lane_bucket(x1, W, expected_roots) if enforce_same_lane else None
        best_j, best_dist = None, float("inf")

        for j, (y2, x2) in enumerate(endpoints):
            if i == j or used[j]:
                continue
            comp2 = component_labels[y2, x2]
            if comp2 == 0 or comp2 == comp1:
                continue

            dx = abs(int(x2) - int(x1))
            dy = abs(int(y2) - int(y1))
            if dx > max_dx or dy < min_dy:
                continue
            if enforce_same_lane and _lane_bucket(x2, W, expected_roots) != lane1:
                continue

            dist = math.hypot(dx, dy)
            if dist <= max_dist and dist < best_dist:
                best_dist, best_j = dist, j

        if best_j is not None:
            y2, x2 = endpoints[best_j]
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness)
            used[i] = used[best_j] = True
            bridges_drawn += 1

    logger.debug("bridge_mask_gaps: %d bridge(s) drawn", bridges_drawn)
    return out


def postprocess_mask(
    prob_map: np.ndarray,
    pad_h: tuple[int, int],
    pad_w: tuple[int, int],
    low: float = config.HYSTERESIS_LOW,
    high: float = config.HYSTERESIS_HIGH,
    kernel_size: int = config.MICRO_CLOSE_KERNEL_SIZE,
    iterations: int = config.MICRO_CLOSE_ITERATIONS,
    max_dist: float = config.BRIDGE_MAX_DIST,
    max_dx: int = config.BRIDGE_MAX_DX,
    min_dy: int = config.BRIDGE_MIN_DY,
    thickness: int = config.BRIDGE_THICKNESS,
    enforce_same_lane: bool = config.BRIDGE_ENFORCE_SAME_LANE,
    expected_roots: int = config.EXPECTED_ROOTS,
) -> np.ndarray:
    """Run the full post-processing chain on a single probability map.

    This is the function ``pipeline.py`` calls.  It chains the four
    in-memory primitives in order:

    1. ``apply_double_threshold`` — probability map → binary mask
    2. ``unpad``                  — remove inference padding
    3. ``micro_close``            — seal small gaps
    4. ``bridge_mask_gaps``       — reconnect split root segments

    All parameters have defaults from ``config.py``, so ``pipeline.py`` only
    needs to supply *prob_map*, *pad_h*, and *pad_w*.

    Parameters
    ----------
    prob_map:
        Float probability map of shape ``(H_padded, W_padded)`` as returned
        by ``inference.predict_smooth_windowing``.
    pad_h:
        ``(top, bottom)`` padding rows to remove after thresholding.
    pad_w:
        ``(left, right)`` padding columns to remove after thresholding.
    low:
        Lower hysteresis threshold.
    high:
        Upper hysteresis threshold.
    kernel_size:
        Closing kernel size in pixels.
    iterations:
        Number of closing iterations.
    max_dist:
        Maximum bridge distance in pixels.
    max_dx:
        Maximum horizontal delta for bridging.
    min_dy:
        Minimum vertical delta for bridging.
    thickness:
        Bridge line thickness in pixels.
    enforce_same_lane:
        Restrict bridging to same-lane endpoints.
    expected_roots:
        Number of root lanes.

    Returns
    -------
    np.ndarray
        Binary uint8 mask ready to pass to ``segmentation.py``.
    """
    logger.info(
        "postprocess_mask: prob_map shape=%s pad_h=%s pad_w=%s",
        prob_map.shape,
        pad_h,
        pad_w,
    )

    mask = apply_double_threshold(prob_map, low=low, high=high)
    logger.debug("postprocess_mask: after threshold — %d root pixels", mask.sum())

    mask = unpad(mask, pad_h, pad_w)
    logger.debug("postprocess_mask: after unpad — shape %s", mask.shape)

    mask = micro_close(mask, kernel_size=kernel_size, iterations=iterations)
    logger.debug("postprocess_mask: after closing — %d root pixels", mask.sum())

    mask = bridge_mask_gaps(
        mask,
        max_dist=max_dist,
        max_dx=max_dx,
        min_dy=min_dy,
        thickness=thickness,
        enforce_same_lane=enforce_same_lane,
        expected_roots=expected_roots,
    )
    logger.debug("postprocess_mask: after bridging — %d root pixels", mask.sum())

    return mask


# =============================================================================
# Cell 7 — Batch file wrapper  (offline / notebook use)
# =============================================================================


def apply_micro_close_and_bridge(
    in_dir: str,
    out_dir: str,
    micro_cfg: Optional[MicroCloseConfig] = None,
    bridge_cfg: Optional[BridgeConfig] = None,
) -> str:
    """Apply closing and bridging to every mask in *in_dir*, save results.

    This is the file-based batch version of ``micro_close`` +
    ``bridge_mask_gaps``.  It reads every ``*_prediction.png`` from
    *in_dir*, runs both operations in memory using the in-memory primitives,
    then writes results as
    ``{stem}_prediction_microclose_bridge.png`` in *out_dir*.

    For single-array use inside ``pipeline.py`` call ``postprocess_mask``
    instead.

    Parameters
    ----------
    in_dir:
        Directory containing ``*_prediction.png`` masks (output of cell 5).
    out_dir:
        Directory where processed masks are written.  Created if absent.
    micro_cfg:
        Closing parameters.  ``None`` uses the defaults from ``config.py``.
    bridge_cfg:
        Bridging parameters.  ``None`` uses the defaults from ``config.py``.

    Returns
    -------
    str
        *out_dir* — returned for convenient call chaining.

    Raises
    ------
    FileNotFoundError
        If *in_dir* contains no ``*_prediction.png`` files.
    OSError
        If an output mask cannot be written to disk.
    """
    micro_cfg = micro_cfg or MicroCloseConfig()
    bridge_cfg = bridge_cfg or BridgeConfig()
    os.makedirs(out_dir, exist_ok=True)

    pred_paths = sorted(glob(os.path.join(in_dir, "*_prediction.png")))
    if not pred_paths:
        raise FileNotFoundError(f"No *_prediction.png masks found in: {in_dir}")

    logger.info("apply_micro_close_and_bridge: found %d masks in %s", len(pred_paths), in_dir)
    logger.debug("Closing kernel=%dpx iterations=%d", micro_cfg.kernel_size, micro_cfg.iterations)

    skipped = 0
    for path in pred_paths:
        m255 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m255 is None:
            logger.warning("Could not read mask, skipping: %s", path)
            skipped += 1
            continue

        mask = (m255 > 127).astype(np.uint8)
        mask = micro_close(
            mask,
            kernel_size=micro_cfg.kernel_size,
            iterations=micro_cfg.iterations,
        )
        mask = bridge_mask_gaps(
            mask,
            max_dist=bridge_cfg.max_dist,
            max_dx=bridge_cfg.max_dx,
            min_dy=bridge_cfg.min_dy,
            thickness=bridge_cfg.thickness,
            enforce_same_lane=bridge_cfg.enforce_same_lane,
            expected_roots=bridge_cfg.expected_roots,
        )

        stem = Path(path).stem.replace("_prediction", "")
        out_file = os.path.join(out_dir, f"{stem}_prediction_microclose_bridge.png")
        if not cv2.imwrite(out_file, (mask * 255).astype(np.uint8)):
            raise OSError(f"Failed to write output mask: {out_file}")

    processed = len(pred_paths) - skipped
    logger.info(
        "apply_micro_close_and_bridge: %d processed, %d skipped → %s",
        processed,
        skipped,
        out_dir,
    )
    if skipped:
        logger.warning("%d mask(s) could not be read and were skipped", skipped)
    return out_dir


# =============================================================================
# Internal helper
# =============================================================================


def _lane_bucket(x: float, width: int, n_lanes: int) -> int:
    """Map an x-coordinate to a coarse lane bucket index in [0, n_lanes - 1].

    Divides the image into *n_lanes* equal-width vertical strips.  Used only
    by ``bridge_mask_gaps`` when ``enforce_same_lane=True``.
    """
    bins = np.linspace(0, width, n_lanes + 1)
    return max(0, min(n_lanes - 1, int(np.searchsorted(bins, x, side="right")) - 1))
