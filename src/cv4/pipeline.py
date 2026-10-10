"""
Pipeline orchestrator for the cv4 plant root segmentation system.

Chains the five module-level steps (preprocessing, inference,
postprocessing, segmentation, localization) into a single
``run_pipeline`` call that takes a petri dish image path and returns
structured results: 5 plant instance masks, root tip coordinates in
pixels and mm, per-plant confidence scores, and timing info.

The pipeline is the entry point that ``cli.py`` and the FastAPI
endpoint will both call. It owns module composition only — every step
delegates to the relevant module function with no inline algorithm
logic.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from cv4 import (
    config,
    inference,
    localization,
    postprocessing,
    preprocessing,
    segmentation,
)

logger = logging.getLogger(__name__)


def _resolve_lane_edges(
    lane_edges: np.ndarray | None,
    lane_edges_path: str | None,
) -> np.ndarray:
    """Load lane edges from the given source, with a clear error if missing.

    Resolution order:
    1. If ``lane_edges`` is provided directly, use it.
    2. If ``lane_edges_path`` is provided, load from that path.
    3. Otherwise fall back to ``config.DEFAULT_LANE_EDGES_PATH``.
    """
    if lane_edges is not None:
        return lane_edges

    path = Path(lane_edges_path or config.DEFAULT_LANE_EDGES_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Lane edges not found at {path}. Run scripts/generate_lane_edges.py to create the file, or pass `lane_edges` directly to run_pipeline()."
        )

    logger.info("Loading lane edges from %s", path)
    return np.load(path)


def _resolve_model(model, model_path: str | None):
    """Use a pre-loaded model if given, otherwise load from disk.

    A pre-loaded ``model`` takes precedence over ``model_path``. Loading
    is the slow part (~30s on cold cache), so the API keeps the model
    in memory and passes it in for every request.
    """
    if model is not None:
        logger.debug("Using pre-loaded model")
        return model
    return inference.load_model(model_path)


def run_pipeline(
    image_path: str,
    model_path: str | None = None,
    lane_edges: np.ndarray | None = None,
    lane_edges_path: str | None = None,
    model=None,
) -> dict:
    """Run the full plant root segmentation pipeline on a single image.

    Parameters
    ----------
    image_path:
        Path to a petri dish image (PNG, JPG, or TIFF).
    model_path:
        Path to the trained UNet weights. Used only if ``model`` is None.
        Defaults to ``config.DEFAULT_MODEL_PATH``.
    lane_edges:
        Precomputed lane edge array. If provided, takes precedence over
        ``lane_edges_path``. Useful when the caller has already loaded
        the file once and wants to reuse it across many pipeline calls.
    lane_edges_path:
        Path to ``lane_edges.npy``. Used only if ``lane_edges`` is None.
        Defaults to ``config.DEFAULT_LANE_EDGES_PATH``.
    model:
        Pre-loaded Keras model. Takes precedence over ``model_path``.
        For long-running services (FastAPI), load the model once at
        startup and pass it in for every request to skip the load cost.

    Returns
    -------
    dict
        ``image_path``: input path, echoed for traceability.
        ``binary_mask``: full clean binary mask (np.ndarray, unpadded).
        ``plant_masks``: list of 5 per-lane masks (np.ndarray each).
        ``plants``: list of 5 dicts, one per lane. Always contains
            ``lane`` (int), ``has_root`` (bool), and ``main_root_path``
            (list of (y, x) tuples, empty if no root). When ``has_root``
            is True, also contains ``root_tip_px`` ({"x", "y"} dict),
            ``root_tip_mm`` ({"x", "y"} dict), and ``confidence`` (float).
            When ``has_root`` is False, those three fields are None so
            callers can't confuse them with valid measurements at (0, 0).
        ``processing_time_seconds``: float, end-to-end wall time.
        ``plate_bbox``: (x, y, side) tuple from preprocessing.

    Raises
    ------
    FileNotFoundError
        If ``image_path`` doesn't exist or lane edges can't be located.
    ValueError
        If the image is unreadable or the pipeline produces no plants.
    """
    start_time = time.time()
    logger.info("run_pipeline: starting on %s", image_path)

    edges = _resolve_lane_edges(lane_edges, lane_edges_path)
    model = _resolve_model(model, model_path)

    # 1. Preprocessing — load, detect dish, crop, pad
    pre = preprocessing.preprocess_image(image_path)
    bbox = pre["bbox"]
    pad_h = pre["pad_h"]
    pad_w = pre["pad_w"]

    # 2. Inference — patch-based prediction in padded space
    prob_map_padded = inference.predict_smooth_windowing(pre["padded_dish"], model)

    # 3. Postprocessing — clean binary mask in unpadded (cropped) space.
    # Also unpad the probability map separately so it shares a coordinate
    # space with downstream root-tip operations (used for confidence).
    binary_mask = postprocessing.postprocess_mask(prob_map_padded, pad_h, pad_w)
    prob_map_unpadded = postprocessing.unpad(prob_map_padded, pad_h, pad_w)

    # 4. Segmentation — split clean mask into 5 per-lane plant masks
    _, plant_masks, debug_info = segmentation.lane_clean_and_instances(binary_mask, edges)
    logger.info(
        "segmentation debug — chosen_labels=%s chosen_scores=%s",
        debug_info.get("chosen_labels"),
        debug_info.get("chosen_scores"),
    )

    # 5. Localization — per plant, find main root, root tip, mm conversion, confidence
    plants = []
    plate_size_px = bbox[2]
    for lane_idx, plant_mask in enumerate(plant_masks):
        plant_entry = _process_plant(
            lane_idx=lane_idx,
            plant_mask=plant_mask,
            plate_size_px=plate_size_px,
            prob_map_unpadded=prob_map_unpadded,
        )
        plants.append(plant_entry)

    elapsed = time.time() - start_time
    logger.info(
        "run_pipeline: done in %.2fs (%d/5 plants with detected roots)",
        elapsed,
        sum(1 for p in plants if p["has_root"]),
    )

    return {
        "image_path": image_path,
        "binary_mask": binary_mask,
        "plant_masks": plant_masks,
        "plants": plants,
        "processing_time_seconds": elapsed,
        "plate_bbox": bbox,
    }


def _process_plant(
    lane_idx: int,
    plant_mask: np.ndarray,
    plate_size_px: int,
    prob_map_unpadded: np.ndarray,
) -> dict:
    """Run localization on a single plant mask, return its result dict.

    Returns a complete plant entry whether or not a root was found, with
    ``has_root=False`` and zero-valued coordinates for empty lanes so the
    output structure stays uniform across all 5 plants.
    """
    if plant_mask.sum() == 0:
        logger.debug("Lane %d: empty mask, no root", lane_idx)
        return _empty_plant_entry(lane_idx)

    path = localization.find_main_root(plant_mask)
    if path is None:
        logger.debug("Lane %d: no main root found", lane_idx)
        return _empty_plant_entry(lane_idx)

    tip = localization.get_root_tip(path)
    if tip is None:
        logger.debug("Lane %d: no root tip extracted", lane_idx)
        return _empty_plant_entry(lane_idx)

    tip_y, tip_x = tip
    tip_y_mm, tip_x_mm = localization.pixel_to_mm((tip_y, tip_x), plate_size_px)
    confidence = localization.get_confidence_score(prob_map_unpadded, path)

    logger.info(
        "Lane %d: root tip px=(%d, %d) mm=(%.2f, %.2f) confidence=%.3f",
        lane_idx,
        tip_x,
        tip_y,
        tip_x_mm,
        tip_y_mm,
        confidence,
    )

    return {
        "lane": lane_idx,
        "root_tip_px": {"x": int(tip_x), "y": int(tip_y)},
        "root_tip_mm": {"x": float(tip_x_mm), "y": float(tip_y_mm)},
        "confidence": float(confidence),
        "has_root": True,
        "main_root_path": [(int(y), int(x)) for y, x in path],
    }


def _empty_plant_entry(lane_idx: int) -> dict:
    """Uniform empty result for lanes where no root was detected.

    Coordinates and confidence are explicitly None so callers can't
    confuse them with valid measurements at (0, 0). Callers should
    always check `has_root` before reading the other fields.
    """
    return {
        "lane": lane_idx,
        "root_tip_px": None,
        "root_tip_mm": None,
        "confidence": None,
        "has_root": False,
        "main_root_path": [],
    }
