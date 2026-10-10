"""Lane-based instance segmentation: split a full root mask into 5 per-plant masks."""

import logging

import cv2
import numpy as np
from skimage import measure, morphology

from cv4 import config

logger = logging.getLogger(__name__)


DEFAULT_CFG = {
    "expected_roots": config.EXPECTED_ROOTS,
    "speck_area": config.SPECK_AREA,
    "exclude_top_frac": config.EXCLUDE_TOP_FRAC,
    "top_band_frac": config.TOP_BAND_FRAC,
    "top_bonus": config.TOP_BONUS,
    "min_area_candidate": config.MIN_AREA_CANDIDATE,
    "min_height_px": config.MIN_HEIGHT_PX,
    "horiz_penalty": config.HORIZ_PENALTY,
    "horiz_aspect": config.HORIZ_ASPECT,
    "horiz_max_height_px": config.HORIZ_MAX_HEIGHT_PX,
    "horiz_theta_deg": config.HORIZ_THETA_DEG,
    "w_height": config.W_HEIGHT,
    "w_major": config.W_MAJOR,
    "w_area": config.W_AREA,
    "merge_within_lane": config.MERGE_WITHIN_LANE,
    "merge_dist_px": config.MERGE_DIST_PX,
    "merge_min_area": config.MERGE_MIN_AREA,
    "min_lane_score": config.MIN_LANE_SCORE,
}


def x_crown_from_region(region):
    """Crown-x: median x at the component's topmost y (fallback: centroid x)."""
    coords = region.coords
    ymin = coords[:, 0].min()
    xs = coords[coords[:, 0] == ymin][:, 1]
    return float(np.median(xs)) if len(xs) else float(region.centroid[1])


def compute_lane_edges(masks, expected_roots=5, min_area=50):
    """Compute global lane boundaries from a list of binary masks.

    Masks may have different widths. x-coordinates are normalized to
    [0, 1] internally and scaled back to the most common mask width at
    the end, so all valid masks contribute to the lane-edge averages.

    Returns an array of shape (expected_roots + 1,): [0, e1, ..., e_{K-1}, W].
    """
    if not masks:
        raise ValueError("No masks provided")

    K = expected_roots
    slot_xs_normalized: list[list[float]] = [[] for _ in range(K)]
    widths: list[int] = []
    used = 0

    for idx, m in enumerate(masks):
        if m is None:
            continue
        _, W = m.shape

        mask01 = (m > 0).astype(np.uint8)

        labels = measure.label(mask01, connectivity=2)
        regions = measure.regionprops(labels)

        regs = [r for r in regions if r.area >= min_area]
        if len(regs) < 1:
            continue

        regs = sorted(regs, key=lambda r: r.area, reverse=True)[: max(K, 8)]

        comps = [(r.area, x_crown_from_region(r)) for r in regs]
        comps = sorted(comps, key=lambda t: t[0], reverse=True)[:K]
        if len(comps) < K:
            continue

        comps_lr = sorted(comps, key=lambda t: t[1])
        for i in range(K):
            slot_xs_normalized[i].append(float(comps_lr[i][1]) / float(W))

        widths.append(W)
        used += 1

    if used == 0:
        raise ValueError(f"No usable masks found (need >={K} components per mask)")

    target_width = max(set(widths), key=widths.count)

    normalized_centers = np.array([np.mean(slot_xs_normalized[i]) for i in range(K)], dtype=np.float32)
    lane_centers = normalized_centers * target_width

    edges = [0.0]
    for i in range(K - 1):
        edges.append(float(0.5 * (lane_centers[i] + lane_centers[i + 1])))
    edges.append(float(target_width))
    lane_edges = np.array(edges, dtype=np.float32)

    logger.info(
        "compute_lane_edges: used %d/%d masks (target width=%d). centers=%s edges=%s",
        used,
        len(masks),
        target_width,
        lane_centers.tolist(),
        lane_edges.tolist(),
    )
    return lane_edges


def lane_from_x(x, lane_edges, expected_roots=5):
    """Map an x-coordinate to a lane index using global lane edges."""
    lane = np.searchsorted(lane_edges, x, side="right") - 1
    return max(0, min(expected_roots - 1, int(lane)))


def _is_horizontal_artifact(region, horiz_aspect, horiz_max_height_px, horiz_theta_deg):
    """Heuristic for thin horizontal components (soft penalty, not a hard reject)."""
    minr, minc, maxr, maxc = region.bbox
    height = maxr - minr
    width = maxc - minc
    aspect = width / (height + 1e-6)
    theta = abs(np.degrees(region.orientation))
    return height <= horiz_max_height_px and aspect >= horiz_aspect and theta <= horiz_theta_deg


def lane_clean_and_instances(mask01, lane_edges, cfg=None):
    """Select one main component per lane, optionally merge fragments.

    Always returns `expected_roots` (default 5) lane masks. An empty lane mask
    means that plant has no detectable root — the plant still exists.
    """
    if cfg is None:
        cfg = DEFAULT_CFG

    mask01 = (mask01 > 0).astype(np.uint8)
    H, W = mask01.shape
    K = cfg["expected_roots"]

    mask01 = morphology.remove_small_objects(mask01.astype(bool), min_size=cfg["speck_area"]).astype(np.uint8)

    labels = measure.label(mask01, connectivity=2)
    regions = measure.regionprops(labels)

    top_band = int(cfg["top_band_frac"] * H)
    y_cut = int(cfg["exclude_top_frac"] * H)

    lane_cands = [[] for _ in range(K)]
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        height = maxr - minr

        if maxr <= y_cut:
            continue
        if r.area < cfg["min_area_candidate"] or height < cfg["min_height_px"]:
            continue

        touches_top = minr <= top_band
        x_crown = x_crown_from_region(r)
        lane = lane_from_x(x_crown, lane_edges, K)

        score = cfg["w_height"] * float(height) + cfg["w_major"] * float(r.major_axis_length) + cfg["w_area"] * float(r.area)

        if touches_top:
            score += cfg["top_bonus"]

        if (not touches_top) and _is_horizontal_artifact(
            r,
            cfg["horiz_aspect"],
            cfg["horiz_max_height_px"],
            cfg["horiz_theta_deg"],
        ):
            score -= cfg["horiz_penalty"]

        lane_cands[lane].append((score, r.label))

    chosen_labels = [None] * K
    chosen_scores = [None] * K

    for lane in range(K):
        if not lane_cands[lane]:
            continue
        lane_cands[lane].sort(key=lambda x: x[0], reverse=True)
        best_score, best_label = lane_cands[lane][0]
        if best_score >= cfg["min_lane_score"]:
            chosen_labels[lane] = best_label
            chosen_scores[lane] = best_score

    lane_masks = []
    for lane in range(K):
        lbl = chosen_labels[lane]
        if lbl is None:
            lane_masks.append(np.zeros_like(mask01, dtype=np.uint8))
        else:
            lane_masks.append((labels == lbl).astype(np.uint8))

    if cfg["merge_within_lane"]:
        region_lane = {r.label: lane_from_x(x_crown_from_region(r), lane_edges, K) for r in regions}

        for lane in range(K):
            main = lane_masks[lane]
            if main.sum() == 0:
                continue

            inv = ((main == 0) * 255).astype(np.uint8)
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

            for r in regions:
                if r.label == chosen_labels[lane]:
                    continue
                if r.area < cfg["merge_min_area"]:
                    continue
                if region_lane.get(r.label, -1) != lane:
                    continue

                comp = labels == r.label
                ys, xs = np.where(comp)
                if len(xs) == 0:
                    continue

                if float(dist[ys, xs].min()) <= cfg["merge_dist_px"]:
                    main[comp] = 1

            lane_masks[lane] = main.astype(np.uint8)

    n_with_root = sum(1 for lbl in chosen_labels if lbl is not None)
    logger.info("Detected roots for %d/%d plants", n_with_root, K)
    for lane in range(K):
        if chosen_labels[lane] is None:
            logger.warning("Plant %d: no root detected", lane)

    if n_with_root == 0:
        logger.warning("No plants detected, all lanes are empty")

    clean_union = np.zeros_like(mask01, dtype=np.uint8)
    for m in lane_masks:
        clean_union |= m

    dbg = {
        "top_band_px": top_band,
        "y_cut_px": y_cut,
        "chosen_labels": chosen_labels,
        "chosen_scores": chosen_scores,
    }
    return clean_union, lane_masks, dbg


def crop_to_root(mask01, margin=25):
    """Tight crop around nonzero pixels. Returns None if mask is empty."""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mask01.shape[0] - 1, y1 + margin)
    x1 = min(mask01.shape[1] - 1, x1 + margin)
    return mask01[y0 : y1 + 1, x0 : x1 + 1]
