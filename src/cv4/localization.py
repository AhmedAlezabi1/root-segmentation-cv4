"""
Localization module for the cv4 plant root segmentation pipeline.

Finds the main root and its tip for each plant using skeletonization
and Dijkstra-based graph pathfinding. Source: reference notebook cells
11 and 14.

Module functions are grouped as follows:
    Skeleton analysis:   analyze_components
    Pathfinding:         score_start_candidate, get_path_from_component
    Convenience wrapper: find_main_root
    Coordinate/output:   get_root_tip, pixel_to_mm, get_confidence_score
"""

from __future__ import annotations

import logging

import cv2
import networkx as nx
import numpy as np
from skimage import measure, morphology

from cv4 import config

logger = logging.getLogger(__name__)


# Skeleton analysis


def analyze_components(
    mask01: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Find the best skeleton component in a plant mask.

    Skeletonizes the binary mask, labels connected skeleton components,
    and picks the component with the highest combined score of height
    and average thickness. Components that are wider than tall are
    penalized since they are not main roots.

    Parameters
    ----------
    mask01 : np.ndarray
        Binary plant mask (dtype uint8 or bool, values 0 or 1). Shape
        should match the cropped petri dish.

    Returns
    -------
    winner_mask : np.ndarray or None
        Binary mask (uint8) of the chosen skeleton component. ``None``
        if the mask has no viable skeleton (too few pixels, no
        components).
    distance_transform : np.ndarray or None
        L2 distance transform of the cleaned mask, used downstream for
        thickness-aware edge weighting in ``get_path_from_component``.
        ``None`` if no winner was found.

    Notes
    -----
    Returns ``(None, None)`` rather than raising when the mask is too
    sparse. Empty or tiny plants are valid inputs (for example,
    non-germinated seedlings), not errors.
    """
    if mask01 is None or mask01.size == 0:
        logger.warning("analyze_components: received empty or None mask")
        return None, None

    # To-do: scikit-image 2.0 will replace `min_size` with `max_size` (with
    # inverted semantics: removes objects <= threshold, not < threshold).
    # Migrate when we upgrade scikit-image past 1.x.

    # Clean tiny noise before skeletonization.
    clean = morphology.remove_small_objects(mask01.astype(bool), min_size=config.MIN_COMPONENT_SIZE).astype(np.uint8)

    # Distance transform gives the thickness (in px) at every pixel.
    # Used later to weight graph edges so the path prefers thick regions.
    dt = cv2.distanceTransform(clean, cv2.DIST_L2, 3)

    # Skeletonize to a 1-pixel-wide representation of the root system.
    skel = morphology.skeletonize(clean).astype(np.uint8)
    if np.sum(skel) < 5:
        logger.warning("analyze_components: skeleton too small (%d pixels)", int(np.sum(skel)))
        return None, None

    # Label each connected skeleton component, pick the best one.
    labels = measure.label(skel, connectivity=2)
    best_label = -1
    best_score = -float("inf")

    for label_id in range(1, labels.max() + 1):
        ys, xs = np.where(labels == label_id)

        height = int(ys.max() - ys.min())
        width = int(xs.max() - xs.min())
        avg_thickness = float(np.mean(dt[ys, xs]))

        # Tall, thick components score highest (height^a * thickness^b).
        score = (height**config.HEIGHT_POWER) * (avg_thickness**config.THICKNESS_POWER)
        # Wider-than-tall components are almost certainly not main roots
        # (they're horizontal artefacts, lateral fragments, etc.).
        if width > height:
            score *= 0.1

        if score > best_score:
            best_score = score
            best_label = label_id

    if best_label == -1:
        logger.warning("analyze_components: no viable component found")
        return None, None

    winner_mask = (labels == best_label).astype(np.uint8)
    logger.debug(
        "analyze_components: picked component %d with score %.2f",
        best_label,
        best_score,
    )
    return winner_mask, dt


# Pathfinding


def score_start_candidate(
    graph: nx.Graph,
    start_node: int,
    end_node: int,
    dt: np.ndarray,
    plant_center_x: float,
) -> float:
    """
    Score a candidate start node for huge plants by path quality.

    Runs Dijkstra from the candidate to the end node, samples the first
    N nodes of the resulting path, and penalizes the path for direction
    changes, upward movement, off-centeredness, and horizontal spread.
    Lower score means a straighter, more vertical, more central path
    and so a better candidate.

    Only used for huge plants (area > ``config.HUGE_PLANT_THRESH``),
    where a naive "topmost node" heuristic often picks a lateral root
    instead of the main root.

    Parameters
    ----------
    graph : networkx.Graph
        Weighted skeleton graph built in ``get_path_from_component``.
    start_node : int
        Candidate start node index.
    end_node : int
        Fixed end node (the bottom of the skeleton).
    dt : np.ndarray
        Distance transform (unused here but kept in the signature for
        future extension where a thickness-based scoring term may be
        added).
    plant_center_x : float
        Median x-coordinate of the plant, used as the "central axis"
        for off-center penalty.

    Returns
    -------
    float
        Path quality score. ``float('inf')`` if no path exists or the
        sampled path is too short to score reliably.
    """
    try:
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="weight")
    except nx.NetworkXNoPath:
        return float("inf")

    sample = path[: min(config.HUGE_PATH_SAMPLE_DEPTH, len(path))]
    if len(sample) < 10:
        return float("inf")

    xs = [graph.nodes[n]["x"] for n in sample]
    ys = [graph.nodes[n]["y"] for n in sample]

    # Penalty 1, zigzags: count times the x-direction reverses.
    direction_changes = 0
    for i in range(2, len(xs)):
        dx1 = xs[i - 1] - xs[i - 2]
        dx2 = xs[i] - xs[i - 1]
        if (dx1 > 0 and dx2 < 0) or (dx1 < 0 and dx2 > 0):
            direction_changes += 1

    # Penalty 2, upward moves: roots grow down, so going up is wrong.
    upward_steps = sum(1 for i in range(1, len(ys)) if ys[i] < ys[i - 1])

    # Penalty 3: average distance from the plant's central axis.
    avg_off_center = float(np.mean([abs(x - plant_center_x) for x in xs]))

    # Penalty 4, horizontal wander: how far left-right the path drifts.
    x_spread = max(xs) - min(xs)

    return (
        direction_changes * config.HUGE_DIRECTION_CHANGE_PENALTY
        + upward_steps * config.HUGE_UPWARD_STEP_PENALTY
        + avg_off_center * config.HUGE_OFF_CENTER_PENALTY
        + x_spread * config.HUGE_SPREAD_PENALTY
    )


def get_path_from_component(
    winner_mask: np.ndarray,
    dt: np.ndarray,
    plant_area: int,
) -> list[tuple[int, int]] | None:
    """
    Build a weighted graph from the skeleton and find the main root path.

    Constructs a graph where each skeleton pixel is a node and adjacent
    pixels are connected with edge weights that reflect:

    - Euclidean distance (diagonal moves cost more)
    - Horizontal penalty (roots are vertical, so horizontal moves cost
      more, relaxed near the top where the root emerges)
    - Inverse thickness (thicker regions are preferred since the main
      root is thicker than laterals)
    - Centrality bias (huge plants only, the main root is closer to
      the plant's central axis)

    Then selects a start node (top of the skeleton) and end node
    (bottom), and runs Dijkstra.

    Three strategies based on plant size:

    - Small plants (area < ``config.SMALL_PLANT_THRESH``) just pick
      the topmost skeleton pixel as start. Skeletons are too sparse for
      smarter heuristics.
    - Standard plants pick the start from the top band, preferring
      endpoints (degree-1 nodes) and thicker regions. Then run a
      "bottleneck check": if the resulting path passes through a thin
      pinch point just below the top, try the next-best candidate.
    - Huge plants (area > ``config.HUGE_PLANT_THRESH``) restrict
      candidates to a narrow central corridor at the top, then score
      each candidate by running Dijkstra and measuring path quality
      (see ``score_start_candidate``). Pick the candidate with the
      best score.

    Parameters
    ----------
    winner_mask : np.ndarray
        Binary skeleton mask from ``analyze_components``.
    dt : np.ndarray
        Distance transform from ``analyze_components``, used for
        thickness-aware edge weights.
    plant_area : int
        Total pixel count of the plant (used to branch between
        strategies).

    Returns
    -------
    list of (y, x) tuples, or None
        The main root path, from top to bottom, as (y, x) pixel
        coordinates. ``None`` if no valid path exists (for example,
        disconnected start/end, or Dijkstra fails).
    """
    ys, xs = np.where(winner_mask > 0)
    if len(ys) == 0:
        logger.warning("get_path_from_component: empty winner mask")
        return None

    nodes = list(zip(ys, xs))
    node_to_idx = {pt: i for i, pt in enumerate(nodes)}
    min_comp_y = int(np.min(ys))

    is_huge = plant_area > config.HUGE_PLANT_THRESH
    plant_center_x = float(np.median(xs))

    # Graph construction: one node per skeleton pixel, edges to 4 neighbors
    # (up, upper-right, right, lower-right). The other directions are
    # covered by their own iteration.
    graph = nx.Graph()
    for i, (y, x) in enumerate(nodes):
        graph.add_node(i, y=int(y), x=int(x))
        t1 = dt[y, x]
        dist_from_center_i = abs(x - plant_center_x) / 50.0 if is_huge else 0.0

        for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1)]:
            ny, nx_coord = y + dy, x + dx
            if (ny, nx_coord) not in node_to_idx:
                continue

            j = node_to_idx[(ny, nx_coord)]
            t2 = dt[ny, nx_coord]
            dist = np.sqrt(dy**2 + dx**2)

            # Horizontal penalty: penalize moves that are more
            # horizontal than vertical. Relaxed near the top where the
            # root emerges and naturally fans out a bit.
            dir_penalty = 1.0
            if abs(dx) > abs(dy):
                dir_penalty = config.HUGE_HORIZONTAL_PENALTY if is_huge else config.HORIZONTAL_PENALTY
                if y <= min_comp_y + config.TOP_RELAX_BAND:
                    dir_penalty = config.TOP_HORIZONTAL_PENALTY

            # Thickness factor: thicker regions have lower edge weight,
            # so Dijkstra prefers them. +0.1 avoids divide-by-zero at
            # edge-of-mask pixels where dt == 0.
            avg_thick = (t1 + t2) / 2.0
            thick_factor = 1.0 / (avg_thick + 0.1)

            # Centrality bias (huge plants only): paths away from the
            # central axis cost more. The main root sits near the
            # center; laterals branch sideways.
            center_cost = 1.0
            if is_huge:
                dist_from_center_j = abs(nx_coord - plant_center_x) / 50.0
                avg_off_center = (dist_from_center_i + dist_from_center_j) / 2.0
                center_cost = 1.0 + (config.CENTER_BIAS_STRENGTH * avg_off_center)

            weight = dist * dir_penalty * thick_factor * center_cost
            graph.add_edge(i, j, weight=weight)

    # Start node selection: three strategies based on plant size.
    max_y = max(graph.nodes[n]["y"] for n in graph.nodes)
    end_candidates = [n for n in graph.nodes if graph.nodes[n]["y"] == max_y]
    end_node = end_candidates[0]

    if plant_area < config.SMALL_PLANT_THRESH:
        # Small plants: grab the topmost node. Skeletons are too sparse
        # for smarter heuristics.
        start_node = min(graph.nodes, key=lambda n: graph.nodes[n]["y"])
        logger.debug("get_path_from_component: small-plant strategy (area=%d)", plant_area)

    elif is_huge:
        # Huge plants: central corridor plus path-quality scoring.
        degrees = dict(graph.degree())
        candidates = [n for n in graph.nodes if degrees[n] == 1 and graph.nodes[n]["y"] <= min_comp_y + config.HUGE_START_SEARCH_BAND]
        if not candidates:
            # No endpoint candidates available, fall back to any top-band node.
            candidates = [n for n in graph.nodes if graph.nodes[n]["y"] <= min_comp_y + config.HUGE_START_SEARCH_BAND]

        if candidates:
            # Prefer candidates inside the central corridor.
            central = [n for n in candidates if abs(graph.nodes[n]["x"] - plant_center_x) <= config.HUGE_CORRIDOR_WIDTH]
            pool = central if central else candidates

            scored = [(score_start_candidate(graph, n, end_node, dt, plant_center_x), n) for n in pool]
            scored.sort(key=lambda s: s[0])
            start_node = scored[0][1]
            logger.debug(
                "get_path_from_component: huge-plant strategy (area=%d, %d candidates, best score=%.2f)",
                plant_area,
                len(pool),
                scored[0][0],
            )
        else:
            start_node = min(graph.nodes, key=lambda n: graph.nodes[n]["y"])
            logger.debug("get_path_from_component: huge-plant fallback (no candidates found)")

    else:
        # Standard plants: thickness-ranked top band plus bottleneck check.
        degrees = dict(graph.degree())
        start_pool = [
            (n, dt[graph.nodes[n]["y"], graph.nodes[n]["x"]], degrees[n])
            for n in graph.nodes
            if graph.nodes[n]["y"] <= min_comp_y + config.START_BAND
        ]

        if not start_pool:
            start_pool = [(n, 0, 0) for n in graph.nodes if graph.nodes[n]["y"] == min_comp_y]

        if config.REQUIRE_ENDPOINT_START:
            endpoints = [z for z in start_pool if z[2] == 1]
            if endpoints:
                start_pool = endpoints

        # Sort by thickness (desc), then degree (desc).
        start_pool.sort(key=lambda z: (z[1], z[2]), reverse=True)

        # Bottleneck check: try up to the top 3 candidates and pick the
        # first one whose path doesn't pass through a thin pinch point
        # just below the top. The "pinch point" is a Dijkstra artifact
        # where the path squeezes through a 1-pixel-wide bridge between
        # otherwise-disconnected components.
        start_node = start_pool[0][0]
        for cand in start_pool[:3]:
            try:
                temp_path = nx.shortest_path(graph, source=cand[0], target=end_node, weight="weight")
                has_bridge = False
                for node_idx in temp_path:
                    y_curr = graph.nodes[node_idx]["y"]
                    if min_comp_y + config.BOTTLENECK_Y_OFFSET_START <= y_curr <= min_comp_y + config.BOTTLENECK_Y_OFFSET_END:
                        if dt[y_curr, graph.nodes[node_idx]["x"]] <= config.BOTTLENECK_THICKNESS_THRESH:
                            has_bridge = True
                            break
                if not has_bridge:
                    start_node = cand[0]
                    break
            except nx.NetworkXNoPath:
                continue

        logger.debug("get_path_from_component: standard-plant strategy (area=%d)", plant_area)

    # Run final Dijkstra from chosen start to chosen end.
    try:
        path_indices = nx.shortest_path(graph, source=start_node, target=end_node, weight="weight")
    except nx.NetworkXNoPath:
        logger.warning("get_path_from_component: no path from start to end")
        return None

    return [(int(graph.nodes[i]["y"]), int(graph.nodes[i]["x"])) for i in path_indices]


# Convenience wrapper


def find_main_root(mask01: np.ndarray) -> list[tuple[int, int]] | None:
    """
    Find the main root path for a single plant mask.

    Thin wrapper that chains ``analyze_components`` and
    ``get_path_from_component``. Use this as the main entry point when
    you have a plant mask and want the root path.

    Parameters
    ----------
    mask01 : np.ndarray
        Binary plant mask (dtype uint8 or bool, values 0 or 1).

    Returns
    -------
    list of (y, x) tuples, or None
        The main root path from top to bottom. ``None`` if the mask is
        empty, the skeleton is too small, or no valid path was found.
    """
    plant_area = int(np.sum(mask01 > 0))
    if plant_area == 0:
        logger.info("find_main_root: empty mask, no root to find")
        return None

    winner_mask, dt = analyze_components(mask01)
    if winner_mask is None:
        return None
    return get_path_from_component(winner_mask, dt, plant_area)


# Coordinate and confidence extraction


def get_root_tip(path: list[tuple[int, int]] | None) -> tuple[int, int] | None:
    """
    Extract the root tip (lowest point) from a main root path.

    By convention, the path returned by ``find_main_root`` is ordered
    from top (seedling end) to bottom (root tip), so the last point is
    the tip.

    Parameters
    ----------
    path : list of (y, x) tuples, or None
        Main root path. Typically the output of ``find_main_root``.

    Returns
    -------
    (y, x) tuple, or None
        Pixel coordinates of the root tip. ``None`` if the path is
        ``None`` or empty.
    """
    if path is None or len(path) == 0:
        logger.warning("get_root_tip: no path provided")
        return None
    tip_y, tip_x = path[-1]
    return int(tip_y), int(tip_x)


def pixel_to_mm(
    pixel_coords: tuple[int, int],
    plate_size_px: int,
    plate_size_mm: float = config.PLATE_SIZE_MM,
) -> tuple[float, float]:
    """
    Convert pixel coordinates to millimeters relative to the plate.

    The scale factor is derived from the plate's physical size (default
    150mm) divided by its pixel size (``bbox[2]`` from preprocessing).

    Parameters
    ----------
    pixel_coords : (y, x) tuple
        Pixel coordinates in the cropped plate image space.
    plate_size_px : int
        Side length of the plate in pixels (``bbox[2]`` from
        ``preprocessing.preprocess_image``).
    plate_size_mm : float, default from config
        Physical side length of the plate in millimeters.

    Returns
    -------
    (y_mm, x_mm) tuple
        Coordinates in millimeters relative to the plate's top-left corner.

    Raises
    ------
    ValueError
        If ``plate_size_px`` is not positive.
    """
    if plate_size_px <= 0:
        raise ValueError(f"plate_size_px must be positive, got {plate_size_px}")

    scale = plate_size_mm / plate_size_px
    y_px, x_px = pixel_coords
    return float(y_px * scale), float(x_px * scale)


def get_confidence_score(
    prob_map_unpadded: np.ndarray,
    main_root_path: list[tuple[int, int]],
    n_samples: int = config.CONFIDENCE_PATH_SAMPLES,
) -> float:
    """
    Score model confidence by averaging probabilities along the root path.

    Sampling on the actual root produces an honest reading of how
    confident the model was where it mattered. A square window centered
    on the tip is biased low because the tip sits at the edge of the
    root, so the window mostly covers below-tip background where
    probabilities are near zero.

    Parameters
    ----------
    prob_map_unpadded : np.ndarray
        Probability map (dtype float32, values 0.0-1.0) in the same
        coordinate space as the root path (cropped plate, no padding).
    main_root_path : list of (y, x) tuples
        Root path from top of skeleton to root tip, as returned by
        ``find_main_root``. The last element is the tip.
    n_samples : int, default from config
        How many of the path's last points to sample. Covers the segment
        of root closest to the tip, which is the meaningful region for
        confidence scoring.

    Returns
    -------
    float
        Mean probability across the sampled path points, clipped to
        [0.0, 1.0]. Returns 0.0 if the path is empty or every sampled
        point falls outside the probability map.

    Raises
    ------
    ValueError
        If ``prob_map_unpadded`` is not 2D or ``n_samples`` is not
        positive.
    """
    if prob_map_unpadded.ndim != 2:
        raise ValueError(f"prob_map_unpadded must be 2D, got shape {prob_map_unpadded.shape}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if len(main_root_path) == 0:
        logger.warning("get_confidence_score: empty root path, returning 0.0")
        return 0.0

    h, w = prob_map_unpadded.shape

    # Take the last n_samples points of the path. Path is ordered top
    # to bottom, so the tail is the segment closest to the tip.
    sample_points = main_root_path[-n_samples:]

    probs = []
    for y, x in sample_points:
        if 0 <= y < h and 0 <= x < w:
            probs.append(prob_map_unpadded[y, x])

    if not probs:
        logger.warning(
            "get_confidence_score: no path points fell inside prob map of shape %s",
            prob_map_unpadded.shape,
        )
        return 0.0

    return float(np.clip(np.mean(probs), 0.0, 1.0))
