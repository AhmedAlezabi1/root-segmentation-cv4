"""
Generate lane_edges.npy from a directory of petri dish images.

This is a one-time setup script. The pipeline needs a precomputed
lane_edges.npy file that defines where the 5 plant lanes sit on a
petri plate. This script takes a directory of original petri dish
images, runs preprocessing -> inference -> postprocessing on each,
collects the resulting masks in memory, calls
segmentation.compute_lane_edges, and saves the result to
weights/lane_edges.npy.

Run once before using the pipeline. Re-run if the camera setup,
plate type, or model changes.

Usage:
    uv run python scripts/generate_lane_edges.py
    uv run python scripts/generate_lane_edges.py --image-dir data/my_images
    uv run python scripts/generate_lane_edges.py --output weights/custom.npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from cv4 import config, inference, postprocessing, preprocessing, segmentation

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_IMAGE_DIR = _REPO_ROOT / "data" / "lane_edges_inputs"
_DEFAULT_OUTPUT = Path(config.DEFAULT_LANE_EDGES_PATH)


def _setup_logging(verbose: bool) -> None:
    """Configure root logging so script output is human-readable."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s] %(message)s",
    )


def _collect_image_paths(image_dir: Path) -> list[Path]:
    """Return sorted PNG/JPG/TIF paths in image_dir."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(image_dir.glob(ext))
    return sorted(paths)


def _produce_mask(image_path: Path, model) -> np.ndarray | None:
    """Run preprocessing + inference + postprocessing on a single image.

    Returns the unpadded binary mask, or None if anything fails so the
    caller can skip the image and continue with the rest.
    """
    log = logging.getLogger(__name__)
    try:
        pre = preprocessing.preprocess_image(str(image_path))
        prob_map = inference.predict_smooth_windowing(pre["padded_dish"], model)
        mask = postprocessing.postprocess_mask(prob_map, pre["pad_h"], pre["pad_w"])
        return mask
    except Exception as exc:
        log.warning("Skipping %s: %s", image_path.name, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate lane_edges.npy from a directory of petri dish images.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_DEFAULT_IMAGE_DIR,
        help=f"Directory of petri dish images. Default: {_DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Path to write lane_edges.npy. Default: {_DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model weights. Default: config.DEFAULT_MODEL_PATH",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    if not args.image_dir.exists() or not args.image_dir.is_dir():
        log.error("Image directory not found: %s", args.image_dir)
        return 1

    image_paths = _collect_image_paths(args.image_dir)
    if not image_paths:
        log.error("No images found in %s", args.image_dir)
        return 1

    log.info("Found %d image(s) in %s", len(image_paths), args.image_dir)

    log.info("Loading model from %s", args.model or "(config default)")
    model = inference.load_model(str(args.model) if args.model else None)

    masks: list[np.ndarray] = []
    for i, path in enumerate(image_paths, start=1):
        log.info("[%d/%d] Processing %s", i, len(image_paths), path.name)
        mask = _produce_mask(path, model)
        if mask is not None:
            masks.append(mask)

    if not masks:
        log.error("No usable masks were produced — cannot compute lane edges")
        return 1

    log.info("Computing lane edges from %d mask(s)", len(masks))
    lane_edges = segmentation.compute_lane_edges(masks, expected_roots=config.EXPECTED_ROOTS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, lane_edges)
    log.info("Saved lane edges to %s", args.output)
    log.info("Edge values: %s", np.round(lane_edges, 2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
