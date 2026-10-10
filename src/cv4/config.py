"""
Central configuration for the cv4 plant root segmentation pipeline.

All default parameters for inference, postprocessing, segmentation, and
localization live here. No logic, just values.

These defaults were derived from the original Task_8.ipynb reference
pipeline (pipeline 245769, Attention U-Net, F1=0.88). See
docs/pipeline_selection.md for the pipeline selection rationale.

Usage:
    from cv4 import config
    print(config.PATCH_SIZE)

    # Or import specific values directly:
    from cv4.config import PATCH_SIZE, STRIDE
"""

import os
from pathlib import Path

# PATHS

# Resolved relative to the repository root (two levels up from this file:
# src/cv4/config.py -> src/cv4 -> src -> <repo root>).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Model weights directory. Defaults to <repo root>/weights for local
# development. CV4_MODEL_DIR overrides this in containers and the cloud,
# where weights live elsewhere (e.g. /app/models inside the backend image).
MODEL_DIR: Path = Path(os.environ.get("CV4_MODEL_DIR", _REPO_ROOT / "weights"))

DEFAULT_MODEL_PATH: str = str(MODEL_DIR / "unet_256px.h5")
DEFAULT_LANE_EDGES_PATH: str = str(MODEL_DIR / "lane_edges.npy")


# INFERENCE  (source: notebook cells 0, 3)

PATCH_SIZE: int = 256  # Sliding-window patch edge length (pixels)
STRIDE: int = 128  # Stride between patches; 50% overlap with PATCH_SIZE
BATCH_SIZE: int = 32  # Patches per forward pass during prediction
SPLINE_WINDOW_POWER: int = 2  # Hann window exponent for smooth patch blending


# POSTPROCESSING  (source: notebook cells 3, 7)

# Hysteresis thresholding: pixels >= HYSTERESIS_HIGH are definite roots;
# pixels in [HYSTERESIS_LOW, HYSTERESIS_HIGH) are kept only if connected to
# a definite pixel. Tuned empirically on the validation set.
HYSTERESIS_LOW: float = 0.05
HYSTERESIS_HIGH: float = 0.25

# Morphological closing to fill tiny single-pixel gaps in the binary mask.
MICRO_CLOSE_KERNEL_SIZE: int = 4  # Elliptical kernel edge (pixels)
MICRO_CLOSE_ITERATIONS: int = 1

# Bridging: reconnect nearby skeleton endpoints that belong to the same root
# but were split into separate components. Vertical bias (min_dy, max_dx)
# prevents bridging across adjacent lanes.
BRIDGE_MAX_DIST: int = 60  # Max Euclidean distance between endpoints (px)
BRIDGE_MAX_DX: int = 4  # Max horizontal delta (px), keep it vertical
BRIDGE_MIN_DY: int = 8  # Min vertical delta (px), avoid self-loops
BRIDGE_THICKNESS: int = 3  # Line thickness when drawing the bridge
BRIDGE_ENFORCE_SAME_LANE: bool = True


# SEGMENTATION  (source: notebook cell 10)

EXPECTED_ROOTS: int = 5  # 5 plants per petri dish, one per lane

# Global speck removal, very small to preserve thin root fragments.
SPECK_AREA: int = 3

# Components entirely within the top EXCLUDE_TOP_FRAC of the image are
# discarded as top-only noise (mostly from the plate label area).
EXCLUDE_TOP_FRAC: float = 0.15

# Components touching the top TOP_BAND_FRAC get a scoring bonus (roots grow
# downward from the seedling, which sits at the top of the plate).
TOP_BAND_FRAC: float = 0.30
TOP_BONUS: float = 250.0

# Permissive per-component gating before lane scoring.
MIN_AREA_CANDIDATE: int = 10
MIN_HEIGHT_PX: int = 20

# Horizontal-line artefact detection (e.g., plate shadows). Soft penalty,
# not a hard reject, touches-top components are always kept.
HORIZ_PENALTY: float = 500.0
HORIZ_ASPECT: float = 8.0  # width / height ratio threshold
HORIZ_MAX_HEIGHT_PX: int = 18
HORIZ_THETA_DEG: int = 20  # orientation angle threshold (degrees)

# Scoring weights for picking the "winner" component in each lane.
W_HEIGHT: float = 2.0  # Dominant factor, tallest component wins
W_MAJOR: float = 1.0  # Major axis length (regionprops)
W_AREA: float = 0.01  # Area has a small influence

# If no lane candidate reaches this score, the lane is marked empty.
MIN_LANE_SCORE: float = 120.0

# Within-lane fragment merging: absorb detached laterals into the main root.
MERGE_WITHIN_LANE: bool = True
MERGE_DIST_PX: int = 22  # Max distance between fragment and main (px)
MERGE_MIN_AREA: int = 8


# LOCALIZATION  (source: notebook cell 11)

# Pre-skeletonization component cleanup.
MIN_COMPONENT_SIZE: int = 15
CROP_MARGIN: int = 50

# Winner-component scoring: score = height^HEIGHT_POWER * thickness^THICKNESS_POWER.
# Components wider than tall get a 0.1x penalty (they're not main roots).
HEIGHT_POWER: float = 2.0
THICKNESS_POWER: float = 1.5

# Dijkstra edge-weight penalty for horizontal moves (keeps paths vertical).
HORIZONTAL_PENALTY: float = 3.0

# Standard-plant start-node selection.
START_BAND: int = 12  # Search top START_BAND pixels for start nodes
REQUIRE_ENDPOINT_START: bool = True  # Prefer degree-1 skeleton endpoints

# Path relaxation in the top region, allow more horizontal movement near
# the seedling where the root emerges and fans out.
TOP_RELAX_BAND: int = 35
TOP_HORIZONTAL_PENALTY: float = 1.1

# Standard-plant bottleneck detection: reject start candidates whose path
# passes through a thin "pinch point" just below the seedling (a common
# failure mode where Dijkstra routes through a single-pixel-wide bridge).
# Offsets are measured from the top of the skeleton component.
BOTTLENECK_Y_OFFSET_START: int = 15
BOTTLENECK_Y_OFFSET_END: int = 25
BOTTLENECK_THICKNESS_THRESH: float = 1.1

# --- Huge plants (area > HUGE_PLANT_THRESH pixels) ---
# Large root systems need stricter vertical bias + central-corridor start
# selection, otherwise Dijkstra picks a lateral instead of the main root.
HUGE_PLANT_THRESH: int = 35000
HUGE_HORIZONTAL_PENALTY: float = 8.0
CENTER_BIAS_STRENGTH: float = 2.0  # Pulls path toward plant's median x

# Huge-plant start selection: central-corridor strategy.
HUGE_START_SEARCH_BAND: int = 100  # Wider than STANDARD_START_BAND
HUGE_CORRIDOR_WIDTH: int = 15  # Only consider starts within corridor

# Huge-plant path quality scoring (used to rank start candidates).
HUGE_PATH_SAMPLE_DEPTH: int = 100  # First N nodes of each candidate path
HUGE_DIRECTION_CHANGE_PENALTY: float = 3.0  # Penalty per zigzag
HUGE_UPWARD_STEP_PENALTY: float = 2.0  # Penalty per upward move
HUGE_OFF_CENTER_PENALTY: float = 0.5  # Penalty per px off centerline
HUGE_SPREAD_PENALTY: float = 1.0  # Penalty for horizontal spread

# --- Small plants (area < SMALL_PLANT_THRESH pixels) ---
# Too small for start-node filtering; just pick the topmost skeleton pixel.
SMALL_PLANT_THRESH: int = 3050


# AZURE ML WORKSPACE

AZURE_SUBSCRIPTION_ID: str = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
AZURE_RESOURCE_GROUP: str = "buas-y2"
AZURE_WORKSPACE_NAME: str = "team-CV4-2026"

# AZURE ML COMPUTE
#
# The workspace is hub-based (Azure AI Foundry). Compute is provided by the
# shared lambda-0 Kubernetes cluster (namespace: cv4). Use compute="lambda-0"
# for all jobs and pick the instance type from the two available:
#   "defaultinstancetype" — 16 CPU, 32Gi RAM, no GPU
#   "gpu"                 — 16 CPU, 32Gi RAM, 1 GPU
AZURE_COMPUTE: str = "lambda-0"
AZURE_GPU_INSTANCE: str = "gpu"
AZURE_CPU_INSTANCE: str = "defaultinstancetype"


# COORDINATE CONVERSION  (source: notebook cell 14)

PLATE_SIZE_MM: float = 150.0  # Physical edge length of the petri plate


# CONFIDENCE SCORING  (new, not in original notebook)

# Number of trailing path points sampled when scoring root-tip confidence.
# 20 covers roughly the last millimetre of root at typical plate scales,
# enough to be robust to localization noise without diluting the
# tip-relevance with very upstream pixels.
CONFIDENCE_PATH_SAMPLES: int = 20
