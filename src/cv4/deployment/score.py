"""
Scoring script for the cv4 plant root segmentation Azure ML endpoint.

Azure ML Managed/Online Endpoints run this script to serve predictions.
The endpoint runtime calls two functions:

    init() : run once when the deployment starts. Loads the model and the
             lane-edges array into module-level globals so they are reused
             across requests (the model load is the slow part, ~30s).

    run(raw_data) : run once per request. Accepts a JSON body containing a
             base64-encoded image, runs the cv4 pipeline, and returns the
             per-lane root measurements as JSON.

The response shape mirrors the prediction data returned by the local
FastAPI ``POST /predict`` endpoint, so the two are interchangeable from the
backend's point of view. The endpoint is stateless: it does not write to the
database or assign a prediction_id. The backend is responsible for persisting
the prediction and generating the id when it stores the result.

Expected request body:

    {
        "image": "<base64-encoded image bytes>"
    }

Response body (HTTP 200):

    {
        "roots": [
            {"lane": 0, "has_root": true,
             "root_tip_px": {"x": 145, "y": 890}, "confidence": 0.91},
            ... 5 entries total, one per lane ...
        ],
        "plate_bbox": [x, y, side],
        "processing_time_s": 4.21
    }

``plate_bbox`` contains the [x, y, side] bounding box of the petri dish
crop within the original image, identical to the value returned by the
local pipeline. The backend uses it to populate the ``crop`` field in
``PredictResponse`` so tip coordinates can be mapped back to the original
image space.

Error response:

    {"error": "<human-readable description>"}
"""

import base64
import binascii
import json
import logging
import os
import sys
import tempfile

import numpy as np

# The endpoint ships the whole src/ tree and runs this script from
# src/cv4/deployment/. The cv4 package lives at src/cv4, so src/ must be on
# sys.path for "import cv4" to resolve. The inference server does not install
# the package or add src/ to the path automatically, so we do it here: walk
# three levels up from this file (deployment -> cv4 -> src) and prepend it.
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from cv4.inference import load_model  # noqa: E402
from cv4.pipeline import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

# Module-level globals populated by init() and reused by every run() call.
_model = None
_lane_edges = None


def _enable_gpu_memory_growth() -> None:
    """Tell TensorFlow to grow GPU memory on demand instead of reserving it all.

    By default TensorFlow grabs the entire GPU memory at startup. On the shared
    cluster (16 GPUs across all teams) that means one endpoint locks a whole
    GPU. Enabling memory growth makes TF allocate only what it actually needs,
    leaving the rest free for others. Must be set before any GPU operation.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Enabled memory growth on %s", gpu.name)
        except RuntimeError as exc:
            # Raised if GPUs were already initialised; safe to log and continue.
            logger.warning("Could not set memory growth on %s: %s", gpu.name, exc)
    if not gpus:
        logger.info("[DEVICE] Scoring on: CPU (no GPU detected)")
    else:
        logger.info("[DEVICE] Scoring on: GPU x%d (%s)", len(gpus), ", ".join(g.name for g in gpus))


def init() -> None:
    """Load the model and lane edges once when the deployment starts.

    Azure ML sets the AZUREML_MODEL_DIR environment variable to the folder
    where the registered model artifacts are mounted. The registered model
    (cv4_root_segmentation) contains a single file, model.h5, so the model
    path is AZUREML_MODEL_DIR/model.h5.

    The lane-edges array is not part of the registered model, so it is
    bundled alongside this script in the deployment code folder and loaded
    from the same directory as score.py.
    """
    global _model, _lane_edges

    # Configure GPU memory growth before loading the model (which touches the
    # GPU). Without this, TF reserves the whole GPU on the shared cluster.
    _enable_gpu_memory_growth()

    model_dir = os.environ.get("AZUREML_MODEL_DIR", "")
    model_path = os.path.join(model_dir, "model.h5")
    logger.info("Loading model from %s", model_path)
    _model = load_model(model_path)

    # lane_edges.npy ships in the same folder as this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lane_edges_path = os.path.join(script_dir, "lane_edges.npy")
    logger.info("Loading lane edges from %s", lane_edges_path)
    _lane_edges = np.load(lane_edges_path)

    logger.info("Initialisation complete; endpoint is ready.")


def run(raw_data: str) -> str:
    """Run the cv4 pipeline on one request and return predictions as JSON.

    Parameters
    ----------
    raw_data : str
        JSON string with a single key, ``image``, whose value is the
        base64-encoded bytes of a PNG/JPEG/TIFF petri-dish image.

    Returns
    -------
    str
        JSON string. On success:
        ``{"roots": [...], "plate_bbox": [x, y, side], "processing_time_s": ...}``
        matching the local /predict prediction schema (minus prediction_id,
        which the backend assigns when it persists the result).
        On failure: ``{"error": "..."}`` with a human-readable message.
    """
    tmp_path = None
    try:
        # Parse the request body and pull out the base64 image string.
        try:
            payload = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError) as exc:
            return json.dumps({"error": f"Request body is not valid JSON: {exc}"})

        image_b64 = payload.get("image")
        if not image_b64:
            return json.dumps({"error": "Request must include a base64 'image' field."})

        # Decode the base64 string back into raw image bytes.
        try:
            image_bytes = base64.b64decode(image_b64)
        except (binascii.Error, ValueError) as exc:
            return json.dumps({"error": f"'image' is not valid base64: {exc}"})

        # The cv4 preprocessing step opens the image by file path, so write
        # the decoded bytes to a temporary file the pipeline can read.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # Run the full pipeline with the pre-loaded model and lane edges.
        result = run_pipeline(tmp_path, model=_model, lane_edges=_lane_edges)

        # Map the pipeline output to the same per-lane schema the local
        # /predict endpoint returns. prediction_id is intentionally omitted;
        # the backend assigns it when it persists the prediction.
        roots = [
            {
                "lane": plant["lane"],
                "has_root": plant["has_root"],
                "root_tip_px": plant["root_tip_px"],
                "confidence": plant["confidence"],
            }
            for plant in result["plants"]
        ]

        return json.dumps(
            {
                "roots": roots,
                # plate_bbox is the [x, y, side] bounding box of the petri dish
                # crop within the original image. The backend uses this to
                # populate PredictResponse.crop so tip coordinates can be
                # converted back to original-image pixel positions.
                "plate_bbox": result["plate_bbox"],
                "processing_time_s": result["processing_time_seconds"],
            }
        )

    except FileNotFoundError as exc:
        # Missing lane edges, or a path issue inside the pipeline.
        logger.exception("File not found during inference")
        return json.dumps({"error": str(exc)})

    except ValueError as exc:
        # Unreadable image or no petri dish detected.
        logger.exception("Invalid input during inference")
        return json.dumps({"error": str(exc)})

    except MemoryError as exc:
        logger.exception("Out of memory during inference")
        return json.dumps({"error": f"Insufficient memory to run inference: {exc}"})

    except Exception as exc:
        # Catch-all so the endpoint always returns a structured error
        # instead of a bare 500 with no body.
        logger.exception("Unexpected error during inference")
        return json.dumps({"error": f"Unexpected error during inference: {exc}"})

    finally:
        # Always clean up the temp file, success or failure.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
