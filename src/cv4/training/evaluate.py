"""Evaluation entry point for the cv4 segmentation model.

Loads a trained .h5 model, runs it against the validation set, computes
segmentation quality metrics, and writes a metrics.json plus a small set of
sample prediction overlays. Designed to be invoked either directly from the
command line for local development or by an Azure ML pipeline component,
which then feeds the metrics.json into the conditional registration step.

Patch-level metrics (F1, IoU, precision, recall) are the primary signal used
for the registration decision. Full-image overlays are saved as MLflow
artifacts so reviewers can visually confirm the model behaves sensibly on
real petri dishes, not just on random patches.
"""

import argparse
import json
import logging
import os
from glob import glob

import cv2
import mlflow
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

from cv4.inference import predict_smooth_windowing
from cv4.preprocessing import pad_to_patch_size
from cv4.training.config import PATCH_SIZE
from cv4.training.losses import focal_tversky_loss, hybrid_loss
from cv4.training.metrics import f1_metric
from cv4.training.mlflow_setup import configure_mlflow, in_azure_ml_job

logger = logging.getLogger(__name__)

BINARY_THRESHOLD = 0.5

# Number of full validation images to render as overlay artifacts. Eight is
# enough for a reviewer to spot systematic failure modes (missed root tips,
# false positives on petri dish edges) without bloating the MLflow run.
NUM_SAMPLE_OVERLAYS = 8


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Argparse rather than Typer for the same reason as train.py: Azure ML
    components launch via a shell command.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained cv4 segmentation model.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained .h5 model.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=("Root directory containing val_images/val/ and val_masks/val/ patches for patch-level metric computation."),
    )
    parser.add_argument(
        "--full-images-path",
        required=False,
        default=None,
        help=(
            "Optional path to a folder containing full-resolution validation "
            "images and masks (Ahmed's raw val asset layout: val_images/val/*.png "
            "and val_masks/val/*.tif). When provided, sample overlays are "
            "rendered from these. Omit to skip overlay generation."
        ),
    )
    parser.add_argument(
        "--metrics-output-path",
        required=True,
        help="Destination JSON file for the computed evaluation metrics.",
    )
    parser.add_argument(
        "--overlays-output-dir",
        required=True,
        help="Destination directory for sample overlay PNGs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for patch-level prediction. Larger is faster but uses more memory.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name. Falls back to env var or package default.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=("Optional MLflow run name for this evaluation. Ignored inside Azure ML jobs, where the run name is bound to the job name."),
    )
    parser.add_argument(
        "--training-run-id",
        type=str,
        default=None,
        help="Optional MLflow run ID of the training run that produced this model, for traceability.",
    )
    return parser.parse_args()


def load_trained_model(model_path: str):
    """Load a .h5 model saved by train.py.

    The model was compiled with a custom loss and metric, so Keras needs
    custom_objects to reconstruct the graph. We pass compile=False because
    evaluation only needs the forward pass, not the optimizer state.

    Args:
        model_path: Path to the .h5 file.

    Returns:
        A loaded Keras model ready for prediction.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    custom_objects = {
        "hybrid_loss": hybrid_loss,
        "focal_tversky_loss": focal_tversky_loss,
        "f1_metric": f1_metric,
    }
    logger.info("Loading model from %s", model_path)
    return load_model(model_path, custom_objects=custom_objects, compile=False)


def load_val_patches(data_path: str, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load all validation patches and masks into memory.

    The patches are small (256x256, single channel) and the val set is small
    too (about 1200 patches for the subset), so loading everything into
    memory at once is simpler than streaming and well within budget.

    Args:
        data_path: Root data directory.
        patch_size: Expected square edge of each patch, used for validation.

    Returns:
        Tuple of (images, masks), each shape (N, patch_size, patch_size, 1)
        with images normalized to [0, 1] and masks binarized to {0, 1}.
    """
    img_dir = os.path.join(data_path, "val_images", "val")
    mask_dir = os.path.join(data_path, "val_masks", "val")

    img_files = sorted(glob(os.path.join(img_dir, "*.png")))
    if not img_files:
        raise FileNotFoundError(f"No validation patches found in {img_dir}")

    logger.info("Loading %d validation patches from %s", len(img_files), img_dir)

    images = np.empty((len(img_files), patch_size, patch_size, 1), dtype=np.float32)
    masks = np.empty((len(img_files), patch_size, patch_size, 1), dtype=np.float32)

    for i, img_path in enumerate(img_files):
        patch_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, patch_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for patch {patch_name}: {mask_path}")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        images[i, :, :, 0] = img.astype(np.float32) / 255.0
        # Binarize at 0.5 because the saved masks are 0/255 uint8.
        masks[i, :, :, 0] = (msk.astype(np.float32) / 255.0 > BINARY_THRESHOLD).astype(np.float32)

    return images, masks


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """Compute segmentation metrics on flattened binary predictions.

    All metrics are computed at the pixel level over the entire validation
    set treated as one large array. This is more stable than averaging
    per-image metrics on small datasets, where a single empty mask can skew
    the average.

    Args:
        y_true: Ground truth array with values in {0, 1}.
        y_pred_prob: Predicted probabilities in [0, 1], same shape as y_true.

    Returns:
        Dict with keys f1, iou, precision, recall, plus the threshold used.
    """
    y_pred = (y_pred_prob > BINARY_THRESHOLD).astype(np.float32)

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    true_positives = float(np.sum(y_true_flat * y_pred_flat))
    false_positives = float(np.sum((1 - y_true_flat) * y_pred_flat))
    false_negatives = float(np.sum(y_true_flat * (1 - y_pred_flat)))

    eps = 1e-7
    precision = true_positives / (true_positives + false_positives + eps)
    recall = true_positives / (true_positives + false_negatives + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = true_positives / (true_positives + false_positives + false_negatives + eps)

    return {
        "f1": f1,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "threshold": BINARY_THRESHOLD,
        "total_pixels": int(y_true_flat.size),
        "positive_pixels_true": int(y_true_flat.sum()),
        "positive_pixels_pred": int(y_pred_flat.sum()),
    }


def render_overlay(image: np.ndarray, mask_true: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    """Render a three-panel side-by-side comparison for one image.

    Panels: original grayscale image, ground truth mask overlay (green),
    predicted mask overlay (red). The two overlays are on copies of the
    same image so spatial alignment is obvious to the eye.

    Args:
        image: 2-D uint8 grayscale image.
        mask_true: 2-D binary ground-truth mask matching image shape.
        mask_pred: 2-D binary predicted mask matching image shape.

    Returns:
        3-D uint8 BGR image of the assembled three panels, suitable for
        cv2.imwrite.
    """
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    true_overlay = base.copy()
    true_overlay[mask_true == 1] = (0, 255, 0)

    pred_overlay = base.copy()
    pred_overlay[mask_pred == 1] = (0, 0, 255)

    return np.hstack([base, true_overlay, pred_overlay])


def generate_sample_overlays(
    model,
    full_images_path: str,
    overlays_dir: str,
    num_samples: int,
    patch_size: int,
) -> list[str]:
    """Generate full-image overlay PNGs for visual inspection.

    Picks the first num_samples full validation images sorted by filename
    (deterministic across runs), pads each to a multiple of patch_size, runs
    the full inference patch-and-stitch via predict_smooth_windowing, then
    saves a three-panel overlay to overlays_dir.

    full_images_path is expected to match the raw val asset layout
    registered by Ahmed: val_images/val/*.png and val_masks/val/*.tif. This
    is the same structure the preprocess component reads, so the pipeline
    can pass the raw val asset to both preprocess and evaluate.

    Args:
        model: Loaded Keras model.
        full_images_path: Root path to a folder containing val_images/val/
            and val_masks/val/ subdirectories with full-resolution files.
        overlays_dir: Destination directory; created if missing.
        num_samples: Number of overlays to generate.
        patch_size: Patch size for padding alignment.

    Returns:
        List of paths to the saved overlay PNG files.
    """
    img_dir = os.path.join(full_images_path, "val_images", "val")
    mask_dir = os.path.join(full_images_path, "val_masks", "val")

    full_imgs = sorted(glob(os.path.join(img_dir, "*.png")))
    if not full_imgs:
        logger.warning("No full validation images found in %s; skipping overlays.", img_dir)
        return []

    os.makedirs(overlays_dir, exist_ok=True)
    saved_paths: list[str] = []

    for img_path in full_imgs[:num_samples]:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        # Validation masks are .tif at the full-image level.
        mask_path = os.path.join(mask_dir, f"{stem}_root_mask.tif")
        if not os.path.exists(mask_path):
            logger.warning("No full-image mask for %s; skipping.", stem)
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            logger.warning("Failed to read %s or %s; skipping.", img_path, mask_path)
            continue

        mx = float(msk.max())
        mask_bin_full = (msk.astype(np.float32) / mx > BINARY_THRESHOLD).astype(np.uint8) if mx > 0 else (msk * 0).astype(np.uint8)

        padded, (top, bottom), (left, right) = pad_to_patch_size(img, patch_size=patch_size)
        prob_map = predict_smooth_windowing(padded, model)

        h, w = img.shape
        prob_unpadded = prob_map[top : top + h, left : left + w]
        pred_bin = (prob_unpadded > BINARY_THRESHOLD).astype(np.uint8)

        overlay = render_overlay(img, mask_bin_full, pred_bin)
        out_path = os.path.join(overlays_dir, f"{stem}_overlay.png")
        cv2.imwrite(out_path, overlay)
        saved_paths.append(out_path)
        logger.info("Saved overlay for %s to %s", stem, out_path)

    return saved_paths


def _run_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation and log to the currently active MLflow run.

    Caller is responsible for opening the MLflow run before invoking this
    and closing it afterward. Locally that means wrapping with
    mlflow.start_run(); inside Azure ML the platform already opened the run.
    """
    model = load_trained_model(args.model_path)

    logger.info("Loading validation patches")
    val_images, val_masks = load_val_patches(args.data_path, args.patch_size)

    logger.info("Running prediction on %d patches", len(val_images))
    val_pred_prob = model.predict(val_images, batch_size=args.batch_size, verbose=1)

    logger.info("Computing patch-level metrics")
    metrics = compute_metrics(val_masks, val_pred_prob)
    metrics["model_path"] = args.model_path
    metrics["num_val_patches"] = int(len(val_images))
    if args.training_run_id is not None:
        metrics["training_run_id"] = args.training_run_id

    logger.info(
        "Patch metrics: F1=%.4f IoU=%.4f precision=%.4f recall=%.4f",
        metrics["f1"],
        metrics["iou"],
        metrics["precision"],
        metrics["recall"],
    )

    if args.training_run_id is not None:
        mlflow.set_tag("training_run_id", args.training_run_id)
    mlflow.log_param("model_path", args.model_path)
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_param("threshold", BINARY_THRESHOLD)
    mlflow.log_param("num_val_patches", len(val_images))

    mlflow.log_metric("val_f1", metrics["f1"])
    mlflow.log_metric("val_iou", metrics["iou"])
    mlflow.log_metric("val_precision", metrics["precision"])
    mlflow.log_metric("val_recall", metrics["recall"])

    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output_path)), exist_ok=True)
    with open(args.metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", args.metrics_output_path)

    if args.full_images_path:
        logger.info("Generating sample overlays from %s", args.full_images_path)
        overlay_paths = generate_sample_overlays(
            model=model,
            full_images_path=args.full_images_path,
            overlays_dir=args.overlays_output_dir,
            num_samples=NUM_SAMPLE_OVERLAYS,
            patch_size=args.patch_size,
        )
        logger.info("Generated %d overlay PNGs", len(overlay_paths))
    else:
        logger.info("No --full-images-path provided; skipping overlay generation.")
        overlay_paths = []
        os.makedirs(args.overlays_output_dir, exist_ok=True)

    if not in_azure_ml_job():
        mlflow.log_artifact(args.metrics_output_path)
        for path in overlay_paths:
            mlflow.log_artifact(path, artifact_path="sample_overlays")
        logger.info("Logged %d overlays to MLflow", len(overlay_paths))
    else:
        logger.info("Skipping mlflow.log_artifact in Azure ML job; outputs captured via job mechanism.")


def main() -> None:
    """Run the evaluation pipeline and write outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    configure_mlflow(args.experiment_name)

    if in_azure_ml_job():
        # Azure ML pre-creates an MLflow run bound to the job; opening our own
        # would conflict with it. Log into the already-active run directly.
        _run_evaluation(args)
    else:
        # Local execution: open and close our own MLflow run.
        with mlflow.start_run(run_name=args.run_name):
            _run_evaluation(args)


if __name__ == "__main__":
    main()
