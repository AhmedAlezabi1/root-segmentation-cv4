"""Training entry point for the cv4 segmentation model.

Runs end to end: builds a Residual Attention U-Net, compiles it with the
hybrid loss, fits it against paired patch generators, and writes both the
trained model and a metrics JSON to the requested output paths. Designed to
be invoked either directly from the command line for local development or by
an Azure ML pipeline component, which calls the same argparse interface.

MLflow tracks hyperparameters at run start, per-epoch metrics during training,
and the final model and metrics file as artifacts at run end. Tracking URI
and experiment name are resolved by cv4.training.mlflow_setup.
"""

import argparse
import json
import logging
import os

import mlflow
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from cv4.training.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    MIN_LEARNING_RATE,
    PATCH_SIZE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
)
from cv4.training.data import build_generators
from cv4.training.losses import hybrid_loss
from cv4.training.metrics import f1_metric
from cv4.training.mlflow_setup import configure_mlflow, in_azure_ml_job
from cv4.training.model import build_res_attention_unet

logger = logging.getLogger(__name__)


def enable_gpu_memory_growth() -> None:
    """Allocate GPU memory on demand instead of reserving each device fully.

    TensorFlow otherwise reserves all memory of every visible GPU at startup,
    which locks an entire GPU per job on the shared cluster and starves other
    jobs. Must run before any GPU is initialized, or set_memory_growth raises.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning("Could not set memory growth on %s: %s", gpu, e)


class MLflowEpochLogger(Callback):
    """Keras callback that streams per-epoch metrics to MLflow.

    Keras computes metrics at the end of each epoch and exposes them via the
    logs dict passed to on_epoch_end. Forwarding them through mlflow.log_metric
    with step=epoch makes them appear as curves in the MLflow UI rather than
    point estimates.
    """

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if not logs:
            return
        for key, value in logs.items():
            mlflow.log_metric(key, float(value), step=epoch)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Argparse rather than Typer because Azure ML pipeline components launch
    via a plain shell command and argparse maps to that contract most
    directly. The user-facing cv4-train CLI (Issue 3.7) wraps this module
    with Typer and is built separately.
    """
    parser = argparse.ArgumentParser(description="Train the cv4 segmentation model.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Root directory containing train_images/, train_masks/, val_images/, val_masks/.",
    )
    parser.add_argument(
        "--model-output-path",
        required=True,
        help="Destination .h5 file for the trained model.",
    )
    parser.add_argument(
        "--metrics-output-path",
        required=True,
        help="Destination JSON file for the best train/val metrics.",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--augmentation-strength",
        choices=["none", "mild", "strong"],
        default="mild",
        help="Augmentation preset to apply to the training generator.",
    )
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Cap training steps per epoch. Used for fast local smoke tests; leave unset for real runs.",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=None,
        help="Cap validation steps per epoch. Used for fast local smoke tests; leave unset for real runs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name. Falls back to CV4_MLFLOW_EXPERIMENT env var or the package default.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional MLflow run name to make this run easier to identify in the UI. "
            "Ignored inside Azure ML jobs, where the run name is bound to the job name."
        ),
    )
    return parser.parse_args()


def _run_training(args: argparse.Namespace) -> None:
    """Train the model and log everything to the currently active MLflow run.

    Caller is responsible for opening the MLflow run before invoking this and
    closing it afterward. Locally that means wrapping with mlflow.start_run();
    inside Azure ML the platform already opened the run.
    """
    logger.info("Building generators from %s", args.data_path)
    train_gen, val_gen, train_steps, val_steps = build_generators(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        augmentation_strength=args.augmentation_strength,
    )

    if args.max_steps_per_epoch is not None:
        train_steps = min(train_steps, args.max_steps_per_epoch)
        logger.info("Capping training steps per epoch at %d", train_steps)
    if args.max_val_steps is not None:
        val_steps = min(val_steps, args.max_val_steps)
        logger.info("Capping validation steps at %d", val_steps)

    logger.info("Building Residual Attention U-Net at %dx%d", args.patch_size, args.patch_size)
    model = build_res_attention_unet(args.patch_size, args.patch_size, 1)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=hybrid_loss,
        metrics=["accuracy", f1_metric],
    )

    callbacks = [
        # restore_best_weights ensures the saved model is the best-validation
        # epoch, not whichever epoch happened to be last when training stopped.
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        MLflowEpochLogger(),
    ]

    # Logged at run start so they survive in the UI even if fit crashes.
    mlflow.log_params(
        {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "augmentation_strength": args.augmentation_strength,
            "patch_size": args.patch_size,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "reduce_lr_factor": REDUCE_LR_FACTOR,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
        }
    )

    logger.info("Training for up to %d epochs", args.epochs)
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.model_output_path)), exist_ok=True)
    model.save(args.model_output_path)
    logger.info("Saved model to %s", args.model_output_path)

    metrics = {
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_f1": float(max(history.history["val_f1_metric"])),
        "best_train_loss": float(min(history.history["loss"])),
        "best_train_f1": float(max(history.history["f1_metric"])),
        "epochs_run": len(history.history["loss"]),
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "augmentation_strength": args.augmentation_strength,
            "patch_size": args.patch_size,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output_path)), exist_ok=True)
    with open(args.metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", args.metrics_output_path)

    # Skipped inside Azure ML jobs: the platform captures output paths as job
    # artifacts, and the cluster's azureml-mlflow plugin is incompatible with
    # the mlflow 3.x artifact API (TypeError on tracking_uri).
    if not in_azure_ml_job():
        mlflow.log_artifact(args.model_output_path)
        mlflow.log_artifact(args.metrics_output_path)
    else:
        logger.info("Skipping mlflow.log_artifact in Azure ML job; outputs captured via job mechanism.")

    mlflow.log_metric("best_val_loss", metrics["best_val_loss"])
    mlflow.log_metric("best_val_f1", metrics["best_val_f1"])
    mlflow.log_metric("best_train_loss", metrics["best_train_loss"])
    mlflow.log_metric("best_train_f1", metrics["best_train_f1"])


def main() -> None:
    """Build, train, and save the model, then write best-epoch metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    enable_gpu_memory_growth()
    gpus = tf.config.list_physical_devices("GPU")
    logger.info("[DEVICE] Training on: %s", f"GPU x{len(gpus)} ({', '.join(g.name for g in gpus)})" if gpus else "CPU (no GPU detected)")
    args = parse_args()

    configure_mlflow(args.experiment_name)

    if in_azure_ml_job():
        # Azure ML pre-creates an MLflow run bound to the job; opening our own
        # would conflict with it.
        _run_training(args)
    else:
        with mlflow.start_run(run_name=args.run_name):
            _run_training(args)


if __name__ == "__main__":
    main()
