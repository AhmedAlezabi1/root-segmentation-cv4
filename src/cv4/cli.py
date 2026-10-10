"""Command line interface for the cv4 plant root segmentation system.

Provides two commands:
    cv4-predict: run inference on a plant image
    cv4-train: train the segmentation model

Predict usage:
    cv4-predict --image path/to/plant.png
    cv4-predict --image plant.png --output results.json
    cv4-predict --image plant.png --visualize

Train usage:
    cv4-train --data path/to/dataset
    cv4-train --data dataset/ --epochs 20 --batch-size 16
    cv4-train --data dataset/ --smoke-test
"""

import json
import logging
import sys
from pathlib import Path

import typer

predict_app = typer.Typer(help="CV4 plant root segmentation - predict")
train_app = typer.Typer(help="CV4 plant root segmentation - train")


def setup_logging() -> None:
    """Set up logging format for the CLI.

    Configures timestamps, module names, and log levels for all
    cv4 modules. This is the single place where log formatting
    is defined for the whole application. Log file is always
    saved to the outputs/ directory.
    """
    log_dir = Path("outputs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "cv4.log"),
        ],
    )


def format_results(results: dict) -> dict:
    """Convert pipeline results to a JSON-friendly format.

    Numpy arrays and other non-serializable types are converted
    to plain Python types so the output can be saved as JSON.

    Args:
        results (dict): Raw pipeline output from run_pipeline().

    Returns:
        dict: Clean dictionary with only JSON-serializable values.
    """
    output = {
        "image_path": results["image_path"],
        "processing_time_seconds": round(results["processing_time_seconds"], 2),
        "plate_bbox": {
            "x": int(results["plate_bbox"][0]),
            "y": int(results["plate_bbox"][1]),
            "side": int(results["plate_bbox"][2]),
        },
        "plants": [],
    }

    for plant in results["plants"]:
        plant_data = {
            "lane": plant["lane"],
            "has_root": plant["has_root"],
        }

        if plant["has_root"]:
            plant_data["root_tip_px"] = plant["root_tip_px"]
            plant_data["root_tip_mm"] = {
                "x": round(plant["root_tip_mm"]["x"], 3),
                "y": round(plant["root_tip_mm"]["y"], 3),
            }
            plant_data["confidence"] = round(plant["confidence"], 3)
        else:
            plant_data["root_tip_px"] = None
            plant_data["root_tip_mm"] = None
            plant_data["confidence"] = None

        output["plants"].append(plant_data)

    return output


def save_visualization(results: dict, output_path: str) -> None:
    """Save an image with the segmentation mask and root tips drawn on it.

    The mask from the pipeline is cropped to the petri dish region,
    so we place it back at the correct position on the full image
    using the bounding box coordinates.

    Args:
        results (dict): Raw pipeline output from run_pipeline().
        output_path (str): Path to save the visualization image.
    """
    import cv2

    image = cv2.imread(results["image_path"])
    if image is None:
        typer.echo(f"Could not read image for visualization: {results['image_path']}")
        return

    bbox_x, bbox_y, bbox_side = results["plate_bbox"]

    # Draw the binary mask as a green overlay in the dish region
    mask = results["binary_mask"]
    if mask is not None and mask.sum() > 0:
        overlay = image.copy()
        region = overlay[bbox_y : bbox_y + bbox_side, bbox_x : bbox_x + bbox_side]
        region[mask > 0] = [0, 255, 0]
        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Draw root tips as red circles
    for plant in results["plants"]:
        if plant["has_root"] and plant["root_tip_px"] is not None:
            abs_x = bbox_x + plant["root_tip_px"]["x"]
            abs_y = bbox_y + plant["root_tip_px"]["y"]
            cv2.circle(image, (abs_x, abs_y), 8, (0, 0, 255), -1)
            cv2.putText(
                image,
                f"L{plant['lane']}",
                (abs_x + 12, abs_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    cv2.imwrite(output_path, image)
    typer.echo(f"Visualization saved to {output_path}")


@predict_app.command()
def predict(
    image: str = typer.Option(..., "--image", help="Path to the input plant image (required)"),
    model: str = typer.Option(None, "--model", help="Path to model weights (.h5 file). Defaults to config value"),
    output: str = typer.Option(None, "--output", help="Path to save results as JSON. Defaults to outputs/ folder"),
    visualize: bool = typer.Option(False, "--visualize", help="Save an image with mask and root tips drawn on it"),
) -> None:
    """Run plant root segmentation on a single image.

    Takes a petri dish image, detects 5 plant roots, and outputs
    their tip coordinates in pixels and millimeters along with
    confidence scores. Results are always saved to a JSON file
    and printed to the terminal.

    Examples:
        cv4-predict --image data/plant.png
        cv4-predict --image plant.png --output results.json
        cv4-predict --image plant.png --visualize
        cv4-predict --image plant.png --model weights/custom.h5 --output out.json --visualize
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    if not Path(image).exists():
        typer.echo(f"Error: Image file not found: {image}")
        raise typer.Exit(code=1)

    try:
        from cv4.pipeline import run_pipeline

        logger.info("Running pipeline on %s", image)
        results = run_pipeline(image_path=image, model_path=model)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}")
        logger.exception("Pipeline failed")
        raise typer.Exit(code=1)

    formatted = format_results(results)

    if output is None:
        image_name = Path(image).stem
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output = str(output_dir / f"{image_name}_results.json")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(formatted, f, indent=2)
    typer.echo(f"Results saved to {output}")

    typer.echo(json.dumps(formatted, indent=2))

    if visualize:
        viz_path = str(Path(output).with_suffix("")) + "_viz.png"
        save_visualization(results, viz_path)

    roots_found = sum(1 for p in results["plants"] if p["has_root"])
    typer.echo(f"\nDone! {roots_found}/5 roots detected in {formatted['processing_time_seconds']}s")


@train_app.command()
def train(
    data: str = typer.Option(..., "--data", help="Path to dataset root with train_images/, train_masks/, val_images/, val_masks/"),
    model_output: str = typer.Option(None, "--model-output", help="Where to save the trained model (.h5). Defaults to outputs/"),
    epochs: int = typer.Option(50, "--epochs", help="Maximum number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="Number of patches per training step"),
    learning_rate: float = typer.Option(1e-4, "--learning-rate", help="Initial learning rate for the optimizer"),
    augmentation: str = typer.Option("mild", "--augmentation", help="Augmentation preset: none, mild, or strong"),
    experiment_name: str = typer.Option(None, "--experiment-name", help="MLflow experiment name for tracking"),
    run_name: str = typer.Option(None, "--run-name", help="MLflow run name to identify this training run"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Quick test run with only 2 steps per epoch"),
) -> None:
    """Train the plant root segmentation model.

    Builds a Residual Attention UNet, trains it on paired image/mask
    patches, logs metrics to MLflow, and saves the trained model.
    Uses early stopping and learning rate reduction to prevent
    overfitting.

    Examples:
        cv4-train --data path/to/dataset
        cv4-train --data dataset/ --epochs 20 --batch-size 16
        cv4-train --data dataset/ --augmentation strong --run-name "experiment_1"
        cv4-train --data dataset/ --smoke-test
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    if not Path(data).exists():
        typer.echo(f"Error: Data directory not found: {data}")
        raise typer.Exit(code=1)

    required_dirs = ["train_images", "train_masks", "val_images", "val_masks"]
    for subdir in required_dirs:
        if not (Path(data) / subdir).exists():
            typer.echo(f"Error: Missing required directory: {Path(data) / subdir}")
            raise typer.Exit(code=1)

    if augmentation not in ("none", "mild", "strong"):
        typer.echo(f"Error: Invalid augmentation preset '{augmentation}'. Choose from: none, mild, strong")
        raise typer.Exit(code=1)

    if model_output is None:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_output = str(output_dir / "trained_model.h5")

    metrics_output = str(Path(model_output).with_suffix("")) + "_metrics.json"

    train_args = [
        "--data-path",
        data,
        "--model-output-path",
        model_output,
        "--metrics-output-path",
        metrics_output,
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--augmentation-strength",
        augmentation,
    ]

    if experiment_name:
        train_args.extend(["--experiment-name", experiment_name])
    if run_name:
        train_args.extend(["--run-name", run_name])
    if smoke_test:
        train_args.extend(["--max-steps-per-epoch", "2", "--max-val-steps", "2"])

    logger.info("Starting training with data from %s", data)
    logger.info("Model will be saved to %s", model_output)

    # train.py reads its arguments from sys.argv via argparse, so we swap in the
    # built argument list while calling its main(), then always restore the
    # original sys.argv in the finally block. Using finally (not a line after
    # train_main()) guarantees the restore runs even if training raises, so an
    # in-process caller (a test or future wrapper) never sees a corrupted argv.
    original_argv = sys.argv
    try:
        sys.argv = ["train"] + train_args

        from cv4.training.train import main as train_main

        train_main()

        typer.echo("\nTraining complete!")
        typer.echo(f"Model saved to {model_output}")
        typer.echo(f"Metrics saved to {metrics_output}")

    except Exception as e:
        typer.echo(f"Training failed: {e}")
        logger.exception("Training failed")
        raise typer.Exit(code=1)
    finally:
        sys.argv = original_argv
