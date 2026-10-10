"""Evaluate the best sweep trial's model and conditionally register it.

The hyperparameter sweep ranks trials by val_f1_metric but does not run the
evaluate or register components. This script closes that gap for the winning
trial: it evaluates the trial's saved model with the same evaluate component
that produced the current baseline (cv4_root_segmentation:2), then runs the
register component on the resulting metrics. Registering through evaluate
means the new model's F1 is computed identically to the baseline's and is
directly comparable.

The best child run id is taken as an argument rather than discovered
programmatically: the sweep's trial comparison in Studio shows the winner
clearly, and reading it there avoids the unreliable best-run SDK lookup.
Find the best trial in the sweep job's Trials tab, copy its run name (the
parent sweep name followed by an underscore and a number), and pass it.

Evaluate runs on GPU (it loads the model and predicts); register runs on
CPU. Both are built with command() at construction time and submitted as
standalone jobs, then streamed to completion in sequence.

Usage:
    uv run --group azure python scripts/register_best_sweep_model.py \\
        --best-run-id cv4-sweep-full_7
"""

import argparse
import logging
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import JobResourceConfiguration
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_GPU_INSTANCE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

PATCHED_FULL = "azureml:cv4-patched-full:1"
RAW_VAL = "azureml:cv4-raw-val:2"
MODEL_NAME = "cv4_root_segmentation"
F1_THRESHOLD = 0.80


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and register the best sweep trial's model.")
    parser.add_argument(
        "--best-run-id",
        required=True,
        help="Run name of the winning sweep trial, e.g. cv4-sweep-full_7. Read it from the sweep's Trials tab.",
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="Registered asset URI of the winning trial's trained_model folder (azureml:<asset_name>:<version>), from the trial's Outputs tab.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the winning trial's model, then conditionally register it."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    # The winning trial's trained_model output is the model to promote.
    winner_model = Input(
        type=AssetTypes.URI_FOLDER,
        path=args.model_uri,
    )

    # Phase 1: evaluate the winner with the same component that scored the
    # baseline, so the resulting f1 is comparable to 0.8488. Patched data
    # supplies the val patches for metrics; the raw val asset supplies full
    # images for overlays.
    evaluate_job = command(
        code=str(SRC_DIR),
        command=(
            "python -m cv4.training.evaluate "
            "--model-path ${{inputs.trained_model}}/model.h5 "
            "--data-path ${{inputs.patched_data}} "
            "--full-images-path ${{inputs.full_images}} "
            "--metrics-output-path ${{outputs.eval_metrics}}/metrics.json "
            "--overlays-output-dir ${{outputs.overlays}} "
            "--batch-size 32 "
            "--patch-size 256"
        ),
        inputs={
            "trained_model": winner_model,
            "patched_data": Input(type=AssetTypes.URI_FOLDER, path=PATCHED_FULL, mode="download"),
            "full_images": Input(type=AssetTypes.URI_FOLDER, path=RAW_VAL),
        },
        outputs={
            "eval_metrics": Output(type=AssetTypes.URI_FOLDER),
            "overlays": Output(type=AssetTypes.URI_FOLDER),
        },
        environment="azureml:cv4-training:6",
        compute=AZURE_COMPUTE,
        resources=JobResourceConfiguration(instance_type=AZURE_GPU_INSTANCE),
        experiment_name="cv4-sweep-register",
        display_name=f"evaluate-{args.best_run_id}",
    )

    logger.info("Submitting evaluate job for %s...", args.best_run_id)
    eval_submitted = client.jobs.create_or_update(evaluate_job)
    logger.info("Studio URL: %s", eval_submitted.studio_url)
    client.jobs.stream(eval_submitted.name)

    # Registration is run as a separate step via submit_register.py. Azure
    # registers each job output as a data asset, and register reads the
    # eval_metrics asset by its azureml:<name>:<version> URI. Passing this
    # job's output as a jobs/ URI to a downstream job is rejected by Azure,
    # so the two phases are decoupled rather than chained.
    print()
    print(f"Evaluate job : {eval_submitted.name}")
    print("Evaluation complete. To register the model, find the eval_metrics")
    print("output asset on this job's Outputs tab (azureml:..._eval_metrics:1)")
    print("and run:")
    print()
    print("  uv run --group azure python scripts/submit_register.py \\")
    print(f"    --model-uri {args.model_uri} \\")
    print("    --metrics-uri azureml:<eval_metrics asset from the Outputs tab>")
    print()


if __name__ == "__main__":
    main()
