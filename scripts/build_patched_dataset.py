"""Preprocess raw data once and register the patched output as a data asset.

The hyperparameter sweep reuses the train component, which consumes the
four-folder patched dataset (train_images/, train_masks/, val_images/,
val_masks/), not the raw assets. A sweep wraps a single command, so re-running
preprocess inside every trial would waste compute. This script runs preprocess
once as a standalone job and registers its output as a versioned data asset
that the sweep points all trials at.

The job is built with command() rather than load_component() because a one-off
standalone job needs its inputs, outputs, environment, and resources declared
at construction time to keep the output binding intact through submission.
components/preprocess.yml remains the source of truth for the pipeline.

SMOKE = True patches the 10-pair sample (cv4-raw-*:1) into cv4-patched-smoke
for validating the sweep plumbing. SMOKE = False patches the full data
(cv4-raw-*:2) into cv4-patched-full for the real sweep.

The script prints the azureml:<name>:<version> to pin in the sweep. Pin the
printed version explicitly rather than @latest, which is unreliable for data
assets in the SDK.

Usage:
    uv run --group azure python scripts/build_patched_dataset.py
"""

import logging
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_CPU_INSTANCE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

SMOKE = False

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def main() -> None:
    """Submit a standalone preprocess job, then register its patched output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    if SMOKE:
        train_asset = "azureml:cv4-raw-train:1"
        val_asset = "azureml:cv4-raw-val:1"
        asset_name = "cv4-patched-smoke"
        display_name = "cv4-preprocess-smoke"
        logger.info("SMOKE run: patching 10-pair sample")
    else:
        train_asset = "azureml:cv4-raw-train:2"
        val_asset = "azureml:cv4-raw-val:2"
        asset_name = "cv4-patched-full"
        display_name = "cv4-preprocess-full"
        logger.info("FULL run: patching complete dataset")

    logger.info("Train asset: %s", train_asset)
    logger.info("Val asset:   %s", val_asset)

    # Preprocess is CPU-only OpenCV and numpy work, so it runs on the CPU
    # environment and the CPU instance type. The command matches the one in
    # components/preprocess.yml.
    job = command(
        code=str(SRC_DIR),
        command=(
            "python -m cv4.training.preprocess "
            "--train-data-path ${{inputs.train_data}} "
            "--val-data-path ${{inputs.val_data}} "
            "--output-path ${{outputs.patched_data}} "
            "--patch-size ${{inputs.patch_size}} "
            "--mask-suffix ${{inputs.mask_suffix}}"
        ),
        inputs={
            "train_data": Input(type=AssetTypes.URI_FOLDER, path=train_asset),
            "val_data": Input(type=AssetTypes.URI_FOLDER, path=val_asset),
            "patch_size": 256,
            "mask_suffix": "_root_mask.tif",
        },
        outputs={
            "patched_data": Output(
                type=AssetTypes.URI_FOLDER,
                name=asset_name,
                version="1",
            )
        },
        environment="azureml:cv4-cpu:3",
        compute=AZURE_COMPUTE,
        instance_type=AZURE_CPU_INSTANCE,
        experiment_name="cv4-preprocess",
        display_name=display_name,
    )

    logger.info("Submitting preprocess job...")
    submitted = client.jobs.create_or_update(job)
    logger.info("Submitted %s, streaming until completion...", submitted.name)
    logger.info("Studio URL: %s", submitted.studio_url)

    client.jobs.stream(submitted.name)

    print()
    print(f"Preprocess job : {submitted.name}")
    print(f"Patched asset  : azureml:{asset_name}:1")
    print("Pin this exact id in the sweep (the version, not @latest).")
    print()


if __name__ == "__main__":
    main()
