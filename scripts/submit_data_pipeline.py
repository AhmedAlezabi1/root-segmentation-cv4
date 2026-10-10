"""Submit the standalone cv4 data pipeline to Azure ML.

Runs cv4_data_pipeline against the registered raw data assets and registers the
patched output as a versioned data asset. The pipeline materializes as one job
in the studio with a visual graph, and its output appears in the data assets
list once the run completes.

SMOKE = True patches the 10-pair sample (cv4-raw-*:1) into cv4-patched-smoke
for validating the plumbing. SMOKE = False patches the full data (cv4-raw-*:2)
into cv4-patched-full for the real processed dataset.

The output asset name and version are set on the pipeline output here rather
than in the pipeline definition, so the same pipeline can register different
named assets across smoke and full runs. Pin the printed version explicitly in
downstream training and sweep jobs rather than using @latest, which is
unreliable for data assets in the SDK.

Usage:
    uv run --group azure python scripts/submit_data_pipeline.py
"""

import logging

from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)
from cv4.training.data_pipeline import cv4_data_pipeline

logger = logging.getLogger(__name__)

SMOKE = False

# The version registered for the patched asset. Bump this when reprocessing
# produces a dataset you want to keep alongside the previous one.
ASSET_VERSION = "3"


def main() -> None:
    """Build and submit the cv4 data pipeline, registering its patched output."""
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
        display_name = "cv4-data-pipeline-smoke"
        logger.info("SMOKE run: patching 10-pair sample")
    else:
        train_asset = "azureml:cv4-raw-train:2"
        val_asset = "azureml:cv4-raw-val:2"
        asset_name = "cv4-patched-full"
        display_name = "cv4-data-pipeline-full"
        logger.info("FULL run: patching complete dataset")

    logger.info("Train asset: %s", train_asset)
    logger.info("Val asset:   %s", val_asset)

    job = cv4_data_pipeline(
        train_data=Input(type=AssetTypes.URI_FOLDER, path=train_asset),
        val_data=Input(type=AssetTypes.URI_FOLDER, path=val_asset),
        patch_size=256,
        mask_suffix="_root_mask.tif",
    )

    # Naming the pipeline output is what registers it as a versioned data asset
    # when the run completes. Set on the job output here so smoke and full runs
    # register under different names.
    job.outputs.patched_data = Output(
        type=AssetTypes.URI_FOLDER,
        name=asset_name,
        version=ASSET_VERSION,
    )

    job.settings.default_compute = AZURE_COMPUTE
    job.experiment_name = "cv4-data-pipeline"
    job.display_name = display_name

    logger.info("Submitting data pipeline...")
    submitted = client.jobs.create_or_update(job)

    print()
    print(f"Data pipeline job : {submitted.name}")
    print(f"Status            : {submitted.status}")
    print(f"Studio URL        : {submitted.studio_url}")
    print(f"Patched asset     : azureml:{asset_name}:{ASSET_VERSION}")
    print("Pin this exact id in training and sweep jobs (the version, not @latest).")
    print()


if __name__ == "__main__":
    main()
