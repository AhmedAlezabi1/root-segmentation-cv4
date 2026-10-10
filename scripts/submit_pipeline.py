"""Submit the end-to-end cv4 training pipeline to Azure ML.

Phase 2 step 5: submits cv4_pipeline against the project's registered
data assets. The pipeline runs four components in sequence
(preprocess -> train -> evaluate -> register) and materializes as one
job in the studio with a visual DAG.

Default config is a smoke run: cv4-raw-train:1 / cv4-raw-val:1 (10-pair
sample), smoke caps on train (2 steps/epoch, 2 val steps, 2 epochs). This
keeps the first end-to-end pipeline test under ~15 minutes wall time so
DAG wiring bugs surface quickly. Switch SMOKE = False to run on full data
(cv4-raw-train:2 / cv4-raw-val:2) without caps for a real training run.

Usage:
    uv run --group azure python scripts/submit_pipeline.py
"""

import logging

from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)
from cv4.training.pipeline import cv4_pipeline

logger = logging.getLogger(__name__)

SMOKE = False


def main() -> None:
    """Build and submit the cv4 training pipeline."""
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
        epochs = 2
        display_name = "cv4-pipeline-smoke"
        # Smoke caps are passed as kwargs only in this branch so the train
        # component's optional inputs stay unbound on full runs.
        smoke_kwargs = {"max_steps_per_epoch": 2, "max_val_steps": 2}
        logger.info("SMOKE run: 10-pair sample, capped training steps")
    else:
        train_asset = "azureml:cv4-raw-train:2"
        val_asset = "azureml:cv4-raw-val:2"
        epochs = 50
        display_name = "cv4-pipeline-full"
        smoke_kwargs = {}
        logger.info("FULL run: complete dataset, no training caps")

    logger.info("Train asset: %s", train_asset)
    logger.info("Val asset:   %s", val_asset)

    job = cv4_pipeline(
        train_data=Input(type=AssetTypes.URI_FOLDER, path=train_asset),
        val_data=Input(type=AssetTypes.URI_FOLDER, path=val_asset),
        patch_size=256,
        mask_suffix="_root_mask.tif",
        learning_rate=1e-4,
        batch_size=16,
        epochs=epochs,
        augmentation_strength="mild",
        eval_batch_size=32,
        model_name="cv4_root_segmentation",
        f1_threshold=0.80,
        **smoke_kwargs,
    )

    job.settings.default_compute = AZURE_COMPUTE
    job.experiment_name = "cv4-pipeline-tests"
    job.display_name = display_name

    logger.info("Submitting pipeline...")
    submitted = client.jobs.create_or_update(job)

    print()
    print(f"Pipeline job  : {submitted.name}")
    print(f"Status        : {submitted.status}")
    print(f"Studio URL    : {submitted.studio_url}")
    print()


if __name__ == "__main__":
    main()
