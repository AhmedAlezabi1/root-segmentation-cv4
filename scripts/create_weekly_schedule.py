"""Create (or update) the weekly Azure ML retraining schedule for cv4.

Registers a cron-based schedule that submits the cv4 training pipeline
every Sunday at 02:00 UTC.

DATA ASSET VERSIONS: the versions below are hardcoded because the Azure ML
SDK resolves @latest at schedule registration time (not at run time), which
causes validation errors. When a new data version is registered, bump the
version numbers and rerun this script to update the schedule.

Usage:
    uv run --group azure python scripts/create_weekly_schedule.py

To disable the schedule without deleting it:
    uv run --group azure python scripts/create_weekly_schedule.py --disable

To re-enable it:
    uv run --group azure python scripts/create_weekly_schedule.py --enable
"""

import argparse
import logging

from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import CronTrigger, JobSchedule
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)
from cv4.training.pipeline import cv4_pipeline

logger = logging.getLogger(__name__)

SCHEDULE_NAME = "cv4-weekly-retrain"
CRON_EXPRESSION = "0 2 * * 0"  # every Sunday at 02:00 UTC

# Bump these when new data versions are registered, then rerun this script.
TRAIN_DATA_VERSION = "2"
VAL_DATA_VERSION = "2"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Manage the cv4 weekly retrain schedule")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--disable", action="store_true", help="Disable the schedule")
    group.add_argument("--enable", action="store_true", help="Re-enable the schedule")
    args = parser.parse_args()

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    if args.disable:
        client.schedules.begin_disable(SCHEDULE_NAME).result()
        logger.info("Schedule '%s' disabled", SCHEDULE_NAME)
        return

    if args.enable:
        client.schedules.begin_enable(SCHEDULE_NAME).result()
        logger.info("Schedule '%s' enabled", SCHEDULE_NAME)
        return

    job = cv4_pipeline(
        train_data=Input(
            type=AssetTypes.URI_FOLDER,
            path=f"azureml:cv4-raw-train:{TRAIN_DATA_VERSION}",
        ),
        val_data=Input(
            type=AssetTypes.URI_FOLDER,
            path=f"azureml:cv4-raw-val:{VAL_DATA_VERSION}",
        ),
        patch_size=256,
        mask_suffix="_root_mask.tif",
        learning_rate=1e-4,
        batch_size=16,
        epochs=50,
        augmentation_strength="mild",
        eval_batch_size=32,
        model_name="cv4_root_segmentation",
        f1_threshold=0.80,
    )
    job.settings.default_compute = AZURE_COMPUTE
    job.experiment_name = "cv4-scheduled-retrain"
    job.display_name = "cv4-weekly-retrain"

    schedule = JobSchedule(
        name=SCHEDULE_NAME,
        trigger=CronTrigger(expression=CRON_EXPRESSION),
        create_job=job,
        description="Weekly cv4 retraining on NPEC data. Runs every Sunday at 02:00 UTC.",
    )

    logger.info("Creating/updating schedule '%s' (cron: %s)", SCHEDULE_NAME, CRON_EXPRESSION)
    created = client.schedules.begin_create_or_update(schedule).result()

    print()
    print(f"Schedule name  : {created.name}")
    print(f"Status         : {created.provisioning_state}")
    print(f"Cron           : {CRON_EXPRESSION}  (every Sunday 02:00 UTC)")
    print(f"Train asset    : cv4-raw-train:{TRAIN_DATA_VERSION}")
    print(f"Val asset      : cv4-raw-val:{VAL_DATA_VERSION}")
    print()


if __name__ == "__main__":
    main()
