"""Create (or update) the weekly Azure ML hyperparameter sweep schedule for cv4.

Registers a cron-based schedule that submits the cv4 sweep job every Sunday
at 04:00 UTC (two hours after the regular retrain at 02:00 UTC, so they do
not race for the GPU cluster). The sweep explores learning rate, batch size,
and augmentation strength using random sampling with median stopping.

DATA ASSET VERSION: hardcoded because the Azure ML SDK resolves @latest at
schedule registration time, not at run time. Bump and rerun when new patched
data is registered.

Usage:
    uv run --group azure python scripts/create_weekly_sweep_schedule.py

To disable the schedule without deleting it:
    uv run --group azure python scripts/create_weekly_sweep_schedule.py --disable

To re-enable it:
    uv run --group azure python scripts/create_weekly_sweep_schedule.py --enable
"""

import argparse
import logging
import math
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import CronTrigger, JobSchedule
from azure.ai.ml.sweep import Choice, LogUniform, MedianStoppingPolicy
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_GPU_INSTANCE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

SCHEDULE_NAME = "cv4-weekly-sweep"
CRON_EXPRESSION = "0 4 * * 0"  # every Sunday at 04:00 UTC

PATCHED_DATA_VERSION = "1"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _build_sweep_job():
    """Return a configured sweep job ready to pass to JobSchedule."""
    base_command = (
        "python -m cv4.training.train "
        "--data-path ${{inputs.patched_data}} "
        "--model-output-path ${{outputs.trained_model}}/model.h5 "
        "--metrics-output-path ${{outputs.metrics}}/metrics.json "
        "--learning-rate ${{inputs.learning_rate}} "
        "--batch-size ${{inputs.batch_size}} "
        "--epochs ${{inputs.epochs}} "
        "--augmentation-strength ${{inputs.augmentation_strength}} "
        "--patch-size ${{inputs.patch_size}}"
    )

    base_job = command(
        code=str(SRC_DIR),
        command=base_command,
        inputs={
            "patched_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml:cv4-patched-full:{PATCHED_DATA_VERSION}",
                mode="download",
            ),
            "epochs": 50,
            "patch_size": 256,
            "learning_rate": 1e-4,
            "batch_size": 16,
            "augmentation_strength": "mild",
        },
        outputs={
            "trained_model": Output(type=AssetTypes.URI_FOLDER),
            "metrics": Output(type=AssetTypes.URI_FOLDER),
        },
        environment="azureml:cv4-training:6",
        compute=AZURE_COMPUTE,
        instance_type=AZURE_GPU_INSTANCE,
    )

    sweep_trial = base_job(
        learning_rate=LogUniform(min_value=math.log(1e-5), max_value=math.log(1e-2)),
        batch_size=Choice([8, 16, 32]),
        augmentation_strength=Choice(["none", "mild", "strong"]),
    )

    sweep_job = sweep_trial.sweep(
        compute=AZURE_COMPUTE,
        sampling_algorithm="random",
        primary_metric="val_f1_metric",
        goal="Maximize",
        max_total_trials=10,
        max_concurrent_trials=2,
        early_termination_policy=MedianStoppingPolicy(delay_evaluation=5, evaluation_interval=2),
    )
    sweep_job.experiment_name = "cv4-sweep"
    sweep_job.display_name = "cv4-weekly-sweep"
    sweep_job.set_resources(instance_type=AZURE_GPU_INSTANCE)
    return sweep_job


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Manage the cv4 weekly sweep schedule")
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

    sweep_job = _build_sweep_job()

    schedule = JobSchedule(
        name=SCHEDULE_NAME,
        trigger=CronTrigger(expression=CRON_EXPRESSION),
        create_job=sweep_job,
        description="Weekly cv4 hyperparameter sweep. Runs every Sunday at 04:00 UTC.",
    )

    logger.info("Creating/updating schedule '%s' (cron: %s)", SCHEDULE_NAME, CRON_EXPRESSION)
    created = client.schedules.begin_create_or_update(schedule).result()

    print()
    print(f"Schedule name  : {created.name}")
    print(f"Status         : {created.provisioning_state}")
    print(f"Cron           : {CRON_EXPRESSION}  (every Sunday 04:00 UTC)")
    print(f"Patched asset  : cv4-patched-full:{PATCHED_DATA_VERSION}")
    print()


if __name__ == "__main__":
    main()
