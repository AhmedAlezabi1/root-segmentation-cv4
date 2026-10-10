"""Submit a hyperparameter sweep over the cv4 training component.

Task 3.4 (ILO 8.9). Sweeps three hyperparameters on top of the existing
training component: learning rate (log-uniform), batch size, and augmentation
strength. The sweep reuses the patched dataset produced by
build_patched_dataset.py rather than re-running preprocess per trial.

The base job is built with command() so inputs, environment, code, and the GPU
instance type are set at construction. Calling that job with sweep
distributions and then .sweep() produces the sweep job. Sampling is random
with median early termination; the two are paired deliberately, since Bayesian
sampling cannot be combined with an early termination policy.

The primary metric is val_f1_metric, which is the exact per-epoch metric name
emitted by the training script's MLflow epoch logger. It must match that name
for the sweep to rank and terminate trials correctly.

SMOKE = True runs a 2-trial plumbing check on cv4-patched-smoke with capped
steps. SMOKE = False runs the real 10-trial sweep on cv4-patched-full.

Usage:
    uv run --group azure python scripts/submit_sweep.py
"""

import logging
import math
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
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

SMOKE = False

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def main() -> None:
    """Build and submit the hyperparameter sweep job."""
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
        patched_asset = "azureml:cv4-patched-smoke:1"
        epochs = 2
        max_total_trials = 2
        max_concurrent_trials = 2
        display_name = "cv4-sweep-smoke"
        # Smoke caps keep each trial to a couple of steps so the plumbing is
        # validated without paying for real training.
        extra_inputs = {"max_steps_per_epoch": 2, "max_val_steps": 2}
        logger.info("SMOKE run: 2 trials on 10-pair patched sample, capped steps")
    else:
        patched_asset = "azureml:cv4-patched-full:1"
        epochs = 50
        max_total_trials = 10
        max_concurrent_trials = 2
        display_name = "cv4-sweep-full"
        extra_inputs = {}
        logger.info("FULL run: 10 trials on full patched data, 4 concurrent")

    logger.info("Patched asset: %s", patched_asset)

    # Base job. Fixed inputs and the GPU instance type are set here at
    # construction. The command matches components/train.yml; the swept inputs
    # are given placeholder defaults that the sweep overrides per trial.
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
    for cap in extra_inputs:
        flag = cap.replace("_", "-")
        base_command += f" --{flag} ${{{{inputs.{cap}}}}}"

    base_inputs = {
        "patched_data": Input(
            type=AssetTypes.URI_FOLDER,
            path=patched_asset,
            mode="download",
        ),
        "epochs": epochs,
        "patch_size": 256,
        "learning_rate": 1e-4,
        "batch_size": 16,
        "augmentation_strength": "mild",
        **extra_inputs,
    }

    base_job = command(
        code=str(SRC_DIR),
        command=base_command,
        inputs=base_inputs,
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
        max_total_trials=max_total_trials,
        max_concurrent_trials=max_concurrent_trials,
        early_termination_policy=MedianStoppingPolicy(delay_evaluation=5, evaluation_interval=2),
    )
    sweep_job.experiment_name = "cv4-sweep"
    sweep_job.display_name = display_name
    sweep_job.set_resources(instance_type=AZURE_GPU_INSTANCE)

    logger.info("Submitting sweep job...")
    submitted = client.jobs.create_or_update(sweep_job)

    print()
    print(f"Sweep job  : {submitted.name}")
    print(f"Status     : {submitted.status}")
    print(f"Studio URL : {submitted.studio_url}")
    print()


if __name__ == "__main__":
    main()
