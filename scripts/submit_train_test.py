"""Submit a standalone train job to validate cv4.training.train as a component.

Phase 2 step 3 (train validation): re-uses the cloud-context dispatch in
train.py to run training as a single-node job consuming local patched data
uploaded as job-scoped blobs. Same upload pattern as submit_train_smoke.py;
the difference here is that this submission shapes the job to match the
component contract (URI_FOLDER inputs/outputs at the boundary of train.yml),
so the workflow we validate is what the pipeline will run in step 5.

Like submit_preprocess_test.py, we build the job with command() directly
rather than load_component(...) because the SDK's component-mutation flow
on Windows drops bindings that should be sticky. The YAML at
components/train.yml remains the canonical declaration for pipeline
assembly in step 5.

Usage:
    uv run --group azure python scripts/submit_train_test.py
"""

import logging
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_GPU_INSTANCE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_PATH = REPO_ROOT / "src"
LOCAL_DATA_PATH = REPO_ROOT / "data" / "dataset"


def main() -> None:
    """Build and submit a single-node train job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not CODE_PATH.exists():
        raise FileNotFoundError(f"Source folder not found: {CODE_PATH}")
    if not LOCAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Patched data folder not found: {LOCAL_DATA_PATH}. Run preprocess locally or copy the smoke-test dataset there first."
        )

    logger.info("Code path: %s", CODE_PATH)
    logger.info("Local patched data: %s", LOCAL_DATA_PATH)
    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    # Smoke-test caps (2 steps each) keep the run under a minute so we can
    # iterate on component plumbing without paying the full training cost.
    # Remove or raise these for real training runs.
    job = command(
        code=str(CODE_PATH),
        command=(
            "python -m cv4.training.train "
            "--data-path ${{inputs.patched_data}} "
            "--model-output-path ${{outputs.trained_model}}/model.h5 "
            "--metrics-output-path ${{outputs.metrics}}/metrics.json "
            "--learning-rate ${{inputs.learning_rate}} "
            "--batch-size ${{inputs.batch_size}} "
            "--epochs ${{inputs.epochs}} "
            "--augmentation-strength ${{inputs.augmentation_strength}} "
            "--patch-size ${{inputs.patch_size}} "
            "--max-steps-per-epoch ${{inputs.max_steps_per_epoch}} "
            "--max-val-steps ${{inputs.max_val_steps}}"
        ),
        inputs={
            "patched_data": Input(type=AssetTypes.URI_FOLDER, path=str(LOCAL_DATA_PATH)),
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 2,
            "augmentation_strength": "mild",
            "patch_size": 256,
            "max_steps_per_epoch": 2,
            "max_val_steps": 2,
        },
        outputs={
            "trained_model": Output(type=AssetTypes.URI_FOLDER),
            "metrics": Output(type=AssetTypes.URI_FOLDER),
        },
        environment="cv4-training:6",
        compute=AZURE_COMPUTE,
        instance_type=AZURE_GPU_INSTANCE,
        experiment_name="cv4-train-tests",
        display_name="train-component-standalone",
    )

    logger.info("Submitting job...")
    submitted = client.jobs.create_or_update(job)

    print()
    print(f"Job submitted : {submitted.name}")
    print(f"Status        : {submitted.status}")
    print(f"Studio URL    : {submitted.studio_url}")
    print()


if __name__ == "__main__":
    main()
