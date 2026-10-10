"""Submit a standalone preprocess job to validate cv4.training.preprocess.

Phase 2 step 3 (preprocess validation): builds the job directly with the
command() helper rather than loading the YAML component, because the SDK's
load_component + mutation flow on Windows trips on multiple resolution
quirks (code path, environment, outputs) that aren't worth fighting for a
single-node test job. The YAML at components/preprocess.yml remains the
canonical declaration for when we wire the pipeline in step 5.

Pulls Ahmed's sample assets (cv4-raw-train:1, cv4-raw-val:1), runs
preprocessing on the cv4-cpu environment, and writes a job-scoped output
folder.

Usage:
    uv run --group azure python scripts/submit_preprocess_test.py
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

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_PATH = REPO_ROOT / "src"


def main() -> None:
    """Build and submit a single-node preprocess job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not CODE_PATH.exists():
        raise FileNotFoundError(f"Source folder not found: {CODE_PATH}")

    logger.info("Code path: %s", CODE_PATH)
    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    # Preprocess is CPU-bound (image I/O + OpenCV), so we run it on the
    # default CPU instance type with the cv4-cpu environment, which has
    # opencv-python-headless + scipy + numpy without the heavy TF stack.
    job = command(
        code=str(CODE_PATH),
        command=(
            "python -m cv4.training.preprocess "
            "--train-data-path ${{inputs.train_data}} "
            "--val-data-path ${{inputs.val_data}} "
            "--output-path ${{outputs.patched_data}} "
            "--patch-size ${{inputs.patch_size}} "
            "--mask-suffix ${{inputs.mask_suffix}}"
        ),
        inputs={
            "train_data": Input(type=AssetTypes.URI_FOLDER, path="azureml:cv4-raw-train:1"),
            "val_data": Input(type=AssetTypes.URI_FOLDER, path="azureml:cv4-raw-val:1"),
            "patch_size": 256,
            "mask_suffix": "_root_mask.tif",
        },
        outputs={
            "patched_data": Output(type=AssetTypes.URI_FOLDER),
        },
        environment="cv4-cpu:2",
        compute=AZURE_COMPUTE,
        instance_type=AZURE_CPU_INSTANCE,
        experiment_name="cv4-preprocess-tests",
        display_name="preprocess-component-standalone",
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
