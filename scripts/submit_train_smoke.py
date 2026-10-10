"""Submit train.py as a standalone Azure ML smoke-test job.

Phase 2 step 1: verifies the full cloud training chain end to end without
committing to components or pipelines yet. Uploads the local patched subset
at data/dataset/ as a job-scoped blob rather than registering it as a named
data asset, since this data is for one-time validation only and Azure ML
data assets cannot be deleted once registered.

The smoke-test caps (--max-steps-per-epoch 2 --max-val-steps 2) keep the run
under 10 minutes so we can iterate quickly. Drop these flags when running a
real training job.

Usage:
    uv run --group azure python scripts/submit_train_smoke.py

The script prints the Azure ML studio URL on submission. Open it in a
browser to watch the job progress through Queued -> Preparing -> Running ->
Completed.
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

# Absolute paths derived from this script's location. The Azure ML SDK
# interprets bare relative strings like "../src" as registry references and
# prefixes them with "azureml:", which fails. Absolute paths sidestep that.
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_PATH = REPO_ROOT / "src"
DATA_PATH = REPO_ROOT / "data" / "dataset"


def main() -> None:
    """Build and submit the smoke-test training job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Fail fast if the local data isn't where we expect, since the SDK's error
    # message in this case is misleading (it talks about registry paths).
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_PATH}")
    if not CODE_PATH.exists():
        raise FileNotFoundError(f"Source folder not found: {CODE_PATH}")

    logger.info("Code path: %s", CODE_PATH)
    logger.info("Data path: %s", DATA_PATH)
    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    # Job-scoped input: SDK uploads data/dataset/ as a one-off blob for this
    # job only. No permanent data asset gets registered, keeping the workspace
    # tidy. The cluster mounts it read-only at the path bound to ${{inputs.data}}.
    job = command(
        code=str(CODE_PATH),
        command=(
            "python -m cv4.training.train "
            "--data-path ${{inputs.data}} "
            "--model-output-path ${{outputs.model}}/model.h5 "
            "--metrics-output-path ${{outputs.metrics}}/metrics.json "
            "--epochs 2 "
            "--max-steps-per-epoch 2 "
            "--max-val-steps 2 "
            "--run-name smoke-test-step1"
        ),
        inputs={
            "data": Input(
                type=AssetTypes.URI_FOLDER,
                path=str(DATA_PATH),
            ),
        },
        outputs={
            "model": Output(type=AssetTypes.URI_FOLDER),
            "metrics": Output(type=AssetTypes.URI_FOLDER),
        },
        environment="cv4-training@latest",
        compute=AZURE_COMPUTE,
        instance_type=AZURE_GPU_INSTANCE,
        experiment_name="cv4-smoke-tests",
        display_name="smoke-test-train-py-standalone",
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
