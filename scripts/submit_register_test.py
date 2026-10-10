"""Submit a standalone register job to validate cv4.training.register.

Phase 2 step 4 (real Azure ML registration): runs the conditional
registration component as a single-node job. Inputs are the trained model
and the evaluation metrics; output is a decision JSON. The component is
CPU-only and lightweight; no GPU, no large data uploads.

Phase 2 step 4 replaced register.py's stub with a real
MLClient.models.create_or_update call. After this job completes successfully,
the workspace's Models tab will show a new cv4_root_segmentation entry.

Usage:
    uv run --group azure python scripts/submit_register_test.py

Before running, place these files locally:
    weights/eval_input/model.h5    -- the trained model used in this test
    metrics/metrics.json           -- evaluate.py's metrics output

For the standalone test, the simplest source for metrics.json is to copy
it from a successful evaluate job's Outputs tab in the studio.
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
LOCAL_MODEL_DIR = REPO_ROOT / "weights" / "eval_input"
LOCAL_METRICS_DIR = REPO_ROOT / "metrics"

# Pin to cv4-cpu:3 which adds azure-ai-ml + azure-identity to the env.
# Earlier versions lack these packages, so register.py's real registration
# branch would fail on import.
CV4_CPU_ENV = "cv4-cpu:3"


def main() -> None:
    """Build and submit a single-node register job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not CODE_PATH.exists():
        raise FileNotFoundError(f"Source folder not found: {CODE_PATH}")

    model_h5 = LOCAL_MODEL_DIR / "model.h5"
    if not model_h5.exists():
        raise FileNotFoundError(f"Model not found: {model_h5}. Re-use the file from the evaluate test.")

    metrics_json = LOCAL_METRICS_DIR / "metrics.json"
    if not metrics_json.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_json}. Download metrics.json from "
            "a successful evaluate job's Outputs tab into a local metrics/ folder."
        )

    logger.info("Code path: %s", CODE_PATH)
    logger.info("Local model folder: %s", LOCAL_MODEL_DIR)
    logger.info("Local metrics folder: %s", LOCAL_METRICS_DIR)
    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    # Register is CPU-only and small; cv4-cpu env is plenty.
    job = command(
        code=str(CODE_PATH),
        command=(
            "python -m cv4.training.register "
            "--metrics-path ${{inputs.eval_metrics}}/metrics.json "
            "--model-path ${{inputs.trained_model}}/model.h5 "
            "--decision-output-path ${{outputs.decision}}/decision.json "
            "--model-name ${{inputs.model_name}} "
            "--f1-threshold ${{inputs.f1_threshold}}"
        ),
        inputs={
            "trained_model": Input(type=AssetTypes.URI_FOLDER, path=str(LOCAL_MODEL_DIR)),
            "eval_metrics": Input(type=AssetTypes.URI_FOLDER, path=str(LOCAL_METRICS_DIR)),
            "model_name": "cv4_root_segmentation",
            "f1_threshold": 0.80,
        },
        outputs={
            "decision": Output(type=AssetTypes.URI_FOLDER),
        },
        environment=CV4_CPU_ENV,
        compute=AZURE_COMPUTE,
        instance_type=AZURE_CPU_INSTANCE,
        experiment_name="cv4-register-tests",
        display_name="register-component-standalone",
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
