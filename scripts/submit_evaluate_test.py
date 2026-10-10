"""Submit a standalone evaluate job to validate cv4.training.evaluate.

Phase 2 step 3 (evaluate validation): re-uses the cloud-context dispatch in
evaluate.py to run patch-level metric computation and overlay generation
as a single-node job. Inputs are a previously-trained model (passed as a
local model.h5 file uploaded as a job-scoped folder) and the patched
dataset.

Like the other component tests, we build the job with command() directly
rather than load_component(...) because the SDK's component-mutation flow
on Windows drops bindings that should be sticky. The YAML at
components/evaluate.yml remains the canonical declaration for pipeline
assembly in step 5.

Usage:
    uv run --group azure python scripts/submit_evaluate_test.py

Before running, place a trained model at weights/model.h5 in the repo root.
The smoke-test train runs from Phase 2 step 1 produce a suitable model;
download it from the studio's Outputs tab and save it locally.
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
LOCAL_MODEL_DIR = REPO_ROOT / "weights" / "eval_input"


def main() -> None:
    """Build and submit a single-node evaluate job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not CODE_PATH.exists():
        raise FileNotFoundError(f"Source folder not found: {CODE_PATH}")
    if not LOCAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Patched data folder not found: {LOCAL_DATA_PATH}.")
    model_h5 = LOCAL_MODEL_DIR / "model.h5"
    if not model_h5.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_h5}. Download model.h5 from a successful "
            "train job's Outputs tab in the studio and place it under weights/."
        )

    logger.info("Code path: %s", CODE_PATH)
    logger.info("Local patched data: %s", LOCAL_DATA_PATH)
    logger.info("Local model folder: %s", LOCAL_MODEL_DIR)
    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    # Evaluate runs on GPU to keep prediction throughput reasonable; the val
    # set is small (~1200 patches) but the .h5 model loads onto GPU memory by
    # default and overlay generation also uses GPU inference via
    # predict_smooth_windowing.
    job = command(
        code=str(CODE_PATH),
        command=(
            "python -m cv4.training.evaluate "
            "--model-path ${{inputs.trained_model}}/model.h5 "
            "--data-path ${{inputs.patched_data}} "
            "--metrics-output-path ${{outputs.eval_metrics}}/metrics.json "
            "--overlays-output-dir ${{outputs.overlays}} "
            "--batch-size ${{inputs.batch_size}} "
            "--patch-size ${{inputs.patch_size}}"
        ),
        inputs={
            "trained_model": Input(type=AssetTypes.URI_FOLDER, path=str(LOCAL_MODEL_DIR)),
            "patched_data": Input(type=AssetTypes.URI_FOLDER, path=str(LOCAL_DATA_PATH)),
            "batch_size": 32,
            "patch_size": 256,
        },
        outputs={
            "eval_metrics": Output(type=AssetTypes.URI_FOLDER),
            "overlays": Output(type=AssetTypes.URI_FOLDER),
        },
        environment="cv4-training:6",
        compute=AZURE_COMPUTE,
        instance_type=AZURE_GPU_INSTANCE,
        experiment_name="cv4-evaluate-tests",
        display_name="evaluate-component-standalone",
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
