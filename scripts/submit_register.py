"""Register a model via the register component, given pre-computed metrics.

Standalone register step decoupled from evaluate. The evaluate job already
produced a metrics.json (with the f1 key) and Azure registered it as a data
asset; this script points the register component at that metrics asset and
the model asset, gating registration on the F1 threshold. Splitting register
out avoids passing job-output URIs as job inputs, which Azure rejects.

Usage:
    uv run --group azure python scripts/submit_register.py \\
        --model-uri azureml:azureml_..._trained_model:1 \\
        --metrics-uri azureml:azureml_..._eval_metrics:1
"""

import argparse
import logging
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import JobResourceConfiguration
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_CPU_INSTANCE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
MODEL_NAME = "cv4_root_segmentation"
F1_THRESHOLD = 0.80


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Conditionally register a model given its evaluation metrics.")
    parser.add_argument(
        "--model-uri",
        required=True,
        help="Registered asset URI of the model's trained_model folder (azureml:<name>:<version>).",
    )
    parser.add_argument(
        "--metrics-uri",
        required=True,
        help="Registered asset URI of the evaluate eval_metrics folder (azureml:<name>:<version>).",
    )
    return parser.parse_args()


def main() -> None:
    """Submit the register component as a standalone job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    logger.info("Connecting to Azure ML workspace: %s", AZURE_WORKSPACE_NAME)
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    register_job = command(
        code=str(SRC_DIR),
        command=(
            "python -m cv4.training.register "
            "--metrics-path ${{inputs.eval_metrics}}/metrics.json "
            "--model-path ${{inputs.trained_model}}/model.h5 "
            "--decision-output-path ${{outputs.decision}}/decision.json "
            "--model-name ${{inputs.model_name}} "
            "--f1-threshold ${{inputs.f1_threshold}}"
        ),
        inputs={
            "trained_model": Input(type=AssetTypes.URI_FOLDER, path=args.model_uri),
            "eval_metrics": Input(type=AssetTypes.URI_FOLDER, path=args.metrics_uri),
            "model_name": MODEL_NAME,
            "f1_threshold": F1_THRESHOLD,
        },
        outputs={"decision": Output(type=AssetTypes.URI_FOLDER)},
        environment="azureml:cv4-cpu:3",
        compute=AZURE_COMPUTE,
        resources=JobResourceConfiguration(instance_type=AZURE_CPU_INSTANCE),
        experiment_name="cv4-sweep-register",
        display_name="register-cv4-sweep-full_7",
    )

    logger.info("Submitting register job...")
    submitted = client.jobs.create_or_update(register_job)
    logger.info("Studio URL: %s", submitted.studio_url)
    client.jobs.stream(submitted.name)

    print()
    print(f"Register job : {submitted.name}")
    print("Check the decision output for should_register and the new version.")
    print()


if __name__ == "__main__":
    main()
