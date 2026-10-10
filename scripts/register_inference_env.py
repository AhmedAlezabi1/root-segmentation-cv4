"""
Register the cv4-inference environment in the Azure ML workspace.

This environment is for serving the model as an online endpoint, not for
training. It contains TensorFlow and the CV packages the cv4 pipeline needs,
plus azureml-inference-server-http — the package the endpoint runtime uses to
receive HTTP requests and call score.py's init() and run(). The cv4-training
environment lacks that package (it was built for training jobs), which is why
the endpoint container crashed with a CrashLoopBackOff.

Run from the repo root:
    uv run --group azure python scripts/register_inference_env.py

Re-running with a changed YAML registers a new version; old versions are kept.
"""

import logging
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def main() -> None:
    """Build and register the cv4-inference environment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    env = Environment(
        name="cv4-inference",
        description=(
            "Inference environment for the cv4 online endpoint: TensorFlow 2.18, the cv4 CV dependencies, and azureml-inference-server-http."
        ),
        conda_file=str(CONFIGS_DIR / "env-inference.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    )

    registered = client.environments.create_or_update(env)
    logger.info("Registered: %s version=%s", registered.name, registered.version)


if __name__ == "__main__":
    main()
