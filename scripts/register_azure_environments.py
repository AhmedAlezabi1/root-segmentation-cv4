"""
Register the cv4 training and CPU environments in the Azure ML workspace.

Run from the repo root:

    uv run --group azure python scripts/register_azure_environments.py

Each environment is built from a conda YAML file in configs/ and a
CUDA-enabled base Docker image (training) or a plain Python image (CPU).
Azure ML validates and builds the Docker image on first use — this script
only registers the definition; the build happens when a job first runs.

Environments are versioned automatically: re-running this script with
changed YAML files registers a new version, the old one is preserved.
"""

from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

from cv4.config import AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID, AZURE_WORKSPACE_NAME

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

ENVIRONMENTS = [
    Environment(
        name="cv4-training",
        description=(
            "TensorFlow 2.18 GPU training environment. CUDA installed via the tensorflow[and-cuda] pip extra — no CUDA base image required."
        ),
        conda_file=str(CONFIGS_DIR / "env-training.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    ),
    Environment(
        name="cv4-cpu",
        description="Lightweight CPU-only environment for preprocessing and evaluation steps.",
        conda_file=str(CONFIGS_DIR / "env-cpu.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
    ),
]


def main() -> None:
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    for env in ENVIRONMENTS:
        registered = client.environments.create_or_update(env)
        print(f"Registered: {registered.name}  version={registered.version}")


if __name__ == "__main__":
    main()
