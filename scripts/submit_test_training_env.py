"""
Submit test_training_env.py as an Azure ML job to verify the cv4-training
environment can import TensorFlow and see GPU devices.

Run from the repo root:

    uv run --group azure python scripts/submit_test_training_env.py

The job runs on the lambda-0 Kubernetes cluster, which is the shared compute
for this hub-based Azure AI Foundry workspace (no persistent clusters needed).
"""

from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_GPU_INSTANCE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


def main() -> None:
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    job = command(
        display_name="cv4-test-training-env",
        description="Verify cv4-training environment: TensorFlow import + GPU check.",
        code=str(SCRIPTS_DIR),
        command="python test_training_env.py",
        environment="cv4-training@latest",
        compute=AZURE_COMPUTE,
        resources={"instance_type": AZURE_GPU_INSTANCE},
    )

    submitted = client.jobs.create_or_update(job)
    print(f"Job submitted : {submitted.name}")
    print(f"Status        : {submitted.status}")
    print(f"Studio URL    : {submitted.studio_url}")


if __name__ == "__main__":
    main()
