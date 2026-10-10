"""
List all compute targets available in the team Azure ML workspace.

Run from the repo root:

    uv run --group azure python scripts/list_azure_compute.py

Requires az login and the azure dependency group:

    az login
    uv sync --group azure
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from cv4.config import AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID, AZURE_WORKSPACE_NAME


def main() -> None:
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    print(f"Compute targets in workspace '{AZURE_WORKSPACE_NAME}':\n")
    for compute in client.compute.list():
        print(f"  Name : {compute.name}")
        print(f"  Type : {compute.type}")
        if hasattr(compute, "size"):
            print(f"  Size : {compute.size}")
        print()


if __name__ == "__main__":
    main()
