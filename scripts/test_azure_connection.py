"""
Verify that the local machine can authenticate and connect to the team's
Azure ML workspace using the Azure ML SDK v2.

Run once per team member to confirm access is set up correctly:

    uv run --group azure python scripts/test_azure_connection.py

Requires the azure dependency group:

    uv sync --group azure

Authentication uses DefaultAzureCredential, which tries (in order):
  1. Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
  2. Azure CLI login (`az login`)
  3. VS Code / Azure Tools extension login
  4. Interactive browser prompt

The simplest way to authenticate is to run `az login` once beforehand.
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from cv4.config import AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID, AZURE_WORKSPACE_NAME


def main() -> None:
    print(f"Connecting to workspace '{AZURE_WORKSPACE_NAME}'...")

    credential = DefaultAzureCredential()
    client = MLClient(
        credential=credential,
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    ws = client.workspaces.get(AZURE_WORKSPACE_NAME)
    print("Connected successfully.")
    print(f"  Workspace : {ws.name}")
    print(f"  Location  : {ws.location}")
    print(f"  Resource  : {ws.resource_group}")


if __name__ == "__main__":
    main()
