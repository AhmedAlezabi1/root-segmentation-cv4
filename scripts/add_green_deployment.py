"""
Add a second deployment ("green") and split traffic with "blue".

This demonstrates the versioned-deployment requirement: two model versions can
coexist behind one endpoint, and traffic can be split between them. It is the
foundation of a blue/green or canary rollout.

  - blue  serves model version 2 (the full-data model, the current production one)
  - green serves model version 1 (the sample model) to show a *different*
          version running side by side

After both deployments are live, traffic is split (default 80% blue / 20% green),
which is the canary pattern: most traffic stays on the proven version while a
small slice goes to the new one for validation.

This script assumes the endpoint and the "blue" deployment already exist
(created by scripts/create_endpoint.py). Run that first.

Run from the repo root:
    uv run --group azure python scripts/add_green_deployment.py

Prerequisites:
    az login
    uv sync --group azure
"""

import logging

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    KubernetesOnlineDeployment,
    OnlineRequestSettings,
    TargetUtilizationScaleSettings,
)
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

# Must match the endpoint created by create_endpoint.py.
ENDPOINT_NAME = "cv4-root-seg"

# The new deployment.
GREEN_NAME = "green"
GREEN_MODEL_VERSION = "1"  # sample model, a *different* version from blue's v2

MODEL_NAME = "cv4_root_segmentation"
ENVIRONMENT_NAME = "cv4-inference"
ENVIRONMENT_VERSION = "2"
CODE_DIR = "src"
SCORING_SCRIPT = "cv4/deployment/score.py"
INSTANCE_TYPE = "gpu"
REQUEST_TIMEOUT_MS = 90000
MIN_INSTANCES = 1
MAX_INSTANCES = 3
TARGET_UTILIZATION_PERCENT = 70

# Traffic split: most traffic stays on the proven blue, a slice goes to green.
BLUE_TRAFFIC = 80
GREEN_TRAFFIC = 20


def main() -> None:
    """Create the green deployment and split traffic with blue."""
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
    logger.info("Connected to workspace: %s", AZURE_WORKSPACE_NAME)

    # Look up the model version for green and the serving environment.
    model = client.models.get(name=MODEL_NAME, version=GREEN_MODEL_VERSION)
    logger.info("Green will serve model %s version %s", model.name, model.version)

    env = client.environments.get(name=ENVIRONMENT_NAME, version=ENVIRONMENT_VERSION)
    logger.info("Using environment %s version %s", env.name, env.version)

    # Create the green deployment (same scoring code, different model version).
    logger.info("Creating deployment '%s'...", GREEN_NAME)
    green = KubernetesOnlineDeployment(
        name=GREEN_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=CODE_DIR,
            scoring_script=SCORING_SCRIPT,
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=MIN_INSTANCES,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=REQUEST_TIMEOUT_MS,
        ),
        scale_settings=TargetUtilizationScaleSettings(
            min_instances=MIN_INSTANCES,
            max_instances=MAX_INSTANCES,
            target_utilization_percentage=TARGET_UTILIZATION_PERCENT,
        ),
    )
    client.online_deployments.begin_create_or_update(green).result()
    logger.info("Deployment '%s' is live.", GREEN_NAME)

    # Split traffic between the two deployments.
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {"blue": BLUE_TRAFFIC, GREEN_NAME: GREEN_TRAFFIC}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Traffic split: blue=%d%%  green=%d%%", BLUE_TRAFFIC, GREEN_TRAFFIC)

    # Show the final state.
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    print()
    print(f"Endpoint    : {endpoint.name}")
    print(f"Traffic     : {endpoint.traffic}")
    print(f"Scoring URI : {endpoint.scoring_uri}")
    print()
    print("Two model versions now coexist behind one endpoint with split traffic.")


if __name__ == "__main__":
    main()
