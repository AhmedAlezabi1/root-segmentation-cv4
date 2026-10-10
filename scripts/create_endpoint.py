"""
Create the cv4 Kubernetes online endpoint and deploy the registered model.

This is Task 2 of the deployment work: it creates an online endpoint on the
lambda-0 Kubernetes cluster, attaches a deployment ("blue") that pairs the
registered model (cv4_root_segmentation) with the score.py scoring script,
and routes 100% of traffic to it.

We use KubernetesOnlineEndpoint (not ManagedOnlineEndpoint) because this is a
hub-based Azure AI Foundry workspace whose compute is the shared lambda-0
Kubernetes cluster. The concept is identical to a managed endpoint — an HTTPS
URL that serves the model — it just runs on the team's Kubernetes compute.

Run from the repo root:
    uv run --group azure python scripts/create_endpoint.py

Prerequisites:
    az login
    uv sync --group azure
"""

import logging

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    KubernetesOnlineDeployment,
    KubernetesOnlineEndpoint,
    OnlineRequestSettings,
    TargetUtilizationScaleSettings,
)
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_COMPUTE,
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

# ── Deployment configuration ──────────────────────────────
# Committed here so the deployment is reproducible from the repo.
ENDPOINT_NAME = "cv4-root-seg"
DEPLOYMENT_NAME = "blue"

MODEL_NAME = "cv4_root_segmentation"
MODEL_VERSION = "2"  # the full-data model registered by the training pipeline

# Use cv4-inference: TensorFlow + CV packages + azureml-inference-server-http.
# cv4-training lacks the inference server package, so the endpoint container
# crashed (CrashLoopBackOff). cv4-inference is built specifically for serving.
# v2 uses tensorflow[and-cuda] so TF can use the GPU; v1's plain TF ran on CPU
# even on the GPU node, making the pipeline too slow and causing 504 timeouts.
ENVIRONMENT_NAME = "cv4-inference"
ENVIRONMENT_VERSION = "2"

# Ship the whole src/ folder so the cv4 package is importable inside the
# endpoint container. The scoring script path is relative to that code root.
# (Shipping only the deployment/ folder fails: score.py imports cv4.*, which
# wouldn't be on the container's Python path.)
CODE_DIR = "src"
SCORING_SCRIPT = "cv4/deployment/score.py"

# Kubernetes instance type from the cluster. GPU is used because the pipeline
# (patch-based UNet inference + classical CV) is slow on CPU (~75s), which
# exceeds the gateway timeout. GPU brings it well under the timeout.
INSTANCE_TYPE = "gpu"

# Per-request timeout. The pipeline is heavy, so the default (~5s) is far too
# short and returns 504 Gateway Timeout before scoring finishes. 90s gives
# plenty of headroom even on a cold worker.
REQUEST_TIMEOUT_MS = 90000

# Autoscaling: keep at least 1 instance always running (so the endpoint is
# never cold), and scale up to 3 when busy. Azure adds/removes instances to
# keep average CPU around the target utilisation percentage.
MIN_INSTANCES = 1
MAX_INSTANCES = 3
TARGET_UTILIZATION_PERCENT = 70


def main() -> None:
    """Create the endpoint, deploy model v1, and route traffic to it."""
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

    # 1. Create (or update) the endpoint. Key-based auth means callers must
    #    pass an endpoint key; the key lives only on the backend.
    logger.info("Creating endpoint '%s' on compute '%s'...", ENDPOINT_NAME, AZURE_COMPUTE)
    endpoint = KubernetesOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="cv4 plant root segmentation inference endpoint.",
        auth_mode="key",
        compute=AZURE_COMPUTE,
    )
    client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Endpoint ready.")

    # 2. Look up the registered model and environment to deploy.
    model = client.models.get(name=MODEL_NAME, version=MODEL_VERSION)
    logger.info("Using model %s version %s", model.name, model.version)

    env = client.environments.get(name=ENVIRONMENT_NAME, version=ENVIRONMENT_VERSION)
    logger.info("Using environment %s version %s", env.name, env.version)

    # 3. Define the deployment: model + environment + scoring script.
    #    Autoscaling keeps min 1 / max 3 instances based on CPU utilisation.
    logger.info("Creating deployment '%s'...", DEPLOYMENT_NAME)
    deployment = KubernetesOnlineDeployment(
        name=DEPLOYMENT_NAME,
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
    client.online_deployments.begin_create_or_update(deployment).result()
    logger.info("Deployment '%s' is live.", DEPLOYMENT_NAME)

    # 4. Route 100% of traffic to this deployment.
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Traffic: 100%% -> %s", DEPLOYMENT_NAME)

    # 5. Print the endpoint details for testing.
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    keys = client.online_endpoints.get_keys(name=ENDPOINT_NAME)
    print()
    print(f"Endpoint name  : {endpoint.name}")
    print(f"Scoring URI    : {endpoint.scoring_uri}")
    print(f"Auth mode      : {endpoint.auth_mode}")
    print(f"Primary key    : {keys.primary_key}")
    print()
    print("Test it with scripts/test_endpoint_azure.py")


if __name__ == "__main__":
    main()
