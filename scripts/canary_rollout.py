"""
Canary rollout of a new model version on the cv4 online endpoint.

This is the canary deployment strategy for Issue 4.2. It promotes a new model
version into production gradually, rather than switching all traffic at once,
so a problem with the new version is caught while it is only serving a small
slice of requests.

The endpoint keeps two deployments behind one URL:

  - blue  serves the current production model (version 2)
  - green serves the new candidate model (version 3, the sweep winner)

The rollout walks the traffic split through a series of stages
(90/10 -> 75/25 -> 50/50 -> 25/75 -> 0/100). Between stages it pauses so the
operator can check the endpoint's health with scripts/measure_endpoint.py. If
the new version misbehaves, scripts/rollback_canary.py (or rerunning this with
--rollback) sends all traffic back to blue.

This is distinct from scripts/add_green_deployment.py, which demonstrates a
blue/green split at a fixed 80/20. This script handles the staged shift and
ensures green is on the candidate version (v3) rather than the older sample
model.

Run from the repo root:
    uv run --group azure python scripts/canary_rollout.py

    # advance through every stage automatically, pausing between each
    uv run --group azure python scripts/canary_rollout.py --auto

    # jump straight to a single split (for example 50/50)
    uv run --group azure python scripts/canary_rollout.py --green-traffic 50

    # send all traffic back to blue
    uv run --group azure python scripts/canary_rollout.py --rollback

Prerequisites:
    az login
    uv sync --group azure
    The endpoint and its blue deployment must already exist
    (scripts/create_endpoint.py).
"""

import argparse
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

# Must match the endpoint and blue deployment from create_endpoint.py.
ENDPOINT_NAME = "cv4-root-seg"
BLUE_NAME = "blue"

# Green serves the candidate. v3 is the sweep winner (F1 0.8553), a genuine
# improvement on blue's v2 (F1 0.8488), so this is a real promotion rather
# than a rollout of the older sample model.
GREEN_NAME = "green"
GREEN_MODEL_VERSION = "3"

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

# The percentage of traffic sent to green at each stage of the rollout. Each
# value is a checkpoint: shift, then verify health before moving on.
CANARY_STAGES = [10, 25, 50, 75, 100]


def connect() -> MLClient:
    """Return an MLClient for the cv4 workspace."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )


def ensure_green_deployment(client: MLClient) -> None:
    """Create or update the green deployment so it serves the candidate model.

    Green uses the same scoring code, environment, and hardware as blue, so the
    only difference between the two deployments is the model version. This makes
    the canary a clean comparison: same serving stack, different weights.

    Calling this when green already exists on the right version is safe; Azure
    treats it as an idempotent update.
    """
    model = client.models.get(name=MODEL_NAME, version=GREEN_MODEL_VERSION)
    logger.info("Green will serve model %s version %s", model.name, model.version)

    env = client.environments.get(name=ENVIRONMENT_NAME, version=ENVIRONMENT_VERSION)

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
    logger.info("Ensuring green deployment is live...")
    client.online_deployments.begin_create_or_update(green).result()
    logger.info("Green deployment is live on version %s.", GREEN_MODEL_VERSION)


def set_split(client: MLClient, green_percent: int) -> None:
    """Route green_percent of traffic to green and the remainder to blue."""
    blue_percent = 100 - green_percent
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {BLUE_NAME: blue_percent, GREEN_NAME: green_percent}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Traffic split: blue=%d%%  green=%d%%", blue_percent, green_percent)


def rollback(client: MLClient) -> None:
    """Send all traffic back to blue, the proven version.

    This is the canary safety net: if metrics degrade at any stage, one call
    returns the endpoint to a known-good state without removing green, so the
    rollout can be retried later.
    """
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {BLUE_NAME: 100, GREEN_NAME: 0}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.warning("Rolled back: blue=100%%  green=0%%")


def run_staged_rollout(client: MLClient, auto: bool) -> None:
    """Walk the traffic split through each canary stage.

    After each shift the operator should check endpoint health with
    measure_endpoint.py before continuing. Without --auto the script stops
    after the first stage so the check can happen between manual runs; with
    --auto it pauses for confirmation between stages instead.
    """
    for stage in CANARY_STAGES:
        set_split(client, stage)
        print(f"\nGreen is now at {stage}% of traffic.")
        print("Check health:")
        print("  uv run --group azure python scripts/measure_endpoint.py --image plant.png --requests 10 --from-workspace")

        if stage == 100:
            print("\nGreen now serves all traffic. The rollout is complete.")
            break

        if auto:
            answer = input("\nMetrics look healthy? Continue to the next stage? [y/N] ")
            if answer.strip().lower() != "y":
                print("Stopping the rollout. Run with --rollback to return to blue.")
                return
        else:
            print("\nRun this script again to advance, or with --rollback to undo.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Canary rollout of a new model version.")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Walk through every stage in one run, pausing for confirmation between each",
    )
    parser.add_argument(
        "--green-traffic",
        type=int,
        default=None,
        metavar="PERCENT",
        help="Set a single explicit split (green gets PERCENT, blue gets the rest)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Send all traffic back to blue and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    client = connect()
    logger.info("Connected to workspace: %s", AZURE_WORKSPACE_NAME)

    if args.rollback:
        rollback(client)
        return

    # Green must exist on the candidate version before any traffic reaches it.
    ensure_green_deployment(client)

    if args.green_traffic is not None:
        if not 0 <= args.green_traffic <= 100:
            parser.error("--green-traffic must be between 0 and 100")
        set_split(client, args.green_traffic)
        print(f"\nGreen is at {args.green_traffic}% of traffic.")
        return

    run_staged_rollout(client, auto=args.auto)


if __name__ == "__main__":
    main()
