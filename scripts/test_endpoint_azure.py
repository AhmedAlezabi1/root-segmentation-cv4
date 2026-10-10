"""
Send a real image to the live cv4 online endpoint and print the prediction.

This is the end-to-end test for the deployed endpoint: it base64-encodes a
local plant image, posts it to the scoring URI with the endpoint key, and
prints the per-lane root predictions returned by the model in the cloud.

The endpoint URI and key are read from environment variables so the key is
never hard-coded or committed:

    set CV4_ENDPOINT_URL=http://.../score          (Windows)
    set CV4_ENDPOINT_KEY=your-primary-key

    export CV4_ENDPOINT_URL=http://.../score       (mac/linux)
    export CV4_ENDPOINT_KEY=your-primary-key

Usage (from repo root):
    uv run python scripts/test_endpoint.py --image path/to/plant.png

Alternatively the script can fetch the URI and key directly from the
workspace with --from-workspace (requires az login + the azure group):
    uv run --group azure python scripts/test_endpoint.py --image plant.png --from-workspace
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

ENDPOINT_NAME = "cv4-root-seg"
DEPLOYMENT_NAME = "blue"


def get_uri_and_key_from_workspace() -> tuple[str, str]:
    """Fetch the scoring URI and primary key from the workspace via the SDK."""
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    from cv4.config import (
        AZURE_RESOURCE_GROUP,
        AZURE_SUBSCRIPTION_ID,
        AZURE_WORKSPACE_NAME,
    )

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )
    endpoint = client.online_endpoints.get(name=ENDPOINT_NAME)
    keys = client.online_endpoints.get_keys(name=ENDPOINT_NAME)
    return endpoint.scoring_uri, keys.primary_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the live cv4 endpoint.")
    parser.add_argument("--image", required=True, help="Path to a test plant image")
    parser.add_argument(
        "--from-workspace",
        action="store_true",
        help="Fetch URI and key from the workspace instead of env vars",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    # Resolve URI and key.
    if args.from_workspace:
        url, key = get_uri_and_key_from_workspace()
    else:
        url = os.environ.get("CV4_ENDPOINT_URL")
        key = os.environ.get("CV4_ENDPOINT_KEY")
        if not url or not key:
            print("ERROR: set CV4_ENDPOINT_URL and CV4_ENDPOINT_KEY env vars, or pass --from-workspace.")
            sys.exit(1)

    # Build the request body: base64 image inside JSON.
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = json.dumps({"image": image_b64})

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    print(f"POST {url}")
    print(f"Image: {image_path.name} ({len(image_b64)} base64 chars)")
    print("Sending request (first call may be slow while the worker warms up)...\n")

    start = time.time()
    response = requests.post(url, data=payload, headers=headers, timeout=300)
    elapsed = time.time() - start

    print(f"HTTP {response.status_code}  (round-trip {elapsed:.1f}s)\n")

    if response.status_code != 200:
        print("Request failed. Response body:")
        print(response.text)
        sys.exit(1)

    # The endpoint returns a JSON string; it may be double-encoded depending on
    # how the inference server wraps score.py's return value, so handle both.
    body = response.json()
    if isinstance(body, str):
        body = json.loads(body)

    print("Prediction:")
    print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
