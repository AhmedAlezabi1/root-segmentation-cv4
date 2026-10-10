"""
Measure latency and throughput of the live cv4 online endpoint.

Sends the same image to the endpoint several times and reports:
  - per-request latency (round-trip time for each call)
  - average / min / max latency
  - throughput (successful requests per minute)

Latency is measured sequentially (one request after another), which reflects
single-client, single-instance performance. This is the headline number for
the deployment task. Real throughput under autoscaling would be higher, since
up to MAX_INSTANCES handle requests in parallel.

The endpoint URI and key are fetched from the workspace with --from-workspace,
or read from CV4_ENDPOINT_URL / CV4_ENDPOINT_KEY env vars.

Usage (from repo root):
    uv run --group azure python scripts/measure_endpoint.py --image plant.png --requests 5 --from-workspace
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
    parser = argparse.ArgumentParser(description="Measure endpoint latency and throughput.")
    parser.add_argument("--image", required=True, help="Path to a test plant image")
    parser.add_argument("--requests", type=int, default=5, help="Number of requests to send")
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

    if args.from_workspace:
        url, key = get_uri_and_key_from_workspace()
    else:
        url = os.environ.get("CV4_ENDPOINT_URL")
        key = os.environ.get("CV4_ENDPOINT_KEY")
        if not url or not key:
            print("ERROR: set CV4_ENDPOINT_URL and CV4_ENDPOINT_KEY, or pass --from-workspace.")
            sys.exit(1)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = json.dumps({"image": image_b64})
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    print(f"Endpoint: {url}")
    print(f"Image:    {image_path.name}")
    print(f"Sending {args.requests} sequential requests...\n")

    latencies = []
    failures = 0
    wall_start = time.time()

    for i in range(args.requests):
        start = time.time()
        try:
            resp = requests.post(url, data=payload, headers=headers, timeout=300)
            elapsed = time.time() - start
            if resp.status_code == 200:
                latencies.append(elapsed)
                print(f"  Request {i + 1}: {resp.status_code}  {elapsed:.2f}s")
            else:
                failures += 1
                print(f"  Request {i + 1}: {resp.status_code}  FAILED  {elapsed:.2f}s")
        except requests.RequestException as exc:
            failures += 1
            print(f"  Request {i + 1}: ERROR  {exc}")

    wall_total = time.time() - wall_start

    print("\n── Results ───────────────────────────────")
    print(f"Successful requests : {len(latencies)} / {args.requests}")
    print(f"Failed requests     : {failures}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"Average latency     : {avg:.2f}s")
        print(f"Min latency         : {min(latencies):.2f}s")
        print(f"Max latency         : {max(latencies):.2f}s")
        # Throughput: successful requests over total wall-clock time.
        throughput_per_min = len(latencies) / wall_total * 60
        print(f"Total wall time     : {wall_total:.2f}s")
        print(f"Throughput          : {throughput_per_min:.1f} requests/min (sequential, 1 client)")


if __name__ == "__main__":
    main()
