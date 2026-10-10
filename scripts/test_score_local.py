"""
Local test for score.py — simulates how Azure ML calls init() and run()
without deploying to the cloud.

What it does:
  1. Creates a temp folder with the model copied in as model.h5 (the name
     Azure uses), and points AZUREML_MODEL_DIR at it.
  2. Calls init() to load the model and lane edges.
  3. Base64-encodes a test image, builds the JSON request, and calls run().
  4. Prints the parsed response.

Usage (from repo root):
    uv run python scripts/test_score_local.py --image path/to/test_plant.png
"""

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = REPO_ROOT / "weights" / "unet_256px.h5"
DEPLOYMENT_DIR = REPO_ROOT / "src" / "cv4" / "deployment"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test score.py locally.")
    parser.add_argument("--image", required=True, help="Path to a test plant image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: test image not found: {image_path}")
        sys.exit(1)

    if not LOCAL_MODEL.exists():
        print(f"ERROR: model not found: {LOCAL_MODEL}")
        sys.exit(1)

    # Azure mounts the registered model in a folder as model.h5. Recreate that
    # locally: a temp folder with the model copied in under the name model.h5.
    tmp_model_dir = tempfile.mkdtemp(prefix="azureml_model_")
    shutil.copy2(LOCAL_MODEL, os.path.join(tmp_model_dir, "model.h5"))
    os.environ["AZUREML_MODEL_DIR"] = tmp_model_dir
    print(f"AZUREML_MODEL_DIR set to: {tmp_model_dir}")

    # Import score AFTER setting the env var, since init() reads it.
    sys.path.insert(0, str(DEPLOYMENT_DIR))
    import score

    print("\nCalling init()...")
    score.init()
    print("init() done.\n")

    # Build the request body the way the backend would: base64 image in JSON.
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    raw_data = json.dumps({"image": image_b64})

    print("Calling run()...")
    response = score.run(raw_data)
    print("run() done.\n")

    # run() returns a JSON string; parse and pretty-print it.
    parsed = json.loads(response)
    print("Response:")
    print(json.dumps(parsed, indent=2))

    # Clean up the temp model dir.
    shutil.rmtree(tmp_model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
