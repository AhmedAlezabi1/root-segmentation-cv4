"""
Register local data folders as versioned Data Assets in Azure ML.

Usage (sample data):
    uv run --group azure python scripts/register_data_assets.py --base-path D:/myfile/Y2-D/sample_dataset

Usage (full data):
    uv run --group azure python scripts/register_data_assets.py --base-path D:/myfile/Y2-D/Data

Dry run (shows what would be uploaded without uploading):
    uv run --group azure python scripts/register_data_assets.py --base-path D:/myfile/Y2-D/sample_dataset --dry-run

This script uploads two Data Assets:
  - cv4-raw-train: contains train_images/ and train_masks/ together
  - cv4-raw-val:   contains val_images/ and val_masks/ together

Keeping images and masks in the same asset prevents version skew (uploading
new images but forgetting to bump masks). The folder structure inside each
asset is preserved, so Keras flow_from_directory continues to work.
"""

import argparse
import logging
import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

from cv4.config import (
    AZURE_RESOURCE_GROUP,
    AZURE_SUBSCRIPTION_ID,
    AZURE_WORKSPACE_NAME,
)

logger = logging.getLogger(__name__)

# Each split is uploaded as one asset containing both images and masks.
# The base-path should contain train/ and val/ subfolders, each holding
# {split}_images/ and {split}_masks/ directories.
DATASETS = {
    "train": "cv4-raw-train",
    "val": "cv4-raw-val",
}

# Expected subfolders inside each split folder
REQUIRED_SUBFOLDERS = {
    "train": ["train_images", "train_masks"],
    "val": ["val_images", "val_masks"],
}


def count_files(folder: Path) -> int:
    """Count all files in a folder recursively."""
    return sum(1 for f in folder.rglob("*") if f.is_file())


def main() -> None:
    """Parse CLI arguments, connect to Azure ML, and upload each split folder
    as a versioned Data Asset.

    Each asset contains both images and masks for one split, preventing
    version skew between the two. The script validates that all expected
    subfolders exist before uploading and exits with code 1 if any upload
    fails.

    Use ``--dry-run`` to preview what would be uploaded without making changes.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Upload data to Azure ML as versioned Data Assets.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Path to the folder containing train/ and val/ subfolders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)

    if not base_path.exists():
        logger.error("Path does not exist: %s", base_path)
        sys.exit(1)

    # Validate folder structure
    for split_name, subfolders in REQUIRED_SUBFOLDERS.items():
        split_path = base_path / split_name
        if not split_path.exists():
            logger.error("Missing split folder: %s", split_path)
            sys.exit(1)
        for sub in subfolders:
            if not (split_path / sub).exists():
                logger.error("Missing subfolder: %s", split_path / sub)
                sys.exit(1)

    if args.dry_run:
        logger.info("=== DRY RUN — nothing will be uploaded ===")
    else:
        logger.info("Connecting to Azure ML workspace...")

    client = None
    if not args.dry_run:
        client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=AZURE_SUBSCRIPTION_ID,
            resource_group_name=AZURE_RESOURCE_GROUP,
            workspace_name=AZURE_WORKSPACE_NAME,
        )
        logger.info("Connected to workspace: %s", AZURE_WORKSPACE_NAME)

    has_failures = False

    for split_name, asset_name in DATASETS.items():
        folder_path = base_path / split_name
        file_count = count_files(folder_path)

        logger.info(
            "Folder: %s  |  Files: %d  |  Asset: %s",
            folder_path,
            file_count,
            asset_name,
        )

        if args.dry_run:
            logger.info("[DRY RUN] Would upload as '%s'", asset_name)
            continue

        try:
            data_asset = Data(
                name=asset_name,
                description=f"{split_name} split (images + masks) for cv4 plant root segmentation",
                path=str(folder_path),
                type=AssetTypes.URI_FOLDER,
            )

            registered = client.data.create_or_update(data_asset)
            logger.info("Uploaded: %s  version %s", registered.name, registered.version)

        except Exception as e:
            logger.exception("Failed to upload %s: %s", asset_name, e)
            has_failures = True

    if has_failures:
        logger.error("Some uploads failed — see errors above.")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
