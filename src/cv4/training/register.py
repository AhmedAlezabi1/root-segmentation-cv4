"""Conditional model registration component.

Reads the evaluation metrics JSON and decides whether the trained model
should be promoted to the model registry. A model is registered only if its
validation F1 score meets or exceeds the configured threshold. The threshold
acts as a regression guard so that pipelines silently fail closed rather
than promoting a model that has degraded compared to the baseline.

When run inside an Azure ML job, registration is performed via the Azure ML
SDK against the workspace's model registry, using DefaultAzureCredential
which picks up the job's managed identity automatically. When run locally
without Azure access, registration falls back to a logged "would register"
stub so that the component can be tested end to end on a laptop without
Azure credentials.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

from cv4.training.config import REGISTRATION_F1_THRESHOLD
from cv4.training.mlflow_setup import in_azure_ml_job

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Conditionally register a trained cv4 model based on evaluation metrics.")
    parser.add_argument(
        "--metrics-path",
        required=True,
        help="Path to the evaluation metrics JSON produced by evaluate.py.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained .h5 model that would be registered.",
    )
    parser.add_argument(
        "--decision-output-path",
        required=True,
        help="Destination JSON file for the registration decision and metadata.",
    )
    parser.add_argument(
        "--model-name",
        default="cv4_root_segmentation",
        help="Name under which the model would be registered.",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=REGISTRATION_F1_THRESHOLD,
        help="Validation F1 floor for registration. Models at or above this score are registered.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Optional version tag for the training data asset, recorded in metadata.",
    )
    parser.add_argument(
        "--training-run-id",
        default=None,
        help="Optional MLflow run ID of the training run, recorded in metadata.",
    )
    return parser.parse_args()


def load_metrics(metrics_path: str) -> dict:
    """Load and validate the evaluation metrics file.

    Args:
        metrics_path: Path to the JSON file written by evaluate.py.

    Returns:
        The parsed metrics dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is missing the f1 key.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path) as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Metrics file is not valid JSON: {metrics_path}") from e

    if "f1" not in metrics:
        raise ValueError(f"Metrics file is missing required 'f1' key: {metrics_path}. Found keys: {list(metrics)}")

    return metrics


def build_decision(
    metrics: dict,
    model_path: str,
    model_name: str,
    f1_threshold: float,
    dataset_version: str | None,
    training_run_id: str | None,
) -> dict:
    """Assemble the registration decision and metadata.

    The returned dict is the single source of truth for what would be sent
    to the model registry, and is written to disk as the component output.

    Args:
        metrics: Parsed metrics dict from evaluate.py.
        model_path: Path to the .h5 model.
        model_name: Registry name for the model.
        f1_threshold: F1 floor for registration.
        dataset_version: Optional dataset asset version tag.
        training_run_id: Optional training run ID tag.

    Returns:
        Dict with keys: should_register, reason, model_name, model_path,
        f1_threshold, metrics, registration_date, tags.
    """
    f1 = float(metrics["f1"])
    should_register = f1 >= f1_threshold

    tags = {
        "f1": f"{f1:.4f}",
        "iou": f"{float(metrics.get('iou', 0.0)):.4f}",
        "precision": f"{float(metrics.get('precision', 0.0)):.4f}",
        "recall": f"{float(metrics.get('recall', 0.0)):.4f}",
        "registration_date": datetime.now(timezone.utc).isoformat(),
    }
    if dataset_version is not None:
        tags["dataset_version"] = dataset_version
    if training_run_id is not None:
        tags["training_run_id"] = training_run_id

    return {
        "should_register": should_register,
        "reason": (f"F1 {f1:.4f} >= threshold {f1_threshold:.4f}" if should_register else f"F1 {f1:.4f} < threshold {f1_threshold:.4f}"),
        "model_name": model_name,
        "model_path": model_path,
        "f1_threshold": f1_threshold,
        "metrics": metrics,
        "tags": tags,
    }


def register_to_azureml(decision: dict) -> dict:
    """Register the model with Azure ML, or stub when running locally.

    Inside an Azure ML job, this uses MLClient.models.create_or_update against
    the workspace pointed to by AZUREML_ARM_* environment variables that the
    runtime sets. Auth is handled by DefaultAzureCredential, which inside a
    job picks up the job's managed identity. Locally, falls back to a logged
    stub so the component can be exercised on a laptop without Azure access.

    Args:
        decision: The decision dict from build_decision().

    Returns:
        Dict with at least a 'registration_status' key, and (when registered)
        a 'registered_version' key. The dict is merged into the saved
        decision JSON for traceability.
    """
    model_name = decision["model_name"]
    model_path = decision["model_path"]
    tags = decision["tags"]

    if not in_azure_ml_job():
        logger.info(
            "WOULD REGISTER (local stub): name=%s path=%s tags=%s",
            model_name,
            model_path,
            tags,
        )
        logger.info("Skipping real registration: not running inside an Azure ML job.")
        return {"registration_status": "stub"}

    # Lazy imports so local invocations without azure-ai-ml installed still
    # work via the stub branch above. Inside the cloud job, the cv4-cpu env
    # has both packages.
    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.identity import DefaultAzureCredential

    from cv4.config import AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID, AZURE_WORKSPACE_NAME

    logger.info("Creating MLClient for workspace: %s", AZURE_WORKSPACE_NAME)
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )

    model_entity = Model(
        path=model_path,
        name=model_name,
        description=f"cv4 root segmentation model; F1={tags['f1']}",
        type=AssetTypes.CUSTOM_MODEL,
        tags=tags,
    )

    logger.info("Registering model %s from %s", model_name, model_path)
    registered = client.models.create_or_update(model_entity)
    logger.info(
        "Registered model: name=%s version=%s",
        registered.name,
        registered.version,
    )

    return {
        "registration_status": "registered",
        "registered_version": registered.version,
    }


def main() -> None:
    """Run the conditional registration component."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    metrics = load_metrics(args.metrics_path)
    decision = build_decision(
        metrics=metrics,
        model_path=args.model_path,
        model_name=args.model_name,
        f1_threshold=args.f1_threshold,
        dataset_version=args.dataset_version,
        training_run_id=args.training_run_id,
    )

    if decision["should_register"]:
        logger.info("Registration decision: REGISTER (%s)", decision["reason"])
        registration_result = register_to_azureml(decision)
        decision.update(registration_result)
    else:
        logger.info("Registration decision: SKIP (%s)", decision["reason"])
        decision["registration_status"] = "skipped"

    os.makedirs(os.path.dirname(os.path.abspath(args.decision_output_path)), exist_ok=True)
    with open(args.decision_output_path, "w") as f:
        json.dump(decision, f, indent=2)
    logger.info("Wrote decision to %s", args.decision_output_path)


if __name__ == "__main__":
    main()
