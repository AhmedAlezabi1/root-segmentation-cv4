"""MLflow tracking configuration.

Centralizes the experiment-name and tracking-URI configuration so train.py
stays focused on training mechanics. When the training component runs inside
an Azure ML job, the tracking URI is set automatically by Azure ML's MLflow
bridge and CV4_MLFLOW_TRACKING_URI should be left unset; locally we fall back
to a file-store under ./mlruns so runs persist between invocations.
"""

import logging
import os

import mlflow

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "cv4_training_experiments"


def in_azure_ml_job() -> bool:
    """Return True when the current process is running inside an Azure ML job.

    Azure ML sets AZUREML_RUN_ID to the job name on the cluster, and never sets
    it during local execution. This is the most reliable signal documented by
    Microsoft for detecting cloud vs local context.
    """
    return "AZUREML_RUN_ID" in os.environ


def configure_mlflow(experiment_name: str | None = None) -> str | None:
    """Configure MLflow tracking for the current process.

    Inside an Azure ML job, MLflow's tracking URI and the active experiment
    are pre-set by the platform so that the auto-created run is bound to the
    job in the studio UI. Calling set_experiment() in that context creates a
    mismatch between the platform's run binding and ours, which makes
    mlflow.start_run() refuse to open a new run. So inside a job we leave
    both alone and return None to signal "platform-managed".

    Locally, we honor the explicit argument, then CV4_MLFLOW_EXPERIMENT, then
    fall back to the package default. Tracking URI honors CV4_MLFLOW_TRACKING_URI
    if set, otherwise falls back to MLflow's local file store.

    Args:
        experiment_name: Optional explicit experiment name. Ignored when
            running inside an Azure ML job. When None locally, falls back to
            the env var or default.

    Returns:
        The experiment name that ends up active for this process, or None
        when running inside an Azure ML job (platform managed).
    """
    if in_azure_ml_job():
        logger.info("MLflow tracking and experiment managed by Azure ML; configure_mlflow is a no-op.")
        return None

    tracking_uri = os.environ.get("CV4_MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
    else:
        logger.info("MLflow tracking URI: default file store (./mlruns)")

    resolved = experiment_name or os.environ.get("CV4_MLFLOW_EXPERIMENT") or DEFAULT_EXPERIMENT_NAME
    mlflow.set_experiment(resolved)
    logger.info("MLflow experiment: %s", resolved)
    return resolved
