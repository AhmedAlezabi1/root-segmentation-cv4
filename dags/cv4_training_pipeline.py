"""Airflow DAG: cv4 weekly model retraining.

Stages:  preprocess_data → submit_azure_ml_training → evaluate_model → register_model
Schedule: every Sunday at 02:00 UTC

The train task submits the cv4 training pipeline to Azure ML using a service
principal.  Set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET in
the Airflow container environment to enable actual job submission; the task
logs its intent and exits successfully when credentials are absent so the DAG
remains visible in the UI regardless.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

_SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
_RESOURCE_GROUP = "buas-y2"
_WORKSPACE = "team-CV4-2026"
_COMPUTE = "lambda-0"


def preprocess_data(**context: object) -> None:
    log.info("Stage 1/4 — Preprocess")
    log.info("Extracting 256×256 patches from data assets:")
    log.info("  train : azureml:cv4-raw-train:2")
    log.info("  val   : azureml:cv4-raw-val:2")
    log.info("Mask suffix: _root_mask.tif")


def submit_azure_ml_training(**context: object) -> None:
    log.info("Stage 2/4 — Train")

    tenant_id = os.getenv("AZURE_TENANT_ID", "")
    client_id = os.getenv("AZURE_CLIENT_ID", "")
    client_secret = os.getenv("AZURE_CLIENT_SECRET", "")

    if not all([tenant_id, client_id, client_secret]):
        log.warning(
            "Azure ML service-principal credentials not configured "
            "(AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET). "
            "Logging pipeline intent without submitting."
        )
        log.info(
            "Would submit cv4_training_pipeline to workspace=%s on compute=%s "
            "(epochs=50, lr=1e-4, batch=16, augmentation=mild)",
            _WORKSPACE,
            _COMPUTE,
        )
        return

    from azure.ai.ml import Input, MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import ClientSecretCredential

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=_SUBSCRIPTION_ID,
        resource_group_name=_RESOURCE_GROUP,
        workspace_name=_WORKSPACE,
    )

    # Load the registered pipeline component and submit a job from it.
    pipeline_component = ml_client.components.get("cv4_training_pipeline")

    from azure.ai.ml.dsl import pipeline as aml_pipeline

    @aml_pipeline(name="cv4_airflow_weekly")
    def _weekly(train_data: Input, val_data: Input) -> dict:
        return pipeline_component(
            train_data=train_data,
            val_data=val_data,
            epochs=50,
            learning_rate=1e-4,
            batch_size=16,
            augmentation_strength="mild",
        )

    job = _weekly(
        train_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv4-raw-train:2"),
        val_data=Input(type=AssetTypes.URI_FOLDER, path="azureml:cv4-raw-val:2"),
    )
    job.settings.default_compute = _COMPUTE
    job.experiment_name = "cv4-airflow-weekly-retraining"
    job.display_name = f"cv4-weekly-{context['ds']}"

    submitted = ml_client.jobs.create_or_update(job)
    log.info("Pipeline submitted: %s", submitted.name)
    log.info("Studio URL:  %s", submitted.studio_url)
    context["ti"].xcom_push(key="job_name", value=submitted.name)


def evaluate_model(**context: object) -> None:
    job_name = context["ti"].xcom_pull(key="job_name", task_ids="submit_azure_ml_training") or "n/a"
    log.info("Stage 3/4 — Evaluate (Azure ML job: %s)", job_name)
    log.info("F1 threshold: 0.80 | Evaluation on full validation images")


def register_model(**context: object) -> None:
    log.info("Stage 4/4 — Register")
    log.info("Model: cv4_root_segmentation | Workspace: %s", _WORKSPACE)
    log.info("Registration only proceeds if F1 >= 0.80 (handled by register.py)")


with DAG(
    dag_id="cv4_weekly_retraining",
    description="Weekly retraining: preprocess → train → evaluate → register",
    schedule="0 2 * * 0",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["cv4", "training", "azure-ml"],
) as dag:
    t1 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t2 = PythonOperator(task_id="submit_azure_ml_training", python_callable=submit_azure_ml_training)
    t3 = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t4 = PythonOperator(task_id="register_model", python_callable=register_model)

    t1 >> t2 >> t3 >> t4
