"""Standalone Azure ML data pipeline for cv4.

Converts the raw image and mask assets into the patched four-folder dataset
that training consumes, as a dedicated data pipeline rather than as a step
inside the training pipeline:

    preprocess -> registered patched data asset

The single preprocess node crops the petri dishes and tiles each pair into
256x256 patches. Its output is given a name and version, which is how Azure ML
registers a pipeline output as a versioned data asset, so each run of this
pipeline produces a reproducible, versioned processed dataset that the training
pipeline and the sweep can point at by id.

This is separate from cv4.training.pipeline, where preprocessing is one step of
the train/evaluate/register DAG. Here the processed dataset itself is the
deliverable.

The same DSL constraint from the training pipeline applies: the component must
be declared at module scope, not inside the @pipeline function, or command()
receives PipelineInput wrappers and rejects them.
"""

import logging
from pathlib import Path

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from cv4.config import AZURE_CPU_INSTANCE

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_PATH = str(REPO_ROOT / "src")

ENV_CPU = "cv4-cpu:3"


_preprocess_op = command(
    name="cv4_preprocess",
    display_name="preprocess",
    code=CODE_PATH,
    command=(
        "python -m cv4.training.preprocess "
        "--train-data-path ${{inputs.train_data}} "
        "--val-data-path ${{inputs.val_data}} "
        "--output-path ${{outputs.patched_data}} "
        "--patch-size ${{inputs.patch_size}} "
        "--mask-suffix ${{inputs.mask_suffix}}"
    ),
    inputs={
        "train_data": Input(type=AssetTypes.URI_FOLDER),
        "val_data": Input(type=AssetTypes.URI_FOLDER),
        "patch_size": Input(type="integer", default=256),
        "mask_suffix": Input(type="string", default="_root_mask.tif"),
    },
    outputs={"patched_data": Output(type=AssetTypes.URI_FOLDER)},
    environment=ENV_CPU,
)


@pipeline(
    name="cv4_data_pipeline",
    display_name="cv4 Data Pipeline (preprocess + register patched asset)",
    description=(
        "Crops and patches the raw petri dish images and masks into the "
        "four-folder patched dataset, and registers the result as a versioned "
        "data asset for training to consume."
    ),
)
def cv4_data_pipeline(
    train_data: Input,
    val_data: Input,
    patch_size: int = 256,
    mask_suffix: str = "_root_mask.tif",
):
    """Run preprocessing as a one-node data pipeline.

    The returned patched_data output is registered as a versioned data asset by
    the submission script, which sets its name and version on the pipeline
    output.
    """
    preprocess = _preprocess_op(
        train_data=train_data,
        val_data=val_data,
        patch_size=patch_size,
        mask_suffix=mask_suffix,
    )

    # Preprocessing is CPU-only OpenCV and numpy work, so it runs on the CPU
    # instance type rather than inheriting a GPU node.
    preprocess.resources = {"instance_type": AZURE_CPU_INSTANCE}

    return {"patched_data": preprocess.outputs.patched_data}
