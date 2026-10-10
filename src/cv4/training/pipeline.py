"""End-to-end Azure ML training pipeline for cv4 root segmentation.

Wires the four cv4 training components into a single DAG:

    preprocess -> train -> evaluate -> register

Each node consumes the output of the previous one through typed
URI_FOLDER bindings; Azure ML schedules them in order and surfaces the
whole pipeline as one job in the studio with a visual graph. The register
node always runs and decides internally whether the F1 score meets the
threshold for registration, so the pipeline does not need conditional
execution; the gating happens inside register.py.

The pipeline DSL imposes two constraints worth knowing:

1. Components must be declared at module scope, not inside the @pipeline
   function. Component definitions inside the pipeline body receive
   PipelineInput wrappers in their inputs dict and command() rejects
   those.
2. Pipeline parameters can't be evaluated with Python operators inside
   the pipeline body. They become PipelineExpression objects that defer
   to submission time, so `if max_steps_per_epoch > 0` raises. Conditional
   wiring (such as whether to bind optional inputs) must be decided in
   the submission script and either pass a concrete int or omit the
   kwarg entirely.
"""

import logging
from pathlib import Path

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from cv4.config import AZURE_CPU_INSTANCE, AZURE_GPU_INSTANCE

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_PATH = str(REPO_ROOT / "src")

ENV_CPU = "cv4-cpu:3"
ENV_TRAINING = "cv4-training:6"


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


_train_op = command(
    name="cv4_train",
    display_name="train",
    code=CODE_PATH,
    command=(
        "python -m cv4.training.train "
        "--data-path ${{inputs.patched_data}} "
        "--model-output-path ${{outputs.trained_model}}/model.h5 "
        "--metrics-output-path ${{outputs.metrics}}/metrics.json "
        "--learning-rate ${{inputs.learning_rate}} "
        "--batch-size ${{inputs.batch_size}} "
        "--epochs ${{inputs.epochs}} "
        "--augmentation-strength ${{inputs.augmentation_strength}} "
        "--patch-size ${{inputs.patch_size}} "
        "$[[--max-steps-per-epoch ${{inputs.max_steps_per_epoch}}]] "
        "$[[--max-val-steps ${{inputs.max_val_steps}}]]"
    ),
    inputs={
        "patched_data": Input(type=AssetTypes.URI_FOLDER, mode="download"),
        "learning_rate": Input(type="number", default=1e-4),
        "batch_size": Input(type="integer", default=16),
        "epochs": Input(type="integer", default=50),
        "augmentation_strength": Input(type="string", default="mild"),
        "patch_size": Input(type="integer", default=256),
        "max_steps_per_epoch": Input(type="integer", optional=True),
        "max_val_steps": Input(type="integer", optional=True),
    },
    outputs={
        "trained_model": Output(type=AssetTypes.URI_FOLDER),
        "metrics": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENV_TRAINING,
)


_evaluate_op = command(
    name="cv4_evaluate",
    display_name="evaluate",
    code=CODE_PATH,
    command=(
        "python -m cv4.training.evaluate "
        "--model-path ${{inputs.trained_model}}/model.h5 "
        "--data-path ${{inputs.patched_data}} "
        "--full-images-path ${{inputs.raw_val_data}} "
        "--metrics-output-path ${{outputs.eval_metrics}}/metrics.json "
        "--overlays-output-dir ${{outputs.overlays}} "
        "--batch-size ${{inputs.batch_size}} "
        "--patch-size ${{inputs.patch_size}}"
    ),
    inputs={
        "trained_model": Input(type=AssetTypes.URI_FOLDER),
        "patched_data": Input(type=AssetTypes.URI_FOLDER, mode="download"),
        "raw_val_data": Input(type=AssetTypes.URI_FOLDER, mode="download"),
        "batch_size": Input(type="integer", default=32),
        "patch_size": Input(type="integer", default=256),
    },
    outputs={
        "eval_metrics": Output(type=AssetTypes.URI_FOLDER),
        "overlays": Output(type=AssetTypes.URI_FOLDER),
    },
    environment=ENV_TRAINING,
)


_register_op = command(
    name="cv4_register",
    display_name="register",
    code=CODE_PATH,
    command=(
        "python -m cv4.training.register "
        "--metrics-path ${{inputs.eval_metrics}}/metrics.json "
        "--model-path ${{inputs.trained_model}}/model.h5 "
        "--decision-output-path ${{outputs.decision}}/decision.json "
        "--model-name ${{inputs.model_name}} "
        "--f1-threshold ${{inputs.f1_threshold}}"
    ),
    inputs={
        "trained_model": Input(type=AssetTypes.URI_FOLDER),
        "eval_metrics": Input(type=AssetTypes.URI_FOLDER),
        "model_name": Input(type="string", default="cv4_root_segmentation"),
        "f1_threshold": Input(type="number", default=0.80),
    },
    outputs={"decision": Output(type=AssetTypes.URI_FOLDER)},
    environment=ENV_CPU,
)


@pipeline(
    name="cv4_training_pipeline",
    display_name="cv4 Training Pipeline (preprocess + train + evaluate + register)",
    description=(
        "End-to-end DAG: raw petri dish images and masks are patched, used to train "
        "a Residual Attention U-Net, evaluated for F1/IoU, and conditionally "
        "registered to the workspace model registry."
    ),
)
def cv4_pipeline(
    train_data: Input,
    val_data: Input,
    patch_size: int = 256,
    mask_suffix: str = "_root_mask.tif",
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 50,
    augmentation_strength: str = "mild",
    eval_batch_size: int = 32,
    model_name: str = "cv4_root_segmentation",
    f1_threshold: float = 0.80,
    max_steps_per_epoch: int = None,
    max_val_steps: int = None,
):
    """Wire the four cv4 training components into a DAG."""
    preprocess = _preprocess_op(
        train_data=train_data,
        val_data=val_data,
        patch_size=patch_size,
        mask_suffix=mask_suffix,
    )

    train = _train_op(
        patched_data=preprocess.outputs.patched_data,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        augmentation_strength=augmentation_strength,
        patch_size=patch_size,
        max_steps_per_epoch=max_steps_per_epoch,
        max_val_steps=max_val_steps,
    )

    evaluate = _evaluate_op(
        trained_model=train.outputs.trained_model,
        patched_data=preprocess.outputs.patched_data,
        raw_val_data=val_data,
        batch_size=eval_batch_size,
        patch_size=patch_size,
    )

    register = _register_op(
        trained_model=train.outputs.trained_model,
        eval_metrics=evaluate.outputs.eval_metrics,
        model_name=model_name,
        f1_threshold=f1_threshold,
    )

    preprocess.resources = {"instance_type": AZURE_CPU_INSTANCE}
    train.resources = {"instance_type": AZURE_GPU_INSTANCE}
    evaluate.resources = {"instance_type": AZURE_GPU_INSTANCE}
    register.resources = {"instance_type": AZURE_CPU_INSTANCE}

    return {
        "patched_data": preprocess.outputs.patched_data,
        "trained_model": train.outputs.trained_model,
        "train_metrics": train.outputs.metrics,
        "eval_metrics": evaluate.outputs.eval_metrics,
        "overlays": evaluate.outputs.overlays,
        "decision": register.outputs.decision,
    }
