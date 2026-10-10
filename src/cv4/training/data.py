"""Data generators for training and validation.

Builds paired (image, mask) generators on top of Keras ImageDataGenerator.
Augmentation strength is parameterized so the hyperparameter sweep (Issue 3.4)
can treat it as a categorical knob rather than tuning each augmentation flag
independently. Validation generators never augment.
"""

import os
from typing import Tuple

from tensorflow.keras.preprocessing.image import (
    DirectoryIterator,
    ImageDataGenerator,
)

# Augmentation presets. The sweep job picks one of these by name. Mild is a
# reasonable default that worked in Block B; strong is for exploration and
# may hurt convergence on small datasets; none is a control for sanity checks.
AUGMENTATION_PRESETS = {
    "none": {},
    "mild": {
        "rotation_range": 20,
        "horizontal_flip": True,
        "vertical_flip": True,
        "zoom_range": [0.8, 1.2],
        "fill_mode": "reflect",
    },
    "strong": {
        "rotation_range": 45,
        "horizontal_flip": True,
        "vertical_flip": True,
        "zoom_range": [0.6, 1.4],
        "shear_range": 15,
        "fill_mode": "reflect",
    },
}


def _combined_generator(img_iter: DirectoryIterator, mask_iter: DirectoryIterator):
    """Zip image and mask iterators into the (x, y) tuples model.fit expects.

    Both iterators must share the same seed and batch size so each call yields
    a correctly paired batch. The pairing is positional, not by filename, so
    upstream patch directories must contain identically named files.
    """
    while True:
        yield next(img_iter), next(mask_iter)


def build_generators(
    data_dir: str,
    batch_size: int,
    patch_size: int,
    augmentation_strength: str,
) -> Tuple[object, object, int, int]:
    """Build paired training and validation generators.

    Expects the on-disk layout produced by the preprocessing notebook:

        data_dir/
          train_images/train/*.png
          train_masks/train/*.png
          val_images/val/*.png
          val_masks/val/*.png

    The extra subfolder under each split is a Keras requirement:
    flow_from_directory treats the immediate subfolder as a class label, so
    we group all patches under a single dummy class.

    Args:
        data_dir: Path to the dataset root.
        batch_size: Samples per gradient step.
        patch_size: Square edge length in pixels for resizing.
        augmentation_strength: Key into AUGMENTATION_PRESETS.

    Returns:
        Tuple of (train_generator, val_generator, train_steps, val_steps).
        The step counts equal len(image_iterator) and should be passed to
        model.fit's steps_per_epoch and validation_steps.
    """
    if augmentation_strength not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown augmentation_strength '{augmentation_strength}'. Expected one of {list(AUGMENTATION_PRESETS)}.")

    aug_kwargs = AUGMENTATION_PRESETS[augmentation_strength]

    train_img_gen = ImageDataGenerator(rescale=1.0 / 255, **aug_kwargs)
    train_mask_gen = ImageDataGenerator(rescale=1.0 / 255, **aug_kwargs)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        target_size=(patch_size, patch_size),
        class_mode=None,
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
    )

    train_img_iter = train_img_gen.flow_from_directory(os.path.join(data_dir, "train_images"), **common)
    train_mask_iter = train_mask_gen.flow_from_directory(os.path.join(data_dir, "train_masks"), **common)
    val_img_iter = val_gen.flow_from_directory(os.path.join(data_dir, "val_images"), shuffle=True, **common)
    val_mask_iter = val_gen.flow_from_directory(os.path.join(data_dir, "val_masks"), shuffle=True, **common)

    train_generator = _combined_generator(train_img_iter, train_mask_iter)
    val_generator = _combined_generator(val_img_iter, val_mask_iter)

    return train_generator, val_generator, len(train_img_iter), len(val_img_iter)
