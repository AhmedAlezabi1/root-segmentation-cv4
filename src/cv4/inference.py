"""Inference module for plant root segmentation.

Handles loading the trained UNet model and running patch-based
predictions on preprocessed plant images. Uses overlapping patches
with test-time augmentation and smooth spline blending.

Refactored from Block B notebook (Abdelrahman 245769, 256px UNet pipeline).

Example:
    model = load_model("models/unet_256px.h5")
    prob_map = predict_smooth_windowing(padded_image, model)
"""

import logging
import os
import time

import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

from cv4.config import BATCH_SIZE, DEFAULT_MODEL_PATH, PATCH_SIZE, STRIDE

logger = logging.getLogger(__name__)


def f1(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculate the F1 score between true labels and predictions.

    This function is needed when loading the model because it was
    used as a custom metric during training. It has to match the
    original training code exactly or the model won't load.

    Args:
        y_true (tf.Tensor): The real labels (0 or 1).
        y_pred (tf.Tensor): The model's predicted probabilities.

    Returns:
        tf.Tensor: The F1 score as a single number.
    """

    def recall_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        return tp / (positives + tf.keras.backend.epsilon())

    def precision_m(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        pred_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        return tp / (pred_positives + tf.keras.backend.epsilon())

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


def load_model(model_path: str | None = None) -> tf.keras.Model:
    """Load a trained UNet model from a .h5 file.

    Loads the model in inference-only mode so it's ready to make
    predictions right away. The custom F1 metric is passed to Keras
    via custom_objects so it can reconstruct the model.

    If no path is given, it uses the default path from config.

    Args:
        model_path (str | None): Path to the .h5 model file.
            If None, uses DEFAULT_MODEL_PATH from config.

    Returns:
        tf.keras.Model: The loaded model, ready for prediction.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        ValueError: If the model file is corrupted or can't be loaded.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        logger.info("Loading model from %s", model_path)
        model = keras_load_model(model_path, custom_objects={"f1": f1}, compile=False)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load model from %s: %s", model_path, e)
        raise ValueError(f"Could not load model from {model_path}: {e}") from e


def get_spline_window(patch_size: int, power: int = 2) -> np.ndarray:
    """Create a 2D blending window for smooth patch stitching.

    When we predict on overlapping patches, the edges of each patch
    can create visible seams. This window gives higher weight to the
    center of each patch and lower weight to the edges, so when
    patches overlap the result looks smooth.

    Args:
        patch_size (int): Size of each square patch in pixels.
        power (int): How sharply the window falls off at the edges.
            Higher values mean sharper falloff. Default is 2.

    Returns:
        np.ndarray: 2D array of shape (patch_size, patch_size) with
            blending weights between 0 and 1.
    """
    window_1d = scipy.signal.windows.hann(patch_size) ** power
    window_2d = np.outer(window_1d, window_1d)
    return window_2d


def predict_smooth_windowing(
    img_padded: np.ndarray,
    model: tf.keras.Model,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Run the UNet model on a single preprocessed image.

    Splits the image into overlapping patches, predicts each one
    (plus a flipped version for better accuracy), then blends
    everything back together into one smooth probability map.

    The horizontal flip is called test-time augmentation (TTA). By
    predicting on both the original and flipped patch and averaging
    the results, we get more accurate predictions.

    Args:
        img_padded (np.ndarray): Preprocessed grayscale image, already
            padded so height and width are multiples of patch_size.
            Shape: (H, W), dtype: uint8.
        model (tf.keras.Model): The loaded UNet model.
        patch_size (int): Size of each square patch. Default from config.
        stride (int): How far apart patches are. Default from config.
            128 gives 50% overlap between patches.
        batch_size (int): How many patches to predict at once.
            Default from config.

    Returns:
        np.ndarray: Probability map of shape (H, W) with float values
            between 0 and 1. Higher values mean more likely to be root.

    Raises:
        MemoryError: If there isn't enough memory to process all patches.
    """
    start_time = time.time()

    h, w = img_padded.shape
    spline_window = get_spline_window(patch_size)

    coords = []
    patches = []

    n_h = (h - patch_size) // stride + 1
    n_w = (w - patch_size) // stride + 1

    logger.info(
        "Extracting patches: %dx%d grid, %d total patches",
        n_h,
        n_w,
        n_h * n_w,
    )

    for i in range(n_h):
        for j in range(n_w):
            y = i * stride
            x = j * stride

            patch = img_padded[y : y + patch_size, x : x + patch_size]
            patch_norm = patch.astype(np.float32) / 255.0
            patch_norm = np.expand_dims(patch_norm, axis=-1)

            patches.append(patch_norm)
            coords.append((y, x))

    patches_array = np.array(patches)

    # Flip each patch horizontally for test-time augmentation
    patches_flipped = np.fliplr(patches_array)
    full_batch = np.concatenate([patches_array, patches_flipped], axis=0)

    # Some models expect 3 channels (RGB) instead of 1 (grayscale)
    if model.input_shape[-1] == 3:
        full_batch = np.repeat(full_batch, 3, axis=-1)

    logger.info("Predicting %d patches (including flipped versions)", len(full_batch))

    try:
        all_preds = model.predict(full_batch, batch_size=batch_size, verbose=0)
    except tf.errors.ResourceExhaustedError as e:
        logger.error("Out of memory during prediction: %s", e)
        raise MemoryError(f"Not enough memory to predict {len(full_batch)} patches. Try reducing batch_size (current: {batch_size}).") from e

    # Average the original and flipped predictions for better accuracy
    n_patches = len(patches)
    preds_orig = all_preds[:n_patches]
    preds_flip = np.fliplr(all_preds[n_patches:])
    final_preds = (preds_orig + preds_flip) / 2.0

    # Blend all patches back into one full-size probability map
    prob_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for idx, (y, x) in enumerate(coords):
        pred_patch = final_preds[idx, :, :, 0]
        prob_map[y : y + patch_size, x : x + patch_size] += pred_patch * spline_window
        weight_map[y : y + patch_size, x : x + patch_size] += spline_window

    weight_map[weight_map == 0] = 1.0
    prob_map = prob_map / weight_map

    elapsed = time.time() - start_time
    logger.info(
        "Prediction complete, output shape: %s, took %.2f seconds",
        prob_map.shape,
        elapsed,
    )

    return prob_map
