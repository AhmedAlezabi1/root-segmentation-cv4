"""Custom loss functions for root segmentation.

Root pixels make up well under 1% of any petri dish image, and the roots
themselves are thin and often faint near the tip. Plain binary cross-entropy
under-penalizes the resulting false negatives. The hybrid loss below combines
BCE for global pixel-level stability with focal Tversky for sensitivity to
the under-represented positive class.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

from cv4.training.config import TVERSKY_ALPHA, TVERSKY_BETA, TVERSKY_GAMMA


def focal_tversky_loss(y_true, y_pred):
    """Compute focal Tversky loss.

    The Tversky index generalizes the Dice coefficient by weighting false
    positives and false negatives asymmetrically. alpha < beta penalizes
    false negatives more, which suits the missed-root cost asymmetry. The
    focal exponent gamma > 1 then concentrates gradient on examples where
    the model is currently failing, accelerating learning on faint roots.

    Args:
        y_true: Ground truth mask tensor with values in {0, 1}.
        y_pred: Predicted mask tensor with values in [0, 1].

    Returns:
        Scalar tensor: the focal Tversky loss for the batch.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(y_true_f * y_pred_f)
    false_positives = K.sum((1 - y_true_f) * y_pred_f)
    false_negatives = K.sum(y_true_f * (1 - y_pred_f))

    tversky = (true_positives + K.epsilon()) / (true_positives + TVERSKY_ALPHA * false_positives + TVERSKY_BETA * false_negatives + K.epsilon())
    return K.pow((1 - tversky), TVERSKY_GAMMA)


def hybrid_loss(y_true, y_pred):
    """Combine binary cross-entropy with focal Tversky loss.

    BCE provides a stable pixel-wise gradient signal that prevents training
    from collapsing in early epochs, while focal Tversky drives the model
    toward correctly recovering thin and faint root structures once basic
    foreground/background separation is learned.

    Args:
        y_true: Ground truth mask tensor with values in {0, 1}.
        y_pred: Predicted mask tensor with values in [0, 1].

    Returns:
        Scalar tensor: the summed hybrid loss for the batch.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ftl = focal_tversky_loss(y_true, y_pred)
    return bce + ftl
