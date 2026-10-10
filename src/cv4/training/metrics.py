"""Custom Keras metrics for training and evaluation.

The F1 metric defined here is computed on every batch during training and
logged per epoch. Block B research showed accuracy is misleading for root
segmentation because the background class dominates by orders of magnitude,
so F1 is the primary metric we optimize and gate registration on.
"""

from tensorflow.keras import backend as K


def f1_metric(y_true, y_pred):
    """Compute the F1 score on a binary segmentation batch.

    Pixels with predicted probability above 0.5 are counted as positive after
    rounding. Returns the harmonic mean of precision and recall over the
    flattened batch, with an epsilon term to prevent division by zero when a
    batch contains no positives (early epochs or all-background patches).

    Args:
        y_true: Ground truth mask tensor of shape (batch, H, W, 1) with values
            in {0, 1}.
        y_pred: Predicted mask tensor of the same shape with values in [0, 1].

    Returns:
        Scalar tensor: the batch F1 score.
    """

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (positives + K.epsilon())

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
