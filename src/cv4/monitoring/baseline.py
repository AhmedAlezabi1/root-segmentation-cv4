"""
monitoring/baseline.py
======================
Compute and persist the baseline confidence distribution used by drift
detection.

After deployment, the first ``n`` predictions logged in ``predictions_log``
are summarised into a per-landmark statistical profile (mean, std, and key
quantiles) and stored as a row in ``confidence_baseline``, tagged with the
model version. Drift detection later compares live confidence against this
baseline.

The single public entry point is :func:`compute_and_store_baseline`.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sqlalchemy.orm import Session

from cv4.api.models import ConfidenceBaseline, PredictionLog

logger = logging.getLogger(__name__)

# Number of landmarks (lanes) per prediction. Matches the fixed length of
# predictions_log.confidence_scores.
NUM_LANDMARKS = 5

# Percentiles stored per landmark, keyed by their string label in the
# quantiles JSON object.
QUANTILE_PERCENTILES = (5, 25, 50, 75, 95)


def _to_float_or_none(value: np.floating) -> float | None:
    """Convert a numpy scalar to a JSON-safe Python float, NaN -> None."""
    return None if np.isnan(value) else float(value)


def compute_and_store_baseline(
    n: int,
    model_version: str,
    session: Session | None = None,
) -> ConfidenceBaseline:
    """Compute the baseline confidence distribution and store it.

    Reads the first ``n`` rows of ``predictions_log`` (ordered by timestamp
    ascending), computes the per-landmark mean, standard deviation, and key
    quantiles of the confidence scores, and inserts the result as a row in
    ``confidence_baseline`` tagged with ``model_version``.

    Statistics are computed per landmark (lane index 0-4): for each landmark,
    only its non-``None`` confidence values across the ``n`` rows are used. A
    landmark with no usable samples gets ``None`` for every statistic.

    Parameters
    ----------
    n : int
        Number of predictions to use for the baseline.
    model_version : str
        Identifier of the model the baseline characterises.
    session : Session, optional
        Database session to use. If omitted, one is created from the cv4
        session factory and closed before returning.

    Returns
    -------
    ConfidenceBaseline
        The persisted baseline row (refreshed, so ``id`` and ``computed_at``
        are populated).

    Raises
    ------
    ValueError
        If ``n`` is not positive, or if fewer than ``n`` predictions exist
        in ``predictions_log``.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}.")

    # Lazily build a session if the caller did not provide one, and make sure
    # we only close sessions we ourselves opened.
    own_session = session is None
    if own_session:
        from cv4.db import _get_session_factory

        session = _get_session_factory()()

    try:
        count = session.query(PredictionLog).count()
        if count < n:
            raise ValueError(f"Need at least {n} predictions to compute a baseline, but only {count} exist.")

        rows = session.query(PredictionLog).order_by(PredictionLog.timestamp.asc()).limit(n).all()

        scores = _build_score_matrix(rows)
        mean, std, quantiles = _summarise(scores)

        baseline = ConfidenceBaseline(
            model_version=model_version,
            n_samples=n,
            mean=mean,
            std=std,
            quantiles=quantiles,
        )
        session.add(baseline)
        session.commit()
        session.refresh(baseline)

        logger.info(
            "Computed confidence baseline: model_version=%s n_samples=%d mean=%s std=%s",
            model_version,
            n,
            mean,
            std,
        )
        return baseline
    finally:
        if own_session:
            session.close()


def _build_score_matrix(rows: list[PredictionLog]) -> np.ndarray:
    """Stack the rows' confidence vectors into an ``(n, NUM_LANDMARKS)`` array.

    ``None`` confidence values (lanes with no detected root) become ``NaN``.
    Rows that are shorter than ``NUM_LANDMARKS`` are padded with ``NaN`` and
    longer rows are truncated, so a malformed row never breaks the baseline.
    """
    matrix = np.full((len(rows), NUM_LANDMARKS), np.nan, dtype=float)
    for i, row in enumerate(rows):
        scores = row.confidence_scores or []
        for j in range(min(NUM_LANDMARKS, len(scores))):
            value = scores[j]
            if value is not None:
                matrix[i, j] = float(value)
    return matrix


def _summarise(scores: np.ndarray) -> tuple[list, list, dict]:
    """Compute per-landmark mean, std, and quantiles from a score matrix.

    All-NaN landmark columns (no samples) yield ``None`` in every output,
    rather than NaN, so the stored JSON stays clean and serialisable.
    """
    # np.nanmean/nanstd/nanpercentile warn on all-NaN columns; we handle those
    # columns explicitly below, so the warning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(scores, axis=0)
        col_std = np.nanstd(scores, axis=0)
        col_quantiles = np.nanpercentile(scores, QUANTILE_PERCENTILES, axis=0)  # shape: (len(QUANTILE_PERCENTILES), NUM_LANDMARKS)

    mean = [_to_float_or_none(v) for v in col_mean]
    std = [_to_float_or_none(v) for v in col_std]
    quantiles = {str(pct): [_to_float_or_none(v) for v in col_quantiles[idx]] for idx, pct in enumerate(QUANTILE_PERCENTILES)}
    return mean, std, quantiles
