"""
monitoring/drift.py
===================
Scheduled drift-detection job for per-landmark confidence scores.

Each run takes the last 24 hours of confidence vectors logged in
``predictions_log``, loads the most recent ``confidence_baseline``, and runs a
per-landmark Kolmogorov-Smirnov test of the rolling scores against the baseline.
Because the baseline stores only per-landmark mean/std, the comparison is a
one-sample KS test against a normal distribution ``N(mean, std)`` (the user-
approved normal approximation).

Every landmark gets exactly one row written to ``drift_events`` per run — even
when it cannot be tested — so the table is a complete audit log. A landmark
whose p-value falls below :data:`DRIFT_P_VALUE_THRESHOLD` is flagged with
``drift_detected=True``.

The single public entry point is :func:`run_drift_check`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from scipy import stats
from sqlalchemy.orm import Session

from cv4.api.models import ConfidenceBaseline, DriftEvent, PredictionLog

logger = logging.getLogger(__name__)

# Number of landmarks (lanes) per prediction. Matches the fixed length of
# predictions_log.confidence_scores and baseline.NUM_LANDMARKS.
NUM_LANDMARKS = 5

# Significance threshold: a landmark drifts when its KS p-value is below this.
DRIFT_P_VALUE_THRESHOLD = 0.05

# Size of the rolling window compared against the baseline.
ROLLING_WINDOW_HOURS = 24

# Minimum rolling samples needed to run the KS test for a landmark. Below this
# the test is meaningless, so the landmark is logged as untestable.
MIN_ROLLING_SAMPLES = 2


def run_drift_check(session: Session | None = None) -> list[DriftEvent]:
    """Compare the last 24h of confidence against the baseline and log drift.

    Fetches every ``predictions_log`` row from the last
    :data:`ROLLING_WINDOW_HOURS` hours, loads the most recent
    ``confidence_baseline``, and for each landmark (0-4) runs a one-sample KS
    test of the rolling scores against ``N(mean, std)`` from the baseline. One
    ``drift_events`` row is inserted per landmark (full audit log); a landmark
    with ``p_value < 0.05`` is flagged ``drift_detected=True``.

    Parameters
    ----------
    session : Session, optional
        Database session to use. If omitted, one is created from the cv4
        session factory and closed before returning.

    Returns
    -------
    list[DriftEvent]
        The persisted drift-event rows (one per landmark), or an empty list if
        no baseline exists yet (nothing to compare against).
    """
    # Lazily build a session if the caller did not provide one, and make sure
    # we only close sessions we ourselves opened.
    own_session = session is None
    if own_session:
        from cv4.db import _get_session_factory

        session = _get_session_factory()()

    try:
        baseline = session.query(ConfidenceBaseline).order_by(ConfidenceBaseline.computed_at.desc()).first()
        if baseline is None:
            logger.warning("Drift check skipped: no confidence_baseline exists yet.")
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=ROLLING_WINDOW_HOURS)
        rows = session.query(PredictionLog).filter(PredictionLog.timestamp >= cutoff).all()

        events = [_check_landmark(i, rows, baseline) for i in range(NUM_LANDMARKS)]

        session.add_all(events)
        session.commit()
        for event in events:
            session.refresh(event)

        n_drifted = sum(1 for e in events if e.drift_detected)
        logger.info(
            "Drift check complete: baseline_id=%s n_rolling=%d p_values=%s landmarks_drifted=%d",
            baseline.id,
            len(rows),
            [e.p_value for e in events],
            n_drifted,
        )
        return events
    finally:
        if own_session:
            session.close()


def _rolling_scores(landmark_index: int, rows: list[PredictionLog]) -> list[float]:
    """Collect the non-``None`` confidence values for one landmark.

    Mirrors the indexing in ``baseline._build_score_matrix``: a row's
    ``confidence_scores`` is treated as a length-``NUM_LANDMARKS`` vector with
    ``None`` placeholders, and shorter rows simply contribute nothing for the
    missing index.
    """
    scores: list[float] = []
    for row in rows:
        values = row.confidence_scores or []
        if landmark_index < len(values):
            value = values[landmark_index]
            if value is not None:
                scores.append(float(value))
    return scores


def _check_landmark(
    landmark_index: int,
    rows: list[PredictionLog],
    baseline: ConfidenceBaseline,
) -> DriftEvent:
    """Run the KS test for one landmark and build its (unsaved) DriftEvent.

    Returns an untestable row (``p_value=None``, ``drift_detected=False``, with
    a reason in ``notes``) when the landmark has too few rolling samples or a
    degenerate baseline (missing mean/std or ``std <= 0``). Otherwise runs the
    one-sample KS test of the rolling scores against ``N(mean, std)``.
    """
    scores = _rolling_scores(landmark_index, rows)
    mean = baseline.mean[landmark_index] if landmark_index < len(baseline.mean) else None
    std = baseline.std[landmark_index] if landmark_index < len(baseline.std) else None

    if len(scores) < MIN_ROLLING_SAMPLES:
        return DriftEvent(
            landmark_index=landmark_index,
            p_value=None,
            drift_detected=False,
            notes=f"untestable: only {len(scores)} rolling samples",
        )
    if mean is None or std is None or std <= 0:
        return DriftEvent(
            landmark_index=landmark_index,
            p_value=None,
            drift_detected=False,
            notes=f"untestable: degenerate baseline (mean={mean}, std={std})",
        )

    _, p_value = stats.kstest(scores, "norm", args=(mean, std))
    p_value = float(p_value)
    return DriftEvent(
        landmark_index=landmark_index,
        p_value=p_value,
        drift_detected=p_value < DRIFT_P_VALUE_THRESHOLD,
        notes=f"n_rolling={len(scores)}",
    )
