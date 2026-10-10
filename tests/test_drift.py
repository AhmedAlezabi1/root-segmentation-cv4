"""
tests/test_drift.py
====================
Unit tests for :func:`cv4.monitoring.drift.run_drift_check`.

These tests run entirely against an in-memory SQLite database (no external
database, no model weights). PredictionLog rows and a ConfidenceBaseline are
seeded, the drift check is run, and the resulting drift_events rows are
inspected. Where drift behaviour matters, the rolling scores are drawn from a
fixed-seed RNG so the KS-test outcome is deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from cv4.api.models import ConfidenceBaseline, DriftEvent, PredictionLog
from cv4.db import Base
from cv4.monitoring.drift import NUM_LANDMARKS, run_drift_check

# ---------------------------------------------------------------------------
# In-memory database setup
# ---------------------------------------------------------------------------
# StaticPool forces all connections to share the same in-memory database, so
# tables created by Base.metadata.create_all stay visible to every session.

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def session():
    """Provide a clean in-memory DB session for each test."""
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


def _add_baseline(session, mean, std, model_version="v1"):
    """Insert a ConfidenceBaseline with the given per-landmark mean/std."""
    baseline = ConfidenceBaseline(
        model_version=model_version,
        n_samples=100,
        mean=mean,
        std=std,
        quantiles={"5": mean, "25": mean, "50": mean, "75": mean, "95": mean},
    )
    session.add(baseline)
    session.commit()
    return baseline


def _add_logs(session, score_rows, base_time=None):
    """Insert PredictionLog rows with recent (within-window) timestamps."""
    base_time = base_time or datetime.now(timezone.utc) - timedelta(hours=1)
    for i, scores in enumerate(score_rows):
        session.add(
            PredictionLog(
                prediction_id=str(i),
                confidence_scores=scores,
                timestamp=base_time + timedelta(seconds=i),
            )
        )
    session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_drift_when_rolling_matches_baseline(session):
    """Rolling scores from the baseline distribution => no drift, all tested."""
    rng = np.random.default_rng(0)
    mean = [0.5] * NUM_LANDMARKS
    std = [0.1] * NUM_LANDMARKS
    _add_baseline(session, mean, std)
    # 200 samples drawn from exactly N(0.5, 0.1) for every landmark.
    rows = rng.normal(0.5, 0.1, size=(200, NUM_LANDMARKS)).tolist()
    _add_logs(session, rows)

    events = run_drift_check(session=session)

    assert len(events) == NUM_LANDMARKS
    assert all(not e.drift_detected for e in events)
    assert all(e.p_value is not None for e in events)


def test_drift_detected_when_rolling_far_from_baseline(session):
    """Rolling scores far from the baseline mean => drift flagged."""
    rng = np.random.default_rng(1)
    mean = [0.2] * NUM_LANDMARKS
    std = [0.05] * NUM_LANDMARKS
    _add_baseline(session, mean, std)
    # Samples centred at 0.8 are nowhere near N(0.2, 0.05).
    rows = rng.normal(0.8, 0.05, size=(200, NUM_LANDMARKS)).tolist()
    _add_logs(session, rows)

    events = run_drift_check(session=session)

    assert len(events) == NUM_LANDMARKS
    assert all(e.drift_detected for e in events)
    assert all(e.p_value is not None and e.p_value < 0.05 for e in events)


def test_audit_row_written_for_every_landmark(session):
    """One drift_events row per landmark is persisted each run."""
    _add_baseline(session, [0.5] * NUM_LANDMARKS, [0.1] * NUM_LANDMARKS)
    _add_logs(session, [[0.5] * NUM_LANDMARKS for _ in range(10)])

    run_drift_check(session=session)

    stored = session.query(DriftEvent).all()
    assert len(stored) == NUM_LANDMARKS
    assert sorted(e.landmark_index for e in stored) == list(range(NUM_LANDMARKS))
    assert all(e.detected_at is not None for e in stored)


def test_no_baseline_returns_empty_and_writes_nothing(session):
    """With no baseline, the check is a no-op (no rows, no exception)."""
    _add_logs(session, [[0.5] * NUM_LANDMARKS for _ in range(10)])

    events = run_drift_check(session=session)

    assert events == []
    assert session.query(DriftEvent).count() == 0


def test_landmark_with_no_rolling_samples_is_untestable(session):
    """A landmark that is always None gets a null-p_value audit row."""
    _add_baseline(session, [0.5] * NUM_LANDMARKS, [0.1] * NUM_LANDMARKS)
    # Landmark index 1 is None in every row; others have plenty of samples.
    rows = [[0.5, None, 0.5, 0.5, 0.5] for _ in range(20)]
    _add_logs(session, rows)

    events = run_drift_check(session=session)
    by_index = {e.landmark_index: e for e in events}

    assert by_index[1].p_value is None
    assert by_index[1].drift_detected is False
    assert "untestable" in by_index[1].notes
    # A landmark with samples is still tested normally.
    assert by_index[0].p_value is not None


def test_only_last_24h_rows_are_used(session):
    """Rows older than the rolling window are excluded from the comparison."""
    mean = [0.2] * NUM_LANDMARKS
    std = [0.05] * NUM_LANDMARKS
    _add_baseline(session, mean, std)

    # Old rows (2 days ago) far from baseline — must be ignored.
    old_time = datetime.now(timezone.utc) - timedelta(days=2)
    _add_logs(session, [[0.9] * NUM_LANDMARKS for _ in range(50)], base_time=old_time)

    # Recent rows matching the baseline — must drive the result.
    rng = np.random.default_rng(2)
    recent = rng.normal(0.2, 0.05, size=(200, NUM_LANDMARKS)).tolist()
    _add_logs(session, recent)

    events = run_drift_check(session=session)

    # If the old, far-off rows had leaked in, drift would fire; they don't.
    assert all(not e.drift_detected for e in events)
