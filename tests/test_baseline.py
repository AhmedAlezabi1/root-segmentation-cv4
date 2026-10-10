"""
tests/test_baseline.py
======================
Unit tests for :func:`cv4.monitoring.baseline.compute_and_store_baseline`.

These tests run entirely against an in-memory SQLite database (no external
database, no model weights). PredictionLog rows with known confidence vectors
are inserted, the baseline is computed, and the stored statistics are checked
against values computed independently with numpy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from cv4.api.models import ConfidenceBaseline, PredictionLog
from cv4.db import Base
from cv4.monitoring.baseline import compute_and_store_baseline

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


def _insert_logs(session, score_rows, base_time=None):
    """Insert PredictionLog rows with ascending timestamps.

    ``score_rows`` is a list of length-5 confidence vectors (values or None).
    Returns nothing; rows are committed.
    """
    base_time = base_time or datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, scores in enumerate(score_rows):
        session.add(
            PredictionLog(
                prediction_id=str(i),
                confidence_scores=scores,
                timestamp=base_time + timedelta(minutes=i),
            )
        )
    session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_baseline_statistics_are_correct(session):
    """Mean, std, and quantiles match an independent numpy computation."""
    score_rows = [
        [0.10, 0.20, 0.30, 0.40, 0.50],
        [0.20, 0.30, 0.40, 0.50, 0.60],
        [0.30, 0.40, 0.50, 0.60, 0.70],
        [0.40, 0.50, 0.60, 0.70, 0.80],
    ]
    _insert_logs(session, score_rows)

    baseline = compute_and_store_baseline(4, "v1.2.3", session=session)

    expected = np.array(score_rows, dtype=float)
    expected_mean = np.mean(expected, axis=0)
    expected_std = np.std(expected, axis=0)
    expected_q = np.percentile(expected, [5, 25, 50, 75, 95], axis=0)

    assert baseline.model_version == "v1.2.3"
    assert baseline.n_samples == 4
    assert baseline.mean == pytest.approx(list(expected_mean))
    assert baseline.std == pytest.approx(list(expected_std))
    for idx, pct in enumerate((5, 25, 50, 75, 95)):
        assert baseline.quantiles[str(pct)] == pytest.approx(list(expected_q[idx]))


def test_baseline_row_is_persisted(session):
    """Exactly one confidence_baseline row is stored after the call."""
    _insert_logs(session, [[0.5] * 5 for _ in range(3)])

    compute_and_store_baseline(3, "model-a", session=session)

    rows = session.query(ConfidenceBaseline).all()
    assert len(rows) == 1
    assert rows[0].model_version == "model-a"
    assert rows[0].computed_at is not None


def test_only_first_n_rows_used_in_timestamp_order(session):
    """Only the earliest n predictions (by timestamp) contribute."""
    # First 2 rows are all 0.0; later rows are 1.0. Using n=2 must yield mean 0.
    score_rows = [[0.0] * 5, [0.0] * 5, [1.0] * 5, [1.0] * 5]
    _insert_logs(session, score_rows)

    baseline = compute_and_store_baseline(2, "v1", session=session)

    assert baseline.n_samples == 2
    assert baseline.mean == pytest.approx([0.0] * 5)


def test_raises_when_too_few_predictions(session):
    """A ValueError is raised when fewer than n rows exist."""
    _insert_logs(session, [[0.5] * 5 for _ in range(2)])

    with pytest.raises(ValueError, match="at least 5"):
        compute_and_store_baseline(5, "v1", session=session)


def test_landmark_with_all_none_yields_none(session):
    """A landmark that is None in every used row produces None statistics."""
    # Landmark index 1 is always None.
    score_rows = [
        [0.10, None, 0.30, 0.40, 0.50],
        [0.20, None, 0.40, 0.50, 0.60],
        [0.30, None, 0.50, 0.60, 0.70],
    ]
    _insert_logs(session, score_rows)

    baseline = compute_and_store_baseline(3, "v1", session=session)

    assert baseline.mean[1] is None
    assert baseline.std[1] is None
    assert all(baseline.quantiles[str(pct)][1] is None for pct in (5, 25, 50, 75, 95))
    # A landmark with samples is still computed normally.
    assert baseline.mean[0] == pytest.approx(0.2)


def test_partial_none_landmark_ignores_missing(session):
    """Per landmark, None values are dropped before computing statistics."""
    # Landmark 0 has one None among three rows; mean should use the two values.
    score_rows = [
        [0.10, 0.5, 0.5, 0.5, 0.5],
        [None, 0.5, 0.5, 0.5, 0.5],
        [0.30, 0.5, 0.5, 0.5, 0.5],
    ]
    _insert_logs(session, score_rows)

    baseline = compute_and_store_baseline(3, "v1", session=session)

    assert baseline.mean[0] == pytest.approx(0.2)  # mean of 0.10 and 0.30


def test_non_positive_n_raises(session):
    """n must be a positive integer."""
    with pytest.raises(ValueError, match="positive"):
        compute_and_store_baseline(0, "v1", session=session)
