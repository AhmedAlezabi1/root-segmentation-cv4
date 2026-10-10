"""
api/dashboard.py
================
Dashboard metric endpoints for the cv4 stakeholder dashboard.

All endpoints are authenticated via the standard JWT cookie dependency
(``get_current_user``) and are mounted on the FastAPI ``app`` instance
in ``app.py`` via ``app.include_router(dashboard_router)``.

Metrics served
--------------
GET /dashboard/summary
    High-level KPIs: total predictions, total feedback submissions,
    overall correction rate, average confidence, model version.

GET /dashboard/activity
    Images processed per day for the last 30 days.

GET /dashboard/confidence
    Average confidence per day for the last 30 days (trend line).

GET /dashboard/correction-rate
    Correction rate per day for the last 30 days.

GET /dashboard/drift-status
    A simple drift indicator based on the 7-day vs 30-day correction rate.
    Returns green / yellow / red with an explanation.

Caching
-------
All endpoints use a module-level in-memory cache keyed by endpoint name.
Each cache entry stores the computed result and the UTC timestamp of
computation. Entries are invalidated after ``CACHE_TTL_SECONDS`` (default
300 s = 5 minutes). This avoids re-running the same aggregation queries
on every page refresh without requiring Redis or any external service.

The cache is bypassed when ``force_refresh=true`` is passed as a query
parameter, allowing a manual refresh button on the frontend to work.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from cv4.api.auth import get_current_user
from cv4.api.models import Feedback, Prediction, User
from cv4.db import get_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

dashboard_router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS: int = int(os.environ.get("DASHBOARD_CACHE_TTL_SECONDS", "300"))

_cache: dict[str, dict[str, Any]] = {}


def _get_cached(key: str) -> Any | None:
    """Return cached value if it exists and has not expired, else None."""
    entry = _cache.get(key)
    if entry is None:
        return None
    age = (datetime.now(timezone.utc) - entry["cached_at"]).total_seconds()
    if age > CACHE_TTL_SECONDS:
        return None
    return entry["value"]


def _set_cached(key: str, value: Any) -> None:
    """Store a value in the cache with the current UTC timestamp."""
    _cache[key] = {"value": value, "cached_at": datetime.now(timezone.utc)}


# ---------------------------------------------------------------------------
# Shared query helpers
# ---------------------------------------------------------------------------


def _predictions_last_n_days(db: Session, days: int) -> list[Prediction]:
    """Return all predictions created within the last *days* days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return db.query(Prediction).filter(Prediction.created_at >= cutoff).all()


def _feedback_last_n_days(db: Session, days: int) -> list[Feedback]:
    """Return all feedback rows created within the last *days* days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return db.query(Feedback).filter(Feedback.created_at >= cutoff).all()


def _date_key(dt: datetime) -> str:
    """Format a datetime as a YYYY-MM-DD string for grouping."""
    return dt.strftime("%Y-%m-%d")


def _confidence_from_prediction(prediction: Prediction) -> list[float]:
    """Extract all non-None confidence values from a prediction's results JSON."""
    return [lane["confidence"] for lane in (prediction.results or []) if lane.get("has_root") and lane.get("confidence") is not None]


def _format_model_version() -> str:
    """Return the active model version formatted as X.X.

    The major version is the integer model registration number in Azure ML,
    read from ``CV4_MODEL_VERSION`` (set once per registered model, e.g.
    ``CV4_MODEL_VERSION=1``).

    The minor version encodes the active deployment:

    * ``blue``  → ``.0``  (stable / production deployment)
    * ``green`` → ``.1``  (canary / new deployment)
    * unset     → ``.0``  (default)

    Examples
    --------
    ``CV4_MODEL_VERSION=1``, ``AZURE_ML_DEPLOYMENT_NAME=blue``  → ``"1.0"``
    ``CV4_MODEL_VERSION=1``, ``AZURE_ML_DEPLOYMENT_NAME=green`` → ``"1.1"``
    ``CV4_MODEL_VERSION=2``, ``AZURE_ML_DEPLOYMENT_NAME=green`` → ``"2.1"``

    In local mode (``CV4_INFERENCE_MODE != remote``) the raw value of
    ``CV4_MODEL_VERSION`` is returned as-is so dev environments show
    something meaningful without requiring a deployment name.
    """
    mode = os.environ.get("CV4_INFERENCE_MODE", "local").strip().lower()
    if mode != "remote":
        return os.environ.get("CV4_MODEL_VERSION", "local")

    major = os.environ.get("CV4_MODEL_VERSION", "1")
    deployment = os.environ.get("AZURE_ML_DEPLOYMENT_NAME", "").strip().lower()
    minor = "1" if deployment == "green" else "0"
    return f"{major}.{minor}"


# ---------------------------------------------------------------------------
# GET /dashboard/summary
# ---------------------------------------------------------------------------


@dashboard_router.get(
    "/summary",
    summary="High-level KPI summary",
    description=(
        "Returns total predictions, total feedback submissions, overall correction rate, "
        "average model confidence, and the currently deployed model version. "
        "Results are cached for 5 minutes."
    ),
)
def get_summary(
    force_refresh: bool = Query(False, description="Bypass cache and recompute."),
    db: Session = Depends(get_session),
    _user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return high-level KPI metrics for the stakeholder dashboard.

    Returns
    -------
    dict
        ``total_predictions``, ``total_feedback_lanes``, ``correction_rate_pct``,
        ``avg_confidence_pct``, ``model_version``, ``cached_at``.
    """
    cache_key = "summary"
    if not force_refresh:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    total_predictions: int = db.query(Prediction).count()
    total_feedback: int = db.query(Feedback).count()
    incorrect: int = db.query(Feedback).filter(Feedback.is_correct.is_(False)).count()

    correction_rate = round((incorrect / total_feedback * 100) if total_feedback > 0 else 0.0, 1)

    # Compute average confidence across all lanes in all predictions
    all_predictions = db.query(Prediction).all()
    confidences: list[float] = []
    for p in all_predictions:
        confidences.extend(_confidence_from_prediction(p))
    avg_confidence = round((sum(confidences) / len(confidences) * 100) if confidences else 0.0, 1)

    result = {
        "total_predictions": total_predictions,
        "total_feedback_lanes": total_feedback,
        "correction_rate_pct": correction_rate,
        "avg_confidence_pct": avg_confidence,
        "model_version": _format_model_version(),
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    _set_cached(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# GET /dashboard/activity
# ---------------------------------------------------------------------------


@dashboard_router.get(
    "/activity",
    summary="Images processed per day (last 30 days)",
    description=(
        "Returns a list of {date, count} entries for the last 30 days, "
        "showing how many images were processed each day. Days with no "
        "activity are included as zero so charts render continuous lines."
    ),
)
def get_activity(
    force_refresh: bool = Query(False, description="Bypass cache and recompute."),
    db: Session = Depends(get_session),
    _user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return images processed per day for the last 30 days.

    Returns
    -------
    dict
        ``data``: list of ``{date, count}`` dicts sorted ascending by date.
        ``cached_at``: ISO timestamp of when this result was computed.
    """
    cache_key = "activity"
    if not force_refresh:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    predictions = _predictions_last_n_days(db, 30)

    counts: dict[str, int] = defaultdict(int)
    for p in predictions:
        counts[_date_key(p.created_at)] += 1

    # Fill in zeros for days with no activity
    data = []
    for i in range(30):
        date = _date_key(datetime.now(timezone.utc) - timedelta(days=29 - i))
        data.append({"date": date, "count": counts.get(date, 0)})

    result = {"data": data, "cached_at": datetime.now(timezone.utc).isoformat()}
    _set_cached(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# GET /dashboard/confidence
# ---------------------------------------------------------------------------


@dashboard_router.get(
    "/confidence",
    summary="Average confidence trend (last 30 days)",
    description=(
        "Returns average model confidence per day for the last 30 days. "
        "Confidence is the model's probability at the detected root tip pixel, "
        "expressed as a percentage. Only lanes where a root was detected are included. "
        "Days with no predictions show null."
    ),
)
def get_confidence(
    force_refresh: bool = Query(False, description="Bypass cache and recompute."),
    db: Session = Depends(get_session),
    _user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return average confidence per day for the last 30 days.

    Returns
    -------
    dict
        ``data``: list of ``{date, avg_confidence_pct}`` dicts sorted ascending.
        ``cached_at``: ISO timestamp of when this result was computed.
    """
    cache_key = "confidence"
    if not force_refresh:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    predictions = _predictions_last_n_days(db, 30)

    daily: dict[str, list[float]] = defaultdict(list)
    for p in predictions:
        day = _date_key(p.created_at)
        daily[day].extend(_confidence_from_prediction(p))

    data = []
    for i in range(30):
        date = _date_key(datetime.now(timezone.utc) - timedelta(days=29 - i))
        values = daily.get(date)
        avg = round(sum(values) / len(values) * 100, 1) if values else None
        data.append({"date": date, "avg_confidence_pct": avg})

    result = {"data": data, "cached_at": datetime.now(timezone.utc).isoformat()}
    _set_cached(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# GET /dashboard/correction-rate
# ---------------------------------------------------------------------------


@dashboard_router.get(
    "/correction-rate",
    summary="Correction rate per day (last 30 days)",
    description=(
        "Returns the percentage of feedback lanes marked as incorrect per day "
        "for the last 30 days. A rising correction rate may indicate model drift "
        "or a change in image quality. Days with no feedback show null."
    ),
)
def get_correction_rate(
    force_refresh: bool = Query(False, description="Bypass cache and recompute."),
    db: Session = Depends(get_session),
    _user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return correction rate per day for the last 30 days.

    Returns
    -------
    dict
        ``data``: list of ``{date, correction_rate_pct}`` dicts sorted ascending.
        ``cached_at``: ISO timestamp of when this result was computed.
    """
    cache_key = "correction_rate"
    if not force_refresh:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    feedback = _feedback_last_n_days(db, 30)

    daily_total: dict[str, int] = defaultdict(int)
    daily_incorrect: dict[str, int] = defaultdict(int)
    for f in feedback:
        day = _date_key(f.created_at)
        daily_total[day] += 1
        if not f.is_correct:
            daily_incorrect[day] += 1

    data = []
    for i in range(30):
        date = _date_key(datetime.now(timezone.utc) - timedelta(days=29 - i))
        total = daily_total.get(date, 0)
        incorrect = daily_incorrect.get(date, 0)
        rate = round(incorrect / total * 100, 1) if total > 0 else None
        data.append({"date": date, "correction_rate_pct": rate})

    result = {"data": data, "cached_at": datetime.now(timezone.utc).isoformat()}
    _set_cached(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# GET /dashboard/drift-status
# ---------------------------------------------------------------------------


@dashboard_router.get(
    "/drift-status",
    summary="Model drift indicator",
    description=(
        "Compares the 7-day correction rate to the 30-day correction rate to detect "
        "potential model drift. Returns a status of 'green', 'yellow', or 'red' with "
        "a plain-English explanation suitable for non-technical stakeholders. "
        "Green: correction rate is stable. Yellow: rate has risen noticeably. "
        "Red: rate has risen significantly — model retraining may be needed."
    ),
)
def get_drift_status(
    force_refresh: bool = Query(False, description="Bypass cache and recompute."),
    db: Session = Depends(get_session),
    _user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return a traffic-light drift indicator based on correction rate trends.

    Compares the correction rate over the last 7 days against the last 30 days.
    A significant rise suggests the model is making more mistakes than usual,
    which may indicate drift in the input data distribution.

    Thresholds
    ----------
    - green  : 7-day rate ≤ 30-day rate + 5 percentage points
    - yellow : 7-day rate > 30-day rate + 5 pp but ≤ 30-day rate + 15 pp
    - red    : 7-day rate > 30-day rate + 15 pp

    Returns
    -------
    dict
        ``status`` ('green'|'yellow'|'red'), ``explanation``,
        ``rate_7d_pct``, ``rate_30d_pct``, ``cached_at``.
    """
    cache_key = "drift_status"
    if not force_refresh:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    def _rate(feedback: list[Feedback]) -> float | None:
        if not feedback:
            return None
        incorrect = sum(1 for f in feedback if not f.is_correct)
        return round(incorrect / len(feedback) * 100, 1)

    feedback_30d = _feedback_last_n_days(db, 30)
    feedback_7d = _feedback_last_n_days(db, 7)

    rate_30d = _rate(feedback_30d)
    rate_7d = _rate(feedback_7d)

    if rate_7d is None or rate_30d is None:
        status = "green"
        explanation = "Not enough feedback data to detect drift yet."
    else:
        delta = rate_7d - rate_30d
        if delta <= 5:
            status = "green"
            explanation = (
                f"The model is performing consistently. "
                f"The correction rate over the last 7 days ({rate_7d}%) is in line "
                f"with the 30-day average ({rate_30d}%)."
            )
        elif delta <= 15:
            status = "yellow"
            explanation = (
                f"The correction rate has risen slightly. "
                f"The 7-day rate ({rate_7d}%) is {delta:.1f} percentage points above "
                f"the 30-day average ({rate_30d}%). Worth monitoring."
            )
        else:
            status = "red"
            explanation = (
                f"The correction rate has risen significantly. "
                f"The 7-day rate ({rate_7d}%) is {delta:.1f} percentage points above "
                f"the 30-day average ({rate_30d}%). Model retraining may be needed."
            )

    result = {
        "status": status,
        "explanation": explanation,
        "rate_7d_pct": rate_7d,
        "rate_30d_pct": rate_30d,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    _set_cached(cache_key, result)
    return result
