"""
api/models.py
=============
SQLAlchemy ORM models for the cv4 feedback system.

Three tables are defined:

    predictions
        Stores the result of each POST /predict call. Created automatically
        when a prediction is made. Required so that feedback can reference
        the prediction it is correcting.

    feedback
        Stores one row per lane per feedback submission. Each row references
        a prediction via a foreign key and records whether the model was
        correct, the corrected tip coordinates (if wrong), and an optional
        comment.

    users
        Stores one row per registered user account. Passwords are stored
        as PBKDF2-HMAC-SHA256 digests — never in plain text.

    predictions_log
        Append-only log of the per-landmark confidence vector from each
        prediction, written for downstream drift detection.

    confidence_baseline
        Statistical summary (mean, std, key quantiles) of the per-landmark
        confidence distribution, computed from the first N logged predictions
        and tagged with a model version. Used as the reference for drift
        detection.

Schema changes must be made via Alembic migrations, not by modifying
these models directly without generating a corresponding migration.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cv4.db import Base


class Prediction(Base):
    """ORM model for the ``predictions`` table.

    Stores one row per POST /predict call. The full per-lane results are
    stored as a JSON blob in ``results`` so that the predictions table does
    not need to be normalised across lanes — feedback references the
    prediction at the lane level.

    Attributes
    ----------
    id : int
        Auto-incremented primary key.
    image_filename : str
        Original filename of the uploaded image as reported by the client.
        Not guaranteed to be unique — the same filename can appear in
        multiple predictions.
    results : list[dict]
        The full ``roots`` list from the PredictResponse, serialised as
        JSON. Each entry is a dict with keys: lane, has_root, root_tip_px,
        confidence.
    processing_time_s : float
        End-to-end wall-clock time in seconds reported by the pipeline.
    created_at : datetime
        UTC timestamp set automatically when the row is inserted.
    feedbacks : list[Feedback]
        Back-populated relationship to all feedback rows for this prediction.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    results: Mapped[list] = mapped_column(JSON, nullable=False)
    processing_time_s: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # One prediction has many feedback rows (one per lane submitted)
    feedbacks: Mapped[list[Feedback]] = relationship("Feedback", back_populates="prediction", cascade="all, delete-orphan")
    # At most one stored image per prediction (set when feedback is submitted)
    feedback_image: Mapped[FeedbackImage | None] = relationship(
        "FeedbackImage", back_populates="prediction", cascade="all, delete-orphan", uselist=False
    )


class Feedback(Base):
    """ORM model for the ``feedback`` table.

    Stores one row per lane per feedback submission. A user may submit
    feedback for one or more lanes from a single prediction in one request,
    producing one Feedback row per lane.

    Attributes
    ----------
    id : int
        Auto-incremented primary key.
    prediction_id : int
        Foreign key to ``predictions.id``. Required — feedback must
        reference an existing prediction.
    lane : int
        0-indexed lane number (0–4) this feedback applies to.
    is_correct : bool
        True if the user confirms the model's output was correct.
        False if the user is providing a correction.
    predicted_tip_x : int | None
        The x pixel coordinate the model returned for this lane.
        Copied from the prediction at submission time for convenience.
        None if the model returned no root for this lane.
    predicted_tip_y : int | None
        The y pixel coordinate the model returned for this lane.
        None if the model returned no root for this lane.
    correct_tip_x : int | None
        The x pixel coordinate the user says is the correct tip.
        None when is_correct is True (no correction needed).
    correct_tip_y : int | None
        The y pixel coordinate the user says is the correct tip.
        None when is_correct is True (no correction needed).
    comment : str | None
        Optional free-text comment from the user.
    created_at : datetime
        UTC timestamp set automatically when the row is inserted.
    prediction : Prediction
        Back-populated relationship to the parent prediction.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    lane: Mapped[int] = mapped_column(Integer, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_tip_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_tip_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correct_tip_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correct_tip_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    prediction: Mapped[Prediction] = relationship("Prediction", back_populates="feedbacks")


class User(Base):
    """ORM model for the ``users`` table.

    Stores one row per registered account. Passwords are never stored in
    plain text — only the PBKDF2-HMAC-SHA256 digest produced by
    ``cv4.api.auth.hash_password`` is persisted.

    Attributes
    ----------
    id : int
        Auto-incremented primary key.
    email : str
        Unique email address used as the login identifier.
        Stored lower-cased; enforced at the application layer.
    hashed_password : str
        ``pbkdf2_sha256$iterations$salt$hash`` string produced by
        ``cv4.api.auth.hash_password``. Never log or expose this field.
    is_active : bool
        False means the account is disabled and cannot log in.
        Defaults to True on registration.
    created_at : datetime
        UTC timestamp set automatically when the row is inserted.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(254), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class PredictionLog(Base):
    """ORM model for the ``predictions_log`` table.

    Append-only log of the per-landmark confidence vector produced by each
    POST /predict call, written for downstream drift detection. Kept separate
    from ``predictions`` so drift monitoring can query confidence cheaply
    without parsing the full ``predictions.results`` payload.

    The write is best-effort in the endpoint: a failure here never blocks or
    fails the prediction response.

    Attributes
    ----------
    id : int
        Auto-incremented primary key.
    prediction_id : str
        ``str(predictions.id)`` of the prediction this log row describes.
        Stored as a string; links back to the predictions table.
    confidence_scores : list
        The 5 per-landmark confidence scores (one per lane) as a JSON array.
        Always length 5; ``None`` placeholders are preserved for lanes with
        no detected root (a meaningful signal for drift detection).
    timestamp : datetime
        UTC timestamp set automatically when the row is inserted.
    image_reference : str | None
        Path or filename of the input image as reported by the client.
    """

    __tablename__ = "predictions_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    confidence_scores: Mapped[list] = mapped_column(JSON, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    image_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)


class ConfidenceBaseline(Base):
    """ORM model for the ``confidence_baseline`` table.

    A statistical summary of the per-landmark confidence distribution,
    computed from the first ``n_samples`` predictions logged in
    ``predictions_log`` after a deployment. Drift detection compares live
    confidence against this baseline, so each baseline is tagged with the
    model version it characterises.

    The arrays are per landmark (one value per lane, indices 0-4), matching
    the shape of ``predictions_log.confidence_scores``. They are stored as
    JSON for portability across SQLite and PostgreSQL.

    Attributes
    ----------
    id : int
        Auto-incremented primary key.
    model_version : str
        Identifier of the model whose predictions this baseline summarises.
    n_samples : int
        Number of ``predictions_log`` rows used to compute the statistics.
    mean : list
        Per-landmark mean confidence (length 5). A landmark with no
        confidence samples is ``None``.
    std : list
        Per-landmark standard deviation of confidence (length 5). A landmark
        with no confidence samples is ``None``.
    quantiles : dict
        Key percentiles of the confidence distribution per landmark, keyed by
        percentile label: ``{"5": [...], "25": [...], "50": [...],
        "75": [...], "95": [...]}``. Each value is a length-5 per-landmark
        list; entries are ``None`` for landmarks with no samples.
    computed_at : datetime
        UTC timestamp set automatically when the baseline row is inserted.
    """

    __tablename__ = "confidence_baseline"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    mean: Mapped[list] = mapped_column(JSON, nullable=False)
    std: Mapped[list] = mapped_column(JSON, nullable=False)
    quantiles: Mapped[dict] = mapped_column(JSON, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class DriftEvent(Base):
    """ORM model for the ``drift_events`` table.

    Append-only audit log written by the scheduled drift-detection job
    (:func:`cv4.monitoring.drift.run_drift_check`). Each run compares the last
    24 hours of logged confidence against the most recent
    ``confidence_baseline`` using a per-landmark KS test and inserts one row
    per landmark — whether or not drift was detected — so the table is a
    complete history of every check.

    Attributes
    ----------
    id : int
        Auto-incremented primary key.
    detected_at : datetime
        UTC timestamp set automatically when the row is inserted (the time the
        drift check ran).
    p_value : float | None
        The KS-test p-value for this landmark. ``None`` when the landmark could
        not be tested (too few rolling samples or a degenerate baseline); the
        reason is recorded in ``notes``.
    drift_detected : bool
        ``True`` when ``p_value`` fell below the drift threshold (0.05).
    landmark_index : int
        Which of the 5 landmarks (lanes) this row describes (0-4). ``-1`` is
        reserved for an aggregate row across all landmarks.
    notes : str | None
        Optional context, e.g. the number of rolling samples used or why a
        landmark was skipped.
    """

    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    drift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    landmark_index: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(String(255), nullable=True)


class FeedbackImage(Base):
    """ORM model for the ``feedback_images`` table.

    Stores one row per prediction for which the user has submitted feedback,
    recording where on disk the original uploaded image was saved. Images are
    written to ``CV4_FEEDBACK_IMAGE_DIR`` (default ``./feedback_images``) by
    ``POST /feedback/{prediction_id}/image`` and can be downloaded as a ZIP
    archive via ``GET /feedback/export``.

    At most one image is stored per prediction (the upload is idempotent —
    re-uploading for the same prediction_id overwrites the previous file and
    updates this row). The stored filename is ``{prediction_id}{extension}``
    so that names remain filesystem-safe regardless of the original filename.

    Attributes
    ----------
    id : int
        Auto-incremented primary key.
    prediction_id : int
        Foreign key to ``predictions.id``. Unique — one image per prediction.
    original_filename : str
        The filename as reported by the client (e.g. ``plate_42.png``).
        Stored for display purposes only; the file on disk uses a normalised
        name derived from ``prediction_id``.
    stored_path : str
        Absolute path to the image file on disk.
    content_type : str
        MIME type of the uploaded image (e.g. ``image/png``).
    created_at : datetime
        UTC timestamp set when the row was first inserted.
    updated_at : datetime
        UTC timestamp updated each time the image is re-uploaded.
    prediction : Prediction
        Back-populated relationship to the parent prediction.
    """

    __tablename__ = "feedback_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    prediction: Mapped[Prediction] = relationship("Prediction", back_populates="feedback_image")
