"""
api/schemas.py
==============
Pydantic request and response schemas for the cv4 feedback and auth endpoints.

Separated from app.py to keep the application file focused on routing
and request handling. These schemas are imported by app.py and used as
FastAPI endpoint parameter and response types.

Schemas
-------
LaneFeedback
    A single lane's feedback within a feedback submission request.

FeedbackRequest
    Request body for POST /feedback. Contains the prediction ID and a
    list of per-lane feedback entries.

FeedbackResponse
    Response body returned by a successful POST /feedback (HTTP 201).
    Contains the generated feedback IDs for each submitted lane.

UserCreate
    Request body for POST /auth/register.

UserLogin
    Request body for POST /auth/login.

UserResponse
    Public representation of a user, safe to return to the client.

TokenResponse
    Response body returned by a successful login or token refresh.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class LaneFeedback(BaseModel):
    """Feedback for a single plant lane within a prediction.

    Validation rules
    ----------------
    - If ``is_correct`` is False, ``correct_tip_x`` and ``correct_tip_y``
      must both be provided — a correction without coordinates is not useful.
    - If ``is_correct`` is True, ``correct_tip_x`` and ``correct_tip_y``
      are ignored even if provided.
    """

    lane: int = Field(
        ...,
        ge=0,
        le=4,
        description="0-indexed lane number (0–4) this feedback applies to.",
    )
    is_correct: bool = Field(
        ...,
        description=("True if the model's output for this lane was correct. False if you are providing a correction."),
    )
    predicted_tip_x: int | None = Field(
        None,
        description="The x coordinate the model returned. Copy from the /predict response.",
    )
    predicted_tip_y: int | None = Field(
        None,
        description="The y coordinate the model returned. Copy from the /predict response.",
    )
    correct_tip_x: int | None = Field(
        None,
        description="The correct x coordinate. Required when is_correct is False.",
    )
    correct_tip_y: int | None = Field(
        None,
        description="The correct y coordinate. Required when is_correct is False.",
    )
    comment: str | None = Field(
        None,
        max_length=1000,
        description="Optional free-text comment (max 1000 characters).",
    )

    @model_validator(mode="after")
    def correction_requires_coordinates(self) -> LaneFeedback:
        """Require correct_tip_x and correct_tip_y when is_correct is False.

        Raises
        ------
        ValueError
            If is_correct is False and either correct_tip_x or correct_tip_y
            is missing.
        """
        if not self.is_correct:
            if self.correct_tip_x is None or self.correct_tip_y is None:
                raise ValueError("correct_tip_x and correct_tip_y are required when is_correct is False.")
        return self


class CropBox(BaseModel):
    """Bounding box of the petri dish crop within the original full-resolution image.

    The root tip coordinates in ``RootResult.root_tip_px`` are expressed in the
    crop coordinate space (top-left = 0, 0). Add ``crop.x`` / ``crop.y`` to
    convert them to original image coordinates.
    """

    x: int = Field(..., description="Left edge of the crop in the original image (pixels).")
    y: int = Field(..., description="Top edge of the crop in the original image (pixels).")
    side: int = Field(..., description="Side length of the square crop (pixels).")


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback.

    A single submission can contain feedback for one or more lanes from
    the same prediction. Each lane produces one row in the feedback table.

    Example
    -------
    .. code-block:: json

        {
            "prediction_id": 42,
            "lanes": [
                {
                    "lane": 0,
                    "is_correct": false,
                    "predicted_tip_x": 145,
                    "predicted_tip_y": 890,
                    "correct_tip_x": 148,
                    "correct_tip_y": 895,
                    "comment": "Tip was slightly lower"
                },
                {
                    "lane": 1,
                    "is_correct": true,
                    "predicted_tip_x": 300,
                    "predicted_tip_y": 750
                }
            ]
        }
    """

    prediction_id: int = Field(
        ...,
        description="ID of the prediction this feedback refers to. Must exist in the database.",
    )
    lanes: list[LaneFeedback] = Field(
        ...,
        min_length=1,
        description="List of per-lane feedback entries. At least one lane is required.",
    )


class FeedbackLaneResult(BaseModel):
    """Result for a single lane within a feedback submission response."""

    lane: int = Field(..., description="Lane number this result corresponds to.")
    feedback_id: int = Field(..., description="Database ID assigned to this feedback row.")


class FeedbackResponse(BaseModel):
    """Response body returned by a successful POST /feedback (HTTP 201).

    Contains the generated feedback IDs for each submitted lane so the
    client can reference them in future requests if needed.
    """

    prediction_id: int = Field(..., description="ID of the prediction feedback was submitted for.")
    submitted: list[FeedbackLaneResult] = Field(
        ...,
        description="One entry per submitted lane, containing the assigned feedback ID.",
    )


# ---------------------------------------------------------------------------
# Auth schemas
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """Request body for POST /auth/register.

    Validation rules
    ----------------
    - ``email`` is stored lower-cased at the application layer.
    - ``password`` must be 8–128 characters.
    - ``password`` and ``password_confirm`` must match.
    """

    email: str = Field(..., description="Email address for the new account.")
    password: str = Field(..., min_length=8, max_length=128, description="Password (8–128 characters).")
    password_confirm: str = Field(..., description="Must match password.")

    @model_validator(mode="after")
    def passwords_match(self) -> "UserCreate":
        if self.password != self.password_confirm:
            raise ValueError("password and password_confirm do not match.")
        return self


class UserLogin(BaseModel):
    """Request body for POST /auth/login."""

    email: str = Field(..., description="Registered email address.")
    password: str = Field(..., description="Account password.")


class UserResponse(BaseModel):
    """Public representation of a user — safe to return to the client.

    Never includes ``hashed_password`` or other sensitive fields.
    """

    id: int
    email: str
    is_active: bool

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Response body returned by a successful login or token refresh.

    The actual tokens are delivered as httpOnly cookies by the endpoint;
    this body is informational only.
    """

    token_type: str = Field(default="bearer", description="Always 'bearer'.")
    user: UserResponse = Field(..., description="The authenticated user.")


# ---------------------------------------------------------------------------
# Password reset schemas
# ---------------------------------------------------------------------------


class PasswordResetRequest(BaseModel):
    """Request body for POST /auth/reset-password/request."""

    email: str = Field(..., description="Email address of the account to reset.")


class PasswordResetConfirm(BaseModel):
    """Request body for POST /auth/reset-password/confirm."""

    token: str = Field(..., description="Reset token returned by POST /auth/reset-password/request.")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password (8–128 characters).")
    new_password_confirm: str = Field(..., description="Must match new_password.")

    @model_validator(mode="after")
    def passwords_match(self) -> "PasswordResetConfirm":
        if self.new_password != self.new_password_confirm:
            raise ValueError("new_password and new_password_confirm do not match.")
        return self


class PasswordResetRequestResponse(BaseModel):
    """Response body for POST /auth/reset-password/request.

    In production this would simply confirm the email was sent without
    revealing whether the address is registered. In development the reset
    token is returned directly so it can be used without an email server.
    """

    message: str = Field(..., description="Human-readable status message.")
    reset_token: str | None = Field(
        None,
        description=("Reset token. Only present when CV4_ENV=development. Pass this to POST /auth/reset-password/confirm."),
    )
