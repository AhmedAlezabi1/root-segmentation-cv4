"""
tests/test_api.py
=================
Unit tests for the cv4 FastAPI application (``cv4.api.app``).

These tests cover the HTTP layer only. The pipeline (``cv4.pipeline``) and
model loading (``cv4.inference.load_model``) are mocked out in every test,
so no GPU, no model weights, and no real images are required to run the
suite. This makes the tests fast (~10s) and runnable in CI without any
ML infrastructure.

Test strategy
-------------
Each test targets a single behaviour of the API layer:
    - Correct HTTP status codes for valid and invalid requests
    - Authentication enforcement (missing / invalid JWT cookie)
    - Content-type validation (wrong type, missing type)
    - Response schema correctness (field presence, field values)
    - Health endpoint behaviour under degraded startup
    - Feedback endpoint (201 success, 404 unknown prediction, 401 auth)

Mocking approach
----------------
``run_pipeline`` and ``load_model`` are patched at the point of use
(``cv4.api.app.run_pipeline``, ``cv4.api.app.load_model``) rather than
at the source module. This is the correct pattern for ``unittest.mock``
and ensures the patches apply to the references already imported into
``app.py`` at import time.

The database session is overridden via FastAPI's dependency override
mechanism. A SQLite in-memory database is created for each test session,
with the schema applied via Alembic's migration metadata. This means
tests run against a real database without touching the dev ``feedback.db``.

The ``client`` fixture patches both pipeline functions and uses FastAPI's
``TestClient`` (backed by ``httpx``) to make in-process HTTP requests
without starting a real server. This means no ports are opened and no
network calls are made.

Environment setup
-----------------
``CV4_ENV`` is set to ``"development"`` before ``app`` (and ``cv4.api.auth``)
is imported. In development mode ``cv4.api.auth`` falls back to a hardcoded
JWT secret when ``JWT_SECRET_KEY`` is unset, and the auth cookies are not
marked Secure, so the in-process test client stores and resends them over
plain HTTP. Without it, importing the auth module raises ``RuntimeError`` for
the missing secret, so the ``os.environ`` assignment must precede the
``from cv4.api.app`` import below.

Running the tests
-----------------
From the project root::

    uv run pytest tests/test_api.py -v

Dependencies: ``pytest``, ``httpx`` (required by Starlette's TestClient).
Both are listed as dev dependencies in ``pyproject.toml``.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# CV4_ENV must be set before importing app, because cv4.api.auth resolves the
# JWT secret at import time and raises RuntimeError unless either JWT_SECRET_KEY
# or CV4_ENV=development is set. In development it also makes the auth cookies
# non-Secure so the test client stores and resends them over plain HTTP.
os.environ["CV4_ENV"] = "development"

from cv4.api.app import app  # noqa: E402 - import after env var setup is intentional
from cv4.db import Base, get_session  # noqa: E402

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

# A user registered and logged in by the `client` fixture so that requests
# carry a valid JWT access_token cookie. Authentication is JWT-only; there is
# no API key.
TEST_USER = {
    "email": "tester@example.com",
    "password": "test-password-123",
    "password_confirm": "test-password-123",
}

# ---------------------------------------------------------------------------
# Mock pipeline result
# ---------------------------------------------------------------------------
# Mimics the dict structure returned by cv4.pipeline.run_pipeline.
# Contains 5 plant entries (lanes 0-4) matching the pipeline contract:
#   - Lane 0: root detected, tip at (145, 890), confidence 0.91
#   - Lanes 1-4: no root detected, all measurement fields None
#
# root_tip_mm and main_root_path are intentionally excluded - they are not
# part of the current response schema and are not consumed by app.py.
MOCK_PIPELINE_RESULT = {
    "plants": [
        {
            "lane": i,
            "has_root": i == 0,
            # root_tip_px is a dict with "x" and "y" keys, matching
            # the structure expected by RootTipPx(**plant["root_tip_px"])
            "root_tip_px": {"x": 145, "y": 890} if i == 0 else None,
            # confidence is None for empty lanes, matching _empty_plant_entry
            "confidence": 0.91 if i == 0 else None,
        }
        for i in range(5)
    ],
    # Key name matches pipeline output; app.py maps it to processing_time_s
    "processing_time_seconds": 1.23,
    # Bounding box of the petri dish crop within the original image.
    # (x, y, side) — matches the tuple returned by preprocessing.preprocess_image.
    "plate_bbox": (0, 0, 1024),
}


# ---------------------------------------------------------------------------
# In-memory database setup
# ---------------------------------------------------------------------------
# Creates a fresh SQLite in-memory database for each test session.
# The schema is created directly from the SQLAlchemy metadata so tests
# don't depend on Alembic being run first.
# The get_db dependency is overridden on the app so all endpoints use
# this test database instead of the real feedback.db.

TEST_DATABASE_URL = "sqlite:///:memory:"

# StaticPool forces all connections to share the same in-memory database.
# Without it, each new SQLite connection gets a fresh empty database, so
# tables created by Base.metadata.create_all are invisible to the session
# the app uses inside the request handler.
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Test database session dependency override.

    Replaces the real get_db dependency with an in-memory SQLite session
    so tests never touch the development feedback.db file.
    """
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create all tables before each test and drop them after.

    autouse=True means this fixture runs for every test automatically
    without needing to be declared as a parameter. This ensures each
    test starts with a clean empty database.
    """
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_session] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def anon_client():
    """Provide an unauthenticated TestClient with pipeline and model mocked out.

    Use this for tests that exercise the unauthenticated path (no cookie) or
    that deliberately send an invalid cookie. No login is performed, so
    requests to protected endpoints are rejected with 401.

    Patches:
        cv4.api.app.run_pipeline -> returns ``MOCK_PIPELINE_RESULT``
        cv4.api.app.load_model   -> returns a ``MagicMock`` (model is truthy)

    Yields
    ------
    TestClient
        In-process HTTP client bound to the FastAPI app, with no auth cookie.
    """
    with (
        patch("cv4.api.app.run_pipeline", return_value=MOCK_PIPELINE_RESULT),
        patch("cv4.api.app.load_model", return_value=MagicMock()),
    ):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client():
    """Provide an authenticated TestClient (registered + logged in via JWT).

    Registers ``TEST_USER`` and logs in, so the httpOnly ``access_token``
    cookie is stored on the client and sent automatically on every subsequent
    request. Used by all tests that need an authenticated caller. The pipeline
    and model loading are mocked out exactly as in ``anon_client``.

    Patches:
        cv4.api.app.run_pipeline -> returns ``MOCK_PIPELINE_RESULT``
        cv4.api.app.load_model   -> returns a ``MagicMock`` (model is truthy)

    Yields
    ------
    TestClient
        In-process HTTP client carrying a valid JWT access cookie.
    """
    with (
        patch("cv4.api.app.run_pipeline", return_value=MOCK_PIPELINE_RESULT),
        patch("cv4.api.app.load_model", return_value=MagicMock()),
    ):
        with TestClient(app) as c:
            c.post("/auth/register", json=TEST_USER)
            login = c.post(
                "/auth/login",
                json={"email": TEST_USER["email"], "password": TEST_USER["password"]},
            )
            assert login.status_code == 200, f"test login failed: {login.status_code} {login.text}"
            yield c


@pytest.fixture
def prediction_id(client):
    """Create a real prediction in the test database via POST /predict.

    Used by feedback tests that need a valid prediction_id to reference.
    Makes a real /predict request through the test client so the
    prediction row is inserted by the actual endpoint logic.

    Returns
    -------
    int
        The prediction_id returned by the /predict response.
    """
    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    response = client.post(
        "/predict",
        files={"image": ("test.png", fake_image, "image/png")},
    )
    assert response.status_code == 200
    return response.json()["prediction_id"]


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


def test_health(client):
    """GET /health returns 200 and correct body when model is loaded.

    Verifies that a healthy container reports status "ok". The ``client``
    fixture mocks ``load_model`` to return a truthy object, simulating a
    successful model load at startup.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_model_not_loaded():
    """GET /health returns 503 when model failed to load at startup.

    Verifies that the health endpoint returns 503 (not 200) when
    ``app.state.model`` is None. This is the behaviour relied upon by
    Docker HEALTHCHECK and container orchestrators to avoid routing
    traffic to a container that cannot serve /predict requests.

    This test does NOT use the ``client`` fixture because the fixture
    mocks load_model to succeed. Instead, it patches load_model to raise
    FileNotFoundError, simulating a missing weights file at startup.
    """
    with patch("cv4.api.app.load_model", side_effect=FileNotFoundError("weights not found")):
        with TestClient(app) as c:
            response = c.get("/health")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Predict endpoint - success path
# ---------------------------------------------------------------------------


def test_predict_valid(client):
    """POST /predict returns 200 and correct JSON for a valid authenticated request.

    Verifies the full happy path:
        - An authenticated request (valid JWT cookie) is accepted
        - Response status is 200
        - Response contains a prediction_id
        - Response contains exactly 5 root entries (one per lane)
        - Lane 0 has has_root=True and correct root_tip_px coordinates
        - root_tip_px matches the mock pipeline output {x: 145, y: 890}

    The image bytes are a minimal PNG header followed by null bytes.
    The content is never decoded - run_pipeline is mocked - but the
    content-type header must be image/png to pass validation.
    """
    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # valid PNG magic bytes
    response = client.post(
        "/predict",
        files={"image": ("test.png", fake_image, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert isinstance(data["prediction_id"], int)
    assert len(data["roots"]) == 5
    assert data["roots"][0]["has_root"] is True
    assert data["roots"][0]["root_tip_px"] == {"x": 145, "y": 890}


# ---------------------------------------------------------------------------
# Predict endpoint - authentication tests
# ---------------------------------------------------------------------------


def test_predict_unauthenticated(anon_client):
    """POST /predict returns 401 when the request carries no auth cookie.

    Verifies that unauthenticated requests are rejected by the
    ``get_current_user`` dependency before the pipeline is called — a 401,
    not a 422 (which would indicate a schema/validation failure).
    """
    response = anon_client.post(
        "/predict",
        files={"image": ("test.png", b"...", "image/png")},
    )
    assert response.status_code == 401


def test_predict_invalid_cookie(anon_client):
    """POST /predict returns 401 when the access_token cookie is invalid.

    Verifies that a malformed or unverifiable JWT is rejected with 401 by
    the token-decoding path in ``get_current_user``.
    """
    anon_client.cookies.set("access_token", "not-a-valid-jwt")
    response = anon_client.post(
        "/predict",
        files={"image": ("test.png", b"...", "image/png")},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Predict endpoint - content-type validation tests
# ---------------------------------------------------------------------------


def test_predict_wrong_content_type(client):
    """POST /predict returns 400 when content-type is not in the allowlist.

    Verifies that non-image uploads are rejected before any disk I/O
    occurs. The content-type check runs before the temp file is written,
    so even a large non-image upload is rejected cheaply.
    """
    response = client.post(
        "/predict",
        files={"image": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert response.status_code == 400


def test_predict_no_content_type(client):
    """POST /predict returns 400 when no explicit image content-type is set.

    When a multipart field has no Content-Type header, FastAPI defaults
    image.content_type to "application/octet-stream" rather than None or
    empty string. This test sends that value explicitly to verify the
    allowlist check rejects it correctly.
    """
    response = client.post(
        "/predict",
        files={"image": ("test.png", b"...", "application/octet-stream")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Feedback endpoint tests
# ---------------------------------------------------------------------------


def test_feedback_valid_correction(client, prediction_id):
    """POST /feedback returns 201 with feedback IDs for a valid correction.

    Verifies the full happy path for a correction (is_correct=False):
        - An authenticated request (valid JWT cookie) is accepted
        - Response status is 201
        - Response contains the prediction_id
        - Response contains one submitted entry with a feedback_id
        - feedback_id is an integer assigned by the database
    """
    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "lanes": [
                {
                    "lane": 0,
                    "is_correct": False,
                    "predicted_tip_x": 145,
                    "predicted_tip_y": 890,
                    "correct_tip_x": 150,
                    "correct_tip_y": 895,
                    "comment": "tip was slightly lower",
                }
            ],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["prediction_id"] == prediction_id
    assert len(data["submitted"]) == 1
    assert data["submitted"][0]["lane"] == 0
    assert isinstance(data["submitted"][0]["feedback_id"], int)


def test_feedback_valid_confirmation(client, prediction_id):
    """POST /feedback returns 201 for a confirmation (is_correct=True).

    Verifies that confirming a correct prediction is accepted without
    requiring correct_tip coordinates.
    """
    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "lanes": [
                {
                    "lane": 0,
                    "is_correct": True,
                    "predicted_tip_x": 145,
                    "predicted_tip_y": 890,
                }
            ],
        },
    )
    assert response.status_code == 201


def test_feedback_unknown_prediction(client):
    """POST /feedback returns 404 when prediction_id does not exist.

    Verifies that submitting feedback for a non-existent prediction
    returns 404 rather than 500, giving the client a clear error.
    """
    response = client.post(
        "/feedback",
        json={
            "prediction_id": 99999,
            "lanes": [
                {
                    "lane": 0,
                    "is_correct": True,
                }
            ],
        },
    )
    assert response.status_code == 404


def test_feedback_unauthenticated(anon_client):
    """POST /feedback returns 401 when the request carries no auth cookie.

    Authentication is enforced before the prediction lookup, so no real
    prediction needs to exist for this check.
    """
    response = anon_client.post(
        "/feedback",
        json={"prediction_id": 1, "lanes": [{"lane": 0, "is_correct": True}]},
    )
    assert response.status_code == 401


def test_feedback_correction_missing_coordinates(client, prediction_id):
    """POST /feedback returns 422 when is_correct=False but no coordinates given.

    Verifies that the LaneFeedback Pydantic validator rejects a correction
    without correct_tip_x and correct_tip_y, returning 422 Unprocessable
    Content with a clear validation error before the endpoint runs.
    """
    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "lanes": [
                {
                    "lane": 0,
                    "is_correct": False,
                    # correct_tip_x and correct_tip_y intentionally omitted
                }
            ],
        },
    )
    assert response.status_code == 422


def test_feedback_multiple_lanes(client, prediction_id):
    """POST /feedback returns 201 with one result per submitted lane.

    Verifies that submitting feedback for multiple lanes in one request
    produces one feedback_id per lane in the response.
    """
    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "lanes": [
                {"lane": 0, "is_correct": True, "predicted_tip_x": 145, "predicted_tip_y": 890},
                {"lane": 1, "is_correct": True, "predicted_tip_x": 300, "predicted_tip_y": 750},
            ],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert len(data["submitted"]) == 2
    assert data["submitted"][0]["lane"] == 0
    assert data["submitted"][1]["lane"] == 1
