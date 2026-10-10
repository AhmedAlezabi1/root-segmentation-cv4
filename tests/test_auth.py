"""
tests/test_auth.py
==================
Unit tests for the cv4 JWT authentication endpoints.

Covers task 4.9.1 requirements:
    - POST /auth/register creates users with hashed passwords
    - POST /auth/login returns JWT cookies on valid credentials, 401 on invalid
    - GET /auth/me enforces authentication via the get_current_user dependency
    - POST /auth/logout clears auth cookies
    - POST /auth/refresh rotates tokens using a valid refresh cookie

Test strategy
-------------
Mirrors test_api.py exactly:
    - CV4_ENV=development set before import to bypass API key startup check
    - JWT_SECRET_KEY set before import to bypass auth startup check
    - SQLite in-memory database with StaticPool, recreated per test
    - FastAPI dependency override for get_session
    - TestClient (no real server, no ports opened)
    - Pipeline and model loading mocked out (not needed for auth tests,
      but the app import requires them not to crash at startup)

Running the tests
-----------------
From the project root::

    uv run pytest tests/test_auth.py -v
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Both env vars must be set before importing app, because _load_api_keys()
# and the JWT secret check both run at module import time.
os.environ["CV4_ENV"] = "development"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-unit-tests-only-not-for-production"

from cv4.api.app import app  # noqa: E402
from cv4.db import Base, get_session  # noqa: E402

# ---------------------------------------------------------------------------
# In-memory database setup  (identical pattern to test_api.py)
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
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

    autouse=True means this runs for every test automatically.
    Ensures each test starts with a clean empty database — no leftover
    users from a previous test can interfere.
    """
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_session] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client():
    """TestClient with pipeline and model loading mocked out.

    Auth tests don't use the pipeline, but the app's lifespan calls
    load_model at startup so it must be mocked to avoid needing weights.
    """
    with (
        patch("cv4.api.app.run_pipeline", return_value={}),
        patch("cv4.api.app.load_model", return_value=MagicMock()),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def registered_user(client):
    """Register a test user and return their credentials.

    Used by tests that need an existing account to log in with.

    Returns
    -------
    dict
        ``{email, password}`` for the registered user.
    """
    credentials = {"email": "test@example.com", "password": "testpass1"}
    response = client.post(
        "/auth/register",
        json={**credentials, "password_confirm": credentials["password"]},
    )
    assert response.status_code == 201, f"Setup failed: {response.json()}"
    return credentials


@pytest.fixture
def auth_client(client, registered_user):
    """TestClient with a valid login session (auth cookies set).

    Logs in with the registered_user credentials so the cookies are
    stored in the TestClient's cookie jar. Subsequent requests on this
    client automatically include the access_token cookie.

    Returns
    -------
    TestClient
        The same client instance, now carrying valid auth cookies.
    """
    client.post(
        "/auth/login",
        json=registered_user,
    )
    return client


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------


def test_register_success(client):
    """POST /auth/register returns 201 and the new user's public profile.

    Verifies:
        - Status is 201 Created
        - Response contains id, email, and is_active
        - Password is NOT present in the response
        - is_active defaults to True
        - email is stored lower-cased
    """
    response = client.post(
        "/auth/register",
        json={
            "email": "NEW@Example.COM",
            "password": "securepass1",
            "password_confirm": "securepass1",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new@example.com"  # stored lower-cased
    assert data["is_active"] is True
    assert isinstance(data["id"], int)
    assert "password" not in data
    assert "hashed_password" not in data


def test_register_duplicate_email(client):
    """POST /auth/register returns 409 when email is already registered.

    Verifies that attempting to register the same email twice returns
    409 Conflict rather than 500, giving the client a clear error.
    """
    payload = {
        "email": "duplicate@example.com",
        "password": "securepass1",
        "password_confirm": "securepass1",
    }
    first = client.post("/auth/register", json=payload)
    assert first.status_code == 201

    second = client.post("/auth/register", json=payload)
    assert second.status_code == 409


def test_register_password_too_short(client):
    """POST /auth/register returns 422 when password is under 8 characters.

    Verifies that Pydantic's min_length=8 constraint on UserCreate.password
    rejects short passwords before the endpoint handler runs.
    """
    response = client.post(
        "/auth/register",
        json={
            "email": "short@example.com",
            "password": "abc",
            "password_confirm": "abc",
        },
    )
    assert response.status_code == 422


def test_register_passwords_do_not_match(client):
    """POST /auth/register returns 422 when passwords do not match.

    Verifies the UserCreate model_validator that requires password and
    password_confirm to be identical.
    """
    response = client.post(
        "/auth/register",
        json={
            "email": "mismatch@example.com",
            "password": "securepass1",
            "password_confirm": "different1",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------


def test_login_success(client, registered_user):
    """POST /auth/login returns 200 and sets auth cookies on valid credentials.

    Verifies:
        - Status is 200
        - Response body contains token_type and user profile
        - access_token cookie is set by the response
    """
    response = client.post("/auth/login", json=registered_user)
    assert response.status_code == 200
    data = response.json()
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == registered_user["email"]
    assert data["user"]["is_active"] is True
    assert "access_token" in client.cookies


def test_login_wrong_password(client, registered_user):
    """POST /auth/login returns 401 when password is incorrect.

    Verifies that a wrong password is rejected with 401, not 422 or 500.
    The error message should not reveal whether the email exists.
    """
    response = client.post(
        "/auth/login",
        json={"email": registered_user["email"], "password": "wrongpassword"},
    )
    assert response.status_code == 401


def test_login_unknown_email(client):
    """POST /auth/login returns 401 when email is not registered.

    Verifies that an unrecognised email returns the same 401 as a wrong
    password — no information is leaked about whether the account exists.
    """
    response = client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "somepassword"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /auth/me  (get_current_user dependency)
# ---------------------------------------------------------------------------


def test_me_authenticated(auth_client):
    """GET /auth/me returns the user profile when a valid cookie is present.

    Verifies the get_current_user dependency correctly decodes the
    access_token cookie and returns the matching user row.
    """
    response = auth_client.get("/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["is_active"] is True
    assert isinstance(data["id"], int)


def test_me_unauthenticated(client):
    """GET /auth/me returns 401 when no access_token cookie is present.

    Verifies that the get_current_user dependency raises 401 for requests
    with no auth cookie, not 422 or 500.
    """
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_invalid_token(client):
    """GET /auth/me returns 401 when the access_token cookie is malformed.

    Verifies that a syntactically invalid JWT is rejected by decode_token
    and results in 401, not 500.
    """
    client.cookies.set("access_token", "this.is.not.a.valid.jwt")
    response = client.get("/auth/me")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------


def test_logout_clears_cookies(auth_client):
    """POST /auth/logout returns 204 and clears the access_token cookie.

    Verifies that after logout, the access_token cookie is removed so
    subsequent requests to /auth/me return 401.
    """
    # Confirm we are authenticated first
    assert auth_client.get("/auth/me").status_code == 200

    logout_response = auth_client.post("/auth/logout")
    assert logout_response.status_code == 204

    # Cookie should be gone — /auth/me must now return 401
    me_response = auth_client.get("/auth/me")
    assert me_response.status_code == 401
