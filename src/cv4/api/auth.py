"""
api/auth.py
===========
Authentication utilities for the cv4 JWT multi-user system.

Responsibilities
----------------
Password hashing
    ``hash_password`` / ``verify_password`` use PBKDF2-HMAC-SHA256 via
    Python's built-in ``hashlib`` and ``secrets`` — no extra dependency.
    The digest is stored as ``pbkdf2_sha256$<iterations>$<salt>$<hash>``
    so the scheme and parameters are self-describing and upgradeable.

JWT issuance and verification
    ``create_access_token`` signs a short-lived access token (default 30 min).
    ``create_refresh_token`` signs a longer-lived refresh token (default 7 days).
    ``decode_token`` verifies signature, expiry, and the ``type`` claim,
    returning the payload dict or raising ``HTTPException 401``.

FastAPI dependency
    ``get_current_user`` is a dependency that reads the ``access_token``
    httpOnly cookie, decodes it, and returns the matching ``User`` ORM row.
    Inject it with ``Depends(get_current_user)`` on any protected endpoint.

Environment variables
---------------------
JWT_SECRET_KEY : str, required
    Secret used to sign all tokens. Must be at least 32 characters.
    Generate with: ``python -c "import secrets; print(secrets.token_hex(32))"``
    If absent and ``CV4_ENV=development``, a hardcoded dev secret is used
    with a logged warning. In any other environment the server refuses to start.

JWT_ALGORITHM : str, optional
    Algorithm passed to PyJWT. Defaults to ``HS256``.

ACCESS_TOKEN_EXPIRE_MINUTES : int, optional
    Lifetime of access tokens in minutes. Defaults to 30.

REFRESH_TOKEN_EXPIRE_DAYS : int, optional
    Lifetime of refresh tokens in days. Defaults to 7.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from cv4.api.models import User
from cv4.db import get_session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Random per process, so no secret is ever hard-coded. Dev tokens stop working after a restart.
_DEV_SECRET = secrets.token_hex(32)

_RAW_SECRET = os.environ.get("JWT_SECRET_KEY", "")
if _RAW_SECRET:
    JWT_SECRET_KEY: str = _RAW_SECRET
elif os.environ.get("CV4_ENV") == "development":
    logger.warning("JWT_SECRET_KEY not set — using a random per-process development secret. Never deploy this configuration.")
    JWT_SECRET_KEY = _DEV_SECRET
else:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is not set. "
        'Generate one with: python -c "import secrets; print(secrets.token_hex(32))" '
        "and add it to your .env file."
    )

ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# ---------------------------------------------------------------------------
# Password hashing  (PBKDF2-HMAC-SHA256, no external dependency)
# ---------------------------------------------------------------------------

_HASH_ALGORITHM = "sha256"
_ITERATIONS = 600_000  # OWASP 2023 recommendation for PBKDF2-SHA256


def hash_password(plain: str) -> str:
    """Return a salted PBKDF2-HMAC-SHA256 digest of *plain*.

    The returned string is self-describing::

        pbkdf2_sha256$600000$<hex-salt>$<hex-digest>

    Parameters
    ----------
    plain : str
        The plain-text password to hash.

    Returns
    -------
    str
        The ``algorithm$iterations$salt$hash`` encoded string suitable
        for storing in the database ``hashed_password`` column.
    """
    salt = secrets.token_hex(32)  # 256-bit salt
    digest = hashlib.pbkdf2_hmac(
        _HASH_ALGORITHM,
        plain.encode(),
        salt.encode(),
        _ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest}"


def verify_password(plain: str, stored: str) -> bool:
    """Verify *plain* against a stored ``hash_password`` digest.

    Uses ``hmac.compare_digest`` under the hood (via ``secrets.compare_digest``)
    to prevent timing attacks.

    Parameters
    ----------
    plain : str
        The plain-text password supplied by the user.
    stored : str
        The ``pbkdf2_sha256$…`` string retrieved from the database.

    Returns
    -------
    bool
        ``True`` if the password matches, ``False`` otherwise.
    """
    try:
        _algo, iterations_str, salt, expected_hex = stored.split("$")
        iterations = int(iterations_str)
    except (ValueError, AttributeError):
        # Malformed stored hash — fail closed
        return False

    candidate_hex = hashlib.pbkdf2_hmac(
        _HASH_ALGORITHM,
        plain.encode(),
        salt.encode(),
        iterations,
    ).hex()
    return secrets.compare_digest(candidate_hex, expected_hex)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _make_token(data: dict[str, Any], expires_delta: timedelta, token_type: str) -> str:
    """Internal helper — build and sign a JWT with an expiry and type claim."""
    payload = data.copy()
    payload.update(
        {
            "exp": datetime.now(timezone.utc) + expires_delta,
            "iat": datetime.now(timezone.utc),
            "type": token_type,
        }
    )
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: int, email: str) -> str:
    """Return a signed JWT access token for the given user.

    Parameters
    ----------
    user_id : int
        The ``users.id`` primary key.
    email : str
        The user's email address, embedded as the ``sub`` claim.

    Returns
    -------
    str
        Signed JWT string, valid for ``ACCESS_TOKEN_EXPIRE_MINUTES`` minutes.
    """
    return _make_token(
        {"sub": email, "user_id": user_id},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "access",
    )


def create_refresh_token(user_id: int, email: str) -> str:
    """Return a signed JWT refresh token for the given user.

    Parameters
    ----------
    user_id : int
        The ``users.id`` primary key.
    email : str
        The user's email address, embedded as the ``sub`` claim.

    Returns
    -------
    str
        Signed JWT string, valid for ``REFRESH_TOKEN_EXPIRE_DAYS`` days.
    """
    return _make_token(
        {"sub": email, "user_id": user_id},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "refresh",
    )


def decode_token(token: str, expected_type: str) -> dict[str, Any]:
    """Decode and validate a JWT, returning its payload.

    Parameters
    ----------
    token : str
        The raw JWT string to decode.
    expected_type : str
        Either ``"access"`` or ``"refresh"``. Raises 401 if the ``type``
        claim does not match.

    Returns
    -------
    dict[str, Any]
        The decoded payload dictionary.

    Raises
    ------
    HTTPException
        ``401 Unauthorized`` for any of: expired token, invalid signature,
        malformed token, or wrong token type.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise credentials_exc

    if payload.get("type") != expected_type:
        raise credentials_exc

    return payload


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_current_user(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(get_session),
) -> User:
    """FastAPI dependency that authenticates the request via httpOnly cookie.

    Reads the ``access_token`` httpOnly cookie set by ``POST /auth/login``,
    decodes it, and returns the corresponding ``User`` ORM row.

    Usage::

        @app.get("/me")
        def me(user: User = Depends(get_current_user)):
            return {"email": user.email}

    Parameters
    ----------
    access_token : str | None
        Injected automatically from the ``access_token`` httpOnly cookie.
    db : Session
        Database session injected by FastAPI.

    Returns
    -------
    User
        The authenticated, active user row from the database.

    Raises
    ------
    HTTPException
        ``401 Unauthorized`` if the cookie is absent, the token is invalid
        or expired, the user no longer exists, or the account is inactive.
    """
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(access_token, expected_type="access")

    user_id: int | None = payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is disabled.")

    return user


RESET_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("RESET_TOKEN_EXPIRE_MINUTES", "60"))


def create_reset_token(user_id: int, email: str) -> str:
    """Return a signed JWT password-reset token for the given user.

    The token has a ``type`` claim of ``"reset"`` so it cannot be used as
    an access or refresh token, and a short expiry (default 60 minutes).

    Parameters
    ----------
    user_id : int
        The ``users.id`` primary key.
    email : str
        The user's email address, embedded as the ``sub`` claim.

    Returns
    -------
    str
        Signed JWT string, valid for ``RESET_TOKEN_EXPIRE_MINUTES`` minutes.
    """
    return _make_token(
        {"sub": email, "user_id": user_id},
        timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES),
        "reset",
    )
