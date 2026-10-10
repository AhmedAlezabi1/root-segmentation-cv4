"""
db.py
=====
Database engine and session factory for the cv4 application.

The connection URL is read from the ``DATABASE_URL`` environment variable.
A ``.env`` file in the project root is loaded automatically via
``python-dotenv``. ``DATABASE_URL`` is only required when something
actually touches the database (e.g. the feedback endpoint inserts a row).
Importing this module without ``DATABASE_URL`` set is fine; the check is
deferred until ``engine``, ``SessionLocal``, or ``get_session`` is used.

This matters during Sprint 2, when most of the app does not yet need a
database. From Sprint 4 onwards, setting ``DATABASE_URL`` (locally via
``.env``, in Docker via docker-compose) is enough to wire it back in.

SQLite is supported for local development. PostgreSQL (via psycopg) is
used in production. The correct driver is selected automatically from
the URL scheme.

See ``.env.example`` for a minimal local development configuration.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    """Parent class for all ORM models in the cv4 codebase.

    All models that should be picked up by Alembic autogenerate must
    inherit from this class and be imported in ``alembic/env.py``.
    """


# Singletons, built on first access via the helpers below.
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine() -> Engine:
    """Build the SQLAlchemy engine from ``DATABASE_URL``.

    Raises ``RuntimeError`` if the variable is missing, so callers get a
    clear error at the moment they try to use the database, not at import.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file or set it in your shell before using cv4.db. See .env.example for the correct format."
        )
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def get_engine() -> Engine:
    """Return the singleton engine, building it on first call."""
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _get_session_factory() -> sessionmaker[Session]:
    """Return the singleton session factory, building it on first call."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _session_factory


def __getattr__(name: str) -> Any:
    """PEP 562 module-level lazy attributes.

    Lets existing code keep using ``from cv4.db import engine`` and
    ``from cv4.db import SessionLocal``. The actual creation (and the
    ``DATABASE_URL`` check) is deferred to the moment of first access.
    """
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return _get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_session() -> Generator[Session, None, None]:
    """Yield a database session and close it when the caller is done.

    Designed for FastAPI dependency injection::

        @app.post("/feedback")
        def handler(db: Session = Depends(get_session)):
            ...

    The session is closed on exit even if the caller raises, which keeps
    connections from leaking back into the pool in a half-used state.
    """
    session = _get_session_factory()()
    try:
        yield session
    finally:
        session.close()
