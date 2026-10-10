"""
Integration tests that exercise the live database.

These tests require the Postgres container from docker-compose to be
running. They are skipped automatically if the database is not reachable,
so unit-test runs on CI (where the db is not up) do not fail.

Run the db first:

    docker compose -f docker/docker-compose.yml up -d db

Then run just these tests:

    uv run pytest tests/test_db_connection.py -v
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

try:
    from cv4.db import SessionLocal, engine
except RuntimeError as exc:
    pytest.skip(
        f"cv4.db could not be imported (DATABASE_URL missing): {exc}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def db_available() -> bool:
    """Skip the module if the database container is not running."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(f"Database not reachable: {exc}")
    return True


def test_engine_can_connect(db_available: bool) -> None:
    """The engine opens a connection and runs a trivial query."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_schema_has_expected_tables(db_available: bool) -> None:
    """Alembic migrations ran and created the tables we expect."""
    # alembic_version is included because its presence is what tells us
    # the schema was set up by Alembic rather than by an ad-hoc init.sql.
    expected = {"predictions", "feedback", "alembic_version"}
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).scalars().all()
    assert expected.issubset(set(rows))


def test_session_can_write_and_read(db_available: bool) -> None:
    """A session can insert a row, read it back, and the row survives commit."""
    image_filename = "test-db-connection.jpg"

    with SessionLocal() as session:
        # Clean up anything left over from a previous failed run so the
        # test is idempotent.
        session.execute(
            text("DELETE FROM predictions WHERE image_filename = :f"),
            {"f": image_filename},
        )
        session.commit()

        session.execute(
            text("INSERT INTO predictions (image_filename, results, processing_time_s) VALUES (:f, CAST(:r AS JSON), :t)"),
            {"f": image_filename, "r": "[]", "t": 0.0},
        )
        session.commit()

        row = session.execute(
            text("SELECT image_filename, processing_time_s FROM predictions WHERE image_filename = :f"),
            {"f": image_filename},
        ).one()

        assert row.image_filename == image_filename
        assert row.processing_time_s == 0.0

        # Tidy up so the test leaves the db in the state it found it.
        session.execute(
            text("DELETE FROM predictions WHERE image_filename = :f"),
            {"f": image_filename},
        )
        session.commit()
