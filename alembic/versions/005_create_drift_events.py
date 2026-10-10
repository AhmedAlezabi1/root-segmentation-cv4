"""Create drift_events table

Revision ID: 005
Revises: 004
Create Date: 2026-06-04

Adds the audit table written by the scheduled drift-detection job:
    - drift_events: one row per landmark per check, recording the KS-test
      p-value, whether drift was detected, which landmark it describes, and
      optional notes. Append-only history of every drift check.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    """True if the table already exists in the connected database.

    Lets this migration run safely against a database whose schema predates
    Alembic tracking (table present but ``alembic_version`` not at head). In
    that case the create is skipped, but Alembic still stamps this revision
    on success, retroactively tracking the existing database with no data loss.
    """
    inspector = sa.inspect(op.get_bind())
    return inspector.has_table(name)


def upgrade() -> None:
    """Create the drift_events table (skipping it if it already exists)."""

    # drift_events table
    # Append-only audit log of the daily drift check. Each run inserts one row
    # per landmark (landmark_index 0-4; -1 reserved for an aggregate row).
    # p_value is nullable because a landmark may be untestable (too few rolling
    # samples or a degenerate baseline) yet still gets an audit row.
    if not _has_table("drift_events"):
        op.create_table(
            "drift_events",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "detected_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("p_value", sa.Float, nullable=True),
            sa.Column("drift_detected", sa.Boolean, nullable=False),
            sa.Column("landmark_index", sa.Integer, nullable=False),
            sa.Column("notes", sa.String(255), nullable=True),
        )


def downgrade() -> None:
    """Drop the drift_events table."""
    if _has_table("drift_events"):
        op.drop_table("drift_events")
