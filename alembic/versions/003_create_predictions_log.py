"""Create predictions_log table

Revision ID: 003
Revises: 002
Create Date: 2026-06-04

Adds an append-only log of per-prediction confidence vectors for downstream
drift detection:
    - predictions_log: one row per POST /predict call, recording the 5
      per-landmark confidence scores plus minimal metadata, so drift
      monitoring can query confidence cheaply without parsing the full
      predictions.results payload.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
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
    """Create the predictions_log table (skipping it if it already exists)."""

    # predictions_log table
    # Stores one row per POST /predict call. confidence_scores holds the 5
    # per-landmark confidence floats as a JSON array (portable across SQLite
    # and PostgreSQL). prediction_id links back to predictions.id (stored as
    # a string). This table is append-only and intended for drift detection.
    if not _has_table("predictions_log"):
        op.create_table(
            "predictions_log",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("prediction_id", sa.String(64), nullable=False, index=True),
            sa.Column("confidence_scores", sa.JSON, nullable=False),
            sa.Column(
                "timestamp",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("image_reference", sa.String(255), nullable=True),
        )


def downgrade() -> None:
    """Drop the predictions_log table."""
    if _has_table("predictions_log"):
        op.drop_table("predictions_log")
