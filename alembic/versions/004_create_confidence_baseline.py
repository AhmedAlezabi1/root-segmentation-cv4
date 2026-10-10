"""Create confidence_baseline table

Revision ID: 004
Revises: 003
Create Date: 2026-06-04

Adds a table holding the baseline confidence distribution used by drift
detection:
    - confidence_baseline: one row per computed baseline, storing the
      per-landmark mean, std, and key quantiles of the confidence scores
      from the first N predictions in predictions_log, tagged with the
      model version the baseline characterises.
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
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
    """Create the confidence_baseline table (skipping it if it already exists)."""

    # confidence_baseline table
    # Stores one row per computed baseline. mean/std hold the per-landmark
    # confidence statistics as JSON arrays (length 5); quantiles holds the
    # key percentiles per landmark as a JSON object. JSON is portable across
    # SQLite and PostgreSQL. Each row is tagged with the model_version it
    # characterises so drift detection can pick the matching baseline.
    if not _has_table("confidence_baseline"):
        op.create_table(
            "confidence_baseline",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_version", sa.String(64), nullable=False),
            sa.Column("n_samples", sa.Integer, nullable=False),
            sa.Column("mean", sa.JSON, nullable=False),
            sa.Column("std", sa.JSON, nullable=False),
            sa.Column("quantiles", sa.JSON, nullable=False),
            sa.Column(
                "computed_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    """Drop the confidence_baseline table."""
    if _has_table("confidence_baseline"):
        op.drop_table("confidence_baseline")
