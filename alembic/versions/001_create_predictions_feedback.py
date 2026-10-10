"""Create predictions and feedback tables

Revision ID: 001
Revises:
Create Date: 2026-05-13

Creates the initial schema for the cv4 feedback system:
    - predictions: stores POST /predict results for feedback reference
    - feedback: stores per-lane user corrections linked to predictions
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "00ee6ac1e65d"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    """True if the table already exists in the connected database.

    Lets this baseline migration run safely against a database whose schema
    predates Alembic tracking (tables present but ``alembic_version`` not at
    head). In that case the create is skipped, but Alembic still stamps this
    revision on success, retroactively tracking the existing database with no
    data loss.
    """
    inspector = sa.inspect(op.get_bind())
    return inspector.has_table(name)


def upgrade() -> None:
    """Create predictions and feedback tables (skipping any that exist)."""

    # predictions table
    # Stores one row per POST /predict call. Results are stored as JSON
    # so the table does not need to be normalised across lanes.
    if not _has_table("predictions"):
        op.create_table(
            "predictions",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("image_filename", sa.String(255), nullable=False),
            sa.Column("results", sa.JSON, nullable=False),
            sa.Column("processing_time_s", sa.Float, nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )

    # feedback table
    # Stores one row per lane per feedback submission. Each row references
    # a prediction via a foreign key with CASCADE delete — deleting a
    # prediction removes all associated feedback rows.
    if not _has_table("feedback"):
        op.create_table(
            "feedback",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "prediction_id",
                sa.Integer,
                sa.ForeignKey("predictions.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("lane", sa.Integer, nullable=False),
            sa.Column("is_correct", sa.Boolean, nullable=False),
            sa.Column("predicted_tip_x", sa.Integer, nullable=True),
            sa.Column("predicted_tip_y", sa.Integer, nullable=True),
            sa.Column("correct_tip_x", sa.Integer, nullable=True),
            sa.Column("correct_tip_y", sa.Integer, nullable=True),
            sa.Column("comment", sa.Text, nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    """Drop feedback and predictions tables in reverse dependency order."""
    # feedback must be dropped before predictions due to the foreign key
    if _has_table("feedback"):
        op.drop_table("feedback")
    if _has_table("predictions"):
        op.drop_table("predictions")
