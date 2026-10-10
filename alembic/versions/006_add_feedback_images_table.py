"""Add feedback_images table

Revision ID: 006
Revises: 005
Create Date: 2026-06-09

Adds the ``feedback_images`` table, which stores the on-disk path to the
original petri-dish image for any prediction that has received feedback:

    - feedback_images: one row per prediction (unique on prediction_id),
      written by POST /feedback/{prediction_id}/image and read by
      GET /feedback/export. Cascades delete from predictions so removing
      a prediction also removes its image record (the file on disk must
      be cleaned up separately if needed).
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
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
    """Create the feedback_images table (skipping it if it already exists)."""

    # feedback_images table
    # Stores one row per prediction that has had an image uploaded after
    # feedback submission. original_filename preserves the client-reported
    # name for display; stored_path is the absolute path on disk where the
    # file was written. content_type records the MIME type so the export
    # endpoint can set correct headers per file if needed in the future.
    # unique=True on prediction_id enforces at most one image per prediction.
    if not _has_table("feedback_images"):
        op.create_table(
            "feedback_images",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "prediction_id",
                sa.Integer,
                sa.ForeignKey("predictions.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
            ),
            sa.Column("original_filename", sa.String(255), nullable=False),
            sa.Column("stored_path", sa.String(512), nullable=False),
            sa.Column("content_type", sa.String(64), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    """Drop the feedback_images table."""
    if _has_table("feedback_images"):
        op.drop_table("feedback_images")
