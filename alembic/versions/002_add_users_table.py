"""Add users table

Revision ID: 002
Revises: 00ee6ac1e65d
Create Date: 2026-06-01
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "00ee6ac1e65d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    """True if the table already exists in the connected database.

    Lets this migration run safely against a database whose schema predates
    Alembic tracking (table present but ``alembic_version`` not at head). In
    that case the create is skipped, but Alembic still stamps this revision on
    success, retroactively tracking the existing database with no data loss.
    """
    inspector = sa.inspect(op.get_bind())
    return inspector.has_table(name)


def upgrade() -> None:
    """Create the users table for JWT multi-user authentication."""
    if not _has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("email", sa.String(254), nullable=False),
            sa.Column("hashed_password", sa.String(255), nullable=False),
            sa.Column(
                "is_active", sa.Boolean(), nullable=False, server_default=sa.true()
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)


def downgrade() -> None:
    """Drop the users table."""
    if _has_table("users"):
        op.drop_index("ix_users_email", table_name="users")
        op.drop_table("users")
