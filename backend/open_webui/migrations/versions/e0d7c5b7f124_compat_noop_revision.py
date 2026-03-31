"""compat no-op revision for legacy databases

Revision ID: e0d7c5b7f124
Revises: c440947495f3
Create Date: 2026-03-31
"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "e0d7c5b7f124"
down_revision: Union[str, None] = "c440947495f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Intentionally empty: compatibility bridge for environments stamped
    # with legacy revision e0d7c5b7f124.
    pass


def downgrade() -> None:
    # Intentionally empty.
    pass
