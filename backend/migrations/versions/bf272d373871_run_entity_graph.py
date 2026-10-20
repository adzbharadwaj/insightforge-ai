"""run entity graph

Revision ID: bf272d373871
Revises: 8c26b688ff94
Create Date: 2026-10-18 20:00:57.066045

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'bf272d373871'
down_revision: str | Sequence[str] | None = '8c26b688ff94'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('runs', sa.Column('entity_graph', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('runs', 'entity_graph')
