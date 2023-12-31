from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Base

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=40,
    pool_recycle=3600,
    pool_timeout=120,
    future=True,
    echo=False,
)

async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autoflush=False,
)


async def initialize_database():
    """Initialize the database."""
    async with engine.begin() as conn:
        # uuid-ossp is needed for the UUID type
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a new session."""
    async with async_session() as session:
        yield session
