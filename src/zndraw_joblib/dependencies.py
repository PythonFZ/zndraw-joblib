# queue_package/dependencies.py
from typing import AsyncGenerator
from redis.asyncio import Redis
from sqlmodel.ext.asyncio.session import AsyncSession

# Stub: Will be overridden
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    raise NotImplementedError("Dependency not configured")

# Stub: Will be overridden
async def get_redis_client() -> Redis:
    raise NotImplementedError("Dependency not configured")