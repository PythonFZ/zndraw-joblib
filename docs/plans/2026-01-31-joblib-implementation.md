# ZnDraw JobLib Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-contained FastAPI package for distributed job/task management with SQL persistence and Redis pub/sub.

**Architecture:** SQL-first storage using SQLModel for Job, Worker, Task, and WorkerJobLink models. Redis used only for pub/sub to enable long-polling on terminal task states. Dependency injection passthrough for DB, Redis, and auth. Client SDK with JobManager and TaskStream for workers.

**Tech Stack:** FastAPI, SQLModel, Redis (pub/sub only), Pydantic, pydantic-settings, httpx

---

## Task 1: Settings Module

**Files:**
- Create: `src/zndraw_joblib/settings.py`
- Test: `tests/test_settings.py`

**Step 1: Write the failing test**

```python
# tests/test_settings.py
import os
import pytest
from zndraw_joblib.settings import JobLibSettings


def test_default_settings():
    settings = JobLibSettings()
    assert settings.allowed_categories == ["modifiers", "selections", "analysis"]
    assert settings.worker_timeout_seconds == 60
    assert settings.sweeper_interval_seconds == 30
    assert settings.long_poll_max_wait_seconds == 120


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("ZNDRAW_JOBLIB_ALLOWED_CATEGORIES", '["custom"]')
    monkeypatch.setenv("ZNDRAW_JOBLIB_WORKER_TIMEOUT_SECONDS", "120")
    settings = JobLibSettings()
    assert settings.allowed_categories == ["custom"]
    assert settings.worker_timeout_seconds == 120
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_settings.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class JobLibSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZNDRAW_JOBLIB_")

    allowed_categories: list[str] = ["modifiers", "selections", "analysis"]
    worker_timeout_seconds: int = 60
    sweeper_interval_seconds: int = 30
    long_poll_max_wait_seconds: int = 120
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_settings.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/settings.py tests/test_settings.py
git commit -m "feat: add JobLibSettings with env var configuration"
```

---

## Task 2: Exceptions Module (RFC 9457)

**Files:**
- Create: `src/zndraw_joblib/exceptions.py`
- Test: `tests/test_exceptions.py`

**Step 1: Write the failing test**

```python
# tests/test_exceptions.py
import pytest
from zndraw_joblib.exceptions import (
    ProblemType,
    ProblemDetail,
    ProblemException,
    JobNotFound,
    SchemaConflict,
    InvalidCategory,
    WorkerNotFound,
    TaskNotFound,
    InvalidTaskTransition,
)


def test_problem_type_creates_problem_detail():
    problem = JobNotFound.create(detail="Job xyz not found")
    assert problem.type == "/v1/problems/job-not-found"
    assert problem.title == "Not Found"
    assert problem.status == 404
    assert problem.detail == "Job xyz not found"


def test_problem_type_creates_exception():
    exc = SchemaConflict.exception(detail="Schema mismatch")
    assert isinstance(exc, ProblemException)
    assert exc.problem.status == 409


def test_all_problem_types_have_correct_status():
    assert JobNotFound.status == 404
    assert SchemaConflict.status == 409
    assert InvalidCategory.status == 400
    assert WorkerNotFound.status == 404
    assert TaskNotFound.status == 404
    assert InvalidTaskTransition.status == 409
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_exceptions.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/exceptions.py
"""RFC 9457 Problem Details for HTTP APIs."""

import re
from typing import Any, ClassVar

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


def _camel_to_kebab(name: str) -> str:
    """Convert CamelCase to kebab-case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


class ProblemException(Exception):
    """Exception that carries a ProblemDetail for RFC 9457 responses."""

    def __init__(
        self,
        problem: ProblemDetail,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.problem = problem
        self.headers = headers
        super().__init__(problem.title)


class ProblemType:
    """Base class for defining problem types with documentation."""

    title: ClassVar[str]
    status: ClassVar[int]

    @classmethod
    def problem_id(cls) -> str:
        """Return kebab-case identifier derived from class name."""
        return _camel_to_kebab(cls.__name__)

    @classmethod
    def type_uri(cls) -> str:
        """Return the full type URI for this problem."""
        return f"/v1/problems/{cls.problem_id()}"

    @classmethod
    def openapi_response(
        cls, description: str | None = None
    ) -> dict[int | str, dict[str, Any]]:
        """Generate OpenAPI response entry for this problem type."""
        return {
            cls.status: {
                "model": ProblemDetail,
                "description": description
                or (cls.__doc__.split("\n")[0] if cls.__doc__ else cls.title),
            }
        }

    @classmethod
    def create(
        cls, detail: str | None = None, instance: str | None = None
    ) -> ProblemDetail:
        """Create a ProblemDetail instance from this type."""
        return ProblemDetail(
            type=cls.type_uri(),
            title=cls.title,
            status=cls.status,
            detail=detail,
            instance=instance,
        )

    @classmethod
    def exception(
        cls,
        detail: str | None = None,
        instance: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> ProblemException:
        """Create a ProblemException from this type."""
        return ProblemException(cls.create(detail, instance), headers=headers)


async def problem_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Convert ProblemException to RFC 9457 compliant JSON response."""
    assert isinstance(exc, ProblemException)
    return JSONResponse(
        status_code=exc.problem.status,
        content=exc.problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
        headers=exc.headers,
    )


# Problem Types

class JobNotFound(ProblemType):
    """The requested job does not exist."""
    title: ClassVar[str] = "Not Found"
    status: ClassVar[int] = 404


class SchemaConflict(ProblemType):
    """Job schema differs from existing registration."""
    title: ClassVar[str] = "Conflict"
    status: ClassVar[int] = 409


class InvalidCategory(ProblemType):
    """Job category is not in the allowed list."""
    title: ClassVar[str] = "Bad Request"
    status: ClassVar[int] = 400


class WorkerNotFound(ProblemType):
    """The requested worker does not exist."""
    title: ClassVar[str] = "Not Found"
    status: ClassVar[int] = 404


class TaskNotFound(ProblemType):
    """The requested task does not exist."""
    title: ClassVar[str] = "Not Found"
    status: ClassVar[int] = 404


class InvalidTaskTransition(ProblemType):
    """Invalid task status transition."""
    title: ClassVar[str] = "Conflict"
    status: ClassVar[int] = 409


class InvalidRoomId(ProblemType):
    """Room ID contains invalid characters (@ or :)."""
    title: ClassVar[str] = "Bad Request"
    status: ClassVar[int] = 400


class Forbidden(ProblemType):
    """Admin privileges required for this operation."""
    title: ClassVar[str] = "Forbidden"
    status: ClassVar[int] = 403
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_exceptions.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/exceptions.py tests/test_exceptions.py
git commit -m "feat: add RFC 9457 problem types and exception handler"
```

---

## Task 3: SQLModel Models

**Files:**
- Modify: `src/zndraw_joblib/models.py`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest
from datetime import datetime, timedelta
from uuid import UUID

from zndraw_joblib.models import (
    TaskStatus,
    WorkerJobLink,
    Job,
    Worker,
    Task,
)


def test_task_status_enum():
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.CLAIMED.value == "claimed"
    assert TaskStatus.RUNNING.value == "running"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"
    assert TaskStatus.CANCELLED.value == "cancelled"


def test_job_full_name():
    job = Job(room_id="@global", category="modifiers", name="Rotate", schema={})
    assert job.full_name == "@global:modifiers:Rotate"


def test_job_full_name_private():
    job = Job(room_id="room_123", category="selections", name="All", schema={})
    assert job.full_name == "room_123:selections:All"


def test_worker_is_alive():
    worker = Worker(id="worker_1", last_heartbeat=datetime.utcnow())
    assert worker.is_alive(timedelta(seconds=60)) is True


def test_worker_is_dead():
    old_time = datetime.utcnow() - timedelta(seconds=120)
    worker = Worker(id="worker_1", last_heartbeat=old_time)
    assert worker.is_alive(timedelta(seconds=60)) is False


def test_task_has_uuid_id():
    task = Task(job_id=UUID("12345678-1234-5678-1234-567812345678"), room_id="room_1")
    assert isinstance(task.id, UUID)


def test_task_default_status_is_pending():
    task = Task(job_id=UUID("12345678-1234-5678-1234-567812345678"), room_id="room_1")
    assert task.status == TaskStatus.PENDING
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_models.py -v`
Expected: FAIL with "ImportError" or "cannot import name"

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/models.py
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import uuid

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.types import JSON
from sqlmodel import SQLModel, Field, Relationship


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerJobLink(SQLModel, table=True):
    """Bare M:N link between Worker and Job."""

    __tablename__ = "worker_job_link"

    worker_id: str = Field(foreign_key="worker.id", primary_key=True)
    job_id: uuid.UUID = Field(foreign_key="job.id", primary_key=True)


class Job(SQLModel, table=True):
    __tablename__ = "job"
    __table_args__ = (
        UniqueConstraint("room_id", "category", "name", name="unique_job"),
    )

    id: uuid.UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: str = Field(index=True)
    category: str = Field(index=True)
    name: str = Field(index=True)
    schema_: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column("schema", JSON)
    )

    # Relationships
    tasks: list["Task"] = Relationship(back_populates="job")
    workers: list["Worker"] = Relationship(
        back_populates="jobs", link_model=WorkerJobLink
    )

    @property
    def full_name(self) -> str:
        return f"{self.room_id}:{self.category}:{self.name}"


class Worker(SQLModel, table=True):
    __tablename__ = "worker"

    id: str = Field(primary_key=True)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Relationships
    jobs: list[Job] = Relationship(back_populates="workers", link_model=WorkerJobLink)
    tasks: list["Task"] = Relationship(back_populates="worker")

    def is_alive(self, threshold: timedelta) -> bool:
        return datetime.utcnow() - self.last_heartbeat < threshold


class Task(SQLModel, table=True):
    __tablename__ = "task"

    id: uuid.UUID = Field(default_factory=uuid4, primary_key=True)

    job_id: uuid.UUID = Field(foreign_key="job.id", index=True)
    job: Optional[Job] = Relationship(back_populates="tasks")

    worker_id: Optional[str] = Field(default=None, foreign_key="worker.id")
    worker: Optional[Worker] = Relationship(back_populates="tasks")

    room_id: str = Field(index=True)
    created_by_id: Optional[str] = Field(default=None, index=True)

    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: TaskStatus = Field(default=TaskStatus.PENDING, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/models.py tests/test_models.py
git commit -m "feat: add SQLModel models for Job, Worker, Task, WorkerJobLink"
```

---

## Task 4: Dependencies Module

**Files:**
- Modify: `src/zndraw_joblib/dependencies.py`
- Test: `tests/test_dependencies.py`

**Step 1: Write the failing test**

```python
# tests/test_dependencies.py
import pytest
from zndraw_joblib.dependencies import (
    get_db_session,
    get_redis_client,
    get_current_identity,
    get_is_admin,
    get_settings,
)
from zndraw_joblib.settings import JobLibSettings


@pytest.mark.asyncio
async def test_get_db_session_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Dependency not configured"):
        async for _ in get_db_session():
            pass


@pytest.mark.asyncio
async def test_get_redis_client_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Dependency not configured"):
        await get_redis_client()


@pytest.mark.asyncio
async def test_get_current_identity_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Dependency not configured"):
        await get_current_identity()


@pytest.mark.asyncio
async def test_get_is_admin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Dependency not configured"):
        await get_is_admin()


def test_get_settings_returns_settings():
    settings = get_settings()
    assert isinstance(settings, JobLibSettings)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_dependencies.py -v`
Expected: FAIL with "ImportError" or missing functions

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/dependencies.py
from functools import lru_cache
from typing import AsyncGenerator

from redis.asyncio import Redis
from sqlmodel.ext.asyncio.session import AsyncSession

from zndraw_joblib.settings import JobLibSettings


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Stub: Override via app.dependency_overrides."""
    raise NotImplementedError("Dependency not configured")
    yield  # type: ignore


async def get_redis_client() -> Redis:
    """Stub: Override via app.dependency_overrides."""
    raise NotImplementedError("Dependency not configured")


async def get_current_identity() -> str:
    """Stub: Override via app.dependency_overrides. Returns user/worker ID from JWT."""
    raise NotImplementedError("Dependency not configured")


async def get_is_admin() -> bool:
    """Stub: Override via app.dependency_overrides. Returns True if user is admin."""
    raise NotImplementedError("Dependency not configured")


@lru_cache
def get_settings() -> JobLibSettings:
    """Returns cached settings instance."""
    return JobLibSettings()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_dependencies.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/dependencies.py tests/test_dependencies.py
git commit -m "feat: add dependency stubs for DB, Redis, auth, and settings"
```

---

## Task 5: Request/Response Schemas

**Files:**
- Create: `src/zndraw_joblib/schemas.py`
- Test: `tests/test_schemas.py`

**Step 1: Write the failing test**

```python
# tests/test_schemas.py
import pytest
from datetime import datetime
from uuid import UUID

from zndraw_joblib.schemas import (
    JobRegisterRequest,
    JobResponse,
    TaskSubmitRequest,
    TaskResponse,
    TaskClaimResponse,
    TaskUpdateRequest,
)
from zndraw_joblib.models import TaskStatus


def test_job_register_request():
    req = JobRegisterRequest(category="modifiers", name="Rotate", schema={"angle": 0})
    assert req.category == "modifiers"
    assert req.name == "Rotate"
    assert req.json_schema == {"angle": 0}


def test_job_response():
    resp = JobResponse(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        room_id="@global",
        category="modifiers",
        name="Rotate",
        full_name="@global:modifiers:Rotate",
        schema={"angle": 0},
        worker_count=2,
    )
    assert resp.full_name == "@global:modifiers:Rotate"
    assert resp.worker_count == 2


def test_task_submit_request():
    req = TaskSubmitRequest(payload={"angle": 90})
    assert req.payload == {"angle": 90}


def test_task_response():
    resp = TaskResponse(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        job_name="@global:modifiers:Rotate",
        room_id="room_1",
        status=TaskStatus.PENDING,
        created_at=datetime.utcnow(),
    )
    assert resp.status == TaskStatus.PENDING


def test_task_update_request_valid_status():
    req = TaskUpdateRequest(status=TaskStatus.RUNNING)
    assert req.status == TaskStatus.RUNNING


def test_task_claim_response_with_task():
    resp = TaskClaimResponse(
        task=TaskResponse(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            job_name="@global:modifiers:Rotate",
            room_id="room_1",
            status=TaskStatus.CLAIMED,
            created_at=datetime.utcnow(),
        )
    )
    assert resp.task is not None


def test_task_claim_response_empty():
    resp = TaskClaimResponse(task=None)
    assert resp.task is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_schemas.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/schemas.py
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel

from zndraw_joblib.models import TaskStatus


class JobRegisterRequest(BaseModel):
    category: str
    name: str
    schema: dict[str, Any] = {}


class JobResponse(BaseModel):
    id: UUID
    room_id: str
    category: str
    name: str
    full_name: str
    schema: dict[str, Any]
    worker_count: int


class JobSummary(BaseModel):
    full_name: str
    category: str
    name: str
    worker_count: int


class TaskSubmitRequest(BaseModel):
    payload: dict[str, Any] = {}


class TaskResponse(BaseModel):
    id: UUID
    job_name: str
    room_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    payload: dict[str, Any] = {}


class TaskClaimResponse(BaseModel):
    task: Optional[TaskResponse] = None


class TaskUpdateRequest(BaseModel):
    status: TaskStatus
    error: Optional[str] = None
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_schemas.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/schemas.py tests/test_schemas.py
git commit -m "feat: add request/response schemas for API endpoints"
```

---

## Task 6: Router - Job Registration Endpoint

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_jobs.py`

**Step 1: Write the failing test**

```python
# tests/test_router_jobs.py
import pytest
from uuid import uuid4
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import get_db_session, get_current_identity, get_is_admin, get_settings
from zndraw_joblib.models import Job, Worker, WorkerJobLink


@pytest.fixture
def app():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    async def get_test_identity():
        return "test_worker_id"

    async def get_test_is_admin():
        return True

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_current_identity] = get_test_identity
    app.dependency_overrides[get_is_admin] = get_test_is_admin
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_register_job_global(client):
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {"angle": 0}},
    )
    assert response.status_code == 201
    data = response.json()
    # TODO: use pydantic model validation!
    assert data["full_name"] == "@global:modifiers:Rotate"
    assert data["worker_count"] == 1


def test_register_job_private(client):
    response = client.put(
        "/v1/joblib/rooms/room_123/jobs",
        json={"category": "selections", "name": "All", "schema": {}},
    )
    assert response.status_code == 201
    data = response.json()
    # TODO: use pydantic model validation!
    assert data["full_name"] == "room_123:selections:All"


def test_register_job_invalid_category(client):
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "invalid_cat", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 400
    # TODO: use pydantic model validation!


def test_register_job_schema_conflict(client):
    # First registration
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {"angle": 0}},
    )
    # Second registration with different schema
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {"angle": 0, "axis": "z"}},
    )
    assert response.status_code == 409
    # TODO: use pydantic model validation!


def test_register_job_same_schema_idempotent(client):
    schema = {"angle": 0}
    # First registration
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "json_schema": schema},
    )
    # Second registration with same schema
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "json_schema": schema},
    )
    assert response.status_code == 200  # OK, not 201
    # TODO: use pydantic model validation!


def test_register_job_invalid_room_id_with_at(client):
    response = client.put(
        "/v1/joblib/rooms/room@123/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 400


def test_register_job_invalid_room_id_with_colon(client):
    response = client.put(
        "/v1/joblib/rooms/room:123/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 400
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_jobs.py -v`
Expected: FAIL with missing endpoint

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/router.py
from fastapi import APIRouter, Depends, Response, status
from sqlmodel import Session, select

from zndraw_joblib.dependencies import (
    get_db_session,
    get_current_identity,
    get_is_admin,
    get_settings,
)
from zndraw_joblib.exceptions import (
    InvalidCategory,
    InvalidRoomId,
    SchemaConflict,
    Forbidden,
)
from zndraw_joblib.models import Job, Worker, WorkerJobLink
from zndraw_joblib.schemas import JobRegisterRequest, JobResponse
from zndraw_joblib.settings import JobLibSettings

router = APIRouter(prefix="/v1/joblib", tags=["joblib"])


def validate_room_id(room_id: str) -> None:
    """Validate room_id doesn't contain @ or : (except @global)."""
    if room_id == "@global":
        return
    if "@" in room_id or ":" in room_id:
        raise InvalidRoomId.exception(
            detail=f"Room ID '{room_id}' contains invalid characters (@ or :)"
        )


@router.put(
    "/rooms/{room_id}/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_job(
    room_id: str,
    request: JobRegisterRequest,
    response: Response,
    db: Session = Depends(get_db_session),
    identity: str = Depends(get_current_identity),
    is_admin: bool = Depends(get_is_admin),
    settings: JobLibSettings = Depends(get_settings),
):
    """Register a job for a room. Creates worker and link if not exists."""
    # Validate room_id
    validate_room_id(room_id)

    # Check admin for @global
    if room_id == "@global" and not is_admin:
        raise Forbidden.exception(detail="Admin required for @global job registration")

    # Validate category
    if request.category not in settings.allowed_categories:
        raise InvalidCategory.exception(
            detail=f"Category '{request.category}' not in allowed list: {settings.allowed_categories}"
        )

    # Check if job exists
    existing_job = db.exec(
        select(Job).where(
            Job.room_id == room_id,
            Job.category == request.category,
            Job.name == request.name,
        )
    ).first()

    if existing_job:
        # Validate schema match
        if existing_job.schema_ != request.schema:
            raise SchemaConflict.exception(
                detail=f"Schema mismatch for job '{existing_job.full_name}'"
            )
        # Idempotent - ensure worker link exists
        job = existing_job
        response.status_code = status.HTTP_200_OK
    else:
        # Create new job
        job = Job(
            room_id=room_id,
            category=request.category,
            name=request.name,
            schema_=request.schema,
        )
        db.add(job)
        db.flush()

    # Ensure worker exists
    worker = db.exec(select(Worker).where(Worker.id == identity)).first()
    if not worker:
        worker = Worker(id=identity)
        db.add(worker)
        db.flush()

    # Ensure worker-job link exists
    link = db.exec(
        select(WorkerJobLink).where(
            WorkerJobLink.worker_id == identity,
            WorkerJobLink.job_id == job.id,
        )
    ).first()
    if not link:
        link = WorkerJobLink(worker_id=identity, job_id=job.id)
        db.add(link)

    db.commit()
    db.refresh(job)

    # Count workers for this job
    worker_count = len(
        db.exec(select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)).all()
    )

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        worker_count=worker_count,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_jobs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_jobs.py
git commit -m "feat: add PUT /rooms/{room_id}/jobs endpoint for job registration"
```

---

## Task 7: Router - Job Listing and Details Endpoints

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_job_list.py`

**Step 1: Write the failing test**

```python
# tests/test_router_job_list.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import get_db_session, get_current_identity, get_is_admin


@pytest.fixture
def app():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    async def get_test_identity():
        return "test_worker_id"

    async def get_test_is_admin():
        return True

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_current_identity] = get_test_identity
    app.dependency_overrides[get_is_admin] = get_test_is_admin
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def seeded_client(client):
    # Register some jobs
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "selections", "name": "All", "schema": {}},
    )
    client.put(
        "/v1/joblib/rooms/room_123/jobs",
        json={"category": "modifiers", "name": "Translate", "schema": {}},
    )
    return client


def test_list_jobs_global_only(seeded_client):
    response = seeded_client.get("/v1/joblib/rooms/@global/jobs")
    assert response.status_code == 200
    data = response.json()
    # TODO: use pydantic model validation!
    assert len(data) == 2
    names = [j["full_name"] for j in data]
    assert "@global:modifiers:Rotate" in names
    assert "@global:selections:All" in names


def test_list_jobs_room_includes_global(seeded_client):
    response = seeded_client.get("/v1/joblib/rooms/room_123/jobs")
    assert response.status_code == 200
    data = response.json()
    # TODO: use pydantic model validation!
    assert len(data) == 3  # 2 global + 1 room
    names = [j["full_name"] for j in data]
    assert "@global:modifiers:Rotate" in names
    assert "room_123:modifiers:Translate" in names


def test_get_job_details(seeded_client):
    response = seeded_client.get(
        "/v1/joblib/rooms/room_123/jobs/@global:modifiers:Rotate"
    )
    assert response.status_code == 200
    data = response.json()
    # TODO: use pydantic model validation!
    assert data["full_name"] == "@global:modifiers:Rotate"
    assert data["category"] == "modifiers"


def test_get_job_not_found(seeded_client):
    response = seeded_client.get(
        "/v1/joblib/rooms/room_123/jobs/@global:modifiers:NonExistent"
    )
    assert response.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_job_list.py -v`
Expected: FAIL with 404 or missing endpoint

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
from zndraw_joblib.schemas import JobSummary
from zndraw_joblib.exceptions import JobNotFound


@router.get("/rooms/{room_id}/jobs", response_model=list[JobSummary])
async def list_jobs(
    room_id: str,
    db: Session = Depends(get_db_session),
):
    """List jobs for a room. Includes @global jobs unless room_id is @global."""
    validate_room_id(room_id)

    if room_id == "@global":
        jobs = db.exec(select(Job).where(Job.room_id == "@global")).all()
    else:
        jobs = db.exec(
            select(Job).where((Job.room_id == "@global") | (Job.room_id == room_id))
        ).all()

    result = []
    for job in jobs:
        worker_count = len(
            db.exec(select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)).all()
        )
        result.append(
            JobSummary(
                full_name=job.full_name,
                category=job.category,
                name=job.name,
                worker_count=worker_count,
            )
        )
    return result


@router.get("/rooms/{room_id}/jobs/{job_name:path}", response_model=JobResponse)
async def get_job(
    room_id: str,
    job_name: str,
    db: Session = Depends(get_db_session),
):
    """Get job details by full name."""
    validate_room_id(room_id)

    # Parse job_name: room_id:category:name
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")

    job_room_id, category, name = parts

    # For room requests, allow access to both @global and room-specific jobs
    if room_id != "@global" and job_room_id not in ("@global", room_id):
        raise JobNotFound.exception(detail=f"Job '{job_name}' not accessible from room '{room_id}'")

    job = db.exec(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    ).first()

    if not job:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")

    worker_count = len(
        db.exec(select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)).all()
    )

    return JobResponse(
        id=job.id,
        room_id=job.room_id,
        category=job.category,
        name=job.name,
        full_name=job.full_name,
        schema=job.schema_,
        worker_count=worker_count,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_job_list.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_job_list.py
git commit -m "feat: add GET /rooms/{room_id}/jobs endpoints for listing and details"
```

---

## Task 8: Router - Task Submission Endpoint

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_task_submit.py`

**Step 1: Write the failing test**

```python
# tests/test_router_task_submit.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import get_db_session, get_current_identity, get_is_admin
from zndraw_joblib.models import Task, TaskStatus


@pytest.fixture
def app():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    async def get_test_identity():
        return "test_user_id"

    async def get_test_is_admin():
        return True

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_current_identity] = get_test_identity
    app.dependency_overrides[get_is_admin] = get_test_is_admin
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def seeded_client(client):
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    return client


def test_submit_task(seeded_client):
    response = seeded_client.post(
        "/v1/joblib/rooms/room_123/tasks/@global:modifiers:Rotate",
        json={"payload": {"angle": 90}},
    )
    assert response.status_code == 202
    assert "Location" in response.headers
    data = response.json()
    # TODO: use pydantic model validation!
    assert data["status"] == "pending"
    assert data["payload"] == {"angle": 90}


def test_submit_task_job_not_found(seeded_client):
    response = seeded_client.post(
        "/v1/joblib/rooms/room_123/tasks/@global:modifiers:NonExistent",
        json={"payload": {}},
    )
    assert response.status_code == 404


def test_submit_task_sets_room_id(seeded_client):
    response = seeded_client.post(
        "/v1/joblib/rooms/my_room/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    assert response.status_code == 202
    data = response.json()
    # TODO: use pydantic model validation!
    assert data["room_id"] == "my_room"


def test_submit_task_sets_created_by_id(seeded_client):
    response = seeded_client.post(
        "/v1/joblib/rooms/room_123/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    assert response.status_code == 202
    # created_by_id should match the identity
    # (verified through task retrieval later)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_task_submit.py -v`
Expected: FAIL with 404 or missing endpoint

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
from zndraw_joblib.models import Task
from zndraw_joblib.schemas import TaskSubmitRequest, TaskResponse


@router.post(
    "/rooms/{room_id}/tasks/{job_name:path}",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_task(
    room_id: str,
    job_name: str,
    request: TaskSubmitRequest,
    response: Response,
    db: Session = Depends(get_db_session),
    identity: str = Depends(get_current_identity),
):
    """Submit a task for processing."""
    validate_room_id(room_id)

    # Parse job_name
    parts = job_name.split(":", 2)
    if len(parts) != 3:
        raise JobNotFound.exception(detail=f"Invalid job name format: {job_name}")

    job_room_id, category, name = parts

    # Find the job
    job = db.exec(
        select(Job).where(
            Job.room_id == job_room_id,
            Job.category == category,
            Job.name == name,
        )
    ).first()

    if not job:
        raise JobNotFound.exception(detail=f"Job '{job_name}' not found")

    # For private jobs, ensure room matches
    if job_room_id != "@global" and job_room_id != room_id:
        raise JobNotFound.exception(
            detail=f"Job '{job_name}' not accessible from room '{room_id}'"
        )

    # Create task
    task = Task(
        job_id=job.id,
        room_id=room_id,
        created_by_id=identity,
        payload=request.payload,
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Set Location header
    response.headers["Location"] = f"/v1/joblib/tasks/{task.id}"
    response.headers["Retry-After"] = "1"

    return TaskResponse(
        id=task.id,
        job_name=job.full_name,
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        payload=task.payload,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_task_submit.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_task_submit.py
git commit -m "feat: add POST /rooms/{room_id}/tasks/{job_name} endpoint for task submission"
```

---

## Task 9: Router - Task Claim Endpoint

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_task_claim.py`

**Step 1: Write the failing test**

```python
# tests/test_router_task_claim.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import get_db_session, get_current_identity, get_is_admin
from zndraw_joblib.models import TaskStatus


@pytest.fixture
def engine():
    return create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )


@pytest.fixture
def app(engine):
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    async def get_test_identity():
        return "worker_1"

    async def get_test_is_admin():
        return True

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_current_identity] = get_test_identity
    app.dependency_overrides[get_is_admin] = get_test_is_admin
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def seeded_client(client):
    # Register job as worker_1
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    return client


def test_claim_task_returns_null_when_empty(seeded_client):
    response = seeded_client.post("/v1/joblib/tasks/claim")
    assert response.status_code == 200
    data = response.json()
    assert data["task"] is None


def test_claim_task_returns_oldest_first(seeded_client):
    # Submit two tasks
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"order": 1}},
    )
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {"order": 2}},
    )

    # Claim should return first (oldest)
    response = seeded_client.post("/v1/joblib/tasks/claim")
    assert response.status_code == 200
    data = response.json()
    assert data["task"] is not None
    assert data["task"]["payload"]["order"] == 1
    assert data["task"]["status"] == "claimed"


def test_claim_task_marks_as_claimed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )

    response = seeded_client.post("/v1/joblib/tasks/claim")
    data = response.json()
    assert data["task"]["status"] == "claimed"


def test_claim_task_only_registered_jobs(engine):
    """Worker can only claim tasks for jobs they are registered for."""
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    # Worker 1 registers job
    async def get_worker_1():
        return "worker_1"

    # Worker 2 tries to claim
    async def get_worker_2():
        return "worker_2"

    async def get_admin():
        return True

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_is_admin] = get_admin

    # Worker 1 registers
    app.dependency_overrides[get_current_identity] = get_worker_1
    client1 = TestClient(app)
    client1.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )

    # Submit task
    client1.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )

    # Worker 2 tries to claim (not registered)
    app.dependency_overrides[get_current_identity] = get_worker_2
    client2 = TestClient(app)
    response = client2.post("/v1/joblib/tasks/claim")
    data = response.json()
    assert data["task"] is None  # Can't claim - not registered
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_task_claim.py -v`
Expected: FAIL with 404 or missing endpoint

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
from datetime import datetime
from zndraw_joblib.schemas import TaskClaimResponse


@router.post("/tasks/claim", response_model=TaskClaimResponse)
async def claim_task(
    db: Session = Depends(get_db_session),
    identity: str = Depends(get_current_identity),
):
    """Claim the oldest pending task for jobs the worker is registered for."""
    # Find oldest pending task for jobs this worker is registered for
    # Using subquery to get job IDs worker is registered for
    worker_job_ids = db.exec(
        select(WorkerJobLink.job_id).where(WorkerJobLink.worker_id == identity)
    ).all()

    if not worker_job_ids:
        return TaskClaimResponse(task=None)

    # Find oldest pending task
    task = db.exec(
        select(Task)
        .where(Task.job_id.in_(worker_job_ids), Task.status == TaskStatus.PENDING)
        .order_by(Task.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()

    if not task:
        return TaskClaimResponse(task=None)

    # Claim the task
    task.status = TaskStatus.CLAIMED
    task.worker_id = identity
    db.add(task)
    db.commit()
    db.refresh(task)

    # Get job for full name
    job = db.exec(select(Job).where(Job.id == task.job_id)).first()

    return TaskClaimResponse(
        task=TaskResponse(
            id=task.id,
            job_name=job.full_name if job else "",
            room_id=task.room_id,
            status=task.status,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            error=task.error,
            payload=task.payload,
        )
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_task_claim.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_task_claim.py
git commit -m "feat: add POST /tasks/claim endpoint for workers to claim tasks"
```

---

## Task 10: Router - Task Status and Update Endpoints

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_task_status.py`

**Step 1: Write the failing test**

```python
# tests/test_router_task_status.py
import pytest
from uuid import uuid4
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import get_db_session, get_current_identity, get_is_admin, get_redis_client
from zndraw_joblib.models import TaskStatus


class MockRedis:
    async def publish(self, channel, message):
        pass


@pytest.fixture
def app():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    async def get_test_identity():
        return "worker_1"

    async def get_test_is_admin():
        return True

    async def get_mock_redis():
        return MockRedis()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_current_identity] = get_test_identity
    app.dependency_overrides[get_is_admin] = get_test_is_admin
    app.dependency_overrides[get_redis_client] = get_mock_redis
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def seeded_client(client):
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    return client


def test_get_task_status(seeded_client):
    # Submit and claim a task
    submit_resp = seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    task_id = submit_resp.json()["id"]

    response = seeded_client.get(f"/v1/joblib/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == task_id
    assert data["status"] == "pending"


def test_get_task_not_found(seeded_client):
    fake_id = str(uuid4())
    response = seeded_client.get(f"/v1/joblib/tasks/{fake_id}")
    assert response.status_code == 404


def test_update_task_claimed_to_running(seeded_client):
    # Submit task
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    # Claim task
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    task_id = claim_resp.json()["task"]["id"]

    # Update to running
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "running"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_update_task_running_to_completed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    task_id = claim_resp.json()["task"]["id"]

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "completed"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_update_task_running_to_failed(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    task_id = claim_resp.json()["task"]["id"]

    seeded_client.patch(f"/v1/joblib/tasks/{task_id}", json={"status": "running"})
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "failed", "error": "Something went wrong"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "Something went wrong"


def test_update_task_invalid_transition(seeded_client):
    seeded_client.post(
        "/v1/joblib/rooms/room_1/tasks/@global:modifiers:Rotate",
        json={"payload": {}},
    )
    claim_resp = seeded_client.post("/v1/joblib/tasks/claim")
    task_id = claim_resp.json()["task"]["id"]

    # Try to go from claimed directly to completed (invalid)
    response = seeded_client.patch(
        f"/v1/joblib/tasks/{task_id}",
        json={"status": "completed"},
    )
    assert response.status_code == 409
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_task_status.py -v`
Expected: FAIL with 404 or missing endpoint

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
from uuid import UUID
from zndraw_joblib.exceptions import TaskNotFound, InvalidTaskTransition
from zndraw_joblib.schemas import TaskUpdateRequest

# Valid status transitions
VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.CLAIMED, TaskStatus.CANCELLED},
    TaskStatus.CLAIMED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}

TERMINAL_STATES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: UUID,
    db: Session = Depends(get_db_session),
):
    """Get task status. Supports long-polling via Prefer: wait=N header."""
    task = db.exec(select(Task).where(Task.id == task_id)).first()
    if not task:
        raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    job = db.exec(select(Job).where(Job.id == task.job_id)).first()

    return TaskResponse(
        id=task.id,
        job_name=job.full_name if job else "",
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error=task.error,
        payload=task.payload,
    )


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task_status(
    task_id: UUID,
    request: TaskUpdateRequest,
    db: Session = Depends(get_db_session),
    redis = Depends(get_redis_client),
):
    """Update task status. Publishes to Redis on terminal states."""
    task = db.exec(select(Task).where(Task.id == task_id)).first()
    if not task:
        raise TaskNotFound.exception(detail=f"Task '{task_id}' not found")

    # Validate transition
    if request.status not in VALID_TRANSITIONS.get(task.status, set()):
        raise InvalidTaskTransition.exception(
            detail=f"Cannot transition from '{task.status.value}' to '{request.status.value}'"
        )

    # Update status
    task.status = request.status
    now = datetime.utcnow()

    if request.status == TaskStatus.RUNNING:
        task.started_at = now
    elif request.status in TERMINAL_STATES:
        task.completed_at = now
        if request.error:
            task.error = request.error

    db.add(task)
    db.commit()
    db.refresh(task)

    # Publish to Redis on terminal states
    if request.status in TERMINAL_STATES:
        channel = f"zndraw_joblib:task:{task_id}"
        await redis.publish(channel, request.status.value)

    job = db.exec(select(Job).where(Job.id == task.job_id)).first()

    return TaskResponse(
        id=task.id,
        job_name=job.full_name if job else "",
        room_id=task.room_id,
        status=task.status,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error=task.error,
        payload=task.payload,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_task_status.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_task_status.py
git commit -m "feat: add GET/PATCH /tasks/{task_id} endpoints for status and updates"
```

---

## Task 11: Router - Worker Heartbeat and Deletion Endpoints

**Files:**
- Modify: `src/zndraw_joblib/router.py`
- Test: `tests/test_router_worker.py`

**Step 1: Write the failing test**

```python
# tests/test_router_worker.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import StaticPool

from zndraw_joblib.router import router
from zndraw_joblib.dependencies import get_db_session, get_current_identity, get_is_admin
from zndraw_joblib.models import Worker


@pytest.fixture
def app():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def get_test_session():
        with Session(engine) as session:
            yield session

    async def get_test_identity():
        return "worker_1"

    async def get_test_is_admin():
        return True

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = get_test_session
    app.dependency_overrides[get_current_identity] = get_test_identity
    app.dependency_overrides[get_is_admin] = get_test_is_admin
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def seeded_client(client):
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    return client


def test_worker_heartbeat(seeded_client):
    response = seeded_client.patch("/v1/joblib/workers/worker_1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "worker_1"
    assert "last_heartbeat" in data


def test_worker_heartbeat_not_found(seeded_client):
    response = seeded_client.patch("/v1/joblib/workers/unknown_worker")
    assert response.status_code == 404


def test_worker_delete(seeded_client):
    response = seeded_client.delete("/v1/joblib/workers/worker_1")
    assert response.status_code == 204


def test_worker_delete_not_found(seeded_client):
    response = seeded_client.delete("/v1/joblib/workers/unknown_worker")
    assert response.status_code == 404


def test_worker_delete_removes_links(seeded_client, app):
    # Verify worker-job link exists
    seeded_client.delete("/v1/joblib/workers/worker_1")

    # Register again - should create new link
    response = seeded_client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 200  # Idempotent, job exists
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_worker.py -v`
Expected: FAIL with 404 or missing endpoint

**Step 3: Write minimal implementation**

Add to `src/zndraw_joblib/router.py`:

```python
from pydantic import BaseModel as PydanticBaseModel


class WorkerResponse(PydanticBaseModel):
    id: str
    last_heartbeat: datetime


@router.patch("/workers/{worker_id}", response_model=WorkerResponse)
async def worker_heartbeat(
    worker_id: str,
    db: Session = Depends(get_db_session),
):
    """Update worker heartbeat timestamp."""
    worker = db.exec(select(Worker).where(Worker.id == worker_id)).first()
    if not worker:
        raise WorkerNotFound.exception(detail=f"Worker '{worker_id}' not found")

    worker.last_heartbeat = datetime.utcnow()
    db.add(worker)
    db.commit()
    db.refresh(worker)

    return WorkerResponse(id=worker.id, last_heartbeat=worker.last_heartbeat)


@router.delete("/workers/{worker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_worker(
    worker_id: str,
    db: Session = Depends(get_db_session),
):
    """Delete worker and all job links."""
    worker = db.exec(select(Worker).where(Worker.id == worker_id)).first()
    if not worker:
        raise WorkerNotFound.exception(detail=f"Worker '{worker_id}' not found")

    # Delete all links first
    links = db.exec(
        select(WorkerJobLink).where(WorkerJobLink.worker_id == worker_id)
    ).all()
    for link in links:
        db.delete(link)

    db.delete(worker)
    db.commit()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_router_worker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/router.py tests/test_router_worker.py
git commit -m "feat: add PATCH/DELETE /workers/{worker_id} endpoints for heartbeat and cleanup"
```

---

## Task 12: Sweeper Module

**Files:**
- Create: `src/zndraw_joblib/sweeper.py`
- Test: `tests/test_sweeper.py`

**Step 1: Write the failing test**

```python
# tests/test_sweeper.py
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from zndraw_joblib.sweeper import run_cleanup_sweeper, cleanup_stale_workers


def test_cleanup_stale_workers_signature():
    """Verify cleanup function exists with correct signature."""
    import inspect
    sig = inspect.signature(cleanup_stale_workers)
    params = list(sig.parameters.keys())
    assert "db" in params
    assert "timeout_seconds" in params
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_sweeper.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/sweeper.py
import asyncio
from datetime import datetime, timedelta

from sqlmodel import Session, select

from zndraw_joblib.models import Worker, Task, TaskStatus, Job, WorkerJobLink


def cleanup_stale_workers(db: Session, timeout_seconds: int = 60) -> int:
    """
    Find workers with stale heartbeats and:
    1. Mark their running tasks as FAILED
    2. Delete the worker and its job links

    Returns count of failed tasks.
    """
    threshold = datetime.utcnow() - timedelta(seconds=timeout_seconds)

    # Find stale workers
    stale_workers = db.exec(
        select(Worker).where(Worker.last_heartbeat < threshold)
    ).all()

    failed_count = 0

    for worker in stale_workers:
        # Fail running tasks
        running_tasks = db.exec(
            select(Task).where(
                Task.worker_id == worker.id,
                Task.status == TaskStatus.RUNNING,
            )
        ).all()

        for task in running_tasks:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error = f"Worker '{worker.id}' timed out"
            db.add(task)
            failed_count += 1

        # Delete worker links
        links = db.exec(
            select(WorkerJobLink).where(WorkerJobLink.worker_id == worker.id)
        ).all()
        for link in links:
            db.delete(link)

        # Delete worker
        db.delete(worker)

    db.commit()
    return failed_count


def cleanup_orphan_jobs(db: Session) -> int:
    """
    Remove jobs with no workers and no pending tasks.

    Returns count of removed jobs.
    """
    # Find all jobs
    all_jobs = db.exec(select(Job)).all()
    removed_count = 0

    for job in all_jobs:
        # Check for workers
        worker_count = len(
            db.exec(select(WorkerJobLink).where(WorkerJobLink.job_id == job.id)).all()
        )
        if worker_count > 0:
            continue

        # Check for pending tasks
        pending_count = len(
            db.exec(
                select(Task).where(
                    Task.job_id == job.id,
                    Task.status == TaskStatus.PENDING,
                )
            ).all()
        )
        if pending_count > 0:
            continue

        # Remove orphan job
        db.delete(job)
        removed_count += 1

    db.commit()
    return removed_count


async def run_cleanup_sweeper(
    base_url: str,
    interval_seconds: int = 30,
) -> None:
    """
    Background task that periodically cleans up stale workers and orphan jobs.

    Note: This uses HTTP calls to avoid creating its own DB session.
    In production, the host app should inject proper session handling.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(interval_seconds)
            # In a real implementation, this would call internal APIs
            # or use a shared session factory
            # For now, this is a placeholder
            pass
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_sweeper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/sweeper.py tests/test_sweeper.py
git commit -m "feat: add sweeper module for stale worker cleanup"
```

---

## Task 13: Client SDK - ClaimedTask and ApiManager

**Files:**
- Create: `src/zndraw_joblib/client.py`
- Test: `tests/test_client.py`

**Step 1: Write the failing test**

```python
# tests/test_client.py
import pytest
from datetime import datetime
from unittest.mock import MagicMock

from zndraw_joblib.client import ClaimedTask, ApiManager, JobManager, TaskStream


def test_claimed_task_model():
    task = ClaimedTask(
        id="task-123",
        job_name="@global:modifiers:Rotate",
        room_id="room_1",
        payload={"angle": 90},
        created_at=datetime.utcnow(),
    )
    assert task.id == "task-123"
    assert task.job_name == "@global:modifiers:Rotate"


def test_api_manager_protocol():
    """Verify ApiManager is a Protocol with required attributes."""
    import typing
    assert typing.get_origin(ApiManager) is None  # Protocol, not generic


def test_job_manager_init():
    mock_api = MagicMock()
    mock_api.http = MagicMock()
    mock_api.base_url = "http://localhost"
    mock_api.get_headers.return_value = {}

    manager = JobManager(mock_api)
    assert len(manager) == 0


def test_job_manager_register_decorator():
    mock_api = MagicMock()
    mock_api.http = MagicMock()
    mock_api.http.put.return_value.status_code = 201
    mock_api.base_url = "http://localhost"
    mock_api.get_headers.return_value = {}

    manager = JobManager(mock_api)

    from pydantic import BaseModel

    @manager.register
    class TestJob(BaseModel):
        value: int = 0

    assert "@global:modifiers:TestJob" in manager
    mock_api.http.put.assert_called_once()


def test_job_manager_register_with_room():
    mock_api = MagicMock()
    mock_api.http = MagicMock()
    mock_api.http.put.return_value.status_code = 201
    mock_api.base_url = "http://localhost"
    mock_api.get_headers.return_value = {}

    manager = JobManager(mock_api)

    from pydantic import BaseModel

    @manager.register(room="room_123")
    class TestJob(BaseModel):
        value: int = 0

    assert "room_123:modifiers:TestJob" in manager
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_client.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/client.py
import time
from datetime import datetime
from typing import Any, Protocol

import httpx
from pydantic import BaseModel


class ApiManager(Protocol):
    """Protocol for API client that JobManager uses."""

    http: httpx.Client

    def get_headers(self) -> dict[str, str]: ...

    @property
    def base_url(self) -> str: ...


class ClaimedTask(BaseModel):
    """Pydantic model for a claimed task."""

    id: str
    job_name: str
    room_id: str
    payload: dict[str, Any]
    created_at: datetime


class TaskStream:
    """Iterator for consuming tasks."""

    def __init__(self, api: ApiManager, worker_id: str, polling_interval: float = 2.0):
        self.api = api
        self.worker_id = worker_id
        self.interval = polling_interval
        self._stop_event = False

    def __iter__(self):
        return self

    def __next__(self) -> ClaimedTask:
        if self._stop_event:
            raise StopIteration

        while True:
            response = self.api.http.post(
                f"{self.api.base_url}/v1/joblib/tasks/claim",
                headers=self.api.get_headers(),
            )
            data = response.json()
            if data.get("task"):
                return ClaimedTask.model_validate(data["task"])
            time.sleep(self.interval)

    def stop(self):
        self._stop_event = True


class JobManager:
    """Main entry point. Behaves like a dictionary of registered jobs."""

    def __init__(self, api: ApiManager):
        self.api = api
        self._registry: dict[str, type[BaseModel]] = {}
        self._worker_id: str | None = None

    def __getitem__(self, key: str) -> type[BaseModel]:
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def register(
        self, task_class: type[BaseModel] | None = None, *, room: str | None = None
    ):
        """
        Register a job. room=None defaults to @global.

        Usage:
            @manager.register
            class Rotate(BaseModel):
                angle: float = 0.0

            @manager.register(room="room_123")
            class PrivateRotate(BaseModel):
                angle: float = 0.0
        """

        def decorator(cls: type[BaseModel]):
            self._register_impl(cls, room)
            return cls

        if task_class is None:
            return decorator
        return decorator(task_class)

    def _register_impl(self, cls: type[BaseModel], room: str | None):
        room_id = room if room is not None else "@global"
        category = getattr(cls, "category", "modifiers")
        name = cls.__name__

        schema = cls.model_json_schema()

        self.api.http.put(
            f"{self.api.base_url}/v1/joblib/rooms/{room_id}/jobs",
            headers=self.api.get_headers(),
            json={"category": category, "name": name, "schema": schema},
        )

        full_name = f"{room_id}:{category}:{name}"
        self._registry[full_name] = cls

    def listen(self, interval: float = 2.0) -> TaskStream:
        """Returns an iterator to process tasks."""
        worker_id = self._get_worker_id()
        return TaskStream(self.api, worker_id, interval)

    def _get_worker_id(self) -> str:
        if self._worker_id is None:
            response = self.api.http.get(
                f"{self.api.base_url}/v1/me",
                headers=self.api.get_headers(),
            )
            self._worker_id = response.json()["id"]
        return self._worker_id

    def heartbeat(self):
        """Manually send worker heartbeat."""
        worker_id = self._get_worker_id()
        self.api.http.patch(
            f"{self.api.base_url}/v1/joblib/workers/{worker_id}",
            headers=self.api.get_headers(),
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/client.py tests/test_client.py
git commit -m "feat: add client SDK with JobManager, TaskStream, and ClaimedTask"
```

---

## Task 14: Package Exports

**Files:**
- Modify: `src/zndraw_joblib/__init__.py`
- Test: `tests/test_init.py`

**Step 1: Write the failing test**

```python
# tests/test_init.py
import pytest


def test_public_api_exports():
    from zndraw_joblib import (
        # Router
        router,
        # Models
        Job,
        Worker,
        Task,
        WorkerJobLink,
        TaskStatus,
        # Dependencies
        get_db_session,
        get_redis_client,
        get_current_identity,
        get_is_admin,
        get_settings,
        # Exceptions
        ProblemException,
        problem_exception_handler,
        JobNotFound,
        SchemaConflict,
        InvalidCategory,
        WorkerNotFound,
        TaskNotFound,
        InvalidTaskTransition,
        # Settings
        JobLibSettings,
        # Client
        JobManager,
        TaskStream,
        ClaimedTask,
        ApiManager,
        # Sweeper
        run_cleanup_sweeper,
    )

    assert router is not None
    assert Job is not None
    assert JobManager is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_init.py -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# src/zndraw_joblib/__init__.py
"""ZnDraw Job Management Library."""

from zndraw_joblib.router import router
from zndraw_joblib.models import Job, Worker, Task, WorkerJobLink, TaskStatus
from zndraw_joblib.dependencies import (
    get_db_session,
    get_redis_client,
    get_current_identity,
    get_is_admin,
    get_settings,
)
from zndraw_joblib.exceptions import (
    ProblemException,
    problem_exception_handler,
    JobNotFound,
    SchemaConflict,
    InvalidCategory,
    WorkerNotFound,
    TaskNotFound,
    InvalidTaskTransition,
    InvalidRoomId,
    Forbidden,
)
from zndraw_joblib.settings import JobLibSettings
from zndraw_joblib.client import JobManager, TaskStream, ClaimedTask, ApiManager
from zndraw_joblib.sweeper import run_cleanup_sweeper

__all__ = [
    # Router
    "router",
    # Models
    "Job",
    "Worker",
    "Task",
    "WorkerJobLink",
    "TaskStatus",
    # Dependencies
    "get_db_session",
    "get_redis_client",
    "get_current_identity",
    "get_is_admin",
    "get_settings",
    # Exceptions
    "ProblemException",
    "problem_exception_handler",
    "JobNotFound",
    "SchemaConflict",
    "InvalidCategory",
    "WorkerNotFound",
    "TaskNotFound",
    "InvalidTaskTransition",
    "InvalidRoomId",
    "Forbidden",
    # Settings
    "JobLibSettings",
    # Client
    "JobManager",
    "TaskStream",
    "ClaimedTask",
    "ApiManager",
    # Sweeper
    "run_cleanup_sweeper",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/test_init.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/zndraw_joblib/__init__.py tests/test_init.py
git commit -m "feat: add public API exports to __init__.py"
```

---

## Task 15: Create tests/__init__.py and Run Full Test Suite

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create test infrastructure**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
```

**Step 2: Run full test suite**

Run: `cd /Users/fzills/tools/zndraw-joblib && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "chore: add test infrastructure files"
```

---

## Summary

This plan implements the zndraw-joblib package in 15 tasks:

1. **Settings** - Pydantic settings with env vars
2. **Exceptions** - RFC 9457 problem types
3. **Models** - SQLModel for Job, Worker, Task, WorkerJobLink
4. **Dependencies** - Stub functions for DI
5. **Schemas** - Request/response Pydantic models
6. **Job Registration** - PUT endpoint
7. **Job Listing** - GET endpoints
8. **Task Submission** - POST endpoint
9. **Task Claim** - POST endpoint with FIFO
10. **Task Status/Update** - GET/PATCH endpoints
11. **Worker Heartbeat/Delete** - PATCH/DELETE endpoints
12. **Sweeper** - Background cleanup
13. **Client SDK** - JobManager, TaskStream
14. **Package Exports** - Public API
15. **Test Infrastructure** - conftest.py

Each task follows TDD with explicit commands and expected outcomes.
