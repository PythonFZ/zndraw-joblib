# ZnDraw Job Management Library

A self-contained FastAPI package for distributed job/task management with SQL persistence.

## Integration into your APP

```python
# main.py
import asyncio
from fastapi import FastAPI
from zndraw_joblib.router import router
from zndraw_joblib.dependencies import (
    get_db_session, get_current_identity, get_is_admin
)
from zndraw_joblib.exceptions import ProblemException, problem_exception_handler
from zndraw_joblib.sweeper import run_cleanup_sweeper

# 1. Your actual infrastructure
async def my_real_db_session():
    async with async_session_maker() as session:
        yield session

async def my_get_current_identity(token: str = Depends(oauth2_scheme)) -> str:
    payload = decode_jwt(token)
    return str(payload["sub"])

async def my_get_is_admin(token: str = Depends(oauth2_scheme)) -> bool:
    payload = decode_jwt(token)
    return payload.get("is_admin", False)

app = FastAPI()

# 2. Inject your infra into the package
app.dependency_overrides[get_db_session] = my_real_db_session
app.dependency_overrides[get_current_identity] = my_get_current_identity
app.dependency_overrides[get_is_admin] = my_get_is_admin

# 3. Register exception handler and router
app.add_exception_handler(ProblemException, problem_exception_handler)
app.include_router(router)

# 4. Start background sweeper
@app.on_event("startup")
async def startup():
    asyncio.create_task(
        run_cleanup_sweeper(
            base_url=settings.internal_api_url,
            interval_seconds=30,
        )
    )
```

Also import the SQLModels in your `models.py`:

```python
from zndraw_joblib.models import Job, Worker, Task, WorkerJobLink
```

## Authentication Dependencies

The package uses **dependency injection passthrough** for authentication:

| Dependency | Returns | Used For |
|------------|---------|----------|
| `get_current_identity` | `str` | Identifies user/worker for `Task.created_by_id` and `Worker.id` |
| `get_is_admin` | `bool` | Controls access to `@global` job registration |

## Host App Requirements

The host app must provide the following endpoint for the client SDK:

```
GET /v1/me
```

Returns the current authenticated user/worker identity:

```json
{
  "id": "user_123"
}
```

## Configuration

Settings via environment variables with `ZNDRAW_JOBLIB_` prefix:

```python
class JobLibSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZNDRAW_JOBLIB_")

    allowed_categories: list[str] = ["modifiers", "selections", "analysis"]
    worker_timeout_seconds: int = 60
    sweeper_interval_seconds: int = 30
    long_poll_max_wait_seconds: int = 120
```

## Job Naming Convention

Jobs use the format: `<room_id>:<category>:<name>`

- `@global:modifiers:Rotate` - global job available to all rooms
- `room_123:modifiers:Rotate` - private job for room_123 only

**Validation rules:**
- `room_id` cannot contain `@` (reserved for `@global`) or `:` (delimiter)
- `category` must be in `settings.allowed_categories`
- Same job name in same room: schema must match (409 Conflict otherwise)
- Different rooms can have same job name with different schemas

## REST Endpoints

All endpoints prefixed with `/v1/joblib` (hardcoded in router).

### Job Registration

```
PUT /v1/joblib/rooms/{room_id}/jobs
```

Register a job for a room. Use `@global` as `room_id` for global jobs.

Request body:
```json
{
  "category": "modifiers",
  "name": "Rotate",
  "schema": { ... }
}
```

- `@global` registration requires admin (`get_is_admin`)
- Re-registering validates schema match (409 if mismatch)
- Creates Worker and WorkerJobLink if not exists

### Job Listing

```
GET /v1/joblib/rooms/{room_id}/jobs     # room + @global jobs
GET /v1/joblib/rooms/@global/jobs       # only global jobs
```

### Job Details

```
GET /v1/joblib/rooms/{room_id}/jobs/{job_name}
```

`job_name` format: `<room_id>:<category>:<name>`

### Task Submission

```
POST /v1/joblib/rooms/{room_id}/tasks/{job_name}
```

Returns `202 Accepted` with `Location` header pointing to task status.

Request body:
```json
{
  "payload": { "angle": 90 }
}
```

### Task Claim (Worker)

```
POST /v1/joblib/tasks/claim
```

Claims oldest pending task (FIFO) across all jobs the worker is registered for:

```sql
SELECT task.* FROM task
JOIN worker_job_link ON task.job_id = worker_job_link.job_id
WHERE worker_job_link.worker_id = :worker_id
  AND task.status = 'pending'
ORDER BY task.created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED
```

### Task Status

```
GET /v1/joblib/tasks/{task_id}
```

Supports long-polling via `Prefer: wait=N` header (max 120s). Returns immediately on terminal states.

### Task Update

```
PATCH /v1/joblib/tasks/{task_id}
```

Valid transitions:
- `pending` → `claimed` (via `/claim`)
- `claimed` → `running`
- `running` → `completed` | `failed`
- Any → `cancelled`


### Worker Heartbeat

```
PATCH /v1/joblib/workers/{worker_id}
```

Updates `last_heartbeat`. Workers must call periodically during execution.

### Worker Deletion

```
DELETE /v1/joblib/workers/{worker_id}
```

Explicit cleanup on graceful shutdown.

## SQLModel

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Any
from uuid import uuid4
import uuid

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSON
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
    worker_id: str = Field(foreign_key="worker.id", primary_key=True)
    job_id: uuid.UUID = Field(foreign_key="job.id", primary_key=True)


class Job(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("room_id", "category", "name", name="unique_job"),
    )

    id: uuid.UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: str = Field(index=True)      # "@global" or "room_123"
    category: str = Field(index=True)     # validated against allowed list
    name: str = Field(index=True)         # e.g., "Rotate"
    schema: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Relationships
    tasks: list["Task"] = Relationship(back_populates="job")
    workers: list["Worker"] = Relationship(back_populates="jobs", link_model=WorkerJobLink)

    @property
    def full_name(self) -> str:
        return f"{self.room_id}:{self.category}:{self.name}"


class Worker(SQLModel, table=True):
    id: str = Field(primary_key=True)  # from JWT identity
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Relationships
    jobs: list[Job] = Relationship(back_populates="workers", link_model=WorkerJobLink)
    tasks: list["Task"] = Relationship(back_populates="worker")

    def is_alive(self, threshold: timedelta) -> bool:
        return datetime.utcnow() - self.last_heartbeat < threshold


class Task(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid4, primary_key=True)

    job_id: uuid.UUID = Field(foreign_key="job.id", index=True)
    job: Job = Relationship(back_populates="tasks")

    worker_id: Optional[str] = Field(default=None, foreign_key="worker.id")
    worker: Optional[Worker] = Relationship(back_populates="tasks")

    room_id: str = Field(index=True)  # room where task was submitted
    created_by_id: Optional[str] = Field(default=None, index=True)

    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: TaskStatus = Field(default=TaskStatus.PENDING, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
```

## Long-Polling

`GET /v1/joblib/tasks/{task_id}` with `Prefer: wait=N`:
- Server polls database until task reaches terminal state or timeout
- Returns immediately on terminal state (`completed`, `failed`, `cancelled`)
- Returns `Preference-Applied: wait` header if honored
- Maximum wait capped by `settings.long_poll_max_wait_seconds`

## Background Sweeper

Host app starts explicitly:

```python
asyncio.create_task(
    run_cleanup_sweeper(base_url=..., interval_seconds=30)
)
```

The sweeper:
1. Finds workers with stale `last_heartbeat`
2. Marks their `running` tasks as `FAILED`
3. Removes orphan jobs (no workers, no pending tasks)

## Error Handling (RFC 9457)

```python
class JobNotFound(ProblemType):
    title = "Not Found"
    status = 404

class SchemaConflict(ProblemType):
    title = "Conflict"
    status = 409

class InvalidCategory(ProblemType):
    title = "Bad Request"
    status = 400

class WorkerNotFound(ProblemType):
    title = "Not Found"
    status = 404

class TaskNotFound(ProblemType):
    title = "Not Found"
    status = 404

class InvalidTaskTransition(ProblemType):
    title = "Conflict"
    status = 409
```

## Client

```python
import time
import httpx
from datetime import datetime
from typing import Any, Protocol
from pydantic import BaseModel


class ApiManager(Protocol):
    http: httpx.Client
    def get_headers(self) -> dict[str, str]: ...
    @property
    def base_url(self) -> str: ...


class ClaimedTask(BaseModel):
    """Pydantic model for a claimed task."""
    id: str
    job_name: str  # full name: room_id:category:name
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

    def __getitem__(self, key: str):
        return self._registry[key]

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def register(self, task_class: type[BaseModel] | None = None, *, room: str | None = None):
        """
        Register a job. room=None defaults to @global.

        Extension classes are Pydantic BaseModels. Schema is derived via model_json_schema().

        Usage:
            @vis.jobs.register
            class Rotate(BaseModel):
                angle: float = 0.0

            @vis.jobs.register(room="room_123")
            class PrivateRotate(BaseModel):
                angle: float = 0.0

            vis.jobs.register(MyJob, room="room_123")
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

        # Schema from Pydantic model
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
        response = self.api.http.get(
            f"{self.api.base_url}/v1/me",
            headers=self.api.get_headers(),
        )
        return response.json()["id"]

    def heartbeat(self):
        """Manually send worker heartbeat. Call during long task execution."""
        worker_id = self._get_worker_id()
        self.api.http.patch(
            f"{self.api.base_url}/v1/joblib/workers/{worker_id}",
            headers=self.api.get_headers(),
        )
```

### Usage Example

```python
from pydantic import BaseModel

# Extension classes are Pydantic BaseModels
class Rotate(BaseModel):
    category: str = "modifiers"  # class attribute for category
    angle: float = 0.0
    axis: str = "z"

# 1. Register jobs
@vis.jobs.register  # @global by default
class Rotate(BaseModel):
    angle: float = 0.0

@vis.jobs.register(room="room_123")
class PrivateRotate(BaseModel):
    angle: float = 0.0

print(len(vis.jobs))  # 2 Jobs Registered

# 2. Consume tasks
for task in vis.jobs.listen():
    vis.jobs.heartbeat()  # keep alive during processing
    task.run(vis)
```

### Notes

- Extension classes are **Pydantic BaseModels**
- Schema is generated via `cls.model_json_schema()`
- Re-registering validates schema match (raises on mismatch)
- `room=None` (default) registers to `@global`
- Worker must call `heartbeat()` explicitly during long-running tasks
- Same identity can both submit tasks and process them (unified `get_current_identity`)
