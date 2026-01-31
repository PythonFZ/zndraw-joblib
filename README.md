# ZnDraw Job Management Library

## Integration into your APP
```python
# main.py
from fastapi import FastAPI
from zndraw_joblib.router import queue_router
from zndraw_joblib.dependencies import get_db_session, get_redis_client

# 1. Your actual infrastructure
async def my_real_db_session():
    async with async_session_maker() as session:
        yield session

async def my_real_redis():
    return global_redis_pool

app = FastAPI()

# 2. THE MAGIC: Inject your infra into the package
app.dependency_overrides[get_db_session] = my_real_db_session
app.dependency_overrides[get_redis_client] = my_real_redis

# 3. Register the router
app.include_router(queue_router)
```

## JWT / Authentication

The package uses **dependency injection passthrough** for authentication. It defines placeholder dependencies that your app must override:

```python
from zndraw_joblib.dependencies import get_current_user_id, get_is_admin

# Your app's implementation
async def my_get_current_user_id(
    token: str = Depends(oauth2_scheme)
) -> int: ...

async def my_get_is_admin(
    token: str = Depends(oauth2_scheme)
) -> bool: ...

# Inject your auth
app.dependency_overrides[get_current_user_id] = my_get_current_user_id
app.dependency_overrides[get_is_admin] = my_get_is_admin
```

Also make sure to import the SQLModels inside your `models.py` 

```py
from zndraw_joblib.models import Task
```

## REST Endpoints
PUT /v1/joblib/rooms/{room_id}/jobs register a job for that room
PUT /v1/joblib/rooms/@global/jobs register a job for all rooms

GET /v1/joblib/rooms/{room_id}/jobs list all jobs for that room ([...@global, ...room_id])
GET /v1/joblib/rooms/@global/jobs list all global jobs

GET /v1/joblib/rooms/{room_id}/jobs/{job_name} -> get job details (both [...@global, ...room_id])
GET /v1/joblib/rooms/@global/jobs/{job_name} -> get job details (only [...@global])

POST /v1/joblib/rooms/{room_id}/tasks/{job_name} -> push task (job name is always <prefix>:<category>:<name>) where prefix is `public` or `private`

POST /v1/joblib/tasks/claim -> Worker claims next task

GET  /v1/joblib/tasks/{task_id} -> Get task status (supports Prefer: wait=N long-polling)
PATCH /v1/joblib/tasks/{task_id} -> Update task status (RUNNING/COMPLETED/FAILED)

PATCH /v1/joblib/workers/{worker_id} -> Worker heartbeat (updates last_heartbeat)


# TODO: worker heartbeat!

## SQLModel

```python
class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Job(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("name", "room_id", name="unique_job_per_room"),)
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True)      # e.g., "send_email"
    room_id: str = Field(index=True)   # "@global" or "room_123"
    
    schema: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Relationships
    tasks: List["Task"] = Relationship(back_populates="job")
    workers: List["Worker"] = Relationship(back_populates="jobs", link_model=WorkerJobLink)

class Worker(SQLModel, table=True):
    id: str = Field(primary_key=True) # The specific worker instance ID
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Relationships
    jobs: List[Job] = Relationship(back_populates="workers", link_model=WorkerJobLink)
    tasks: List["Task"] = Relationship(back_populates="worker")

    def is_alive(self, threshold: timedelta) -> bool:
        return datetime.utcnow() - self.last_heartbeat < threshold

class Task(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    job_id: uuid.UUID = Field(foreign_key="job.id", index=True)
    job: Job = Relationship(back_populates="tasks")
    
    worker_id: Optional[str] = Field(default=None, foreign_key="worker.id")
    worker: Optional[Worker] = Relationship(back_populates="tasks")

    room_id: str = Field(index=True) 
    created_by_id: Optional[int] = Field(default=None, index=True)

    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: TaskStatus = Field(default=TaskStatus.PENDING, index=True)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
```

- Long-polling is realized, by using pub/sub on Redis because all updates go through PATCH /v1/joblib/tasks/{task_id} which publishes to a Redis channel specific to that task ID and the long-polling GET /v1/joblib/tasks/{task_id} subscribes to that channel waiting for updates.
- worker_timeout_threshold is defined via `pydantic-settings`
- periodic background task checks for 
    - running jobs with no alive workers and marks them as FAILED.
    - jobs without workers and without pending tasks and remove them.



## Client
```python
from collections.abc import MutableMapping, Iterator
from typing import Type, Union, Optional
import time

class TaskStream(Iterator):
    """A dedicated iterator object for consuming tasks."""
    def __init__(self, api, polling_interval=2.0):
        self.api = api
        self.interval = polling_interval
        self._stop_event = False

    def __next__(self):
        if self._stop_event:
            raise StopIteration
        
        # Blocking poll loop
        while True:
            task = self.api.claim_task()
            if task:
                return task
            time.sleep(self.interval)

    def stop(self):
        self._stop_event = True


class JobManager(MutableMapping):
    """
    Main Entry Point.
    Behaves like a Dictionary of registered jobs.
    """
    def __init__(self, api, context):
        self.api = api
        self.context = context
        self._registry = {}

    # --- Mapping ABC Implementation ---
    def __getitem__(self, key): return self._registry[key]
    def __setitem__(self, key, value): self._registry[key] = value
    def __delitem__(self, key): del self._registry[key]
    def __len__(self): return len(self._registry)
    def __iter__(self): return iter(self._registry) # Iterates registered job names

    # --- Registration (Hybrid Decorator) ---
    def register(self, task_class=None, *, public=False):
        if task_class is None:
            def wrapper(cls):
                self._register_impl(cls, public)
                return cls
            return wrapper
        self._register_impl(task_class, public)
        return task_class

    def _register_impl(self, cls, public):
        # Logic to register with API
        name = cls.__name__
        self[name] = cls # Uses __setitem__ logic

    # --- The Consumer Factory ---
    def listen(self, interval=2.0) -> TaskStream:
        """Returns an Iterator to process tasks."""
        return TaskStream(self.api, interval)

# 1. Registry (Dict-like)
@vis.jobs.register
class Rotate(Extension): ...

print(len(vis.jobs)) # "1 Job Registered"

# 2. Consumption (Iterator-like)
# "listen()" clearly indicates we are switching modes to consuming
for task in vis.jobs.listen():
    task.run(vis)
```

- Registering an Extension which is already registered will check if the schema MATCH and raise an Error if not.