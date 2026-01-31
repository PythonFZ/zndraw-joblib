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
