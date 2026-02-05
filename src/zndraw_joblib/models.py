# src/zndraw_joblib/models.py
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import uuid

from sqlalchemy import UniqueConstraint, ForeignKey, String, Boolean, DateTime, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from zndraw_auth import Base


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class WorkerJobLink(Base):
    """Bare M:N link between Worker and Job."""

    __tablename__ = "worker_job_link"

    worker_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("worker.id", ondelete="CASCADE"), primary_key=True
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("job.id", ondelete="CASCADE"), primary_key=True
    )


class Job(Base):
    __tablename__ = "job"
    __table_args__ = (
        UniqueConstraint("room_id", "category", "name", name="unique_job"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid4)
    room_id: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    schema_: Mapped[dict[str, Any]] = mapped_column("schema", JSON, default=dict)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Relationships
    tasks: Mapped[list["Task"]] = relationship(back_populates="job")
    workers: Mapped[list["Worker"]] = relationship(
        back_populates="jobs", secondary="worker_job_link"
    )

    @property
    def full_name(self) -> str:
        return f"{self.room_id}:{self.category}:{self.name}"


class Worker(Base):
    __tablename__ = "worker"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), index=True
    )
    last_heartbeat: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )

    # Relationships
    jobs: Mapped[list[Job]] = relationship(
        back_populates="workers", secondary="worker_job_link"
    )
    tasks: Mapped[list["Task"]] = relationship(back_populates="worker")

    def is_alive(self, threshold: timedelta) -> bool:
        return datetime.now(timezone.utc) - self.last_heartbeat < threshold


class Task(Base):
    __tablename__ = "task"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid4)

    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("job.id"), index=True)
    job: Mapped[Optional[Job]] = relationship(back_populates="tasks")

    worker_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("worker.id"), default=None, index=True, nullable=True
    )
    worker: Mapped[Optional[Worker]] = relationship(back_populates="tasks")

    room_id: Mapped[str] = mapped_column(String, index=True)
    created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        default=None, index=True, nullable=True
    )

    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[TaskStatus] = mapped_column(default=TaskStatus.PENDING, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=None, nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, default=None, nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(Text, default=None, nullable=True)
