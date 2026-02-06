# src/zndraw_joblib/client.py
"""Client SDK for ZnDraw JobLib workers."""

import logging
import time
from enum import Enum
from typing import ClassVar, Generic, Iterator, Protocol, TypeVar
from uuid import UUID

import httpx
from pydantic import BaseModel

from zndraw_joblib.schemas import (
    JobRegisterRequest,
    TaskSubmitRequest,
    TaskClaimResponse,
)

logger = logging.getLogger(__name__)


class Category(str, Enum):
    """Extension category types."""

    MODIFIER = "modifiers"
    SELECTION = "selections"
    ANALYSIS = "analysis"


class Extension(BaseModel):
    """Base class for all ZnDraw extensions.

    Extensions are Pydantic models that define their parameters as fields.
    The JSON schema is generated from the model for the frontend forms.
    """

    category: ClassVar[Category]

    def run(self) -> None:
        """Execute the extension logic. Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")


E = TypeVar("E", bound=Extension)


class ApiManager(Protocol):
    """Protocol for API client that JobManager uses."""

    http: httpx.Client

    def get_headers(self) -> dict[str, str]: ...

    @property
    def base_url(self) -> str: ...


class ClaimedTask(Generic[E]):
    """A claimed task with its Extension instance and metadata."""

    def __init__(
        self,
        task_id: str,
        job_name: str,
        room_id: str,
        extension: E,
    ):
        self.task_id = task_id
        self.job_name = job_name
        self.room_id = room_id
        self.extension = extension


class JobManager:
    """Main entry point for workers. Registers jobs and claims tasks."""

    def __init__(self, api: ApiManager):
        self.api = api
        self._registry: dict[str, type[Extension]] = {}
        self._worker_id: UUID | None = None

    def __getitem__(self, key: str) -> type[Extension]:
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    @property
    def worker_id(self) -> UUID | None:
        """The worker ID for this manager (set after create_worker or first job registration)."""
        return self._worker_id

    def create_worker(self) -> UUID:
        """Create a new worker and return its ID.

        The worker is linked to the authenticated user. Use this to explicitly
        create a worker before registering jobs.
        """
        response = self.api.http.post(
            f"{self.api.base_url}/v1/joblib/workers",
            headers=self.api.get_headers(),
        )
        response.raise_for_status()
        self._worker_id = UUID(response.json()["id"])
        return self._worker_id

    def register(
        self,
        extension_class: type[Extension] | None = None,
        *,
        room: str | None = None,
    ):
        """
        Register an Extension class as a job.

        Usage:
            @manager.register
            class Rotate(Extension):
                category: ClassVar[Category] = Category.MODIFIER
                angle: float = 0.0

            @manager.register(room="room_123")
            class PrivateRotate(Extension):
                category: ClassVar[Category] = Category.MODIFIER
                angle: float = 0.0
        """

        def decorator(cls: type[Extension]) -> type[Extension]:
            self._register_impl(cls, room)
            return cls

        if extension_class is None:
            return decorator
        return decorator(extension_class)

    def _register_impl(self, cls: type[Extension], room: str | None) -> None:
        """Internal implementation of job registration."""
        room_id = room if room is not None else "@global"
        category = cls.category.value
        name = cls.__name__

        schema = cls.model_json_schema()

        # Build request using Pydantic model
        request = JobRegisterRequest(
            category=category,
            name=name,
            schema=schema,
            worker_id=self._worker_id,
        )

        resp = self.api.http.put(
            f"{self.api.base_url}/v1/joblib/rooms/{room_id}/jobs",
            headers=self.api.get_headers(),
            json=request.model_dump(exclude_none=True, mode="json"),
        )
        resp.raise_for_status()

        data = resp.json()
        full_name = f"{room_id}:{category}:{name}"

        # Extract worker_id from response if auto-created
        if "worker_id" in data and data["worker_id"]:
            self._worker_id = UUID(data["worker_id"])

        if resp.status_code == 200:
            logger.info("Already registered: %s", full_name)
        self._registry[full_name] = cls

    def claim(self) -> ClaimedTask | None:
        """
        Attempt to claim a single task.

        Returns ClaimedTask with the Extension instance, or None if no tasks available.

        Raises:
            ValueError: If worker_id is not set. Call create_worker() or register a job first.
        """
        if self._worker_id is None:
            raise ValueError("Worker ID not set. Call create_worker() or register a job first.")

        response = self.api.http.post(
            f"{self.api.base_url}/v1/joblib/tasks/claim",
            headers=self.api.get_headers(),
            json={"worker_id": str(self._worker_id)},
        )
        response.raise_for_status()
        claim_response = TaskClaimResponse.model_validate(response.json())

        if claim_response.task is None:
            return None

        task = claim_response.task
        job_name = task.job_name

        # Look up the Extension class from registry
        if job_name not in self._registry:
            raise KeyError(f"Job '{job_name}' not registered with this JobManager")

        extension_cls = self._registry[job_name]
        extension = extension_cls.model_validate(task.payload)

        return ClaimedTask(
            task_id=str(task.id),
            job_name=job_name,
            room_id=task.room_id,
            extension=extension,
        )

    def listen(self, polling_interval: float = 2.0) -> Iterator[ClaimedTask]:
        """
        Generator that yields claimed tasks indefinitely.

        Polls the server for tasks at the specified interval.
        """
        while True:
            claimed = self.claim()
            if claimed is not None:
                yield claimed
            else:
                time.sleep(polling_interval)

    def heartbeat(self) -> None:
        """Send a heartbeat to keep the worker alive."""
        if self._worker_id is None:
            raise ValueError(
                "Worker ID not set. Call create_worker() or register a job first."
            )
        response = self.api.http.patch(
            f"{self.api.base_url}/v1/joblib/workers/{self._worker_id}",
            headers=self.api.get_headers(),
        )
        response.raise_for_status()

    def submit(
        self, extension: Extension, room: str, *, job_room: str = "@global"
    ) -> str:
        """
        Submit a task for processing.

        Args:
            extension: The Extension instance with parameters
            room: The room to submit the task to
            job_room: The room where the job is registered (default: @global)

        Returns:
            The task ID
        """
        category = extension.category.value
        name = extension.__class__.__name__
        job_name = f"{job_room}:{category}:{name}"

        # Build request using Pydantic model
        request = TaskSubmitRequest(payload=extension.model_dump())

        response = self.api.http.post(
            f"{self.api.base_url}/v1/joblib/rooms/{room}/tasks/{job_name}",
            headers=self.api.get_headers(),
            json=request.model_dump(),
        )
        response.raise_for_status()
        return response.json()["id"]
