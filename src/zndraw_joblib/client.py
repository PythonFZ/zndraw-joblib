# src/zndraw_joblib/client.py
"""Client SDK for ZnDraw JobLib workers."""
import time
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Generic, Iterator, Protocol, TypeVar

import httpx
from pydantic import BaseModel


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
        self._worker_id: str | None = None

    def __getitem__(self, key: str) -> type[Extension]:
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

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

        self.api.http.put(
            f"{self.api.base_url}/v1/joblib/rooms/{room_id}/jobs",
            headers=self.api.get_headers(),
            json={"category": category, "name": name, "schema": schema},
        )

        full_name = f"{room_id}:{category}:{name}"
        self._registry[full_name] = cls

    def claim(self) -> ClaimedTask | None:
        """
        Attempt to claim a single task.

        Returns ClaimedTask with the Extension instance, or None if no tasks available.
        """
        response = self.api.http.post(
            f"{self.api.base_url}/v1/joblib/tasks/claim",
            headers=self.api.get_headers(),
        )
        data = response.json()

        if data.get("task") is None:
            return None

        task_data = data["task"]
        job_name = task_data["job_name"]

        # Look up the Extension class from registry
        if job_name not in self._registry:
            raise KeyError(f"Job '{job_name}' not registered with this JobManager")

        extension_cls = self._registry[job_name]
        extension = extension_cls.model_validate(task_data["payload"])

        return ClaimedTask(
            task_id=task_data["id"],
            job_name=job_name,
            room_id=task_data["room_id"],
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
        worker_id = self._get_worker_id()
        self.api.http.patch(
            f"{self.api.base_url}/v1/joblib/workers/{worker_id}",
            headers=self.api.get_headers(),
        )

    def _get_worker_id(self) -> str:
        """Get or fetch the worker ID."""
        if self._worker_id is None:
            # In a real implementation, this would come from auth
            # For now, we assume it's set externally or via registration
            raise ValueError("Worker ID not set. Set manager._worker_id or authenticate first.")
        return self._worker_id
