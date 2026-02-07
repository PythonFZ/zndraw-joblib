"""Tests for internal job registry."""

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

from zndraw_joblib.client import Extension, Category
from zndraw_joblib.registry import InternalRegistry, register_internal_tasks


class Rotate(Extension):
    category: ClassVar[Category] = Category.MODIFIER
    angle: float = 0.0


class Scale(Extension):
    category: ClassVar[Category] = Category.MODIFIER
    factor: float = 1.0


def make_mock_broker():
    """Create a mock broker with register_task that returns a mock task handle."""
    broker = MagicMock()
    task_handles = {}

    def mock_register_task(fn, task_name, **kwargs):
        handle = MagicMock()
        handle.kiq = AsyncMock()
        task_handles[task_name] = handle
        return handle

    broker.register_task = mock_register_task
    broker._task_handles = task_handles
    return broker


async def mock_executor(
    extension_cls: type[Extension],
    payload: dict[str, Any],
    room_id: str,
    task_id: str,
    base_url: str,
) -> None:
    pass


def test_register_internal_tasks_returns_registry():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [Rotate, Scale], executor=mock_executor)

    assert isinstance(registry, InternalRegistry)
    assert "@internal:modifiers:Rotate" in registry.tasks
    assert "@internal:modifiers:Scale" in registry.tasks
    assert len(registry.tasks) == 2


def test_register_internal_tasks_stores_extension_classes():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [Rotate], executor=mock_executor)

    assert registry.extensions["@internal:modifiers:Rotate"] is Rotate


def test_register_internal_tasks_registers_on_broker():
    broker = make_mock_broker()
    register_internal_tasks(broker, [Rotate], executor=mock_executor)

    assert "@internal:modifiers:Rotate" in broker._task_handles


def test_register_internal_tasks_empty_list():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [], executor=mock_executor)

    assert len(registry.tasks) == 0
    assert len(registry.extensions) == 0


def test_internal_registry_executor_stored():
    broker = make_mock_broker()
    registry = register_internal_tasks(broker, [Rotate], executor=mock_executor)

    assert registry.executor is mock_executor
