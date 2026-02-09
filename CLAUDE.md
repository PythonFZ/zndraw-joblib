# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`zndraw-joblib` is a self-contained FastAPI package for distributed job/task management with SQL persistence. It provides a pluggable router, ORM models, a client SDK, and a background sweeper — designed to be mounted into a host FastAPI app via dependency injection overrides.

## Commands

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_router.py     # run a single test file
uv run pytest -k "test_claim"          # run tests matching a name
uv run ruff check                      # lint
uv run ruff format                     # format
uv run ruff check --fix                # auto-fix lint issues
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` — all async test functions run automatically without `@pytest.mark.asyncio`.

## Architecture

### Module Layout (`src/zndraw_joblib/`)

- **router.py** — FastAPI router with all REST endpoints under `/v1/joblib`. The central module; handles job registration, task submission/claim/update, worker management, and long-polling.
- **models.py** — SQLAlchemy 2.0 ORM models (`Job`, `Worker`, `Task`, `WorkerJobLink`) inheriting from `zndraw_auth.Base`.
- **schemas.py** — Pydantic request/response models including `PaginatedResponse[T]` generic envelope.
- **client.py** — Synchronous client SDK (`JobManager`, `Extension`, `ClaimedTask`) using `httpx`. Workers use `Extension` subclasses with a `category` ClassVar and `run()` method.
- **dependencies.py** — FastAPI dependencies. `get_async_session_maker` is the single override point for all DB access; `get_locked_async_session` and `get_session_factory` derive from it via DI.
- **settings.py** — `JobLibSettings` using pydantic-settings with `ZNDRAW_JOBLIB_` env prefix.
- **exceptions.py** — RFC 9457 Problem Details error types (`JobNotFound`, `SchemaConflict`, `InvalidTaskTransition`, etc.).
- **sweeper.py** — Background coroutine that cleans up stale workers and orphan jobs.

### Key Design Decisions

**Dependency injection passthrough**: The package does NOT own database sessions or auth. Host apps override `get_async_session_maker` (single point for all DB access), plus `current_active_user` and `current_superuser` from `zndraw_auth`. Both `get_locked_async_session` and `get_session_factory` derive from `get_async_session_maker` via DI.

**Job naming**: `{room_id}:{category}:{name}` — `@global` for cross-room jobs, otherwise room-scoped. Room IDs cannot contain `@` or `:`.

**Task lifecycle**: `PENDING → CLAIMED → RUNNING → COMPLETED|FAILED|CANCELLED`. Claiming uses optimistic locking (atomic UPDATE with WHERE clause + exponential backoff).

**SQLite compatibility**: An optional `asyncio.Lock` serializes DB access when `enable_db_lock=True`. Disable for PostgreSQL.

**Long-polling**: `GET /tasks/{id}` supports `Prefer: wait=N` header. Uses a session factory to create short-lived sessions per poll iteration.

**Soft deletion**: Jobs are marked `deleted=True` rather than removed, preserving task history. Re-registration reactivates them.

### Test Structure

Tests use in-memory SQLite via `conftest.py` fixtures. Key fixtures:
- `client` — `TestClient` with full dependency overrides
- `seeded_client` — pre-registered `@global:modifiers:Rotate` job
- `client_factory` — creates clients with distinct user identities (for multi-worker tests)
- `async_client` — `httpx.AsyncClient` for concurrent stress tests

### Relationship to zndraw-fastapi

The sibling repo at `/Users/fzills/tools/zndraw-fastapi` is the host app. It currently has its own parallel worker system in `src/zndraw/workers/` but does not yet depend on this package. Both repos share `zndraw-auth` for authentication models.
