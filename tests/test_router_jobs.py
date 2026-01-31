# tests/test_router_jobs.py
"""Tests for job registration endpoint using shared fixtures from conftest.py."""
import pytest


def test_register_job_global(client):
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": {"angle": 0}},
    )
    assert response.status_code == 201
    data = response.json()
    # TODO: use pydantic model to validate response
    assert data["full_name"] == "@global:modifiers:Rotate"
    assert data["worker_count"] == 1


def test_register_job_private(client):
    response = client.put(
        "/v1/joblib/rooms/room_123/jobs",
        json={"category": "selections", "name": "All", "schema": {}},
    )
    assert response.status_code == 201
    data = response.json()
    # TODO: use pydantic model to validate response
    assert data["full_name"] == "room_123:selections:All"


def test_register_job_invalid_category(client):
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "invalid_cat", "name": "Rotate", "schema": {}},
    )
    assert response.status_code == 400


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


def test_register_job_same_schema_idempotent(client):
    schema = {"angle": 0}
    # First registration
    client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": schema},
    )
    # Second registration with same schema
    response = client.put(
        "/v1/joblib/rooms/@global/jobs",
        json={"category": "modifiers", "name": "Rotate", "schema": schema},
    )
    assert response.status_code == 200  # OK, not 201


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
