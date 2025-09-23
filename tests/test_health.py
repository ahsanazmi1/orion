"""
Tests for the health check endpoint.
"""

from fastapi.testclient import TestClient

from orion.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test that the health endpoint returns the expected response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "repo": "orion"}


def test_health_endpoint_content_type() -> None:
    """Test that the health endpoint returns JSON content type."""
    response = client.get("/health")

    assert response.headers["content-type"] == "application/json"
