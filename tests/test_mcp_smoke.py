"""
Smoke tests for MCP (Model Context Protocol) endpoints.
"""

from fastapi.testclient import TestClient

from orion.api import app

client = TestClient(app)


def test_mcp_get_status() -> None:
    """Test MCP getStatus verb returns expected response."""
    response = client.post("/mcp/invoke", json={"verb": "getStatus", "args": {}})

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "ok" in data
    assert "data" in data
    assert "agent" in data

    # Check response values
    assert data["ok"] is True
    assert data["agent"] == "orion"
    assert data["data"]["status"] == "active"
    assert data["data"]["service"] == "payout-management"


def test_mcp_get_payout_options() -> None:
    """Test MCP getPayoutOptions verb returns expected response."""
    response = client.post("/mcp/invoke", json={"verb": "getPayoutOptions", "args": {}})

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "ok" in data
    assert "data" in data
    assert "agent" in data

    # Check response values
    assert data["ok"] is True
    assert data["agent"] == "orion"

    # Check payout options structure
    payout_data = data["data"]
    assert "payout_options" in payout_data
    assert "total_options" in payout_data
    assert "best_rail" in payout_data
    assert "explanation" in payout_data

    # Check payout options count
    assert (
        payout_data["total_options"] == 4
    )  # All 4 rails should be available for $1000
    assert payout_data["best_rail"]["rail"] in ["ACH", "WIRE", "RTP", "V_CARD"]

    # Check first payout option structure
    first_option = payout_data["payout_options"][0]
    expected_fields = [
        "rail",
        "name",
        "description",
        "processing_time",
        "fees",
        "limits",
        "score",
        "enabled",
    ]
    for field in expected_fields:
        assert field in first_option

    # Check specific values
    assert first_option["enabled"] is True
    assert isinstance(first_option["score"], (int, float))
    assert first_option["score"] > 0


def test_mcp_unsupported_verb() -> None:
    """Test MCP endpoint returns error for unsupported verb."""
    response = client.post("/mcp/invoke", json={"verb": "unsupportedVerb", "args": {}})

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Unsupported verb" in data["detail"]


def test_mcp_missing_verb() -> None:
    """Test MCP endpoint returns error for missing verb."""
    response = client.post("/mcp/invoke", json={"args": {}})

    assert response.status_code == 422  # Validation error


def test_mcp_invoke_content_type() -> None:
    """Test that MCP endpoint returns JSON content type."""
    response = client.post("/mcp/invoke", json={"verb": "getStatus", "args": {}})

    assert response.headers["content-type"] == "application/json"
