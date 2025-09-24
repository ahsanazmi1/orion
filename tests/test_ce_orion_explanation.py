"""
Tests for Orion CloudEvents explanation schema validation.
"""

from orion.ce import (
    create_explanation_payload,
    emit_explanation_ce,
    format_ce_for_logging,
    get_trace_id,
    validate_ce_schema,
)


def test_emit_explanation_ce_required_fields() -> None:
    """Test that emitted CloudEvent has all required fields."""
    trace_id = "test-trace-123"
    payload = {
        "reason": "Test reason",
        "signals": ["test_signal"],
        "mitigation": "Test mitigation",
        "confidence": 0.8,
    }

    event = emit_explanation_ce(trace_id, payload)

    # Check required CloudEvent fields
    required_fields = [
        "specversion",
        "type",
        "source",
        "id",
        "time",
        "subject",
        "datacontenttype",
        "data",
    ]

    for field in required_fields:
        assert field in event
        assert event[field] is not None

    # Check specific values
    assert event["specversion"] == "1.0"
    assert event["type"] == "ocn.orion.explanation.v1"
    assert event["source"] == "orion"
    assert event["subject"] == trace_id
    assert event["datacontenttype"] == "application/json"
    assert event["data"] == payload


def test_emit_explanation_ce_unique_ids() -> None:
    """Test that each emitted event has unique ID."""
    trace_id = "test-trace-123"
    payload = {
        "reason": "Test reason",
        "signals": ["test_signal"],
        "mitigation": "Test mitigation",
        "confidence": 0.8,
    }

    event1 = emit_explanation_ce(trace_id, payload)
    event2 = emit_explanation_ce(trace_id, payload)

    # IDs should be different
    assert event1["id"] != event2["id"]

    # But other fields should be the same
    assert event1["type"] == event2["type"]
    assert event1["source"] == event2["source"]
    assert event1["subject"] == event2["subject"]
    assert event1["data"] == event2["data"]


def test_validate_ce_schema_valid_event() -> None:
    """Test validation of a valid CloudEvent."""
    valid_event = {
        "specversion": "1.0",
        "type": "ocn.orion.explanation.v1",
        "source": "orion",
        "id": "test-id-123",
        "time": "2024-01-01T00:00:00Z",
        "subject": "test-trace-123",
        "datacontenttype": "application/json",
        "data": {
            "reason": "Test reason",
            "signals": ["test_signal"],
            "mitigation": "Test mitigation",
            "confidence": 0.8,
        },
    }

    assert validate_ce_schema(valid_event) is True


def test_validate_ce_schema_invalid_event() -> None:
    """Test validation of invalid CloudEvents."""
    # Missing required field
    invalid_event = {
        "specversion": "1.0",
        "type": "ocn.orion.explanation.v1",
        "source": "orion",
        # Missing "id"
        "time": "2024-01-01T00:00:00Z",
        "subject": "test-trace-123",
        "datacontenttype": "application/json",
        "data": {
            "reason": "Test reason",
            "signals": ["test_signal"],
            "mitigation": "Test mitigation",
            "confidence": 0.8,
        },
    }

    assert validate_ce_schema(invalid_event) is False

    # Wrong type
    invalid_event["id"] = "test-id-123"
    invalid_event["type"] = "wrong.type.v1"
    assert validate_ce_schema(invalid_event) is False

    # Wrong source
    invalid_event["type"] = "ocn.orion.explanation.v1"
    invalid_event["source"] = "wrong-source"
    assert validate_ce_schema(invalid_event) is False

    # Invalid confidence (out of range)
    invalid_event["source"] = "orion"
    invalid_event["data"]["confidence"] = 1.5
    assert validate_ce_schema(invalid_event) is False

    # Invalid signals (not a list)
    invalid_event["data"]["confidence"] = 0.8
    invalid_event["data"]["signals"] = "not_a_list"
    assert validate_ce_schema(invalid_event) is False


def test_create_explanation_payload() -> None:
    """Test creation of explanation payload."""
    best_rail = {
        "rail_id": "ach",
        "description": "ACH (Automated Clearing House)",
        "scores": {"total": 85.0},
    }

    explanation = {
        "reason": "ACH was selected for cost-effectiveness",
        "signals": ["cost_effective", "standard_rail_selected"],
        "mitigation": "No mitigation needed",
        "confidence": 0.9,
    }

    context = {"amount": 1000.0, "urgency": "normal", "vendor_id": "test_vendor"}

    payload = create_explanation_payload(best_rail, explanation, context)

    # Check required fields
    assert "rail_selection" in payload
    assert "explanation" in payload
    assert "context" in payload
    assert "timestamp" in payload
    assert "metadata" in payload

    # Check rail selection
    assert payload["rail_selection"]["rail_id"] == "ach"
    assert payload["rail_selection"]["score"] == 85.0

    # Check explanation
    assert payload["explanation"] == explanation

    # Check context
    assert payload["context"]["amount"] == 1000.0
    assert payload["context"]["urgency"] == "normal"
    assert payload["context"]["vendor_id"] == "test_vendor"

    # Check metadata
    assert payload["metadata"]["service"] == "orion"
    assert payload["metadata"]["version"] == "1.0.0"
    assert payload["metadata"]["feature"] == "payout_optimization"


def test_get_trace_id() -> None:
    """Test trace ID generation."""
    trace_id1 = get_trace_id()
    trace_id2 = get_trace_id()

    # Should generate different IDs
    assert trace_id1 != trace_id2

    # Should be strings
    assert isinstance(trace_id1, str)
    assert isinstance(trace_id2, str)

    # Should be non-empty
    assert len(trace_id1) > 0
    assert len(trace_id2) > 0


def test_format_ce_for_logging() -> None:
    """Test CloudEvent formatting for logging."""
    event = {
        "specversion": "1.0",
        "type": "ocn.orion.explanation.v1",
        "source": "orion",
        "id": "test-id-123",
        "time": "2024-01-01T00:00:00Z",
        "subject": "test-trace-123",
        "datacontenttype": "application/json",
        "data": {
            "reason": "Test reason",
            "signals": ["test_signal"],
            "mitigation": "Test mitigation",
            "confidence": 0.8,
        },
    }

    formatted = format_ce_for_logging(event)

    # Should be a string
    assert isinstance(formatted, str)

    # Should contain key information
    assert "ocn.orion.explanation.v1" in formatted
    assert "test-trace-123" in formatted
    assert "Test reason" in formatted


def test_end_to_end_ce_validation() -> None:
    """Test end-to-end CloudEvent creation and validation."""
    # Create explanation payload
    best_rail = {
        "rail_id": "rtp",
        "description": "Real-Time Payments",
        "scores": {"total": 90.0},
    }

    explanation = {
        "reason": "RTP selected for speed and efficiency",
        "signals": ["fast_processing", "instant_rail_selected"],
        "mitigation": "Monitor for any processing delays",
        "confidence": 0.95,
    }

    context = {"amount": 5000.0, "urgency": "high", "vendor_id": "urgent_vendor"}

    payload = create_explanation_payload(best_rail, explanation, context)

    # Emit CloudEvent
    trace_id = get_trace_id()
    event = emit_explanation_ce(trace_id, payload)

    # Validate the event
    assert validate_ce_schema(event) is True

    # Check that data matches payload
    assert event["data"] == payload
    assert event["subject"] == trace_id

    # Format for logging
    formatted = format_ce_for_logging(event)
    assert isinstance(formatted, str)
    assert len(formatted) > 0
