"""
Orion CloudEvents module.

Handles emission and validation of CloudEvents for payout explanations.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ExplanationEvent(BaseModel):
    """CloudEvent model for Orion explanations."""

    specversion: str = "1.0"
    type: str = "ocn.orion.explanation.v1"
    source: str = "orion"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    subject: str  # trace_id
    datacontenttype: str = "application/json"
    data: dict[str, Any]


def emit_explanation_ce(trace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Emit a CloudEvent for payout explanation.

    Args:
        trace_id: Trace ID for the request
        payload: Explanation payload

    Returns:
        CloudEvent envelope
    """
    event = ExplanationEvent(subject=trace_id, data=payload)

    return event.model_dump()


def validate_ce_schema(event: dict[str, Any]) -> bool:
    """
    Validate CloudEvent against ocn.orion.explanation.v1 schema.

    Args:
        event: CloudEvent to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
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
            if field not in event:
                return False

        # Validate specversion
        if event["specversion"] != "1.0":
            return False

        # Validate type
        if event["type"] != "ocn.orion.explanation.v1":
            return False

        # Validate source
        if event["source"] != "orion":
            return False

        # Validate datacontenttype
        if event["datacontenttype"] != "application/json":
            return False

        # Validate data structure
        data = event["data"]
        if not isinstance(data, dict):
            return False

        # Check required data fields - handle nested structure
        if "explanation" in data:
            # Nested structure from create_explanation_payload
            explanation = data["explanation"]
            required_fields = ["reason", "signals", "mitigation", "confidence"]
            for field in required_fields:
                if field not in explanation:
                    return False

            # Validate confidence is a number between 0 and 1
            confidence = explanation["confidence"]
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                return False

            # Validate signals is a list
            if not isinstance(explanation["signals"], list):
                return False
        else:
            # Direct structure (legacy)
            required_data_fields = ["reason", "signals", "mitigation", "confidence"]
            for field in required_data_fields:
                if field not in data:
                    return False

            # Validate confidence is a number between 0 and 1
            confidence = data["confidence"]
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                return False

            # Validate signals is a list
            if not isinstance(data["signals"], list):
                return False

        return True

    except Exception:
        return False


def create_explanation_payload(
    best_rail: dict[str, Any], explanation: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """
    Create explanation payload for CloudEvent.

    Args:
        best_rail: Selected rail information
        explanation: Explanation details
        context: Request context

    Returns:
        Payload dictionary
    """
    return {
        "rail_selection": {
            "rail_id": best_rail["rail_id"],
            "description": best_rail["description"],
            "score": best_rail["scores"]["total"],
        },
        "explanation": explanation,
        "context": {
            "amount": context["amount"],
            "urgency": context["urgency"],
            "vendor_id": context.get("vendor_id", "unknown"),
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "metadata": {
            "service": "orion",
            "version": "1.0.0",
            "feature": "payout_optimization",
        },
    }


def get_trace_id() -> str:
    """
    Generate or retrieve trace ID.

    Returns:
        Trace ID string
    """
    return str(uuid.uuid4())


def format_ce_for_logging(event: dict[str, Any]) -> str:
    """
    Format CloudEvent for logging purposes.

    Args:
        event: CloudEvent dictionary

    Returns:
        Formatted string for logging
    """
    return json.dumps(event, indent=2)
