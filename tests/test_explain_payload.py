"""
Tests for Orion explanation payload generation.
"""

from orion.explain import explain_choice


def test_explain_choice_has_required_fields() -> None:
    """Test that explanation contains all required fields."""
    best_rail = {
        "rail_id": "ach",
        "description": "ACH (Automated Clearing House)",
        "max_amount": 25000.0,
        "scores": {"cost": 85.0, "speed": 60.0, "limits": 70.0, "total": 75.0},
    }

    ranked_rails = [best_rail]
    context = {"amount": 1000.0, "urgency": "normal", "vendor_id": "test_vendor"}

    explanation = explain_choice(best_rail, ranked_rails, context)

    # Check required fields
    required_fields = ["reason", "signals", "mitigation", "confidence"]
    for field in required_fields:
        assert field in explanation
        assert explanation[field] is not None

    # Check field types
    assert isinstance(explanation["reason"], str)
    assert isinstance(explanation["signals"], list)
    assert isinstance(explanation["mitigation"], str)
    assert isinstance(explanation["confidence"], (int, float))

    # Check confidence range
    assert 0.0 <= explanation["confidence"] <= 1.0


def test_explain_choice_reason_generation() -> None:
    """Test that reason generation includes key information."""
    best_rail = {
        "rail_id": "rtp",
        "description": "Real-Time Payments",
        "max_amount": 100000.0,
        "scores": {"cost": 70.0, "speed": 95.0, "limits": 80.0, "total": 85.0},
    }

    ranked_rails = [best_rail]
    context = {"amount": 5000.0, "urgency": "high", "vendor_id": "test_vendor"}

    explanation = explain_choice(best_rail, ranked_rails, context)

    reason = explanation["reason"]

    # Should mention the rail name
    assert "Real-Time Payments" in reason

    # Should mention the score
    assert "85.0" in reason

    # Should mention the amount
    assert "$5,000.00" in reason

    # Should mention urgency context for high urgency
    assert "high urgency" in reason or "urgent" in reason


def test_explain_choice_signals_identification() -> None:
    """Test that signals are correctly identified."""
    best_rail = {
        "rail_id": "wire",
        "description": "Wire Transfer",
        "max_amount": 1000000.0,
        "scores": {"cost": 40.0, "speed": 80.0, "limits": 90.0, "total": 75.0},
    }

    ranked_rails = [best_rail]

    # Test high value transaction signal
    high_value_context = {
        "amount": 150000.0,
        "urgency": "normal",
        "vendor_id": "test_vendor",
    }

    explanation = explain_choice(best_rail, ranked_rails, high_value_context)
    assert "high_value_transaction" in explanation["signals"]

    # Test urgent payment signal
    urgent_context = {"amount": 1000.0, "urgency": "high", "vendor_id": "test_vendor"}

    explanation = explain_choice(best_rail, ranked_rails, urgent_context)
    assert "urgent_payment" in explanation["signals"]

    # Test rail-specific signals
    explanation = explain_choice(best_rail, ranked_rails, urgent_context)
    assert "premium_rail_selected" in explanation["signals"]


def test_explain_choice_mitigation_suggestions() -> None:
    """Test that mitigation suggestions are appropriate."""
    best_rail = {
        "rail_id": "ach",
        "description": "ACH (Automated Clearing House)",
        "scores": {
            "cost": 85.0,
            "speed": 60.0,
            "limits": 70.0,
            "total": 65.0,  # Low score
        },
        "max_amount": 25000.0,
    }

    ranked_rails = [best_rail]
    context = {
        "amount": 20000.0,  # High utilization
        "urgency": "normal",
        "vendor_id": "test_vendor",
    }

    explanation = explain_choice(best_rail, ranked_rails, context)

    mitigation = explanation["mitigation"]

    # Should suggest mitigation for low score
    assert "mitigation" in mitigation.lower() or "consider" in mitigation.lower()


def test_explain_choice_confidence_calculation() -> None:
    """Test that confidence is calculated appropriately."""
    # High confidence scenario
    best_rail = {
        "rail_id": "ach",
        "description": "ACH (Automated Clearing House)",
        "max_amount": 25000.0,
        "scores": {
            "cost": 90.0,
            "speed": 80.0,
            "limits": 85.0,
            "total": 85.0,  # High score
        },
    }

    ranked_rails = [best_rail]
    context = {"amount": 1000.0, "urgency": "normal", "vendor_id": "test_vendor"}

    explanation = explain_choice(best_rail, ranked_rails, context)
    high_confidence = explanation["confidence"]

    # Low confidence scenario
    low_score_rail = {
        "rail_id": "v_card",
        "description": "Virtual Card",
        "scores": {
            "cost": 40.0,
            "speed": 50.0,
            "limits": 60.0,
            "total": 50.0,  # Low score
        },
    }

    explanation = explain_choice(low_score_rail, ranked_rails, context)
    low_confidence = explanation["confidence"]

    # High confidence should be greater than low confidence
    assert high_confidence > low_confidence

    # Both should be in valid range
    assert 0.0 <= high_confidence <= 1.0
    assert 0.0 <= low_confidence <= 1.0


def test_explain_choice_empty_inputs() -> None:
    """Test explanation with empty or None inputs."""
    # Test with None best rail
    explanation = explain_choice(None, [], {})

    assert explanation["reason"] == "No suitable payment rails available"
    assert "insufficient_rails" in explanation["signals"]
    assert explanation["confidence"] == 0.0

    # Test with empty ranked list
    best_rail = {"rail_id": "ach", "description": "ACH", "scores": {"total": 80.0}}

    explanation = explain_choice(best_rail, [], {})

    assert explanation["reason"] == "No suitable payment rails available"
    assert explanation["confidence"] == 0.0


def test_explain_choice_competition_signals() -> None:
    """Test signals for close competition vs clear winner."""
    best_rail = {
        "rail_id": "ach",
        "description": "ACH",
        "max_amount": 25000.0,
        "scores": {"cost": 80.0, "speed": 70.0, "limits": 75.0, "total": 75.0},
    }

    second_rail = {
        "rail_id": "rtp",
        "description": "RTP",
        "max_amount": 100000.0,
        "scores": {
            "cost": 70.0,
            "speed": 80.0,
            "limits": 70.0,
            "total": 73.0,  # Close score
        },
    }

    ranked_rails = [best_rail, second_rail]
    context = {"amount": 1000.0, "urgency": "normal", "vendor_id": "test"}

    explanation = explain_choice(best_rail, ranked_rails, context)

    # Should detect close competition
    assert "close_competition" in explanation["signals"]

    # Test clear winner scenario
    clear_winner_rail = {
        "rail_id": "ach",
        "description": "ACH",
        "max_amount": 25000.0,
        "scores": {
            "cost": 90.0,
            "speed": 85.0,
            "limits": 90.0,
            "total": 95.0,  # Much higher score (gap of 22)
        },
    }

    ranked_rails = [clear_winner_rail, second_rail]
    explanation = explain_choice(clear_winner_rail, ranked_rails, context)

    # Should detect clear winner
    assert "clear_winner" in explanation["signals"]
