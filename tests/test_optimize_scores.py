"""
Tests for Orion payout optimization scoring.
"""

from orion.optimize import get_best_rail, score_rails, validate_context


def test_score_rails_deterministic_ranking() -> None:
    """Test that rail scoring produces deterministic rankings."""
    context = {"amount": 1000.0, "urgency": "normal", "vendor_id": "test_vendor"}

    # Run scoring multiple times
    results = []
    for _ in range(3):
        scored_rails = score_rails(context)
        results.append([rail["rail_id"] for rail in scored_rails])

    # All results should be identical
    assert all(result == results[0] for result in results)

    # Should have all 4 rails (none exceed limits for $1000)
    assert len(results[0]) == 4

    # Rails should be ranked by score (descending)
    scored_rails = score_rails(context)
    scores = [rail["scores"]["total"] for rail in scored_rails]
    assert scores == sorted(scores, reverse=True)


def test_score_rails_amount_limits() -> None:
    """Test that rails exceeding amount limits are rejected."""
    # Test with amount exceeding all rail limits
    context = {
        "amount": 2000000.0,  # Exceeds all max amounts
        "urgency": "normal",
    }

    scored_rails = score_rails(context)

    # Should return empty list (no rails can handle this amount)
    assert len(scored_rails) == 0


def test_score_rails_urgency_impact() -> None:
    """Test that urgency affects speed scoring."""
    normal_context = {"amount": 1000.0, "urgency": "normal"}
    high_urgency_context = {"amount": 1000.0, "urgency": "high"}

    normal_rails = score_rails(normal_context)
    high_urgency_rails = score_rails(high_urgency_context)

    # Should have same rails
    assert len(normal_rails) == len(high_urgency_rails)

    # High urgency should favor faster rails
    # If RTP is available, it should score higher with high urgency
    rtp_normal = next((r for r in normal_rails if r["rail_id"] == "rtp"), None)
    rtp_high = next((r for r in high_urgency_rails if r["rail_id"] == "rtp"), None)

    if rtp_normal and rtp_high:
        assert rtp_high["scores"]["speed"] > rtp_normal["scores"]["speed"]


def test_score_rails_cost_scoring() -> None:
    """Test that cost scoring works correctly."""
    context = {"amount": 1000.0, "urgency": "normal"}

    scored_rails = score_rails(context)

    # ACH should have highest cost score (lowest cost)
    ach_rail = next((r for r in scored_rails if r["rail_id"] == "ach"), None)
    wire_rail = next((r for r in scored_rails if r["rail_id"] == "wire"), None)

    assert ach_rail is not None
    assert wire_rail is not None
    assert ach_rail["scores"]["cost"] > wire_rail["scores"]["cost"]


def test_get_best_rail() -> None:
    """Test getting the best rail from scored list."""
    context = {"amount": 1000.0, "urgency": "normal"}

    scored_rails = score_rails(context)
    best_rail = get_best_rail(scored_rails)

    assert best_rail is not None
    assert best_rail == scored_rails[0]  # Best should be first in sorted list

    # Test with empty list
    empty_rails = []
    assert get_best_rail(empty_rails) is None


def test_validate_context() -> None:
    """Test context validation and normalization."""
    # Test with valid context
    context = {
        "amount": 1000.0,
        "urgency": "high",
        "vendor_id": "test_vendor",
        "currency": "USD",
    }

    validated = validate_context(context)

    assert validated["amount"] == 1000.0
    assert validated["urgency"] == "high"
    assert validated["vendor_id"] == "test_vendor"
    assert validated["currency"] == "USD"

    # Test with defaults
    minimal_context = {}
    validated_minimal = validate_context(minimal_context)

    assert validated_minimal["amount"] == 0.0
    assert validated_minimal["urgency"] == "normal"
    assert validated_minimal["vendor_id"] == "unknown"
    assert validated_minimal["currency"] == "USD"

    # Test with invalid urgency
    invalid_context = {"urgency": "invalid"}
    validated_invalid = validate_context(invalid_context)

    assert validated_invalid["urgency"] == "normal"  # Should default to normal

    # Test with negative amount
    negative_context = {"amount": -100.0}
    validated_negative = validate_context(negative_context)

    assert validated_negative["amount"] == 0.0  # Should be normalized to 0


def test_rail_configurations() -> None:
    """Test that rail configurations are correct."""
    from orion.optimize import RAIL_CONFIGS

    # Test that all expected rails are present
    expected_rails = {"ach", "wire", "rtp", "v_card"}
    assert set(RAIL_CONFIGS.keys()) == expected_rails

    # Test that each rail has required properties
    for _rail_id, config in RAIL_CONFIGS.items():
        required_fields = [
            "cost_per_transaction",
            "processing_time_hours",
            "max_amount",
            "success_rate",
            "description",
        ]
        for field in required_fields:
            assert field in config
            assert config[field] is not None

        # Test that values are reasonable
        assert config["cost_per_transaction"] >= 0
        assert config["processing_time_hours"] >= 0
        assert config["max_amount"] > 0
        assert 0 <= config["success_rate"] <= 1
        assert isinstance(config["description"], str)


def test_scoring_weights() -> None:
    """Test that scoring weights sum to 1.0."""
    from orion.optimize import COST_WEIGHT, LIMITS_WEIGHT, SPEED_WEIGHT

    total_weight = COST_WEIGHT + SPEED_WEIGHT + LIMITS_WEIGHT
    assert abs(total_weight - 1.0) < 0.001  # Allow for floating point precision
