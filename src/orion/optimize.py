"""
Orion payout optimization module.

Provides deterministic scoring and ranking of payment rails for vendor payouts.
"""

from typing import Any

# Rail configurations with deterministic properties
RAIL_CONFIGS = {
    "ach": {
        "cost_per_transaction": 0.25,
        "processing_time_hours": 24,
        "max_amount": 25000.0,
        "success_rate": 0.98,
        "description": "ACH (Automated Clearing House)",
    },
    "wire": {
        "cost_per_transaction": 15.0,
        "processing_time_hours": 4,
        "max_amount": 1000000.0,
        "success_rate": 0.99,
        "description": "Wire Transfer",
    },
    "rtp": {
        "cost_per_transaction": 0.50,
        "processing_time_hours": 0.5,
        "max_amount": 100000.0,
        "success_rate": 0.97,
        "description": "Real-Time Payments",
    },
    "v_card": {
        "cost_per_transaction": 2.0,
        "processing_time_hours": 1,
        "max_amount": 5000.0,
        "success_rate": 0.95,
        "description": "Virtual Card",
    },
}

# Scoring weights
COST_WEIGHT = 0.5
SPEED_WEIGHT = 0.3
LIMITS_WEIGHT = 0.2


def score_rails(context: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Score payment rails based on context and return ranked list.

    Args:
        context: Context containing amount, urgency, vendor_id, etc.

    Returns:
        List of rails with scores, ranked by best score first
    """
    amount = context.get("amount", 0.0)
    urgency = context.get("urgency", "normal")  # low, normal, high

    scored_rails = []

    for rail_id, config in RAIL_CONFIGS.items():
        # Check if rail exceeds limits
        if amount > config["max_amount"]:
            continue

        # Calculate scores (higher is better)
        cost_score = _calculate_cost_score(config["cost_per_transaction"], amount)
        speed_score = _calculate_speed_score(config["processing_time_hours"], urgency)
        limits_score = _calculate_limits_score(amount, config["max_amount"])

        # Weighted total score
        total_score = (
            cost_score * COST_WEIGHT
            + speed_score * SPEED_WEIGHT
            + limits_score * LIMITS_WEIGHT
        )

        rail_result = {
            "rail_id": rail_id,
            "description": config["description"],
            "cost_per_transaction": config["cost_per_transaction"],
            "processing_time_hours": config["processing_time_hours"],
            "max_amount": config["max_amount"],
            "success_rate": config["success_rate"],
            "scores": {
                "cost": cost_score,
                "speed": speed_score,
                "limits": limits_score,
                "total": round(total_score, 3),
            },
            "context": {"amount": amount, "urgency": urgency},
        }

        scored_rails.append(rail_result)

    # Sort by total score (descending) - deterministic ordering
    scored_rails.sort(key=lambda x: x["scores"]["total"], reverse=True)

    return scored_rails


def _calculate_cost_score(cost_per_transaction: float, amount: float) -> float:
    """Calculate cost score (lower cost = higher score)."""
    # Normalize cost as percentage of amount, then invert
    cost_percentage = (cost_per_transaction / amount) * 100 if amount > 0 else 0
    # Use exponential decay for cost score
    return max(0.0, 100.0 * (0.5 ** (cost_percentage / 1.0)))


def _calculate_speed_score(processing_hours: float, urgency: str) -> float:
    """Calculate speed score based on processing time and urgency."""
    # Base speed score (faster = higher score)
    base_score = max(0.0, 100.0 - (processing_hours * 2))

    # Urgency multiplier
    urgency_multipliers = {"low": 0.8, "normal": 1.0, "high": 1.5}

    multiplier = urgency_multipliers.get(urgency, 1.0)
    return min(100.0, base_score * multiplier)


def _calculate_limits_score(amount: float, max_amount: float) -> float:
    """Calculate limits score based on amount vs max amount."""
    if amount > max_amount:
        return 0.0

    # Higher score for rails that can handle larger amounts
    utilization = amount / max_amount
    return 100.0 * (1.0 - utilization * 0.5)  # Slight penalty for high utilization


def get_best_rail(scored_rails: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Get the best rail from scored list.

    Args:
        scored_rails: List of scored rails (should be pre-sorted)

    Returns:
        Best rail or None if no rails available
    """
    return scored_rails[0] if scored_rails else None


def validate_context(context: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize context.

    Args:
        context: Input context

    Returns:
        Validated context with defaults
    """
    validated = {
        "amount": float(context.get("amount", 0.0)),
        "urgency": context.get("urgency", "normal"),
        "vendor_id": context.get("vendor_id", "unknown"),
        "currency": context.get("currency", "USD"),
    }

    # Validate urgency
    if validated["urgency"] not in ["low", "normal", "high"]:
        validated["urgency"] = "normal"

    # Ensure positive amount
    validated["amount"] = max(0.0, validated["amount"])

    return validated
