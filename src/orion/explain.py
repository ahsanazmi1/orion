"""
Orion explanation module.

Provides human-readable explanations for payout optimization decisions.
"""

from typing import Any


def explain_choice(
    best: dict[str, Any], ranked: list[dict[str, Any]], context: dict[str, Any]
) -> dict[str, Any]:
    """
    Generate explanation for payout rail choice.

    Args:
        best: The selected rail
        ranked: All ranked rails
        context: Request context

    Returns:
        Explanation with reason, signals, mitigation, and confidence
    """
    if not best or not ranked:
        return {
            "reason": "No suitable payment rails available",
            "signals": ["insufficient_rails"],
            "mitigation": "Contact support for alternative payment methods",
            "confidence": 0.0,
        }

    reason = _generate_reason(best, ranked, context)
    signals = _identify_signals(best, ranked, context)
    mitigation = _suggest_mitigation(best, ranked, context)
    confidence = _calculate_confidence(best, ranked, context)

    return {
        "reason": reason,
        "signals": signals,
        "mitigation": mitigation,
        "confidence": confidence,
    }


def _generate_reason(
    best: dict[str, Any], ranked: list[dict[str, Any]], context: dict[str, Any]
) -> str:
    """Generate human-readable reason for the choice."""
    rail_name = best["description"]
    total_score = best["scores"]["total"]
    amount = context["amount"]
    urgency = context["urgency"]

    # Get score breakdown
    cost_score = best["scores"]["cost"]
    speed_score = best["scores"]["speed"]
    limits_score = best["scores"]["limits"]

    # Find the primary strength
    strengths = []
    if cost_score >= 80:
        strengths.append("cost-effective")
    if speed_score >= 80:
        strengths.append("fast processing")
    if limits_score >= 80:
        strengths.append("high capacity")

    primary_strength = strengths[0] if strengths else "balanced performance"

    # Generate contextual reason
    if urgency == "high":
        urgency_context = "Given the high urgency of this payment,"
    elif urgency == "low":
        urgency_context = "For this low-urgency payment,"
    else:
        urgency_context = "For this payment,"

    reason = (
        f"{urgency_context} {rail_name} was selected as the optimal payment rail "
        f"with a score of {total_score:.1f}/100. This rail is particularly strong in "
        f"{primary_strength} (cost: {cost_score:.1f}, speed: {speed_score:.1f}, "
        f"capacity: {limits_score:.1f}) and can handle the ${amount:,.2f} amount efficiently."
    )

    return reason


def _identify_signals(
    best: dict[str, Any], ranked: list[dict[str, Any]], context: dict[str, Any]
) -> list[str]:
    """Identify key signals that influenced the decision."""
    signals = []

    # Amount-based signals
    amount = context["amount"]
    if amount > 100000:
        signals.append("high_value_transaction")
    elif amount < 100:
        signals.append("low_value_transaction")

    # Urgency signals
    urgency = context["urgency"]
    if urgency == "high":
        signals.append("urgent_payment")
    elif urgency == "low":
        signals.append("non_urgent_payment")

    # Rail-specific signals
    rail_id = best["rail_id"]
    if rail_id == "wire":
        signals.append("premium_rail_selected")
    elif rail_id == "ach":
        signals.append("standard_rail_selected")
    elif rail_id == "rtp":
        signals.append("instant_rail_selected")
    elif rail_id == "v_card":
        signals.append("card_rail_selected")

    # Score-based signals
    if best["scores"]["total"] > 90:
        signals.append("excellent_rail_match")
    elif best["scores"]["total"] < 60:
        signals.append("suboptimal_rail_match")

    # Competition signals
    if len(ranked) > 1:
        second_best_score = ranked[1]["scores"]["total"]
        score_gap = best["scores"]["total"] - second_best_score
        if score_gap < 5:
            signals.append("close_competition")
        elif score_gap > 20:
            signals.append("clear_winner")

    return signals


def _suggest_mitigation(
    best: dict[str, Any], ranked: list[dict[str, Any]], context: dict[str, Any]
) -> str:
    """Suggest mitigation strategies if needed."""
    mitigations = []

    # Check for potential issues
    total_score = best["scores"]["total"]
    amount = context["amount"]

    if total_score < 70:
        mitigations.append("Consider splitting large payments across multiple rails")

    if "max_amount" in best and amount > best["max_amount"] * 0.8:
        mitigations.append("Monitor for potential capacity constraints")

    # Check if there are better alternatives for specific criteria
    if len(ranked) > 1:
        best_cost_rail = max(ranked, key=lambda r: r["scores"]["cost"])
        best_speed_rail = max(ranked, key=lambda r: r["scores"]["speed"])

        if best_cost_rail["rail_id"] != best["rail_id"]:
            cost_savings = best_cost_rail["scores"]["cost"] - best["scores"]["cost"]
            if cost_savings > 20:
                mitigations.append(
                    f"Consider {best_cost_rail['description']} for significant cost savings"
                )

        if best_speed_rail["rail_id"] != best["rail_id"]:
            speed_gain = best_speed_rail["scores"]["speed"] - best["scores"]["speed"]
            if speed_gain > 20:
                mitigations.append(
                    f"Consider {best_speed_rail['description']} for faster processing"
                )

    if not mitigations:
        return "No mitigation needed - optimal rail selected"

    return "; ".join(mitigations)


def _calculate_confidence(
    best: dict[str, Any], ranked: list[dict[str, Any]], context: dict[str, Any]
) -> float:
    """Calculate confidence score (0-1) for the decision."""
    base_confidence = 0.5

    # Score-based confidence
    total_score = best["scores"]["total"]
    score_confidence = min(0.4, (total_score - 50) / 125)  # 0.4 max from score

    # Competition-based confidence
    competition_confidence = 0.1
    if len(ranked) > 1:
        second_best_score = ranked[1]["scores"]["total"]
        score_gap = best["scores"]["total"] - second_best_score
        competition_confidence = min(0.3, score_gap / 50)  # 0.3 max from gap

    # Context-based confidence
    context_confidence = 0.1
    if context.get("urgency") == "normal" and 100 <= context["amount"] <= 10000:
        context_confidence = 0.1  # Optimal context

    total_confidence = (
        base_confidence + score_confidence + competition_confidence + context_confidence
    )

    return min(1.0, max(0.0, total_confidence))
