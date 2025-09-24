"""
MCP (Model Context Protocol) server for Orion service.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from orion.explain import explain_choice
from orion.optimize import get_best_rail, score_rails, validate_context

# Create MCP router
mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPRequest(BaseModel):
    """MCP request model."""

    verb: str
    args: dict[str, Any] = {}


class MCPResponse(BaseModel):
    """MCP response model."""

    ok: bool
    data: Any
    agent: str


@mcp_router.post("/invoke")
async def mcp_invoke(request: MCPRequest) -> MCPResponse:
    """
    MCP protocol endpoint for Orion service operations.

    Args:
        request: MCP request containing verb and arguments

    Returns:
        MCP response with operation result

    Raises:
        HTTPException: If verb is not supported
    """
    verb = request.verb
    args = request.args

    if verb == "getStatus":
        return await get_status()
    elif verb == "getPayoutOptions":
        return await get_payout_options(args)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported verb: {verb}. Supported verbs: getStatus, getPayoutOptions",
        )


async def get_status() -> MCPResponse:
    """
    Get the current status of the Orion agent.

    Returns:
        MCP response with agent status
    """
    return MCPResponse(
        ok=True,
        data={"status": "active", "service": "payout-management"},
        agent="orion",
    )


async def get_payout_options(args: dict[str, Any]) -> MCPResponse:
    """
    Get optimized payout options using scoring and explanations.

    Args:
        args: Arguments containing amount, urgency, vendor_id, etc.

    Returns:
        MCP response with optimized payout options and explanation
    """
    # Extract context from args
    context = {
        "amount": args.get("amount", 1000.0),
        "urgency": args.get("urgency", "normal"),
        "vendor_id": args.get("vendor_id", "unknown"),
        "currency": args.get("currency", "USD"),
    }

    # Validate context
    context = validate_context(context)

    # Score and rank rails
    ranked_rails = score_rails(context)

    if not ranked_rails:
        return MCPResponse(
            ok=False,
            data={"error": "No suitable payment rails available for this amount"},
            agent="orion",
        )

    # Get best rail
    best_rail = get_best_rail(ranked_rails)

    if not best_rail:
        return MCPResponse(
            ok=False,
            data={"error": "No suitable payment rails available for this amount"},
            agent="orion",
        )

    # Generate explanation
    explanation = explain_choice(best_rail, ranked_rails, context)

    # Format payout options for MCP response
    payout_options = []
    for rail in ranked_rails:
        payout_options.append(
            {
                "rail": rail["rail_id"].upper(),
                "name": rail["description"],
                "description": f"{rail['description']} with {rail['processing_time_hours']}h processing",
                "processing_time": f"{rail['processing_time_hours']} hours",
                "fees": {"fixed": rail["cost_per_transaction"], "percentage": 0.0},
                "limits": {"min": 1.00, "max": rail["max_amount"]},
                "score": rail["scores"]["total"],
                "enabled": True,
            }
        )

    return MCPResponse(
        ok=True,
        data={
            "payout_options": payout_options,
            "best_rail": {
                "rail": best_rail["rail_id"].upper(),
                "score": best_rail["scores"]["total"],
                "reason": explanation["reason"],
            },
            "explanation": explanation,
            "context": context,
            "total_options": len(payout_options),
        },
        agent="orion",
    )
