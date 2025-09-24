"""
FastAPI application for Orion service.
"""

from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel

from orion.ce import create_explanation_payload, emit_explanation_ce, get_trace_id
from orion.explain import explain_choice
from orion.mcp import mcp_router
from orion.optimize import get_best_rail, score_rails, validate_context

# Create FastAPI application
app = FastAPI(
    title="Orion Service",
    description="Orion service for the Open Checkout Network (OCN)",
    version="0.1.0",
    contact={
        "name": "OCN Team",
        "email": "team@ocn.ai",
        "url": "https://github.com/ahsanazmi1/orion",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Include MCP router
app.include_router(mcp_router)


class OptimizationRequest(BaseModel):
    """Request model for payout optimization."""

    amount: float
    urgency: str = "normal"
    vendor_id: str = "unknown"
    currency: str = "USD"


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {"ok": True, "repo": "orion"}


@app.post("/optimize")
async def optimize_payout(
    request: OptimizationRequest,
    emit_ce: bool = Query(False, description="Emit CloudEvent for explanation"),
) -> dict[str, Any]:
    """
    Optimize payout rail selection with explanations.

    Args:
        request: Optimization request with amount, urgency, etc.
        emit_ce: Whether to emit CloudEvent for explanation

    Returns:
        Optimization result with best rail, rankings, and explanation
    """
    # Validate and normalize context
    context = validate_context(request.model_dump())

    # Score and rank rails
    ranked_rails = score_rails(context)

    if not ranked_rails:
        return {
            "error": "No suitable payment rails available for this amount",
            "context": context,
        }

    # Get best rail
    best_rail = get_best_rail(ranked_rails)

    # Generate explanation
    explanation = explain_choice(best_rail, ranked_rails, context)

    # Prepare response
    response = {
        "best": best_rail,
        "ranked": ranked_rails,
        "explanation": explanation,
        "context": context,
    }

    # Emit CloudEvent if requested
    if emit_ce:
        trace_id = get_trace_id()
        payload = create_explanation_payload(best_rail, explanation, context)
        ce_event = emit_explanation_ce(trace_id, payload)

        # Add CloudEvent to response
        response["cloud_event"] = ce_event
        response["trace_id"] = trace_id

    return response


def main() -> None:
    """Main entry point for running the application."""
    import uvicorn

    uvicorn.run(
        "orion.api:app",
        host="127.0.0.1",  # Use localhost for development security
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
