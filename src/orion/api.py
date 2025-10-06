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

# Import ML-enhanced optimizer
try:
    from orion.ml_enhanced_optimizer import get_ml_enhanced_optimizer
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML models not available: {e}")
    ML_AVAILABLE = False

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
    return {
        "ok": True, 
        "repo": "orion",
        "ml_enabled": ML_AVAILABLE
    }


@app.get("/ml/status")
async def get_ml_status() -> dict[str, Any]:
    """Get ML model status and configuration."""
    if not ML_AVAILABLE:
        return {
            "ml_enabled": False,
            "error": "ML models not available"
        }
    
    try:
        from orion.ml.route_optimization import get_route_optimizer
        from orion.ml.cost_prediction import get_cost_predictor
        
        route_optimizer = get_route_optimizer()
        cost_predictor = get_cost_predictor()
        
        return {
            "ml_enabled": True,
            "ml_weight": 0.7,  # From MLEnhancedOptimizer
            "models": {
                "route_optimization": {
                    "loaded": route_optimizer.is_loaded,
                    "model_type": route_optimizer.metadata.get("model_type", "unknown"),
                    "version": route_optimizer.metadata.get("version", "unknown"),
                    "training_date": route_optimizer.metadata.get("trained_on", "unknown"),
                    "features": len(route_optimizer.feature_names) if route_optimizer.feature_names else 0
                },
                "cost_prediction": {
                    "loaded": cost_predictor.is_loaded,
                    "model_type": cost_predictor.metadata.get("model_type", "unknown"),
                    "version": cost_predictor.metadata.get("version", "unknown"),
                    "training_date": cost_predictor.metadata.get("trained_on", "unknown"),
                    "features": len(cost_predictor.feature_names) if cost_predictor.feature_names else 0
                }
            }
        }
    except Exception as e:
        return {
            "ml_enabled": False,
            "error": f"Failed to get ML status: {str(e)}"
        }


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
    # Use ML-enhanced optimizer if available, otherwise fallback to traditional
    if ML_AVAILABLE:
        ml_optimizer = get_ml_enhanced_optimizer()
        response = ml_optimizer.optimize_payout(request.model_dump())
    else:
        # Traditional optimization logic
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

        if not best_rail:
            return {
                "error": "No suitable payment rails available for this amount",
                "context": context,
            }

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
        payload = create_explanation_payload(response["best"], response["explanation"], response["context"])
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
        host="0.0.0.0",  # Use 0.0.0.0 for Docker container access
        port=8081,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
