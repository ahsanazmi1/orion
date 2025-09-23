"""
MCP (Model Context Protocol) server for Orion service.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    Get available payout options and rails.

    Args:
        args: Optional arguments (currently unused)

    Returns:
        MCP response with payout options
    """
    payout_options = [
        {
            "rail": "ACH",
            "name": "Automated Clearing House",
            "description": "Direct bank transfer via ACH network",
            "processing_time": "1-3 business days",
            "fees": {"fixed": 0.25, "percentage": 0.0},
            "limits": {"min": 1.00, "max": 25000.00},
            "supported_countries": ["US"],
            "enabled": True,
        },
        {
            "rail": "Wire",
            "name": "Wire Transfer",
            "description": "International wire transfer",
            "processing_time": "1-2 business days",
            "fees": {"fixed": 15.00, "percentage": 0.0},
            "limits": {"min": 100.00, "max": 100000.00},
            "supported_countries": ["US", "CA", "GB", "DE", "FR"],
            "enabled": True,
        },
        {
            "rail": "Card",
            "name": "Card Payout",
            "description": "Direct to card payout",
            "processing_time": "Instant",
            "fees": {"fixed": 0.50, "percentage": 0.0},
            "limits": {"min": 1.00, "max": 5000.00},
            "supported_countries": ["US", "CA"],
            "enabled": True,
        },
    ]

    return MCPResponse(
        ok=True,
        data={
            "payout_options": payout_options,
            "total_options": len(payout_options),
            "default_rail": "ACH",
        },
        agent="orion",
    )
