"""
FastAPI application for Orion service.
"""

from typing import Any

from fastapi import FastAPI

from orion.mcp import mcp_router

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


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {"ok": True, "repo": "orion"}


def main() -> None:
    """Main entry point for running the application."""
    import uvicorn

    uvicorn.run(
        "orion.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
