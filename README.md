# Orion Service

[![CI](https://github.com/ahsanazmi1/orion/workflows/CI/badge.svg)](https://github.com/ahsanazmi1/orion/actions/workflows/ci.yml)
[![Contracts](https://github.com/ahsanazmi1/orion/workflows/Contracts/badge.svg)](https://github.com/ahsanazmi1/orion/actions/workflows/contracts.yml)
[![Security](https://github.com/ahsanazmi1/orion/workflows/Security/badge.svg)](https://github.com/ahsanazmi1/orion/actions/workflows/security.yml)

**Orion** is the **Vendor Payout service** for the [Open Checkout Network (OCN)](https://github.com/ahsanazmi1/ocn-common).

## Phase 2 — Explainability

🚧 **Currently in development** - Phase 2 focuses on AI-powered explainability and human-readable vendor payout decision reasoning.

- **Status**: Active development on `phase-2-explainability` branch
- **Features**: LLM integration, explainability API endpoints, decision audit trails
- **Issue Tracker**: [Phase 2 Issues](https://github.com/ahsanazmi1/orion/issues?q=is%3Aopen+is%3Aissue+label%3Aphase-2)
- **Timeline**: Weeks 4-8 of OCN development roadmap

Orion provides intelligent vendor payout management and payment rail optimization for the OCN ecosystem. Unlike traditional black-box payout systems, Orion offers:

## Quickstart (≤ 60s)

Get up and running with Orion OCN Agent in under a minute:

```bash
# Clone the repository
git clone https://github.com/ahsanazmi1/orion.git
cd orion

# Setup everything (venv, deps, pre-commit hooks)
make setup

# Run tests to verify everything works
make test

# Start the service
make run
```

**That's it!** 🎉

The service will be running at `http://localhost:8000`. Test the endpoints:

```bash
# Health check
curl http://localhost:8000/health

# MCP getStatus
curl -X POST http://localhost:8000/mcp/invoke \
  -H "Content-Type: application/json" \
  -d '{"verb": "getStatus", "args": {}}'

# MCP getPayoutOptions
curl -X POST http://localhost:8000/mcp/invoke \
  -H "Content-Type: application/json" \
  -d '{"verb": "getPayoutOptions", "args": {}}'
```

### Additional Makefile Targets

```bash
make lint        # Run code quality checks
make fmt         # Format code with black/ruff
make clean       # Remove virtual environment and cache
make help        # Show all available targets
```

## Manual Setup (Alternative)

If you prefer manual setup over the Makefile:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest -q

# Start the service
uvicorn orion.api:app --reload
```

## API Endpoints

- `GET /health` - Health check endpoint

## Development

This project uses:
- **FastAPI** for the web framework
- **pytest** for testing
- **ruff** and **black** for code formatting
- **mypy** for type checking

## License

MIT License - see [LICENSE](LICENSE) file for details.
