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
- `POST /optimize` - Optimize payout rail selection with explanations

### Payout Optimization Example

**Request:**
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000.0,
    "urgency": "high",
    "vendor_id": "vendor_123",
    "currency": "USD"
  }'
```

**Response:**
```json
{
  "best": {
    "rail_id": "rtp",
    "description": "Real-Time Payments",
    "cost_per_transaction": 0.5,
    "processing_time_hours": 0.5,
    "max_amount": 100000.0,
    "success_rate": 0.97,
    "scores": {
      "cost": 89.5,
      "speed": 99.0,
      "limits": 95.0,
      "total": 94.7
    },
    "context": {
      "amount": 5000.0,
      "urgency": "high",
      "vendor_id": "vendor_123"
    }
  },
  "ranked": [
    {
      "rail_id": "rtp",
      "description": "Real-Time Payments",
      "scores": {"total": 94.7}
    },
    {
      "rail_id": "ach",
      "description": "ACH (Automated Clearing House)",
      "scores": {"total": 78.2}
    }
  ],
  "explanation": {
    "reason": "Given the high urgency of this payment, Real-Time Payments was selected as the optimal payment rail with a score of 94.7/100. This rail is particularly strong in fast processing (cost: 89.5, speed: 99.0, capacity: 95.0) and can handle the $5,000.00 amount efficiently.",
    "signals": ["urgent_payment", "instant_rail_selected", "clear_winner"],
    "mitigation": "No mitigation needed - optimal rail selected",
    "confidence": 0.95
  },
  "context": {
    "amount": 5000.0,
    "urgency": "high",
    "vendor_id": "vendor_123",
    "currency": "USD"
  }
}
```

### CloudEvent Emission

To emit a CloudEvent for the explanation, add `?emit_ce=true` to the request:

```bash
curl -X POST "http://localhost:8000/optimize?emit_ce=true" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000.0,
    "urgency": "high",
    "vendor_id": "vendor_123"
  }'
```

**CloudEvent Envelope:**
```json
{
  "specversion": "1.0",
  "type": "ocn.orion.explanation.v1",
  "source": "orion",
  "id": "12345678-1234-5678-9012-123456789012",
  "time": "2024-01-15T10:30:00Z",
  "subject": "trace-abc-123",
  "datacontenttype": "application/json",
  "data": {
    "rail_selection": {
      "rail_id": "rtp",
      "description": "Real-Time Payments",
      "score": 94.7
    },
    "explanation": {
      "reason": "Given the high urgency of this payment, Real-Time Payments was selected...",
      "signals": ["urgent_payment", "instant_rail_selected", "clear_winner"],
      "mitigation": "No mitigation needed - optimal rail selected",
      "confidence": 0.95
    },
    "context": {
      "amount": 5000.0,
      "urgency": "high",
      "vendor_id": "vendor_123"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {
      "service": "orion",
      "version": "1.0.0",
      "feature": "payout_optimization"
    }
  }
}
```

## Development

This project uses:
- **FastAPI** for the web framework
- **pytest** for testing
- **ruff** and **black** for code formatting
- **mypy** for type checking

## License

MIT License - see [LICENSE](LICENSE) file for details.
