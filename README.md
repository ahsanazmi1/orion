# Orion Service

[![CI](https://github.com/ahsanazmi1/orion/workflows/CI/badge.svg)](https://github.com/ahsanazmi1/orion/actions/workflows/ci.yml)

Orion is a minimal Python service for the [Open Checkout Network (OCN)](https://github.com/ahsanazmi1/ocn-common). It provides core functionality and serves as a foundation for building more complex services within the OCN ecosystem. Orion follows modern Python development practices with FastAPI, comprehensive testing, and automated CI/CD workflows.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ahsanazmi1/orion.git
cd orion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

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
