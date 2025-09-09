# trade_system_modules

Shared Python package for trade systems: config, schemas, data adapters (Polygon, IBKR), storage (GCS), execution (IBKR), utilities, and templates for strategy repos.

## Quickstart

1. Install the package:
   ```
   pip install -e .[dev]
   ```

2. Set up environment variables from `.env.example`:
   ```
   cp .env.example .env
   # Edit .env with your values
   ```

3. Use the public API:
   ```python
   from trade_system_modules.config.settings import Settings
   from trade_system_modules.data.polygon_adapter import get_agg_minute
   from trade_system_modules.storage import GCSClient
   # etc.
   ```

See usage examples in docs or tests.

## Installation

### From Registry (Private PyPI or GitHub Packages)
```
pip install trade-system-modules==0.1.0 --index-url https://pypi.org/simple/  # or private index
```

### From Git URL (Pinned)
```
pip install "trade-system-modules @ git+https://github.com/wienbej01/trade_system_modules.git@v0.1.0"
```

## Versioning

We follow Semantic Versioning (SemVer): MAJOR.MINOR.PATCH.
- Public API frozen at v0.1.0.
- Breaking changes only on major versions.
- See [CHANGELOG.md] for details (to be added).

## Development

- Run tests: `pytest`
- Lint: `ruff check .`
- Type check: `mypy src`

## Environment

Copy `.env.example` to `.env` and fill in:
- GCS_BUCKET
- GCP_PROJECT
- POLYGON_API_KEY
- IB_HOST, IB_PORT, IB_CLIENT_ID

## Cookiecutter for New Strategies

Use the provided template:
```
cookiecutter cookiecutter-strategy/