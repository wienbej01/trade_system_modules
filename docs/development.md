# Development Guide

This guide covers development workflows, coding standards, testing practices, and contribution guidelines for the `trade_system_modules` package.

## Development Environment Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# Git
git --version

# Virtual environment tools
python -m pip install --user virtualenv
```

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/wienbej01/trade_system_modules.git
cd trade_system_modules

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode with all tools
pip install -e .[dev]

# Verify installation
python -c "import trade_system_modules; print('✓ Import successful')"
pytest --version
ruff --version
mypy --version
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll.ruff": "explicit"
    }
}
```

#### PyCharm

1. Open project
2. Set Python interpreter to `.venv/bin/python`
3. Enable Ruff integration
4. Configure pytest as test runner

## Code Quality Standards

### Linting and Formatting

```bash
# Run all quality checks
make quality  # If available, or manually:

# Format code
ruff format .

# Lint code
ruff check . --fix

# Type checking
mypy src/

# Security checks
bandit -r src/
```

### Pre-commit Hooks

Setup pre-commit for automatic quality checks:

```bash
pip install pre-commit
pre-commit install

# Manual run
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-r, src/]
```

## Testing Practices

### Test Organization

```
tests/
├── unit/              # Unit tests
│   ├── test_gcs_unit.py
│   └── test_polygon_adapter_unit.py
├── integration/       # Integration tests
├── fixtures/          # Test data and fixtures
└── conftest.py        # Test configuration
```

### Writing Tests

```python
import pytest
from trade_system_modules import get_agg_minute

class TestPolygonAdapter:
    """Test cases for Polygon data adapter."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample market data."""
        return pd.DataFrame({
            "ts": pd.date_range("2023-01-01", periods=100, freq="1min"),
            "open": [100 + i*0.01 for i in range(100)],
            "close": [100.5 + i*0.01 for i in range(100)],
        })

    def test_get_agg_minute_success(self, mocker):
        """Test successful data retrieval."""
        # Arrange
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "results": [
                {"t": 1672531200000, "o": 100.0, "c": 100.5, "h": 101.0, "l": 99.0, "v": 1000, "n": 10}
            ]
        }
        mocker.patch("requests.get", return_value=mock_response)

        # Act
        result = get_agg_minute("AAPL", "2023-01-01", "2023-01-01")

        # Assert
        assert not result.empty
        assert len(result) == 1
        assert list(result.columns) == ["ts", "open", "high", "low", "close", "volume", "trades"]

    def test_get_agg_minute_empty_results(self, mocker):
        """Test handling of empty results."""
        # Arrange
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"results": []}
        mocker.patch("requests.get", return_value=mock_response)

        # Act
        result = get_agg_minute("INVALID", "2023-01-01", "2023-01-01")

        # Assert
        assert result.empty
```

### Test Coverage

```bash
# Run tests with coverage
pytest --cov=src/trade_system_modules --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html  # Opens in browser

# Coverage requirements
# - Minimum 80% coverage
# - All public APIs tested
# - Error conditions covered
```

### Test Data Management

```python
# tests/conftest.py
import pytest
import pandas as pd
from trade_system_modules import Settings

@pytest.fixture(scope="session")
def test_settings():
    """Test configuration."""
    return Settings(
        gcs_bucket="test-bucket",
        polygon_api_key="test_key"
    )

@pytest.fixture
def sample_market_data():
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=100, freq="1min", tz="UTC"),
        "open": [100 + i*0.01 for i in range(100)],
        "high": [101 + i*0.01 for i in range(100)],
        "low": [99 + i*0.01 for i in range(100)],
        "close": [100.5 + i*0.01 for i in range(100)],
        "volume": [1000 + i*10 for i in range(100)],
        "trades": [50 + i for i in range(100)]
    })
```

## Development Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/new-data-adapter

# Make changes
# ... development work ...

# Run quality checks
make quality

# Run tests
pytest

# Commit changes
git add .
git commit -m "feat: add new data adapter

- Add support for new data source
- Include comprehensive tests
- Update documentation"

# Push branch
git push origin feature/new-data-adapter

# Create pull request
# ... GitHub PR process ...
```

### Commit Message Convention

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Examples:
```
feat(polygon): add minute-level data support
fix(gcs): handle connection timeouts
docs(api): update parameter descriptions
test(storage): add integration tests
```

## Code Review Process

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No breaking changes
```

### Review Criteria

- **Functionality**: Code works as intended
- **Testing**: Adequate test coverage
- **Documentation**: Code and API docs updated
- **Style**: Follows project conventions
- **Performance**: No performance regressions
- **Security**: No security vulnerabilities

## Architecture Guidelines

### Module Structure

```
src/trade_system_modules/
├── __init__.py          # Public API exports
├── config/              # Configuration management
├── data/                # Data adapters
├── storage/             # Cloud storage
├── execution/           # Trading execution
├── schemas/             # Data validation
├── utils/               # Common utilities
└── types.py             # Type definitions
```

### Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: Dependencies passed explicitly
3. **Error Handling**: Comprehensive error handling with custom exceptions
4. **Type Safety**: Full type hints throughout
5. **Testability**: Code designed for easy testing
6. **Documentation**: All public APIs documented

### Interface Design

```python
from abc import ABC, abstractmethod
from typing import Protocol

class DataAdapter(Protocol):
    """Protocol for data adapters."""

    @abstractmethod
    def get_historical_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Get historical market data."""
        ...

class StorageAdapter(ABC):
    """Abstract base class for storage adapters."""

    @abstractmethod
    def save(self, data: pd.DataFrame, path: str) -> None:
        """Save data to storage."""
        ...

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load data from storage."""
        ...
```

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def profile_function():
    """Profile a function's performance."""
    profiler = cProfile.Profile()
    profiler.enable()

    # Code to profile
    data = get_agg_minute("AAPL", "2023-01-01", "2023-01-31")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

profile_function()
```

### Memory Optimization

```python
import psutil
import os

def memory_usage():
    """Monitor memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def optimize_dataframe(df):
    """Optimize DataFrame memory usage."""
    # Downcast numeric types
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert object columns to category if appropriate
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
            df[col] = df[col].astype('category')

    return df
```

## Security Considerations

### API Key Management

```python
# Never commit secrets
# Use environment variables or secure vaults

import os
from pathlib import Path

def get_api_key():
    """Securely retrieve API key."""
    # Check environment first
    api_key = os.environ.get('POLYGON_API_KEY')
    if api_key:
        return api_key

    # Check secure file
    key_file = Path.home() / '.trading' / 'polygon_key'
    if key_file.exists():
        return key_file.read_text().strip()

    raise ValueError("API key not found")
```

### Input Validation

```python
from pydantic import BaseModel, validator
import re

class TradeRequest(BaseModel):
    symbol: str
    quantity: int
    price: float

    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(r'^[A-Z]{1,5}$', v):
            raise ValueError('Invalid symbol format')
        return v.upper()

    @validator('quantity')
    def validate_quantity(cls, v):
        if not 1 <= v <= 1000000:
            raise ValueError('Quantity must be between 1 and 1,000,000')
        return v

    @validator('price')
    def validate_price(cls, v):
        if not 0.01 <= v <= 10000.0:
            raise ValueError('Price must be between $0.01 and $10,000')
        return round(v, 2)
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Run tests
      run: pytest --cov=src/trade_system_modules --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -e .[dev]

    - name: Run quality checks
      run: |
        ruff check .
        ruff format --check .
        mypy src/
```

## Release Process

### Version Management

```bash
# Update version in pyproject.toml
# Update version in src/trade_system_modules/__init__.py

# Create git tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag
git push origin v0.2.0

# Build and publish
python -m build
twine upload dist/*
```

### Changelog Management

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New data adapter for XYZ source
- Performance improvements for large datasets

### Fixed
- Memory leak in GCS client
- Timezone handling in polygon adapter

### Changed
- Updated dependencies
- Improved error messages

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Polygon.io data adapter
- Google Cloud Storage integration
- Interactive Brokers execution
- Basic testing framework
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Maintain professional standards

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/org/trade_system_modules/issues)
- **Discussions**: [GitHub Discussions](https://github.com/org/trade_system_modules/discussions)
- **Documentation**: [Full Docs](https://github.com/org/trade_system_modules/tree/main/docs)
- **Email**: engineering@hedgefund.com

## See Also

- [Installation Guide](installation.md)
- [API Reference](api-reference.md)
- [Testing Guide](testing.md)
- [Troubleshooting](troubleshooting.md)