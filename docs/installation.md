# Installation Guide

This guide covers all methods for installing and setting up the `trade_system_modules` package.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

## Quick Installation

### For Development
```bash
# Clone the repository
git clone https://github.com/wienbej01/trade_system_modules.git
cd trade_system_modules

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

### For Production Use
```bash
# Install from PyPI
pip install trade-system-modules==0.1.0

# Or install from Git (pinned version)
pip install "trade-system-modules @ git+https://github.com/wienbej01/trade_system_modules.git@v0.1.0"
```

## Installation Methods

### 1. Editable Development Installation

Best for development and testing:

```bash
pip install -e .
```

This creates a symlink to your source code, allowing live edits without reinstallation.

### 2. Production Installation

For production deployments:

```bash
# From PyPI
pip install trade-system-modules

# From Git with specific version
pip install "trade-system-modules @ git+https://github.com/wienbej01/trade_system_modules.git@v0.1.0"

# From local package
pip install .
```

### 3. Development Installation with Extras

For development with all tools:

```bash
pip install -e .[dev]
```

This installs:
- Core dependencies
- pytest (testing)
- ruff (linting)
- mypy (type checking)
- types-requests (type stubs)

## Virtual Environment Setup

### Using venv (Recommended)

```bash
# Create environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install package
pip install -e .[dev]
```

### Using conda

```bash
# Create environment
conda create -n trade-system python=3.11
conda activate trade-system

# Install package
pip install -e .[dev]
```

## Dependency Management

### Core Dependencies

The package requires these core dependencies:

```python
# From pyproject.toml
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings",
    "pandas>=2.0",
    "pyarrow>=12.0",
    "google-cloud-storage>=2.10",
    "ib-insync>=0.9.70",
    "tenacity>=8.2",
    "python-dotenv>=1.0",
    "requests",
]
```

### Optional Dependencies

Development dependencies:

```python
dev = [
    "pytest>=7.4",
    "ruff>=0.1",
    "mypy>=1.5",
    "types-requests>=2.31",
]
```

## Platform-Specific Notes

### Linux/macOS

No special requirements. Standard Python installation works.

### Windows

Ensure you have:
- Python 3.10+ installed
- Microsoft Visual C++ Build Tools (for some dependencies)
- Use PowerShell or Command Prompt with proper permissions

### Docker

For containerized deployments:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "your_script.py"]
```

## Post-Installation Setup

### 1. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Required for data adapters
POLYGON_API_KEY=your_polygon_api_key

# Required for Google Cloud Storage
GCS_BUCKET=your_gcs_bucket
GCP_PROJECT=your_gcp_project

# Required for Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

### 2. Google Cloud Setup (Optional)

If using GCS storage:

```bash
# Install gcloud CLI
# Follow: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project your-project-id
```

### 3. Interactive Brokers Setup (Optional)

If using IBKR execution:

1. Install Trader Workstation (TWS) or IB Gateway
2. Configure API settings in TWS
3. Start TWS/IB Gateway before running your application

## Verification

Test your installation:

```bash
# Run tests
pytest

# Check import
python -c "from trade_system_modules import Settings; print('Installation successful!')"

# Check version
python -c "import trade_system_modules; print(trade_system_modules.__version__)"
```

## Troubleshooting

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'trade_system_modules'`
- Ensure you're in the correct virtual environment
- Try reinstalling: `pip install -e .`

**Permission Error**: When installing packages
- Use `pip install --user` or run as administrator
- Check virtual environment activation

**Google Cloud Authentication Error**
- Run `gcloud auth application-default login`
- Verify `GOOGLE_APPLICATION_CREDENTIALS` environment variable

**Interactive Brokers Connection Error**
- Ensure TWS/IB Gateway is running
- Check firewall settings
- Verify host/port configuration

### Getting Help

- Check the [troubleshooting guide](troubleshooting.md)
- Review [GitHub Issues](https://github.com/org/trade_system_modules/issues)
- Contact: engineering@hedgefund.com

## Next Steps

After installation:
1. Review the [configuration guide](configuration.md)
2. Check out [usage examples](examples/)
3. Read the [API reference](api-reference.md)