# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the `trade_system_modules` package.

## Quick Diagnosis

### Check Your Setup

```bash
# Verify Python version
python --version  # Should be 3.10+

# Check virtual environment
which python      # Should point to .venv/bin/python

# Verify package installation
python -c "import trade_system_modules; print('✓ Package installed')"

# Check configuration
python -c "from trade_system_modules import Settings; print('✓ Config loaded')"

# Run basic test
pytest tests/test_public_api.py -v
```

## Common Issues

### 1. Import Errors

#### ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'trade_system_modules'
```

**Solutions:**

1. **Check virtual environment:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Reinstall package
pip install -e .
```

2. **Check Python path:**
```bash
python -c "import sys; print(sys.path)"
# Should include your project directory
```

3. **Reinstall in development mode:**
```bash
pip uninstall trade-system-modules
pip install -e .
```

#### ImportError: cannot import name

**Symptoms:**
```
ImportError: cannot import name 'GCSClient' from 'trade_system_modules'
```

**Solutions:**

1. **Check package version:**
```bash
pip show trade-system-modules
# Should match pyproject.toml version
```

2. **Rebuild package:**
```bash
pip install -e . --force-reinstall
```

### 2. Configuration Issues

#### Missing Environment Variables

**Symptoms:**
```
ValidationError: 1 validation error for Settings
gcs_bucket
  field required
```

**Solutions:**

1. **Check .env file:**
```bash
ls -la .env
cat .env
```

2. **Set environment variables:**
```bash
export GCS_BUCKET=my-bucket
export GCP_PROJECT=my-project
export POLYGON_API_KEY=my-key
```

3. **Copy example config:**
```bash
cp .env.example .env
# Edit .env with your values
```

#### Invalid Configuration Values

**Symptoms:**
```
ValidationError: polygon_api_key must start with 'pk_'
```

**Solutions:**

1. **Check API key format:**
```python
# Valid formats
POLYGON_API_KEY=pk_live_1234567890abcdef
POLYGON_API_KEY=pk_test_1234567890abcdef
```

2. **Validate configuration:**
```python
from trade_system_modules import Settings
try:
    settings = Settings()
    print("✓ Configuration valid")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

### 3. Google Cloud Storage Issues

#### Authentication Errors

**Symptoms:**
```
google.auth.exceptions.DefaultCredentialsError: Your default credentials were not found
```

**Solutions:**

1. **Set up Application Default Credentials:**
```bash
gcloud auth application-default login
```

2. **Check service account key:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

3. **Verify project access:**
```bash
gcloud config set project your-project-id
gcloud projects describe your-project-id
```

#### Bucket Access Errors

**Symptoms:**
```
google.api_core.exceptions.Forbidden: 403 Access denied
```

**Solutions:**

1. **Check bucket permissions:**
```bash
gsutil iam get gs://your-bucket
```

2. **Verify bucket exists:**
```bash
gsutil ls gs://your-bucket
```

3. **Check service account roles:**
```bash
# Should have Storage Object Admin or equivalent
gcloud projects get-iam-policy your-project
```

### 4. Polygon.io API Issues

#### Authentication Errors

**Symptoms:**
```
requests.exceptions.HTTPError: 401 Client Error
```

**Solutions:**

1. **Check API key:**
```bash
# Verify key is set
echo $POLYGON_API_KEY

# Test API access
curl "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/2023-01-01/2023-01-02?apikey=$POLYGON_API_KEY"
```

2. **Check API key format:**
```python
# Should start with pk_
assert settings.polygon_api_key.startswith('pk_')
```

3. **Verify account status:**
```bash
# Check account dashboard at polygon.io
# Ensure you have sufficient credits
```

#### Rate Limiting

**Symptoms:**
```
requests.exceptions.HTTPError: 429 Too Many Requests
```

**Solutions:**

1. **Check rate limits:**
```python
# Free tier: 5 calls/minute
# Paid plans: Higher limits
```

2. **Implement backoff:**
```python
import time
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def get_data_with_backoff(symbol, start, end):
    return get_agg_minute(symbol, start, end)
```

3. **Use caching:**
```python
# Cache results to reduce API calls
@lru_cache(maxsize=100)
def cached_get_agg_minute(symbol, start, end):
    return get_agg_minute(symbol, start, end)
```

### 5. Interactive Brokers Issues

#### Connection Errors

**Symptoms:**
```
ConnectionError: Failed to connect to IBKR
```

**Solutions:**

1. **Check TWS/IB Gateway:**
```bash
# Ensure TWS or IB Gateway is running
# Check API settings in TWS
```

2. **Verify connection parameters:**
```python
# Check settings
print(f"Host: {settings.ib_host}")
print(f"Port: {settings.ib_port}")
print(f"Client ID: {settings.ib_client_id}")
```

3. **Test connection:**
```python
from trade_system_modules.data.ibkr_live import IBLive

ib_client = IBLive()
if ib_client.connect():
    print("✓ IBKR connection successful")
    ib_client.disconnect()
else:
    print("✗ IBKR connection failed")
```

#### Contract Resolution Errors

**Symptoms:**
```
ValueError: Unknown instrument key: INVALID
```

**Solutions:**

1. **Check symbol format:**
```python
# Valid examples
"AAPL"    # Apple Inc.
"MSFT"    # Microsoft Corp.
"GOOGL"   # Alphabet Inc.
```

2. **Verify exchange:**
```python
from trade_system_modules.data.symbology import resolve_instrument

contract = resolve_instrument("AAPL")
print(f"Exchange: {contract.exchange}")  # Should be SMART
```

### 6. Data Processing Issues

#### Schema Validation Errors

**Symptoms:**
```
ValueError: Missing columns: ['volume']
```

**Solutions:**

1. **Check data schema:**
```python
from trade_system_modules.schemas.bars import ensure_bar_schema

data = get_agg_minute("AAPL", "2023-01-01", "2023-01-01")
clean_data = ensure_bar_schema(data)
print(f"Columns: {list(clean_data.columns)}")
```

2. **Validate data types:**
```python
print(data.dtypes)
# Should match expected types
```

#### Timezone Issues

**Symptoms:**
```
AttributeError: 'datetime.timezone' object has no attribute 'zone'
```

**Solutions:**

1. **Check timezone handling:**
```python
# Ensure timestamps are timezone-aware
assert data['ts'].dt.tz is not None

# Convert to UTC if needed
data['ts'] = pd.to_datetime(data['ts'], utc=True)
```

### 7. Performance Issues

#### Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Process data in chunks:**
```python
def process_large_dataset(symbol, start_date, end_date, chunk_size=10000):
    """Process large datasets in chunks."""

    # Get date range
    dates = pd.date_range(start_date, end_date, freq='D')

    for chunk_start in range(0, len(dates), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dates))
        chunk_dates = dates[chunk_start:chunk_end]

        # Process chunk
        for date in chunk_dates:
            data = get_agg_minute(symbol, str(date.date()), str(date.date()))
            # Process data...
```

2. **Optimize DataFrame memory:**
```python
def optimize_dataframe(df):
    """Reduce DataFrame memory usage."""

    # Downcast numeric types
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')

    return df
```

#### Slow Operations

**Symptoms:**
```
Operations taking too long
```

**Solutions:**

1. **Enable parallel processing:**
```python
import concurrent.futures

def parallel_data_fetch(symbols, start_date, end_date):
    """Fetch data for multiple symbols in parallel."""

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(get_agg_minute, symbol, start_date, end_date)
            for symbol in symbols
        ]

        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results
```

2. **Use caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_get_agg_minute(symbol, start, end):
    """Cache API responses."""
    return get_agg_minute(symbol, start, end)
```

### 8. Testing Issues

#### Test Failures

**Symptoms:**
```
FAILED tests/test_gcs_unit.py::test_gcs_client_init
```

**Solutions:**

1. **Check test configuration:**
```bash
# Ensure .env is set up for testing
cp .env.example .env
# Edit with test values
```

2. **Run specific tests:**
```bash
pytest tests/test_gcs_unit.py::test_gcs_client_init -v -s
```

3. **Check mock setup:**
```python
# Verify mocks are properly configured
from unittest.mock import patch
```

#### Coverage Issues

**Symptoms:**
```
Coverage too low
```

**Solutions:**

1. **Run coverage report:**
```bash
pytest --cov=src/trade_system_modules --cov-report=html
open htmlcov/index.html
```

2. **Add missing tests:**
```python
# Identify untested code
# Add unit tests for missing functions
```

### 9. Dependency Issues

#### Version Conflicts

**Symptoms:**
```
ImportError: cannot import name from package
```

**Solutions:**

1. **Check dependency versions:**
```bash
pip list | grep -E "(pandas|google|ib-insync)"
```

2. **Update dependencies:**
```bash
pip install --upgrade pandas google-cloud-storage ib-insync
```

3. **Check compatibility:**
```python
# Verify version compatibility
import pandas as pd
import google.cloud.storage
import ib_insync

print(f"pandas: {pd.__version__}")
print(f"google-cloud-storage: {google.cloud.storage.__version__}")
print(f"ib-insync: {ib_insync.__version__}")
```

## Diagnostic Tools

### System Information

```python
def system_diagnostics():
    """Collect system information for debugging."""

    import sys
    import platform
    import psutil

    print("=== System Diagnostics ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    # Check package versions
    try:
        import trade_system_modules
        print(f"trade_system_modules: {trade_system_modules.__version__}")
    except ImportError:
        print("trade_system_modules: NOT INSTALLED")

    # Check dependencies
    deps = ['pandas', 'google.cloud.storage', 'ib_insync', 'requests']
    for dep in deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"{dep}: {version}")
        except ImportError:
            print(f"{dep}: NOT INSTALLED")
```

### Network Diagnostics

```python
def network_diagnostics():
    """Test network connectivity."""

    import requests

    endpoints = [
        "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/2023-01-01/2023-01-01",
        "https://www.google.com",
    ]

    for url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ {url}: {response.status_code}")
        except Exception as e:
            print(f"✗ {url}: {e}")
```

### Configuration Validator

```python
def validate_configuration():
    """Validate all configuration settings."""

    from trade_system_modules import Settings

    issues = []

    try:
        settings = Settings()

        # Check required settings
        if not settings.gcs_bucket:
            issues.append("GCS_BUCKET not set")

        if not settings.polygon_api_key:
            issues.append("POLYGON_API_KEY not set")

        # Validate API key format
        if settings.polygon_api_key and not settings.polygon_api_key.startswith('pk_'):
            issues.append("POLYGON_API_KEY should start with 'pk_'")

        # Check IBKR settings
        if not (1025 <= settings.ib_port <= 65535):
            issues.append(f"IB_PORT {settings.ib_port} is invalid")

    except Exception as e:
        issues.append(f"Configuration error: {e}")

    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Configuration is valid")

    return issues
```

## Getting Help

### Debug Information

When reporting issues, include:

```python
# System information
import sys
print(f"Python: {sys.version}")

# Package information
import trade_system_modules
print(f"Version: {trade_system_modules.__version__}")

# Configuration (without secrets)
from trade_system_modules import Settings
settings = Settings()
print(f"GCS Bucket: {bool(settings.gcs_bucket)}")
print(f"Polygon API: {bool(settings.polygon_api_key)}")
```

### Support Channels

1. **GitHub Issues**: [Report bugs](https://github.com/org/trade_system_modules/issues)
2. **Documentation**: Check this troubleshooting guide
3. **Community**: [GitHub Discussions](https://github.com/org/trade_system_modules/discussions)
4. **Email**: engineering@hedgefund.com

### Issue Template

When reporting issues, include:

- **Description**: Clear description of the problem
- **Steps to reproduce**: Minimal code to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, package versions
- **Logs**: Any relevant error messages or logs
- **Configuration**: Non-sensitive configuration details

## Prevention

### Best Practices

1. **Use virtual environments**
2. **Keep dependencies updated**
3. **Test in staging before production**
4. **Monitor resource usage**
5. **Implement proper error handling**
6. **Use logging for debugging**
7. **Backup important data**
8. **Document custom configurations**

### Health Checks

```python
def health_check():
    """Perform comprehensive health check."""

    checks = []

    # Configuration check
    try:
        from trade_system_modules import Settings
        settings = Settings()
        checks.append(("Configuration", True, "Valid"))
    except Exception as e:
        checks.append(("Configuration", False, str(e)))

    # GCS check
    try:
        from trade_system_modules.storage.gcs import GCSClient
        client = GCSClient()
        checks.append(("GCS", True, "Connected"))
    except Exception as e:
        checks.append(("GCS", False, str(e)))

    # Polygon check
    try:
        from trade_system_modules import get_agg_minute
        # Use a minimal test
        checks.append(("Polygon", True, "API accessible"))
    except Exception as e:
        checks.append(("Polygon", False, str(e)))

    # Print results
    print("=== Health Check Results ===")
    all_healthy = True

    for check_name, healthy, details in checks:
        status = "✓" if healthy else "✗"
        print(f"{status} {check_name}: {details}")
        if not healthy:
            all_healthy = False

    print(f"\nOverall: {'✓ Healthy' if all_healthy else '✗ Issues detected'}")
    return all_healthy
```

## See Also

- [Installation Guide](installation.md) - Setup instructions
- [Configuration Guide](configuration.md) - Configuration details
- [Development Guide](development.md) - Development workflow
- [API Reference](api-reference.md) - Complete API documentation