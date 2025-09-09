# Configuration Guide

This guide explains how to configure the `trade_system_modules` package for different environments and use cases.

## Overview

The package uses Pydantic Settings for configuration management, providing:
- Type-safe configuration
- Environment variable support
- Validation and defaults
- Hierarchical configuration

## Basic Configuration

### Environment File Setup

1. Copy the example configuration:
```bash
cp .env.example .env
```

2. Edit `.env` with your values:
```bash
# Google Cloud Configuration
GCS_BUCKET=my-trading-bucket
GCP_PROJECT=my-gcp-project

# Data Providers
POLYGON_API_KEY=your_polygon_api_key_here

# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

### Programmatic Configuration

```python
from trade_system_modules.config.settings import Settings

# Load from environment
settings = Settings()

# Access configuration
print(f"GCS Bucket: {settings.gcs_bucket}")
print(f"IB Host: {settings.ib_host}")
```

## Configuration Options

### Google Cloud Storage

```bash
# Required for GCS operations
GCS_BUCKET=your-bucket-name
GCP_PROJECT=your-project-id
```

**Setup Steps:**
1. Create a GCS bucket in Google Cloud Console
2. Enable Cloud Storage API
3. Set up authentication (see [Installation Guide](installation.md))

### Polygon.io Data

```bash
# Required for market data
POLYGON_API_KEY=your_api_key_here
```

**Setup Steps:**
1. Sign up at [Polygon.io](https://polygon.io)
2. Get your API key from the dashboard
3. Choose appropriate plan (free tier available)

### Interactive Brokers

```bash
# Required for live trading
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

**Setup Steps:**
1. Install Trader Workstation (TWS) or IB Gateway
2. Enable API connections in TWS settings
3. Configure API settings (read-only for paper trading)

## Advanced Configuration

### Custom Settings Class

```python
from pydantic_settings import BaseSettings
from typing import Optional

class CustomSettings(BaseSettings):
    # Inherit base settings
    gcs_bucket: Optional[str] = None
    gcp_project: Optional[str] = None

    # Add custom settings
    custom_timeout: int = 30
    enable_debug_logging: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False

# Use custom settings
settings = CustomSettings()
```

### Environment-Specific Configuration

```python
import os
from pathlib import Path

class EnvironmentSettings(BaseSettings):
    environment: str = "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def config_path(self) -> Path:
        return Path(f"config/{self.environment}.yaml")

# Set environment
os.environ["ENVIRONMENT"] = "production"
settings = EnvironmentSettings()
```

### Validation and Defaults

```python
from pydantic import Field, validator

class ValidatedSettings(BaseSettings):
    # Required fields
    api_key: str = Field(..., min_length=10)

    # Optional with defaults
    timeout: int = Field(default=30, ge=1, le=300)

    # Custom validation
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v.startswith('pk_'):
            raise ValueError('API key must start with pk_')
        return v
```

## Configuration Patterns

### Hierarchical Configuration

```python
# config/base.py
class BaseConfig(BaseSettings):
    app_name: str = "trading_system"
    log_level: str = "INFO"

# config/database.py
class DatabaseConfig(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    database: str = "trading"

# config/trading.py
class TradingConfig(BaseSettings):
    max_position_size: float = 100000.0
    risk_limit: float = 0.02
```

### Configuration Factory

```python
from typing import Dict, Type, TypeVar
from pydantic_settings import BaseSettings

T = TypeVar('T', bound=BaseSettings)

class ConfigFactory:
    @staticmethod
    def create_config(config_class: Type[T], **overrides) -> T:
        """Create configuration with optional overrides."""
        return config_class(**overrides)

# Usage
from trade_system_modules.config.settings import Settings

config = ConfigFactory.create_config(
    Settings,
    gcs_bucket="override-bucket"
)
```

## Environment Variables

### Naming Convention

The package uses these environment variable patterns:

- `UPPER_CASE` for settings
- `GCP_PROJECT` for Google Cloud
- `POLYGON_API_KEY` for data providers
- `IB_HOST`, `IB_PORT` for IBKR

### Loading Order

Configuration is loaded in this order (later sources override earlier):

1. Default values in code
2. Environment variables
3. `.env` file values
4. Runtime overrides

### Environment File Examples

#### Development
```bash
# .env.development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
GCS_BUCKET=dev-trading-bucket
POLYGON_API_KEY=pk_test_...
```

#### Production
```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=WARNING
GCS_BUCKET=prod-trading-bucket
POLYGON_API_KEY=pk_live_...
```

#### Testing
```bash
# .env.test
ENVIRONMENT=test
LOG_LEVEL=DEBUG
GCS_BUCKET=test-bucket
POLYGON_API_KEY=pk_test_...
```

## Security Best Practices

### API Keys

```bash
# Never commit API keys to version control
# Use environment variables or secure vaults

# Good: Use environment variables
POLYGON_API_KEY=${POLYGON_API_KEY}

# Better: Use secret management
# AWS Secrets Manager, Google Secret Manager, etc.
```

### File Permissions

```bash
# Set restrictive permissions on config files
chmod 600 .env
chmod 600 credentials.json
```

### Sensitive Data

```python
import os
from pathlib import Path

class SecureSettings(BaseSettings):
    api_key: str

    @property
    def api_key_secure(self) -> str:
        """Get API key from secure location."""
        key_path = Path.home() / ".trading" / "api_key"
        if key_path.exists():
            return key_path.read_text().strip()
        return self.api_key
```

## Configuration Validation

### Runtime Validation

```python
from trade_system_modules.config.settings import settings

def validate_configuration():
    """Validate configuration at startup."""
    required_settings = [
        settings.gcs_bucket,
        settings.polygon_api_key,
    ]

    missing = [name for name, value in zip(
        ['GCS_BUCKET', 'POLYGON_API_KEY'],
        required_settings
    ) if not value]

    if missing:
        raise ValueError(f"Missing required configuration: {missing}")

# Call at application startup
validate_configuration()
```

### Type Validation

```python
from pydantic import ValidationError

try:
    settings = Settings()
except ValidationError as e:
    print(f"Configuration error: {e}")
    exit(1)
```

## Troubleshooting

### Common Issues

**Configuration not loading:**
```python
# Check if .env file exists
from pathlib import Path
print(Path('.env').exists())

# Check environment variables
import os
print(os.environ.get('GCS_BUCKET'))
```

**Validation errors:**
```python
from pydantic import ValidationError
from trade_system_modules.config.settings import Settings

try:
    settings = Settings()
except ValidationError as e:
    for error in e.errors():
        print(f"{error['loc']}: {error['msg']}")
```

**Environment conflicts:**
```python
# Check loaded values
settings = Settings()
print(settings.dict())  # Shows all loaded values
```

## Best Practices

1. **Use environment-specific configs**
2. **Validate configuration at startup**
3. **Never commit sensitive data**
4. **Use secure vaults for production**
5. **Document all configuration options**
6. **Provide sensible defaults**
7. **Test configuration loading**

## Next Steps

- Review [API Reference](api-reference.md) for available settings
- Check [Examples](examples/) for configuration patterns
- Read [Development Guide](development.md) for extending configuration