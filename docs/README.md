# Trade System Modules Documentation

## Overview

`trade_system_modules` is a comprehensive Python package providing shared infrastructure for quantitative trading systems. It offers standardized interfaces for data acquisition, storage, execution, and configuration management across multiple trading strategies.

## Architecture

The package follows a modular architecture designed for scalability and maintainability:

```
trade_system_modules/
├── config/          # Configuration management
├── data/            # Data adapters and symbology
├── storage/         # Cloud storage interfaces
├── execution/       # Trading execution engines
├── schemas/         # Data validation schemas
├── utils/           # Common utilities
└── __init__.py      # Public API exports
```

### Core Components

- **Configuration**: Centralized settings management using Pydantic
- **Data Adapters**: Standardized interfaces for market data sources
- **Storage**: Cloud storage abstraction (Google Cloud Storage)
- **Execution**: Trading execution interfaces (Interactive Brokers)
- **Schemas**: Data validation and standardization
- **Utilities**: Common helper functions and time utilities

## Quick Start

```python
from trade_system_modules import Settings, get_agg_minute, GCSClient, IBLive

# Access configuration
settings = Settings()

# Get market data
data = get_agg_minute("AAPL", "2023-01-01", "2023-01-02")

# Store data in cloud
client = GCSClient()
client.to_parquet(data, "data/aapl_minute.parquet")
```

## Documentation Sections

- [Installation](installation.md) - Setup and deployment
- [Configuration](configuration.md) - Environment and settings
- [API Reference](api-reference.md) - Complete API documentation
- [Data Adapters](data-adapters.md) - Market data interfaces
- [Storage](storage.md) - Cloud storage operations
- [Execution](execution.md) - Trading execution
- [Examples](examples/) - Usage examples
- [Cookiecutter Template](cookiecutter.md) - Strategy scaffolding
- [Development](development.md) - Contributing guidelines
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Architecture](architecture.md) - System design and diagrams

## Key Features

- **🔧 Modular Design**: Clean separation of concerns with pluggable components
- **📊 Multi-Source Data**: Support for Polygon.io and Interactive Brokers data
- **☁️ Cloud Storage**: Google Cloud Storage integration with parquet support
- **⚡ High Performance**: Optimized for real-time trading applications
- **🛡️ Type Safety**: Full type hints and Pydantic validation
- **🧪 Well Tested**: Comprehensive test suite with 14+ test cases
- **📦 Reproducible**: Deterministic builds with pinned dependencies
- **🚀 Developer Experience**: Cookiecutter templates for rapid strategy development

## Versioning

This package follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes to public API
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Current version: `0.1.0` (Public API frozen)

## Support

For questions, issues, or contributions:
- 📧 Email: engineering@hedgefund.com
- 🐛 Issues: [GitHub Issues](https://github.com/org/trade_system_modules/issues)
- 📖 Docs: [Full Documentation](https://github.com/org/trade_system_modules/tree/main/docs)

## License

MIT License - see [LICENSE](../LICENSE) file for details.