# API Reference

Complete reference for all public APIs in `trade_system_modules`.

## Table of Contents

- [Configuration](#configuration)
- [Data Adapters](#data-adapters)
- [Storage](#storage)
- [Execution](#execution)
- [Schemas](#schemas)
- [Utilities](#utilities)

## Configuration

### Settings

```python
from trade_system_modules.config.settings import Settings, settings
```

Main configuration class using Pydantic Settings.

#### Attributes

- `gcs_bucket: Optional[str]` - Google Cloud Storage bucket name
- `gcp_project: Optional[str]` - Google Cloud project ID
- `polygon_api_key: Optional[str]` - Polygon.io API key
- `ib_host: str` - Interactive Brokers host (default: "127.0.0.1")
- `ib_port: int` - Interactive Brokers port (default: 7497)
- `ib_client_id: int` - Interactive Brokers client ID (default: 1)

#### Usage

```python
# Access global settings instance
from trade_system_modules.config.settings import settings

bucket = settings.gcs_bucket
api_key = settings.polygon_api_key

# Create custom settings
custom_settings = Settings(gcs_bucket="my-bucket")
```

## Data Adapters

### get_agg_minute

```python
from trade_system_modules.data.polygon_adapter import get_agg_minute, get_agg_minute_batch
import asyncio

async def get_agg_minute(symbol: str, start: str, end: str, concurrency: int = 20) -> pd.DataFrame
```

Get minute-level aggregate bars from Polygon.io.

#### Parameters

- `symbol: str` - Stock symbol (e.g., "AAPL")
- `start: str` - Start date in YYYY-MM-DD format
- `end: str` - End date in YYYY-MM-DD format
- `concurrency: int = 20` - Number of concurrent requests

#### Returns

`pd.DataFrame` with columns:
- `ts` - Timestamp (America/New_York timezone-aware)
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume
- `trades` - Number of trades

#### Example

```python
from trade_system_modules.data.polygon_adapter import get_agg_minute

# Get AAPL data for January 2023
data = asyncio.run(get_agg_minute("AAPL", "2023-01-01", "2023-01-31", concurrency=20))
print(data.head())
```

### resolve_instrument

```python
from trade_system_modules.data.symbology import resolve_instrument

def resolve_instrument(symbol: str) -> ib_insync.Stock
```

Resolve a stock symbol to an Interactive Brokers contract.

#### Parameters

- `symbol: str` - Stock symbol (e.g., "AAPL")

#### Returns

`ib_insync.Stock` - IBKR stock contract object

#### Example

```python
from trade_system_modules.data.symbology import resolve_instrument

contract = resolve_instrument("AAPL")
print(f"Symbol: {contract.symbol}")
print(f"Exchange: {contract.exchange}")

### get_agg_minute_batch

```python
from trade_system_modules.data.polygon_adapter import get_agg_minute_batch
import asyncio

async def get_agg_minute_batch(symbols: List[str], start: str, end: str, concurrency: int = 20) -> Dict[str, pd.DataFrame]
```

Get minute aggregates for multiple symbols in parallel using async downloads.

#### Parameters
- `symbols: List[str]` - List of stock symbols
- `start: str` - Start date in YYYY-MM-DD format
- `end: str` - End date in YYYY-MM-DD format
- `concurrency: int = 20` - Number of concurrent requests per symbol batch

#### Returns
`Dict[str, pd.DataFrame]` - Dictionary mapping symbol to its DataFrame

#### Example
```python
symbols = ["AAPL", "MSFT", "GOOGL"]
data_dict = asyncio.run(get_agg_minute_batch(symbols, "2023-01-01", "2023-01-02"))
aapl_data = data_dict["AAPL"]
```
```

## Storage

### GCSClient

```python
from trade_system_modules.storage.gcs import GCSClient

class GCSClient:
    def __init__(self, bucket: Optional[str] = None)
    def to_parquet(self, df: pd.DataFrame, path: str) -> None
    def read_parquet(self, path: str) -> pd.DataFrame
```

Google Cloud Storage client for DataFrame operations.

#### Constructor

- `bucket: Optional[str]` - GCS bucket name (uses settings if None)

#### Methods

#### to_parquet(df, path)

Write DataFrame to parquet file in GCS.

**Parameters:**
- `df: pd.DataFrame` - Data to write
- `path: str` - GCS path (e.g., "data/aapl.parquet")

**Example:**
```python
from trade_system_modules.storage.gcs import GCSClient

client = GCSClient()
client.to_parquet(data, "data/aapl_minute.parquet")
```

#### read_parquet(path)

Read parquet file from GCS into DataFrame.

**Parameters:**
- `path: str` - GCS path to parquet file

**Returns:**
- `pd.DataFrame` - Loaded data

**Example:**
```python
data = client.read_parquet("data/aapl_minute.parquet")
```

## Execution

### IBLive

```python
from trade_system_modules.data.ibkr_live import IBLive

class IBLive:
    def __init__(self)
    def connect(self) -> bool
    def disconnect(self)
    def get_historical_data(self, contract, duration, bar_size) -> pd.DataFrame
    def subscribe_realtime(self, contract, callback)
    def unsubscribe_realtime(self, contract)
```

Interactive Brokers live data client.

#### Methods

#### connect()

Connect to IBKR TWS/Gateway.

**Returns:**
- `bool` - True if connection successful

#### disconnect()

Disconnect from IBKR.

#### get_historical_data(contract, duration, bar_size)

Get historical market data.

**Parameters:**
- `contract` - IBKR contract object
- `duration: str` - Duration (e.g., "1 D")
- `bar_size: str` - Bar size (e.g., "1 min")

**Returns:**
- `pd.DataFrame` - Historical data

#### subscribe_realtime(contract, callback)

Subscribe to real-time data.

**Parameters:**
- `contract` - IBKR contract object
- `callback` - Function to handle tick data

#### unsubscribe_realtime(contract)

Unsubscribe from real-time data.

**Parameters:**
- `contract` - IBKR contract object

### IBExec

```python
from trade_system_modules.execution.ibkr_exec import IBExec

class IBExec:
    def __init__(self)
    def connect(self) -> bool
    def place_order(self, contract, order) -> ib_insync.Trade
    def cancel_order(self, order_id: int)
    def get_positions(self) -> List[ib_insync.Position]
    def get_account_summary(self) -> Dict[str, str]
```

Interactive Brokers execution client.

#### Methods

#### connect()

Connect to IBKR for trading.

**Returns:**
- `bool` - True if connection successful

#### place_order(contract, order)

Place a trading order.

**Parameters:**
- `contract` - IBKR contract object
- `order` - IBKR order object

**Returns:**
- `ib_insync.Trade` - Trade object

#### cancel_order(order_id)

Cancel an order by ID.

**Parameters:**
- `order_id: int` - Order ID to cancel

#### get_positions()

Get current positions.

**Returns:**
- `List[ib_insync.Position]` - List of positions

#### get_account_summary()

Get account summary information.

**Returns:**
- `Dict[str, str]` - Account information

## Schemas

### BAR_COLUMNS

```python
from trade_system_modules.schemas.bars import BAR_COLUMNS
```

Standard column names for bar data.

**Value:**
```python
["ts", "open", "high", "low", "close", "volume", "trades"]
```

### ensure_bar_schema

```python
from trade_system_modules.schemas.bars import ensure_bar_schema

def ensure_bar_schema(df: pd.DataFrame) -> pd.DataFrame
```

Ensure DataFrame has standard bar schema.

#### Parameters

- `df: pd.DataFrame` - Input DataFrame

#### Returns

- `pd.DataFrame` - DataFrame with standard bar columns

#### Behavior

- Adds missing columns with NaN values
- Converts `ts` column to UTC timezone-aware datetime
- Reorders columns to standard format

## Utilities

### Time Utilities

```python
from trade_system_modules.utils.time import utcnow

def utcnow() -> datetime
```

Get current UTC datetime.

**Returns:**
- `datetime` - Current UTC time (timezone-aware)

**Example:**
```python
from trade_system_modules.utils.time import utcnow

current_time = utcnow()
print(f"Current UTC: {current_time}")
```

## Error Handling

### Custom Exceptions

The package defines these custom exceptions:

- `ConfigurationError` - Configuration-related errors
- `DataAdapterError` - Data source errors
- `StorageError` - Storage operation errors
- `ExecutionError` - Trading execution errors

### Retry Behavior

Data adapters use tenacity for automatic retries:

- **Max attempts:** 3
- **Backoff:** Exponential (1s, 4s, 10s)
- **Exceptions:** Retries on network errors, timeouts
- **No retry:** HTTP 4xx errors (client errors)

## Type Hints

All public APIs include comprehensive type hints:

```python
from typing import Optional, List, Dict
import pandas as pd
import ib_insync

# Example type hints
def get_agg_minute(symbol: str, start: str, end: str) -> pd.DataFrame:
    ...

def resolve_instrument(symbol: str) -> ib_insync.Stock:
    ...
```

## Constants

### Version Information

```python
import trade_system_modules

print(trade_system_modules.__version__)  # "0.1.0"
```

### Default Values

```python
# Default IBKR connection settings
DEFAULT_IB_HOST = "127.0.0.1"
DEFAULT_IB_PORT = 7497
DEFAULT_IB_CLIENT_ID = 1
```

## Thread Safety

- **GCSClient**: Thread-safe for concurrent operations
- **IBLive/IBExec**: Not thread-safe, use one instance per thread
- **Settings**: Thread-safe singleton

## Performance Considerations

- **DataFrame operations**: Use pandas vectorized operations
- **GCS operations**: Batch operations when possible
- **IBKR connections**: Maintain persistent connections
- **Memory usage**: Monitor DataFrame sizes for large datasets

## Examples

See [Examples](examples/) directory for complete usage examples:

- [Basic Data Retrieval](examples/basic_data.py)
- [Storage Operations](examples/storage_example.py)
- [Live Trading Setup](examples/live_trading.py)
- [Backtesting Integration](examples/backtesting.py)

## See Also

- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [Data Adapters](data-adapters.md)
- [Storage Guide](storage.md)
- [Execution Guide](execution.md)
- [Troubleshooting](troubleshooting.md)