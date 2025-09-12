# Data Adapters Guide

This guide covers the data adapter modules for retrieving market data from various sources.

## Overview

The package provides standardized interfaces for market data:

- **Polygon.io**: High-performance market data API
- **Interactive Brokers**: Live market data and symbology
- **Symbology**: Symbol resolution and contract mapping

## Polygon.io Adapter

### get_agg_minute() - Async Enhanced

Retrieve minute-level aggregate bars from Polygon.io using asynchronous parallel downloads with day-wise chunking for efficient 5+ year range handling.

```python
from trade_system_modules.data.polygon_adapter import get_agg_minute
import asyncio

# Get AAPL minute data for a specific date range (async)
data = asyncio.run(get_agg_minute("AAPL", "2023-01-01", "2023-01-02", concurrency=20))
print(data.head())
```

#### Features

- **Automatic retries** with exponential backoff
- **Rate limiting** handled by tenacity
- **Data validation** and schema enforcement
- **Timezone handling** (UTC conversion)
- **Error handling** for API failures

#### Parameters
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `symbol` | `str` | Stock symbol | `"AAPL"` |
| `start` | `str` | Start date (YYYY-MM-DD) | `"2023-01-01"` |
| `end` | `str` | End date (YYYY-MM-DD) | `"2023-01-02"` |
| `concurrency` | `int` | Concurrent requests (default: 20) | `20` |


#### Response Schema

```python
# DataFrame columns
{
    "ts": "datetime64[ns, America/New_York]",  # Timestamp (NY timezone)
    "open": "float64",             # Opening price
    "high": "float64",             # High price
    "low": "float64",              # Low price
    "close": "float64",            # Closing price
    "volume": "int64",             # Trading volume
    "trades": "int64"              # Number of trades
}
```

#### Error Handling

```python
from trade_system_modules.data.polygon_adapter import get_agg_minute
from tenacity import RetryError

try:
    data = get_agg_minute("AAPL", "2023-01-01", "2023-01-02")
except RetryError as e:
    print(f"Failed after retries: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

#### Rate Limits

Polygon.io has these rate limits:
- **Free tier**: 5 API calls/minute
- **Paid plans**: Higher limits based on subscription (unlimited for top tiers)

The async adapter maximizes throughput through:
- Parallel async requests with configurable concurrency
- Day-wise chunking for long date ranges (5+ years)
- Automatic retries with backoff per request
- Efficient session management with connection pooling

## Interactive Brokers Live Data

### IBLive Class

Real-time and historical market data from Interactive Brokers.

```python
from trade_system_modules.data.ibkr_live import IBLive
from trade_system_modules.data.symbology import resolve_instrument

# Initialize client
ib_client = IBLive()

# Connect to IBKR
if ib_client.connect():
    print("Connected to IBKR")

    # Resolve symbol
    contract = resolve_instrument("AAPL")

    # Get historical data
    data = ib_client.get_historical_data(
        contract=contract,
        duration="1 D",
        bar_size="1 min"
    )
    print(data.head())

    ib_client.disconnect()
```

#### Connection Management

```python
# Connect with error handling
try:
    connected = ib_client.connect()
    if not connected:
        print("Failed to connect to IBKR")
        exit(1)
except Exception as e:
    print(f"Connection error: {e}")
    exit(1)
```

#### Historical Data

```python
# Available durations
durations = ["1 D", "2 D", "1 W", "2 W", "1 M"]

# Available bar sizes
bar_sizes = ["1 min", "5 mins", "15 mins", "1 hour", "1 day"]

# Get daily bars for past week
daily_data = ib_client.get_historical_data(
    contract=contract,
    duration="1 W",
    bar_size="1 day"
)
```

#### Real-time Data

```python
def on_tick(tick):
    """Handle real-time tick data."""
    print(f"Price: {tick.last} at {tick.time}")

# Subscribe to real-time data
ib_client.subscribe_realtime(contract, on_tick)

# Keep connection alive
import time
time.sleep(300)  # Monitor for 5 minutes

# Unsubscribe
ib_client.unsubscribe_realtime(contract)
```

## Symbology Resolution

### resolve_instrument()

Convert stock symbols to IBKR contract objects.

```python
from trade_system_modules.data.symbology import resolve_instrument

# Resolve common stocks
aapl_contract = resolve_instrument("AAPL")
msft_contract = resolve_instrument("MSFT")
googl_contract = resolve_instrument("GOOGL")

print(f"AAPL: {aapl_contract.symbol} on {aapl_contract.exchange}")
```

#### Supported Symbols

- **US Stocks**: Standard tickers (AAPL, MSFT, GOOGL)
- **Exchange**: SMART routing by default
- **Currency**: USD

#### Contract Details

```python
contract = resolve_instrument("AAPL")

print(f"Symbol: {contract.symbol}")
print(f"Security Type: {contract.secType}")  # "STK"
print(f"Exchange: {contract.exchange}")      # "SMART"
print(f"Currency: {contract.currency}")      # "USD"
print(f"Primary Exchange: {contract.primaryExchange}")
```

#### Error Handling

```python
try:
    contract = resolve_instrument("INVALID")
except ValueError as e:
    print(f"Symbol resolution failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Data Processing Pipeline

### Typical Workflow

```python
from trade_system_modules import get_agg_minute, resolve_instrument, IBLive
from trade_system_modules.schemas.bars import ensure_bar_schema
import pandas as pd

def get_market_data(symbol: str, start_date: str, end_date: str):
    """Get market data with fallback options."""

    # Try Polygon.io first (fast, reliable)
    try:
        data = get_agg_minute(symbol, start_date, end_date)
        print(f"Retrieved {len(data)} bars from Polygon.io")
        return ensure_bar_schema(data)
    except Exception as e:
        print(f"Polygon.io failed: {e}")

    # Fallback to IBKR live data
    try:
        ib_client = IBLive()
        if ib_client.connect():
            contract = resolve_instrument(symbol)
            data = ib_client.get_historical_data(
                contract=contract,
                duration="1 M",
                bar_size="1 min"
            )
            ib_client.disconnect()
            print(f"Retrieved {len(data)} bars from IBKR")
            return ensure_bar_schema(data)
    except Exception as e:
        print(f"IBKR fallback failed: {e}")

    raise RuntimeError(f"Failed to get data for {symbol}")
```

## Data Quality and Validation

### Schema Enforcement

```python
from trade_system_modules.schemas.bars import ensure_bar_schema, BAR_COLUMNS

# Raw data
raw_data = get_agg_minute("AAPL", "2023-01-01", "2023-01-02")

# Ensure consistent schema
clean_data = ensure_bar_schema(raw_data)

# Verify columns
assert list(clean_data.columns) == BAR_COLUMNS
assert clean_data["ts"].dt.tz is not None  # Timezone-aware
```

### Data Validation

```python
def validate_market_data(df):
    """Validate market data quality."""

    # Check for required columns
    required_cols = ["ts", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check for timezone awareness
    if df["ts"].dt.tz is None:
        raise ValueError("Timestamp column must be timezone-aware")

    # Check for reasonable price ranges
    if (df["low"] > df["high"]).any():
        raise ValueError("Low price cannot be higher than high price")

    # Check for negative volumes
    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")

    return True
```

## Performance Optimization

### Batch Processing

```python
from trade_system_modules.data.polygon_adapter import get_agg_minute_batch
import asyncio

# Get data for multiple symbols in parallel (native async)
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
data_dict = asyncio.run(get_agg_minute_batch(symbols, "2023-01-01", "2023-01-02", concurrency=20))

# Access individual DataFrames
aapl_data = data_dict["AAPL"]
print(aapl_data.head())
```

### Caching Strategy

```python
from functools import lru_cache
import pickle
from pathlib import Path

@lru_cache(maxsize=100)
def cached_get_agg_minute(symbol: str, start: str, end: str):
    """Cached version of get_agg_minute."""

    cache_key = f"{symbol}_{start}_{end}"
    cache_file = Path(f".cache/{cache_key}.pkl")

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    data = get_agg_minute(symbol, start, end)

    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    return data
```

## Configuration

### Environment Variables

```bash
# Polygon.io API
POLYGON_API_KEY=your_api_key_here

# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
```

### Runtime Configuration

```python
from trade_system_modules.config.settings import settings

# Check API key availability
if not settings.polygon_api_key:
    print("Warning: Polygon.io API key not configured")

# Configure IBKR connection
print(f"IBKR Host: {settings.ib_host}:{settings.ib_port}")
```

## Error Handling Patterns

### Graceful Degradation

```python
def get_data_with_fallback(symbol, start, end):
    """Get data with multiple fallback strategies."""

    sources = [
        ("polygon", lambda: get_agg_minute(symbol, start, end)),
        ("ibkr", lambda: get_ibkr_data(symbol, start, end)),
        ("cache", lambda: get_cached_data(symbol, start, end)),
    ]

    for source_name, getter in sources:
        try:
            data = getter()
            print(f"Successfully retrieved data from {source_name}")
            return data
        except Exception as e:
            print(f"{source_name} failed: {e}")
            continue

    raise RuntimeError(f"All data sources failed for {symbol}")
```

### Circuit Breaker Pattern

```python
from time import sleep

def get_data_with_circuit_breaker(symbol, start, end, max_retries=3):
    """Get data with circuit breaker for failing services."""

    failures = 0

    for attempt in range(max_retries):
        try:
            return get_agg_minute(symbol, start, end)
        except Exception as e:
            failures += 1
            if failures >= 2:  # Open circuit after 2 failures
                print(f"Circuit breaker open for {symbol}")
                raise RuntimeError(f"Service unavailable: {e}")

            sleep(2 ** attempt)  # Exponential backoff

    raise RuntimeError("Max retries exceeded")
```

## Monitoring and Logging

### Request Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_get_agg_minute(symbol, start, end):
    """Get aggregate data with logging."""

    logger.info(f"Requesting data for {symbol} from {start} to {end}")

    try:
        data = get_agg_minute(symbol, start, end)
        logger.info(f"Retrieved {len(data)} bars for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Failed to get data for {symbol}: {e}")
        raise
```

### Performance Monitoring

```python
import time

def timed_get_agg_minute(symbol, start, end):
    """Get data with performance timing."""

    start_time = time.time()
    try:
        data = get_agg_minute(symbol, start, end)
        duration = time.time() - start_time

        print(f"Data retrieval took {duration:.2f}s")
        print(f"Retrieved {len(data)} bars ({len(data)/duration:.0f} bars/sec)")

        return data
    except Exception as e:
        duration = time.time() - start_time
        print(f"Request failed after {duration:.2f}s: {e}")
        raise
```

## Best Practices

1. **Use appropriate data sources** for your use case
2. **Implement proper error handling** and fallbacks
3. **Cache frequently accessed data** to reduce API calls
4. **Validate data quality** before processing
5. **Monitor API usage** to stay within rate limits
6. **Handle timezone conversions** consistently
7. **Log important events** for debugging
8. **Test with different market conditions**

## See Also

- [API Reference](api-reference.md) - Complete function signatures
- [Configuration Guide](configuration.md) - Setup instructions
- [Examples](examples/) - Usage examples
- [Troubleshooting](troubleshooting.md) - Common issues