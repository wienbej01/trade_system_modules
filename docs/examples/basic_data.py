#!/usr/bin/env python3
"""
Basic data retrieval example for trade_system_modules.

This example demonstrates:
- Retrieving market data from Polygon.io
- Basic data processing and validation
- Storing data in Google Cloud Storage
"""

from trade_system_modules import (
    Settings,
    get_agg_minute,
    GCSClient,
    resolve_instrument
)
from trade_system_modules.schemas.bars import ensure_bar_schema
import pandas as pd
import asyncio

def main():
    """Main example function."""

    print("=== Trade System Modules - Basic Data Example ===\n")

    # 1. Configuration
    print("1. Configuration")
    settings = Settings()
    print(f"   GCS Bucket: {settings.gcs_bucket}")
    print(f"   Polygon API Key configured: {bool(settings.polygon_api_key)}")
    print()

    # 2. Get market data
    print("2. Retrieving market data from Polygon.io")
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-02"

    try:
        data = asyncio.run(get_agg_minute(symbol, start_date, end_date))
        print(f"   Retrieved {len(data)} bars for {symbol}")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Columns: {list(data.columns)}")
        print()
    except Exception as e:
        print(f"   Error retrieving data: {e}")
        return

    # 3. Data validation and processing
    print("3. Data validation and processing")
    data = ensure_bar_schema(data)
    print(f"   Schema validated: {list(data.columns)}")
    print(f"   Timezone: {data['ts'].dt.tz}")
    print(f"   Data types:\n{data.dtypes}")
    print()

    # 4. Basic analysis
    print("4. Basic market analysis")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print()

    # 5. Store in cloud storage
    print("5. Storing data in Google Cloud Storage")
    try:
        client = GCSClient()
        path = f"examples/{symbol}_sample.parquet"
        client.to_parquet(data, path)
        print(f"   Data stored at: gs://{settings.gcs_bucket}/{path}")
        print()
    except Exception as e:
        print(f"   Error storing data: {e}")
        print("   Note: GCS requires proper authentication")
        print()

    # 6. Symbol resolution example
    print("6. Symbol resolution for Interactive Brokers")
    try:
        contract = resolve_instrument("AAPL")
        print(f"   Symbol: {contract.symbol}")
        print(f"   Exchange: {contract.exchange}")
        print(f"   Currency: {contract.currency}")
        print()
    except Exception as e:
        print(f"   Error resolving symbol: {e}")
        print()

    # 7. Data preview
    print("7. Data preview (first 5 rows)")
    print(data.head().to_string())
    print()

    print("=== Example completed successfully! ===")

if __name__ == "__main__":
    main()