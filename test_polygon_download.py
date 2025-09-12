#!/usr/bin/env python3
"""
Test Polygon.io data download for AAPL 1-minute data
"""

import sys
from pathlib import Path

# Add module path
sys.path.append(str(Path(__file__).parent / 'src'))

from trade_system_modules.data.polygon_adapter import get_agg_minute
from trade_system_modules.config.settings import settings

def test_polygon_download():
    """Test downloading AAPL 1-minute data from Polygon.io."""

    print("=== Polygon.io Data Download Test ===\n")

    # 1. Check API key configuration
    print("1. Configuration Check")
    if not settings.polygon_api_key:
        print("❌ POLYGON_API_KEY not found in environment variables")
        print("Please set your API key in the .env file")
        return

    print(f"✅ Polygon API Key configured: {settings.polygon_api_key[:8]}...")
    print()

    # 2. Download data
    print("2. Downloading AAPL 1-minute data")
    symbol = "AAPL"
    start_date = "2025-06-06"
    end_date = "2025-06-06"  # Same day for single day test

    print(f"Symbol: {symbol}")
    print(f"Date: {start_date}")
    print(f"Timeframe: 1 minute")
    print("Fetching data...")
    try:
        data = get_agg_minute(symbol, start_date, end_date)

        print("✅ Data download completed!")
        print(f"   Records retrieved: {len(data)}")
        print(f"   Data shape: {data.shape}")
        print()

        # 3. Data validation
        print("3. Data Validation")
        if data.empty:
            print("⚠️  No data returned - this could be because:")
            print("   - The date is in the future (2025-06-06)")
            print("   - No trading occurred on this date")
            print("   - API key issues")
            print("   - Date format issues")
        else:
            print("✅ Data validation:")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data['ts'].min()} to {data['ts'].max()}")
            print(f"   Timezone: {data['ts'].dt.tz}")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(f"   Total volume: {data['volume'].sum():,.0f}")
            print()

            # 4. Show sample data
            print("4. Sample Data (first 5 rows)")
            print(data.head().to_string())
            print()

            # 5. Show data statistics
            print("5. Data Statistics")
            print(f"   Trading minutes: {len(data)}")
            print(".2f")
            print(".2f")
            print(".2f")
            print(f"   Average volume per minute: {data['volume'].mean():,.0f}")
            print(f"   Max volume in a minute: {data['volume'].max():,.0f}")
            print()

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Possible causes:")
        print("   - Invalid API key")
        print("   - Network connectivity issues")
        print("   - Polygon.io service issues")
        print("   - Rate limiting")

def main():
    """Main function."""
    print("Polygon.io Data Download Test")
    print("Testing AAPL 1-minute data download for 2025-06-06\n")

    test_polygon_download()

if __name__ == "__main__":
    main()