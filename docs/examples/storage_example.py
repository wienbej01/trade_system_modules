#!/usr/bin/env python3
"""
Storage operations example for trade_system_modules.

This example demonstrates:
- Batch data uploads to Google Cloud Storage
- File organization and metadata
- Performance monitoring
- Error handling and retries
"""

from trade_system_modules import GCSClient, get_agg_minute
from trade_system_modules.schemas.bars import ensure_bar_schema
import pandas as pd
import time
from pathlib import Path

def main():
    """Main storage example function."""

    print("=== Trade System Modules - Storage Example ===\n")

    # 1. Initialize GCS client
    print("1. Initializing GCS client")
    try:
        client = GCSClient()
        print(f"   Connected to bucket: {client.bucket_name}")
        print()
    except Exception as e:
        print(f"   GCS connection failed: {e}")
        print("   Make sure GOOGLE_APPLICATION_CREDENTIALS is set")
        return

    # 2. Prepare sample data
    print("2. Preparing sample data")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    data_frames = {}

    for symbol in symbols:
        try:
            print(f"   Fetching data for {symbol}...")
            data = get_agg_minute(symbol, "2023-01-01", "2023-01-02")
            data = ensure_bar_schema(data)
            data_frames[symbol] = data
            print(f"   ✓ {symbol}: {len(data)} rows")
        except Exception as e:
            print(f"   ✗ {symbol}: {e}")

    print()

    # 3. Individual file uploads
    print("3. Individual file uploads")
    upload_times = {}

    for symbol, data in data_frames.items():
        start_time = time.time()
        try:
            path = f"examples/individual/{symbol}_2023.parquet"
            client.to_parquet(data, path)
            upload_time = time.time() - start_time
            upload_times[symbol] = upload_time
            print(".2f")
        except Exception as e:
            print(f"   ✗ {symbol}: {e}")

    print()

    # 4. Batch upload
    print("4. Batch upload example")
    batch_start = time.time()

    try:
        # Combine all data
        combined_data = pd.concat([
            df.assign(symbol=symbol)
            for symbol, df in data_frames.items()
        ], ignore_index=True)

        # Upload combined dataset
        batch_path = "examples/batch/combined_2023.parquet"
        client.to_parquet(combined_data, batch_path)

        batch_time = time.time() - batch_start
        print(".2f")
        print(f"   Combined size: {len(combined_data)} rows")
        print()

    except Exception as e:
        print(f"   ✗ Batch upload failed: {e}")
        print()

    # 5. File organization
    print("5. File organization and metadata")

    # Create organized structure
    organized_data = {
        "stocks": {
            "tech": data_frames,
            "indices": {}  # Could add index data here
        }
    }

    for category, subcategories in organized_data.items():
        for subcategory, symbol_data in subcategories.items():
            for symbol, data in symbol_data.items():
                try:
                    path = f"organized/{category}/{subcategory}/{symbol}_2023.parquet"
                    client.to_parquet(data, path)
                    print(f"   ✓ Organized: {path}")
                except Exception as e:
                    print(f"   ✗ {symbol}: {e}")

    print()

    # 6. Performance comparison
    print("6. Performance analysis")

    if upload_times:
        avg_time = sum(upload_times.values()) / len(upload_times)
        print(".2f")

        print("   Individual upload times:")
        for symbol, upload_time in upload_times.items():
            print(".2f")

    print()

    # 7. Data retrieval example
    print("7. Data retrieval and verification")

    for symbol in symbols[:2]:  # Test first 2 symbols
        try:
            path = f"examples/individual/{symbol}_2023.parquet"
            retrieved_data = client.read_parquet(path)

            print(f"   ✓ {symbol}: Retrieved {len(retrieved_data)} rows")
            print(f"     Columns: {list(retrieved_data.columns)}")
            print(f"     Date range: {retrieved_data['ts'].min()} to {retrieved_data['ts'].max()}")

        except Exception as e:
            print(f"   ✗ {symbol}: {e}")

    print()

    # 8. Cost optimization example
    print("8. Cost optimization - compression comparison")

    test_data = data_frames.get("AAPL", pd.DataFrame())
    if not test_data.empty:
        # Test different compression methods
        compressions = ["snappy", "gzip", "brotli"]

        for compression in compressions:
            try:
                # Note: This is a simplified example
                # In practice, you'd measure actual file sizes
                path = f"examples/compression/{compression}_test.parquet"

                start_time = time.time()
                client.to_parquet(test_data, path)
                upload_time = time.time() - start_time

                print(".2f")

            except Exception as e:
                print(f"   ✗ {compression}: {e}")

    print()

    # 9. Error handling demonstration
    print("9. Error handling and recovery")

    # Test with invalid path
    try:
        invalid_data = client.read_parquet("nonexistent/file.parquet")
    except Exception as e:
        print(f"   ✓ Gracefully handled missing file: {type(e).__name__}")

    # Test with corrupted data handling
    try:
        # Create empty dataframe and try to upload
        empty_data = pd.DataFrame()
        client.to_parquet(empty_data, "examples/empty_test.parquet")
        print("   ✓ Handled empty data gracefully")
    except Exception as e:
        print(f"   ✗ Empty data handling: {e}")

    print()

    print("=== Storage example completed! ===")
    print("\nNext steps:")
    print("- Check your GCS bucket for uploaded files")
    print("- Review the organized folder structure")
    print("- Monitor costs in Google Cloud Console")

if __name__ == "__main__":
    main()