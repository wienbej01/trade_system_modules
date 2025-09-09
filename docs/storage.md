# Storage Guide

This guide covers cloud storage operations using Google Cloud Storage (GCS) for DataFrame persistence and retrieval.

## Overview

The storage module provides:
- **Cloud Storage**: Google Cloud Storage integration
- **DataFrame Operations**: Parquet format support
- **Batch Operations**: Efficient bulk data handling
- **Error Handling**: Robust retry mechanisms

## GCSClient Class

### Initialization

```python
from trade_system_modules.storage.gcs import GCSClient

# Use default bucket from settings
client = GCSClient()

# Use specific bucket
client = GCSClient(bucket="my-custom-bucket")

# Check bucket name
print(f"Bucket: {client.bucket_name}")
```

### Connection Management

```python
# Client automatically handles authentication
# Uses Application Default Credentials (ADC)

# Verify connection
try:
    # GCS client is lazy-loaded on first operation
    print("GCS client ready")
except Exception as e:
    print(f"GCS connection failed: {e}")
```

## DataFrame Operations

### Writing DataFrames

```python
import pandas as pd
from trade_system_modules.storage.gcs import GCSClient

# Sample data
data = pd.DataFrame({
    "ts": pd.date_range("2023-01-01", periods=100, freq="1min"),
    "open": [100 + i*0.01 for i in range(100)],
    "high": [101 + i*0.01 for i in range(100)],
    "low": [99 + i*0.01 for i in range(100)],
    "close": [100.5 + i*0.01 for i in range(100)],
    "volume": [1000 + i*10 for i in range(100)],
    "trades": [50 + i for i in range(100)]
})

client = GCSClient()

# Write to GCS
client.to_parquet(data, "market_data/aapl_minute_2023_01_01.parquet")
print("Data written to GCS")
```

### Reading DataFrames

```python
# Read from GCS
data = client.read_parquet("market_data/aapl_minute_2023_01_01.parquet")
print(f"Loaded {len(data)} rows")
print(data.head())
```

### Batch Operations

```python
import os
from pathlib import Path

def upload_directory(local_dir: str, gcs_prefix: str):
    """Upload all parquet files from local directory to GCS."""

    local_path = Path(local_dir)

    for parquet_file in local_path.glob("**/*.parquet"):
        # Create GCS path
        relative_path = parquet_file.relative_to(local_path)
        gcs_path = f"{gcs_prefix}/{relative_path}"

        # Read and upload
        data = pd.read_parquet(parquet_file)
        client.to_parquet(data, gcs_path)
        print(f"Uploaded {parquet_file} to {gcs_path}")

# Usage
upload_directory("data/historical", "historical_data")
```

## File Organization

### Recommended Structure

```
gs://your-bucket/
├── market_data/
│   ├── stocks/
│   │   ├── aapl/
│   │   │   ├── minute/
│   │   │   │   ├── 2023/
│   │   │   │   │   ├── 01/
│   │   │   │   │   │   ├── 2023-01-01.parquet
│   │   │   │   │   │   └── 2023-01-02.parquet
│   │   │   │   │   └── 02/
│   │   │   │   └── 2024/
│   │   └── msft/
│   │       └── ...
│   └── forex/
│       └── ...
├── signals/
│   ├── strategy_a/
│   └── strategy_b/
├── backtests/
│   ├── results/
│   └── reports/
└── models/
    ├── trained/
    └── artifacts/
```

### Path Utilities

```python
from trade_system_modules.storage.paths import hist_path

# Generate standardized paths
path = hist_path(
    vendor="polygon",
    asset="stocks",
    symbol="AAPL",
    year=2023
)
# Result: "polygon/stocks/AAPL/2023.parquet"

# Use with GCS client
client.to_parquet(data, path)
```

## Performance Optimization

### Compression Options

```python
# Default compression (snappy)
client.to_parquet(data, "data/file.parquet")

# Custom compression
data.to_parquet(
    "/tmp/temp.parquet",
    compression="gzip",  # gzip, snappy, lz4, zstd
    index=False
)

# Upload compressed file
import io
with open("/tmp/temp.parquet", "rb") as f:
    client.bucket.blob("data/file.parquet").upload_from_file(f)
```

### Chunked Uploads

```python
def upload_large_dataframe(df, path, chunk_size=100000):
    """Upload large DataFrame in chunks."""

    total_rows = len(df)

    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_path = f"{path}/part_{i//chunk_size:04d}.parquet"
        client.to_parquet(chunk, chunk_path)
        print(f"Uploaded chunk {i//chunk_size + 1}")

# Usage
upload_large_dataframe(large_data, "large_dataset")
```

### Parallel Operations

```python
import concurrent.futures
import threading

def parallel_upload(files, max_workers=4):
    """Upload multiple files in parallel."""

    def upload_file(file_info):
        local_path, gcs_path = file_info
        data = pd.read_parquet(local_path)
        client.to_parquet(data, gcs_path)
        return f"Uploaded {local_path}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(upload_file, files))

    for result in results:
        print(result)

# Usage
files_to_upload = [
    ("data/aapl.parquet", "stocks/aapl.parquet"),
    ("data/msft.parquet", "stocks/msft.parquet"),
]
parallel_upload(files_to_upload)
```

## Error Handling

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import google.api_core.exceptions

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.DeadlineExceeded,
    ))
)
def reliable_upload(data, path):
    """Upload with automatic retries."""
    client.to_parquet(data, path)

# Usage
try:
    reliable_upload(data, "data/file.parquet")
except Exception as e:
    print(f"Upload failed after retries: {e}")
```

### Error Types

```python
import google.api_core.exceptions

try:
    client.to_parquet(data, "data/file.parquet")
except google.api_core.exceptions.NotFound:
    print("Bucket not found")
except google.api_core.exceptions.Forbidden:
    print("Access denied")
except google.api_core.exceptions.BadRequest:
    print("Invalid request")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Metadata and Tagging

### File Metadata

```python
# Add metadata when uploading
blob = client.bucket.blob("data/file.parquet")
blob.metadata = {
    "source": "polygon",
    "symbol": "AAPL",
    "date_range": "2023-01-01_2023-01-31",
    "created_at": pd.Timestamp.now().isoformat(),
    "row_count": len(data),
    "columns": list(data.columns)
}

# Upload with metadata
with io.BytesIO() as buffer:
    data.to_parquet(buffer, index=False)
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type="application/parquet")
```

### Listing Files

```python
# List all files in a prefix
blobs = list(client.bucket.list_blobs(prefix="market_data/stocks/"))

for blob in blobs:
    print(f"Name: {blob.name}")
    print(f"Size: {blob.size} bytes")
    print(f"Created: {blob.time_created}")
    print(f"Metadata: {blob.metadata}")
    print("---")
```

## Cost Optimization

### Storage Classes

```python
# Standard storage (default)
blob = client.bucket.blob("data/file.parquet")
blob.storage_class = "STANDARD"

# Nearline for infrequent access
blob.storage_class = "NEARLINE"

# Coldline for archival
blob.storage_class = "COLDLINE"

# Archive for long-term storage
blob.storage_class = "ARCHIVE"
```

### Lifecycle Management

```python
from google.cloud.storage import Lifecycle

# Set lifecycle rules
bucket = client.bucket
bucket.lifecycle_rules = [
    Lifecycle.Rule(
        action=Lifecycle.Action(type="Delete"),
        condition=Lifecycle.Condition(age=365)  # Delete after 1 year
    )
]
bucket.patch()
```

## Security

### Access Control

```python
# Make file public
blob = client.bucket.blob("public/data.parquet")
blob.make_public()

# Generate signed URL
signed_url = blob.generate_signed_url(
    version="v4",
    expiration=3600,  # 1 hour
    method="GET"
)

print(f"Signed URL: {signed_url}")
```

### Encryption

```python
# Server-side encryption (default)
blob = client.bucket.blob("encrypted/data.parquet")
# GCS automatically encrypts data at rest

# Customer-managed encryption keys
from google.cloud.storage import CustomerEncryption

encryption_key = CustomerEncryption(
    encryption_key=b"your-32-byte-encryption-key"
)

blob = client.bucket.blob("encrypted/data.parquet")
with io.BytesIO() as buffer:
    data.to_parquet(buffer, index=False)
    buffer.seek(0)
    blob.upload_from_file(buffer, encryption=encryption_key)
```

## Monitoring and Logging

### Operation Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggedGCSClient(GCSClient):
    """GCS client with enhanced logging."""

    def to_parquet(self, df, path):
        """Upload with logging."""
        start_time = pd.Timestamp.now()
        row_count = len(df)
        size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        logger.info(f"Starting upload of {row_count} rows ({size_mb:.2f} MB) to {path}")

        try:
            super().to_parquet(df, path)
            duration = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(f"Upload completed in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    def read_parquet(self, path):
        """Download with logging."""
        start_time = pd.Timestamp.now()

        logger.info(f"Starting download from {path}")

        try:
            data = super().read_parquet(path)
            duration = (pd.Timestamp.now() - start_time).total_seconds()
            size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"Download completed in {duration:.2f}s ({size_mb:.2f} MB)")
            return data
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
```

### Performance Metrics

```python
import time

def benchmark_gcs_operations(client, data_sizes):
    """Benchmark GCS operations for different data sizes."""

    results = []

    for size in data_sizes:
        # Generate test data
        data = pd.DataFrame({
            "ts": pd.date_range("2023-01-01", periods=size, freq="1min"),
            "price": np.random.randn(size).cumsum() + 100
        })

        # Benchmark upload
        start_time = time.time()
        client.to_parquet(data, f"benchmark/test_{size}.parquet")
        upload_time = time.time() - start_time

        # Benchmark download
        start_time = time.time()
        loaded_data = client.read_parquet(f"benchmark/test_{size}.parquet")
        download_time = time.time() - start_time

        results.append({
            "size": size,
            "upload_time": upload_time,
            "download_time": download_time,
            "upload_speed": size / upload_time,
            "download_speed": size / download_time
        })

    return pd.DataFrame(results)

# Usage
sizes = [1000, 10000, 100000]
benchmark_results = benchmark_gcs_operations(client, sizes)
print(benchmark_results)
```

## Best Practices

1. **Use appropriate storage classes** for cost optimization
2. **Implement retry logic** for reliable operations
3. **Batch operations** when possible
4. **Monitor costs** and usage patterns
5. **Secure sensitive data** with encryption
6. **Log operations** for debugging and monitoring
7. **Test with realistic data sizes**
8. **Use lifecycle policies** for automatic cleanup

## Integration Examples

### With Data Processing Pipeline

```python
def process_and_store_data(symbol, start_date, end_date):
    """Complete data processing pipeline."""

    # Get raw data
    from trade_system_modules import get_agg_minute
    data = get_agg_minute(symbol, start_date, end_date)

    # Process data
    data = process_market_data(data)

    # Store in GCS
    path = f"processed/{symbol}/{start_date}_{end_date}.parquet"
    client.to_parquet(data, path)

    return path

def process_market_data(data):
    """Apply data processing logic."""
    # Add technical indicators
    data["sma_20"] = data["close"].rolling(20).mean()
    data["returns"] = data["close"].pct_change()

    return data
```

### With Machine Learning Workflow

```python
def train_and_store_model(features, target, model_path):
    """Train ML model and store artifacts."""

    # Train model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100)
    model.fit(features, target)

    # Save model to GCS
    import joblib
    import io

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    blob = client.bucket.blob(model_path)
    blob.upload_from_file(buffer, content_type="application/octet-stream")

    return model_path
```

## See Also

- [API Reference](api-reference.md) - GCSClient documentation
- [Configuration Guide](configuration.md) - GCS setup
- [Examples](examples/) - Storage usage examples
- [Troubleshooting](troubleshooting.md) - Common GCS issues