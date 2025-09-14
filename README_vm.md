# Polygon S&P 500 Stocks Downloader - VM Setup and Run Instructions

This guide provides step-by-step instructions to deploy and run the Polygon data downloader on your Google VM for downloading S&P 500 minute aggregates from 2020-10-01 to present, stored as monthly Parquet files in GCS bucket `jwss_data_store`.

## Prerequisites

- Google VM with Ubuntu/Debian-based OS
- Python 3.10+ installed
- GCP project `llm-behavior-trading` with GCS access
- VM default service account has `Storage Object Admin` role for bucket `jwss_data_store`
- Internet access for Polygon API calls

## Files Required

Upload these files to your VM home directory (`~`):
- `polygon_stocks_downloader.py` - Main downloader script
- `download_stocks.sh` - Nohup runner script
- `requirements.txt` - Python dependencies
- `src/` directory (entire trade_system_modules package)

## Setup Steps

### 1. SSH to VM
```bash
gcloud compute ssh [vm-name] --project=llm-behavior-trading
```

### 2. Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential
```

### 3. Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

### 4. Verify GCS Access
```bash
# Test GCS access
python3 -c "
from google.cloud import storage
client = storage.Client(project='llm-behavior-trading')
buckets = list(client.list_buckets())
print('GCS access OK' if buckets else 'GCS access failed')
"
```

### 5. Make Scripts Executable
```bash
chmod +x download_stocks.sh
chmod +x polygon_stocks_downloader.py
```

## Running the Downloader

### Start Download (Background)
```bash
./download_stocks.sh
```

This will:
- Start the downloader in background with nohup
- Log output to `stocks_download_YYYYMMDD_HHMMSS.log`
- Save PID to `download.pid`

### Monitor Progress
```bash
# View live logs
tail -f stocks_download_*.log

# Check process status
ps aux | grep downloader

# View recent logs
tail -n 50 stocks_download_*.log
```

### Stop Download
```bash
# Graceful stop (allows checkpoint save)
kill $(cat download.pid)

# Force stop if needed
kill -9 $(cat download.pid)
```

## Data Structure

Downloaded data is stored in GCS as:
```
gs://jwss_data_store/stocks/{SYMBOL}/{YEAR}/{MONTH:02d}.parquet
```

Example:
- `gs://jwss_data_store/stocks/AAPL/2020/10.parquet` - AAPL October 2020
- `gs://jwss_data_store/stocks/MSFT/2024/09.parquet` - MSFT September 2024

Each Parquet file contains minute OHLCV data with columns:
- `ts`: Timestamp (America/New_York timezone)
- `open`, `high`, `low`, `close`: Prices
- `volume`: Volume
- `trades`: Number of trades

## Features

- **Delta Downloads**: Only downloads missing data gaps, not full months
- **Resume Capability**: Uses checkpoint file to resume interrupted runs
- **Async Parallel**: Downloads multiple days concurrently for speed
- **Error Handling**: Retries failed API calls, logs errors
- **Progress Tracking**: tqdm progress bars and detailed logging
- **Graceful Shutdown**: Handles SIGINT/SIGTERM for clean stops

## Troubleshooting

### Common Issues

1. **GCS Access Denied**
   - Ensure VM service account has `Storage Object Admin` role
   - Check project ID is `llm-behavior-trading`

2. **Polygon API Errors**
   - Verify API key is valid (unlimited stocks plan required)
   - Check rate limits (unlimited for stocks, but tickers API has limits)

3. **Memory Issues**
   - Large downloads may consume memory; monitor with `htop`
   - Reduce concurrency in script if needed

4. **Disk Space**
   - Downloaded data ~500GB+ for full S&P 500 history
   - Ensure sufficient VM disk space

### Logs and Debugging

- Main logs: `stocks_download_*.log`
- Checkpoint: `checkpoint.json` (tracks completed symbols)
- Symbol cache: `sp500_symbols.json`
- Downloader logs: `downloader.log`

### Restarting After Interrupt

The script automatically resumes from checkpoint. If needed, delete `checkpoint.json` to restart from beginning.

## Verification

After completion, verify data:
```bash
# List downloaded files
gsutil ls gs://jwss_data_store/stocks/AAPL/

# Check file contents
gsutil cat gs://jwss_data_store/stocks/AAPL/2020/10.parquet | head -c 1000

# Count total files
gsutil ls -r gs://jwss_data_store/stocks/ | wc -l
```

## Scheduling Monthly Runs

To run monthly, add to crontab:
```bash
crontab -e
# Add: 0 2 1 * * /home/user/download_stocks.sh  # Run 1st of each month at 2 AM
```

## API Limits and Costs

- **Polygon Stocks Plan**: Unlimited minute aggregates
- **Tickers API**: 5 requests/minute (one-time symbol fetch)
- **GCS Costs**: Storage (~$0.02/GB/month) + operations
- **VM Costs**: Compute time during downloads

## Support

For issues:
1. Check logs for error messages
2. Verify API key and GCS permissions
3. Test with single symbol first (modify script temporarily)