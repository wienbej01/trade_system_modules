#!/usr/bin/env python3
"""Polygon S&P 500 stocks downloader with delta updates to GCS."""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
import io

import pandas as pd
from pytz import timezone
from tqdm import tqdm
import aiohttp

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trade_system_modules.data.polygon_adapter import get_sp500_symbols, get_agg_minute_range
from trade_system_modules.storage.gcs import GCSClient
from trade_system_modules.utils.time import utcnow
from trade_system_modules.config.settings import settings

API_KEY = os.getenv("POLYGON_API_KEY") or "ZBxeJYOn0_e0UcPgEYLA90CQ9S28_EfU"
BUCKET = "jwss_data_store"
NY_TZ = timezone("America/New_York")
START_DATE = pd.Timestamp("2020-10-01", tz=NY_TZ)
CACHE_FILE = 'sp500_symbols.json'
CHECKPOINT_FILE = 'checkpoint.json'

# Global flag for graceful shutdown
shutdown = False

def signal_handler(signum, frame):
    global shutdown
    logging.info("Received signal, shutting down gracefully...")
    shutdown = True


async def fetch_sp500_from_wikipedia(cache_file: str = 'sp500_symbols.json') -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia and cache locally."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    connector = aiohttp.TCPConnector()
    timeout = aiohttp.ClientTimeout(total=60)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    }
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            html = await resp.text()
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        return []
    df = tables[0]
    tickers = (
        df["Symbol"].astype(str).str.strip().str.replace(" ", "", regex=False).tolist()
    )
    tickers = sorted({t for t in tickers if t and t.lower() != "nan"})
    with open(cache_file, "w") as f:
        json.dump(tickers, f)
    return tickers


async def main():
    global shutdown
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('downloader.log', mode='a')
        ]
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Ensure API key is set in settings for downstream adapter calls
        if not settings.polygon_api_key:
            settings.polygon_api_key = API_KEY
        logging.info("Polygon API key configured for adapter (source=%s)", "env" if os.getenv("POLYGON_API_KEY") else "constant")

        # Fetch S&P 500 symbols from Polygon (indices constituents or large-cap proxy)
        logging.info("Fetching S&P 500 symbols from Polygon...")
        symbols = await get_sp500_symbols(API_KEY, CACHE_FILE)
        logging.info("Fetched %d S&P 500 symbols", len(symbols))
        if not symbols:
            logging.error("No symbols returned from Polygon; aborting.")
            return

        gcs = GCSClient(BUCKET)

        # Current end date in NY tz
        now_utc = utcnow()
        end_date = now_utc.astimezone(NY_TZ)
        logging.info(f"Time check: now_utc={now_utc.isoformat()}, end_date_ny={end_date.isoformat()}")

        # Generate monthly periods
        months = pd.date_range(start=START_DATE, end=end_date, freq='MS')

        # Load checkpoint
        checkpoint = {}
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE) as f:
                checkpoint = json.load(f)

        total_downloaded = 0

        with tqdm(total=len(symbols) * len(months), desc="Processing") as pbar:
            for symbol in symbols:
                if shutdown:
                    break
                if symbol in checkpoint and checkpoint[symbol]:
                    pbar.update(len(months))
                    continue

                for month in months:
                    if shutdown:
                        break

                    year = month.year
                    month_num = month.month
                    path = GCSClient.monthly_path(symbol, year, month_num)

                    # Month start and end
                    month_start = month
                    month_end = (month + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
                    if month_end > end_date:
                        month_end = end_date

                    # Get missing ranges
                    ranges = gcs.get_delta_ranges(path, month_start.to_pydatetime(), month_end.to_pydatetime())

                    if not ranges:
                        pbar.update(1)
                        continue

                    all_data = []
                    # Load existing if present
                    if gcs.exists(path):
                        existing = gcs.read_parquet(path)
                        all_data.append(existing)

                    # Download missing ranges
                    for start, end in ranges:
                        if shutdown:
                            break
                        logging.info(f"Downloading {symbol} {year}-{month_num:02d} from {start} to {end}")
                        df = await get_agg_minute_range(symbol, start, end)
                        if not df.empty:
                            all_data.append(df)
                            total_downloaded += len(df)

                    if all_data:
                        # Merge and dedup
                        combined = pd.concat(all_data, ignore_index=True)
                        combined = combined.drop_duplicates(subset='ts').sort_values('ts').reset_index(drop=True)
                        gcs.to_parquet(combined, path)
                        logging.info(f"Updated {symbol} {year}-{month_num:02d} with {len(combined)} records")

                    pbar.update(1)

                # Checkpoint every symbol
                checkpoint[symbol] = True
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f)

        logging.info(f"Download complete. Total records downloaded: {total_downloaded}")

    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())