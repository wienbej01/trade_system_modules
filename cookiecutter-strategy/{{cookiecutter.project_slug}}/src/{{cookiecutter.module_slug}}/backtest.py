"""Backtest script for {{cookiecutter.project_name}}."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
import os
from trade_system_modules.config.settings import settings
from trade_system_modules.data.polygon_adapter import get_agg_minute
from trade_system_modules.storage import GCSClient
from trade_system_modules.storage.paths import hist_path
from trade_system_modules.utils.logging import setup

def run_backtest(symbol: str = "AAPL", weeks: int = 1):
    """Run backtest: pull bars, write to GCS."""
    setup("INFO")
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(weeks=weeks)
    
    df = get_agg_minute(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    year = start_date.year
    path = hist_path("polygon", "equity", symbol, year)
    
    client = GCSClient()
    client.to_parquet(df, path)
    
    print(f"Backtest data written to {path}")

if __name__ == "__main__":
    symbol = os.getenv("SYMBOL", "AAPL")
    run_backtest(symbol)