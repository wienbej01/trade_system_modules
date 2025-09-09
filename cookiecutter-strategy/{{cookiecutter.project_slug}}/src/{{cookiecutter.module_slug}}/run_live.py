"""Live run script for {{cookiecutter.project_name}}."""

from __future__ import annotations

from typing import Any
import os
from trade_system_modules.config.settings import settings
from trade_system_modules.data.ibkr_live import IBLive
from trade_system_modules.storage import GCSClient
from trade_system_modules.storage.paths import live_path
from trade_system_modules.utils.logging import setup
from trade_system_modules.utils.time import utcnow

def run_live(symbol: str = "AAPL"):
    """Run live: connect IB, stream bars, write to GCS."""
    setup("INFO")
    
    ib = IBLive()
    ib.connect()
    
    date_str = utcnow().strftime("%Y%m%d")
    path = live_path(symbol, date_str)
    
    def on_bar_update(bars, has_new_bar):
        if has_new_bar:
            df = bars.df  # ib_insync provides df
            client = GCSClient()
            client.to_parquet(df, path)
            print(f"Live bars written to {path}")
    
    ib.stream_rt_bars(symbol, callback=on_bar_update)
    
    ib.ib.run()  # Keep running

if __name__ == "__main__":
    symbol = os.getenv("SYMBOL", "AAPL")
    run_live(symbol)