"""Deterministic storage paths for historical and live data."""

from __future__ import annotations

from typing import Any
from datetime import datetime

def hist_path(vendor: str, asset: str, symbol: str, year: int) -> str:
    """Path for historical data."""
    return f"{vendor}/{asset}/{symbol}/{year}.parquet"

def live_path(symbol: str, date_str: str) -> str:
    """Path for live data."""
    return f"live/{symbol}/{date_str}.parquet"