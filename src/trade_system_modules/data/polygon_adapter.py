"""Polygon data adapter for minute aggregates."""

from __future__ import annotations

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import pandas as pd
from pandas import DataFrame
from typing import Any
from ..config.settings import settings

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_not_exception_type(requests.exceptions.HTTPError))
def get_agg_minute(symbol: str, start: str, end: str) -> DataFrame:
    """Get minute aggregates from Polygon."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}"
    params = {"apikey": settings.polygon_api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if "results" not in data:
        return DataFrame()
    
    df = DataFrame(data["results"])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df[["ts", "o", "h", "l", "c", "v", "n"]]
    df.columns = ["ts", "open", "high", "low", "close", "volume", "trades"]
    return df