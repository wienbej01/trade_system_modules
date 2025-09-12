"""Polygon data adapter for minute aggregates with async parallel downloads."""

from __future__ import annotations

import asyncio
import aiohttp
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import pandas as pd
from pandas import DataFrame
from typing import List, Optional, Dict
from ..config.settings import settings

def daterange_days(start: datetime, end: datetime) -> List[datetime]:
    """Generate list of daily datetimes from start to end inclusive."""
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


async def fetch_json(session: aiohttp.ClientSession, endpoint: str, params: dict) -> dict:
    """Fetch JSON from Polygon API with retry."""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _fetch():
        url = f"https://api.polygon.io{endpoint}"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()
    return await _fetch()


async def fetch_day_aggregates(session: aiohttp.ClientSession, api_key: str, symbol: str, day: datetime) -> Optional[pd.DataFrame]:
    """Fetch minute aggregates for a single day."""
    day_str = day.strftime("%Y-%m-%d")
    endpoint = f"/v2/aggs/ticker/{symbol}/range/1/minute/{day_str}/{day_str}"
    js = await fetch_json(session, endpoint, {"apikey": api_key, "limit": 50000})
    results = js.get("results", [])
    if not results:
        return None
    df = pd.DataFrame(results)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("ts").sort_index()
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "n": "trades"
    })
    cols = ["open", "high", "low", "close", "volume", "trades"]
    return df[[c for c in cols if c in df.columns]]


async def get_agg_minute(symbol: str, start: str, end: str, concurrency: int = 20) -> DataFrame:
    """Get minute aggregates from Polygon using async parallel day-wise downloads."""
    api_key = settings.polygon_api_key
    if not api_key:
        raise ValueError("Polygon API key required in settings.")
    
    start_dt = pd.Timestamp(start).tz_localize("America/New_York").to_pydatetime()
    end_dt = pd.Timestamp(end).tz_localize("America/New_York").to_pydatetime()
    
    days = daterange_days(start_dt, end_dt)
    if not days:
        return DataFrame()
    
    connector = aiohttp.TCPConnector(limit=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        all_dfs = []
        for i in range(0, len(days), concurrency):
            chunk = days[i:i + concurrency]
            tasks = [fetch_day_aggregates(session, api_key, symbol, d) for d in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception) or res is None:
                    continue
                all_dfs.append(res)
        
        if not all_dfs:
            return DataFrame()
        
        combined = pd.concat(all_dfs, axis=0)
        combined.reset_index(inplace=True)
        combined.rename(columns={"index": "ts"}, inplace=True)
        combined = combined[["ts", "open", "high", "low", "close", "volume", "trades"]]
        return combined.sort_values("ts").reset_index(drop=True)


async def get_agg_minute_batch(
    symbols: List[str], start: str, end: str, concurrency: int = 20
) -> Dict[str, DataFrame]:
    """Get minute aggregates for multiple symbols using async parallel downloads."""
    api_key = settings.polygon_api_key
    if not api_key:
        raise ValueError("Polygon API key required in settings.")
    
    start_dt = pd.Timestamp(start).tz_localize("America/New_York").to_pydatetime()
    end_dt = pd.Timestamp(end).tz_localize("America/New_York").to_pydatetime()
    days = daterange_days(start_dt, end_dt)
    if not days:
        return {sym: DataFrame() for sym in symbols}
    
    connector = aiohttp.TCPConnector(limit=concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=60)
    results = {}
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i in range(0, len(symbols), concurrency):
            chunk_symbols = symbols[i:i + concurrency]
            tasks = []
            for sym in chunk_symbols:
                sym_tasks = [fetch_day_aggregates(session, api_key, sym, d) for d in days]
                tasks.append(asyncio.create_task(asyncio.gather(*sym_tasks, return_exceptions=True)))
            
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, sym_result in zip(chunk_symbols, chunk_results):
                if isinstance(sym_result, Exception):
                    results[sym] = DataFrame()
                    continue
                all_dfs = []
                for res in sym_result:
                    if isinstance(res, Exception) or res is None:
                        continue
                    all_dfs.append(res)
                if not all_dfs:
                    results[sym] = DataFrame()
                    continue
                combined = pd.concat(all_dfs, axis=0)
                combined.reset_index(inplace=True)
                combined.rename(columns={"index": "ts"}, inplace=True)
                combined = combined[["ts", "open", "high", "low", "close", "volume", "trades"]]
                results[sym] = combined.sort_values("ts").reset_index(drop=True)
    
    return results