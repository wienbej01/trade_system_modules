"""Schemas for bars data."""

from __future__ import annotations

import pandas as pd
from typing import Any

BAR_COLUMNS = ["ts", "open", "high", "low", "close", "volume", "trades"]

def ensure_bar_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has bar schema."""
    for col in BAR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df