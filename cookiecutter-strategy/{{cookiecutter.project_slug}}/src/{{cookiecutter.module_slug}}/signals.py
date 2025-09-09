"""Signal generation for {{cookiecutter.project_name}}."""

from __future__ import annotations

import pandas as pd
from typing import Any
from trade_system_modules.data.polygon_adapter import get_agg_minute
from trade_system_modules.schemas.bars import ensure_bar_schema

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate buy/sell signals from bars DataFrame."""
    df = ensure_bar_schema(df)
    df["signal"] = 0
    df.loc[df["close"] > df["open"], "signal"] = 1  # Simple: buy if green
    df.loc[df["close"] < df["open"], "signal"] = -1  # Sell if red
    return df