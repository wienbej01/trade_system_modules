"""Unit tests for schemas.bars."""

import pytest
import pandas as pd
from pandas import DataFrame, Timestamp
from trade_system_modules.schemas.bars import BAR_COLUMNS, ensure_bar_schema

def test_bar_columns():
    """Test BAR_COLUMNS constant."""
    assert BAR_COLUMNS == ["ts", "open", "high", "low", "close", "volume", "trades"]

def test_ensure_bar_schema_full():
    """Test ensure_bar_schema with full columns."""
    df = DataFrame({
        "ts": [Timestamp("2023-01-01")],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [1000],
        "trades": [10]
    })
    result = ensure_bar_schema(df)
    assert list(result.columns) == BAR_COLUMNS
    assert str(result["ts"].dt.tz) == "UTC"
    assert result.equals(df)  # No change if already correct

def test_ensure_bar_schema_missing_columns():
    """Test ensure_bar_schema with missing columns."""
    df = DataFrame({"ts": [Timestamp("2023-01-01")]})
    result = ensure_bar_schema(df)
    assert list(result.columns) == BAR_COLUMNS
    for col in BAR_COLUMNS[1:]:
        assert result[col].isna().all()
    assert str(result["ts"].dt.tz) == "UTC"

def test_ensure_bar_schema_non_utc_ts():
    """Test ensure_bar_schema converts non-UTC ts to UTC."""
    df = DataFrame({
        "ts": [Timestamp("2023-01-01 00:00:00", tz="America/New_York")],
        "open": [100.0]
    })
    result = ensure_bar_schema(df)
    assert str(result["ts"].dt.tz) == "UTC"