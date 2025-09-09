"""Symbology resolution for instruments."""

from __future__ import annotations

from typing import Dict, Any, Optional
from ib_insync import Stock, Contract

# Stub mapping for stocks; extend for futures
STOCK_MAPPING: Dict[str, str] = {
    "AAPL": "AAPL",
    # Add more as needed
}

def resolve_instrument(key: str) -> Contract:
    """Resolve vendor symbol to IB Contract."""
    if key in STOCK_MAPPING:
        return Stock(key, "SMART", "USD")
    raise ValueError(f"Unknown instrument key: {key}")