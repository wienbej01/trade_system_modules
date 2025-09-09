"""Unit tests for data.symbology."""

from trade_system_modules.data.symbology import resolve_instrument
from ib_insync import Stock

def test_resolve_instrument_stock():
    """Test resolve_instrument for known stock."""
    contract = resolve_instrument("AAPL")
    assert isinstance(contract, Stock)
    assert contract.symbol == "AAPL"
    assert contract.exchange == "SMART"
    assert contract.currency == "USD"

def test_resolve_instrument_unknown():
    """Test resolve_instrument for unknown key."""
    with pytest.raises(ValueError, match="Unknown instrument key: UNKNOWN"):
        resolve_instrument("UNKNOWN")