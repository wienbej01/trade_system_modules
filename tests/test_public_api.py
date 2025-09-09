"""Unit tests for public API imports."""

from trade_system_modules import Settings, GCSClient, get_agg_minute, IBLive, IBExec, resolve_instrument

def test_public_api_symbols():
    """Test public API symbols exist."""
    assert callable(get_agg_minute)
    assert callable(resolve_instrument)

def test_public_api_classes():
    """Test public API classes import."""
    assert Settings
    assert GCSClient
    assert IBLive
    assert IBExec