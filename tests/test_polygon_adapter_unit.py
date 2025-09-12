"""Unit tests for polygon_adapter with offline mocks."""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from pandas import DataFrame, Timestamp
from trade_system_modules.data.polygon_adapter import get_agg_minute

@pytest.fixture
def mock_aio_response():
    """Mock response from aiohttp ClientSession.get."""
    mock = AsyncMock()
    mock.json.return_value = {
        "results": [
            {"t": 1672531200000, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000, "n": 10}
        ]
    }
    mock.raise_for_status.return_value = None
    return mock

@pytest.mark.asyncio
async def test_get_agg_minute(mocker, mock_aio_response):
    """Test get_agg_minute with mocked API call."""
    with patch("trade_system_modules.data.polygon_adapter.aiohttp.ClientSession.get", new_callable=AsyncMock) as mocked_get:
        instance = AsyncMock()
        instance.get.return_value = mock_aio_response
        mocked_get.return_value.__aenter__.return_value = instance
        mocked_get.return_value.__aexit__.return_value = None
        
        df = await get_agg_minute("AAPL", "2023-01-01", "2023-01-01")
        
        assert not df.empty
        assert len(df) == 1
        assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume", "trades"]
        assert str(df["ts"].iloc[0]) == "2023-01-01 05:00:00-05:00"  # NY timezone
        assert df["open"].iloc[0] == 100.0
        assert df["volume"].iloc[0] == 1000
        assert df["trades"].iloc[0] == 10

@pytest.mark.asyncio
async def test_get_agg_minute_no_results(mocker):
    """Test get_agg_minute with no results."""
    mock_no_results = AsyncMock()
    mock_no_results.json.return_value = {"status": "OK", "results": []}
    mock_no_results.raise_for_status.return_value = None
    
    with patch("trade_system_modules.data.polygon_adapter.aiohttp.ClientSession.get", new_callable=AsyncMock) as mocked_get:
        instance = AsyncMock()
        instance.get.return_value = mock_no_results
        mocked_get.return_value.__aenter__.return_value = instance
        mocked_get.return_value.__aexit__.return_value = None
        
        df = await get_agg_minute("INVALID", "2023-01-01", "2023-01-01")
        
        assert df.empty

@pytest.mark.asyncio
async def test_get_agg_minute_error(mocker):
    """Test get_agg_minute with API error."""
    mock_error = AsyncMock()
    mock_error.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=None, history=None, status=500,
        message="API Error", headers=None
    )
    
    with patch("trade_system_modules.data.polygon_adapter.aiohttp.ClientSession.get", new_callable=AsyncMock) as mocked_get:
        instance = AsyncMock()
        instance.get.return_value = mock_error
        mocked_get.return_value.__aenter__.return_value = instance
        mocked_get.return_value.__aexit__.return_value = None
        
        with pytest.raises(aiohttp.ClientResponseError):
            await get_agg_minute("AAPL", "2023-01-01", "2023-01-01")