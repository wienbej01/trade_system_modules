"""Unit tests for polygon_adapter with offline mocks."""

import pytest
import requests
from unittest.mock import Mock
from pandas import DataFrame, Timestamp
from trade_system_modules.data.polygon_adapter import get_agg_minute

@pytest.fixture
def mock_response():
    """Mock response from requests.get."""
    mock = Mock()
    mock.json.return_value = {
        "results": [
            {"t": 1672531200000, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000, "n": 10}
        ]
    }
    mock.raise_for_status.return_value = None
    return mock

def test_get_agg_minute(mocker, mock_response):
    """Test get_agg_minute with mocked API call."""
    mocker.patch("requests.get", return_value=mock_response)
    
    df = get_agg_minute("AAPL", "2023-01-01", "2023-01-01")
    
    assert not df.empty
    assert len(df) == 1
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume", "trades"]
    assert df["ts"].iloc[0] == Timestamp("2023-01-01 00:00:00", tz="UTC")
    assert df["open"].iloc[0] == 100.0
    assert df["volume"].iloc[0] == 1000
    assert df["trades"].iloc[0] == 10

def test_get_agg_minute_no_results(mocker):
    """Test get_agg_minute with no results."""
    mock_response = Mock()
    mock_response.json.return_value = {"status": "OK", "results": []}
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)
    
    df = get_agg_minute("INVALID", "2023-01-01", "2023-01-01")
    
    assert df.empty

def test_get_agg_minute_error(mocker):
    """Test get_agg_minute with API error."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error")
    mocker.patch("requests.get", return_value=mock_response)
    
    with pytest.raises(requests.exceptions.HTTPError):
        get_agg_minute("AAPL", "2023-01-01", "2023-01-01")