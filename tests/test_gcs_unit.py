"""Unit tests for storage.gcs with offline mocks."""

import pytest
import io
from unittest.mock import Mock, patch
from pandas import DataFrame
from trade_system_modules.config.settings import Settings
from trade_system_modules.storage.gcs import GCSClient

@pytest.fixture
def mock_settings():
    """Mock settings with bucket and project."""
    settings = Mock(spec=Settings)
    settings.gcs_bucket = "test-bucket"
    settings.gcp_project = "test-project"
    return settings

@pytest.fixture
def mock_client():
    """Mock GCS client."""
    mock_client = Mock()
    mock_bucket = Mock()
    mock_blob = Mock()
    mock_blob.download_as_bytes.return_value = b"mock_parquet_data"
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket
    return mock_client

def test_gcs_client_init(mock_settings):
    """Test GCSClient initialization."""
    with patch("trade_system_modules.storage.gcs.settings", mock_settings):
        client = GCSClient()
        assert client.bucket_name == "test-bucket"

def test_to_parquet(mock_settings, mock_client):
    """Test to_parquet with mock upload."""
    df = DataFrame({"ts": [1], "open": [100.0]})
    mock_blob = Mock()
    mock_bucket = Mock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = Mock()
    mock_client.bucket.return_value = mock_bucket

    # Mock pandas to_parquet to avoid complex parquet operations
    with patch("trade_system_modules.storage.gcs.settings", mock_settings), \
         patch("trade_system_modules.storage.gcs.storage.Client", return_value=mock_client), \
         patch("pandas.DataFrame.to_parquet") as mock_to_parquet:
        client = GCSClient()
        client.to_parquet(df, "test/path.parquet")
        mock_to_parquet.assert_called_once()
        mock_blob.upload_from_string.assert_called_once()

def test_read_parquet(mock_settings, mock_client):
    """Test read_parquet with mock download."""
    expected_df = DataFrame({"ts": [1], "open": [100.0]})

    with patch("trade_system_modules.storage.gcs.settings", mock_settings), \
         patch("trade_system_modules.storage.gcs.storage.Client", return_value=mock_client), \
         patch("pandas.read_parquet", return_value=expected_df) as mock_read_parquet:
        client = GCSClient()
        df = client.read_parquet("test/path.parquet")
        mock_client.bucket.return_value.blob.return_value.download_as_bytes.assert_called_once()
        mock_read_parquet.assert_called_once()
        assert not df.empty
        assert len(df) == 1