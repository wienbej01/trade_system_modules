#!/usr/bin/env python3
"""Test script for Polygon stocks downloader components."""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytz

# Add src to path</search>
</search_and_replace>
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trade_system_modules.data.polygon_adapter import get_sp500_symbols, get_agg_minute_range
from trade_system_modules.storage.gcs import GCSClient


class TestStocksDownloader(unittest.TestCase):
    """Test cases for downloader components."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_key"
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.temp_dir, 'test_symbols.json')

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('trade_system_modules.data.polygon_adapter.fetch_json')
    def test_get_sp500_symbols(self, mock_fetch):
        """Test fetching S&P 500 symbols."""
        # Mock API response
        mock_fetch.return_value = {
            'tickers': [
                {'ticker': 'AAPL'},
                {'ticker': 'MSFT'},
                {'ticker': 'GOOGL'}
            ]
        }

        async def run_test():
            symbols = await get_sp500_symbols(self.api_key, self.cache_file)
            return symbols

        symbols = asyncio.run(run_test())

        self.assertEqual(len(symbols), 3)
        self.assertIn('AAPL', symbols)
        self.assertIn('MSFT', symbols)
        self.assertIn('GOOGL', symbols)

        # Check caching
        self.assertTrue(os.path.exists(self.cache_file))
        with open(self.cache_file) as f:
            cached = json.load(f)
        self.assertEqual(cached, symbols)

        # Test cache loading
        async def run_test2():
            symbols2 = await get_sp500_symbols(self.api_key, self.cache_file)
            return symbols2

        symbols2 = asyncio.run(run_test2())
        self.assertEqual(symbols, symbols2)
        # Should not call API again
        self.assertEqual(mock_fetch.call_count, 1)

    @patch('trade_system_modules.data.polygon_adapter.fetch_json')
    @patch('trade_system_modules.data.polygon_adapter.fetch_day_aggregates')
    @patch('trade_system_modules.data.polygon_adapter.settings')
    def test_get_agg_minute_range(self, mock_settings, mock_fetch_agg, mock_fetch_json):
        """Test fetching aggregates for date range."""
        # Mock settings
        mock_settings.polygon_api_key = 'test_key'

        # Mock day aggregates - return processed df with correct columns
        processed_df = pd.DataFrame({
            'ts': pd.to_datetime(['2021-01-01 00:00:00-05:00']),
            'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.5],
            'volume': [1000], 'trades': [10]
        })
        mock_fetch_agg.return_value = processed_df

        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)

        async def run_test():
            df = await get_agg_minute_range('AAPL', start, end)
            return df

        df = asyncio.run(run_test())

        self.assertFalse(df.empty)
        self.assertIn('ts', df.columns)
        self.assertIn('open', df.columns)
        self.assertEqual(len(df), 1)

    def test_monthly_path(self):
        """Test monthly path generation."""
        path = GCSClient.monthly_path('AAPL', 2020, 10)
        expected = 'stocks/AAPL/2020/10.parquet'
        self.assertEqual(path, expected)

    @patch('google.cloud.storage.Client')
    def test_gcs_delta_ranges_no_existing(self, mock_client):
        """Test delta ranges when no existing data."""
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value.exists.return_value = False

        gcs = GCSClient('test-bucket')
        start = datetime(2021, 1, 1)
        end = datetime(2021, 1, 2)

        ranges = gcs.get_delta_ranges('test/path.parquet', start, end)

        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0], (start, end))

    @patch('google.cloud.storage.Client')
    def test_gcs_delta_ranges_with_gaps(self, mock_client):
        """Test delta ranges with existing data and gaps."""
        # Mock existing data with gaps
        existing_df = pd.DataFrame({
            'ts': pd.to_datetime([
                '2021-01-01 09:30:00-05:00',  # Start
                '2021-01-01 09:31:00-05:00',  # +1 min
                # Gap here
                '2021-01-01 09:35:00-05:00',  # +4 min gap
                '2021-01-01 09:36:00-05:00',  # End
            ])
        })

        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value.exists.return_value = True

        # Mock read_parquet
        with patch.object(GCSClient, 'read_parquet', return_value=existing_df) as mock_read:
            gcs = GCSClient('test-bucket')
            # Make tz-aware
            ny_tz = pytz.timezone("America/New_York")
            start = datetime(2021, 1, 1, 9, 30, tzinfo=ny_tz)
            end = datetime(2021, 1, 1, 9, 40, tzinfo=ny_tz)

            ranges = gcs.get_delta_ranges('test/path.parquet', start, end)

            # Should have gaps: before first, between 31 and 35, after last
            self.assertGreater(len(ranges), 0)
            # Verify gap detection logic
            mock_read.assert_called_once()


if __name__ == '__main__':
    # Run tests with asyncio support
    unittest.main()