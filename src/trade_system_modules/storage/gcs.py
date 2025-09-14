"""GCS storage client for parquet read/write."""

from __future__ import annotations

import pandas as pd
import io
from datetime import datetime
from google.cloud import storage
from typing import Optional, List, Tuple
from ..config.settings import settings

class GCSClient:
    """Client for GCS operations."""
    
    def __init__(self, bucket: Optional[str] = None):
        self.bucket_name = bucket or settings.gcs_bucket
        self.client = storage.Client(project=settings.gcp_project)
        self.bucket = self.client.bucket(self.bucket_name)
    
    def to_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Write DataFrame to parquet in GCS."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        blob = self.bucket.blob(path)
        blob.upload_from_string(buffer.read(), content_type="application/parquet")
    
    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read parquet from GCS."""
        blob = self.bucket.blob(path)
        data = blob.download_as_bytes()
        buffer = io.BytesIO(data)
        return pd.read_parquet(buffer)

    def exists(self, path: str) -> bool:
        """Check if path exists in GCS."""
        blob = self.bucket.blob(path)
        return blob.exists()

    @staticmethod
    def monthly_path(symbol: str, year: int, month: int) -> str:
        """Generate monthly path for stocks."""
        return f"stocks/{symbol}/{year}/{month:02d}.parquet"

    def get_delta_ranges(self, path: str, target_start: datetime, target_end: datetime) -> List[Tuple[datetime, datetime]]:
        """Get missing date ranges for delta download."""
        if not self.exists(path):
            return [(target_start, target_end)]

        df = self.read_parquet(path)
        if df.empty:
            return [(target_start, target_end)]

        df = df.set_index('ts').sort_index()
        idx = df.index

        # Find gaps > 1 minute
        diffs = idx[1:] - idx[:-1]
        gap_mask = diffs > pd.Timedelta('1min')

        missing_ranges = []

        # Gap from target_start to first ts
        if idx[0] > target_start:
            missing_ranges.append((target_start, idx[0] - pd.Timedelta('1min')))

        # Gaps between consecutive
        gap_starts = idx[:-1][gap_mask]
        gap_ends = idx[1:][gap_mask]
        for start, end in zip(gap_starts, gap_ends):
            missing_ranges.append((start + pd.Timedelta('1min'), end - pd.Timedelta('1min')))

        # Gap from last ts to target_end
        if idx[-1] < target_end:
            missing_ranges.append((idx[-1] + pd.Timedelta('1min'), target_end))

        return missing_ranges