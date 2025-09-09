"""GCS storage client for parquet read/write."""

from __future__ import annotations

import pandas as pd
import io
from google.cloud import storage
from typing import Optional
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