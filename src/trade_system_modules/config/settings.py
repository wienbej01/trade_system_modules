"""Configuration settings using Pydantic."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Trade system settings."""
    
    gcs_bucket: Optional[str] = None
    gcp_project: Optional[str] = None
    polygon_api_key: Optional[str] = None
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    
    class Config:
        env_file = ".env"

settings = Settings()