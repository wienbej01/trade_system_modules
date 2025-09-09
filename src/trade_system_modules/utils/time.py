"""Time utilities with UTC focus."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

def utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.now().replace(tzinfo=timezone.utc)