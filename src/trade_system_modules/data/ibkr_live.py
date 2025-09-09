"""IBKR live data adapter using ib_insync."""

from __future__ import annotations

from ib_insync import IB, BarData
from typing import Callable, Optional
from ..config.settings import settings

class IBLive:
    """IBKR live streaming client."""
    
    def __init__(self):
        self.ib = IB()
    
    def connect(self, host: Optional[str] = None, port: Optional[int] = None, client_id: Optional[int] = None):
        """Connect to IBKR."""
        host = host or settings.ib_host
        port = port or settings.ib_port
        client_id = client_id or settings.ib_client_id
        self.ib.connect(host, port, clientId=client_id)
    
    def stream_rt_bars(self, symbol: str, duration: str = "1 D", what: str = "TRADES", useRth: bool = False, callback: Optional[Callable] = None):
        """Stream real-time bars."""
        contract = self.ib.Stock(symbol, "SMART", "USD")
        bars = self.ib.reqRealTimeBars(contract, 5, what, useRth)
        if callback:
            bars.updateEvent += callback
        return bars