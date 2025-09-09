"""IBKR execution client."""

from __future__ import annotations

from ib_insync import IB, Order, Trade
from typing import List, Optional
from ..config.settings import settings
from ..data.symbology import resolve_instrument

class IBExec:
    """IBKR execution handler."""
    
    def __init__(self):
        self.ib = IB()
        self.connect()
    
    def connect(self):
        """Connect to IBKR."""
        self.ib.connect(settings.ib_host, settings.ib_port, clientId=settings.ib_client_id)
    
    def place_market(self, symbol: str, action: str, quantity: int) -> Trade:
        """Place market order."""
        contract = resolve_instrument(symbol)
        order = Order('MKT', quantity, action, account=None)
        trade = self.ib.placeOrder(contract, order)
        return trade
    
    def place_limit(self, symbol: str, action: str, quantity: int, limit_price: float) -> Trade:
        """Place limit order."""
        contract = resolve_instrument(symbol)
        order = Order('LMT', quantity, action, limitPrice=limit_price)
        trade = self.ib.placeOrder(contract, order)
        return trade
    
    def cancel(self, order_id: int) -> bool:
        """Cancel order."""
        return self.ib.cancelOrder(order_id)
    
    def get_positions(self) -> List[Trade]:
        """Get current positions."""
        return self.ib.positions()