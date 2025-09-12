"""IBKR execution client with advanced order management."""

from __future__ import annotations

from ib_insync import IB, Order, Trade
from typing import List, Optional, Dict, Any
from ..config.settings import settings
from ..data.symbology import resolve_instrument

class IBExec:
    """IBKR execution handler with advanced order management."""

    def __init__(self):
        self.ib = IB()
        self.connect()

    def connect(self):
        """Connect to IBKR."""
        self.ib.connect(settings.ib_host, settings.ib_port, clientId=settings.ib_client_id)

    def place_market(self, symbol: str, action: str, quantity: int) -> Trade:
        """Place market order."""
        contract = resolve_instrument(symbol)
        order = Order(action=action, totalQuantity=quantity, orderType='MKT')
        trade = self.ib.placeOrder(contract, order)
        return trade

    def place_limit(self, symbol: str, action: str, quantity: int, limit_price: float) -> Trade:
        """Place limit order."""
        contract = resolve_instrument(symbol)
        order = Order(action=action, totalQuantity=quantity, orderType='LMT', lmtPrice=limit_price)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def place_stop(self, symbol: str, action: str, quantity: int, stop_price: float) -> Trade:
        """Place stop order."""
        contract = resolve_instrument(symbol)
        order = Order(action=action, totalQuantity=quantity, orderType='STP', auxPrice=stop_price)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def place_stop_limit(self, symbol: str, action: str, quantity: int, limit_price: float, stop_price: float) -> Trade:
        """Place stop-limit order."""
        contract = resolve_instrument(symbol)
        order = Order(action=action, totalQuantity=quantity, orderType='STP LMT',
                     lmtPrice=limit_price, auxPrice=stop_price)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def place_trailing_stop(self, symbol: str, action: str, quantity: int, trailing_percent: float = None, trailing_amount: float = None) -> Trade:
        """Place trailing stop order."""
        contract = resolve_instrument(symbol)
        order = Order(action=action, totalQuantity=quantity, orderType='TRAIL')

        if trailing_percent is not None:
            order.trailingPercent = trailing_percent
        elif trailing_amount is not None:
            order.auxPrice = trailing_amount  # Trailing amount

        trade = self.ib.placeOrder(contract, order)
        return trade

    def modify_order(self, order_id: int, quantity: int = None, limit_price: float = None, stop_price: float = None) -> bool:
        """Modify existing order."""
        try:
            # Find the order
            target_order = None
            for order in self.ib.openOrders():
                if order.orderId == order_id:
                    target_order = order
                    break

            if not target_order:
                return False

            # Create modified order
            modified_order = Order(
                action=target_order.action,
                totalQuantity=quantity or target_order.totalQuantity,
                orderType=target_order.orderType,
                orderId=order_id
            )

            # Set appropriate price fields based on order type
            if limit_price is not None and target_order.orderType in ['LMT', 'STP LMT']:
                modified_order.lmtPrice = limit_price

            if stop_price is not None and target_order.orderType in ['STP', 'STP LMT']:
                modified_order.auxPrice = stop_price

            # Place modified order
            self.ib.placeOrder(target_order.contract, modified_order)
            return True

        except Exception:
            return False

    def cancel(self, order_id: int) -> bool:
        """Cancel order by ID."""
        try:
            # Find the order object with the given ID
            for order in self.ib.openOrders():
                if order.orderId == order_id:
                    return self.ib.cancelOrder(order)
            return False  # Order not found
        except Exception:
            return False

    def get_positions(self) -> List[Trade]:
        """Get current positions."""
        return self.ib.positions()

    def get_account_summary(self) -> List[Dict[str, Any]]:
        """Get detailed account summary."""
        try:
            summary = self.ib.accountSummary()
            return [{"tag": item.tag, "value": item.value, "currency": item.currency} for item in summary]
        except Exception:
            return []

    def get_pnl(self) -> Dict[str, Any]:
        """Get P&L information."""
        try:
            pnl = self.ib.pnl()
            return {
                "dailyPnL": pnl.dailyPnL,
                "unrealizedPnL": pnl.unrealizedPnL,
                "realizedPnL": pnl.realizedPnL,
                "totalPnL": pnl.dailyPnL + pnl.unrealizedPnL + pnl.realizedPnL
            }
        except Exception:
            return {"error": "P&L data not available"}

    def get_account_values(self) -> Dict[str, Any]:
        """Get account values and equity information."""
        try:
            values = {}
            summary = self.ib.accountSummary()

            for item in summary:
                if item.tag in ["TotalCashValue", "NetLiquidation", "EquityWithLoanValue", "BuyingPower"]:
                    values[item.tag] = float(item.value) if item.value else 0.0

            return values
        except Exception:
            return {"error": "Account values not available"}

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders with details from both openOrders and openTrades."""
        try:
            order_details = []

            # Method 1: Check openOrders()
            try:
                orders = self.ib.openOrders()
                for order in orders:
                    order_info = {
                        "orderId": order.orderId,
                        "symbol": order.contract.symbol if hasattr(order.contract, 'symbol') else 'N/A',
                        "action": order.action,
                        "quantity": order.totalQuantity,
                        "orderType": order.orderType,
                        "status": order.orderStatus.status if hasattr(order, 'orderStatus') else 'Unknown',
                        "source": "openOrders"
                    }

                    # Add price information based on order type
                    if hasattr(order, 'lmtPrice') and order.lmtPrice:
                        order_info["limitPrice"] = order.lmtPrice
                    if hasattr(order, 'auxPrice') and order.auxPrice:
                        order_info["stopPrice"] = order.auxPrice

                    order_details.append(order_info)
            except Exception as e:
                print(f"Warning: openOrders() failed: {e}")

            # Method 2: Check openTrades() for any missed orders
            try:
                trades = self.ib.openTrades()
                existing_order_ids = {order['orderId'] for order in order_details}

                for trade in trades:
                    order = trade.order
                    # Only add if not already in the list
                    if order.orderId not in existing_order_ids:
                        order_info = {
                            "orderId": order.orderId,
                            "symbol": trade.contract.symbol if hasattr(trade.contract, 'symbol') else 'N/A',
                            "action": order.action,
                            "quantity": order.totalQuantity,
                            "orderType": order.orderType,
                            "status": trade.orderStatus.status if hasattr(trade, 'orderStatus') else 'Unknown',
                            "source": "openTrades"
                        }

                        # Add price information based on order type
                        if hasattr(order, 'lmtPrice') and order.lmtPrice:
                            order_info["limitPrice"] = order.lmtPrice
                        if hasattr(order, 'auxPrice') and order.auxPrice:
                            order_info["stopPrice"] = order.auxPrice

                        order_details.append(order_info)
            except Exception as e:
                print(f"Warning: openTrades() failed: {e}")

            return order_details
        except Exception:
            return []