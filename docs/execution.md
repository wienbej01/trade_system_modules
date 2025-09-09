# Execution Guide

This guide covers trading execution using Interactive Brokers (IBKR) integration for live trading operations.

## Overview

The execution module provides:
- **Order Management**: Place, cancel, and monitor orders
- **Position Tracking**: Real-time position updates
- **Account Information**: Balance and margin details
- **Risk Management**: Position size limits and controls

## IBExec Class

### Initialization and Connection

```python
from trade_system_modules.execution.ibkr_exec import IBExec

# Initialize execution client
exec_client = IBExec()

# Connect to IBKR
if exec_client.connect():
    print("Connected to IBKR for trading")
else:
    print("Failed to connect")
    exit(1)
```

### Connection Management

```python
# Check connection status
print(f"Connected: {exec_client.is_connected()}")

# Disconnect when done
exec_client.disconnect()
print("Disconnected from IBKR")
```

## Order Management

### Placing Orders

```python
from trade_system_modules.data.symbology import resolve_instrument
from ib_insync import MarketOrder, LimitOrder, StopOrder

# Resolve contract
contract = resolve_instrument("AAPL")

# Market order
market_order = MarketOrder("BUY", 100)  # Buy 100 shares at market
trade = exec_client.place_order(contract, market_order)
print(f"Order placed: {trade.order.orderId}")

# Limit order
limit_order = LimitOrder("SELL", 50, 150.0)  # Sell 50 shares at $150
trade = exec_client.place_order(contract, limit_order)

# Stop order
stop_order = StopOrder("SELL", 25, 140.0)  # Stop sell at $140
trade = exec_client.place_order(contract, stop_order)
```

### Order Types

```python
from ib_insync import (
    MarketOrder,      # Market price
    LimitOrder,       # Limit price
    StopOrder,        # Stop price
    StopLimitOrder,   # Stop with limit
    TrailingStopOrder # Trailing stop
)

# Market order
order = MarketOrder(action="BUY", totalQuantity=100)

# Limit order
order = LimitOrder(action="SELL", totalQuantity=50, lmtPrice=150.0)

# Stop order
order = StopOrder(action="SELL", totalQuantity=25, stopPrice=140.0)

# Stop-limit order
order = StopLimitOrder(
    action="BUY",
    totalQuantity=100,
    stopPrice=145.0,
    lmtPrice=146.0
)

# Trailing stop (percentage)
order = TrailingStopOrder(
    action="SELL",
    totalQuantity=50,
    trailingPercent=2.0  # 2% trailing stop
)
```

### Order Monitoring

```python
# Place order and monitor
trade = exec_client.place_order(contract, order)

# Check order status
print(f"Order ID: {trade.order.orderId}")
print(f"Status: {trade.orderStatus.status}")
print(f"Filled: {trade.orderStatus.filled}")
print(f"Remaining: {trade.orderStatus.remaining}")

# Wait for completion
while not trade.isDone:
    exec_client.sleep(1)
    print(f"Status: {trade.orderStatus.status}")

print(f"Final status: {trade.orderStatus.status}")
print(f"Executed price: {trade.orderStatus.avgFillPrice}")
```

### Canceling Orders

```python
# Cancel by order ID
order_id = 12345
exec_client.cancel_order(order_id)

# Cancel all orders
for trade in exec_client.trades():
    if not trade.isDone:
        exec_client.cancel_order(trade.order.orderId)
```

## Position Management

### Current Positions

```python
# Get all positions
positions = exec_client.get_positions()

for position in positions:
    print(f"Contract: {position.contract.symbol}")
    print(f"Position: {position.position}")
    print(f"Average Cost: {position.avgCost}")
    print(f"Market Value: {position.marketValue}")
    print("---")
```

### Position Tracking

```python
def track_positions():
    """Monitor position changes."""

    positions = exec_client.get_positions()
    position_dict = {pos.contract.symbol: pos for pos in positions}

    while True:
        exec_client.sleep(60)  # Check every minute

        new_positions = exec_client.get_positions()
        new_position_dict = {pos.contract.symbol: pos for pos in new_positions}

        # Check for changes
        for symbol in set(position_dict.keys()) | set(new_position_dict.keys()):
            old_pos = position_dict.get(symbol)
            new_pos = new_position_dict.get(symbol)

            if old_pos != new_pos:
                print(f"Position change for {symbol}:")
                if old_pos:
                    print(f"  Old: {old_pos.position} @ {old_pos.avgCost}")
                if new_pos:
                    print(f"  New: {new_pos.position} @ {new_pos.avgCost}")

        position_dict = new_position_dict

# Usage
track_positions()
```

## Account Information

### Account Summary

```python
# Get account information
account_info = exec_client.get_account_summary()

for key, value in account_info.items():
    print(f"{key}: {value}")
```

### Account Values

```python
# Common account values
account_values = exec_client.accountValues()

for value in account_values:
    if value.tag in ['TotalCashValue', 'NetLiquidation', 'BuyingPower']:
        print(f"{value.tag}: {value.value}")
```

### Margin Information

```python
def check_margin_status():
    """Monitor margin usage."""

    account_values = exec_client.accountValues()

    for value in account_values:
        if value.tag == 'AvailableFunds':
            available = float(value.value)
        elif value.tag == 'BuyingPower':
            buying_power = float(value.value)
        elif value.tag == 'NetLiquidation':
            net_liq = float(value.value)

    margin_used = ((net_liq - available) / net_liq) * 100
    print(f"Margin used: {margin_used:.2f}%")
    print(f"Buying power: ${buying_power:,.2f}")

    return margin_used < 50  # True if margin usage is safe
```

## Risk Management

### Position Size Limits

```python
class RiskManager:
    """Risk management for position sizing."""

    def __init__(self, max_position_pct=0.05, max_portfolio_risk=0.02):
        self.max_position_pct = max_position_pct  # Max 5% of portfolio
        self.max_portfolio_risk = max_portfolio_risk  # Max 2% risk

    def calculate_position_size(self, price, stop_loss, portfolio_value):
        """Calculate safe position size."""

        # Risk per share
        risk_per_share = abs(price - stop_loss)

        # Max position value based on portfolio percentage
        max_pos_value = portfolio_value * self.max_position_pct

        # Max position value based on risk limit
        max_risk_value = portfolio_value * self.max_portfolio_risk
        max_pos_from_risk = max_risk_value / (risk_per_share / price)

        # Use the smaller limit
        max_position_value = min(max_pos_value, max_pos_from_risk)

        # Calculate quantity
        quantity = int(max_position_value / price)

        return quantity

# Usage
risk_mgr = RiskManager()
quantity = risk_mgr.calculate_position_size(
    price=150.0,
    stop_loss=140.0,
    portfolio_value=100000
)
print(f"Safe position size: {quantity} shares")
```

### Circuit Breakers

```python
class CircuitBreaker:
    """Trading circuit breaker."""

    def __init__(self, max_daily_loss=0.05, max_single_loss=0.02):
        self.max_daily_loss = max_daily_loss
        self.max_single_loss = max_single_loss
        self.daily_pnl = 0.0
        self.daily_start_value = None

    def should_trade(self, trade_value, stop_loss):
        """Check if trade should be allowed."""

        # Initialize daily value
        if self.daily_start_value is None:
            account_info = exec_client.get_account_summary()
            self.daily_start_value = float(account_info.get('NetLiquidation', 0))

        # Check single trade risk
        risk_pct = abs(trade_value - stop_loss) / trade_value
        if risk_pct > self.max_single_loss:
            print(f"Trade risk {risk_pct:.2%} exceeds limit {self.max_single_loss:.2%}")
            return False

        # Check daily loss limit
        current_value = float(exec_client.get_account_summary().get('NetLiquidation', 0))
        if self.daily_start_value > 0:
            daily_loss_pct = (self.daily_start_value - current_value) / self.daily_start_value
            if daily_loss_pct > self.max_daily_loss:
                print(f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_daily_loss:.2%}")
                return False

        return True

# Usage
circuit_breaker = CircuitBreaker()

if circuit_breaker.should_trade(150.0, 140.0):
    # Place order
    order = LimitOrder("BUY", quantity, 150.0)
    exec_client.place_order(contract, order)
else:
    print("Trade blocked by circuit breaker")
```

## Order Execution Strategies

### Bracket Orders

```python
from ib_insync import BracketOrder

def place_bracket_order(contract, entry_price, quantity, stop_loss_pct=0.05, target_pct=0.10):
    """Place bracket order with stop loss and target."""

    # Parent order (entry)
    parent = LimitOrder("BUY", quantity, entry_price)

    # Stop loss
    stop_price = entry_price * (1 - stop_loss_pct)
    stop_loss = StopOrder("SELL", quantity, stop_price)

    # Profit target
    target_price = entry_price * (1 + target_pct)
    profit_target = LimitOrder("SELL", quantity, target_price)

    # Create bracket
    bracket = BracketOrder(parent, profit_target, stop_loss)

    # Place bracket order
    trades = exec_client.placeOrder(contract, parent)
    trades[0].bracket = bracket

    return trades

# Usage
trades = place_bracket_order(
    contract=contract,
    entry_price=150.0,
    quantity=100,
    stop_loss_pct=0.05,  # 5% stop loss
    target_pct=0.10      # 10% profit target
)
```

### Scale Orders

```python
def place_scale_order(contract, action, total_quantity, start_price, end_price, scale_increment):
    """Place scale order to fill gradually."""

    orders = []

    # Calculate price levels
    if action == "BUY":
        prices = [start_price - i * scale_increment for i in range(total_quantity // 100)]
    else:  # SELL
        prices = [start_price + i * scale_increment for i in range(total_quantity // 100)]

    # Place orders at each level
    for i, price in enumerate(prices):
        quantity = min(100, total_quantity - i * 100)
        if quantity <= 0:
            break

        order = LimitOrder(action, quantity, price)
        trade = exec_client.place_order(contract, order)
        orders.append(trade)

        # Small delay between orders
        exec_client.sleep(0.1)

    return orders

# Usage
scale_orders = place_scale_order(
    contract=contract,
    action="BUY",
    total_quantity=500,
    start_price=150.0,
    end_price=148.0,
    scale_increment=0.10
)
```

## Monitoring and Logging

### Execution Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoggedExecutionClient(IBExec):
    """Execution client with enhanced logging."""

    def place_order(self, contract, order):
        """Place order with logging."""

        logger.info(f"Placing {order.action} order for {order.totalQuantity} {contract.symbol}")

        try:
            trade = super().place_order(contract, order)

            # Log order details
            logger.info(f"Order {trade.order.orderId} placed: {order.action} {order.totalQuantity} {contract.symbol}")

            # Monitor order status
            def on_status_update(trade_obj, status):
                logger.info(f"Order {trade_obj.order.orderId} status: {status.status}")

            trade.statusEvent += on_status_update

            return trade

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise

    def cancel_order(self, order_id):
        """Cancel order with logging."""

        logger.info(f"Cancelling order {order_id}")

        try:
            super().cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            raise
```

### Performance Tracking

```python
class ExecutionTracker:
    """Track execution performance."""

    def __init__(self):
        self.trades = []

    def record_trade(self, trade):
        """Record completed trade."""

        if trade.isDone and trade.orderStatus.filled > 0:
            trade_record = {
                'order_id': trade.order.orderId,
                'symbol': trade.contract.symbol,
                'action': trade.order.action,
                'quantity': trade.orderStatus.filled,
                'avg_price': trade.orderStatus.avgFillPrice,
                'timestamp': datetime.now(),
                'commission': getattr(trade.orderStatus, 'commission', 0)
            }

            self.trades.append(trade_record)
            self.save_trade_record(trade_record)

    def save_trade_record(self, record):
        """Save trade record to storage."""

        # Save to GCS
        import pandas as pd
        from trade_system_modules.storage.gcs import GCSClient

        df = pd.DataFrame([record])
        client = GCSClient()

        date_str = record['timestamp'].strftime('%Y-%m-%d')
        path = f"trades/{date_str}/{record['order_id']}.parquet"
        client.to_parquet(df, path)

    def get_daily_pnl(self, date):
        """Calculate daily P&L."""

        day_trades = [t for t in self.trades
                     if t['timestamp'].date() == date.date()]

        total_pnl = 0
        for trade in day_trades:
            # Simplified P&L calculation
            # In practice, you'd need entry/exit prices
            pass

        return total_pnl

# Usage
tracker = ExecutionTracker()

# Monitor all trades
for trade in exec_client.trades():
    if trade.isDone:
        tracker.record_trade(trade)
```

## Error Handling

### Connection Recovery

```python
def maintain_connection():
    """Maintain IBKR connection with auto-reconnect."""

    while True:
        if not exec_client.is_connected():
            print("Connection lost, attempting reconnect...")

            try:
                if exec_client.connect():
                    print("Reconnected successfully")
                else:
                    print("Reconnect failed, retrying in 30 seconds...")
                    exec_client.sleep(30)
            except Exception as e:
                print(f"Reconnect error: {e}")
                exec_client.sleep(30)
        else:
            exec_client.sleep(10)  # Check every 10 seconds

# Run in background thread
import threading
connection_thread = threading.Thread(target=maintain_connection, daemon=True)
connection_thread.start()
```

### Order Error Handling

```python
def safe_place_order(contract, order, max_retries=3):
    """Place order with error handling and retries."""

    for attempt in range(max_retries):
        try:
            trade = exec_client.place_order(contract, order)

            # Check for immediate rejection
            if hasattr(trade.orderStatus, 'status'):
                if trade.orderStatus.status == 'Rejected':
                    print(f"Order rejected: {getattr(trade.orderStatus, 'whyHeld', 'Unknown reason')}")
                    return None

            return trade

        except Exception as e:
            print(f"Order attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                exec_client.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Max retries exceeded")
                return None

# Usage
trade = safe_place_order(contract, order)
if trade:
    print(f"Order placed: {trade.order.orderId}")
```

## Best Practices

1. **Test with paper trading** before live execution
2. **Implement proper risk management** and position sizing
3. **Monitor account status** continuously
4. **Use appropriate order types** for different market conditions
5. **Log all trading activity** for analysis and compliance
6. **Handle connection issues** gracefully
7. **Validate orders** before placement
8. **Set reasonable timeouts** for order operations

## Integration Examples

### With Signal Generation

```python
from trade_system_modules import get_agg_minute
from trade_system_modules.schemas.bars import ensure_bar_schema

def signal_based_trading():
    """Execute trades based on signals."""

    # Get market data
    data = get_agg_minute("AAPL", "2023-01-01", "2023-01-02")
    data = ensure_bar_schema(data)

    # Generate signals (simplified example)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['signal'] = (data['close'] > data['sma_20']).astype(int)

    # Execute based on signals
    for idx, row in data.iterrows():
        if row['signal'] == 1 and not has_position("AAPL"):
            # Buy signal
            order = MarketOrder("BUY", 100)
            exec_client.place_order(resolve_instrument("AAPL"), order)
        elif row['signal'] == 0 and has_position("AAPL"):
            # Sell signal
            order = MarketOrder("SELL", 100)
            exec_client.place_order(resolve_instrument("AAPL"), order)

def has_position(symbol):
    """Check if we have a position in the symbol."""
    positions = exec_client.get_positions()
    return any(pos.contract.symbol == symbol and pos.position != 0 for pos in positions)
```

### With Portfolio Management

```python
class PortfolioManager:
    """Manage portfolio allocation and rebalancing."""

    def __init__(self, target_allocations):
        self.target_allocations = target_allocations  # {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}

    def rebalance_portfolio(self):
        """Rebalance portfolio to target allocations."""

        # Get current positions and values
        positions = exec_client.get_positions()
        account_value = float(exec_client.get_account_summary().get('NetLiquidation', 0))

        current_allocations = {}
        for pos in positions:
            symbol = pos.contract.symbol
            position_value = pos.marketValue
            current_allocations[symbol] = position_value / account_value

        # Calculate required trades
        trades = []
        for symbol, target_pct in self.target_allocations.items():
            current_pct = current_allocations.get(symbol, 0)
            diff_pct = target_pct - current_pct

            if abs(diff_pct) > 0.01:  # Rebalance threshold
                contract = resolve_instrument(symbol)

                # Get current price
                ticker = exec_client.reqMktData(contract)
                exec_client.sleep(1)
                price = ticker.last if ticker.last > 0 else ticker.close

                # Calculate trade size
                trade_value = diff_pct * account_value
                quantity = int(abs(trade_value) / price)

                if quantity > 0:
                    action = "BUY" if diff_pct > 0 else "SELL"
                    order = MarketOrder(action, quantity)
                    trades.append((contract, order))

        # Execute trades
        for contract, order in trades:
            exec_client.place_order(contract, order)
            exec_client.sleep(0.5)  # Rate limiting

# Usage
portfolio_mgr = PortfolioManager({
    'AAPL': 0.3,
    'MSFT': 0.3,
    'GOOGL': 0.4
})

portfolio_mgr.rebalance_portfolio()
```

## See Also

- [API Reference](api-reference.md) - IBExec documentation
- [Configuration Guide](configuration.md) - IBKR setup
- [Data Adapters](data-adapters.md) - Symbol resolution
- [Troubleshooting](troubleshooting.md) - Common execution issues