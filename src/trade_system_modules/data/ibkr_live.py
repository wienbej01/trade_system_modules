"""IBKR live data adapter with advanced market data capabilities."""

from __future__ import annotations

from ib_insync import IB, BarData, Stock, Contract
from typing import Callable, Optional, List, Dict, Any
from ..config.settings import settings

class IBLive:
    """IBKR live streaming client with advanced market data capabilities."""

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
        contract = Stock(symbol, "SMART", "USD")
        bars = self.ib.reqRealTimeBars(contract, 5, what, useRth)
        if callback:
            bars.updateEvent += callback
        return bars

    def get_historical_data(self, symbol: str, duration: str = "1 D", bar_size: str = "1 min",
                           what: str = "TRADES", use_rth: bool = True) -> List[BarData]:
        """Get historical bar data."""
        contract = Stock(symbol, "SMART", "USD")

        bars = self.ib.reqHistoricalData(
            contract=contract,
            endDateTime="",  # Empty string = current time
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what,
            useRTH=use_rth,
            formatDate=1
        )

        return bars

    def get_market_depth(self, symbol: str, num_rows: int = 5) -> Dict[str, Any]:
        """Get market depth/order book data."""
        contract = Stock(symbol, "SMART", "USD")

        # Request market depth
        self.ib.reqMktDepth(contract, num_rows, isSmartDepth=False)

        # Get the depth data
        depth = self.ib.marketDepth(contract)

        if depth:
            return {
                "symbol": symbol,
                "timestamp": depth.time,
                "bid": [
                    {"price": level.price, "size": level.size}
                    for level in depth.bid[:num_rows]
                ],
                "ask": [
                    {"price": level.price, "size": level.size}
                    for level in depth.ask[:num_rows]
                ]
            }
        else:
            return {"error": "Market depth data not available"}

    def get_option_chain(self, symbol: str, strike_range: float = 0.1) -> List[Contract]:
        """Get option chain for a symbol."""
        underlying = Stock(symbol, "SMART", "USD")

        # Get current price for strike range calculation
        ticker = self.ib.reqMktData(underlying, '', False, False)

        # Wait for price data
        import time
        timeout = 5
        start_time = time.time()

        while time.time() - start_time < timeout:
            if ticker.last:
                current_price = ticker.last
                break
            time.sleep(0.1)
        else:
            return []

        # Calculate strike range
        min_strike = current_price * (1 - strike_range)
        max_strike = current_price * (1 + strike_range)

        # Get option chain
        chains = self.ib.reqSecDefOptParams(underlying.symbol, "", underlying.secType, underlying.conId)

        options = []
        for chain in chains:
            if chain.exchange == "SMART":
                for strike in chain.strikes:
                    if min_strike <= strike <= max_strike:
                        # Create call option
                        call_contract = Contract(
                            symbol=chain.symbol,
                            secType="OPT",
                            exchange="SMART",
                            currency="USD",
                            lastTradeDateOrContractMonth=chain.tradingClass[:6],  # YYYYMM
                            strike=strike,
                            right="C"
                        )
                        options.append(call_contract)

                        # Create put option
                        put_contract = Contract(
                            symbol=chain.symbol,
                            secType="OPT",
                            exchange="SMART",
                            currency="USD",
                            lastTradeDateOrContractMonth=chain.tradingClass[:6],
                            strike=strike,
                            right="P"
                        )
                        options.append(put_contract)

        return options[:20]  # Limit to first 20 options

    def get_fundamentals(self, symbol: str, report_type: str = "ReportsFinSummary") -> str:
        """Get fundamental data for a symbol."""
        contract = Stock(symbol, "SMART", "USD")

        try:
            fundamentals = self.ib.reqFundamentalData(contract, report_type)
            return fundamentals
        except Exception:
            return "Fundamental data not available"

    def subscribe_news(self, symbol: str, callback: Callable = None) -> None:
        """Subscribe to news bulletins for a symbol."""
        contract = Stock(symbol, "SMART", "USD")

        def news_handler(news):
            if callback:
                callback(news)
            else:
                print(f"News: {news}")

        # Subscribe to news
        self.ib.reqNewsBulletins(True)
        self.ib.newsBulletinEvent += news_handler

    def get_contract_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed contract information."""
        contract = Stock(symbol, "SMART", "USD")

        try:
            details = self.ib.reqContractDetails(contract)

            if details:
                detail = details[0]
                return {
                    "symbol": detail.contract.symbol,
                    "exchange": detail.contract.exchange,
                    "currency": detail.contract.currency,
                    "secType": detail.contract.secType,
                    "marketName": detail.marketName,
                    "tradingHours": detail.tradingHours,
                    "liquidHours": detail.liquidHours,
                    "priceMagnifier": detail.priceMagnifier,
                    "minTick": detail.minTick,
                    "orderTypes": detail.orderTypes,
                    "validExchanges": detail.validExchanges
                }
            else:
                return {"error": "Contract details not found"}

        except Exception as e:
            return {"error": str(e)}

    def get_scanner_data(self, scanner_subscription) -> List[Contract]:
        """Get scanner/market scanner data."""
        try:
            data = self.ib.reqScannerData(scanner_subscription)
            return data
        except Exception:
            return []