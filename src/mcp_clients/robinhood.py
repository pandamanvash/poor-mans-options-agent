from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MCP_ACCOUNT_NUMBER = "512500422"
DEFAULT_MCP_ACCOUNT_NICKNAME = "MCP"


@dataclass(frozen=True)
class OptionLegQuote:
    instrument_id: str
    symbol: str
    expiration_date: str
    dte: int
    strike_price: float
    option_type: str  # "put" or "call"
    bid: float
    ask: float
    mark: float
    delta: float
    implied_volatility: float
    open_interest: int

    @property
    def bid_ask_spread(self) -> float:
        return round(self.ask - self.bid, 4)


@dataclass
class SpreadCandidateQuote:
    ticker: str
    strategy: str  # "BULL_PUT_SPREAD" or "BEAR_CALL_SPREAD"
    expiration_date: str
    dte: int
    short_leg: OptionLegQuote
    long_leg: OptionLegQuote

    @property
    def spread_width(self) -> float:
        return round(abs(self.short_leg.strike_price - self.long_leg.strike_price), 2)

    @property
    def mid_credit(self) -> float:
        return round(self.short_leg.mark - self.long_leg.mark, 2)

    @property
    def natural_credit(self) -> float:
        return round(self.short_leg.bid - self.long_leg.ask, 2)

    @property
    def net_bid_ask_spread(self) -> float:
        return round(
            max(0.05, (self.short_leg.bid_ask_spread + self.long_leg.bid_ask_spread) * 0.25),
            2,
        )


@dataclass
class SimulatedOrder:
    order_id: str
    idempotency_key: str
    ticker: str
    strategy: str
    order_type: str  # "ENTRY_LIMIT", "GTC_TAKE_PROFIT", "DEFENSIVE_EXIT", "DEFENSIVE_ROLL"
    limit_price: float
    contracts: int
    account_number: str = DEFAULT_MCP_ACCOUNT_NUMBER
    account_nickname: str = DEFAULT_MCP_ACCOUNT_NICKNAME
    status: str = "OPEN"  # "OPEN", "FILLED", "CANCELLED"
    filled_price: float | None = None
    steps_walked: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# Live Robinhood MCP option quotes captured 2026-11-16 (MCP Account ••••0422)
LIVE_RH_SPREAD_CANDIDATES: dict[str, SpreadCandidateQuote] = {
    "AMZN_BULL_PUT": SpreadCandidateQuote(
        ticker="AMZN",
        strategy="BULL_PUT_SPREAD",
        expiration_date="2026-12-18",
        dte=32,
        short_leg=OptionLegQuote(
            instrument_id="1e1a02e3-ada1-44d9-8778-9841a541dac8",
            symbol="AMZN",
            expiration_date="2026-12-18",
            dte=32,
            strike_price=250.0,
            option_type="put",
            bid=7.30,
            ask=7.60,
            mark=7.45,
            delta=-0.278,
            implied_volatility=0.3836,
            open_interest=16358,
        ),
        long_leg=OptionLegQuote(
            instrument_id="0bbc954f-c7d1-48a4-923a-03ddc95e228b",
            symbol="AMZN",
            expiration_date="2026-12-18",
            dte=32,
            strike_price=245.0,
            option_type="put",
            bid=5.75,
            ask=6.00,
            mark=5.87,
            delta=-0.224,
            implied_volatility=0.3930,
            open_interest=9813,
        ),
    ),
    "AAPL_BEAR_CALL": SpreadCandidateQuote(
        ticker="AAPL",
        strategy="BEAR_CALL_SPREAD",
        expiration_date="2026-12-18",
        dte=32,
        short_leg=OptionLegQuote(
            instrument_id="6f198511-d3ce-4d8d-a8d5-b7a389a918ee",
            symbol="AAPL",
            expiration_date="2026-12-18",
            dte=32,
            strike_price=365.0,
            option_type="call",
            bid=4.80,
            ask=5.10,
            mark=4.95,
            delta=0.275,
            implied_volatility=0.2261,
            open_interest=7489,
        ),
        long_leg=OptionLegQuote(
            instrument_id="a7e90391-51b2-4086-8f13-6dbf9784bcb0",
            symbol="AAPL",
            expiration_date="2026-12-18",
            dte=32,
            strike_price=370.0,
            option_type="call",
            bid=3.25,
            ask=3.50,
            mark=3.37,
            delta=0.205,
            implied_volatility=0.2200,
            open_interest=19574,
        ),
    ),
    "TSLA_BULL_PUT": SpreadCandidateQuote(
        ticker="TSLA",
        strategy="BULL_PUT_SPREAD",
        expiration_date="2026-12-18",
        dte=32,
        short_leg=OptionLegQuote(
            instrument_id="8689bb60-a2d7-4091-9976-315a32a1a741",
            symbol="TSLA",
            expiration_date="2026-12-18",
            dte=32,
            strike_price=355.0,
            option_type="put",
            bid=11.85,
            ask=12.15,
            mark=12.00,
            delta=-0.278,
            implied_volatility=0.5253,
            open_interest=3790,
        ),
        long_leg=OptionLegQuote(
            instrument_id="7e26e58f-ff66-45fb-bc8d-466bf7aa52bc",
            symbol="TSLA",
            expiration_date="2026-12-18",
            dte=32,
            strike_price=350.0,
            option_type="put",
            bid=10.30,
            ask=10.50,
            mark=10.40,
            delta=-0.248,
            implied_volatility=0.5314,
            open_interest=15795,
        ),
    ),
}


class RobinhoodClient:
    """
    Client wrapper for Robinhood MCP account, option chains, quotes,
    Smart Mid-Price Limit Walker, GTC staging, and Cancel-First exits.
    Defaults strictly to the Robinhood 'MCP' account (512500422).
    """

    def __init__(
        self,
        account_number: str = DEFAULT_MCP_ACCOUNT_NUMBER,
        account_nickname: str = DEFAULT_MCP_ACCOUNT_NICKNAME,
        total_value: float = 5207.86,
        cash: float = 5492.86,
        buying_power: float = 3992.86,
    ) -> None:
        if account_number == "542891403":
            raise PermissionError(
                "Account 542891403 is read-only; default to the 'MCP' account (512500422)."
            )
        self.account_number = account_number
        self.account_nickname = account_nickname
        self.total_value = total_value
        self.cash = cash
        self.buying_power = buying_power
        self.orders: dict[str, SimulatedOrder] = {}
        self.spread_candidates = dict(LIVE_RH_SPREAD_CANDIDATES)

    def get_portfolio_summary(self) -> dict[str, float | str]:
        return {
            "account_number": self.account_number,
            "account_nickname": self.account_nickname,
            "brokerage_trading_type": "limited_margin",
            "option_level": "option_level_3",
            "total_value": self.total_value,
            "cash": self.cash,
            "buying_power": self.buying_power,
        }

    def get_spread_candidate(self, key: str) -> SpreadCandidateQuote:
        if key not in self.spread_candidates:
            raise KeyError(f"Unknown spread candidate key: {key}")
        return self.spread_candidates[key]

    def submit_limit_order(
        self,
        idempotency_key: str,
        ticker: str,
        strategy: str,
        order_type: str,
        limit_price: float,
        contracts: int,
        metadata: dict[str, Any] | None = None,
    ) -> SimulatedOrder:
        order_id = f"rh-{uuid.uuid4().hex[:12]}"
        order = SimulatedOrder(
            order_id=order_id,
            idempotency_key=idempotency_key,
            ticker=ticker,
            strategy=strategy,
            order_type=order_type,
            limit_price=round(limit_price, 2),
            contracts=contracts,
            account_number=self.account_number,
            account_nickname=self.account_nickname,
            status="OPEN",
            metadata=metadata or {},
        )
        self.orders[order_id] = order
        return order

    def update_limit_price(self, order_id: str, new_limit_price: float) -> SimulatedOrder:
        order = self.orders[order_id]
        if order.status != "OPEN":
            raise RuntimeError(f"Cannot walk non-open order {order_id} (status={order.status})")
        order.limit_price = round(new_limit_price, 2)
        order.steps_walked += 1
        return order

    def mark_order_filled(self, order_id: str, fill_price: float | None = None) -> SimulatedOrder:
        order = self.orders[order_id]
        order.status = "FILLED"
        order.filled_price = round(fill_price if fill_price is not None else order.limit_price, 2)
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.orders:
            return True
        order = self.orders[order_id]
        if order.status == "FILLED":
            raise RuntimeError(f"Cannot cancel already-filled order {order_id}")
        order.status = "CANCELLED"
        return True

    def get_order(self, order_id: str) -> SimulatedOrder:
        return self.orders[order_id]
