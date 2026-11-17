from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _ColumnDescriptor:
    """Descriptor that returns a filter builder on the class and the attribute value on an instance."""

    def __init__(self, name: str, default: Any = None) -> None:
        self.name = name
        self.default = default

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self.name] = value

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        return lambda obj: getattr(obj, self.name) == other

    def in_(self, values: list[Any]) -> Any:
        return lambda obj: getattr(obj, self.name) in values

    def is_not(self, other: Any) -> Any:
        return lambda obj: getattr(obj, self.name) is not other


class Base:
    metadata = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class ExecutionIntent(Base):
    __tablename__ = "execution_intents"

    id = _ColumnDescriptor("id", None)
    idempotency_key = _ColumnDescriptor("idempotency_key", "")
    ticker = _ColumnDescriptor("ticker", "")
    strategy = _ColumnDescriptor("strategy", "")
    intent_type = _ColumnDescriptor("intent_type", "ENTRY_INTENT")
    expiration_date = _ColumnDescriptor("expiration_date", "")
    dte = _ColumnDescriptor("dte", 0)
    short_strike = _ColumnDescriptor("short_strike", 0.0)
    long_strike = _ColumnDescriptor("long_strike", 0.0)
    short_option_id = _ColumnDescriptor("short_option_id", None)
    long_option_id = _ColumnDescriptor("long_option_id", None)
    contracts = _ColumnDescriptor("contracts", 1)
    mid_credit = _ColumnDescriptor("mid_credit", 0.0)
    min_credit_floor = _ColumnDescriptor("min_credit_floor", 0.0)
    bid_ask_spread = _ColumnDescriptor("bid_ask_spread", 0.0)
    expected_50pct_profit_usd = _ColumnDescriptor("expected_50pct_profit_usd", 0.0)
    collateral_locked_usd = _ColumnDescriptor("collateral_locked_usd", 0.0)
    vix_regime = _ColumnDescriptor("vix_regime", "")
    rationale = _ColumnDescriptor("rationale", "")
    status = _ColumnDescriptor("status", "QUEUED")

    def __init__(self, **kwargs: Any) -> None:
        self.created_at = utc_now()
        self.updated_at = utc_now()
        self.status = "QUEUED"
        super().__init__(**kwargs)


class Trade(Base):
    __tablename__ = "trades"

    id = _ColumnDescriptor("id", None)
    idempotency_key = _ColumnDescriptor("idempotency_key", "")
    ticker = _ColumnDescriptor("ticker", "")
    strategy = _ColumnDescriptor("strategy", "")
    expiration_date = _ColumnDescriptor("expiration_date", "")
    dte_at_entry = _ColumnDescriptor("dte_at_entry", 0)
    short_strike = _ColumnDescriptor("short_strike", 0.0)
    long_strike = _ColumnDescriptor("long_strike", 0.0)
    short_option_id = _ColumnDescriptor("short_option_id", None)
    long_option_id = _ColumnDescriptor("long_option_id", None)
    contracts = _ColumnDescriptor("contracts", 1)
    entry_credit = _ColumnDescriptor("entry_credit", 0.0)
    target_close_price = _ColumnDescriptor("target_close_price", 0.0)
    current_mark = _ColumnDescriptor("current_mark", 0.0)
    short_delta = _ColumnDescriptor("short_delta", 0.25)
    tv_sr_anchor = _ColumnDescriptor("tv_sr_anchor", 0.0)
    atr_14 = _ColumnDescriptor("atr_14", 1.0)
    collateral_usd = _ColumnDescriptor("collateral_usd", 0.0)
    unrealized_pnl_usd = _ColumnDescriptor("unrealized_pnl_usd", 0.0)
    realized_pnl_usd = _ColumnDescriptor("realized_pnl_usd", None)
    slippage_usd = _ColumnDescriptor("slippage_usd", 0.0)
    entry_order_id = _ColumnDescriptor("entry_order_id", None)
    gtc_order_id = _ColumnDescriptor("gtc_order_id", None)
    exit_order_id = _ColumnDescriptor("exit_order_id", None)
    execution_mode = _ColumnDescriptor("execution_mode", "PAPER")
    status = _ColumnDescriptor("status", "OPEN")
    closed_at = _ColumnDescriptor("closed_at", None)

    def __init__(self, **kwargs: Any) -> None:
        self.opened_at = utc_now()
        self.status = "OPEN"
        self.realized_pnl_usd = None
        super().__init__(**kwargs)


class StressEvent(Base):
    __tablename__ = "stress_events"

    id = _ColumnDescriptor("id", None)
    ticker = _ColumnDescriptor("ticker", "")
    strategy = _ColumnDescriptor("strategy", "")
    contracts = _ColumnDescriptor("contracts", 2)
    dte = _ColumnDescriptor("dte", 30)
    short_delta = _ColumnDescriptor("short_delta", 0.38)
    spread_mark = _ColumnDescriptor("spread_mark", 0.0)
    unrealized_pnl = _ColumnDescriptor("unrealized_pnl", 0.0)
    dist_to_tv_sr_atr = _ColumnDescriptor("dist_to_tv_sr_atr", 0.0)
    tv_1h_close_breach = _ColumnDescriptor("tv_1h_close_breach", False)
    tv_4h_rsi = _ColumnDescriptor("tv_4h_rsi", 50.0)
    uw_darkpool_net_flow_1h = _ColumnDescriptor("uw_darkpool_net_flow_1h", 0.0)
    uw_sweep_ratio = _ColumnDescriptor("uw_sweep_ratio", 0.5)
    vix_level = _ColumnDescriptor("vix_level", 20.0)
    vix_zscore_20d = _ColumnDescriptor("vix_zscore_20d", 0.0)
    vix_4h_rsi = _ColumnDescriptor("vix_4h_rsi", 50.0)
    vix_1h_slope = _ColumnDescriptor("vix_1h_slope", 0.0)
    vix_term_structure_ratio = _ColumnDescriptor("vix_term_structure_ratio", 0.95)
    action_taken = _ColumnDescriptor("action_taken", "HOLD_LOOSE")

    def __init__(self, **kwargs: Any) -> None:
        self.timestamp = utc_now()
        super().__init__(**kwargs)


class ActiveRiskThreshold(Base):
    __tablename__ = "active_risk_thresholds"

    id = _ColumnDescriptor("id", None)
    version = _ColumnDescriptor("version", "")
    active_policy = _ColumnDescriptor("active_policy", "")
    baseline_short_delta_ceiling = _ColumnDescriptor("baseline_short_delta_ceiling", 0.52)
    ml_early_cut_delta_ceiling = _ColumnDescriptor("ml_early_cut_delta_ceiling", 0.43)
    tv_1h_sr_atr_buffer = _ColumnDescriptor("tv_1h_sr_atr_buffer", 0.15)
    uw_breakout_darkpool_zscore_min = _ColumnDescriptor("uw_breakout_darkpool_zscore_min", 1.75)
    uw_breakout_ask_sweep_ratio_min = _ColumnDescriptor("uw_breakout_ask_sweep_ratio_min", 0.68)
    vix_backwardation_ratio_trigger = _ColumnDescriptor("vix_backwardation_ratio_trigger", 1.0)
    payload_json = _ColumnDescriptor("payload_json", "{}")

    def __init__(self, **kwargs: Any) -> None:
        self.promoted_at = utc_now()
        super().__init__(**kwargs)
