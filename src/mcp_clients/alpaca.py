from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config.settings import settings
from src.mcp_clients.robinhood import LIVE_RH_SPREAD_CANDIDATES, SpreadCandidateQuote

logger = logging.getLogger(__name__)

DEFAULT_ALPACA_PAPER_ACCOUNT = "PA35XZXE8ZZE"


def build_occ_symbol(
    ticker: str,
    expiration_date: str,
    option_type: str,
    strike_price: float,
) -> str:
    """
    Build standard 21-char OCC option symbol for Alpaca's options API
    (e.g., AMZN261218P00250000).
    """
    yy_mm_dd = expiration_date.replace("-", "")[2:]
    cp = "C" if option_type.lower().startswith("c") else "P"
    strike_int = int(round(strike_price * 1000))
    return f"{ticker.upper()}{yy_mm_dd}{cp}{strike_int:08d}"


def build_mleg_order_payload(
    idempotency_key: str,
    ticker: str,
    strategy: str,
    expiration_date: str,
    short_strike: float,
    long_strike: float,
    contracts: int,
    limit_price: float,
    is_closing: bool = False,
    time_in_force: str = "day",
) -> dict[str, Any]:
    """
    Build a payload compatible with `alpaca/place_option_order` and `POST /v2/orders`:
      - order_class="mleg"
      - Negative limit_price for net credit (entry), positive limit_price for net debit (closing).
    """
    option_type = "put" if "PUT" in strategy.upper() else "call"
    short_occ = build_occ_symbol(ticker, expiration_date, option_type, short_strike)
    long_occ = build_occ_symbol(ticker, expiration_date, option_type, long_strike)

    if not is_closing:
        short_side, short_intent = "sell", "sell_to_open"
        long_side, long_intent = "buy", "buy_to_open"
        signed_limit = -abs(round(limit_price, 2))
    else:
        short_side, short_intent = "buy", "buy_to_close"
        long_side, long_intent = "sell", "sell_to_close"
        signed_limit = abs(round(limit_price, 2))

    return {
        "client_order_id": idempotency_key[:48],
        "order_class": "mleg",
        "type": "limit",
        "time_in_force": time_in_force,
        "qty": str(contracts),
        "limit_price": f"{signed_limit:.2f}",
        "legs": [
            {
                "symbol": short_occ,
                "ratio_qty": "1",
                "side": short_side,
                "position_intent": short_intent,
            },
            {
                "symbol": long_occ,
                "ratio_qty": "1",
                "side": long_side,
                "position_intent": long_intent,
            },
        ],
    }


@dataclass
class AlpacaSimulatedOrder:
    order_id: str
    idempotency_key: str
    ticker: str
    strategy: str
    order_type: str
    limit_price: float
    contracts: int
    account_number: str = DEFAULT_ALPACA_PAPER_ACCOUNT
    status: str = "OPEN"
    filled_price: float | None = None
    steps_walked: int = 0
    mleg_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AlpacaPaperClient:
    """
    Client for the Alpaca Paper Trading account (PA35XZXE8ZZE, Option Level 3, $10,000 base).
    When ALPACA_API_KEY and ALPACA_SECRET_KEY are configured (e.g. on Render),
    syncs live account balances and submits multi-leg option orders to ALPACA_BASE_URL.
    """

    def __init__(
        self,
        account_number: str = DEFAULT_ALPACA_PAPER_ACCOUNT,
        total_value: float = 10000.0,
        cash: float = 10000.0,
        buying_power: float = 10000.0,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.account_number = account_number
        self.total_value = total_value
        self.cash = cash
        self.buying_power = buying_power
        self.api_key = api_key if api_key is not None else settings.alpaca_api_key
        self.secret_key = secret_key if secret_key is not None else settings.alpaca_secret_key
        self.base_url = (base_url or settings.alpaca_base_url).rstrip("/")
        self.orders: dict[str, AlpacaSimulatedOrder] = {}
        self.spread_candidates: dict[str, SpreadCandidateQuote] = dict(LIVE_RH_SPREAD_CANDIDATES)
        if self._has_live_credentials():
            self.sync_live_account()

    def _has_live_credentials(self) -> bool:
        return bool(
            self.api_key
            and self.secret_key
            and self.api_key != "REPLACE_IN_RENDER_DASHBOARD"
            and self.secret_key != "REPLACE_IN_RENDER_DASHBOARD"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }

    def sync_live_account(self) -> bool:
        if not self._has_live_credentials():
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/v2/account", headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                self.account_number = str(data.get("account_number", self.account_number))
                self.total_value = float(data.get("portfolio_value") or data.get("equity") or self.total_value)
                self.cash = float(data.get("cash") or self.cash)
                self.buying_power = float(
                    data.get("options_buying_power")
                    or data.get("non_marginable_buying_power")
                    or self.buying_power
                )
                return True
        except Exception as exc:
            logger.warning("Alpaca /v2/account sync warning: %s", exc)
            return False

    def get_portfolio_summary(self) -> dict[str, float | str]:
        return {
            "broker": "alpaca_paper",
            "account_number": self.account_number,
            "options_trading_level": 3,
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
    ) -> AlpacaSimulatedOrder:
        meta = metadata or {}
        is_closing = order_type in ("GTC_TAKE_PROFIT", "DEFENSIVE_EXIT", "DEFENSIVE_ROLL")
        tif = "gtc" if order_type == "GTC_TAKE_PROFIT" else "day"
        payload: dict[str, Any] = {}
        if {"expiration_date", "short_strike", "long_strike"}.issubset(meta.keys()):
            payload = build_mleg_order_payload(
                idempotency_key=idempotency_key,
                ticker=ticker,
                strategy=strategy,
                expiration_date=str(meta["expiration_date"]),
                short_strike=float(meta["short_strike"]),
                long_strike=float(meta["long_strike"]),
                contracts=contracts,
                limit_price=limit_price,
                is_closing=is_closing,
                time_in_force=tif,
            )

        order_id = f"alpaca-{uuid.uuid4().hex[:12]}"
        if self._has_live_credentials() and payload:
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.post(
                        f"{self.base_url}/v2/orders",
                        headers=self._headers(),
                        json=payload,
                    )
                    if resp.status_code in (200, 201):
                        data = resp.json()
                        order_id = f"alpaca-{data.get('id', uuid.uuid4().hex[:12])}"
                    else:
                        logger.info(
                            "Alpaca /v2/orders response (%s): %s",
                            resp.status_code,
                            resp.text[:200],
                        )
            except Exception as exc:
                logger.warning("Alpaca /v2/orders submission note: %s", exc)

        order = AlpacaSimulatedOrder(
            order_id=order_id,
            idempotency_key=idempotency_key,
            ticker=ticker,
            strategy=strategy,
            order_type=order_type,
            limit_price=round(limit_price, 2),
            contracts=contracts,
            account_number=self.account_number,
            status="OPEN",
            mleg_payload=payload,
            metadata=meta,
        )
        self.orders[order_id] = order
        return order

    def update_limit_price(self, order_id: str, new_limit_price: float) -> AlpacaSimulatedOrder:
        order = self.orders[order_id]
        if order.status != "OPEN":
            raise RuntimeError(f"Cannot walk non-open order {order_id} (status={order.status})")
        order.limit_price = round(new_limit_price, 2)
        order.steps_walked += 1
        return order

    def mark_order_filled(
        self, order_id: str, fill_price: float | None = None
    ) -> AlpacaSimulatedOrder:
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

    def get_order(self, order_id: str) -> AlpacaSimulatedOrder:
        return self.orders[order_id]
