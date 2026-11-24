from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

import httpx

from src.config.settings import settings
from src.mcp_clients.robinhood import LIVE_RH_SPREAD_CANDIDATES, OptionLegQuote, SpreadCandidateQuote

logger = logging.getLogger(__name__)

DEFAULT_ALPACA_PAPER_ACCOUNT = "PA35XZXE8ZZE"
DEFAULT_WATCHLIST: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "AMZN",
    "AAPL",
    "TSLA",
    "META",
    "NVDA",
    "AMD",
    "MSFT",
)
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _extract_alpaca_uuid(order_id: str) -> str | None:
    raw = order_id[7:] if order_id.startswith("alpaca-") else order_id
    return raw if _UUID_RE.match(raw) else None


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

    def fetch_live_positions(self) -> list[dict[str, Any]]:
        if not self._has_live_credentials():
            return []
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(f"{self.base_url}/v2/positions", headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Alpaca /v2/positions fetch warning: %s", exc)
            return []

    def fetch_live_orders(self, status: str = "all", limit: int = 50) -> list[dict[str, Any]]:
        if not self._has_live_credentials():
            return []
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(
                    f"{self.base_url}/v2/orders",
                    headers=self._headers(),
                    params={"status": status, "nested": "true", "limit": str(limit)},
                )
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Alpaca /v2/orders fetch warning: %s", exc)
            return []

    def refresh_live_option_quotes(self) -> int:
        """
        Cloud-native Alpaca Options Market Data refresh (no local computer needed):
        Updates bid/ask/mark/delta/IV on spread_candidates directly from
        https://data.alpaca.markets/v1beta1/options/snapshots.
        """
        if not self._has_live_credentials():
            return 0
        updated = 0
        with httpx.Client(timeout=8.0) as client:
            for key, cand in list(self.spread_candidates.items()):
                short_occ = build_occ_symbol(
                    cand.ticker, cand.expiration_date, cand.short_leg.option_type, cand.short_leg.strike_price
                )
                long_occ = build_occ_symbol(
                    cand.ticker, cand.expiration_date, cand.long_leg.option_type, cand.long_leg.strike_price
                )
                try:
                    resp = client.get(
                        "https://data.alpaca.markets/v1beta1/options/snapshots",
                        headers=self._headers(),
                        params={"symbols": f"{short_occ},{long_occ}"},
                    )
                    if resp.status_code != 200:
                        continue
                    snaps = resp.json().get("snapshots") or {}
                    s_snap = snaps.get(short_occ) or {}
                    l_snap = snaps.get(long_occ) or {}
                    if s_snap and l_snap:
                        s_q = s_snap.get("latestQuote") or {}
                        l_q = l_snap.get("latestQuote") or {}
                        s_g = s_snap.get("greeks") or {}
                        l_g = l_snap.get("greeks") or {}
                        s_bid = float(s_q.get("bp") or cand.short_leg.bid)
                        s_ask = float(s_q.get("ap") or cand.short_leg.ask)
                        l_bid = float(l_q.get("bp") or cand.long_leg.bid)
                        l_ask = float(l_q.get("ap") or cand.long_leg.ask)
                        from src.mcp_clients.robinhood import OptionLegQuote

                        cand.short_leg = OptionLegQuote(
                            instrument_id=short_occ,
                            symbol=cand.ticker,
                            expiration_date=cand.expiration_date,
                            dte=cand.dte,
                            strike_price=cand.short_leg.strike_price,
                            option_type=cand.short_leg.option_type,
                            bid=s_bid,
                            ask=s_ask,
                            mark=round((s_bid + s_ask) / 2.0, 2),
                            delta=float(s_g.get("delta") or cand.short_leg.delta),
                            implied_volatility=float(
                                s_snap.get("impliedVolatility") or cand.short_leg.implied_volatility
                            ),
                            open_interest=cand.short_leg.open_interest,
                        )
                        cand.long_leg = OptionLegQuote(
                            instrument_id=long_occ,
                            symbol=cand.ticker,
                            expiration_date=cand.expiration_date,
                            dte=cand.dte,
                            strike_price=cand.long_leg.strike_price,
                            option_type=cand.long_leg.option_type,
                            bid=l_bid,
                            ask=l_ask,
                            mark=round((l_bid + l_ask) / 2.0, 2),
                            delta=float(l_g.get("delta") or cand.long_leg.delta),
                            implied_volatility=float(
                                l_snap.get("impliedVolatility") or cand.long_leg.implied_volatility
                            ),
                            open_interest=cand.long_leg.open_interest,
                        )
                        updated += 1
                except Exception as exc:
                    logger.info("Alpaca options snapshot refresh fallback for %s: %s", key, exc)
        return updated

    def scan_live_option_chains(
        self,
        tickers: Sequence[str] | None = None,
        min_dte: int = 28,
        max_dte: int = 45,
        preferred_widths: Sequence[float] = (5.0, 10.0),
        target_delta_min: float = 0.18,
        target_delta_max: float = 0.30,
        panic_regime: bool = False,
        reference_date: date | None = None,
    ) -> dict[str, SpreadCandidateQuote]:
        """
        Dynamic Cloud-Native Option Chain Scanner:
        Queries Alpaca `/v2/options/contracts` + `data.alpaca.markets/v1beta1/options/snapshots`
        across the liquid watchlist to select fresh 0.18-0.30 Delta (or 0.12-0.18 Delta puts in
        panic), 28-45 DTE, $5-wide credit spreads around current spot prices.
        """
        if not self._has_live_credentials():
            return dict(self.spread_candidates)

        today = reference_date or datetime.now(timezone.utc).date()
        exp_gte = (today + timedelta(days=min_dte)).isoformat()
        exp_lte = (today + timedelta(days=max_dte)).isoformat()
        target_tickers = [t.upper() for t in (tickers or DEFAULT_WATCHLIST)]
        discovered: dict[str, SpreadCandidateQuote] = {}

        with httpx.Client(timeout=10.0) as client:
            for sym in target_tickers:
                try:
                    c_resp = client.get(
                        f"{self.base_url}/v2/options/contracts",
                        headers=self._headers(),
                        params={
                            "underlying_symbols": sym,
                            "status": "active",
                            "expiration_date_gte": exp_gte,
                            "expiration_date_lte": exp_lte,
                            "limit": "200",
                        },
                    )
                    if c_resp.status_code != 200:
                        continue
                    contracts = c_resp.json().get("option_contracts") or []
                    if not contracts:
                        continue

                    # Pick expiration closest to 35 DTE
                    by_exp: dict[str, list[dict[str, Any]]] = {}
                    for c in contracts:
                        exp = str(c.get("expiration_date") or "")
                        if exp:
                            by_exp.setdefault(exp, []).append(c)
                    if not by_exp:
                        continue

                    best_exp = min(
                        by_exp.keys(),
                        key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days - 35),
                    )
                    dte = max(1, (datetime.strptime(best_exp, "%Y-%m-%d").date() - today).days)
                    exp_contracts = by_exp[best_exp]

                    # Fetch live snapshots in chunks of 50 symbols
                    occ_symbols = [str(c["symbol"]) for c in exp_contracts if c.get("symbol")]
                    snaps: dict[str, dict[str, Any]] = {}
                    for i in range(0, min(len(occ_symbols), 150), 50):
                        chunk = occ_symbols[i : i + 50]
                        s_resp = client.get(
                            "https://data.alpaca.markets/v1beta1/options/snapshots",
                            headers=self._headers(),
                            params={"symbols": ",".join(chunk)},
                        )
                        if s_resp.status_code == 200:
                            snaps.update(s_resp.json().get("snapshots") or {})

                    if not snaps:
                        continue

                    # Group parsed legs by option_type -> strike
                    legs_by_type: dict[str, dict[float, OptionLegQuote]] = {"put": {}, "call": {}}
                    for c in exp_contracts:
                        occ = str(c.get("symbol") or "")
                        snap = snaps.get(occ)
                        if not snap:
                            continue
                        opt_type = str(c.get("type") or "").lower()
                        if opt_type not in ("put", "call"):
                            continue
                        strike = float(c.get("strike_price") or 0.0)
                        q = snap.get("latestQuote") or {}
                        g = snap.get("greeks") or {}
                        bid = float(q.get("bp") or 0.0)
                        ask = float(q.get("ap") or 0.0)
                        delta = float(g.get("delta") or 0.0)
                        if strike <= 0 or bid <= 0 or ask <= 0 or ask < bid or delta == 0.0:
                            continue
                        oi = int(float(c.get("open_interest") or 0))
                        legs_by_type[opt_type][strike] = OptionLegQuote(
                            instrument_id=occ,
                            symbol=sym,
                            expiration_date=best_exp,
                            dte=dte,
                            strike_price=strike,
                            option_type=opt_type,
                            bid=round(bid, 2),
                            ask=round(ask, 2),
                            mark=round((bid + ask) / 2.0, 2),
                            delta=round(delta, 4),
                            implied_volatility=round(float(snap.get("impliedVolatility") or 0.35), 4),
                            open_interest=oi,
                        )

                    for opt_type, strategy_name, key_suffix in (
                        ("put", "BULL_PUT_SPREAD", "BULL_PUT"),
                        ("call", "BEAR_CALL_SPREAD", "BEAR_CALL"),
                    ):
                        # In Regime 4 panic, target lower-delta puts (0.12-0.18); normal 0.18-0.30
                        d_min = 0.12 if (panic_regime and opt_type == "put") else target_delta_min
                        d_max = 0.18 if (panic_regime and opt_type == "put") else target_delta_max
                        d_target = 0.15 if (panic_regime and opt_type == "put") else 0.24

                        chain = legs_by_type[opt_type]
                        short_candidates = [
                            leg for leg in chain.values() if d_min <= abs(leg.delta) <= d_max
                        ]
                        short_candidates.sort(key=lambda leg: abs(abs(leg.delta) - d_target))

                        chosen: SpreadCandidateQuote | None = None
                        for short_leg in short_candidates:
                            for width in preferred_widths:
                                long_strike = (
                                    round(short_leg.strike_price - width, 2)
                                    if opt_type == "put"
                                    else round(short_leg.strike_price + width, 2)
                                )
                                long_leg = chain.get(long_strike)
                                if long_leg is None:
                                    continue
                                cand = SpreadCandidateQuote(
                                    ticker=sym,
                                    strategy=strategy_name,
                                    expiration_date=best_exp,
                                    dte=dte,
                                    short_leg=short_leg,
                                    long_leg=long_leg,
                                )
                                if cand.mid_credit > 0.20:
                                    chosen = cand
                                    break
                            if chosen is not None:
                                break

                        if chosen is not None:
                            discovered[f"{sym}_{key_suffix}"] = chosen
                except Exception as exc:
                    logger.info("Alpaca live chain scan skipped for %s: %s", sym, exc)

        if discovered:
            self.spread_candidates = discovered
        return dict(self.spread_candidates)

    def walk_open_live_orders(
        self,
        step_decrement: float = 0.02,
        min_credit_ratio: float = 0.30,
        min_age_seconds: int = 120,
    ) -> list[dict[str, Any]]:
        """
        Inspect working open multi-leg credit orders on Alpaca and walk their limit price
        down by $0.01-$0.02 toward live market mid (never below min_credit_floor) via
        PATCH /v2/orders/{order_id}.
        """
        if not self._has_live_credentials():
            return []
        walked: list[dict[str, Any]] = []
        open_orders = self.fetch_live_orders(status="open", limit=25)
        now_utc = datetime.now(timezone.utc)

        with httpx.Client(timeout=8.0) as client:
            for ord_obj in open_orders:
                if str(ord_obj.get("order_class") or "").lower() != "mleg":
                    continue
                raw_limit = ord_obj.get("limit_price")
                if raw_limit is None:
                    continue
                limit_val = float(raw_limit)
                # Only walk credit entry orders (negative limit_price in Alpaca mleg)
                if limit_val >= 0:
                    continue

                created_at_str = str(ord_obj.get("created_at") or "")
                if created_at_str:
                    try:
                        created_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        if (now_utc - created_dt).total_seconds() < min_age_seconds:
                            continue
                    except ValueError:
                        pass

                legs = ord_obj.get("legs") or []
                if len(legs) != 2:
                    continue
                sell_leg = next((lg for lg in legs if lg.get("side") == "sell"), None)
                buy_leg = next((lg for lg in legs if lg.get("side") == "buy"), None)
                if not sell_leg or not buy_leg:
                    continue
                sell_sym = str(sell_leg.get("symbol") or "")
                buy_sym = str(buy_leg.get("symbol") or "")
                if not sell_sym or not buy_sym:
                    continue

                try:
                    s_resp = client.get(
                        "https://data.alpaca.markets/v1beta1/options/snapshots",
                        headers=self._headers(),
                        params={"symbols": f"{sell_sym},{buy_sym}"},
                    )
                    if s_resp.status_code != 200:
                        continue
                    snaps = s_resp.json().get("snapshots") or {}
                    s_q = (snaps.get(sell_sym) or {}).get("latestQuote") or {}
                    b_q = (snaps.get(buy_sym) or {}).get("latestQuote") or {}
                    s_bid = float(s_q.get("bp") or 0)
                    s_ask = float(s_q.get("ap") or 0)
                    b_bid = float(b_q.get("bp") or 0)
                    b_ask = float(b_q.get("ap") or 0)
                    s_mid = (s_bid + s_ask) / 2.0
                    b_mid = (b_bid + b_ask) / 2.0
                    live_mid_credit = round(s_mid - b_mid, 2)
                    live_natural_credit = round(s_bid - b_ask, 2)
                    if live_mid_credit <= 0:
                        continue

                    # Parse strikes from OCC symbols (last 8 digits / 1000)
                    s_strike = int(sell_sym[-8:]) / 1000.0
                    b_strike = int(buy_sym[-8:]) / 1000.0
                    spread_width = abs(s_strike - b_strike)
                    min_credit_floor = round(spread_width * min_credit_ratio, 2)

                    # Alpaca Paper does not simulate midpoint price improvement on mleg orders;
                    # walk down toward natural_credit (short_bid - long_ask) on paper-api while
                    # strictly respecting min_credit_floor.
                    is_paper_api = "paper-api" in self.base_url.lower()
                    target_credit = (
                        live_natural_credit
                        if (is_paper_api and live_natural_credit > 0)
                        else live_mid_credit
                    )

                    current_credit = round(abs(limit_val), 2)
                    if current_credit <= target_credit:
                        continue
                    new_credit = round(max(target_credit, current_credit - step_decrement), 2)
                    if new_credit < min_credit_floor or new_credit >= current_credit:
                        continue

                    raw_order_id = str(ord_obj.get("id") or "")
                    patch_resp = client.patch(
                        f"{self.base_url}/v2/orders/{raw_order_id}",
                        headers=self._headers(),
                        json={"limit_price": f"{-new_credit:.2f}"},
                    )
                    if patch_resp.status_code in (200, 201):
                        replaced = patch_resp.json()
                        walked.append(
                            {
                                "old_order_id": raw_order_id,
                                "new_order_id": replaced.get("id", raw_order_id),
                                "legs": f"{sell_sym}/{buy_sym}",
                                "old_credit": current_credit,
                                "new_credit": new_credit,
                                "live_mid_credit": live_mid_credit,
                                "min_credit_floor": min_credit_floor,
                            }
                        )
                    else:
                        logger.info(
                            "Alpaca PATCH /v2/orders/%s response (%s): %s",
                            raw_order_id,
                            patch_resp.status_code,
                            patch_resp.text[:200],
                        )
                except Exception as exc:
                    logger.info("Alpaca live order walk note: %s", exc)

        return walked

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
        meta = dict(metadata or {})
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
                        live_uuid = str(data.get("id") or "")
                        if live_uuid:
                            order_id = f"alpaca-{live_uuid}"
                            meta["alpaca_latest_order_id"] = live_uuid
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

        raw_uuid = str(order.metadata.get("alpaca_latest_order_id") or "") or _extract_alpaca_uuid(order_id)
        if self._has_live_credentials() and raw_uuid:
            is_closing = order.order_type in ("GTC_TAKE_PROFIT", "DEFENSIVE_EXIT", "DEFENSIVE_ROLL")
            signed_limit = abs(order.limit_price) if is_closing else -abs(order.limit_price)
            try:
                with httpx.Client(timeout=8.0) as client:
                    resp = client.patch(
                        f"{self.base_url}/v2/orders/{raw_uuid}",
                        headers=self._headers(),
                        json={"limit_price": f"{signed_limit:.2f}"},
                    )
                    if resp.status_code in (200, 201):
                        replaced = resp.json()
                        if replaced.get("id"):
                            order.metadata["alpaca_latest_order_id"] = str(replaced["id"])
                    else:
                        logger.info(
                            "Alpaca PATCH /v2/orders/%s response (%s): %s",
                            raw_uuid,
                            resp.status_code,
                            resp.text[:200],
                        )
            except Exception as exc:
                logger.warning("Alpaca PATCH /v2/orders/%s note: %s", raw_uuid, exc)

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

        raw_uuid = str(order.metadata.get("alpaca_latest_order_id") or "") or _extract_alpaca_uuid(order_id)
        if self._has_live_credentials() and raw_uuid:
            try:
                with httpx.Client(timeout=8.0) as client:
                    client.delete(
                        f"{self.base_url}/v2/orders/{raw_uuid}",
                        headers=self._headers(),
                    )
            except Exception as exc:
                logger.warning("Alpaca DELETE /v2/orders/%s note: %s", raw_uuid, exc)

        order.status = "CANCELLED"
        return True

    def get_order(self, order_id: str) -> AlpacaSimulatedOrder:
        return self.orders[order_id]
