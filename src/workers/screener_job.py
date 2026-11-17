from __future__ import annotations

from datetime import date
from typing import Any

from src.config.portfolio_rules import PORTFOLIO_CONSTRAINTS, scale_vertical_contracts
from src.db.models import ExecutionIntent, Trade
from src.mcp_clients.robinhood import RobinhoodClient, SpreadCandidateQuote
from src.mcp_clients.tradingview import TradingViewClient, VixRegime
from src.mcp_clients.unusual_whales import UnusualWhalesClient


def build_idempotency_key(
    ticker: str,
    strategy: str,
    expiration_date: str,
    short_strike: float,
    long_strike: float,
    trade_date: str | None = None,
) -> str:
    dt = trade_date or date.today().isoformat()
    strikes = f"{short_strike:.0f}-{long_strike:.0f}"
    return f"{ticker.upper()}_{strategy.upper()}_{expiration_date}_{strikes}_{dt}"


class ScreenerJob:
    """
    4-Step Multi-MCP Screening & VIX Regime Pipeline:
      Step 1: CBOE:VIX & CBOE:VIX3M Master Regime Filter (TradingView MCP)
      Step 2: Institutional Flow & 40-Day Earnings Hazard Filter (Unusual Whales MCP)
      Step 3: Technical Structure & Strike Anchoring (TradingView MCP)
      Step 4: Option Chain Liquidity, >=$100 50%-Target Scaling & Queue Execution Intent
    """

    def __init__(
        self,
        uw_client: UnusualWhalesClient | None = None,
        tv_client: TradingViewClient | None = None,
        rh_client: RobinhoodClient | None = None,
    ) -> None:
        self.uw = uw_client or UnusualWhalesClient()
        self.tv = tv_client or TradingViewClient()
        self.rh = rh_client or RobinhoodClient()

    def run_pipeline(
        self,
        session: Session,
        trade_date: str = "2026-11-16",
    ) -> dict[str, Any]:
        # Step 1: VIX & VIX3M Master Regime Filter
        vix_analysis = self.tv.analyze_vix_regime()
        panic_regime = vix_analysis.regime == VixRegime.REGIME_4_PANIC_BACKWARDATION

        portfolio = self.rh.get_portfolio_summary()
        account_value = float(portfolio["total_value"])
        available_cash = float(portfolio["buying_power"])

        open_verticals = (
            session.query(Trade)
            .filter(
                Trade.status == "OPEN",
                Trade.strategy.in_(["BULL_PUT_SPREAD", "BEAR_CALL_SPREAD"]),
            )
            .count()
        )
        slots_available = max(0, PORTFOLIO_CONSTRAINTS.max_active_vertical_spreads - open_verticals)

        queued_intents: list[ExecutionIntent] = []
        rejected_candidates: list[dict[str, str]] = []

        for key, candidate in self.rh.spread_candidates.items():
            if len(queued_intents) >= slots_available:
                break

            ticker = candidate.ticker
            strategy = candidate.strategy

            # Regime eligibility check
            if strategy == "BULL_PUT_SPREAD" and not vix_analysis.allow_bull_put_entry:
                rejected_candidates.append(
                    {"key": key, "reason": f"Blocked by VIX Regime ({vix_analysis.regime.value})"}
                )
                continue
            if strategy == "BEAR_CALL_SPREAD" and not vix_analysis.allow_bear_call_entry:
                rejected_candidates.append(
                    {"key": key, "reason": f"Blocked by VIX Regime ({vix_analysis.regime.value})"}
                )
                continue

            # Step 2: Institutional Flow & 40-Day Earnings Filter (Unusual Whales MCP)
            if strategy == "BULL_PUT_SPREAD":
                uw_ok, uw_reason = self.uw.validate_bull_put_flow(ticker)
            else:
                uw_ok, uw_reason = self.uw.validate_bear_call_flow(ticker)

            if not uw_ok:
                rejected_candidates.append({"key": key, "reason": f"UW Filter: {uw_reason}"})
                continue

            # Step 3: Technical Structure & Strike Anchoring (TradingView MCP)
            if strategy == "BULL_PUT_SPREAD":
                tv_ok, tv_reason = self.tv.validate_bull_put_structure(
                    ticker, candidate.short_leg.strike_price
                )
            else:
                tv_ok, tv_reason = self.tv.validate_bear_call_structure(
                    ticker, candidate.short_leg.strike_price
                )

            if not tv_ok:
                rejected_candidates.append({"key": key, "reason": f"TV Filter: {tv_reason}"})
                continue

            # Step 4: Validate Option Liquidity, 1/3 Width & >$100 Profit Floor Scaling
            intent = self._evaluate_and_queue_candidate(
                session=session,
                candidate=candidate,
                vix_regime=vix_analysis.regime.value,
                rationale=f"{vix_analysis.summary} | UW: {uw_reason} | TV: {tv_reason}",
                account_value=account_value,
                available_cash=available_cash,
                panic_regime=panic_regime,
                trade_date=trade_date,
            )
            if intent is not None:
                queued_intents.append(intent)
                available_cash -= intent.collateral_locked_usd
            else:
                rejected_candidates.append(
                    {"key": key, "reason": "Failed option chain liquidity, 1/3 width, or cash floor"}
                )

        return {
            "vix_regime": vix_analysis.regime.value,
            "vix_summary": vix_analysis.summary,
            "term_structure_ratio": vix_analysis.term_structure_ratio,
            "queued_intents_count": len(queued_intents),
            "queued_intents": [
                {
                    "idempotency_key": i.idempotency_key,
                    "ticker": i.ticker,
                    "strategy": i.strategy,
                    "expiration_date": i.expiration_date,
                    "dte": i.dte,
                    "short_strike": i.short_strike,
                    "long_strike": i.long_strike,
                    "contracts": i.contracts,
                    "mid_credit": i.mid_credit,
                    "expected_50pct_profit_usd": i.expected_50pct_profit_usd,
                    "collateral_locked_usd": i.collateral_locked_usd,
                }
                for i in queued_intents
            ],
            "rejected_candidates": rejected_candidates,
        }

    def _evaluate_and_queue_candidate(
        self,
        session: Session,
        candidate: SpreadCandidateQuote,
        vix_regime: str,
        rationale: str,
        account_value: float,
        available_cash: float,
        panic_regime: bool,
        trade_date: str,
    ) -> ExecutionIntent | None:
        if not (PORTFOLIO_CONSTRAINTS.vertical_min_dte <= candidate.dte <= PORTFOLIO_CONSTRAINTS.vertical_max_dte):
            return None
        if candidate.net_bid_ask_spread > PORTFOLIO_CONSTRAINTS.max_bid_ask_spread_usd:
            return None

        try:
            sizing = scale_vertical_contracts(
                credit_per_contract=candidate.mid_credit,
                spread_width=candidate.spread_width,
                account_value=account_value,
                current_cash=available_cash,
                panic_regime=panic_regime,
            )
        except ValueError:
            return None

        if not (sizing.meets_one_third_width and sizing.meets_profit_floor):
            return None

        idem_key = build_idempotency_key(
            ticker=candidate.ticker,
            strategy=candidate.strategy,
            expiration_date=candidate.expiration_date,
            short_strike=candidate.short_leg.strike_price,
            long_strike=candidate.long_leg.strike_price,
            trade_date=trade_date,
        )

        existing = (
            session.query(ExecutionIntent)
            .filter(ExecutionIntent.idempotency_key == idem_key)
            .first()
        )
        if existing is not None:
            return existing

        intent = ExecutionIntent(
            idempotency_key=idem_key,
            ticker=candidate.ticker,
            strategy=candidate.strategy,
            intent_type="ENTRY_INTENT",
            expiration_date=candidate.expiration_date,
            dte=candidate.dte,
            short_strike=candidate.short_leg.strike_price,
            long_strike=candidate.long_leg.strike_price,
            short_option_id=candidate.short_leg.instrument_id,
            long_option_id=candidate.long_leg.instrument_id,
            contracts=sizing.contracts,
            mid_credit=sizing.credit_per_contract,
            min_credit_floor=sizing.min_credit_floor,
            bid_ask_spread=candidate.net_bid_ask_spread,
            expected_50pct_profit_usd=sizing.expected_50pct_profit_usd,
            collateral_locked_usd=sizing.collateral_locked_usd,
            vix_regime=vix_regime,
            rationale=rationale,
            status="QUEUED",
        )
        session.add(intent)
        session.flush()
        return intent
