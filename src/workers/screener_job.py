from __future__ import annotations

from datetime import date
from typing import Any

from src.config.portfolio_rules import (
    DEFAULT_STRATEGY_VERSION,
    PORTFOLIO_CONSTRAINTS,
    evaluate_ml_entry_expectancy,
    get_dynamic_equity_tier,
    get_strategy_profile,
    scale_vertical_contracts,
)
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
    Multi-Version 4-Step Multi-MCP Screening & VIX Regime Pipeline (v0 -> v5):
      Step 1: CBOE:VIX & CBOE:VIX3M Master Regime Filter + v1 Term-Structure Velocity Guard
      Step 2: Institutional Flow, Dark-Pool Z-Score Confirmation & 40-Day Earnings Filter
      Step 3: Technical Structure, ATR Cushion & Strike Anchoring
      Step 4: Dynamic Equity-Tier Slot Scaling (v2+), Credit/Width & ML Plateau Gate (v5)
    """

    def __init__(
        self,
        uw_client: UnusualWhalesClient | None = None,
        tv_client: TradingViewClient | None = None,
        rh_client: RobinhoodClient | None = None,
        version: str = DEFAULT_STRATEGY_VERSION,
    ) -> None:
        self.uw = uw_client or UnusualWhalesClient()
        self.tv = tv_client or TradingViewClient()
        self.rh = rh_client or RobinhoodClient()
        self.version = version.lower().strip()
        self.profile = get_strategy_profile(self.version)

    def run_pipeline(
        self,
        session: Any,
        trade_date: str = "2026-11-16",
        version: str | None = None,
    ) -> dict[str, Any]:
        active_version = (version or self.version).lower().strip()
        profile = get_strategy_profile(active_version)

        # Step 1: VIX & VIX3M Master Regime Filter
        vix_snapshot = self.tv.get_vix_snapshot()
        vix_analysis = self.tv.analyze_vix_regime()
        panic_regime = vix_analysis.regime == VixRegime.REGIME_4_PANIC_BACKWARDATION

        portfolio = self.rh.get_portfolio_summary()
        account_value = float(portfolio["total_value"])
        available_cash = float(portfolio["buying_power"])

        # Step 1B (Loop 2 / v2+): Dynamic Equity-Tier Slot Scaling
        equity_tier = get_dynamic_equity_tier(account_value=account_value, version=active_version)
        max_vertical_slots = int(equity_tier["max_active_vertical_spreads"])

        open_verticals = (
            session.query(Trade)
            .filter(
                Trade.status == "OPEN",
                Trade.strategy.in_(["BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR"]),
            )
            .count()
        )
        slots_available = max(0, max_vertical_slots - open_verticals)

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

            # Loop 1 (v1+): Regime Term-Structure Slope Velocity Guard for Bull Puts
            if (
                profile.enable_regime_slope_filter
                and strategy == "BULL_PUT_SPREAD"
                and vix_analysis.term_structure_ratio > profile.max_bull_put_vix_term_ratio
                and vix_snapshot.vix_1h_slope > profile.max_bull_put_vix_1h_slope
                and not vix_analysis.vix_exhaustion_bull_put_signal
            ):
                rejected_candidates.append(
                    {
                        "key": key,
                        "reason": (
                            f"Loop 1 Term-Structure Guard ({active_version}): "
                            f"VIX/VIX3M={vix_analysis.term_structure_ratio:.3f} > {profile.max_bull_put_vix_term_ratio:.2f} "
                            f"with rising 1H slope={vix_snapshot.vix_1h_slope:+.2f} and no exhaustion wick"
                        ),
                    }
                )
                continue

            # Step 2: Institutional Flow & 40-Day Earnings Filter (Unusual Whales MCP)
            try:
                if strategy == "BULL_PUT_SPREAD":
                    uw_ok, uw_reason = self.uw.validate_bull_put_flow(
                        ticker, min_iv_rank=profile.iv_rank_min
                    )
                else:
                    uw_ok, uw_reason = self.uw.validate_bear_call_flow(
                        ticker, min_iv_rank=profile.iv_rank_min
                    )
            except KeyError as exc:
                rejected_candidates.append({"key": key, "reason": f"UW Data Missing: {exc}"})
                continue

            if not uw_ok:
                rejected_candidates.append({"key": key, "reason": f"UW Filter: {uw_reason}"})
                continue

            flow_snap = self.uw.get_flow_snapshot(ticker)
            aligned_dp_zscore = (
                flow_snap.darkpool_flow_zscore
                if strategy == "BULL_PUT_SPREAD"
                else -flow_snap.darkpool_flow_zscore
            )
            if (
                profile.enable_regime_slope_filter
                and aligned_dp_zscore < profile.min_darkpool_flow_zscore_entry
            ):
                rejected_candidates.append(
                    {
                        "key": key,
                        "reason": (
                            f"Loop 1 Dark-Pool Z-Score Guard ({active_version}): "
                            f"aligned DP Z={aligned_dp_zscore:.2f} < {profile.min_darkpool_flow_zscore_entry:.2f}"
                        ),
                    }
                )
                continue

            # Step 3: Technical Structure & Strike Anchoring (TradingView MCP)
            try:
                if strategy == "BULL_PUT_SPREAD":
                    tv_ok, tv_reason = self.tv.validate_bull_put_structure(
                        ticker, candidate.short_leg.strike_price
                    )
                else:
                    tv_ok, tv_reason = self.tv.validate_bear_call_structure(
                        ticker, candidate.short_leg.strike_price, panic_regime=panic_regime
                    )
            except KeyError as exc:
                rejected_candidates.append({"key": key, "reason": f"TV Data Missing: {exc}"})
                continue

            if not tv_ok:
                rejected_candidates.append({"key": key, "reason": f"TV Filter: {tv_reason}"})
                continue

            tech = self.tv.get_technical_structure(ticker)
            if strategy == "BULL_PUT_SPREAD":
                dist_to_sr_atr = round(
                    (tech.horizontal_support_poc - candidate.short_leg.strike_price)
                    / max(0.01, tech.atr_14),
                    4,
                )
            else:
                dist_to_sr_atr = round(
                    (candidate.short_leg.strike_price - tech.horizontal_resistance_ceiling)
                    / max(0.01, tech.atr_14),
                    4,
                )

            # Loop 5 (v5): Pre-Entry ML Counterfactual Expectancy Gate
            credit_to_width = round(candidate.mid_credit / max(0.01, candidate.spread_width), 4)
            ml_ok, ml_win_prob, ml_reason = evaluate_ml_entry_expectancy(
                iv_rank=flow_snap.iv_rank,
                short_delta=candidate.short_leg.delta,
                credit_to_width=credit_to_width,
                vix_term_ratio=vix_analysis.term_structure_ratio,
                vix_1h_slope=vix_snapshot.vix_1h_slope,
                uw_darkpool_zscore=aligned_dp_zscore,
                uw_sweep_ratio=flow_snap.ask_sweep_ratio,
                dist_to_sr_atr=dist_to_sr_atr,
                version=active_version,
            )
            if not ml_ok:
                rejected_candidates.append({"key": key, "reason": ml_reason})
                continue

            # Step 4: Validate Option Liquidity, 1/3 Width & Dynamic Profit Floor Scaling
            intent = self._evaluate_and_queue_candidate(
                session=session,
                candidate=candidate,
                vix_regime=vix_analysis.regime.value,
                rationale=(
                    f"[{active_version.upper()}|{equity_tier['tier_name']}] "
                    f"{vix_analysis.summary} | UW: {uw_reason} | TV: {tv_reason} | {ml_reason}"
                ),
                account_value=account_value,
                available_cash=available_cash,
                panic_regime=panic_regime,
                trade_date=trade_date,
                version=active_version,
                ml_win_prob=ml_win_prob,
            )
            if intent is not None:
                queued_intents.append(intent)
                available_cash -= intent.collateral_locked_usd
            else:
                rejected_candidates.append(
                    {"key": key, "reason": "Failed option chain liquidity, 1/3 width, or cash floor"}
                )

        return {
            "strategy_version": active_version,
            "equity_tier": equity_tier["tier_name"],
            "max_vertical_slots": max_vertical_slots,
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
        session: Any,
        candidate: SpreadCandidateQuote,
        vix_regime: str,
        rationale: str,
        account_value: float,
        available_cash: float,
        panic_regime: bool,
        trade_date: str,
        version: str = DEFAULT_STRATEGY_VERSION,
        ml_win_prob: float | None = None,
    ) -> ExecutionIntent | None:
        if not (
            PORTFOLIO_CONSTRAINTS.vertical_min_dte
            <= candidate.dte
            <= PORTFOLIO_CONSTRAINTS.vertical_max_dte
        ):
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
                version=version,
                ml_win_prob=ml_win_prob,
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


def main() -> dict[str, Any]:
    import json
    from src.workers.scheduled_paper_check import run_scheduled_paper_cycle

    res = run_scheduled_paper_cycle(mode="screener")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()

