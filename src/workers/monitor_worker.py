from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.portfolio_rules import DEFAULT_STRATEGY_VERSION, get_strategy_profile
from src.config.settings import settings
from src.db.models import StressEvent, Trade


@dataclass(frozen=True)
class PositionStressInput:
    ticker: str
    strategy: str  # "BULL_PUT_SPREAD" or "BEAR_CALL_SPREAD"
    contracts: int
    dte: int
    short_delta: float
    spread_mark: float
    entry_credit: float
    tv_1h_close: float
    tv_sr_level: float
    atr_14: float
    tv_4h_rsi: float
    uw_darkpool_net_flow_1h: float
    uw_darkpool_zscore: float
    uw_sweep_ratio: float
    vix_level: float
    vix3m_level: float
    vix_zscore_20d: float
    vix_4h_rsi: float
    vix_1h_slope: float
    prev_short_delta: float | None = None
    dt_days: float = 1.0
    days_held: int = 7
    version: str = DEFAULT_STRATEGY_VERSION

    @property
    def unrealized_pnl_usd(self) -> float:
        return round((self.entry_credit - self.spread_mark) * 100.0 * self.contracts, 2)

    @property
    def delta_velocity_1d(self) -> float:
        if self.prev_short_delta is None:
            return 0.0
        return round((abs(self.short_delta) - abs(self.prev_short_delta)) / max(0.25, self.dt_days), 4)

    @property
    def vix_term_structure_ratio(self) -> float:
        if self.vix3m_level <= 0:
            return 1.0
        return round(self.vix_level / self.vix3m_level, 4)

    @property
    def dist_to_tv_sr_atr(self) -> float:
        if self.atr_14 <= 0:
            return 0.0
        if self.strategy == "BULL_PUT_SPREAD":
            # Positive means above support, negative means below support
            return round((self.tv_1h_close - self.tv_sr_level) / self.atr_14, 4)
        # For BEAR_CALL_SPREAD, positive means below resistance ceiling, negative means breached above
        return round((self.tv_sr_level - self.tv_1h_close) / self.atr_14, 4)


@dataclass(frozen=True)
class MonitorDecision:
    action: str  # "NORMAL_HOLD", "HOLD_LOOSE", "HOLD_LOOSE_PHANTOM_VEGA", "HOLD_LOOSE_VIX_EXHAUSTION", "EARLY_CUT_ML_OVERRIDE", "EARLY_CUT_DELTA_VELOCITY", "14_DTE_GAMMA_SHIELD", "FAST_42PCT_TAKE_PROFIT", "TREND_RUNNER_65PCT_HARVEST", "HARD_STOP_EXIT", "21_DTE_ROLL"
    polling_interval_seconds: int  # 300s normal, 60s when short_delta >= 0.38
    tv_1h_close_breach: bool
    is_stress_event: bool
    rationale: str


class PositionMonitorWorker:
    """
    Multi-Version Intra-Week Risk & Profit Harvesting Engine (`v0` -> `v5`):
      - Loose-First + VIX + Hard Structural Stop (`v0`..`v5`)
      - Loop 3 (`v3+`): Adaptive Delta-Velocity (`dDelta/dt`) Cut & 14-DTE Gamma Shield
      - Loop 4 (`v4+`): Multi-Speed Velocity Take-Profit (`42%` Fast / `50%` Core / `62.5-65%` Trend Runner)
      - Loop 5 (`v5`): Plateau-Calibrated Early Cut Threshold (`0.41` Delta)
    """

    def __init__(
        self,
        thresholds: dict[str, Any] | None = None,
        jsonl_path: Path | None = None,
        version: str = DEFAULT_STRATEGY_VERSION,
    ) -> None:
        self.thresholds = thresholds or settings.load_risk_thresholds()
        self.jsonl_path = jsonl_path or settings.stress_events_jsonl_path
        self.version = version.lower().strip()

    def evaluate_position(
        self,
        pos: PositionStressInput,
        version: str | None = None,
    ) -> MonitorDecision:
        active_version = (version or pos.version or self.version).lower().strip()
        profile = get_strategy_profile(active_version)

        abs_delta = abs(pos.short_delta)
        stress_trigger = float(self.thresholds.get("stress_polling_delta_trigger", 0.38))
        delta_ceiling = float(self.thresholds.get("baseline_short_delta_ceiling", 0.52))
        ml_early_cut_delta = float(
            self.thresholds.get("ml_early_cut_delta_ceiling", profile.ml_early_cut_delta_ceiling)
        )
        atr_buffer = float(self.thresholds.get("tv_1h_sr_atr_buffer", 0.15))
        roll_dte = int(self.thresholds.get("roll_dte_trigger", profile.roll_dte_trigger))

        polling_interval = 60 if abs_delta >= stress_trigger else 300
        is_stress = abs_delta >= stress_trigger

        # Check 1H structural candle breach beyond TradingView S/R + 0.15 ATR buffer
        if pos.strategy == "BULL_PUT_SPREAD":
            breach_level = pos.tv_sr_level - (atr_buffer * pos.atr_14)
            tv_1h_breach = pos.tv_1h_close < breach_level
        else:
            breach_level = pos.tv_sr_level + (atr_buffer * pos.atr_14)
            tv_1h_breach = pos.tv_1h_close > breach_level

        # 1. Hard Structural Stop or Delta Ceiling Breach
        if tv_1h_breach or abs_delta >= delta_ceiling:
            reasons = []
            if tv_1h_breach:
                reasons.append(
                    f"1H candle closed at ${pos.tv_1h_close:.2f} beyond structural S/R+0.15ATR (${breach_level:.2f})"
                )
            if abs_delta >= delta_ceiling:
                reasons.append(f"Short Delta {abs_delta:.2f} >= ceiling {delta_ceiling:.2f}")
            return MonitorDecision(
                action="HARD_STOP_EXIT",
                polling_interval_seconds=polling_interval,
                tv_1h_close_breach=tv_1h_breach,
                is_stress_event=is_stress,
                rationale=" | ".join(reasons),
            )

        # 2. Loop 3 (v3+): Adaptive Delta-Velocity (dDelta/dt) Pre-Stop Cut
        adverse_flow_direction = (
            pos.uw_darkpool_net_flow_1h < 0
            if pos.strategy == "BULL_PUT_SPREAD"
            else pos.uw_darkpool_net_flow_1h > 0
        )
        if (
            profile.enable_delta_velocity_shield
            and pos.prev_short_delta is not None
            and abs_delta >= profile.delta_velocity_min_abs_delta
            and pos.delta_velocity_1d >= profile.delta_velocity_cut_trigger
            and adverse_flow_direction
        ):
            return MonitorDecision(
                action="EARLY_CUT_DELTA_VELOCITY",
                polling_interval_seconds=60,
                tv_1h_close_breach=tv_1h_breach,
                is_stress_event=True,
                rationale=(
                    f"Loop 3 Delta-Velocity Shield ({active_version}): Short Delta {abs_delta:.2f} "
                    f"expanded +{pos.delta_velocity_1d:.3f}/day (>= +{profile.delta_velocity_cut_trigger:.2f}) "
                    f"with adverse dark-pool flow (${pos.uw_darkpool_net_flow_1h:,.0f}). Cutting before full stop."
                ),
            )

        # 3. Early Cut Override (Backwardation VIX/VIX3M > 1.0 OR Confirmed UW Institutional Breakout)
        backwardation = pos.vix_term_structure_ratio > float(
            self.thresholds.get("vix_backwardation_ratio_trigger", 1.0)
        )
        dp_zscore_min = float(self.thresholds.get("uw_breakout_darkpool_zscore_min", 1.75))
        sweep_min = float(self.thresholds.get("uw_breakout_ask_sweep_ratio_min", 0.68))

        if pos.strategy == "BULL_PUT_SPREAD":
            uw_breakout_against = (
                pos.uw_darkpool_net_flow_1h < 0
                and pos.uw_darkpool_zscore >= dp_zscore_min
                and pos.uw_sweep_ratio >= sweep_min
            )
        else:
            uw_breakout_against = (
                pos.uw_darkpool_net_flow_1h > 0
                and pos.uw_darkpool_zscore >= dp_zscore_min
                and pos.uw_sweep_ratio >= sweep_min
            )

        if abs_delta >= ml_early_cut_delta and (backwardation or uw_breakout_against):
            override_reason = (
                f"VIX Backwardation (VIX/VIX3M={pos.vix_term_structure_ratio:.3f} > 1.0)"
                if backwardation
                else (
                    f"UW Institutional Breakout against strike (DP_Z={pos.uw_darkpool_zscore:.2f}, "
                    f"SweepRatio={pos.uw_sweep_ratio:.2f})"
                )
            )
            return MonitorDecision(
                action="EARLY_CUT_ML_OVERRIDE",
                polling_interval_seconds=polling_interval,
                tv_1h_close_breach=tv_1h_close_breach if "tv_1h_close_breach" in locals() else tv_1h_breach,
                is_stress_event=is_stress,
                rationale=(
                    f"Early Cut Override at {abs_delta:.2f} Delta (>= {ml_early_cut_delta:.2f}): "
                    f"{override_reason}"
                ),
            )

        # 4. Loop 4 (v4+): Multi-Speed Velocity Profit Harvesting
        if profile.enable_multispeed_profit_harvest and pos.entry_credit > 0:
            captured_pct = (pos.entry_credit - pos.spread_mark) / pos.entry_credit
            if (
                pos.days_held <= profile.fast_profit_max_days
                and captured_pct >= profile.fast_profit_target_pct
            ):
                return MonitorDecision(
                    action="FAST_42PCT_TAKE_PROFIT",
                    polling_interval_seconds=polling_interval,
                    tv_1h_close_breach=False,
                    is_stress_event=False,
                    rationale=(
                        f"Loop 4 Fast Velocity Harvest ({active_version}): Captured {captured_pct * 100:.1f}% "
                        f"credit in {pos.days_held}d (<= {profile.fast_profit_max_days}d). Recycling buying power."
                    ),
                )
            if (
                abs_delta <= profile.runner_max_abs_delta
                and pos.dist_to_tv_sr_atr >= 0.80
                and captured_pct >= profile.runner_profit_target_pct
            ):
                return MonitorDecision(
                    action="TREND_RUNNER_65PCT_HARVEST",
                    polling_interval_seconds=polling_interval,
                    tv_1h_close_breach=False,
                    is_stress_event=False,
                    rationale=(
                        f"Loop 4 Trend-Runner Harvest ({active_version}): Captured {captured_pct * 100:.1f}% "
                        f"at low gamma ({abs_delta:.2f} Delta, {pos.dist_to_tv_sr_atr:.2f} ATR cushion)."
                    ),
                )

        # 5. Loop 3 (v3+): 14-DTE Gamma Shield Profit Lock
        if (
            profile.enable_delta_velocity_shield
            and pos.dte <= profile.gamma_shield_dte_threshold
            and pos.entry_credit > 0
            and pos.spread_mark <= (pos.entry_credit * 0.65)
        ):
            return MonitorDecision(
                action="14_DTE_GAMMA_SHIELD",
                polling_interval_seconds=polling_interval,
                tv_1h_close_breach=False,
                is_stress_event=is_stress,
                rationale=(
                    f"Loop 3 Gamma Shield ({active_version}): {pos.ticker} at {pos.dte} DTE "
                    f"(<= {profile.gamma_shield_dte_threshold} DTE) with >=35% credit captured. Locking profit."
                ),
            )

        # 6. 21 DTE (or calibrated 18 DTE) Roll Rule (if not yet at 50% profit target)
        hit_50pct_profit = pos.spread_mark <= (pos.entry_credit * 0.50)
        if pos.dte <= roll_dte and not hit_50pct_profit:
            return MonitorDecision(
                action="21_DTE_ROLL",
                polling_interval_seconds=polling_interval,
                tv_1h_close_breach=tv_1h_breach,
                is_stress_event=is_stress,
                rationale=(
                    f"{pos.ticker} reached {pos.dte} DTE (<= {roll_dte} DTE trigger) "
                    f"with mark ${pos.spread_mark:.2f} > 50% target ${pos.entry_credit * 0.50:.2f}. "
                    "Queueing net-credit roll to next 30-45 DTE cycle."
                ),
            )

        # 7. Phantom Vega Protection (Bull Puts) & Spot-Up / VIX-Up Exhaustion (Bear Calls)
        if is_stress:
            phantom_rsi_min = float(self.thresholds.get("vix_phantom_vega_4h_rsi_min", 68.0))
            if (
                pos.strategy == "BULL_PUT_SPREAD"
                and pos.dist_to_tv_sr_atr >= 0.10
                and pos.vix_4h_rsi >= phantom_rsi_min
                and pos.delta_velocity_1d < 0.07
            ):
                return MonitorDecision(
                    action="HOLD_LOOSE_PHANTOM_VEGA",
                    polling_interval_seconds=polling_interval,
                    tv_1h_close_breach=False,
                    is_stress_event=True,
                    rationale=(
                        f"Phantom Vega Drawdown: {pos.ticker} 1H close ${pos.tv_1h_close:.2f} holds "
                        f"{pos.dist_to_tv_sr_atr:.2f} ATR (>= 0.10) above support ${pos.tv_sr_level:.2f}, "
                        f"VIX 4H RSI={pos.vix_4h_rsi:.1f} >= {phantom_rsi_min:.1f}, "
                        f"dDelta/dt={pos.delta_velocity_1d:+.3f} < 0.07. HOLD LOOSE."
                    ),
                )

            if (
                pos.strategy == "BEAR_CALL_SPREAD"
                and not tv_1h_breach
                and pos.vix_1h_slope >= 0.0
            ):
                return MonitorDecision(
                    action="HOLD_LOOSE_VIX_EXHAUSTION",
                    polling_interval_seconds=polling_interval,
                    tv_1h_close_breach=False,
                    is_stress_event=True,
                    rationale=(
                        f"Spot-Up / VIX-Up Exhaustion: {pos.ticker} tested below 1H resistance breach "
                        f"while VIX 1H slope={pos.vix_1h_slope:+.2f} is rising/flat. HOLD LOOSE."
                    ),
                )

            return MonitorDecision(
                action="HOLD_LOOSE",
                polling_interval_seconds=polling_interval,
                tv_1h_close_breach=False,
                is_stress_event=True,
                rationale=(
                    f"Loose-First Hold: Short Delta {abs_delta:.2f} < {delta_ceiling:.2f}, "
                    f"1H candle above S/R+0.15ATR, no backwardation or UW breakout override."
                ),
            )

        return MonitorDecision(
            action="NORMAL_HOLD",
            polling_interval_seconds=polling_interval,
            tv_1h_close_breach=False,
            is_stress_event=False,
            rationale=f"Healthy position: Short Delta {abs_delta:.2f} < {stress_trigger:.2f}, 5-min polling.",
        )

    def poll_and_log(
        self,
        session: Any,
        pos: PositionStressInput,
        trade: Trade | None = None,
    ) -> MonitorDecision:
        decision = self.evaluate_position(pos)

        if trade is not None:
            trade.current_mark = pos.spread_mark
            trade.short_delta = abs(pos.short_delta)
            trade.unrealized_pnl_usd = pos.unrealized_pnl_usd
            session.flush()

        if decision.is_stress_event:
            now_dt = datetime.now(timezone.utc)
            event = StressEvent(
                timestamp=now_dt,
                ticker=pos.ticker,
                strategy=pos.strategy,
                contracts=pos.contracts,
                dte=pos.dte,
                short_delta=abs(pos.short_delta),
                spread_mark=pos.spread_mark,
                unrealized_pnl=pos.unrealized_pnl_usd,
                dist_to_tv_sr_atr=pos.dist_to_tv_sr_atr,
                tv_1h_close_breach=decision.tv_1h_close_breach,
                tv_4h_rsi=pos.tv_4h_rsi,
                uw_darkpool_net_flow_1h=pos.uw_darkpool_net_flow_1h,
                uw_sweep_ratio=pos.uw_sweep_ratio,
                vix_level=pos.vix_level,
                vix_zscore_20d=pos.vix_zscore_20d,
                vix_4h_rsi=pos.vix_4h_rsi,
                vix_1h_slope=pos.vix_1h_slope,
                vix_term_structure_ratio=pos.vix_term_structure_ratio,
                action_taken=decision.action,
            )
            session.add(event)
            session.flush()

            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": now_dt.isoformat(),
                "ticker": pos.ticker,
                "strategy": pos.strategy,
                "contracts": pos.contracts,
                "dte": pos.dte,
                "short_delta": round(abs(pos.short_delta), 4),
                "spread_mark": pos.spread_mark,
                "unrealized_pnl": pos.unrealized_pnl_usd,
                "dist_to_tv_sr_atr": pos.dist_to_tv_sr_atr,
                "tv_1h_close_breach": decision.tv_1h_close_breach,
                "tv_4h_rsi": pos.tv_4h_rsi,
                "uw_darkpool_net_flow_1h": pos.uw_darkpool_net_flow_1h,
                "uw_sweep_ratio": pos.uw_sweep_ratio,
                "vix_level": pos.vix_level,
                "vix_zscore_20d": pos.vix_zscore_20d,
                "vix_4h_rsi": pos.vix_4h_rsi,
                "vix_1h_slope": pos.vix_1h_slope,
                "vix_term_structure_ratio": pos.vix_term_structure_ratio,
                "action_taken": decision.action,
            }
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

        return decision


def main() -> None:
    import time
    from src.workers.scheduled_paper_check import run_scheduled_paper_cycle

    print("[UnifiedOptionsWorker] Starting combined Execution + Position Monitor loop...")
    tick = 0
    while True:
        summary = run_scheduled_paper_cycle(mode="monitor_and_screen")
        active = summary.get("paper_trades_active", [])
        decisions = summary.get("monitor_decisions", [])
        min_interval = min(
            (int(d.get("polling_interval_seconds", 300)) for d in decisions),
            default=300,
        )
        sleep_s = 60 if min_interval <= 60 else 60
        tick += 1
        print(
            f"[UnifiedOptionsWorker] Tick #{tick}: managed {len(active)} active spread(s), "
            f"evaluated {len(decisions)} monitor decision(s); next tick in {sleep_s}s."
        )
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()

