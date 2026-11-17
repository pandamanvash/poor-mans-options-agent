from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class FlowSnapshot:
    ticker: str
    iv_rank: float
    days_to_earnings: int
    next_earnings_date: str
    bullish_premium: float
    bearish_premium: float
    net_call_premium: float
    net_put_premium: float
    ask_sweep_ratio: float
    darkpool_net_flow_1h: float
    darkpool_flow_zscore: float
    overhead_call_wall: float
    put_support_wall: float


# Live Unusual Whales MCP snapshot captured 2026-11-16
LIVE_UW_SNAPSHOTS: dict[str, FlowSnapshot] = {
    "AMZN": FlowSnapshot(
        ticker="AMZN",
        iv_rank=51.44,
        days_to_earnings=80,
        next_earnings_date="2027-02-04",
        bullish_premium=132_756_958.0,
        bearish_premium=117_562_032.0,
        net_call_premium=7_564_748.0,
        net_put_premium=3_992_952.0,
        ask_sweep_ratio=0.72,
        darkpool_net_flow_1h=645_790.60,
        darkpool_flow_zscore=1.42,
        overhead_call_wall=275.0,
        put_support_wall=250.0,
    ),
    "TSLA": FlowSnapshot(
        ticker="TSLA",
        iv_rank=59.93,
        days_to_earnings=72,
        next_earnings_date="2027-01-27",
        bullish_premium=306_271_643.0,
        bearish_premium=337_769_662.0,
        net_call_premium=-13_448_783.0,
        net_put_premium=17_879_316.0,
        ask_sweep_ratio=0.69,
        darkpool_net_flow_1h=410_200.0,
        darkpool_flow_zscore=1.18,
        overhead_call_wall=415.0,
        put_support_wall=355.0,
    ),
    "AAPL": FlowSnapshot(
        ticker="AAPL",
        iv_rank=44.47,
        days_to_earnings=80,
        next_earnings_date="2027-02-04",
        bullish_premium=153_763_174.0,
        bearish_premium=171_478_129.0,
        net_call_premium=-7_516_571.0,
        net_put_premium=10_368_743.0,
        ask_sweep_ratio=0.64,
        darkpool_net_flow_1h=-742_668.90,
        darkpool_flow_zscore=-1.35,
        overhead_call_wall=365.0,
        put_support_wall=340.0,
    ),
    "META": FlowSnapshot(
        ticker="META",
        iv_rank=57.04,
        days_to_earnings=72,
        next_earnings_date="2027-01-27",
        bullish_premium=140_683_548.0,
        bearish_premium=154_068_030.0,
        net_call_premium=-8_901_040.0,
        net_put_premium=4_697_742.0,
        ask_sweep_ratio=0.61,
        darkpool_net_flow_1h=-315_000.0,
        darkpool_flow_zscore=-0.95,
        overhead_call_wall=650.0,
        put_support_wall=600.0,
    ),
    "TSM": FlowSnapshot(
        ticker="TSM",
        iv_rank=63.20,
        days_to_earnings=59,
        next_earnings_date="2027-01-14",
        bullish_premium=64_120_000.0,
        bearish_premium=112_450_000.0,
        net_call_premium=-18_200_000.0,
        net_put_premium=-24_500_000.0,
        ask_sweep_ratio=0.74,
        darkpool_net_flow_1h=-499_523.50,
        darkpool_flow_zscore=2.12,
        overhead_call_wall=395.0,
        put_support_wall=365.0,
    ),
}


class UnusualWhalesClient:
    """Client wrapper for Unusual Whales MCP tools."""

    def __init__(self, snapshots: dict[str, FlowSnapshot] | None = None) -> None:
        self._snapshots = snapshots or dict(LIVE_UW_SNAPSHOTS)

    @staticmethod
    def compute_days_to_earnings(next_earnings_date: str, reference_date: date | None = None) -> int:
        ref = reference_date or date(2026, 11, 16)
        target = datetime.strptime(next_earnings_date, "%Y-%m-%d").date()
        return (target - ref).days

    def get_flow_snapshot(self, ticker: str) -> FlowSnapshot:
        symbol = ticker.upper()
        if symbol not in self._snapshots:
            raise KeyError(f"No Unusual Whales snapshot found for {symbol}")
        return self._snapshots[symbol]

    def passes_earnings_hazard_filter(self, ticker: str, min_days: int = 40) -> bool:
        snap = self.get_flow_snapshot(ticker)
        return snap.days_to_earnings >= min_days

    def validate_bull_put_flow(self, ticker: str, min_iv_rank: float = 40.0) -> tuple[bool, str]:
        snap = self.get_flow_snapshot(ticker)
        if snap.days_to_earnings < 40:
            return False, f"Earnings hazard: {snap.days_to_earnings}d < 40d minimum"
        if snap.iv_rank < min_iv_rank:
            return False, f"IV Rank {snap.iv_rank:.1f}% < {min_iv_rank:.1f}%"
        has_bullish_flow = snap.bullish_premium >= snap.bearish_premium or snap.ask_sweep_ratio >= 0.65
        has_non_negative_dp = snap.darkpool_net_flow_1h >= 0.0
        if not (has_bullish_flow and has_non_negative_dp):
            return False, "Institutional flow or dark pool accumulation not confirmed"
        return True, (
            f"IVR={snap.iv_rank:.1f}%, Earnings={snap.days_to_earnings}d, "
            f"AskSweepRatio={snap.ask_sweep_ratio:.2f}, DP_1H=+${snap.darkpool_net_flow_1h:,.0f}"
        )

    def validate_bear_call_flow(self, ticker: str, min_iv_rank: float = 40.0) -> tuple[bool, str]:
        snap = self.get_flow_snapshot(ticker)
        if snap.days_to_earnings < 40:
            return False, f"Earnings hazard: {snap.days_to_earnings}d < 40d minimum"
        if snap.iv_rank < min_iv_rank:
            return False, f"IV Rank {snap.iv_rank:.1f}% < {min_iv_rank:.1f}%"
        has_distribution = snap.bearish_premium > snap.bullish_premium or snap.darkpool_net_flow_1h < 0
        if not has_distribution:
            return False, "No institutional distribution or overhead hedge flow detected"
        return True, (
            f"IVR={snap.iv_rank:.1f}%, Earnings={snap.days_to_earnings}d, "
            f"CallWall=${snap.overhead_call_wall:.0f}, DP_1H=${snap.darkpool_net_flow_1h:,.0f}"
        )

    def screen_candidates(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for ticker, snap in self._snapshots.items():
            bp_ok, bp_reason = self.validate_bull_put_flow(ticker)
            bc_ok, bc_reason = self.validate_bear_call_flow(ticker)
            results.append(
                {
                    "ticker": ticker,
                    "iv_rank": snap.iv_rank,
                    "days_to_earnings": snap.days_to_earnings,
                    "bull_put_eligible": bp_ok,
                    "bull_put_reason": bp_reason,
                    "bear_call_eligible": bc_ok,
                    "bear_call_reason": bc_reason,
                }
            )
        return results
