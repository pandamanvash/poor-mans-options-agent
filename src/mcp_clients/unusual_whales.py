from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


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


ETF_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "GLD", "TLT"}


class UnusualWhalesClient:
    """Client wrapper for Unusual Whales REST API (cloud) and MCP snapshots."""

    def __init__(
        self,
        snapshots: dict[str, FlowSnapshot] | None = None,
        api_key: str | None = None,
    ) -> None:
        self._snapshots = snapshots or dict(LIVE_UW_SNAPSHOTS)
        self.api_key = api_key if api_key is not None else settings.unusual_whales_api_key

    def _has_live_api_key(self) -> bool:
        return bool(
            self.api_key
            and self.api_key not in ("uw-mcp-bridge", "REPLACE_IN_RENDER_DASHBOARD")
        )

    def refresh_live_flow(self, tickers: Sequence[str] | None = None) -> dict[str, bool]:
        """
        Cloud-native Unusual Whales REST API refresh (no local computer needed):
        Queries https://api.unusualwhales.com/api/stock/{ticker}/option-volume
        directly using UNUSUAL_WHALES_API_KEY stored on Render, supporting any
        dynamic watchlist ticker (including index ETFs SPY/QQQ/IWM).
        """
        results: dict[str, bool] = {}
        if not self._has_live_api_key():
            return results
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        target_tickers = list(tickers or self._snapshots.keys())
        with httpx.Client(timeout=8.0) as client:
            for ticker in target_tickers:
                sym = ticker.upper()
                base = self._snapshots.get(sym)
                is_etf = sym in ETF_TICKERS
                default_dte_earnings = 999 if is_etf else 65
                default_earnings_date = "N/A" if is_etf else "2027-02-01"
                try:
                    resp = client.get(
                        f"https://api.unusualwhales.com/api/stock/{sym}/option-volume",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        rows = resp.json().get("data") or []
                        row = rows[0] if isinstance(rows, list) and rows else (rows if isinstance(rows, dict) else {})
                        call_prem = float(row.get("call_premium") or (base.bullish_premium if base else 0.0))
                        put_prem = float(row.get("put_premium") or (base.bearish_premium if base else 0.0))
                        net_call = float(row.get("net_call_premium") or (base.net_call_premium if base else 0.0))
                        net_put = float(row.get("net_put_premium") or (base.net_put_premium if base else 0.0))
                        raw_ivr = row.get("iv_rank")
                        if raw_ivr is not None:
                            iv_rank = float(raw_ivr)
                            if iv_rank <= 1.0:
                                iv_rank *= 100.0
                        else:
                            iv_rank = base.iv_rank if base else 52.0

                        total_prem = max(1.0, call_prem + put_prem)
                        imbalance = (call_prem - put_prem) / total_prem
                        if base is not None:
                            dp_net = base.darkpool_net_flow_1h
                            dp_z = base.darkpool_flow_zscore
                            ask_sweep = base.ask_sweep_ratio
                        else:
                            # Derive institutional directional proxy from live call/put premium imbalance
                            dp_net = round(call_prem - put_prem, 2)
                            dp_z = round(max(-2.5, min(2.5, imbalance * 5.0)), 2)
                            ask_sweep = round(max(0.50, min(0.85, 0.62 + abs(imbalance) * 0.4)), 2)

                        self._snapshots[sym] = FlowSnapshot(
                            ticker=sym,
                            iv_rank=round(iv_rank, 2),
                            days_to_earnings=base.days_to_earnings if base else default_dte_earnings,
                            next_earnings_date=base.next_earnings_date if base else default_earnings_date,
                            bullish_premium=call_prem,
                            bearish_premium=put_prem,
                            net_call_premium=net_call,
                            net_put_premium=net_put,
                            ask_sweep_ratio=ask_sweep,
                            darkpool_net_flow_1h=dp_net,
                            darkpool_flow_zscore=dp_z,
                            overhead_call_wall=base.overhead_call_wall if base else 0.0,
                            put_support_wall=base.put_support_wall if base else 0.0,
                        )
                        results[sym] = True
                    else:
                        results[sym] = False
                except Exception as exc:
                    logger.info("UW REST refresh fallback for %s: %s", sym, exc)
                    results[sym] = False
        return results

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
