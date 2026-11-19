from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)


def _compute_ema(values: Sequence[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = float(v) * k + ema * (1.0 - k)
    return round(ema, 4)


def _compute_rsi(values: Sequence[float], period: int = 14) -> float:
    """
    Compute Wilder's Smoothed RSI (RMA), matching TradingView's ta.rsi(close, 14).
    """
    if len(values) < period + 1:
        return 50.0
    changes = [float(values[i]) - float(values[i - 1]) for i in range(1, len(values))]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _aggregate_4h_closes(hourly_closes: Sequence[float]) -> list[float]:
    """Group 1H closes into 4H candle closes (always preserving the latest bar close)."""
    if len(hourly_closes) < 4:
        return [float(c) for c in hourly_closes]
    grouped = [float(hourly_closes[i]) for i in range(3, len(hourly_closes), 4)]
    if (len(hourly_closes) - 1) % 4 != 3:
        grouped.append(float(hourly_closes[-1]))
    return grouped


def _compute_atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> float:
    if len(closes) < 2:
        return 1.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    window = trs[-period:] if len(trs) >= period else trs
    return round(sum(window) / max(1, len(window)), 2)


def _fetch_yahoo_chart_bars(symbol: str, interval: str = "1d", range_str: str = "6mo") -> dict[str, list[float]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OptionsAgent/1.0)"}
    with httpx.Client(timeout=6.0) as client:
        resp = client.get(url, params={"interval": interval, "range": range_str}, headers=headers)
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        opens = [float(x) for x in quote.get("open", []) if x is not None]
        highs = [float(x) for x in quote.get("high", []) if x is not None]
        lows = [float(x) for x in quote.get("low", []) if x is not None]
        closes = [float(x) for x in quote.get("close", []) if x is not None]
        return {"open": opens, "high": highs, "low": lows, "close": closes}


class VixRegime(str, Enum):
    REGIME_1_LOW_VOL = "REGIME_1_LOW_VOL"
    REGIME_2_NORMAL = "REGIME_2_NORMAL"
    REGIME_3_ELEVATED_FEAR = "REGIME_3_ELEVATED_FEAR"
    REGIME_4_PANIC_BACKWARDATION = "REGIME_4_PANIC_BACKWARDATION"


@dataclass(frozen=True)
class VixMarketSnapshot:
    vix_price: float
    vix3m_price: float
    vix_ema20: float
    vix_upper_bb: float
    vix_lower_bb: float
    vix_4h_rsi: float
    vix_1h_slope: float
    vix_zscore_20d: float
    printed_rejection_wick: bool
    bullish_divergence_at_floor: bool

    @property
    def term_structure_ratio(self) -> float:
        if self.vix3m_price <= 0:
            return 1.0
        return round(self.vix_price / self.vix3m_price, 4)


@dataclass(frozen=True)
class TechnicalStructure:
    ticker: str
    price: float
    daily_ema50: float
    daily_ema200: float
    rsi_4h: float
    atr_14: float
    horizontal_support_poc: float
    horizontal_resistance_ceiling: float
    last_1h_close: float


@dataclass(frozen=True)
class VixRegimeAnalysis:
    regime: VixRegime
    term_structure_ratio: float
    allow_pmcc_entry: bool
    allow_bull_put_entry: bool
    allow_bear_call_entry: bool
    vix_exhaustion_bull_put_signal: bool
    vix_floor_bounce_bear_call_signal: bool
    required_cash_reserve_pct: float
    max_bull_put_short_delta: float
    summary: str


def classify_vix_regime(snapshot: VixMarketSnapshot) -> VixRegimeAnalysis:
    """
    Step 1 of Screener Pipeline: CBOE:VIX & CBOE:VIX3M Master Regime Filter.
    """
    ts_ratio = snapshot.term_structure_ratio

    # Regime 4: VIX > 28 or VIX / VIX3M > 1.02 Backwardation
    if snapshot.vix_price > 28.0 or ts_ratio > 1.02:
        return VixRegimeAnalysis(
            regime=VixRegime.REGIME_4_PANIC_BACKWARDATION,
            term_structure_ratio=ts_ratio,
            allow_pmcc_entry=False,
            allow_bull_put_entry=False,
            allow_bear_call_entry=True,
            vix_exhaustion_bull_put_signal=False,
            vix_floor_bounce_bear_call_signal=False,
            required_cash_reserve_pct=0.60,
            max_bull_put_short_delta=0.16,
            summary=(
                f"Systemic panic/backwardation (VIX={snapshot.vix_price:.2f}, "
                f"VIX/VIX3M={ts_ratio:.3f}). Pause standard Bull Puts and raise cash floor to 60%."
            ),
        )

    # Regime 1: VIX < 15 - Low Volatility
    if snapshot.vix_price < 15.0:
        return VixRegimeAnalysis(
            regime=VixRegime.REGIME_1_LOW_VOL,
            term_structure_ratio=ts_ratio,
            allow_pmcc_entry=True,
            allow_bull_put_entry=True,
            allow_bear_call_entry=True,
            vix_exhaustion_bull_put_signal=False,
            vix_floor_bounce_bear_call_signal=False,
            required_cash_reserve_pct=0.40,
            max_bull_put_short_delta=0.28,
            summary=(
                f"Low Volatility (VIX={snapshot.vix_price:.2f}). Ideal window for 200+ DTE PMCC Long Calls; "
                "restrict verticals to single stocks with IVR > 40%."
            ),
        )

    # Regime 2: VIX 15-20 - Normal
    if snapshot.vix_price < 20.0:
        return VixRegimeAnalysis(
            regime=VixRegime.REGIME_2_NORMAL,
            term_structure_ratio=ts_ratio,
            allow_pmcc_entry=snapshot.vix_price < 18.0,
            allow_bull_put_entry=True,
            allow_bear_call_entry=True,
            vix_exhaustion_bull_put_signal=False,
            vix_floor_bounce_bear_call_signal=(
                snapshot.vix_4h_rsi < 35.0 and snapshot.bullish_divergence_at_floor
            ),
            required_cash_reserve_pct=0.40,
            max_bull_put_short_delta=0.28,
            summary=(
                f"Normal Volatility (VIX={snapshot.vix_price:.2f}, VIX/VIX3M={ts_ratio:.3f}). "
                "Run balanced book (1x 2-contract Bull Put + 1x 2-contract Bear Call)."
            ),
        )

    # Regime 3: VIX 20-28 - Elevated Fear / Prime Credit Harvesting
    vix_exhaustion = (
        snapshot.vix_price >= snapshot.vix_upper_bb * 0.96
        and snapshot.vix_4h_rsi > 65.0
        and snapshot.printed_rejection_wick
    )
    vix_floor_bounce = snapshot.vix_4h_rsi < 35.0 and snapshot.bullish_divergence_at_floor

    return VixRegimeAnalysis(
        regime=VixRegime.REGIME_3_ELEVATED_FEAR,
        term_structure_ratio=ts_ratio,
        allow_pmcc_entry=False,
        allow_bull_put_entry=True,
        allow_bear_call_entry=True,
        vix_exhaustion_bull_put_signal=vix_exhaustion,
        vix_floor_bounce_bear_call_signal=vix_floor_bounce,
        required_cash_reserve_pct=0.45,
        max_bull_put_short_delta=0.28,
        summary=(
            f"Elevated Fear / Prime Credit Harvesting (VIX={snapshot.vix_price:.2f}, "
            f"VIX3M={snapshot.vix3m_price:.2f}, Ratio={ts_ratio:.3f}, 4H_RSI={snapshot.vix_4h_rsi:.1f}, "
            f"ExhaustionSignal={vix_exhaustion})."
        ),
    )


# Live TradingView MCP VIX/VIX3M snapshot captured 2026-11-16
LIVE_VIX_SNAPSHOT = VixMarketSnapshot(
    vix_price=22.91,
    vix3m_price=23.02,
    vix_ema20=20.20,
    vix_upper_bb=23.65,
    vix_lower_bb=16.50,
    vix_4h_rsi=65.40,  # Intraday peak rejection wick from 24.52 -> 22.91 close
    vix_1h_slope=-0.23,
    vix_zscore_20d=1.48,
    printed_rejection_wick=True,
    bullish_divergence_at_floor=False,
)

# Live TradingView MCP technical structures captured 2026-11-16
LIVE_TV_STRUCTURES: dict[str, TechnicalStructure] = {
    "AMZN": TechnicalStructure(
        ticker="AMZN",
        price=257.80,
        daily_ema50=256.58,
        daily_ema200=244.69,
        rsi_4h=46.18,
        atr_14=9.15,
        horizontal_support_poc=252.27,
        horizontal_resistance_ceiling=273.23,
        last_1h_close=257.80,
    ),
    "TSLA": TechnicalStructure(
        ticker="TSLA",
        price=379.74,
        daily_ema50=376.55,
        daily_ema200=366.61,
        rsi_4h=47.61,
        atr_14=16.56,
        horizontal_support_poc=362.00,
        horizontal_resistance_ceiling=412.00,
        last_1h_close=379.74,
    ),
    "AAPL": TechnicalStructure(
        ticker="AAPL",
        price=352.78,
        daily_ema50=339.29,
        daily_ema200=301.72,
        rsi_4h=69.20,  # Tested multi-week resistance ceiling $360.83 with overbought 4H structure
        atr_14=8.05,
        horizontal_support_poc=346.00,
        horizontal_resistance_ceiling=360.83,
        last_1h_close=352.78,
    ),
    "TSM": TechnicalStructure(
        ticker="TSM",
        price=373.45,
        daily_ema50=408.83,
        daily_ema200=417.55,
        rsi_4h=37.36,
        atr_14=11.40,
        horizontal_support_poc=375.50,
        horizontal_resistance_ceiling=397.35,
        last_1h_close=373.45,
    ),
}


class TradingViewClient:
    """Client wrapper for TradingView MCP technicals, OHLCV, and VIX regime analysis."""

    def __init__(
        self,
        vix_snapshot: VixMarketSnapshot | None = None,
        structures: dict[str, TechnicalStructure] | None = None,
    ) -> None:
        self._vix_snapshot = vix_snapshot or LIVE_VIX_SNAPSHOT
        self._structures = structures or dict(LIVE_TV_STRUCTURES)

    def get_vix_snapshot(self) -> VixMarketSnapshot:
        return self._vix_snapshot

    def refresh_live_market_data(self, tickers: Sequence[str] | None = None) -> dict[str, bool]:
        """
        Cloud-native live market data refresh (no local computer or TradingView MCP needed):
        Fetches live ^VIX, ^VIX3M, and equity OHLCV bars over HTTPS to compute EMAs,
        Bollinger Bands, RSI, ATR, and S/R levels directly inside Render.
        """
        status: dict[str, bool] = {"vix": False}
        try:
            vix_hourly = _fetch_yahoo_chart_bars("^VIX", interval="1h", range_str="3mo")
            vix_daily = _fetch_yahoo_chart_bars("^VIX", interval="1d", range_str="3mo")
            vix3m_daily = _fetch_yahoo_chart_bars("^VIX3M", interval="1d", range_str="1mo")
            vc_1h = vix_hourly["close"]
            vh_1h = vix_hourly["high"]
            vc_1d = vix_daily["close"]
            v3c_1d = vix3m_daily["close"]
            if len(vc_1d) >= 20 and len(vc_1h) >= 20 and len(v3c_1d) >= 1:
                vix_price = round(vc_1h[-1], 2)
                vix3m_price = round(v3c_1d[-1], 2)
                vc_4h = _aggregate_4h_closes(vc_1h)
                ema20 = _compute_ema(vc_4h[-60:], 20)
                # Compute 20-day Bollinger Bands & Z-Score strictly on daily bars
                w20_daily = vc_1d[-20:]
                mean20_d = sum(w20_daily) / len(w20_daily)
                std20_d = math.sqrt(sum((x - mean20_d) ** 2 for x in w20_daily) / len(w20_daily)) or 1.0
                upper_bb = round(mean20_d + 2.0 * std20_d, 2)
                lower_bb = round(mean20_d - 2.0 * std20_d, 2)
                rsi_4h = _compute_rsi(vc_4h, 14)
                slope_1h = round(vc_1h[-1] - vc_1h[-2], 2)
                zscore_20d = round((vix_price - mean20_d) / std20_d, 2)
                rej_wick = bool(vh_1h and (vh_1h[-1] - vc_1h[-1]) >= 0.40)
                self._vix_snapshot = VixMarketSnapshot(
                    vix_price=vix_price,
                    vix3m_price=vix3m_price,
                    vix_ema20=round(ema20, 2),
                    vix_upper_bb=upper_bb,
                    vix_lower_bb=lower_bb,
                    vix_4h_rsi=rsi_4h,
                    vix_1h_slope=slope_1h,
                    vix_zscore_20d=zscore_20d,
                    printed_rejection_wick=rej_wick,
                    bullish_divergence_at_floor=rsi_4h < 35.0 and slope_1h > 0,
                )
                status["vix"] = True
        except Exception as exc:
            logger.info("Cloud VIX refresh fallback (using last snapshot): %s", exc)

        target_tickers = list(tickers or self._structures.keys())
        for sym in target_tickers:
            try:
                daily = _fetch_yahoo_chart_bars(sym, interval="1d", range_str="1y")
                hourly = _fetch_yahoo_chart_bars(sym, interval="1h", range_str="3mo")
                dc, dh, dl = daily["close"], daily["high"], daily["low"]
                hc = hourly["close"]
                if len(dc) >= 50 and len(hc) >= 20:
                    price = round(hc[-1], 2)
                    ema50 = _compute_ema(dc, 50)
                    ema200 = _compute_ema(dc, min(200, len(dc)))
                    rsi_4h = _compute_rsi(_aggregate_4h_closes(hc), 14)
                    atr_14 = _compute_atr(dh, dl, dc, 14)
                    support_poc = round(min(dl[-20:]), 2)
                    resistance_ceil = round(max(dh[-20:]), 2)
                    self._structures[sym.upper()] = TechnicalStructure(
                        ticker=sym.upper(),
                        price=price,
                        daily_ema50=round(ema50, 2),
                        daily_ema200=round(ema200, 2),
                        rsi_4h=rsi_4h,
                        atr_14=atr_14,
                        horizontal_support_poc=support_poc,
                        horizontal_resistance_ceiling=resistance_ceil,
                        last_1h_close=price,
                    )
                    status[sym.upper()] = True
            except Exception as exc:
                logger.info("Cloud technicals refresh fallback for %s: %s", sym, exc)
                status[sym.upper()] = False
        return status

    def analyze_vix_regime(self) -> VixRegimeAnalysis:
        return classify_vix_regime(self._vix_snapshot)

    def get_technical_structure(self, ticker: str) -> TechnicalStructure:
        symbol = ticker.upper()
        if symbol not in self._structures:
            raise KeyError(f"No TradingView structure found for {symbol}")
        return self._structures[symbol]

    def validate_bull_put_structure(
        self, ticker: str, proposed_short_put_strike: float
    ) -> tuple[bool, str]:
        tech = self.get_technical_structure(ticker)
        above_trend = tech.price > tech.daily_ema50 and tech.price > tech.daily_ema200
        rsi_in_pullback_zone = 35.0 <= tech.rsi_4h <= 48.5
        strike_below_support = proposed_short_put_strike < tech.horizontal_support_poc

        if not above_trend:
            return False, (
                f"{ticker} price ${tech.price:.2f} below 50/200-EMA "
                f"(${tech.daily_ema50:.2f}/${tech.daily_ema200:.2f})"
            )
        if not rsi_in_pullback_zone:
            return False, f"{ticker} 4H RSI {tech.rsi_4h:.2f} outside [35, 48] pullback window"
        if not strike_below_support:
            return False, (
                f"Short put ${proposed_short_put_strike:.2f} is not safely below "
                f"POC/support ${tech.horizontal_support_poc:.2f}"
            )
        return True, (
            f"Price ${tech.price:.2f} > 50/200-EMA, 4H RSI={tech.rsi_4h:.2f}, "
            f"Short Put ${proposed_short_put_strike:.0f} < Support/POC ${tech.horizontal_support_poc:.2f}"
        )

    def validate_bear_call_structure(
        self, ticker: str, proposed_short_call_strike: float
    ) -> tuple[bool, str]:
        tech = self.get_technical_structure(ticker)
        overbought_near_ceiling = tech.rsi_4h > 68.0
        strike_above_resistance = proposed_short_call_strike > tech.horizontal_resistance_ceiling

        if not overbought_near_ceiling:
            return False, f"{ticker} 4H RSI {tech.rsi_4h:.2f} <= 68 overbought threshold"
        if not strike_above_resistance:
            return False, (
                f"Short call ${proposed_short_call_strike:.2f} is not safely above "
                f"resistance ceiling ${tech.horizontal_resistance_ceiling:.2f}"
            )
        return True, (
            f"4H RSI={tech.rsi_4h:.2f} > 68 near ceiling ${tech.horizontal_resistance_ceiling:.2f}, "
            f"Short Call ${proposed_short_call_strike:.0f} anchored above resistance"
        )
