from __future__ import annotations

import unittest

from src.mcp_clients.tradingview import (
    VixMarketSnapshot,
    VixRegime,
    classify_vix_regime,
)


class TestVixRegime(unittest.TestCase):
    def test_regime_1_low_volatility_enables_pmcc(self):
        snap = VixMarketSnapshot(
            vix_price=13.80,
            vix3m_price=16.10,
            vix_ema20=14.20,
            vix_upper_bb=15.50,
            vix_lower_bb=12.80,
            vix_4h_rsi=42.0,
            vix_1h_slope=-0.05,
            vix_zscore_20d=-0.85,
            printed_rejection_wick=False,
            bullish_divergence_at_floor=False,
        )
        res = classify_vix_regime(snap)
        self.assertEqual(res.regime, VixRegime.REGIME_1_LOW_VOL)
        self.assertTrue(res.allow_pmcc_entry)
        self.assertEqual(res.required_cash_reserve_pct, 0.40)

    def test_regime_3_vix_exhaustion_signal_for_bull_puts(self):
        snap = VixMarketSnapshot(
            vix_price=22.91,
            vix3m_price=23.02,
            vix_ema20=20.20,
            vix_upper_bb=23.50,
            vix_lower_bb=16.50,
            vix_4h_rsi=66.8,
            vix_1h_slope=-0.22,
            vix_zscore_20d=1.52,
            printed_rejection_wick=True,
            bullish_divergence_at_floor=False,
        )
        res = classify_vix_regime(snap)
        self.assertEqual(res.regime, VixRegime.REGIME_3_ELEVATED_FEAR)
        self.assertTrue(res.vix_exhaustion_bull_put_signal)
        self.assertTrue(res.allow_bull_put_entry)
        self.assertFalse(res.allow_pmcc_entry)

    def test_regime_4_backwardation_and_panic_filter(self):
        snap = VixMarketSnapshot(
            vix_price=26.50,
            vix3m_price=25.40,
            vix_ema20=22.10,
            vix_upper_bb=26.00,
            vix_lower_bb=18.00,
            vix_4h_rsi=74.0,
            vix_1h_slope=0.45,
            vix_zscore_20d=2.40,
            printed_rejection_wick=False,
            bullish_divergence_at_floor=False,
        )
        res = classify_vix_regime(snap)
        self.assertEqual(res.regime, VixRegime.REGIME_4_PANIC_BACKWARDATION)
        self.assertFalse(res.allow_bull_put_entry)
        self.assertEqual(res.required_cash_reserve_pct, 0.60)
        self.assertEqual(res.max_bull_put_short_delta, 0.16)


if __name__ == "__main__":
    unittest.main()
