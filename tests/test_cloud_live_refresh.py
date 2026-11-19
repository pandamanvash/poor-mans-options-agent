from __future__ import annotations

import unittest

from src.mcp_clients.tradingview import _compute_atr, _compute_ema, _compute_rsi


class TestCloudLiveIndicators(unittest.TestCase):
    def test_cloud_ema_rsi_atr_helpers(self):
        prices = [100.0 + i * 0.5 for i in range(30)]
        ema = _compute_ema(prices, 10)
        self.assertGreater(ema, 105.0)
        self.assertLess(ema, 115.0)

        rsi = _compute_rsi(prices, 14)
        self.assertEqual(rsi, 100.0)

        highs = [p + 1.0 for p in prices]
        lows = [p - 1.0 for p in prices]
        atr = _compute_atr(highs, lows, prices, 14)
        self.assertEqual(atr, 2.0)


if __name__ == "__main__":
    unittest.main()
