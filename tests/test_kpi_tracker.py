from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ml_engine.backtest_agent import BacktestingAgent
from src.ml_engine.kpi_tracker import StrategyKPITracker


class TestKPITracker(unittest.TestCase):
    def test_kpi_tracker_generates_metrics_and_equity_curves(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            tracker = StrategyKPITracker(
                ledger_path=tmp / "ledger.jsonl",
                kpi_json_path=tmp / "kpi_metrics.json",
                kpi_md_path=tmp / "kpi_dashboard.md",
            )
            snap = tracker.compute_and_save_kpis()
            bt = BacktestingAgent(starting_capital_usd=5000.0)
            s30 = bt.run_window_backtest(30)
            s90 = bt.run_window_backtest(90)

            self.assertEqual(snap["account_default"]["nickname"], "MCP")
            self.assertEqual(snap["account_default"]["account_number"], "512500422")
            self.assertEqual(
                snap["core_kpis"]["last_30d_net_pnl_usd"],
                s30.net_profit_policy_d_usd,
            )
            self.assertEqual(
                snap["core_kpis"]["last_90d_net_pnl_usd"],
                s90.net_profit_policy_d_usd,
            )
            self.assertGreater(snap["core_kpis"]["last_90d_net_pnl_usd"], 0.0)
            self.assertGreater(snap["core_kpis"]["phantom_vega_fakeouts_saved"], 0)
            self.assertTrue((tmp / "kpi_metrics.json").exists())
            self.assertTrue((tmp / "kpi_dashboard.md").exists())


if __name__ == "__main__":
    unittest.main()
