from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.mcp_clients.robinhood import (
    DEFAULT_MCP_ACCOUNT_NICKNAME,
    DEFAULT_MCP_ACCOUNT_NUMBER,
    RobinhoodClient,
)
from src.ml_engine.backtest_agent import BacktestingAgent


class TestBacktestAgentAndDefaultAccount(unittest.TestCase):
    def test_robinhood_defaults_strictly_to_mcp_account(self):
        rh = RobinhoodClient()
        summary = rh.get_portfolio_summary()
        self.assertEqual(summary["account_number"], DEFAULT_MCP_ACCOUNT_NUMBER)
        self.assertEqual(summary["account_nickname"], DEFAULT_MCP_ACCOUNT_NICKNAME)
        with self.assertRaises(PermissionError):
            RobinhoodClient(account_number="542891403")

    def test_30_day_and_90_day_backtest_simulations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = BacktestingAgent(starting_capital_usd=5000.0)
            res = agent.run_30d_and_90d_suite(reports_dir=Path(tmp_dir))

            s30 = res["last_30_days"]
            s90 = res["last_90_days"]

            self.assertEqual(s30["account_nickname"], "MCP")
            self.assertEqual(s30["account_number"], "512500422")
            self.assertEqual(s30["calendar_days"], 30)
            self.assertEqual(s90["calendar_days"], 90)

            self.assertGreater(s30["net_profit_policy_d_usd"], s30["net_profit_policy_a_usd"])
            self.assertGreater(s30["net_profit_policy_d_usd"], s30["net_profit_policy_b_usd"])
            self.assertGreater(s90["net_profit_policy_d_usd"], s90["net_profit_policy_a_usd"])
            self.assertGreater(s90["net_profit_policy_d_usd"], s90["net_profit_policy_b_usd"])
            self.assertGreaterEqual(s90["min_cash_reserve_pct_observed"], 40.0)


if __name__ == "__main__":
    unittest.main()
