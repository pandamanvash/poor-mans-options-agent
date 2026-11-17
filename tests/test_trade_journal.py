from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import ExecutionMode
from src.db.models import ExecutionIntent
from src.db.session import LightweightSession
from src.mcp_clients.robinhood import RobinhoodClient
from src.ml_engine.trade_journal import TradeJournalLogger
from src.workers.execution_worker import ExecutionWorker


class TestTradeJournal(unittest.TestCase):
    def test_trade_journal_logs_thesis_dates_score_and_overall_grade(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            session = LightweightSession()
            rh = RobinhoodClient(total_value=5207.86, cash=5492.86, buying_power=3992.86)
            worker = ExecutionWorker(rh_client=rh, execution_mode=ExecutionMode.PAPER)

            intent = ExecutionIntent(
                idempotency_key="AMZN_BULL_PUT_SPREAD_2026-12-18_250-245_2026-11-16",
                ticker="AMZN",
                strategy="BULL_PUT_SPREAD",
                expiration_date="2026-12-18",
                dte=32,
                short_strike=250.0,
                long_strike=245.0,
                contracts=2,
                mid_credit=1.58,
                min_credit_floor=1.50,
                bid_ask_spread=0.10,
                expected_50pct_profit_usd=158.0,
                collateral_locked_usd=684.0,
                vix_regime="REGIME_3_ELEVATED_FEAR",
                rationale="VIX Exhaustion + UW Bullish Flow + TV $252.27 POC anchor",
                status="QUEUED",
            )
            session.add(intent)
            session.commit()

            trade = worker.execute_queued_intent(session=session, intent=intent, simulated_fill_step=0)
            self.assertIsNotNone(trade)
            assert trade is not None

            # Simulate closing at 50% GTC target
            trade.status = "CLOSED_PROFIT"
            trade.realized_pnl_usd = 158.0
            trade.current_mark = 0.79

            logger = TradeJournalLogger(
                journal_jsonl_path=tmp / "trade_journal.jsonl",
                report_md_path=tmp / "trade_journal_report.md",
            )
            entries = logger.record_trades_from_session(session=session)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["category"], "Credit Harvesting / Bull Put Vertical")
            self.assertIn("VIX Exhaustion", entries[0]["working_thesis"])
            self.assertEqual(entries[0]["rating_grade"], "A+")
            self.assertGreaterEqual(entries[0]["performance_score"], 95.0)

            summary = logger.generate_report()
            self.assertEqual(summary["overall_rating_grade"], "A+")
            self.assertTrue((tmp / "trade_journal_report.md").exists())


if __name__ == "__main__":
    unittest.main()
