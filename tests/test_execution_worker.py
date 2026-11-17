from __future__ import annotations

import unittest

from src.config.settings import ExecutionMode
from src.db.models import ExecutionIntent
from src.db.session import LightweightSession
from src.mcp_clients.robinhood import RobinhoodClient
from src.workers.execution_worker import ExecutionWorker


class TestExecutionWorker(unittest.TestCase):
    def test_smart_mid_price_limit_walker_and_50pct_gtc_staging(self):
        session = LightweightSession()
        rh = RobinhoodClient(total_value=5000.0, cash=5000.0, buying_power=4000.0)
        worker = ExecutionWorker(rh_client=rh, execution_mode=ExecutionMode.PAPER)

        intent = ExecutionIntent(
            idempotency_key="AMZN_BULL_PUT_SPREAD_2026-12-18_250-245_2026-11-16",
            ticker="AMZN",
            strategy="BULL_PUT_SPREAD",
            expiration_date="2026-12-18",
            dte=32,
            short_strike=250.0,
            long_strike=245.0,
            short_option_id="1e1a02e3-ada1-44d9-8778-9841a541dac8",
            long_option_id="0bbc954f-c7d1-48a4-923a-03ddc95e228b",
            contracts=2,
            mid_credit=1.60,
            min_credit_floor=1.52,
            bid_ask_spread=0.12,
            expected_50pct_profit_usd=160.0,
            collateral_locked_usd=680.0,
            vix_regime="REGIME_3_ELEVATED_FEAR",
            status="QUEUED",
        )
        session.add(intent)
        session.commit()

        trade = worker.execute_queued_intent(
            session=session,
            intent=intent,
            simulated_fill_step=2,
            step_decrement=0.02,
        )
        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertEqual(trade.entry_credit, 1.56)
        self.assertEqual(trade.target_close_price, 0.78)
        self.assertIsNotNone(trade.gtc_order_id)
        assert trade.gtc_order_id is not None
        self.assertEqual(rh.get_order(trade.gtc_order_id).status, "OPEN")
        self.assertEqual(rh.get_order(trade.gtc_order_id).limit_price, 0.78)

        order_count_before = len(rh.orders)
        dup_trade = worker.execute_queued_intent(session=session, intent=intent)
        self.assertIsNotNone(dup_trade)
        assert dup_trade is not None
        self.assertEqual(dup_trade.id, trade.id)
        self.assertEqual(len(rh.orders), order_count_before)

    def test_cancel_first_lock_and_aggressive_limit_chaser_on_defensive_exit(self):
        session = LightweightSession()
        rh = RobinhoodClient(total_value=5000.0, cash=5000.0, buying_power=4000.0)
        worker = ExecutionWorker(rh_client=rh, execution_mode=ExecutionMode.PAPER)

        intent = ExecutionIntent(
            idempotency_key="AAPL_BEAR_CALL_SPREAD_2026-12-18_365-370_2026-11-16",
            ticker="AAPL",
            strategy="BEAR_CALL_SPREAD",
            expiration_date="2026-12-18",
            dte=32,
            short_strike=365.0,
            long_strike=370.0,
            contracts=2,
            mid_credit=1.58,
            min_credit_floor=1.50,
            bid_ask_spread=0.11,
            expected_50pct_profit_usd=158.0,
            collateral_locked_usd=684.0,
            vix_regime="REGIME_3_ELEVATED_FEAR",
            status="QUEUED",
        )
        session.add(intent)
        session.commit()

        trade = worker.execute_queued_intent(session=session, intent=intent, simulated_fill_step=0)
        self.assertIsNotNone(trade)
        assert trade is not None
        gtc_id = trade.gtc_order_id
        self.assertIsNotNone(gtc_id)
        assert gtc_id is not None
        self.assertEqual(rh.get_order(gtc_id).status, "OPEN")

        exit_res = worker.execute_defensive_exit_or_roll(
            session=session,
            trade=trade,
            exit_reason="HARD_STOP_EXIT",
            current_mid_debit=2.30,
            chase_steps_required=2,
            chase_step_increment=0.02,
        )
        self.assertTrue(exit_res["gtc_order_cancelled_first"])
        self.assertEqual(rh.get_order(gtc_id).status, "CANCELLED")
        self.assertEqual(exit_res["final_fill_debit"], 2.34)
        self.assertEqual(exit_res["realized_pnl_usd"], -152.00)
        self.assertEqual(trade.status, "CLOSED_STOP")


if __name__ == "__main__":
    unittest.main()
