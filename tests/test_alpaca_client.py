from __future__ import annotations

import unittest

from src.config.settings import ExecutionMode
from src.db.models import ExecutionIntent
from src.db.session import LightweightSession
from src.mcp_clients.alpaca import (
    DEFAULT_ALPACA_PAPER_ACCOUNT,
    AlpacaPaperClient,
    build_mleg_order_payload,
    build_occ_symbol,
)
from src.workers.execution_worker import ExecutionWorker


class TestAlpacaClient(unittest.TestCase):
    def test_occ_symbol_and_mleg_payload_formatting(self):
        short_occ = build_occ_symbol("AMZN", "2026-12-18", "put", 250.0)
        long_occ = build_occ_symbol("AMZN", "2026-12-18", "put", 245.0)
        self.assertEqual(short_occ, "AMZN261218P00250000")
        self.assertEqual(long_occ, "AMZN261218P00245000")

        entry_payload = build_mleg_order_payload(
            idempotency_key="AMZN_BULL_PUT_2026_11_17",
            ticker="AMZN",
            strategy="BULL_PUT_SPREAD",
            expiration_date="2026-12-18",
            short_strike=250.0,
            long_strike=245.0,
            contracts=2,
            limit_price=1.68,
            is_closing=False,
        )
        self.assertEqual(entry_payload["order_class"], "mleg")
        self.assertEqual(entry_payload["limit_price"], "-1.68")
        self.assertEqual(entry_payload["legs"][0]["position_intent"], "sell_to_open")
        self.assertEqual(entry_payload["legs"][1]["position_intent"], "buy_to_open")

        close_payload = build_mleg_order_payload(
            idempotency_key="AMZN_BULL_PUT_CLOSE",
            ticker="AMZN",
            strategy="BULL_PUT_SPREAD",
            expiration_date="2026-12-18",
            short_strike=250.0,
            long_strike=245.0,
            contracts=2,
            limit_price=0.84,
            is_closing=True,
            time_in_force="gtc",
        )
        self.assertEqual(close_payload["limit_price"], "0.84")
        self.assertEqual(close_payload["time_in_force"], "gtc")
        self.assertEqual(close_payload["legs"][0]["position_intent"], "buy_to_close")

    def test_execution_worker_routes_paper_spreads_to_alpaca_client(self):
        session = LightweightSession()
        alpaca = AlpacaPaperClient()
        self.assertEqual(alpaca.get_portfolio_summary()["account_number"], DEFAULT_ALPACA_PAPER_ACCOUNT)

        worker = ExecutionWorker(rh_client=alpaca, execution_mode=ExecutionMode.PAPER)  # type: ignore[arg-type]
        intent = ExecutionIntent(
            idempotency_key="AMZN_BULL_PUT_ALPACA_2026_11_17",
            ticker="AMZN",
            strategy="BULL_PUT_SPREAD",
            expiration_date="2026-12-18",
            dte=31,
            short_strike=250.0,
            long_strike=245.0,
            contracts=2,
            mid_credit=1.68,
            min_credit_floor=1.55,
            bid_ask_spread=0.10,
            expected_50pct_profit_usd=168.0,
            collateral_locked_usd=664.0,
            vix_regime="REGIME_3_ELEVATED_FEAR",
            status="QUEUED",
        )
        session.add(intent)
        session.commit()

        trade = worker.execute_queued_intent(session=session, intent=intent, simulated_fill_step=0)
        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertTrue(str(trade.entry_order_id).startswith("alpaca-"))
        self.assertTrue(str(trade.gtc_order_id).startswith("alpaca-"))
        entry_order = alpaca.get_order(str(trade.entry_order_id))
        self.assertEqual(entry_order.mleg_payload["limit_price"], "-1.68")


if __name__ == "__main__":
    unittest.main()
