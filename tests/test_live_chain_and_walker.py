from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from src.mcp_clients.alpaca import AlpacaPaperClient, _extract_alpaca_uuid, is_third_friday_opex
from src.mcp_clients.tradingview import TechnicalStructure, TradingViewClient


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | list) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestLiveChainAndWalker(unittest.TestCase):
    def test_extract_alpaca_uuid(self) -> None:
        uid = "12345678-1234-5678-1234-567812345678"
        self.assertEqual(_extract_alpaca_uuid(f"alpaca-{uid}"), uid)
        self.assertEqual(_extract_alpaca_uuid(uid), uid)
        self.assertIsNone(_extract_alpaca_uuid("alpaca-abcdef123456"))

    def test_regime4_bear_call_validation_allows_downtrend_with_atr_cushion(self) -> None:
        tv = TradingViewClient(
            structures={
                "SPY": TechnicalStructure(
                    ticker="SPY",
                    price=580.0,
                    daily_ema50=592.0,
                    daily_ema200=560.0,
                    rsi_4h=42.0,
                    atr_14=8.5,
                    horizontal_support_poc=570.0,
                    horizontal_resistance_ceiling=602.0,
                    last_1h_close=580.0,
                )
            }
        )
        # Normal regime rejects because 4H RSI <= 68
        ok_norm, _ = tv.validate_bear_call_structure("SPY", 595.0, panic_regime=False)
        self.assertFalse(ok_norm)

        # Panic regime allows downtrending stock with >= 1 ATR cushion ($580 + $8.5 = $588.5 <= $595)
        ok_panic, reason = tv.validate_bear_call_structure("SPY", 595.0, panic_regime=True)
        self.assertTrue(ok_panic, reason)

        # Panic regime rejects strike too close to spot (< 1 ATR and below resistance)
        ok_tight, _ = tv.validate_bear_call_structure("SPY", 584.0, panic_regime=True)
        self.assertFalse(ok_tight)

    @patch("src.mcp_clients.alpaca.httpx.Client")
    def test_scan_live_option_chains_selects_fresh_delta_spreads(self, mock_client_cls: MagicMock) -> None:
        mock_http = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_http

        contracts_resp = _FakeResp(
            200,
            {
                "option_contracts": [
                    {"symbol": "SPY261224P00560000", "type": "put", "strike_price": "560", "expiration_date": "2026-12-24", "open_interest": "1200"},
                    {"symbol": "SPY261224P00555000", "type": "put", "strike_price": "555", "expiration_date": "2026-12-24", "open_interest": "900"},
                    {"symbol": "SPY261224C00605000", "type": "call", "strike_price": "605", "expiration_date": "2026-12-24", "open_interest": "1500"},
                    {"symbol": "SPY261224C00610000", "type": "call", "strike_price": "610", "expiration_date": "2026-12-24", "open_interest": "1100"},
                ]
            },
        )
        snaps_resp = _FakeResp(
            200,
            {
                "snapshots": {
                    "SPY261224P00560000": {"latestQuote": {"bp": 4.20, "ap": 4.30}, "greeks": {"delta": -0.24}, "impliedVolatility": 0.28},
                    "SPY261224P00555000": {"latestQuote": {"bp": 2.45, "ap": 2.55}, "greeks": {"delta": -0.17}, "impliedVolatility": 0.29},
                    "SPY261224C00605000": {"latestQuote": {"bp": 3.80, "ap": 3.90}, "greeks": {"delta": 0.25}, "impliedVolatility": 0.26},
                    "SPY261224C00610000": {"latestQuote": {"bp": 2.05, "ap": 2.15}, "greeks": {"delta": 0.16}, "impliedVolatility": 0.25},
                }
            },
        )
        acct_resp = _FakeResp(
            200,
            {"account_number": "PA35XZXE8ZZE", "portfolio_value": "10000", "cash": "10000", "options_buying_power": "10000"},
        )

        def fake_get(url: str, **kwargs):
            if "/v2/account" in url:
                return acct_resp
            if "/v2/options/contracts" in url:
                return contracts_resp
            if "/v1beta1/options/snapshots" in url:
                return snaps_resp
            raise AssertionError(f"Unexpected GET {url}")

        mock_http.get.side_effect = fake_get

        client = AlpacaPaperClient(api_key="PKTEST", secret_key="SKTEST")
        scanned = client.scan_live_option_chains(
            tickers=["SPY"],
            panic_regime=False,
            reference_date=date(2026, 11, 19),
        )
        self.assertIn("SPY_BULL_PUT", scanned)
        self.assertIn("SPY_BEAR_CALL", scanned)
        self.assertEqual(scanned["SPY_BULL_PUT"].short_leg.strike_price, 560.0)
        self.assertEqual(scanned["SPY_BULL_PUT"].long_leg.strike_price, 555.0)
        self.assertAlmostEqual(scanned["SPY_BULL_PUT"].mid_credit, 1.75, places=2)
        self.assertEqual(scanned["SPY_BEAR_CALL"].short_leg.strike_price, 605.0)
        self.assertEqual(scanned["SPY_BEAR_CALL"].long_leg.strike_price, 610.0)
        self.assertAlmostEqual(scanned["SPY_BEAR_CALL"].mid_credit, 1.75, places=2)

    @patch("src.mcp_clients.alpaca.httpx.Client")
    def test_walk_open_live_orders_patches_working_credit_limit(self, mock_client_cls: MagicMock) -> None:
        mock_http = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_http

        old_uid = "11111111-2222-3333-4444-555555555555"
        new_uid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        created_old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")

        acct_resp = _FakeResp(
            200,
            {"account_number": "PA35XZXE8ZZE", "portfolio_value": "10000", "cash": "10000", "options_buying_power": "10000"},
        )
        open_orders_resp = _FakeResp(
            200,
            [
                {
                    "id": old_uid,
                    "order_class": "mleg",
                    "limit_price": "-1.62",
                    "created_at": created_old,
                    "legs": [
                        {"symbol": "AAPL261218C00365000", "side": "sell"},
                        {"symbol": "AAPL261218C00370000", "side": "buy"},
                    ],
                }
            ],
        )
        snaps_resp = _FakeResp(
            200,
            {
                "snapshots": {
                    "AAPL261218C00365000": {"latestQuote": {"bp": 3.50, "ap": 3.60}},
                    "AAPL261218C00370000": {"latestQuote": {"bp": 1.95, "ap": 2.05}},
                }
            },
        )
        patch_resp = _FakeResp(200, {"id": new_uid})

        def fake_get(url: str, **kwargs):
            if "/v2/account" in url:
                return acct_resp
            if "/v2/orders" in url:
                return open_orders_resp
            if "/v1beta1/options/snapshots" in url:
                return snaps_resp
            raise AssertionError(f"Unexpected GET {url}")

        mock_http.get.side_effect = fake_get
        mock_http.patch.return_value = patch_resp

        client = AlpacaPaperClient(api_key="PKTEST", secret_key="SKTEST")
        walked = client.walk_open_live_orders(step_decrement=0.02, min_credit_ratio=0.30)

        self.assertEqual(len(walked), 1)
        self.assertEqual(walked[0]["old_order_id"], old_uid)
        self.assertEqual(walked[0]["new_order_id"], new_uid)
        self.assertEqual(walked[0]["old_credit"], 1.62)
        self.assertEqual(walked[0]["new_credit"], 1.60)
        mock_http.patch.assert_called_once()
        _, kwargs = mock_http.patch.call_args
        self.assertEqual(kwargs["json"], {"limit_price": "-1.60"})

    @patch("src.mcp_clients.alpaca.httpx.Client")
    def test_walk_open_live_orders_crosses_mid_toward_natural_on_paper_api(self, mock_client_cls: MagicMock) -> None:
        mock_http = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_http

        old_uid = "11111111-2222-3333-4444-555555555555"
        new_uid = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        created_old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")

        acct_resp = _FakeResp(
            200,
            {"account_number": "PA35XZXE8ZZE", "portfolio_value": "10000", "cash": "10000", "options_buying_power": "10000"},
        )
        # Resting order is ALREADY at live mid (-1.55), while natural credit = 3.50 - 2.05 = 1.45
        open_orders_resp = _FakeResp(
            200,
            [
                {
                    "id": old_uid,
                    "order_class": "mleg",
                    "limit_price": "-1.55",
                    "created_at": created_old,
                    "legs": [
                        {"symbol": "AAPL261218C00365000", "side": "sell"},
                        {"symbol": "AAPL261218C00370000", "side": "buy"},
                    ],
                }
            ],
        )
        snaps_resp = _FakeResp(
            200,
            {
                "snapshots": {
                    "AAPL261218C00365000": {"latestQuote": {"bp": 3.50, "ap": 3.60}},
                    "AAPL261218C00370000": {"latestQuote": {"bp": 1.95, "ap": 2.05}},
                }
            },
        )
        patch_resp = _FakeResp(200, {"id": new_uid})

        def fake_get(url: str, **kwargs):
            if "/v2/account" in url:
                return acct_resp
            if "/v2/orders" in url:
                return open_orders_resp
            if "/v1beta1/options/snapshots" in url:
                return snaps_resp
            raise AssertionError(f"Unexpected GET {url}")

        mock_http.get.side_effect = fake_get
        mock_http.patch.return_value = patch_resp

        client = AlpacaPaperClient(
            api_key="PKTEST",
            secret_key="SKTEST",
            base_url="https://paper-api.alpaca.markets",
        )
        walked = client.walk_open_live_orders(step_decrement=0.02, min_credit_ratio=0.30)

        self.assertEqual(len(walked), 1)
        self.assertEqual(walked[0]["old_credit"], 1.55)
        self.assertEqual(walked[0]["new_credit"], 1.53)
        _, kwargs = mock_http.patch.call_args
        self.assertEqual(kwargs["json"], {"limit_price": "-1.53"})

    def test_uw_seed_from_option_candidates_populates_missing_tickers(self) -> None:
        from src.mcp_clients.robinhood import OptionLegQuote, SpreadCandidateQuote
        from src.mcp_clients.unusual_whales import UnusualWhalesClient

        uw = UnusualWhalesClient()
        self.assertNotIn("SPY", uw._snapshots)
        self.assertNotIn("NVDA", uw._snapshots)

        spy_bc = SpreadCandidateQuote(
            ticker="SPY",
            strategy="BEAR_CALL_SPREAD",
            expiration_date="2026-12-31",
            dte=37,
            short_leg=OptionLegQuote("SPY261231C00750000", "SPY", "2026-12-31", 37, 750.0, "call", 3.50, 3.60, 3.55, 0.24, 0.26, 1200),
            long_leg=OptionLegQuote("SPY261231C00755000", "SPY", "2026-12-31", 37, 755.0, "call", 1.80, 1.90, 1.85, 0.16, 0.25, 900),
        )
        nvda_bc = SpreadCandidateQuote(
            ticker="NVDA",
            strategy="BEAR_CALL_SPREAD",
            expiration_date="2026-12-31",
            dte=37,
            short_leg=OptionLegQuote("NVDA261231C00185000", "NVDA", "2026-12-31", 37, 185.0, "call", 4.10, 4.20, 4.15, 0.25, 0.58, 2500),
            long_leg=OptionLegQuote("NVDA261231C00190000", "NVDA", "2026-12-31", 37, 190.0, "call", 2.35, 2.45, 2.40, 0.17, 0.56, 1800),
        )
        techs = {
            "SPY": TechnicalStructure("SPY", 722.0, 752.0, 737.0, 42.6, 18.4, 686.0, 775.0, 722.0),
            "NVDA": TechnicalStructure("NVDA", 165.0, 194.0, 202.0, 48.5, 15.6, 124.0, 220.0, 165.0),
        }
        seeded = uw.seed_from_option_candidates(
            {"SPY_BEAR_CALL": spy_bc, "NVDA_BEAR_CALL": nvda_bc},
            vix_price=28.60,
            technical_structures=techs,
        )
        self.assertIn("SPY", seeded)
        self.assertIn("NVDA", seeded)
        spy_snap = uw.get_flow_snapshot("SPY")
        self.assertEqual(spy_snap.days_to_earnings, 999)
        self.assertGreaterEqual(spy_snap.iv_rank, 42.0)
        bc_ok, _ = uw.validate_bear_call_flow("SPY", min_iv_rank=42.0)
        self.assertTrue(bc_ok)

    @patch("src.mcp_clients.alpaca.httpx.Client")
    def test_scan_prefers_monthly_opex_and_skips_illiquid_natural_credit(self, mock_client_cls: MagicMock) -> None:
        self.assertTrue(is_third_friday_opex("2026-12-18"))
        self.assertTrue(is_third_friday_opex("2027-01-15"))
        self.assertFalse(is_third_friday_opex("2026-12-31"))

        mock_http = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_http

        acct_resp = _FakeResp(
            200,
            {"account_number": "PA35XZXE8ZZE", "portfolio_value": "10000", "cash": "10000", "options_buying_power": "10000"},
        )
        # Provide both 2026-12-18 (monthly OPEX, 24 DTE) and 2026-12-31 (weekly, 37 DTE closer to 35)
        contracts_resp = _FakeResp(
            200,
            {
                "option_contracts": [
                    {"symbol": "SPY261218C00750000", "type": "call", "strike_price": "750", "expiration_date": "2026-12-18", "open_interest": "15000"},
                    {"symbol": "SPY261218C00755000", "type": "call", "strike_price": "755", "expiration_date": "2026-12-18", "open_interest": "12000"},
                    {"symbol": "SPY261231C00750000", "type": "call", "strike_price": "750", "expiration_date": "2026-12-31", "open_interest": "400"},
                    {"symbol": "SPY261231C00755000", "type": "call", "strike_price": "755", "expiration_date": "2026-12-31", "open_interest": "300"},
                    # Illiquid put pair on Dec 18 (natural credit = 3.00 - 2.60 = 0.40 < 1.25)
                    {"symbol": "SPY261218P00690000", "type": "put", "strike_price": "690", "expiration_date": "2026-12-18", "open_interest": "100"},
                    {"symbol": "SPY261218P00685000", "type": "put", "strike_price": "685", "expiration_date": "2026-12-18", "open_interest": "100"},
                ]
            },
        )
        snaps_resp = _FakeResp(
            200,
            {
                "snapshots": {
                    "SPY261218C00750000": {"latestQuote": {"bp": 3.80, "ap": 3.95}, "greeks": {"delta": 0.24}, "impliedVolatility": 0.25},
                    "SPY261218C00755000": {"latestQuote": {"bp": 2.05, "ap": 2.18}, "greeks": {"delta": 0.16}, "impliedVolatility": 0.24},
                    "SPY261218P00690000": {"latestQuote": {"bp": 3.00, "ap": 4.50}, "greeks": {"delta": -0.24}, "impliedVolatility": 0.28},
                    "SPY261218P00685000": {"latestQuote": {"bp": 1.60, "ap": 2.60}, "greeks": {"delta": -0.17}, "impliedVolatility": 0.29},
                }
            },
        )

        def fake_get(url: str, **kwargs):
            if "/v2/account" in url:
                return acct_resp
            if "/v2/options/contracts" in url:
                return contracts_resp
            if "/v1beta1/options/snapshots" in url:
                return snaps_resp
            raise AssertionError(f"Unexpected GET {url}")

        mock_http.get.side_effect = fake_get
        client = AlpacaPaperClient(api_key="PKTEST", secret_key="SKTEST")
        scanned = client.scan_live_option_chains(
            tickers=["SPY"],
            panic_regime=False,
            reference_date=date(2026, 11, 24),
        )
        # Selected monthly 2026-12-18 over weekly 2026-12-31
        self.assertIn("SPY_BEAR_CALL", scanned)
        self.assertEqual(scanned["SPY_BEAR_CALL"].expiration_date, "2026-12-18")
        # Rejected illiquid put spread ($1.50 leg spread & $0.40 natural credit)
        self.assertNotIn("SPY_BULL_PUT", scanned)

    def test_summarize_live_alpaca_spreads_groups_legs_for_discord(self) -> None:
        from src.workers.scheduled_paper_check import _summarize_live_alpaca_spreads

        positions = [
            {"symbol": "AMZN261231P00230000", "qty": "-2", "unrealized_pl": "800"},
            {"symbol": "AMZN261231P00225000", "qty": "2", "unrealized_pl": "-774"},
            {"symbol": "GLD261231C00340000", "qty": "-2", "unrealized_pl": "-20"},
            {"symbol": "GLD261231C00345000", "qty": "2", "unrealized_pl": "-80"},
        ]
        summary = _summarize_live_alpaca_spreads(positions)
        self.assertEqual(summary["active_spreads_count"], 2)
        self.assertEqual(summary["open_legs_count"], 4)
        self.assertAlmostEqual(summary["total_unrealized_pnl_usd"], -74.0, places=2)
        self.assertTrue(any("AMZN 12/31 225/230P" in s for s in summary["spread_labels"]))
        self.assertTrue(any("GLD 12/31 340/345C" in s for s in summary["spread_labels"]))


if __name__ == "__main__":
    unittest.main()
