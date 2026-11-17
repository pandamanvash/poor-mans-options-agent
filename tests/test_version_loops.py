from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.portfolio_rules import (
    DEFAULT_STRATEGY_VERSION,
    VERSION_PROFILES,
    calculate_dynamic_profit_target_pct,
    evaluate_ml_entry_expectancy,
    get_dynamic_equity_tier,
    get_strategy_profile,
    scale_vertical_contracts,
)
from src.db.session import LightweightSession
from src.mcp_clients.robinhood import RobinhoodClient
from src.mcp_clients.tradingview import TradingViewClient, VixMarketSnapshot
from src.mcp_clients.unusual_whales import UnusualWhalesClient
from src.ml_engine.backtest_agent import BacktestingAgent
from src.workers.monitor_worker import PositionMonitorWorker, PositionStressInput
from src.workers.screener_job import ScreenerJob
from src.workers.version_loop_runner import run_5x_version_evolution_loops


class TestVersionLoopsEvolution(unittest.TestCase):
    def test_default_strategy_version_is_v5_and_all_profiles_exist(self):
        self.assertEqual(DEFAULT_STRATEGY_VERSION, "v5")
        for ver in ("v0", "v1", "v2", "v3", "v4", "v5"):
            self.assertIn(ver, VERSION_PROFILES)
            prof = get_strategy_profile(ver)
            self.assertEqual(prof.version, ver)

    def test_loop2_dynamic_equity_tier_slot_and_contract_scaling(self):
        # In v0 and v1, slots remain static at 3 regardless of equity
        t_v0 = get_dynamic_equity_tier(account_value=9200.0, version="v0")
        self.assertEqual(t_v0["max_active_vertical_spreads"], 3)

        # In v2..v5, slots and contract sizing expand as equity grows from $5k -> $6.5k -> $8.5k -> $11k+
        t1 = get_dynamic_equity_tier(account_value=5200.0, version="v5")
        t2 = get_dynamic_equity_tier(account_value=7200.0, version="v5")
        t3 = get_dynamic_equity_tier(account_value=9500.0, version="v5")
        t4 = get_dynamic_equity_tier(account_value=11800.0, version="v5")

        self.assertLess(t1["max_active_vertical_spreads"], t2["max_active_vertical_spreads"])
        self.assertLess(t2["max_active_vertical_spreads"], t3["max_active_vertical_spreads"])
        self.assertLess(t3["max_active_vertical_spreads"], t4["max_active_vertical_spreads"])
        self.assertIn("IRON_CONDOR", t2["unlocked_strategies"])

        # Contract scaling grows with account value while respecting 40% cash reserve floor
        s_5k = scale_vertical_contracts(
            1.60, 5.0, account_value=5000.0, current_cash=4000.0, version="v5"
        )
        s_9k = scale_vertical_contracts(
            1.60, 5.0, account_value=9000.0, current_cash=7500.0, version="v5", ml_win_prob=0.91
        )
        self.assertEqual(s_5k.contracts, 2)
        self.assertGreater(s_9k.contracts, s_5k.contracts)

    def test_loop1_screener_blocks_regime3_term_structure_slope_without_exhaustion(self):
        # Simulate pre-backwardation VIX=21.5, VIX3M=21.8 (ratio=0.986 > 0.96), rising slope=+0.25, no exhaustion wick
        pre_spike_vix = VixMarketSnapshot(
            vix_price=21.50,
            vix3m_price=21.80,
            vix_ema20=19.50,
            vix_upper_bb=23.80,
            vix_lower_bb=16.50,
            vix_4h_rsi=61.0,
            vix_1h_slope=0.25,
            vix_zscore_20d=1.40,
            printed_rejection_wick=False,
            bullish_divergence_at_floor=False,
        )
        tv = TradingViewClient(vix_snapshot=pre_spike_vix)
        uw = UnusualWhalesClient()
        rh = RobinhoodClient(total_value=6800.0, cash=6800.0, buying_power=5500.0)

        # In v0, Bull Puts are still allowed through; in v1..v5, Bull Puts are blocked by Loop 1 Term-Structure Guard
        session_v0 = LightweightSession()
        screener_v0 = ScreenerJob(uw_client=uw, tv_client=tv, rh_client=rh, version="v0")
        res_v0 = screener_v0.run_pipeline(session=session_v0, trade_date="2026-11-02")
        v0_bp_count = sum(
            1 for i in res_v0["queued_intents"] if i["strategy"] == "BULL_PUT_SPREAD"
        )
        self.assertGreater(v0_bp_count, 0)

        session_v1 = LightweightSession()
        screener_v1 = ScreenerJob(uw_client=uw, tv_client=tv, rh_client=rh, version="v1")
        res_v1 = screener_v1.run_pipeline(session=session_v1, trade_date="2026-11-02")
        v1_bp_count = sum(
            1 for i in res_v1["queued_intents"] if i["strategy"] == "BULL_PUT_SPREAD"
        )
        self.assertEqual(v1_bp_count, 0)

    def test_loop3_and_loop4_monitor_decisions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            monitor = PositionMonitorWorker(
                jsonl_path=Path(tmp_dir) / "stress.jsonl",
                version="v5",
            )

            # Loop 3: Delta-Velocity early cut when short_delta jumps from -0.31 to -0.40 (+0.09/day) with adverse darkpool flow
            pos_vel = PositionStressInput(
                ticker="TSLA",
                strategy="BULL_PUT_SPREAD",
                contracts=3,
                dte=24,
                short_delta=-0.40,
                prev_short_delta=-0.31,
                spread_mark=2.15,
                entry_credit=1.64,
                tv_1h_close=364.0,
                tv_sr_level=362.0,
                atr_14=16.5,
                tv_4h_rsi=39.0,
                uw_darkpool_net_flow_1h=-620_000.0,
                uw_darkpool_zscore=1.45,
                uw_sweep_ratio=0.62,
                vix_level=22.4,
                vix3m_level=22.8,
                vix_zscore_20d=1.5,
                vix_4h_rsi=63.0,
                vix_1h_slope=0.22,
                version="v3",
            )
            dec_vel = monitor.evaluate_position(pos_vel, version="v3")
            self.assertEqual(dec_vel.action, "EARLY_CUT_DELTA_VELOCITY")

            # Loop 4: Fast 42% take-profit when captured in <= 4 days
            pos_fast = PositionStressInput(
                ticker="AMZN",
                strategy="BULL_PUT_SPREAD",
                contracts=3,
                dte=29,
                short_delta=-0.19,
                spread_mark=0.92,  # (1.64 - 0.92)/1.64 = 43.9% captured in 3 days
                entry_credit=1.64,
                tv_1h_close=264.0,
                tv_sr_level=252.27,
                atr_14=9.15,
                tv_4h_rsi=55.0,
                uw_darkpool_net_flow_1h=540_000.0,
                uw_darkpool_zscore=1.20,
                uw_sweep_ratio=0.70,
                vix_level=16.2,
                vix3m_level=18.5,
                vix_zscore_20d=-0.2,
                vix_4h_rsi=45.0,
                vix_1h_slope=-0.08,
                days_held=3,
                version="v4",
            )
            dec_fast = monitor.evaluate_position(pos_fast, version="v4")
            self.assertEqual(dec_fast.action, "FAST_42PCT_TAKE_PROFIT")
            self.assertEqual(
                calculate_dynamic_profit_target_pct(days_held=3, short_delta=0.20, version="v4"),
                0.42,
            )

    def test_loop5_ml_entry_expectancy_gate(self):
        # Strong institutional flow & structure passes v5 gate
        ok_good, prob_good, _ = evaluate_ml_entry_expectancy(
            iv_rank=52.0,
            short_delta=-0.24,
            credit_to_width=0.33,
            vix_term_ratio=0.91,
            vix_1h_slope=-0.10,
            uw_darkpool_zscore=1.35,
            uw_sweep_ratio=0.71,
            dist_to_sr_atr=0.65,
            version="v5",
        )
        self.assertTrue(ok_good)
        self.assertGreaterEqual(prob_good, 0.78)

        # Weak flow + high term ratio is rejected by v5 gate
        ok_bad, prob_bad, _ = evaluate_ml_entry_expectancy(
            iv_rank=43.0,
            short_delta=-0.29,
            credit_to_width=0.31,
            vix_term_ratio=0.98,
            vix_1h_slope=0.20,
            uw_darkpool_zscore=-0.40,
            uw_sweep_ratio=0.52,
            dist_to_sr_atr=0.18,
            version="v5",
        )
        self.assertFalse(ok_bad)
        self.assertLess(prob_bad, 0.78)

    def test_5_loop_progression_monotonically_improves_trades_winrate_and_profit(self):
        agent = BacktestingAgent(starting_capital_usd=5000.0)
        all_summaries = agent.run_all_versions_comparison(window_days=90)

        v0 = all_summaries["v0"]
        v1 = all_summaries["v1"]
        v2 = all_summaries["v2"]
        v3 = all_summaries["v3"]
        v4 = all_summaries["v4"]
        v5 = all_summaries["v5"]

        # Goal 1: More trades executed as the account grows
        self.assertLess(v0.total_trades, v1.total_trades)
        self.assertLess(v1.total_trades, v2.total_trades)
        self.assertLess(v2.total_trades, v3.total_trades)
        self.assertLess(v3.total_trades, v4.total_trades)
        self.assertLess(v4.total_trades, v5.total_trades)
        self.assertEqual(v0.total_trades, 14)
        self.assertEqual(v5.total_trades, 35)

        # Goal 2: Losing trades teased out & win rate improved
        self.assertGreater(v1.win_rate_pct, v0.win_rate_pct)
        self.assertGreater(v5.win_rate_pct, v0.win_rate_pct)
        self.assertLess(v5.losing_trades, v0.losing_trades)
        self.assertLess(v5.max_drawdown_pct, v0.max_drawdown_pct)

        # Goal 3: Net profit increases across every loop
        self.assertLess(v0.net_profit_policy_d_usd, v1.net_profit_policy_d_usd)
        self.assertLess(v1.net_profit_policy_d_usd, v2.net_profit_policy_d_usd)
        self.assertLess(v2.net_profit_policy_d_usd, v3.net_profit_policy_d_usd)
        self.assertLess(v3.net_profit_policy_d_usd, v4.net_profit_policy_d_usd)
        self.assertLess(v4.net_profit_policy_d_usd, v5.net_profit_policy_d_usd)

    def test_version_loop_runner_end_to_end_artifacts_and_robustness(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_reports = Path(tmp_dir) / "reports"
            payload = run_5x_version_evolution_loops(reports_dir=tmp_reports, n_simulations=120)

            self.assertEqual(payload["active_default_version"], "v5")
            self.assertEqual(len(payload["loops"]), 6)

            json_file = tmp_reports / "version_loops" / "loop_results_v0_to_v5.json"
            md_file = tmp_reports / "version_loops" / "evolution_5x_report.md"
            prog_chart = tmp_reports / "charts" / "options_v0_to_v5_evolution_progression.png"
            overlay_chart = tmp_reports / "version_loops" / "v0_to_v5_equity_trajectories.png"

            self.assertTrue(json_file.exists() and json_file.stat().st_size > 1000)
            self.assertTrue(md_file.exists() and md_file.stat().st_size > 1000)
            self.assertTrue(prog_chart.exists() and prog_chart.stat().st_size > 5000)
            self.assertTrue(overlay_chart.exists() and overlay_chart.stat().st_size > 5000)

            comp = payload["summary_comparison"]
            self.assertLess(
                comp["v5"]["mc_95th_pct_max_drawdown_pct"],
                comp["v0"]["mc_95th_pct_max_drawdown_pct"],
            )


if __name__ == "__main__":
    unittest.main()
