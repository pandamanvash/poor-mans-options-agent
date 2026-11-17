from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.db.session import LightweightSession
from src.ml_engine.model_trainer import WeeklyModelTrainer
from src.workers.monitor_worker import PositionMonitorWorker, PositionStressInput


class TestLooseStopLogic(unittest.TestCase):
    def test_phantom_vega_protection_holds_loose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            monitor = PositionMonitorWorker(jsonl_path=tmp_path / "stress.jsonl")
            pos = PositionStressInput(
                ticker="AMZN",
                strategy="BULL_PUT_SPREAD",
                contracts=2,
                dte=29,
                short_delta=-0.41,
                spread_mark=2.25,
                entry_credit=1.58,
                tv_1h_close=253.80,
                tv_sr_level=252.27,
                atr_14=9.15,
                tv_4h_rsi=43.5,
                uw_darkpool_net_flow_1h=320_000.0,
                uw_darkpool_zscore=0.90,
                uw_sweep_ratio=0.55,
                vix_level=24.6,
                vix3m_level=25.1,
                vix_zscore_20d=1.80,
                vix_4h_rsi=72.4,
                vix_1h_slope=-0.12,
            )
            decision = monitor.evaluate_position(pos)
            self.assertEqual(decision.action, "HOLD_LOOSE_PHANTOM_VEGA")
            self.assertEqual(decision.polling_interval_seconds, 60)
            self.assertFalse(decision.tv_1h_close_breach)

    def test_early_cut_ml_override_on_backwardation_or_uw_breakout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            monitor = PositionMonitorWorker(jsonl_path=tmp_path / "stress.jsonl")
            pos = PositionStressInput(
                ticker="TSM",
                strategy="BULL_PUT_SPREAD",
                contracts=3,
                dte=25,
                short_delta=-0.44,
                spread_mark=2.35,
                entry_credit=1.76,
                tv_1h_close=374.50,
                tv_sr_level=375.50,
                atr_14=11.40,
                tv_4h_rsi=36.0,
                uw_darkpool_net_flow_1h=-520_000.0,
                uw_darkpool_zscore=2.15,
                uw_sweep_ratio=0.74,
                vix_level=23.5,
                vix3m_level=23.1,
                vix_zscore_20d=1.65,
                vix_4h_rsi=62.0,
                vix_1h_slope=0.25,
            )
            decision = monitor.evaluate_position(pos)
            self.assertEqual(decision.action, "EARLY_CUT_ML_OVERRIDE")
            self.assertEqual(decision.polling_interval_seconds, 60)

    def test_1h_candle_close_beyond_sr_plus_015_atr_triggers_hard_stop(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            monitor = PositionMonitorWorker(jsonl_path=tmp_path / "stress.jsonl")
            pos = PositionStressInput(
                ticker="AMZN",
                strategy="BULL_PUT_SPREAD",
                contracts=2,
                dte=28,
                short_delta=-0.46,
                spread_mark=2.65,
                entry_credit=1.58,
                tv_1h_close=250.20,
                tv_sr_level=252.27,
                atr_14=9.15,
                tv_4h_rsi=34.0,
                uw_darkpool_net_flow_1h=-200_000.0,
                uw_darkpool_zscore=1.10,
                uw_sweep_ratio=0.60,
                vix_level=23.0,
                vix3m_level=23.5,
                vix_zscore_20d=1.40,
                vix_4h_rsi=60.0,
                vix_1h_slope=0.10,
            )
            decision = monitor.evaluate_position(pos)
            self.assertEqual(decision.action, "HARD_STOP_EXIT")
            self.assertTrue(decision.tv_1h_close_breach)

    def test_weekly_ml_counterfactual_trainer_promotes_policy_d(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            session = LightweightSession()
            thresholds_file = tmp_path / "risk_thresholds.json"
            reports_dir = tmp_path / "reports"

            trainer = WeeklyModelTrainer(
                thresholds_path=thresholds_file,
                reports_dir=reports_dir,
            )
            res = trainer.run_weekly_calibration(session=session, audit_date="2026_11_20")
            self.assertEqual(res["best_policy"], "Policy_D_ML_Classifier_GBM")
            self.assertGreater(res["best_ev_per_2contract_spread_usd"], 0.0)
            self.assertTrue(thresholds_file.exists())
            self.assertTrue((reports_dir / "weekly_ml_audit_2026_11_20.md").exists())


if __name__ == "__main__":
    unittest.main()
