from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ml_engine.backtest_agent import BacktestingAgent
from src.ml_engine.robustness_tester import (
    RobustnessTester,
    fetch_spy_daily_data,
    run_monte_carlo_reshuffle,
    run_parameter_sensitivity,
    run_walk_forward_analysis,
    simulate_sma_crossover_strategy,
)
from src.workers.robustness_runner import (
    evaluate_options_strategy_versions,
    evaluate_single_version_robustness,
    run_reference_sma_robustness_suite,
)


def test_spy_daily_data_and_sma_simulation() -> None:
    df = fetch_spy_daily_data(start_date="2021-01-01", end_date="2026-11-16", use_yfinance=False, seed=42)
    assert len(df) > 1000
    assert {"Open", "High", "Low", "Close", "Volume", "Return"}.issubset(df.columns)

    res = simulate_sma_crossover_strategy(df, fast_window=20, slow_window=60, initial_capital=10000.0)
    assert res["total_trades"] >= 10
    assert res["sharpe_ratio"] > 0.0
    assert len(res["equity_curve"]) == len(df)


def test_monte_carlo_reshuffle_bootstrap_and_permutation(tmp_path: Path) -> None:
    agent = BacktestingAgent()
    trades = agent.build_90d_trade_tape()
    chart_file = tmp_path / "mc_test.png"

    mc_boot = run_monte_carlo_reshuffle(
        trades_or_returns=trades,
        starting_capital=5000.0,
        n_simulations=1000,
        n_plot_paths=150,
        mode="bootstrap",
        chart_path=chart_file,
        strategy_name="Test Options MC",
        seed=42,
    )

    assert chart_file.exists() and chart_file.stat().st_size > 15_000
    assert mc_boot.n_simulations == 1000
    assert mc_boot.n_trades == len(trades)
    assert mc_boot.mc_5th_pct_ending_capital <= mc_boot.mc_median_ending_capital <= mc_boot.mc_95th_pct_ending_capital
    assert mc_boot.mc_cdar_95_pct >= mc_boot.mc_95th_pct_max_drawdown_pct
    assert 0.0 <= mc_boot.prob_drawdown_gt_20pct <= 100.0
    assert 0.0 <= mc_boot.prob_of_ruin_pct <= 100.0
    # Verify dict-style access works alongside attribute access
    assert mc_boot["mc_95th_pct_max_drawdown_pct"] == mc_boot.mc_95th_pct_max_drawdown_pct
    assert len(mc_boot.percentile_curves["p50"]) == len(trades) + 1

    # Permutation mode without replacement must preserve exact terminal equity across all runs
    mc_perm = run_monte_carlo_reshuffle(
        trades_or_returns=trades,
        starting_capital=5000.0,
        n_simulations=200,
        mode="permutation",
        seed=42,
    )
    assert abs(mc_perm.mc_5th_pct_ending_capital - mc_perm.historical_ending_capital) < 1e-4
    assert abs(mc_perm.mc_95th_pct_ending_capital - mc_perm.historical_ending_capital) < 1e-4
    assert mc_perm.mc_95th_pct_max_drawdown_pct >= mc_perm.historical_max_drawdown_pct


def test_walk_forward_analysis_rolling_and_expanding(tmp_path: Path) -> None:
    df = fetch_spy_daily_data(start_date="2021-01-01", end_date="2026-11-16", use_yfinance=False, seed=42)
    chart_file = tmp_path / "wfa_test.png"

    wfa_roll = run_walk_forward_analysis(
        data=df,
        param_grid={"fast_window": [15, 20, 25], "slow_window": [50, 70, 90]},
        window_type="rolling",
        n_splits=5,
        is_ratio=0.70,
        starting_capital=10000.0,
        chart_path=chart_file,
        strategy_name="Test WFA Rolling",
    )
    assert chart_file.exists() and chart_file.stat().st_size > 15_000
    assert wfa_roll.n_splits == 5
    assert len(wfa_roll.folds) == 5
    assert wfa_roll["wfe_ratio"] == wfa_roll.wfe_ratio

    # Also test expanding window on options trade tape
    trades = BacktestingAgent().build_90d_trade_tape()
    wfa_opt = run_walk_forward_analysis(
        data=trades,
        window_type="expanding",
        n_splits=5,
        is_ratio=0.70,
        starting_capital=5000.0,
        strategy_name="Test Options WFA Expanding",
    )
    assert wfa_opt.n_splits == 5
    assert wfa_opt.wfe_ratio > 0.0
    assert 0.0 <= wfa_opt.oos_consistency_pct <= 100.0


def test_parameter_sensitivity_plateau_vs_spike(tmp_path: Path) -> None:
    chart_file = tmp_path / "heatmap_test.png"

    # Smooth broad plateau surface
    def _broad_plateau_eval(x: float | int, y: float | int) -> dict[str, float]:
        dist = ((float(x) - 25.0) / 15.0) ** 2 + ((float(y) - 0.50) / 0.20) ** 2
        sharpe = max(0.5, 3.2 - 0.8 * dist)
        return {"sharpe_ratio": sharpe, "net_profit_usd": sharpe * 500.0, "win_rate_pct": 82.0}

    # Fragile isolated spike surface
    def _fragile_spike_eval(x: float | int, y: float | int) -> dict[str, float]:
        if float(x) == 25.0 and abs(float(y) - 0.50) < 1e-6:
            return {"sharpe_ratio": 4.0, "net_profit_usd": 2000.0, "win_rate_pct": 90.0}
        return {"sharpe_ratio": 0.2, "net_profit_usd": 50.0, "win_rate_pct": 45.0}

    x_vals = [15, 20, 25, 30, 35]
    y_vals = [0.40, 0.45, 0.50, 0.55, 0.60]

    res_plateau = run_parameter_sensitivity(
        param_x_name="iv_rank_min",
        param_x_values=x_vals,
        param_y_name="profit_target_pct",
        param_y_values=y_vals,
        eval_fn=_broad_plateau_eval,
        chart_path=chart_file,
        strategy_name="Broad Plateau Test",
    )
    res_spike = run_parameter_sensitivity(
        param_x_name="iv_rank_min",
        param_x_values=x_vals,
        param_y_name="profit_target_pct",
        param_y_values=y_vals,
        eval_fn=_fragile_spike_eval,
        strategy_name="Fragile Spike Test",
    )

    assert chart_file.exists() and chart_file.stat().st_size > 15_000
    assert res_plateau.plateau_stability_score > res_spike.plateau_stability_score
    assert res_plateau.plateau_coverage_pct > res_spike.plateau_coverage_pct
    assert res_plateau.is_robust_plateau is True
    assert res_spike.is_robust_plateau is False


def test_robustness_runner_reference_and_options_hooks(tmp_path: Path) -> None:
    sma_out = run_reference_sma_robustness_suite(
        charts_dir=tmp_path,
        initial_capital=10000.0,
        n_simulations=200,
        use_yfinance=False,
    )
    assert (tmp_path / "sma_mc_equity_curves.png").exists()
    assert (tmp_path / "sma_wfa_is_vs_oos.png").exists()
    assert (tmp_path / "sma_param_heatmap.png").exists()
    assert sma_out["monte_carlo"]["n_simulations"] == 200

    trades_v0 = BacktestingAgent().build_90d_trade_tape()
    single_out = evaluate_single_version_robustness(
        version_tag="v0",
        trades=trades_v0,
        starting_capital=5000.0,
        charts_dir=tmp_path,
        n_simulations=200,
        save_charts=True,
    )
    assert (tmp_path / "options_v0_mc_equity_curves.png").exists()
    assert (tmp_path / "options_v0_wfa_is_vs_oos.png").exists()
    assert (tmp_path / "options_v0_param_heatmap.png").exists()
    assert single_out["monte_carlo"]["mc_median_ending_capital"] > 5000.0

    multi_payload = {
        "versions": {
            "v0": {"trades": trades_v0, "starting_capital_usd": 5000.0},
            "v1": {"trades": trades_v0, "starting_capital_usd": 5000.0},
        }
    }
    multi_out = evaluate_options_strategy_versions(
        version_results_path=multi_payload,
        charts_dir=tmp_path,
        save_per_version_charts=True,
        n_simulations=200,
    )
    assert set(multi_out["evaluated_versions"]) == {"v0", "v1"}
    assert (tmp_path / "options_v0_to_v5_robustness_summary.png").exists()
