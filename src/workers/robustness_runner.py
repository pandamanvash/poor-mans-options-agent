from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ml_engine.backtest_agent import BacktestingAgent
from src.ml_engine.robustness_tester import (
    RobustnessTester,
    fetch_spy_daily_data,
    run_monte_carlo_reshuffle,
    run_parameter_sensitivity,
    run_walk_forward_analysis,
    simulate_sma_crossover_strategy,
)


def run_reference_sma_robustness_suite(
    charts_dir: str | Path = Path("reports/charts"),
    initial_capital: float = 10000.0,
    n_simulations: int = 1000,
    use_yfinance: bool = False,
) -> dict[str, Any]:
    """
    Part A: Standalone Reference Strategy Example (Simple Moving Average Crossover on SPY daily data).
    Generates:
      - `reports/charts/sma_mc_equity_curves.png`
      - `reports/charts/sma_wfa_is_vs_oos.png`
      - `reports/charts/sma_param_heatmap.png`
    """
    out_dir = Path(charts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spy_df = fetch_spy_daily_data(
        start_date="2020-01-01",
        end_date="2026-11-16",
        use_yfinance=use_yfinance,
        seed=42,
    )
    baseline_sma = simulate_sma_crossover_strategy(
        spy_df,
        fast_window=20,
        slow_window=60,
        initial_capital=initial_capital,
    )

    mc_chart_path = out_dir / "sma_mc_equity_curves.png"
    wfa_chart_path = out_dir / "sma_wfa_is_vs_oos.png"
    hm_chart_path = out_dir / "sma_param_heatmap.png"

    mc_res = run_monte_carlo_reshuffle(
        trades_or_returns=baseline_sma["trades"],
        starting_capital=initial_capital,
        n_simulations=n_simulations,
        n_plot_paths=150,
        mode="bootstrap",
        chart_path=mc_chart_path,
        strategy_name="SPY SMA(20/60) Crossover Reference",
        seed=42,
    )

    wfa_res = run_walk_forward_analysis(
        data=spy_df,
        param_grid={
            "fast_window": [10, 15, 20, 25, 30, 35],
            "slow_window": [45, 60, 75, 90, 105, 120],
        },
        window_type="rolling",
        n_splits=6,
        is_ratio=0.70,
        starting_capital=initial_capital,
        chart_path=wfa_chart_path,
        strategy_name="SPY SMA Crossover Reference",
    )

    sens_res = run_parameter_sensitivity(
        data=spy_df,
        param_x_name="fast_window",
        param_x_values=[10, 15, 20, 25, 30, 35, 40],
        param_y_name="slow_window",
        param_y_values=[45, 55, 65, 75, 90, 105, 120],
        starting_capital=initial_capital,
        chart_path=hm_chart_path,
        strategy_name="SPY SMA Crossover Reference",
    )

    return {
        "strategy": "SPY_SMA_Crossover_Reference",
        "baseline_metrics": {
            "fast_window": baseline_sma["fast_window"],
            "slow_window": baseline_sma["slow_window"],
            "sharpe_ratio": baseline_sma["sharpe_ratio"],
            "net_profit_usd": baseline_sma["net_profit_usd"],
            "total_return_pct": baseline_sma["total_return_pct"],
            "annualized_return_pct": baseline_sma["annualized_return_pct"],
            "max_drawdown_pct": baseline_sma["max_drawdown_pct"],
            "win_rate_pct": baseline_sma["win_rate_pct"],
            "total_trades": baseline_sma["total_trades"],
        },
        "monte_carlo": mc_res.to_dict(),
        "walk_forward": wfa_res.to_dict(),
        "parameter_sensitivity": sens_res.to_dict(),
        "charts": {
            "sma_mc_equity_curves": str(mc_chart_path),
            "sma_wfa_is_vs_oos": str(wfa_chart_path),
            "sma_param_heatmap": str(hm_chart_path),
        },
    }


def evaluate_single_version_robustness(
    version_tag: str,
    trades: Sequence[Any],
    starting_capital: float = 5000.0,
    charts_dir: str | Path = Path("reports/charts"),
    version_params: Mapping[str, Any] | None = None,
    n_simulations: int = 1000,
    save_charts: bool = True,
) -> dict[str, Any]:
    """
    Runs Monte Carlo Reshuffles (1,000 runs, 150 plotted curves), Walk-Forward Analysis
    (Rolling & Expanding IS vs OOS), and 2D Parameter Sensitivity Heatmap (`iv_rank_min`
    vs `profit_target_pct` with `delta_stop` context) for a single options strategy version.
    """
    tester = RobustnessTester(
        charts_dir=charts_dir,
        starting_capital=starting_capital,
        n_simulations=n_simulations,
        seed=42,
    )
    fixed = dict(version_params) if version_params else {"delta_stop": 0.43}
    return tester.evaluate_trade_tape(
        trades=trades,
        version_tag=version_tag,
        strategy_label="Poor Man's Options Agent",
        starting_capital=starting_capital,
        param_x_name="iv_rank_min",
        param_x_values=[15, 20, 25, 30, 35, 40],
        param_y_name="profit_target_pct",
        param_y_values=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        fixed_params=fixed,
        save_charts=save_charts,
    )


def _extract_versions_from_payload(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Normalizes various JSON structures (`loop_results_v0_to_v5.json`, `backtest_30d_90d_results.json`,
    or custom dicts) into `{version_tag: {"trades": [...], "starting_capital": 5000.0, "params": {...}}}`.
    """
    extracted: dict[str, dict[str, Any]] = {}

    # Case 1: Dict has "versions" or "loops" sub-collection
    container = payload.get("versions") or payload.get("loops")
    if isinstance(container, Mapping):
        for v_key, v_val in container.items():
            if isinstance(v_val, Mapping) and "trades" in v_val:
                extracted[str(v_key)] = {
                    "trades": list(v_val["trades"]),
                    "starting_capital": float(v_val.get("starting_capital_usd", v_val.get("starting_capital", 5000.0))),
                    "params": dict(v_val.get("params", v_val.get("config", {}))),
                }
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        for idx, item in enumerate(container):
            if isinstance(item, Mapping) and "trades" in item:
                v_key = str(item.get("version", item.get("version_tag", f"v{idx}")))
                extracted[v_key] = {
                    "trades": list(item["trades"]),
                    "starting_capital": float(item.get("starting_capital_usd", item.get("starting_capital", 5000.0))),
                    "params": dict(item.get("params", item.get("config", {}))),
                }

    # Case 2: Top-level keys are "v0", "v1", ..., "v5"
    for k, v in payload.items():
        if str(k).lower().startswith("v") and isinstance(v, Mapping):
            trades = v.get("trades")
            if trades is None and isinstance(v.get("last_90_days"), Mapping):
                trades = v["last_90_days"].get("trades")
            if trades:
                extracted[str(k)] = {
                    "trades": list(trades),
                    "starting_capital": float(v.get("starting_capital_usd", v.get("starting_capital", 5000.0))),
                    "params": dict(v.get("params", v.get("config", {}))),
                }

    # Case 3: Standard `backtest_30d_90d_results.json` structure
    if not extracted and isinstance(payload.get("last_90_days"), Mapping):
        s90 = payload["last_90_days"]
        if "trades" in s90:
            extracted["v0"] = {
                "trades": list(s90["trades"]),
                "starting_capital": float(s90.get("starting_capital_usd", 5000.0)),
                "params": {"delta_stop": 0.43},
            }

    return extracted


def _plot_multi_version_robustness_comparison(
    version_reports: Mapping[str, Mapping[str, Any]],
    out_path: Path,
) -> str:
    """
    Generates a publication-grade 4-panel comparison chart (`options_v0_to_v5_robustness_summary.png`)
    across versions showing:
      1. Median Monte Carlo Ending Equity vs 5th–95th Band
      2. Monte Carlo 95th Percentile Max Drawdown (%) & 95% CDaR (%)
      3. Walk-Forward Efficiency (WFE Ratio) & OOS Consistency (%)
      4. Parameter Sensitivity Plateau Stability Score & Peak Sharpe
    """
    v_keys = list(version_reports.keys())
    x = np.arange(len(v_keys))
    labels = [k.upper() for k in v_keys]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    fig.patch.set_facecolor("#0f172a")
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 1. Ending Equity (Historical + MC P5/P50/P95)
    mc_p50 = [float(version_reports[k]["monte_carlo"]["mc_median_ending_capital"]) for k in v_keys]
    mc_p5 = [float(version_reports[k]["monte_carlo"]["mc_5th_pct_ending_capital"]) for k in v_keys]
    mc_p95 = [float(version_reports[k]["monte_carlo"]["mc_95th_pct_ending_capital"]) for k in v_keys]

    ax1.fill_between(x, mc_p5, mc_p95, color="#0ea5e9", alpha=0.22, label="MC 5th–95th Equity Cone")
    ax1.plot(x, mc_p50, marker="o", linewidth=2.6, color="#facc15", label="MC Median Ending Equity ($)")
    ax1.plot(x, mc_p95, marker="^", linewidth=1.8, linestyle="--", color="#22c55e", label="MC 95th Pct Equity ($)")
    ax1.plot(x, mc_p5, marker="v", linewidth=1.8, linestyle="--", color="#f97316", label="MC 5th Pct Equity ($)")
    for i, val in enumerate(mc_p50):
        ax1.annotate(f"${val:,.0f}", (x[i], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Monte Carlo Terminal Equity Growth Across Versions", fontweight="bold")
    ax1.set_ylabel("Ending Portfolio Equity ($ USD)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.5)

    # 2. Tail Drawdown (MC 95th Pct Max DD & 95% CDaR)
    dd95 = [float(version_reports[k]["monte_carlo"]["mc_95th_pct_max_drawdown_pct"]) for k in v_keys]
    cdar95 = [float(version_reports[k]["monte_carlo"]["mc_cdar_95_pct"]) for k in v_keys]
    w = 0.34
    b1 = ax2.bar(x - w / 2, dd95, width=w, color="#f97316", label="MC 95th Pct Max DD (%)")
    b2 = ax2.bar(x + w / 2, cdar95, width=w, color="#ef4444", label="MC 95% CDaR Tail DD (%)")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Monte Carlo Tail Drawdown Reduction (Lower is Better)", fontweight="bold")
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax2.legend(loc="upper right", fontsize=8.5)

    # 3. Walk-Forward Efficiency & Sharpe Retention
    wfe = [float(version_reports[k]["walk_forward"]["wfe_ratio"]) for k in v_keys]
    sh_ret = [float(version_reports[k]["walk_forward"]["sharpe_retention_ratio"]) for k in v_keys]
    ax3.plot(x, wfe, marker="s", linewidth=2.5, color="#22c55e", label="Walk-Forward Efficiency (OOS/IS)")
    ax3.plot(x, sh_ret, marker="d", linewidth=2.2, color="#38bdf8", linestyle="--", label="OOS/IS Sharpe Retention")
    ax3.axhline(0.60, color="#94a3b8", linestyle=":", linewidth=1.3, label="Robust WFE Threshold (0.60)")
    for i, val in enumerate(wfe):
        ax3.annotate(f"{val:.2f}", (x[i], val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_title("Walk-Forward Out-of-Sample Efficiency Progression", fontweight="bold")
    ax3.set_ylabel("Efficiency / Retention Ratio")
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(loc="lower right", fontsize=8.5)

    # 4. Parameter Sensitivity Plateau Stability Score & Coverage
    plat_score = [float(version_reports[k]["parameter_sensitivity"]["plateau_stability_score"]) for k in v_keys]
    plat_cov = [float(version_reports[k]["parameter_sensitivity"]["plateau_coverage_pct"]) / 100.0 for k in v_keys]
    b3 = ax4.bar(x - w / 2, plat_score, width=w, color="#a855f7", label="Plateau Stability Score (0–1)")
    b4 = ax4.bar(x + w / 2, plat_cov, width=w, color="#2dd4bf", label="Plateau Grid Coverage Ratio")
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax4.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.0)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_title("Parameter Sensitivity Plateau Stability Across Versions", fontweight="bold")
    ax4.set_ylabel("Score / Coverage Ratio")
    ax4.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax4.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Poor Man's Options Agent — Multi-Version Quantitative Robustness Progression",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(out_path)


def evaluate_options_strategy_versions(
    version_results_path: str | Path | Mapping[str, Any] | Sequence[Any] | None = None,
    charts_dir: str | Path = Path("reports/charts"),
    save_per_version_charts: bool = True,
    n_simulations: int = 1000,
) -> dict[str, Any]:
    """
    Part B: Runs all three robustness tests (Monte Carlo 1,000 runs / 150 plotted curves,
    Walk-Forward IS vs OOS, and 2D Parameter Sensitivity Heatmap over `iv_rank_min` vs
    `profit_target_pct` / `delta_stop`) on the `poor-mans-options-agent` strategy versions
    and saves publication-grade high-DPI charts in `reports/charts/`.
    """
    out_dir = Path(charts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    versions_map: dict[str, dict[str, Any]] = {}

    if isinstance(version_results_path, (str, Path)):
        path_obj = Path(version_results_path)
        if path_obj.exists():
            raw_payload = json.loads(path_obj.read_text(encoding="utf-8"))
            if isinstance(raw_payload, Mapping):
                versions_map = _extract_versions_from_payload(raw_payload)
    elif isinstance(version_results_path, Mapping):
        versions_map = _extract_versions_from_payload(version_results_path)
    elif isinstance(version_results_path, Sequence) and not isinstance(version_results_path, (str, bytes)):
        versions_map = {"v0": {"trades": list(version_results_path), "starting_capital": 5000.0, "params": {}}}

    # Fallback: check if `reports/version_loops/loop_results_v0_to_v5.json` or `reports/backtest_30d_90d_results.json` exists
    if not versions_map:
        default_loop_path = Path("reports/version_loops/loop_results_v0_to_v5.json")
        default_bt_path = Path("reports/backtest_30d_90d_results.json")
        if default_loop_path.exists():
            raw_payload = json.loads(default_loop_path.read_text(encoding="utf-8"))
            versions_map = _extract_versions_from_payload(raw_payload)
        elif default_bt_path.exists():
            raw_payload = json.loads(default_bt_path.read_text(encoding="utf-8"))
            versions_map = _extract_versions_from_payload(raw_payload)

    # Final fallback: build directly from BacktestingAgent
    if not versions_map:
        agent = BacktestingAgent()
        tape = agent.build_90d_trade_tape()
        versions_map = {
            "v0": {
                "trades": tape,
                "starting_capital": agent.starting_capital_usd,
                "params": {"delta_stop": 0.43},
            }
        }

    version_reports: dict[str, dict[str, Any]] = {}
    for v_tag, v_info in versions_map.items():
        version_reports[v_tag] = evaluate_single_version_robustness(
            version_tag=v_tag,
            trades=v_info["trades"],
            starting_capital=v_info["starting_capital"],
            charts_dir=out_dir,
            version_params=v_info.get("params"),
            n_simulations=n_simulations,
            save_charts=save_per_version_charts,
        )

    summary_chart_path = out_dir / "options_v0_to_v5_robustness_summary.png"
    _plot_multi_version_robustness_comparison(version_reports, summary_chart_path)

    return {
        "evaluated_versions": list(version_reports.keys()),
        "versions": version_reports,
        "summary_chart": str(summary_chart_path),
    }


def main() -> dict[str, Any]:
    sma_results = run_reference_sma_robustness_suite(
        charts_dir=Path("reports/charts"),
        initial_capital=10000.0,
        n_simulations=1000,
        use_yfinance=False,
    )
    options_results = evaluate_options_strategy_versions(
        version_results_path=None,
        charts_dir=Path("reports/charts"),
        save_per_version_charts=True,
        n_simulations=1000,
    )

    summary_payload = {
        "reference_sma_suite": {
            "baseline_metrics": sma_results["baseline_metrics"],
            "mc_95th_pct_max_drawdown_pct": sma_results["monte_carlo"]["mc_95th_pct_max_drawdown_pct"],
            "mc_cdar_95_pct": sma_results["monte_carlo"]["mc_cdar_95_pct"],
            "wfe_ratio": sma_results["walk_forward"]["wfe_ratio"],
            "plateau_stability_score": sma_results["parameter_sensitivity"]["plateau_stability_score"],
            "charts": sma_results["charts"],
        },
        "options_strategy_suite": {
            "evaluated_versions": options_results["evaluated_versions"],
            "summary_chart": options_results["summary_chart"],
            "per_version_highlights": {
                v: {
                    "mc_median_ending_capital": r["monte_carlo"]["mc_median_ending_capital"],
                    "mc_95th_pct_max_drawdown_pct": r["monte_carlo"]["mc_95th_pct_max_drawdown_pct"],
                    "wfe_ratio": r["walk_forward"]["wfe_ratio"],
                    "plateau_stability_score": r["parameter_sensitivity"]["plateau_stability_score"],
                    "charts": r["charts"],
                }
                for v, r in options_results["versions"].items()
            },
        },
    }
    print(json.dumps(summary_payload, indent=2))
    return summary_payload


if __name__ == "__main__":
    main()
