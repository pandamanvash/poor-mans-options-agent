from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import ExecutionMode, settings
from src.db.models import ExecutionIntent, Trade
from src.db.session import LightweightSession, _build_engine, get_db_session, init_db
from src.mcp_clients.alpaca import DEFAULT_WATCHLIST, AlpacaPaperClient
from src.mcp_clients.robinhood import DEFAULT_MCP_ACCOUNT_NUMBER, RobinhoodClient
from src.mcp_clients.tradingview import TradingViewClient, VixRegime
from src.mcp_clients.unusual_whales import UnusualWhalesClient
from src.ml_engine.kpi_tracker import StrategyKPITracker
from src.ml_engine.model_trainer import WeeklyModelTrainer
from src.ml_engine.trade_journal import TradeJournalLogger
from src.notifications.notifier import Notifier
from src.workers.execution_worker import ExecutionWorker
from src.workers.monitor_worker import PositionMonitorWorker, PositionStressInput
from src.workers.screener_job import ScreenerJob


def run_scheduled_paper_cycle(mode: str = "monitor_and_screen") -> dict[str, Any]:
    """
    Executes a scheduled local paper-trading cycle against the default Robinhood 'MCP' account (512500422)
    and logs telemetry to data/paper_live_ledger.jsonl and reports/latest_paper_status.md.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    db_url = "sqlite:///./data/options_agent_paper.db"
    sim_engine = _build_engine(db_url)
    init_db(sim_engine)

    def make_session() -> LightweightSession:
        return LightweightSession(database_url=sim_engine.database_url, store=sim_engine.store)

    uw_client = UnusualWhalesClient()
    tv_client = TradingViewClient()
    alpaca_client = AlpacaPaperClient(
        account_number=settings.alpaca_paper_account_number,
        total_value=10000.0,
        cash=10000.0,
        buying_power=10000.0,
    )
    # Cloud-native live data refresh (works 100% on Render without local PC/MCPs)
    walked_orders: list[dict[str, Any]] = []
    if alpaca_client._has_live_credentials():
        tv_client.refresh_live_market_data(tickers=DEFAULT_WATCHLIST)
        vix_check = tv_client.analyze_vix_regime()
        is_panic = vix_check.regime == VixRegime.REGIME_4_PANIC_BACKWARDATION
        uw_client.refresh_live_flow(tickers=DEFAULT_WATCHLIST)
        alpaca_client.scan_live_option_chains(
            tickers=DEFAULT_WATCHLIST,
            panic_regime=is_panic,
        )
        walked_orders = alpaca_client.walk_open_live_orders(step_decrement=0.02)
    exec_broker_client: AlpacaPaperClient | RobinhoodClient = (
        alpaca_client
        if not settings.robinhood_trading_enabled
        else RobinhoodClient(account_number=DEFAULT_MCP_ACCOUNT_NUMBER)
    )
    screener = ScreenerJob(uw_client=uw_client, tv_client=tv_client, rh_client=exec_broker_client)  # type: ignore[arg-type]
    exec_worker = ExecutionWorker(
        rh_client=exec_broker_client,
        tv_client=tv_client,
        execution_mode=ExecutionMode.PAPER,
    )
    monitor = PositionMonitorWorker(
        jsonl_path=Path("data/stress_events.jsonl"),
    )

    today_str = datetime.now(timezone.utc).date().isoformat()
    with get_db_session(make_session) as session:
        screen_res = screener.run_pipeline(session=session, trade_date=today_str)
        intents = session.query(ExecutionIntent).filter(ExecutionIntent.status == "QUEUED").all()
        filled_trades: list[dict[str, Any]] = []
        for intent in intents:
            tr = exec_worker.execute_queued_intent(
                session=session,
                intent=intent,
                simulated_fill_step=1,
                step_decrement=0.01,
            )
            if tr is not None:
                filled_trades.append(
                    {
                        "ticker": tr.ticker,
                        "strategy": tr.strategy,
                        "expiration_date": tr.expiration_date,
                        "strikes": f"${tr.short_strike:.0f}/${tr.long_strike:.0f}",
                        "contracts": tr.contracts,
                        "entry_credit": tr.entry_credit,
                        "target_close_50pct": tr.target_close_price,
                        "gtc_order_id": tr.gtc_order_id,
                    }
                )

        # Evaluate open paper trades through the Loose-First + VIX + ML Monitor
        open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
        monitor_checks: list[dict[str, Any]] = []
        vix_snap = tv_client.get_vix_snapshot()
        for tr in open_trades:
            tech = tv_client.get_technical_structure(tr.ticker)
            flow = uw_client.get_flow_snapshot(tr.ticker)
            pos_input = PositionStressInput(
                ticker=tr.ticker,
                strategy=tr.strategy,
                contracts=tr.contracts,
                dte=tr.dte_at_entry,
                short_delta=tr.short_delta,
                spread_mark=tr.current_mark,
                entry_credit=tr.entry_credit,
                tv_1h_close=tech.last_1h_close,
                tv_sr_level=tr.tv_sr_anchor,
                atr_14=tr.atr_14,
                tv_4h_rsi=tech.rsi_4h,
                uw_darkpool_net_flow_1h=flow.darkpool_net_flow_1h,
                uw_darkpool_zscore=flow.darkpool_flow_zscore,
                uw_sweep_ratio=flow.ask_sweep_ratio,
                vix_level=vix_snap.vix_price,
                vix3m_level=vix_snap.vix3m_price,
                vix_zscore_20d=vix_snap.vix_zscore_20d,
                vix_4h_rsi=vix_snap.vix_4h_rsi,
                vix_1h_slope=vix_snap.vix_1h_slope,
            )
            decision = monitor.poll_and_log(session=session, pos=pos_input, trade=tr)
            defensive_exit = None
            if decision.action in ("HARD_STOP_EXIT", "EARLY_CUT_ML_OVERRIDE", "21_DTE_ROLL"):
                exit_reason = "21_DTE_ROLL" if decision.action == "21_DTE_ROLL" else "HARD_STOP_EXIT"
                defensive_exit = exec_worker.execute_defensive_exit_or_roll(
                    session=session,
                    trade=tr,
                    exit_reason=exit_reason,
                    current_mid_debit=tr.current_mark,
                    chase_steps_required=1,
                    chase_step_increment=0.02,
                )
            monitor_checks.append(
                {
                    "ticker": tr.ticker,
                    "strategy": tr.strategy,
                    "short_delta": tr.short_delta,
                    "spread_mark": tr.current_mark,
                    "action": decision.action,
                    "polling_interval_seconds": decision.polling_interval_seconds,
                    "rationale": decision.rationale,
                    "defensive_exit": defensive_exit,
                }
            )

        ml_result = None
        if mode == "friday_ml_retrain":
            trainer = WeeklyModelTrainer()
            ml_result = trainer.run_weekly_calibration(session=session)

        journal = TradeJournalLogger()
        journal.record_trades_from_session(session=session)

    cycle_payload = {
        "timestamp_utc": now_iso,
        "cycle_mode": mode,
        "execution_mode": "ALPACA_PAPER",
        "account": exec_broker_client.get_portfolio_summary(),
        "vix_regime": screen_res["vix_regime"],
        "vix_summary": screen_res["vix_summary"],
        "scanned_candidates_count": len(exec_broker_client.spread_candidates),
        "evaluated_candidates_summary": {
            "queued": screen_res.get("queued_intents", []),
            "rejected": screen_res.get("rejected_candidates", []),
        },
        "walked_orders": walked_orders,
        "paper_trades_active": filled_trades,
        "monitor_decisions": monitor_checks,
        "ml_calibration": ml_result,
    }

    ledger_path = Path("data/paper_live_ledger.jsonl")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(cycle_payload) + "\n")

    kpi_tracker = StrategyKPITracker(ledger_path=ledger_path)
    kpi_summary = kpi_tracker.compute_and_save_kpis()
    journal_report = TradeJournalLogger().generate_report(kpi_snapshot=kpi_summary)
    cycle_payload["kpi_summary"] = kpi_summary["core_kpis"]
    cycle_payload["trade_journal_summary"] = journal_report

    if mode in ("screener", "friday_ml_retrain"):
        Notifier().send_alert(
            title=f"Options Agent ({mode.upper()}) — {screen_res['vix_regime']}",
            message=(
                f"Qualifying Trades: {screen_res['queued_intents_count']} | "
                f"Portfolio Rating: {journal_report['overall_rating_grade']} ({journal_report['overall_performance_score']}/100)"
            ),
            metadata={
                "broker": "alpaca_paper (PA35XZXE8ZZE)",
                "vix_regime": screen_res["vix_regime"],
                "active_spreads": len(filled_trades),
                "rating": journal_report["overall_rating_grade"],
            },
        )

    status_md = Path("reports/latest_paper_status.md")
    status_md.parent.mkdir(parents=True, exist_ok=True)
    status_md.write_text(
        "\n".join(
            [
                f"# Latest Alpaca Account Status (`{now_iso}`)",
                "",
                f"- **Cycle Mode:** `{mode}` (`ACTIVE_EXECUTION_BROKER=alpaca`)",
                f"- **Alpaca Account:** `{exec_broker_client.account_number}` (`$10,000.00` base equity)",
                f"- **Active VIX Regime:** `{screen_res['vix_regime']}` — {screen_res['vix_summary']}",
                f"- **Post-Allocation Buying Power:** `${exec_broker_client.buying_power:,.2f}`",
                "",
                "## 1. Potential Trades Summary (Screener Scan)",
                "",
                "### Qualifying Candidates",
                "",
                "| Ticker | Strategy | Exp (DTE) | Strikes | Contracts | Mid Credit | 50% GTC Target | Collateral |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                *[
                    f"| **{q['ticker']}** | `{q['strategy']}` | `{q['expiration_date']}` ({q['dte']}d) | "
                    f"`${q['short_strike']:.0f}/${q['long_strike']:.0f}` | {q['contracts']}x | "
                    f"`${q['mid_credit']:.2f}` | `+${q['expected_50pct_profit_usd']:.2f}` | `${q['collateral_locked_usd']:.2f}` |"
                    for q in screen_res.get("queued_intents", [])
                ],
                "",
                "### Rejected / Filtered Candidates",
                "",
                "| Candidate | Rejection Reason |",
                "| :--- | :--- |",
                *[
                    f"| `{r['key']}` | {r['reason']} |"
                    for r in screen_res.get("rejected_candidates", [])
                ],
                "",
                "## 2. Active Paper Positions & Monitor Status",
                "",
                "| Ticker | Strategy | Strikes | Contracts | Entry Credit | 50% GTC Target | Monitor Action | Polling Interval |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                *[
                    f"| **{t['ticker']}** | `{t['strategy']}` | `{t['strikes']}` | {t['contracts']}x | "
                    f"`${t['entry_credit']:.2f}` | `${t['target_close_50pct']:.2f}` | "
                    f"`{m['action']}` | `{m['polling_interval_seconds']}s` |"
                    for t, m in zip(filled_trades, monitor_checks)
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    potential_md = Path("reports/potential_trades_summary.md")
    potential_md.write_text(
        "\n".join(
            [
                f"# Potential Trades Scan Summary (`{now_iso}`)",
                "",
                f"- **VIX Regime:** `{screen_res['vix_regime']}` (`VIX/VIX3M={screen_res['term_structure_ratio']}`)",
                f"- **Qualifying Trades:** `{screen_res['queued_intents_count']}` | **Filtered Out:** `{len(screen_res.get('rejected_candidates', []))}`",
                "",
                "## Qualifying Potential Trades",
                "",
                "| Ticker | Strategy | Exp (DTE) | Strikes | Contracts | Mid Credit | 50% Target Profit | Collateral |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                *[
                    f"| **{q['ticker']}** | `{q['strategy']}` | `{q['expiration_date']}` ({q['dte']}d) | "
                    f"`${q['short_strike']:.0f}/${q['long_strike']:.0f}` | {q['contracts']}x | "
                    f"`${q['mid_credit']:.2f}` | `+${q['expected_50pct_profit_usd']:.2f}` | `${q['collateral_locked_usd']:.2f}` |"
                    for q in screen_res.get("queued_intents", [])
                ],
                "",
                "## Filtered / Rejected Candidates",
                "",
                "| Candidate | Reason |",
                "| :--- | :--- |",
                *[
                    f"| `{r['key']}` | {r['reason']} |"
                    for r in screen_res.get("rejected_candidates", [])
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cycle_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled Paper-Trading Check Runner")
    parser.add_argument(
        "--mode",
        choices=["screener", "monitor", "monitor_and_screen", "friday_ml_retrain"],
        default="monitor_and_screen",
    )
    args = parser.parse_args()
    res = run_scheduled_paper_cycle(mode=args.mode)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
