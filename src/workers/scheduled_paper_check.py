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
from src.notifications.notifier import Notifier, format_management_strategy, format_spread_type
from src.workers.execution_worker import ExecutionWorker
from src.workers.monitor_worker import PositionMonitorWorker, PositionStressInput
from src.workers.screener_job import ScreenerJob


def _summarize_live_alpaca_spreads(live_positions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Group raw Alpaca option leg positions (from GET /v2/positions) into vertical spreads
    by (underlying, expiration_yymmdd, put_or_call) so Discord alerts and logs reflect
    the real open book on Alpaca rather than ephemeral container-local SQLite rows.
    """
    import re

    occ_re = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    total_upl = 0.0
    for pos in live_positions:
        sym = str(pos.get("symbol") or "").upper()
        upl = float(pos.get("unrealized_pl") or 0.0)
        total_upl += upl
        m = occ_re.match(sym)
        if m:
            underlying, yymmdd, cp, strike_raw = m.groups()
            key = (underlying, yymmdd, cp)
            groups.setdefault(key, []).append(
                {
                    "symbol": sym,
                    "strike": int(strike_raw) / 1000.0,
                    "qty": int(float(pos.get("qty") or 0)),
                    "unrealized_pl": upl,
                }
            )

    spread_labels: list[str] = []
    for (underlying, yymmdd, cp), legs in sorted(groups.items()):
        mm_dd = f"{yymmdd[2:4]}/{yymmdd[4:6]}"
        net_upl = round(sum(l["unrealized_pl"] for l in legs), 2)
        short_legs = [l for l in legs if l["qty"] < 0]
        contracts = abs(short_legs[0]["qty"]) if short_legs else abs(legs[0]["qty"])
        # Order strikes short/long for readability
        ordered = sorted(legs, key=lambda x: x["strike"], reverse=(cp == "P"))
        strikes = "/".join(f"${l['strike']:g}" for l in ordered)
        spread_type = format_spread_type(cp)
        mgmt = format_management_strategy(cp, compact=True)
        sign = "+" if net_upl >= 0 else "-"
        spread_labels.append(
            f"• **{contracts}x {underlying}** `{mm_dd}` `{strikes}` — **{spread_type}** (`{sign}${abs(net_upl):,.2f}`)\n"
            f"  ↳ *{mgmt}*"
        )

    return {
        "active_spreads_count": len(groups),
        "open_legs_count": len(live_positions),
        "total_unrealized_pnl_usd": round(total_upl, 2),
        "spread_labels": spread_labels,
    }


def _dispatch_live_order_alerts(
    alpaca_client: AlpacaPaperClient,
    notifier: Notifier,
    state_path: Path = Path("data/notified_alpaca_orders.json"),
) -> list[dict[str, Any]]:
    """
    Detect newly submitted or newly filled multi-leg orders on Alpaca (including orders placed
    outside this container) and push Discord alerts with spread type and management strategy.
    """
    import re

    if not alpaca_client._has_live_credentials():
        return []

    occ_re = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
    recent_orders = alpaca_client.fetch_live_orders(status="all", limit=25)
    if not recent_orders:
        return []

    seen: dict[str, str] = {}
    cold_boot = not state_path.exists()
    if not cold_boot:
        try:
            seen = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            seen = {}

    dispatched: list[dict[str, Any]] = []
    for ord_obj in recent_orders:
        if str(ord_obj.get("order_class") or "").lower() != "mleg":
            continue
        oid = str(ord_obj.get("id") or "")
        status = str(ord_obj.get("status") or "").lower()
        if not oid or not status:
            continue

        prev_status = seen.get(oid)
        seen[oid] = status
        if cold_boot or prev_status == status:
            continue

        # Only alert on initial submission (not walk replacements) or fills
        is_new_order = status in ("new", "accepted") and not ord_obj.get("replaces")
        is_filled = status == "filled"
        if not (is_new_order or is_filled):
            continue

        legs = ord_obj.get("legs") or []
        parsed_legs: list[tuple[str, str, str, float, str]] = []
        for lg in legs:
            m = occ_re.match(str(lg.get("symbol") or "").upper())
            if m:
                u, ymd, cp, k_raw = m.groups()
                parsed_legs.append((u, f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}", cp, int(k_raw) / 1000.0, str(lg.get("position_intent") or "")))
        if not parsed_legs:
            continue

        ticker = parsed_legs[0][0]
        exp_date = parsed_legs[0][1]
        cp = parsed_legs[0][2]
        strategy = "BULL_PUT_SPREAD" if cp == "P" else "BEAR_CALL_SPREAD"
        ordered_strikes = sorted([p[3] for p in parsed_legs], reverse=(cp == "P"))
        strikes = "/".join(f"${k:g}" for k in ordered_strikes)
        qty = int(float(ord_obj.get("qty") or 1))
        is_closing = any("close" in p[4].lower() for p in parsed_legs)

        if is_filled:
            fill_px = float(ord_obj.get("filled_avg_price") or ord_obj.get("limit_price") or 0.0)
            notifier.notify_position_filled(
                ticker=ticker,
                strategy=strategy,
                contracts=qty,
                strikes=strikes,
                expiration_date=exp_date,
                fill_price=fill_px,
                is_closing=is_closing,
                order_id=oid,
            )
            dispatched.append({"event": "filled", "order_id": oid, "ticker": ticker, "strikes": strikes})
        elif is_new_order:
            lim_px = float(ord_obj.get("limit_price") or 0.0)
            notifier.notify_order_submitted(
                ticker=ticker,
                strategy=strategy,
                contracts=qty,
                strikes=strikes,
                expiration_date=exp_date,
                limit_price=lim_px,
                order_type="CLOSING_LIMIT" if is_closing else "ENTRY_LIMIT",
                order_id=oid,
            )
            dispatched.append({"event": "submitted", "order_id": oid, "ticker": ticker, "strikes": strikes})

    # Keep the state file compact (latest 100 order IDs)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = dict(list(seen.items())[-100:])
    state_path.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    return dispatched


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
        uw_client.seed_from_option_candidates(
            spread_candidates=alpaca_client.spread_candidates,
            vix_price=tv_client.get_vix_snapshot().vix_price,
            technical_structures=tv_client._structures,
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

    notifier = Notifier()
    live_order_alerts = _dispatch_live_order_alerts(alpaca_client, notifier)
    live_positions = alpaca_client.fetch_live_positions() if alpaca_client._has_live_credentials() else []
    live_book = _summarize_live_alpaca_spreads(live_positions)
    active_spreads_count = (
        live_book["active_spreads_count"]
        if alpaca_client._has_live_credentials()
        else len(filled_trades)
    )

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
        "live_alpaca_book": live_book,
        "live_order_alerts": live_order_alerts,
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
        upl_val = live_book["total_unrealized_pnl_usd"]
        upl_str = f"+${upl_val:,.2f}" if upl_val >= 0 else f"-${abs(upl_val):,.2f}"
        positions_text = (
            "\n".join(live_book["spread_labels"][:6])
            if live_book["spread_labels"]
            else "None"
        )
        notifier.send_alert(
            title=f"Options Agent ({mode.upper()}) — {screen_res['vix_regime']}",
            message=(
                f"Qualifying Trades: {screen_res['queued_intents_count']} | "
                f"Open Spreads: {active_spreads_count} ({upl_str} uPnL) | "
                f"Portfolio Rating: {journal_report['overall_rating_grade']} ({journal_report['overall_performance_score']}/100)"
            ),
            metadata={
                "broker": f"alpaca_paper ({exec_broker_client.account_number})",
                "vix_regime": screen_res["vix_regime"],
                "active_spreads": active_spreads_count,
                "equity": f"${exec_broker_client.total_value:,.2f}",
                "buying_power": f"${exec_broker_client.buying_power:,.2f}",
                "rating": journal_report["overall_rating_grade"],
                "open_positions": positions_text,
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
