from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.db.session import get_db_session, init_db
from src.mcp_clients.alpaca import AlpacaPaperClient
from src.ml_engine.gemini_fill_advisor import GeminiFillAdvisor
from src.ml_engine.kpi_tracker import StrategyKPITracker
from src.ml_engine.model_trainer import WeeklyModelTrainer
from src.ml_engine.trade_journal import TradeJournalLogger
from src.notifications.notifier import Notifier
from src.workers.scheduled_paper_check import _summarize_live_alpaca_spreads


def render_friday_backtest_markdown(
    run_date: str,
    alpaca_summary: dict[str, Any],
    live_book: dict[str, Any],
    ml_result: dict[str, Any],
    kpi_snapshot: dict[str, Any],
    journal_summary: dict[str, Any],
    ai_note: str,
) -> str:
    upl = float(live_book.get("total_unrealized_pnl_usd", 0.0))
    upl_str = f"+${upl:,.2f}" if upl >= 0 else f"-${abs(upl):,.2f}"
    spreads_md = (
        "\n".join(live_book.get("spread_labels", []))
        if live_book.get("spread_labels")
        else "- None currently open."
    )
    core = kpi_snapshot.get("core_kpis", {})
    lines = [
        f"# Friday ML Calibration & Backtest Review (`{run_date}`)",
        "",
        "## 1. Executive Summary",
        f"> {ai_note}",
        "",
        f"- **Alpaca Account:** `{alpaca_summary.get('account_number', 'PA35XZXE8ZZE')}`",
        f"- **Account Equity:** `${float(alpaca_summary.get('total_value', 0.0)):,.2f}` | **Options Buying Power:** `${float(alpaca_summary.get('buying_power', 0.0)):,.2f}`",
        f"- **Open Spreads:** `{live_book.get('active_spreads_count', 0)}` (`{upl_str}` unrealized P&L)",
        f"- **Promoted Policy:** `{ml_result.get('best_policy')}` (`${float(ml_result.get('best_ev_per_2contract_spread_usd', 0.0)):+.2f}` EV per 2x spread)",
        f"- **Portfolio Rating:** `{journal_summary.get('overall_rating_grade')}` (`{journal_summary.get('overall_performance_score')}/100`)",
        "",
        "## 2. Live Open Positions (Alpaca Book)",
        spreads_md,
        "",
        "## 3. Strategy KPI Snapshot (`v5` Stacked Benchmark)",
        "",
        "| Metric | Value |",
        "| :--- | :---: |",
        f"| 30-Day Net P&L | `${float(core.get('last_30d_net_pnl_usd', 0.0)):+,.2f}` (`{core.get('last_30d_win_rate_pct', 0.0)}%` WR) |",
        f"| 90-Day Net P&L | `${float(core.get('last_90d_net_pnl_usd', 0.0)):+,.2f}` (`{core.get('last_90d_win_rate_pct', 0.0)}%` WR) |",
        f"| 90-Day Profit Factor / Sharpe | `{core.get('last_90d_profit_factor', 0.0)}x` / `{core.get('last_90d_sharpe', 0.0)}` |",
        f"| Max Drawdown | `{core.get('max_drawdown_pct', 0.0)}%` (`${float(core.get('max_drawdown_usd', 0.0)):,.2f}`) |",
        f"| Logged Live Trades | `{journal_summary.get('logged_trades_count', 0)}` |",
        "",
        "## 4. Updated ML Risk Thresholds (`config/risk_thresholds.json`)",
        "```json",
        json.dumps(settings.load_risk_thresholds(), indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> dict[str, Any]:
    init_db()
    now_date = datetime.now(timezone.utc).date().isoformat()
    alpaca_client = AlpacaPaperClient(account_number=settings.alpaca_paper_account_number)
    trainer = WeeklyModelTrainer()
    journal = TradeJournalLogger()

    with get_db_session() as session:
        ml_result = trainer.run_weekly_calibration(session=session)
        journal.record_trades_from_session(session=session)

    # Pull live filled/open option orders & positions directly from Alpaca Paper (PA35XZXE8ZZE)
    journal.sync_from_alpaca(alpaca_client)
    live_positions = alpaca_client.fetch_live_positions() if alpaca_client._has_live_credentials() else []
    live_book = _summarize_live_alpaca_spreads(live_positions)

    kpi_snapshot = StrategyKPITracker().compute_and_save_kpis()
    journal_summary = journal.generate_report(kpi_snapshot=kpi_snapshot)

    advisor = GeminiFillAdvisor()
    ai_note = advisor.summarize_friday_backtest(
        equity_usd=alpaca_client.total_value,
        promoted_policy=str(ml_result["best_policy"]),
        best_ev_usd=float(ml_result["best_ev_per_2contract_spread_usd"]),
        grade=str(journal_summary["overall_rating_grade"]),
        open_spreads_count=int(live_book["active_spreads_count"]),
    )

    md_content = render_friday_backtest_markdown(
        run_date=now_date,
        alpaca_summary=alpaca_client.get_portfolio_summary(),
        live_book=live_book,
        ml_result=ml_result,
        kpi_snapshot=kpi_snapshot,
        journal_summary=journal_summary,
        ai_note=ai_note,
    )
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    dated_md = reports_dir / f"friday_ml_backtest_{now_date.replace('-', '_')}.md"
    latest_md = reports_dir / "friday_ml_backtest_latest.md"
    dated_md.write_text(md_content, encoding="utf-8")
    latest_md.write_text(md_content, encoding="utf-8")

    positions_text = (
        "\n".join(live_book["spread_labels"][:6])
        if live_book["spread_labels"]
        else "None"
    )
    Notifier().send_alert(
        title=f"Friday ML Backtest & Calibration ({now_date})",
        message=(
            f"Alpaca Equity: ${alpaca_client.total_value:,.2f} | "
            f"Promoted Policy: {ml_result['best_policy']} | "
            f"Grade: {journal_summary['overall_rating_grade']} ({journal_summary['overall_performance_score']}/100)\n"
            f"Report saved: `{dated_md.as_posix()}`"
        ),
        metadata={
            "alpaca_account": alpaca_client.account_number,
            "equity": f"${alpaca_client.total_value:,.2f}",
            "promoted_policy": ml_result["best_policy"],
            "ev_per_2x_spread": f"${ml_result['best_ev_per_2contract_spread_usd']:+.2f}",
            "overall_grade": journal_summary["overall_rating_grade"],
            "open_positions": positions_text,
        },
    )

    combined = {
        "report_path": dated_md.as_posix(),
        "latest_report_path": latest_md.as_posix(),
        "alpaca_account": alpaca_client.get_portfolio_summary(),
        "live_alpaca_book": live_book,
        "ml_calibration": ml_result,
        "ai_review_note": ai_note,
        "kpi_summary": kpi_snapshot["core_kpis"],
        "trade_journal_summary": journal_summary,
    }
    print(json.dumps(combined, indent=2))
    return combined


if __name__ == "__main__":
    main()
