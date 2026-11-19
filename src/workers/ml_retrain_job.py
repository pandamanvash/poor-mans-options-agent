from __future__ import annotations

import json

from src.config.settings import settings
from src.db.session import get_db_session, init_db
from src.mcp_clients.alpaca import AlpacaPaperClient
from src.ml_engine.kpi_tracker import StrategyKPITracker
from src.ml_engine.model_trainer import WeeklyModelTrainer
from src.ml_engine.trade_journal import TradeJournalLogger
from src.notifications.notifier import Notifier


def main() -> dict:
    init_db()
    alpaca_client = AlpacaPaperClient(account_number=settings.alpaca_paper_account_number)
    trainer = WeeklyModelTrainer()
    journal = TradeJournalLogger()

    with get_db_session() as session:
        ml_result = trainer.run_weekly_calibration(session=session)
        journal.record_trades_from_session(session=session)

    # Pull live filled/open option orders & positions directly from Alpaca Paper (PA35XZXE8ZZE)
    journal.sync_from_alpaca(alpaca_client)

    kpi_snapshot = StrategyKPITracker().compute_and_save_kpis()
    journal_summary = journal.generate_report(kpi_snapshot=kpi_snapshot)

    Notifier().send_alert(
        title="Friday ML Calibration, Alpaca Trade Journal & KPI Scorecard",
        message=(
            f"Alpaca Account {alpaca_client.account_number} Equity: ${alpaca_client.total_value:,.2f} | "
            f"Promoted Policy: {ml_result['best_policy']} | "
            f"Portfolio Grade: {journal_summary['overall_rating_grade']} ({journal_summary['overall_performance_score']}/100)"
        ),
        metadata={
            "alpaca_account": alpaca_client.account_number,
            "equity_usd": f"${alpaca_client.total_value:,.2f}",
            "promoted_policy": ml_result["best_policy"],
            "ev_per_2x_spread": f"${ml_result['best_ev_per_2contract_spread_usd']:+.2f}",
            "overall_grade": journal_summary["overall_rating_grade"],
            "logged_trades": journal_summary["logged_trades_count"],
        },
    )

    combined = {
        "alpaca_account": alpaca_client.get_portfolio_summary(),
        "ml_calibration": ml_result,
        "kpi_summary": kpi_snapshot["core_kpis"],
        "trade_journal_summary": journal_summary,
    }
    print(json.dumps(combined, indent=2))
    return combined


if __name__ == "__main__":
    main()
