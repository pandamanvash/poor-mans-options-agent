from __future__ import annotations

import json

from src.db.session import get_db_session, init_db
from src.ml_engine.kpi_tracker import StrategyKPITracker
from src.ml_engine.model_trainer import WeeklyModelTrainer
from src.ml_engine.trade_journal import TradeJournalLogger


def main() -> dict:
    init_db()
    trainer = WeeklyModelTrainer()
    with get_db_session() as session:
        ml_result = trainer.run_weekly_calibration(session=session)
        TradeJournalLogger().record_trades_from_session(session=session)

    kpi_snapshot = StrategyKPITracker().compute_and_save_kpis()
    journal_summary = TradeJournalLogger().generate_report(kpi_snapshot=kpi_snapshot)

    combined = {
        "ml_calibration": ml_result,
        "kpi_summary": kpi_snapshot["core_kpis"],
        "trade_journal_summary": journal_summary,
    }
    print(json.dumps(combined, indent=2))
    return combined


if __name__ == "__main__":
    main()
