from __future__ import annotations

import json

from src.db.session import get_db_session, init_db
from src.ml_engine.model_trainer import WeeklyModelTrainer


def main() -> dict:
    init_db()
    trainer = WeeklyModelTrainer()
    with get_db_session() as session:
        result = trainer.run_weekly_calibration(session=session)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
