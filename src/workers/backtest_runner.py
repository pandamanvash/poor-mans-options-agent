from __future__ import annotations

import json

from src.ml_engine.backtest_agent import BacktestingAgent


def main() -> dict:
    agent = BacktestingAgent()
    results = agent.run_30d_and_90d_suite()
    print(
        json.dumps(
            {
                "account_default": results["account_default"],
                "last_30_days": {
                    k: v for k, v in results["last_30_days"].items() if k != "trades"
                },
                "last_90_days": {
                    k: v for k, v in results["last_90_days"].items() if k != "trades"
                },
                "report_markdown_path": results["report_markdown_path"],
                "report_json_path": results["report_json_path"],
            },
            indent=2,
        )
    )
    return results


if __name__ == "__main__":
    main()
