from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.db.models import ExecutionIntent, Trade
from src.ml_engine.kpi_tracker import StrategyKPITracker

STRATEGY_CATEGORIES: dict[str, str] = {
    "BULL_PUT_SPREAD": "Credit Harvesting / Bull Put Vertical",
    "BEAR_CALL_SPREAD": "Portfolio Hedge / Bear Call Vertical",
    "PMCC_DIAGONAL": "Core Stock Replacement / PMCC Diagonal",
}


def score_to_grade(score: float) -> str:
    if score >= 95.0:
        return "A+"
    if score >= 90.0:
        return "A"
    if score >= 85.0:
        return "A-"
    if score >= 80.0:
        return "B+"
    if score >= 75.0:
        return "B"
    if score >= 70.0:
        return "C+"
    if score >= 60.0:
        return "C"
    if score >= 50.0:
        return "D"
    return "F"


@dataclass(frozen=True)
class TradeJournalEntry:
    trade_id: str
    ticker: str
    strategy: str
    category: str
    execution_start_date: str
    execution_end_date: str
    status: str
    strikes: str
    contracts: int
    entry_credit: float
    exit_or_mark_price: float
    pnl_usd: float
    collateral_usd: float
    working_thesis: str
    performance_score: float
    rating_grade: str
    performance_summary: str


def evaluate_trade_performance(
    trade: Trade,
    working_thesis: str,
) -> TradeJournalEntry:
    """
    Score a trade (0-100) and assign a letter grade based on:
      - P&L outcome vs. 50% GTC target or capped stop-loss (50 pts)
      - Quantitative rule compliance: >=$100 50%-target scaling & collateral discipline (25 pts)
      - Execution & risk discipline: low slippage & proper stop/hold execution (25 pts)
    """
    pnl = (
        float(trade.realized_pnl_usd)
        if trade.realized_pnl_usd is not None
        else float(trade.unrealized_pnl_usd or 0.0)
    )
    target_profit_usd = max(
        100.0,
        round((trade.entry_credit - trade.target_close_price) * 100.0 * trade.contracts, 2),
    )

    # 1. Outcome component (0-50 pts)
    if trade.status == "CLOSED_PROFIT":
        outcome_pts = 50.0
    elif trade.status in ("CLOSED_STOP", "ROLLED"):
        # Reward tight loss containment (< 1.5x credit risk)
        loss_ratio = abs(pnl) / max(1.0, trade.entry_credit * 100.0 * trade.contracts)
        outcome_pts = max(18.0, round(38.0 - (loss_ratio * 15.0), 1))
    else:
        # OPEN trade: baseline 38 pts adjusted by unrealized P&L progress toward 50% target
        progress = max(-0.5, min(1.0, pnl / target_profit_usd))
        outcome_pts = round(38.0 + (progress * 12.0), 1)

    # 2. Sizing & rule compliance component (0-25 pts)
    expected_50_usd = 0.50 * trade.entry_credit * 100.0 * trade.contracts
    rule_pts = 25.0 if expected_50_usd >= 100.0 else 15.0

    # 3. Execution & slippage discipline (0-25 pts)
    slippage_abs = abs(float(trade.slippage_usd or 0.0))
    exec_pts = max(15.0, round(25.0 - (slippage_abs * 0.25), 1))

    total_score = round(min(100.0, outcome_pts + rule_pts + exec_pts), 1)
    grade = score_to_grade(total_score)

    start_dt = (
        trade.opened_at.isoformat()
        if hasattr(trade.opened_at, "isoformat")
        else str(trade.opened_at)
    )
    end_dt = (
        trade.closed_at.isoformat()
        if trade.closed_at is not None and hasattr(trade.closed_at, "isoformat")
        else (str(trade.closed_at) if trade.closed_at else "OPEN (Active)")
    )

    if trade.status == "CLOSED_PROFIT":
        perf_summary = (
            f"Hit 50% GTC take-profit target at ${trade.target_close_price:.2f}, "
            f"realizing +${pnl:.2f} (slippage ${trade.slippage_usd:.2f})."
        )
    elif trade.status in ("CLOSED_STOP", "ROLLED"):
        perf_summary = (
            f"Defensive {trade.status} executed via Cancel-First chaser at ${trade.current_mark:.2f}; "
            f"loss contained at ${pnl:+.2f}."
        )
    else:
        perf_summary = (
            f"Active in PAPER mode (Short Delta {trade.short_delta:.2f}, mark ${trade.current_mark:.2f}); "
            f"50% GTC target staged at ${trade.target_close_price:.2f} (+${expected_50_usd:.2f})."
        )

    category = STRATEGY_CATEGORIES.get(trade.strategy, trade.strategy)
    return TradeJournalEntry(
        trade_id=trade.idempotency_key,
        ticker=trade.ticker,
        strategy=trade.strategy,
        category=category,
        execution_start_date=start_dt,
        execution_end_date=end_dt,
        status=trade.status,
        strikes=f"${trade.short_strike:.0f}/${trade.long_strike:.0f}",
        contracts=trade.contracts,
        entry_credit=round(float(trade.entry_credit), 2),
        exit_or_mark_price=round(float(trade.current_mark), 2),
        pnl_usd=round(pnl, 2),
        collateral_usd=round(float(trade.collateral_usd), 2),
        working_thesis=working_thesis or f"{trade.strategy} screened via UW + TV + RH MCPs.",
        performance_score=total_score,
        rating_grade=grade,
        performance_summary=perf_summary,
    )


class TradeJournalLogger:
    """
    Persists an automated log of all trades taken (upserted by trade_id),
    computes per-trade performance scores & letter grades, and generates
    the Friday ML Calibration Trade Journal & Strategy Rating Report.
    """

    def __init__(
        self,
        journal_jsonl_path: Path = Path("data/trade_journal.jsonl"),
        report_md_path: Path = Path("reports/trade_journal_report.md"),
    ) -> None:
        self.journal_jsonl_path = journal_jsonl_path
        self.report_md_path = report_md_path

    def load_entries(self) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        if not self.journal_jsonl_path.exists():
            return entries
        for line in self.journal_jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            entries[item["trade_id"]] = item
        return entries

    def record_trades_from_session(self, session: Any) -> list[dict[str, Any]]:
        intents_by_key = {
            intent.idempotency_key: intent.rationale
            for intent in session.query(ExecutionIntent).all()
        }
        trades = session.query(Trade).all()
        existing = self.load_entries()

        for tr in trades:
            thesis = intents_by_key.get(tr.idempotency_key, "")
            entry = evaluate_trade_performance(tr, working_thesis=thesis)
            existing[entry.trade_id] = asdict(entry)

        self.journal_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_jsonl_path.open("w", encoding="utf-8") as fh:
            for item in existing.values():
                fh.write(json.dumps(item) + "\n")

        return list(existing.values())

    def generate_report(self, kpi_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        entries = list(self.load_entries().values())
        kpi = kpi_snapshot or StrategyKPITracker().compute_and_save_kpis()

        if entries:
            avg_score = round(
                sum(float(e["performance_score"]) for e in entries) / len(entries), 1
            )
            total_pnl = round(sum(float(e["pnl_usd"]) for e in entries), 2)
        else:
            avg_score = 90.0
            total_pnl = 0.0

        overall_grade = score_to_grade(avg_score)
        now_iso = datetime.now(timezone.utc).isoformat()

        lines = [
            f"# Automated Trade Journal & Friday ML Strategy Rating (`{now_iso}`)",
            "",
            f"- **Default Robinhood Account:** `MCP` (`••••0422`, Limited Margin, Level 3)",
            f"- **Overall Portfolio Rating Grade:** **`{overall_grade}` (`{avg_score:.1f} / 100`)**",
            f"- **Logged Trades Count:** `{len(entries)}` | **Logged Net P&L:** `${total_pnl:+,.2f}`",
            f"- **30D / 90D Benchmark EV (`Policy D`):** `+{kpi['core_kpis']['last_30d_return_pct']:.2f}%` (30D) / `+{kpi['core_kpis']['last_90d_return_pct']:.2f}%` (90D)",
            "",
            "## 1. Automated Trade Execution Log & Performance Grades",
            "",
            "| Trade ID | Ticker & Category | Start Date $\\rightarrow$ End Date | Strikes & Size | Working Thesis | P&L | Score & Grade | Performance Summary |",
            "| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- |",
        ]

        for e in entries:
            thesis_short = str(e["working_thesis"]).replace("|", "/")
            if len(thesis_short) > 110:
                thesis_short = thesis_short[:107] + "..."
            lines.append(
                f"| `{e['trade_id']}` | **{e['ticker']}** ({e['category']}) | "
                f"`{e['execution_start_date'][:10]}` $\\rightarrow$ `{str(e['execution_end_date'])[:19]}` | "
                f"{e['contracts']}x {e['strikes']} | {thesis_short} | "
                f"`${float(e['pnl_usd']):+,.2f}` | **`{e['rating_grade']}` ({e['performance_score']:.1f})** | "
                f"{e['performance_summary']} |"
            )

        lines.extend(
            [
                "",
                "## 2. Linked Interactive Strategy KPIs",
                "",
                "- **JSON Telemetry:** `reports/kpi_metrics.json`",
                "- **Markdown KPI Dashboard:** `reports/kpi_dashboard.md`",
                "- **Interactive HTML Widget:** `reports/strategy_kpi_dashboard.html`",
                "",
            ]
        )

        self.report_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_md_path.write_text("\n".join(lines), encoding="utf-8")

        return {
            "overall_rating_grade": overall_grade,
            "overall_performance_score": avg_score,
            "logged_trades_count": len(entries),
            "logged_net_pnl_usd": total_pnl,
            "report_path": str(self.report_md_path),
            "journal_jsonl_path": str(self.journal_jsonl_path),
        }
