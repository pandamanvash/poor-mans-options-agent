from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import ExecutionMode
from src.db.models import ExecutionIntent, Trade
from src.db.session import LightweightSession, _build_engine, get_db_session, init_db
from src.mcp_clients.robinhood import RobinhoodClient
from src.mcp_clients.tradingview import TradingViewClient
from src.mcp_clients.unusual_whales import UnusualWhalesClient
from src.ml_engine.model_trainer import WeeklyModelTrainer
from src.workers.execution_worker import ExecutionWorker
from src.workers.monitor_worker import PositionMonitorWorker, PositionStressInput
from src.workers.screener_job import ScreenerJob


def run_weekly_paper_trading_simulation(
    db_url: str = "sqlite:///./data/paper_sim_2026_11_16.db",
) -> dict[str, Any]:
    """
    Runs a full 5-day simulated paper trading test for the week of 2026-11-16 to 2026-11-20
    using live snapshots from Robinhood (Account ••••0422), TradingView (CBOE:VIX=22.91, VIX3M=23.02),
    and Unusual Whales MCPs.
    """
    db_file = db_url.replace("sqlite:///", "")
    if db_file and db_file != ":memory:":
        path_obj = Path(db_file)
        if path_obj.exists():
            path_obj.unlink()

    sim_engine = _build_engine(db_url)
    init_db(sim_engine)

    def make_sim_session() -> LightweightSession:
        return LightweightSession(database_url=sim_engine.database_url, store=sim_engine.store)

    uw_client = UnusualWhalesClient()
    tv_client = TradingViewClient()
    rh_client = RobinhoodClient()
    screener = ScreenerJob(uw_client=uw_client, tv_client=tv_client, rh_client=rh_client)
    exec_worker = ExecutionWorker(
        rh_client=rh_client,
        tv_client=tv_client,
        execution_mode=ExecutionMode.PAPER,
    )
    monitor = PositionMonitorWorker(
        jsonl_path=Path("data/paper_sim_stress_events.jsonl"),
    )

    with get_db_session(make_sim_session) as session:
        # 1. Monday 9:45 AM ET: Run Multi-MCP Screener & VIX Regime Pipeline
        screen_summary = screener.run_pipeline(session=session, trade_date="2026-11-16")

        # 2. Monday 9:50 AM ET: Execute Queued Intents in PAPER Mode via Smart Mid-Price Limit Walker
        intents = session.query(ExecutionIntent).filter(ExecutionIntent.status == "QUEUED").all()
        opened_trades: list[dict[str, Any]] = []
        for idx, intent in enumerate(intents):
            step_to_fill = 1 if idx % 2 == 0 else 0
            trade = exec_worker.execute_queued_intent(
                session=session,
                intent=intent,
                simulated_fill_step=step_to_fill,
                step_decrement=0.01,
            )
            if trade is not None:
                opened_trades.append(
                    {
                        "ticker": trade.ticker,
                        "strategy": trade.strategy,
                        "expiration_date": trade.expiration_date,
                        "strikes": f"${trade.short_strike:.0f}/${trade.long_strike:.0f}",
                        "contracts": trade.contracts,
                        "entry_credit": trade.entry_credit,
                        "target_close_price_50pct": trade.target_close_price,
                        "gtc_order_id": trade.gtc_order_id,
                        "collateral_usd": trade.collateral_usd,
                    }
                )

        # 3. Audit Live Existing Robinhood Position: TSM 2026-11-27 $370/$365 Bull Put (3x, 11 DTE, -0.4305 Delta)
        tsm_gtc = rh_client.submit_limit_order(
            idempotency_key="TSM_LIVE_EXISTING_GTC",
            ticker="TSM",
            strategy="BULL_PUT_SPREAD",
            order_type="GTC_TAKE_PROFIT",
            limit_price=0.88,
            contracts=3,
        )
        tsm_trade = Trade(
            idempotency_key="TSM_BULL_PUT_SPREAD_2026-11-27_370-365_2026-11-16",
            ticker="TSM",
            strategy="BULL_PUT_SPREAD",
            expiration_date="2026-11-27",
            dte_at_entry=11,
            short_strike=370.0,
            long_strike=365.0,
            short_option_id="e41ad2a6-c287-4e86-919f-d2504ede9eb1",
            long_option_id="f370ac42-8285-4a56-9d05-9f9fcce8f9c6",
            contracts=3,
            entry_credit=1.76,
            target_close_price=0.88,
            current_mark=2.05,
            short_delta=0.4305,
            tv_sr_anchor=375.50,
            atr_14=11.40,
            collateral_usd=972.0,
            unrealized_pnl_usd=-87.0,
            gtc_order_id=tsm_gtc.order_id,
            execution_mode="PAPER",
            status="OPEN",
        )
        session.add(tsm_trade)
        session.flush()

        tsm_stress_input = PositionStressInput(
            ticker="TSM",
            strategy="BULL_PUT_SPREAD",
            contracts=3,
            dte=11,
            short_delta=-0.4305,
            spread_mark=2.05,
            entry_credit=1.76,
            tv_1h_close=373.45,
            tv_sr_level=375.50,
            atr_14=11.40,
            tv_4h_rsi=37.36,
            uw_darkpool_net_flow_1h=-499_523.50,
            uw_darkpool_zscore=2.12,
            uw_sweep_ratio=0.74,
            vix_level=22.91,
            vix3m_level=23.02,
            vix_zscore_20d=1.48,
            vix_4h_rsi=61.65,
            vix_1h_slope=0.18,
        )
        tsm_decision = monitor.poll_and_log(session=session, pos=tsm_stress_input, trade=tsm_trade)
        tsm_exit_summary = exec_worker.execute_defensive_exit_or_roll(
            session=session,
            trade=tsm_trade,
            exit_reason="21_DTE_ROLL" if tsm_decision.action == "21_DTE_ROLL" else "HARD_STOP_EXIT",
            current_mid_debit=2.05,
            chase_steps_required=1,
            chase_step_increment=0.02,
        )

        # 4. Wednesday/Thursday Intra-Week Stress Simulation on AMZN & AAPL
        amzn_trade = session.query(Trade).filter(Trade.ticker == "AMZN").first()
        aapl_trade = session.query(Trade).filter(Trade.ticker == "AAPL").first()

        # Thursday 11:00 AM ET: AMZN experiences a Phantom Vega spike (VIX 4H RSI = 72.4, AMZN holds $253.80 > $252.27 POC)
        amzn_stress_input = PositionStressInput(
            ticker="AMZN",
            strategy="BULL_PUT_SPREAD",
            contracts=amzn_trade.contracts if amzn_trade else 2,
            dte=29,
            short_delta=-0.41,
            spread_mark=2.25,
            entry_credit=amzn_trade.entry_credit if amzn_trade else 1.57,
            tv_1h_close=253.80,
            tv_sr_level=252.27,
            atr_14=9.15,
            tv_4h_rsi=43.50,
            uw_darkpool_net_flow_1h=385_000.0,
            uw_darkpool_zscore=0.85,
            uw_sweep_ratio=0.54,
            vix_level=24.60,
            vix3m_level=25.10,
            vix_zscore_20d=1.82,
            vix_4h_rsi=72.40,
            vix_1h_slope=-0.15,
        )
        amzn_thursday_decision = monitor.poll_and_log(
            session=session, pos=amzn_stress_input, trade=amzn_trade
        )

        # Friday 1:30 PM ET: Vega crushes as VIX recedes to 19.8; AMZN, AAPL, and TSLA hit their 50% GTC Take-Profit orders!
        tsla_trade = session.query(Trade).filter(Trade.ticker == "TSLA").first()
        realized_profits: list[dict[str, Any]] = []
        for tr in [amzn_trade, aapl_trade, tsla_trade]:
            if tr is not None:
                if tr.gtc_order_id and tr.gtc_order_id in rh_client.orders:
                    rh_client.mark_order_filled(tr.gtc_order_id, fill_price=tr.target_close_price)
                tr.current_mark = tr.target_close_price
                tr.realized_pnl_usd = round(
                    (tr.entry_credit - tr.target_close_price) * 100.0 * tr.contracts, 2
                )
                tr.unrealized_pnl_usd = 0.0
                tr.status = "CLOSED_PROFIT"
                realized_profits.append(
                    {
                        "ticker": tr.ticker,
                        "strategy": tr.strategy,
                        "contracts": tr.contracts,
                        "entry_credit": tr.entry_credit,
                        "gtc_fill_debit": tr.target_close_price,
                        "realized_pnl_usd": tr.realized_pnl_usd,
                    }
                )
        session.flush()

        # 5. Friday 4:30 PM ET: Run Weekly ML Counterfactual Backtest & Dynamic Threshold Promotion
        trainer = WeeklyModelTrainer()
        ml_summary = trainer.run_weekly_calibration(session=session, audit_date="2026_11_20")

        new_entries_pnl = round(sum(item["realized_pnl_usd"] for item in realized_profits), 2)
        net_weekly_paper_pnl = round(
            new_entries_pnl + float(tsm_exit_summary["realized_pnl_usd"]),
            2,
        )

        summary = {
            "simulation_week": "2026-11-16 to 2026-11-20",
            "execution_mode": "PAPER",
            "account_snapshot": rh_client.get_portfolio_summary(),
            "step_1_screener_and_vix_regime": screen_summary,
            "step_2_paper_entries_and_gtc_staging": opened_trades,
            "step_3_live_tsm_position_audit": {
                "monitor_decision": tsm_decision.action,
                "polling_interval_seconds": tsm_decision.polling_interval_seconds,
                "rationale": tsm_decision.rationale,
                "defensive_execution": tsm_exit_summary,
            },
            "step_4_thursday_phantom_vega_test": {
                "ticker": "AMZN",
                "short_delta": 0.41,
                "vix_4h_rsi": 72.40,
                "decision": amzn_thursday_decision.action,
                "rationale": amzn_thursday_decision.rationale,
                "friday_50pct_gtc_fills": realized_profits,
            },
            "step_5_friday_ml_calibration": ml_summary,
            "new_screened_entries_realized_pnl_usd": new_entries_pnl,
            "net_weekly_paper_pnl_including_tsm_cleanup_usd": net_weekly_paper_pnl,
        }

    report_out = Path("reports/paper_trading_week_2026_11_16.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    output = run_weekly_paper_trading_simulation()
    print(json.dumps(output, indent=2))
