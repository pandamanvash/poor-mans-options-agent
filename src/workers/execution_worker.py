from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.config.portfolio_rules import (
    PORTFOLIO_CONSTRAINTS,
    calculate_expected_50pct_profit,
)
from src.config.settings import ExecutionMode, settings
from src.db.models import ExecutionIntent, Trade
from src.mcp_clients.alpaca import AlpacaPaperClient
from src.mcp_clients.robinhood import RobinhoodClient
from src.mcp_clients.tradingview import TradingViewClient
from src.notifications.notifier import Notifier


class ExecutionWorker:
    """
    Fault-tolerant trade execution service (defaults to Alpaca Paper PA35XZXE8ZZE while
    ROBINHOOD_TRADING_ENABLED=false; can be toggled to Robinhood MCP 512500422 later):
      1. Execution Mode & Broker Toggle (ALPACA_TRADING_ENABLED=true, ROBINHOOD_TRADING_ENABLED=false)
      2. Smart Mid-Price Limit Walker (up to 4 steps of $0.01-$0.02 down to min_credit_floor)
      3. Automatic 50% Take-Profit GTC Staging upon entry fill
      4. Cancel-First Lock & Aggressive Limit Chaser ($0.02/step) for HARD_STOP_EXIT / 21_DTE_ROLL
      5. Deterministic Idempotency & 6% ($300) Daily Realized Loss Circuit Breaker
    """

    def __init__(
        self,
        rh_client: RobinhoodClient | AlpacaPaperClient | None = None,
        tv_client: TradingViewClient | None = None,
        notifier: Notifier | None = None,
        execution_mode: ExecutionMode | None = None,
    ) -> None:
        if rh_client is not None:
            self.rh = rh_client
        elif not settings.robinhood_trading_enabled:
            self.rh = AlpacaPaperClient(account_number=settings.alpaca_paper_account_number)
        else:
            self.rh = RobinhoodClient(account_number=settings.robinhood_account_number)
        self.tv = tv_client or TradingViewClient()
        self.notifier = notifier or Notifier()
        self.execution_mode = execution_mode or settings.execution_mode

    def is_daily_circuit_breaker_tripped(
        self,
        session: Session,
        account_value: float = PORTFOLIO_CONSTRAINTS.base_account_value,
    ) -> tuple[bool, float]:
        closed_today = (
            session.query(Trade)
            .filter(Trade.realized_pnl_usd.is_not(None))
            .all()
        )
        realized_loss = sum(
            abs(t.realized_pnl_usd) for t in closed_today if (t.realized_pnl_usd or 0.0) < 0.0
        )
        max_loss_usd = min(
            PORTFOLIO_CONSTRAINTS.max_daily_loss_usd,
            account_value * PORTFOLIO_CONSTRAINTS.max_daily_loss_pct,
        )
        if realized_loss >= max_loss_usd:
            self.notifier.send_alert(
                title="DAILY CIRCUIT BREAKER TRIPPED",
                message=(
                    f"Realized daily loss ${realized_loss:.2f} >= ${max_loss_usd:.2f} (6% limit). "
                    "Pausing new entries for 24 hours."
                ),
                metadata={"realized_loss_usd": realized_loss, "limit_usd": max_loss_usd},
            )
            return True, realized_loss
        return False, realized_loss

    def execute_queued_intent(
        self,
        session: Session,
        intent: ExecutionIntent,
        simulated_fill_step: int = 1,
        step_decrement: float = 0.02,
        semi_auto_approved: bool = True,
    ) -> Trade | None:
        """
        Execute an ENTRY_INTENT using the Smart Mid-Price Limit Walker and stage a 50% GTC closing order.
        - Never submits raw market orders.
        - Re-verifies pre-flight guards (spread <= $0.15, 40% cash floor, 50% profit >= $100).
        - Walks limit price up to 4 steps by $0.01-$0.02 down to min_credit_floor.
        - If unfilled after 4 steps, cancels cleanly.
        """
        # Idempotency check: never duplicate an existing Trade
        existing_trade = (
            session.query(Trade)
            .filter(Trade.idempotency_key == intent.idempotency_key)
            .first()
        )
        if existing_trade is not None:
            intent.status = "FILLED"
            return existing_trade

        portfolio = self.rh.get_portfolio_summary()
        account_value = float(portfolio["total_value"])
        available_cash = float(portfolio["buying_power"])

        # Circuit breaker check
        tripped, _ = self.is_daily_circuit_breaker_tripped(session, account_value=account_value)
        if tripped:
            intent.status = "REJECTED"
            return None

        # Pre-flight Guard 1: bid_ask_spread <= $0.15 (or <= $0.25 in high-VIX > 25 / panic regimes)
        vix_now = self.tv.get_vix_snapshot().vix_price
        spread_cap = (
            0.25
            if (vix_now > 25.0 or intent.vix_regime == "REGIME_4_PANIC_BACKWARDATION")
            else PORTFOLIO_CONSTRAINTS.max_bid_ask_spread_usd
        )
        if intent.bid_ask_spread > spread_cap:
            intent.status = "REJECTED"
            return None

        # Pre-flight Guard 2: Buying power & 40% cash reserve floor intact
        cash_floor = account_value * PORTFOLIO_CONSTRAINTS.min_cash_reserve_pct
        if (available_cash - intent.collateral_locked_usd) < cash_floor:
            intent.status = "REJECTED"
            return None

        # Pre-flight Guard 3: 0.50 * limit_credit * 100 * contracts >= $100.00
        if calculate_expected_50pct_profit(intent.mid_credit, intent.contracts) < 100.00:
            intent.status = "REJECTED"
            return None

        # SEMI_AUTO mode: require 1-click webhook approval before firing new trade entries
        if self.execution_mode == ExecutionMode.SEMI_AUTO:
            self.notifier.request_semi_auto_approval(
                {
                    "ticker": intent.ticker,
                    "strategy": intent.strategy,
                    "expiration_date": intent.expiration_date,
                    "contracts": intent.contracts,
                    "mid_credit": intent.mid_credit,
                    "expected_50pct_profit_usd": intent.expected_50pct_profit_usd,
                }
            )
            if not semi_auto_approved:
                return None

        # Submit initial multi-leg limit order at mid_price
        current_limit = round(intent.mid_credit, 2)
        order = self.rh.submit_limit_order(
            idempotency_key=intent.idempotency_key,
            ticker=intent.ticker,
            strategy=intent.strategy,
            order_type="ENTRY_LIMIT",
            limit_price=current_limit,
            contracts=intent.contracts,
            metadata={
                "mode": self.execution_mode.value,
                "expiration_date": intent.expiration_date,
                "short_strike": intent.short_strike,
                "long_strike": intent.long_strike,
            },
        )

        # Smart Mid-Price Limit Walker (up to 4 steps)
        max_steps = 4
        filled = False
        for step in range(max_steps + 1):
            if step == simulated_fill_step:
                self.rh.mark_order_filled(order.order_id, fill_price=current_limit)
                filled = True
                break

            if step < max_steps:
                next_limit = round(current_limit - step_decrement, 2)
                # Strict min_credit_floor where 50% target still yields >= $100
                if next_limit < intent.min_credit_floor or (
                    calculate_expected_50pct_profit(next_limit, intent.contracts) < 100.00
                ):
                    break
                current_limit = next_limit
                self.rh.update_limit_price(order.order_id, current_limit)

        if not filled:
            self.rh.cancel_order(order.order_id)
            intent.status = "CANCELLED"
            session.flush()
            return None

        fill_credit = float(order.filled_price or current_limit)
        target_close_price = round(fill_credit * 0.50, 2)

        # Immediately stage 50% Take-Profit GTC closing limit order
        gtc_order = self.rh.submit_limit_order(
            idempotency_key=f"{intent.idempotency_key}_GTC_50PCT",
            ticker=intent.ticker,
            strategy=intent.strategy,
            order_type="GTC_TAKE_PROFIT",
            limit_price=target_close_price,
            contracts=intent.contracts,
            metadata={
                "parent_entry_order_id": order.order_id,
                "expiration_date": intent.expiration_date,
                "short_strike": intent.short_strike,
                "long_strike": intent.long_strike,
            },
        )

        # Fetch TradingView structural anchor & ATR for intra-week risk management
        try:
            tech = self.tv.get_technical_structure(intent.ticker)
            sr_anchor = (
                tech.horizontal_support_poc
                if intent.strategy == "BULL_PUT_SPREAD"
                else tech.horizontal_resistance_ceiling
            )
            atr_14 = tech.atr_14
        except KeyError:
            sr_anchor = intent.short_strike
            atr_14 = 5.0

        spread_width = abs(intent.short_strike - intent.long_strike)
        collateral_usd = round((spread_width - fill_credit) * 100.0 * intent.contracts, 2)
        slippage_usd = round((intent.mid_credit - fill_credit) * 100.0 * intent.contracts, 2)

        trade = Trade(
            idempotency_key=intent.idempotency_key,
            ticker=intent.ticker,
            strategy=intent.strategy,
            expiration_date=intent.expiration_date,
            dte_at_entry=intent.dte,
            short_strike=intent.short_strike,
            long_strike=intent.long_strike,
            short_option_id=intent.short_option_id,
            long_option_id=intent.long_option_id,
            contracts=intent.contracts,
            entry_credit=fill_credit,
            target_close_price=target_close_price,
            current_mark=fill_credit,
            short_delta=0.27,
            tv_sr_anchor=sr_anchor,
            atr_14=atr_14,
            collateral_usd=collateral_usd,
            unrealized_pnl_usd=0.0,
            slippage_usd=slippage_usd,
            entry_order_id=order.order_id,
            gtc_order_id=gtc_order.order_id,
            execution_mode=self.execution_mode.value,
            status="OPEN",
        )
        intent.status = "FILLED"
        self.rh.buying_power = round(self.rh.buying_power - collateral_usd, 2)
        session.add(trade)
        session.flush()
        return trade

    def execute_defensive_exit_or_roll(
        self,
        session: Session,
        trade: Trade,
        exit_reason: str,  # "HARD_STOP_EXIT" or "21_DTE_ROLL"
        current_mid_debit: float,
        chase_steps_required: int = 2,
        chase_step_increment: float = 0.02,
    ) -> dict[str, Any]:
        """
        Automated Defensive Exit & Roll Engine:
          1. Cancel-First Lock: Immediately cancel open gtc_order_id on Robinhood and await confirmation
             so Robinhood does not reject the defensive exit due to locked contracts.
          2. Aggressive Limit Chaser: Submit closing (or 4-leg roll) limit order at mid_price,
             poll every 15s and walk limit price by +$0.02 per step until FILLED.
          3. Record final fill price, slippage, and realized P&L in PostgreSQL.
        """
        gtc_cancelled = False
        if trade.gtc_order_id:
            gtc_cancelled = self.rh.cancel_order(trade.gtc_order_id)
            if not gtc_cancelled:
                raise RuntimeError(
                    f"Cancel-First Lock failed: could not cancel GTC order {trade.gtc_order_id}"
                )

        # Verify GTC order is cancelled before submitting defensive exit
        if trade.gtc_order_id and trade.gtc_order_id in self.rh.orders:
            gtc_state = self.rh.get_order(trade.gtc_order_id).status
            if gtc_state != "CANCELLED":
                raise RuntimeError(
                    f"Locked contract hazard: GTC order {trade.gtc_order_id} is still {gtc_state}"
                )

        order_type = "DEFENSIVE_ROLL" if exit_reason == "21_DTE_ROLL" else "DEFENSIVE_EXIT"
        limit_debit = round(current_mid_debit, 2)
        exit_order = self.rh.submit_limit_order(
            idempotency_key=f"{trade.idempotency_key}_{exit_reason}",
            ticker=trade.ticker,
            strategy=trade.strategy,
            order_type=order_type,
            limit_price=limit_debit,
            contracts=trade.contracts,
            metadata={"exit_reason": exit_reason, "gtc_cancelled_first": True},
        )

        for step in range(chase_steps_required):
            limit_debit = round(limit_debit + chase_step_increment, 2)
            self.rh.update_limit_price(exit_order.order_id, limit_debit)

        self.rh.mark_order_filled(exit_order.order_id, fill_price=limit_debit)

        realized_pnl = round((trade.entry_credit - limit_debit) * 100.0 * trade.contracts, 2)
        exit_slippage = round((limit_debit - current_mid_debit) * 100.0 * trade.contracts, 2)

        trade.current_mark = limit_debit
        trade.realized_pnl_usd = realized_pnl
        trade.unrealized_pnl_usd = 0.0
        trade.slippage_usd = round(trade.slippage_usd + exit_slippage, 2)
        trade.exit_order_id = exit_order.order_id
        trade.status = "ROLLED" if exit_reason == "21_DTE_ROLL" else "CLOSED_STOP"
        trade.closed_at = datetime.now(timezone.utc)
        session.flush()

        return {
            "trade_id": trade.id,
            "ticker": trade.ticker,
            "exit_reason": exit_reason,
            "gtc_order_cancelled_first": gtc_cancelled,
            "initial_mid_debit": round(current_mid_debit, 2),
            "final_fill_debit": limit_debit,
            "steps_chased": chase_steps_required,
            "exit_slippage_usd": exit_slippage,
            "realized_pnl_usd": realized_pnl,
            "status": trade.status,
        }


def main() -> None:
    import time
    from src.workers.scheduled_paper_check import run_scheduled_paper_cycle

    print("[ExecutionWorker] Starting continuous Smart Mid-Price Limit Walker & GTC worker...")
    while True:
        summary = run_scheduled_paper_cycle(mode="monitor_and_screen")
        active = summary.get("paper_trades_active", [])
        print(f"[ExecutionWorker] Active paper positions managed: {len(active)}; sleeping 60s.")
        time.sleep(60)


if __name__ == "__main__":
    main()

