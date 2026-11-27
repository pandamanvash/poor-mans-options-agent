"""Notification service for trade tickets, 1-click approvals, and risk alerts (with Discord Webhook formatting)."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


def format_spread_type(strategy_or_cp: str) -> str:
    raw = str(strategy_or_cp or "").upper().strip()
    if "BULL_PUT" in raw or raw == "P":
        return "Bull Put Credit Spread"
    if "BEAR_CALL" in raw or raw == "C":
        return "Bear Call Credit Spread"
    if "IRON_CONDOR" in raw:
        return "Iron Condor Credit Spread"
    if "PMCC" in raw:
        return "PMCC Diagonal"
    return raw.replace("_", " ").title() or "Credit Spread"


def format_management_strategy(
    strategy_or_cp: str,
    target_debit: float | None = None,
    compact: bool = False,
) -> str:
    tp_label = (
        f"50% GTC TP (${target_debit:.2f} debit)"
        if target_debit is not None and target_debit > 0
        else "50% GTC TP"
    )
    if compact:
        return f"v5: {tp_label} | 0.41Δ ML Cut | 18 DTE Roll | 1H S/R+0.15ATR Stop"
    return (
        f"v5 Stacked: {tp_label} (42% Fast <=4d / 65% Runner) | "
        f"0.41Δ ML Cut & +0.07/d Δ-Velocity | 14 DTE Gamma Shield | "
        f"18 DTE Roll | 1H S/R+0.15ATR Hard Stop"
    )


class Notifier:
    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = webhook_url if webhook_url is not None else settings.webhook_notification_url
        self.sent_messages: list[dict[str, Any]] = []

    def send_alert(self, title: str, message: str, metadata: dict[str, Any] | None = None) -> bool:
        payload: dict[str, Any] = {
            "title": title,
            "message": message,
            "metadata": metadata or {},
        }
        self.sent_messages.append(payload)
        logger.info("[ALERT] %s: %s", title, message)
        if not self.webhook_url:
            return True

        # Format native Discord webhook payload if targeting discord.com/api/webhooks
        if "discord.com/api/webhooks" in self.webhook_url:
            full_width_keys = {"open_positions", "management_strategy", "details"}
            fields = []
            for k, v in list((metadata or {}).items())[:10]:
                val_str = str(v)
                is_multiline = "\n" in val_str or str(k).lower() in full_width_keys or len(val_str) > 48
                fields.append(
                    {
                        "name": str(k),
                        "value": val_str[:1020] if is_multiline else f"`{val_str}`",
                        "inline": not is_multiline,
                    }
                )
            http_body: dict[str, Any] = {
                "username": "Poor Man's Options Agent",
                "content": f"**{title}**\n{message}",
                "embeds": [
                    {
                        "title": title,
                        "description": message,
                        "color": 1096065,
                        "fields": fields,
                    }
                ],
            }
        else:
            http_body = payload

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(self.webhook_url, json=http_body)
                resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Webhook delivery failed: %s", exc)
            return False

    def notify_order_submitted(
        self,
        ticker: str,
        strategy: str,
        contracts: int,
        strikes: str,
        expiration_date: str,
        limit_price: float,
        order_type: str = "ENTRY_LIMIT",
        order_id: str = "",
    ) -> bool:
        spread_type = format_spread_type(strategy)
        is_closing = order_type in ("GTC_TAKE_PROFIT", "DEFENSIVE_EXIT", "DEFENSIVE_ROLL", "CLOSING_LIMIT")
        px_label = f"${abs(limit_price):.2f} debit" if is_closing else f"${abs(limit_price):.2f} credit"
        target_debit = None if is_closing else round(abs(limit_price) * 0.50, 2)
        mgmt = format_management_strategy(strategy, target_debit=target_debit)
        action_label = "CLOSING ORDER SUBMITTED" if is_closing else "ORDER SUBMITTED"
        return self.send_alert(
            title=f"{action_label} — {contracts}x {ticker} {spread_type}",
            message=f"{contracts}x **{ticker}** `{expiration_date}` `{strikes}` ({spread_type}) @ `{px_label}`",
            metadata={
                "spread_type": spread_type,
                "contracts": f"{contracts}x",
                "limit_price": px_label,
                "order_id": order_id[:16] if order_id else "pending",
                "management_strategy": mgmt,
            },
        )

    def notify_position_filled(
        self,
        ticker: str,
        strategy: str,
        contracts: int,
        strikes: str,
        expiration_date: str,
        fill_price: float,
        target_close_price: float | None = None,
        is_closing: bool = False,
        order_id: str = "",
    ) -> bool:
        spread_type = format_spread_type(strategy)
        px_label = f"${abs(fill_price):.2f} debit" if is_closing else f"${abs(fill_price):.2f} credit"
        tp_debit = (
            target_close_price
            if target_close_price is not None
            else (None if is_closing else round(abs(fill_price) * 0.50, 2))
        )
        mgmt = format_management_strategy(strategy, target_debit=tp_debit)
        title_prefix = "POSITION CLOSED" if is_closing else "POSITION FILLED"
        return self.send_alert(
            title=f"{title_prefix} — {contracts}x {ticker} {spread_type}",
            message=(
                f"{contracts}x **{ticker}** `{expiration_date}` `{strikes}` ({spread_type}) "
                f"filled @ `{px_label}`"
            ),
            metadata={
                "spread_type": spread_type,
                "contracts": f"{contracts}x",
                "fill_price": px_label,
                "order_id": order_id[:16] if order_id else "filled",
                "management_strategy": mgmt,
            },
        )

    def request_semi_auto_approval(self, intent_summary: dict[str, Any]) -> bool:
        return self.send_alert(
            title=f"SEMI_AUTO Trade Entry Approval Required: {intent_summary.get('ticker')}",
            message=(
                f"Approve {intent_summary.get('contracts')}x {intent_summary.get('ticker')} "
                f"{intent_summary.get('strategy')} ({intent_summary.get('expiration_date')}) "
                f"@ ${intent_summary.get('mid_credit'):.2f} credit "
                f"(50% Target: +${intent_summary.get('expected_50pct_profit_usd'):.2f})"
            ),
            metadata=intent_summary,
        )
