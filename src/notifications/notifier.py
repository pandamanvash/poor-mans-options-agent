"""Notification service for trade tickets, 1-click approvals, and risk alerts (with Discord Webhook formatting)."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


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
            fields = [
                {"name": str(k), "value": f"`{v}`", "inline": True}
                for k, v in (metadata or {}).items()
            ][:10]
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
