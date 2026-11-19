from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    FULL_AUTO = "FULL_AUTO"
    SEMI_AUTO = "SEMI_AUTO"
    PAPER = "PAPER"


class Settings(BaseModel):
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data/options_agent_paper.db")
    )
    execution_mode: ExecutionMode = Field(
        default_factory=lambda: ExecutionMode(os.getenv("EXECUTION_MODE", "PAPER"))
    )
    robinhood_account_number: str = Field(
        default_factory=lambda: os.getenv("ROBINHOOD_ACCOUNT_NUMBER", "512500422")
    )
    robinhood_account_nickname: str = Field(
        default_factory=lambda: os.getenv("ROBINHOOD_ACCOUNT_NICKNAME", "MCP")
    )
    alpaca_paper_account_number: str = Field(
        default_factory=lambda: os.getenv("ALPACA_PAPER_ACCOUNT_NUMBER", "PA35XZXE8ZZE")
    )
    paper_execution_broker: str = Field(
        default_factory=lambda: os.getenv("PAPER_EXECUTION_BROKER", "alpaca")
    )
    active_execution_broker: str = Field(
        default_factory=lambda: os.getenv("ACTIVE_EXECUTION_BROKER", "alpaca")
    )
    alpaca_trading_enabled: bool = Field(
        default_factory=lambda: os.getenv("ALPACA_TRADING_ENABLED", "true").lower() == "true"
    )
    robinhood_trading_enabled: bool = Field(
        default_factory=lambda: os.getenv("ROBINHOOD_TRADING_ENABLED", "false").lower() == "true"
    )
    unusual_whales_api_key: str = Field(
        default_factory=lambda: os.getenv("UNUSUAL_WHALES_API_KEY", "uw-mcp-bridge")
    )
    tradingview_mcp_url: str = Field(
        default_factory=lambda: os.getenv("TRADINGVIEW_MCP_URL", "http://localhost:8081/mcp")
    )
    robinhood_mcp_url: str = Field(
        default_factory=lambda: os.getenv("ROBINHOOD_MCP_URL", "http://localhost:8082/mcp")
    )
    webhook_notification_url: str = Field(
        default_factory=lambda: os.getenv("WEBHOOK_NOTIFICATION_URL", "")
    )
    risk_thresholds_path: Path = Field(
        default=Path("config/risk_thresholds.json"),
    )
    stress_events_jsonl_path: Path = Field(
        default=Path("data/stress_events.jsonl"),
    )
    reports_dir: Path = Field(
        default=Path("reports"),
    )

    def load_risk_thresholds(self) -> dict[str, Any]:
        if self.risk_thresholds_path.exists():
            return json.loads(self.risk_thresholds_path.read_text(encoding="utf-8"))
        return {
            "version": "default-fallback",
            "stress_polling_delta_trigger": 0.38,
            "baseline_short_delta_ceiling": 0.52,
            "ml_early_cut_delta_ceiling": 0.43,
            "tv_1h_sr_atr_buffer": 0.15,
            "vix_phantom_vega_4h_rsi_min": 70.0,
            "vix_exhaustion_4h_rsi_min": 65.0,
            "vix_floor_bounce_4h_rsi_max": 35.0,
            "vix_backwardation_ratio_trigger": 1.0,
            "vix_panic_regime_threshold": 28.0,
            "vix_panic_term_structure_threshold": 1.02,
            "uw_breakout_darkpool_zscore_min": 1.75,
            "uw_breakout_ask_sweep_ratio_min": 0.68,
            "ml_true_breakout_prob_threshold": 0.62,
            "roll_dte_trigger": 21,
            "max_daily_realized_loss_pct": 0.06,
            "max_daily_realized_loss_usd": 300.0,
        }


settings = Settings()
