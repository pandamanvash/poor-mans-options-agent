from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


class GeminiFillAdvisor:
    """
    Low-effort Google AI Studio (Gemini Flash-Lite) advisor for:
      1. Stalled multi-leg option orders (> 15 minutes unfilled on Alpaca Paper):
         Evaluates live mid vs natural credit, spread width, and $100 profit floor
         to decide REPLACE / HOLD / CANCEL.
      2. Friday ML Backtest concise executive review commentary for Markdown reports.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model or "gemini-2.5-flash-lite"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "REPLACE_IN_RENDER_DASHBOARD")

    def _call_gemini(self, prompt: str, json_mode: bool = True) -> str | None:
        if not self.is_configured():
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        gen_cfg: dict[str, Any] = {
            "temperature": 0.1,
            "maxOutputTokens": 256,
        }
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(url, params={"key": self.api_key}, json=body)
                if resp.status_code != 200:
                    logger.info("Gemini AI Studio HTTP %s: %s", resp.status_code, resp.text[:160])
                    return None
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    return None
                parts = (candidates[0].get("content") or {}).get("parts") or []
                return "".join(str(p.get("text") or "") for p in parts).strip()
        except Exception as exc:
            logger.info("Gemini AI Studio request skipped: %s", exc)
            return None

    def advise_stalled_order(
        self,
        legs_desc: str,
        contracts: int,
        spread_width: float,
        current_limit_credit: float,
        live_mid_credit: float,
        live_natural_credit: float,
        min_credit_floor: float,
        age_minutes: float,
    ) -> dict[str, Any]:
        """
        Ask Gemini (low effort) how to handle a multi-leg credit spread unfilled after >= 15 mins.
        Strictly enforces hard Python safety bounds ($100 50%-profit floor and >= 23% spread width).
        """
        hard_min_for_100 = round(100.0 / max(1.0, 0.50 * 100.0 * contracts), 2)
        hard_floor = round(max(hard_min_for_100, spread_width * 0.23), 2)

        prompt = (
            "You are an options execution assistant for Alpaca Paper Trading (which fills multi-leg "
            "credit spreads at natural_credit = short_bid - long_ask, not mid).\n"
            f"Order: {contracts}x {legs_desc} (width=${spread_width:.2f})\n"
            f"Open age: {age_minutes:.1f} minutes\n"
            f"Current limit credit: ${current_limit_credit:.2f}\n"
            f"Live mid credit: ${live_mid_credit:.2f}\n"
            f"Live natural credit (short_bid - long_ask): ${live_natural_credit:.2f}\n"
            f"Standard min_credit_floor: ${min_credit_floor:.2f} | Hard safety floor: ${hard_floor:.2f}\n\n"
            "Rules:\n"
            "- If live_natural_credit >= hard_floor and live_natural_credit < current_limit_credit, "
            "recommend action='REPLACE' with recommended_credit=live_natural_credit so it fills.\n"
            "- If live_natural_credit < hard_floor * 0.70 and age_minutes >= 20, recommend action='CANCEL' "
            "to free buying power from an illiquid spread.\n"
            "- Otherwise recommend action='HOLD'.\n"
            "Return strict JSON: {\"action\": \"REPLACE\"|\"HOLD\"|\"CANCEL\", "
            "\"recommended_credit\": number|null, \"rationale\": \"short 1-sentence reason\"}"
        )

        raw = self._call_gemini(prompt, json_mode=True)
        if not raw:
            return {"action": "HOLD", "recommended_credit": None, "rationale": "Gemini unavailable"}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"action": "HOLD", "recommended_credit": None, "rationale": "Unparseable Gemini response"}

        action = str(parsed.get("action") or "HOLD").upper().strip()
        if action not in ("REPLACE", "HOLD", "CANCEL"):
            action = "HOLD"

        rec_credit = parsed.get("recommended_credit")
        if action == "REPLACE":
            try:
                rec_val = round(float(rec_credit), 2)
            except (TypeError, ValueError):
                rec_val = 0.0
            # Enforce strict safety floor in Python regardless of LLM output
            if rec_val < hard_floor or rec_val >= current_limit_credit:
                action = "HOLD"
                rec_credit = None
            else:
                rec_credit = rec_val
        else:
            rec_credit = None

        return {
            "action": action,
            "recommended_credit": rec_credit,
            "rationale": str(parsed.get("rationale") or "")[:200],
            "model": self.model,
        }

    def summarize_friday_backtest(
        self,
        equity_usd: float,
        promoted_policy: str,
        best_ev_usd: float,
        grade: str,
        open_spreads_count: int,
    ) -> str:
        prompt = (
            "Write a concise 2-sentence weekly review note for an options trader's Friday ML calibration report.\n"
            f"Account Equity: ${equity_usd:,.2f}, Open Spreads: {open_spreads_count}, "
            f"Promoted Policy: {promoted_policy}, Expected Value per 2x spread: ${best_ev_usd:+.2f}, "
            f"Portfolio Grade: {grade}.\n"
            "Be objective, concise, and highlight one actionable focus for next week."
        )
        text = self._call_gemini(prompt, json_mode=False)
        return text or (
            f"Weekly ML calibration promoted `{promoted_policy}` (${best_ev_usd:+.2f} EV per 2x spread) "
            f"with portfolio grade `{grade}` across {open_spreads_count} open spread(s)."
        )
