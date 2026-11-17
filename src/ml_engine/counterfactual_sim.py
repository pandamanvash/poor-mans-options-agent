from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.ml_engine.feature_store import StressEpisode


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_name: str
    total_pnl_usd: float
    ev_per_2contract_spread_usd: float
    win_rate: float
    avoided_premature_stops: int
    early_breakout_saves: int


class CounterfactualSimulator:
    """
    Simulates 4 Competing Exit Policies across all stressed episodes (normalized to 2-contract spreads):
      - Policy A (Tight Cut): Exit immediately at 0.40 Delta.
      - Policy B (Loose Structural Stop - Baseline): Hold until 1H candle closes past TV S/R + 0.15 ATR or Short Delta >= 0.52.
      - Policy C (Defensive Roll): Roll out 30 days and adjust strikes at 0.45 Delta for a net credit.
      - Policy D (ML Classifier - GradientBoosting): Predicts P(Fakeout_Reversal) vs. P(True_Breakout)
        to dynamically hold fakeouts for +$160 profit (50% target) while cutting true breakouts early at 0.43 Delta.
    """

    def simulate_all_policies(
        self,
        episodes: Sequence[StressEpisode],
        ml_breakout_probs: Sequence[float],
        prob_threshold: float = 0.60,
    ) -> dict[str, PolicyEvaluation]:
        if not episodes:
            raise ValueError("episodes cannot be empty")

        eval_a = self._eval_policy_a(episodes)
        eval_b = self._eval_policy_b(episodes)
        eval_c = self._eval_policy_c(episodes)
        eval_d = self._eval_policy_d(episodes, ml_breakout_probs, prob_threshold=prob_threshold)

        return {
            eval_a.policy_name: eval_a,
            eval_b.policy_name: eval_b,
            eval_c.policy_name: eval_c,
            eval_d.policy_name: eval_d,
        }

    def _eval_policy_a(self, episodes: Sequence[StressEpisode]) -> PolicyEvaluation:
        """Policy A (Tight Cut): Exit immediately at 0.40 Delta (~-$130 loss on 2-contract spread)."""
        pnls: list[float] = []
        early_saves = 0
        for ep in episodes:
            loss_per_contract = max(0.55, ep.spread_mark_at_stress - ep.entry_credit)
            pnl = round(-loss_per_contract * 100.0 * 2, 2)
            pnls.append(pnl)
            if ep.true_breakout == 1:
                early_saves += 1
        return self._summarize("Policy_A_Tight_Cut_040_Delta", pnls, 0, early_saves)

    def _eval_policy_b(self, episodes: Sequence[StressEpisode]) -> PolicyEvaluation:
        """
        Policy B (Loose Structural Stop - Baseline):
        Recovers +50% target (+$160 on 2 contracts) on Fakeout Reversals (true_breakout=0),
        but suffers deeper structural stop loss (~-$290 on 2 contracts) on True Breakouts (true_breakout=1).
        """
        pnls: list[float] = []
        avoided_stops = 0
        for ep in episodes:
            if ep.true_breakout == 0:
                pnl = round(0.50 * ep.entry_credit * 100.0 * 2, 2)
                avoided_stops += 1
            else:
                pnl = -290.00
            pnls.append(pnl)
        return self._summarize("Policy_B_Loose_Structural_Stop", pnls, avoided_stops, 0)

    def _eval_policy_c(self, episodes: Sequence[StressEpisode]) -> PolicyEvaluation:
        """
        Policy C (Defensive Roll at 0.45 Delta):
        Collects small net credit roll; recovers modestly (+$75) on reversals,
        reduces loss (-$195) on sustained breakouts.
        """
        pnls: list[float] = []
        avoided_stops = 0
        for ep in episodes:
            if ep.true_breakout == 0:
                pnl = 75.00
                avoided_stops += 1
            else:
                pnl = -195.00
            pnls.append(pnl)
        return self._summarize("Policy_C_Defensive_Roll_045_Delta", pnls, avoided_stops, 0)

    def _eval_policy_d(
        self,
        episodes: Sequence[StressEpisode],
        ml_breakout_probs: Sequence[float],
        prob_threshold: float = 0.60,
    ) -> PolicyEvaluation:
        """
        Policy D (ML Classifier - GradientBoosting):
        Uses P(True_Breakout) >= prob_threshold to cut true breakouts early at 0.43 Delta (~-$135),
        while holding fakeout/Phantom Vega episodes loose to harvest the full 50% target (+$160).
        """
        pnls: list[float] = []
        avoided_stops = 0
        early_saves = 0
        for ep, prob in zip(episodes, ml_breakout_probs):
            predict_breakout = prob >= prob_threshold
            if predict_breakout:
                loss_per_contract = max(0.60, (ep.spread_mark_at_stress - ep.entry_credit) * 0.90)
                pnl = round(-loss_per_contract * 100.0 * 2, 2)
                if ep.true_breakout == 1:
                    early_saves += 1
            else:
                if ep.true_breakout == 0:
                    pnl = round(0.50 * ep.entry_credit * 100.0 * 2, 2)
                    avoided_stops += 1
                else:
                    pnl = -290.00
            pnls.append(pnl)
        return self._summarize("Policy_D_ML_Classifier_GBM", pnls, avoided_stops, early_saves)

    @staticmethod
    def _summarize(
        name: str,
        pnls: Sequence[float],
        avoided_stops: int,
        early_saves: int,
    ) -> PolicyEvaluation:
        total = sum(pnls)
        mean_ev = total / len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        return PolicyEvaluation(
            policy_name=name,
            total_pnl_usd=round(float(total), 2),
            ev_per_2contract_spread_usd=round(float(mean_ev), 2),
            win_rate=round(wins / len(pnls), 4),
            avoided_premature_stops=avoided_stops,
            early_breakout_saves=early_saves,
        )
