from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.db.models import ActiveRiskThreshold
from src.ml_engine.counterfactual_sim import CounterfactualSimulator, PolicyEvaluation
from src.ml_engine.feature_store import FEATURE_NAMES, FeatureStore


class _PurePythonGBMClassifier:
    """
    Lightweight Gradient Boosting decision-stump ensemble used when scikit-learn is not installed
    in the local sandbox, matching Scikit-Learn's fit() / predict_proba() / feature_importances_ API.
    """

    def __init__(self, n_estimators: int = 40, learning_rate: float = 0.15) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees: list[tuple[int, float, float, float]] = []
        self.feature_importances_: list[float] = []
        self.base_log_odds: float = 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

    def fit(self, x_mat: list[list[float]], y_vec: list[int]) -> "_PurePythonGBMClassifier":
        n_samples = len(x_mat)
        n_features = len(x_mat[0])
        pos_rate = max(0.05, min(0.95, sum(y_vec) / n_samples))
        self.base_log_odds = math.log(pos_rate / (1.0 - pos_rate))
        f_scores = [self.base_log_odds for _ in range(n_samples)]
        feat_gains = [0.0 for _ in range(n_features)]

        for round_idx in range(self.n_estimators):
            residuals = [y_vec[i] - self._sigmoid(f_scores[i]) for i in range(n_samples)]
            best_feat = 0
            best_thresh = 0.0
            best_left_val = 0.0
            best_right_val = 0.0
            best_loss = float("inf")

            # Deterministic feature subsampling across boosting rounds (stochastic GBM style)
            candidate_features = [
                (round_idx * 3 + offset) % n_features for offset in range(4)
            ]
            for f_idx in candidate_features:
                col_vals = sorted(set(row[f_idx] for row in x_mat))
                for thresh in col_vals:
                    left_res = [residuals[i] for i in range(n_samples) if x_mat[i][f_idx] <= thresh]
                    right_res = [residuals[i] for i in range(n_samples) if x_mat[i][f_idx] > thresh]
                    if not left_res or not right_res:
                        continue
                    l_mean = sum(left_res) / len(left_res)
                    r_mean = sum(right_res) / len(right_res)
                    mse = sum((r - l_mean) ** 2 for r in left_res) + sum(
                        (r - r_mean) ** 2 for r in right_res
                    )
                    if mse < best_loss:
                        best_loss = mse
                        best_feat = f_idx
                        best_thresh = thresh
                        best_left_val = l_mean
                        best_right_val = r_mean

            self.trees.append((best_feat, best_thresh, best_left_val, best_right_val))
            feat_gains[best_feat] += abs(best_left_val - best_right_val)
            for i in range(n_samples):
                step_val = (
                    best_left_val if x_mat[i][best_feat] <= best_thresh else best_right_val
                )
                f_scores[i] += self.learning_rate * step_val

        total_gain = sum(feat_gains) or 1.0
        self.feature_importances_ = [g / total_gain for g in feat_gains]
        return self

    def predict_proba(self, x_mat: list[list[float]]) -> list[list[float]]:
        probs: list[list[float]] = []
        for row in x_mat:
            score = self.base_log_odds
            for feat_idx, thresh, l_val, r_val in self.trees:
                score += self.learning_rate * (l_val if row[feat_idx] <= thresh else r_val)
            p1 = self._sigmoid(score)
            probs.append([round(1.0 - p1, 4), round(p1, 4)])
        return probs


def _build_classifier():
    try:
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

        return GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42
        )
    except ImportError:
        return _PurePythonGBMClassifier(n_estimators=40, learning_rate=0.15)


class WeeklyModelTrainer:
    """
    Trains GradientBoostingClassifier on weekly + historical stress episodes,
    evaluates counterfactual EV across Policies A, B, C, D, updates config/risk_thresholds.json
    and PostgreSQL active_risk_thresholds, and writes the weekly Markdown audit report.
    """

    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        simulator: CounterfactualSimulator | None = None,
        thresholds_path: Path | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self.feature_store = feature_store or FeatureStore()
        self.simulator = simulator or CounterfactualSimulator()
        self.thresholds_path = thresholds_path or settings.risk_thresholds_path
        self.reports_dir = reports_dir or settings.reports_dir

    def run_weekly_calibration(
        self,
        session: Any | None = None,
        audit_date: str = "2026_11_20",
    ) -> dict[str, Any]:
        episodes = self.feature_store.build_training_episodes(session=session)
        x_mat, y_vec = self.feature_store.to_matrices(episodes)

        clf = _build_classifier()
        clf.fit(x_mat, y_vec)
        raw_probs = clf.predict_proba(x_mat)
        breakout_probs = [float(row[1]) for row in raw_probs]

        evaluations = self.simulator.simulate_all_policies(
            episodes=episodes,
            ml_breakout_probs=breakout_probs,
            prob_threshold=0.60,
        )

        best_policy_name = max(
            evaluations.keys(),
            key=lambda k: evaluations[k].ev_per_2contract_spread_usd,
        )
        best_eval = evaluations[best_policy_name]

        feature_importances = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(FEATURE_NAMES, clf.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        }

        updated_thresholds = {
            "version": f"ml-calibrated-{audit_date}",
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "active_policy": best_policy_name,
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
            "uw_breakout_darkpool_zscore_min": 1.70,
            "uw_breakout_ask_sweep_ratio_min": 0.68,
            "ml_true_breakout_prob_threshold": 0.60,
            "roll_dte_trigger": 21,
            "max_daily_realized_loss_pct": 0.06,
            "max_daily_realized_loss_usd": 300.0,
            "feature_importances": feature_importances,
        }

        self.thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        self.thresholds_path.write_text(
            json.dumps(updated_thresholds, indent=2) + "\n",
            encoding="utf-8",
        )

        if session is not None:
            db_record = ActiveRiskThreshold(
                version=updated_thresholds["version"],
                active_policy=best_policy_name,
                baseline_short_delta_ceiling=updated_thresholds["baseline_short_delta_ceiling"],
                ml_early_cut_delta_ceiling=updated_thresholds["ml_early_cut_delta_ceiling"],
                tv_1h_sr_atr_buffer=updated_thresholds["tv_1h_sr_atr_buffer"],
                uw_breakout_darkpool_zscore_min=updated_thresholds["uw_breakout_darkpool_zscore_min"],
                uw_breakout_ask_sweep_ratio_min=updated_thresholds["uw_breakout_ask_sweep_ratio_min"],
                vix_backwardation_ratio_trigger=updated_thresholds["vix_backwardation_ratio_trigger"],
                payload_json=json.dumps(updated_thresholds),
            )
            session.add(db_record)
            session.flush()

        report_path = self._write_markdown_report(
            audit_date=audit_date,
            episodes_count=len(episodes),
            evaluations=evaluations,
            best_eval=best_eval,
            feature_importances=feature_importances,
            updated_thresholds=updated_thresholds,
        )

        return {
            "episodes_evaluated": len(episodes),
            "best_policy": best_policy_name,
            "best_ev_per_2contract_spread_usd": best_eval.ev_per_2contract_spread_usd,
            "policy_evaluations": {
                k: {
                    "total_pnl_usd": v.total_pnl_usd,
                    "ev_per_2contract_spread_usd": v.ev_per_2contract_spread_usd,
                    "win_rate": v.win_rate,
                    "avoided_premature_stops": v.avoided_premature_stops,
                    "early_breakout_saves": v.early_breakout_saves,
                }
                for k, v in evaluations.items()
            },
            "feature_importances": feature_importances,
            "report_path": str(report_path),
            "thresholds_path": str(self.thresholds_path),
        }

    def _write_markdown_report(
        self,
        audit_date: str,
        episodes_count: int,
        evaluations: dict[str, PolicyEvaluation],
        best_eval: PolicyEvaluation,
        feature_importances: dict[str, float],
        updated_thresholds: dict[str, Any],
    ) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.reports_dir / f"weekly_ml_audit_{audit_date}.md"

        lines = [
            f"# Weekly ML Counterfactual Risk Audit ({audit_date})",
            "",
            f"- **Episodes Simulated:** `{episodes_count}` (Live Intra-Week Stress Snapshots + Lookalike Episodes)",
            f"- **Promoted Policy:** `{best_eval.policy_name}`",
            f"- **Expected Value (EV) per 2-Contract Spread:** `${best_eval.ev_per_2contract_spread_usd:+.2f}`",
            "",
            "## 1. Competing Exit Policy Comparison (2-Contract Vertical Spreads)",
            "",
            "| Exit Policy | Total Simulated P&L | EV / 2-Contract Spread | Win Rate | Fakeouts Saved | Early Breakout Cuts |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for policy_name, ev in evaluations.items():
            lines.append(
                f"| `{policy_name}` | `${ev.total_pnl_usd:+,.2f}` | **`${ev.ev_per_2contract_spread_usd:+.2f}`** | "
                f"`{ev.win_rate * 100:.1f}%` | `{ev.avoided_premature_stops}` | `{ev.early_breakout_saves}` |"
            )

        lines.extend(
            [
                "",
                "## 2. Top GradientBoosting Feature Importances",
                "",
                "| Feature | Importance |",
                "| :--- | :---: |",
            ]
        )
        for feat, imp in list(feature_importances.items())[:6]:
            lines.append(f"| `{feat}` | `{imp:.4f}` |")

        lines.extend(
            [
                "",
                "## 3. Promoted `risk_thresholds.json` Parameters",
                "",
                "```json",
                json.dumps(updated_thresholds, indent=2),
                "```",
                "",
            ]
        )
        report_file.write_text("\n".join(lines), encoding="utf-8")
        return report_file
