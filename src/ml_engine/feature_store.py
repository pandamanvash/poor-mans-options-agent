from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from src.db.models import StressEvent

FEATURE_NAMES: list[str] = [
    "short_delta",
    "dist_to_tv_sr_atr",
    "tv_4h_rsi",
    "uw_darkpool_flow_zscore",
    "uw_ask_sweep_ratio",
    "vix_level",
    "vix_zscore_20d",
    "vix_4h_rsi",
    "vix_1h_slope",
    "vix_term_structure_ratio",
    "dte",
]


@dataclass(frozen=True)
class StressEpisode:
    episode_id: str
    ticker: str
    strategy: str
    contracts: int
    entry_credit: float
    spread_mark_at_stress: float
    short_delta: float
    dist_to_tv_sr_atr: float
    tv_4h_rsi: float
    uw_darkpool_flow_zscore: float
    uw_ask_sweep_ratio: float
    vix_level: float
    vix_zscore_20d: float
    vix_4h_rsi: float
    vix_1h_slope: float
    vix_term_structure_ratio: float
    dte: int
    true_breakout: int  # 1 = True Structural Breakout (cut early wins), 0 = Fakeout / Phantom Vega Reversal (hold loose wins)

    def to_feature_vector(self) -> list[float]:
        return [
            float(self.short_delta),
            float(self.dist_to_tv_sr_atr),
            float(self.tv_4h_rsi),
            float(self.uw_darkpool_flow_zscore),
            float(self.uw_ask_sweep_ratio),
            float(self.vix_level),
            float(self.vix_zscore_20d),
            float(self.vix_4h_rsi),
            float(self.vix_1h_slope),
            float(self.vix_term_structure_ratio),
            float(self.dte),
        ]


class FeatureStore:
    """Loads intra-week stress telemetry and historical lookalike episodes for weekly ML training."""

    def build_training_episodes(
        self,
        session: Any | None = None,
        stress_jsonl_path: Path = Path("data/stress_events.jsonl"),
    ) -> list[StressEpisode]:
        episodes: list[StressEpisode] = list(self._seed_historical_lookalikes())
        seen_keys: set[str] = set()

        if session is not None:
            db_events: Sequence[StressEvent] = session.query(StressEvent).all()
            for idx, ev in enumerate(db_events):
                is_breakout = int(
                    ev.tv_1h_close_breach
                    or ev.vix_term_structure_ratio > 1.0
                    or (abs(ev.uw_darkpool_net_flow_1h) > 400_000 and ev.uw_sweep_ratio >= 0.68)
                )
                dp_z = round(abs(ev.uw_darkpool_net_flow_1h) / 250_000.0, 2)
                ep_key = f"{ev.ticker}_{ev.short_delta}_{ev.spread_mark}"
                seen_keys.add(ep_key)
                episodes.append(
                    StressEpisode(
                        episode_id=f"LIVE_WK_{idx+1}_{ev.ticker}",
                        ticker=ev.ticker,
                        strategy=ev.strategy,
                        contracts=max(2, ev.contracts),
                        entry_credit=1.60,
                        spread_mark_at_stress=ev.spread_mark,
                        short_delta=ev.short_delta,
                        dist_to_tv_sr_atr=ev.dist_to_tv_sr_atr,
                        tv_4h_rsi=ev.tv_4h_rsi,
                        uw_darkpool_flow_zscore=dp_z,
                        uw_ask_sweep_ratio=ev.uw_sweep_ratio,
                        vix_level=ev.vix_level,
                        vix_zscore_20d=ev.vix_zscore_20d,
                        vix_4h_rsi=ev.vix_4h_rsi,
                        vix_1h_slope=ev.vix_1h_slope,
                        vix_term_structure_ratio=ev.vix_term_structure_ratio,
                        dte=ev.dte,
                        true_breakout=is_breakout,
                    )
                )

        if stress_jsonl_path.exists():
            for idx, line in enumerate(stress_jsonl_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                ev_d = json.loads(line)
                ep_key = f"{ev_d.get('ticker')}_{ev_d.get('short_delta')}_{ev_d.get('spread_mark')}"
                if ep_key in seen_keys:
                    continue
                seen_keys.add(ep_key)
                is_breakout = int(
                    bool(ev_d.get("tv_1h_close_breach"))
                    or float(ev_d.get("vix_term_structure_ratio", 0.95)) > 1.0
                    or (
                        abs(float(ev_d.get("uw_darkpool_net_flow_1h", 0.0))) > 400_000
                        and float(ev_d.get("uw_sweep_ratio", 0.5)) >= 0.68
                    )
                )
                dp_z = round(abs(float(ev_d.get("uw_darkpool_net_flow_1h", 0.0))) / 250_000.0, 2)
                episodes.append(
                    StressEpisode(
                        episode_id=f"ALPACA_LOG_{idx+1}_{ev_d.get('ticker', 'UNK')}",
                        ticker=str(ev_d.get("ticker", "UNK")),
                        strategy=str(ev_d.get("strategy", "BULL_PUT_SPREAD")),
                        contracts=max(2, int(ev_d.get("contracts", 2))),
                        entry_credit=1.60,
                        spread_mark_at_stress=float(ev_d.get("spread_mark", 2.0)),
                        short_delta=float(ev_d.get("short_delta", 0.40)),
                        dist_to_tv_sr_atr=float(ev_d.get("dist_to_tv_sr_atr", 0.0)),
                        tv_4h_rsi=float(ev_d.get("tv_4h_rsi", 50.0)),
                        uw_darkpool_flow_zscore=dp_z,
                        uw_ask_sweep_ratio=float(ev_d.get("uw_sweep_ratio", 0.5)),
                        vix_level=float(ev_d.get("vix_level", 20.0)),
                        vix_zscore_20d=float(ev_d.get("vix_zscore_20d", 0.0)),
                        vix_4h_rsi=float(ev_d.get("vix_4h_rsi", 50.0)),
                        vix_1h_slope=float(ev_d.get("vix_1h_slope", 0.0)),
                        vix_term_structure_ratio=float(ev_d.get("vix_term_structure_ratio", 0.95)),
                        dte=int(ev_d.get("dte", 30)),
                        true_breakout=is_breakout,
                    )
                )
        return episodes

    @staticmethod
    def to_matrices(episodes: list[StressEpisode]) -> tuple[list[list[float]], list[int]]:
        x_mat = [ep.to_feature_vector() for ep in episodes]
        y_vec = [int(ep.true_breakout) for ep in episodes]
        return x_mat, y_vec

    @staticmethod
    def _seed_historical_lookalikes() -> list[StressEpisode]:
        """Curated lookalike stress episodes spanning Phantom Vega fakeouts and True Institutional Breakouts."""
        return [
            # Fakeout Reversals / Phantom Vega Spikes (true_breakout = 0)
            StressEpisode("HIST_01_AMZN_VEGA", "AMZN", "BULL_PUT_SPREAD", 2, 1.60, 2.35, 0.41, 0.35, 42.5, 0.45, 0.52, 23.4, 1.65, 72.8, -0.18, 0.96, 28, 0),
            StressEpisode("HIST_02_AAPL_WICK", "AAPL", "BEAR_CALL_SPREAD", 2, 1.58, 2.25, 0.39, 0.28, 69.5, 0.60, 0.55, 21.8, 1.20, 64.0, 0.14, 0.95, 29, 0),
            StressEpisode("HIST_03_TSLA_FAKE", "TSLA", "BULL_PUT_SPREAD", 2, 1.65, 2.40, 0.42, 0.40, 44.0, 0.80, 0.58, 24.1, 1.78, 74.2, -0.25, 0.97, 26, 0),
            StressEpisode("HIST_04_META_HOLD", "META", "BULL_PUT_SPREAD", 2, 1.62, 2.30, 0.40, 0.52, 41.2, 0.30, 0.49, 22.5, 1.40, 71.1, -0.12, 0.96, 31, 0),
            StressEpisode("HIST_05_AMZN_POC", "AMZN", "BULL_PUT_SPREAD", 2, 1.58, 2.20, 0.39, 0.61, 45.8, 0.55, 0.51, 20.9, 1.05, 68.4, -0.09, 0.94, 27, 0),
            StressEpisode("HIST_06_GOOGL_WICK", "GOOGL", "BULL_PUT_SPREAD", 2, 1.60, 2.45, 0.43, 0.22, 39.8, 0.90, 0.60, 24.8, 1.85, 73.5, -0.22, 0.98, 24, 0),
            StressEpisode("HIST_07_AAPL_EXH", "AAPL", "BEAR_CALL_SPREAD", 2, 1.60, 2.32, 0.41, 0.31, 71.0, 0.40, 0.53, 22.9, 1.50, 66.2, 0.19, 0.97, 30, 0),
            StressEpisode("HIST_08_MSFT_VEGA", "MSFT", "BULL_PUT_SPREAD", 2, 1.56, 2.18, 0.38, 0.48, 43.1, 0.25, 0.48, 23.1, 1.55, 71.9, -0.15, 0.95, 33, 0),
            # True Structural Breakouts / Backwardation & Institutional Sweeps (true_breakout = 1)
            StressEpisode("HIST_09_TSM_BRK", "TSM", "BULL_PUT_SPREAD", 2, 1.76, 2.65, 0.44, -0.18, 34.2, 2.15, 0.74, 25.4, 1.95, 62.1, 0.38, 1.03, 19, 1),
            StressEpisode("HIST_10_NVDA_DUMP", "NVDA", "BULL_PUT_SPREAD", 2, 1.65, 2.80, 0.47, -0.32, 28.5, 2.45, 0.78, 26.8, 2.20, 64.5, 0.45, 1.04, 22, 1),
            StressEpisode("HIST_11_AMD_SWEEP", "AMD", "BULL_PUT_SPREAD", 2, 1.60, 2.70, 0.45, -0.24, 31.0, 1.98, 0.72, 24.9, 1.88, 60.5, 0.31, 1.01, 25, 1),
            StressEpisode("HIST_12_INTC_GAP", "INTC", "BULL_PUT_SPREAD", 2, 1.58, 2.90, 0.49, -0.45, 26.4, 2.60, 0.81, 27.5, 2.35, 66.0, 0.52, 1.05, 18, 1),
            StressEpisode("HIST_13_TSLA_MELT", "TSLA", "BEAR_CALL_SPREAD", 2, 1.62, 2.75, 0.46, -0.28, 76.4, 2.05, 0.75, 19.2, 0.40, 48.0, -0.35, 0.92, 21, 1),
            StressEpisode("HIST_14_META_BRK", "META", "BULL_PUT_SPREAD", 2, 1.60, 2.68, 0.44, -0.19, 33.5, 1.89, 0.70, 25.9, 2.05, 63.0, 0.29, 1.02, 23, 1),
        ]
