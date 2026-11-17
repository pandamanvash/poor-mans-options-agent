# Weekly ML Counterfactual Risk Audit (2026_11_20)

- **Episodes Simulated:** `14` (Live Intra-Week Stress Snapshots + Lookalike Episodes)
- **Promoted Policy:** `Policy_D_ML_Classifier_GBM`
- **Expected Value (EV) per 2-Contract Spread:** `$+5.60`

## 1. Competing Exit Policy Comparison (2-Contract Vertical Spreads)

| Exit Policy | Total Simulated P&L | EV / 2-Contract Spread | Win Rate | Fakeouts Saved | Early Breakout Cuts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Policy_A_Tight_Cut_040_Delta` | `$-2,466.00` | **`$-176.14`** | `0.0%` | `0` | `6` |
| `Policy_B_Loose_Structural_Stop` | `$-461.00` | **`$-32.93`** | `57.1%` | `8` | `0` |
| `Policy_C_Defensive_Roll_045_Delta` | `$-570.00` | **`$-40.71`** | `57.1%` | `8` | `0` |
| `Policy_D_ML_Classifier_GBM` | `$+78.40` | **`$+5.60`** | `57.1%` | `8` | `6` |

## 2. Top GradientBoosting Feature Importances

| Feature | Importance |
| :--- | :---: |
| `short_delta` | `0.3487` |
| `dist_to_tv_sr_atr` | `0.3063` |
| `uw_darkpool_flow_zscore` | `0.2268` |
| `uw_ask_sweep_ratio` | `0.1182` |
| `tv_4h_rsi` | `0.0000` |
| `vix_level` | `0.0000` |

## 3. Promoted `risk_thresholds.json` Parameters

```json
{
  "version": "ml-calibrated-2026_11_20",
  "calibrated_at": "2026-11-17T07:40:31.877779+00:00",
  "active_policy": "Policy_D_ML_Classifier_GBM",
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
  "uw_breakout_darkpool_zscore_min": 1.7,
  "uw_breakout_ask_sweep_ratio_min": 0.68,
  "ml_true_breakout_prob_threshold": 0.6,
  "roll_dte_trigger": 21,
  "max_daily_realized_loss_pct": 0.06,
  "max_daily_realized_loss_usd": 300.0,
  "feature_importances": {
    "short_delta": 0.3487,
    "dist_to_tv_sr_atr": 0.3063,
    "uw_darkpool_flow_zscore": 0.2268,
    "uw_ask_sweep_ratio": 0.1182,
    "tv_4h_rsi": 0.0,
    "vix_level": 0.0,
    "vix_zscore_20d": 0.0,
    "vix_4h_rsi": 0.0,
    "vix_1h_slope": 0.0,
    "vix_term_structure_ratio": 0.0,
    "dte": 0.0
  }
}
```
