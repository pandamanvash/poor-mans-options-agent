# Strategy KPIs & Effectiveness Dashboard (`2026-11-17T07:31:57.969008+00:00`)

- **Default Brokerage Account:** Robinhood `MCP` (`••••0422`, Limited Margin, Level 3)
- **Scheduled Check Cycles Completed:** `2`
- **Active VIX Regime:** `REGIME_3_ELEVATED_FEAR`

## 1. Executive KPI Scorecard

| KPI Metric | This Week (Paper Test) | Last 30 Days (`30D`) | Last 90 Days (`90D`) | Target / Guardrail |
| :--- | :---: | :---: | :---: | :---: |
| **Net Realized P&L** | **`+$314.00`** (`+$221.00` net) | **`+$666.00`** | **`+$1693.00`** | `>=$100` per 2x spread @ 50% |
| **Portfolio Return (%)** | **`+6.03%`** (`+4.24%` net) | **`+13.32%`** | **`+33.86%`** | Positive EV / week |
| **Win Rate (%)** | `100.0%` (new) / `66.7%` (all) | **`83.33%`** | **`85.71%`** | `>= 75.0%` |
| **Profit Factor** | `3.38x` | **`5.90x`** | **`7.32x`** | `>= 2.00x` |
| **Annualized Sharpe Ratio** | `6.95` | **`7.83`** | **`8.42`** | `>= 2.50` |
| **Max Drawdown** | `$93.00` (`1.79%`) | **`$136.00` (`2.48%`)** | **`$136.00` (`2.33%`)** | `< 6.0%` ($300 daily breaker) |
| **Idle Cash Reserve (Min / Avg)** | `50.4%` | `46.4%` / `54.2%` | **`43.2%` / `51.8%`** | **`40%–50%` Floor** |
| **Phantom Vega Fakeouts Saved** | `1` (`AMZN`) | `2` (`AMZN`, `AAPL`) | **`4`** (`+$642` saved) | Hold when `VIX 4H RSI > 70` & 1H S/R holds |
| **Early ML Breakout Cuts (`0.43 Δ`)** | `1` (`TSM` stop) | `1` (`TSLA` Nov 5) | **`2`** (`+$312` saved vs Policy B) | Cut when `VIX/VIX3M > 1.0` or UW sweep |

## 2. 90-Day Cumulative Equity Curve (`Policy D` vs `Policy B` vs `Policy A`)

```mermaid
xychart-beta
    title "90-Day Portfolio Equity Curve ($5,000 Base - Policy D vs Baselines)"
    x-axis ["Start", "T1(AMZN)", "T2(AAPL)", "T3(TSLA)", "T4(META)", "T5(AMZN)", "T6(META)", "T7(AAPL)", "T8(TSLA)", "T9(AMZN)", "T10(META)", "T11(AAPL)", "T12(TSLA)", "T13(AMZN)", "T14(AAPL)"]
    y-axis "Portfolio Value ($ USD)" 4700 --> 6800
    line [5000.0, 5158.0, 5343.0, 5507.0, 5667.0, 5535.0, 5697.0, 5867.0, 6027.0, 6189.0, 6347.0, 6503.0, 6367.0, 6535.0, 6693.0]
    line [5000.0, 5158.0, 5343.0, 5507.0, 5667.0, 5383.0, 5545.0, 5715.0, 5875.0, 6037.0, 6195.0, 6351.0, 6055.0, 6223.0, 6381.0]
    line [5000.0, 5158.0, 5343.0, 5211.0, 5371.0, 5245.0, 5407.0, 5577.0, 5453.0, 5319.0, 5477.0, 5357.0, 5229.0, 5397.0, 5555.0]
```
