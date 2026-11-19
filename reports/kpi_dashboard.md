# Strategy KPIs & Effectiveness Dashboard (`2026-11-19T06:15:04.485158+00:00`)

- **Default Brokerage Account:** Robinhood `MCP` (`••••0422`, Limited Margin, Level 3)
- **Scheduled Check Cycles Completed:** `5`
- **Active VIX Regime:** `REGIME_3_ELEVATED_FEAR`

## 1. Executive KPI Scorecard

| KPI Metric | This Week (Paper Test) | Last 30 Days (`30D`) | Last 90 Days (`90D`) | Target / Guardrail |
| :--- | :---: | :---: | :---: | :---: |
| **Net Realized P&L** | **`+$314.00`** (`+$221.00` net) | **`+$3467.00`** | **`+$7326.00`** | `>=$100` per 2x spread @ 50% |
| **Portfolio Return (%)** | **`+6.03%`** (`+4.24%` net) | **`+69.34%`** | **`+146.52%`** | Positive EV / week |
| **Win Rate (%)** | `100.0%` (new) / `66.7%` (all) | **`93.75%`** | **`97.14%`** | `>= 75.0%` |
| **Profit Factor** | `3.38x` | **`51.99x`** | **`108.74x`** | `>= 2.00x` |
| **Annualized Sharpe Ratio** | `6.95` | **`28.05`** | **`30.48`** | `>= 2.50` |
| **Max Drawdown** | `$93.00` (`1.79%`) | **`$68.00` (`2.48%`)** | **`$68.00` (`0.66%`)** | `< 6.0%` ($300 daily breaker) |
| **Idle Cash Reserve (Min / Avg)** | `50.4%` | `46.4%` / `54.2%` | **`41.8%` / `48.6%`** | **`40%–50%` Floor** |
| **Phantom Vega Fakeouts Saved** | `1` (`AMZN`) | `2` (`AMZN`, `AAPL`) | **`4`** (`+$642` saved) | Hold when `VIX 4H RSI > 70` & 1H S/R holds |
| **Early ML Breakout Cuts (`0.43 Δ`)** | `1` (`TSM` stop) | `1` (`TSLA` Nov 5) | **`0`** (`+$312` saved vs Policy B) | Cut when `VIX/VIX3M > 1.0` or UW sweep |

## 2. 90-Day Cumulative Equity Curve (`Policy D` vs `Policy B` vs `Policy A`)

```mermaid
xychart-beta
    title "90-Day Portfolio Equity Curve ($5,000 Base - Policy D vs Baselines)"
    x-axis ["Start", "T1(AMZN)", "T2(AMZN)", "T3(AAPL)", "T4(META)", "T5(TSLA)", "T6(AMZN)", "T7(AAPL)", "T8(META)", "T9(TSM)", "T10(TSLA)", "T11(META)", "T12(TSM)", "T13(AAPL)", "T14(META)", "T15(AAPL)", "T16(TSLA)", "T17(AAPL)", "T18(META)", "T19(TSM)", "T20(AMZN)", "T21(AAPL)", "T22(TSLA)", "T23(META)", "T24(AAPL)", "T25(TSM)", "T26(AMZN)", "T27(TSLA)", "T28(META)", "T29(AMZN)", "T30(TSLA)", "T31(META)", "T32(TSM)", "T33(AMZN)", "T34(META)", "T35(AAPL)"]
    y-axis "Portfolio Value ($ USD)" 4700 --> 6800
    line [5000.0, 5158.0, 5330.0, 5515.0, 5725.0, 5889.0, 6096.0, 6256.0, 6416.0, 6580.0, 6793.0, 7048.0, 7392.0, 7562.0, 7724.0, 7964.0, 8124.0, 8364.0, 8673.0, 8859.0, 9021.0, 9240.0, 9486.0, 9644.0, 9800.0, 10076.0, 10337.0, 10269.0, 10521.0, 10689.0, 11121.0, 11287.0, 11536.0, 11848.0, 12168.0, 12326.0]
    line [5000.0, 5158.0, 5330.0, 5515.0, 5700.0, 5864.0, 6044.0, 6204.0, 6364.0, 6528.0, 6718.0, 6973.0, 7273.0, 7443.0, 7605.0, 7845.0, 8005.0, 8215.0, 8461.0, 8191.0, 8353.0, 8548.0, 8794.0, 8952.0, 9108.0, 9348.0, 9609.0, 9369.0, 9621.0, 9789.0, 10129.0, 10295.0, 10544.0, 10790.0, 11070.0, 11228.0]
    line [5000.0, 5158.0, 5330.0, 5515.0, 5700.0, 5568.0, 5748.0, 5908.0, 6068.0, 6232.0, 6422.0, 6677.0, 6977.0, 7147.0, 7309.0, 7549.0, 7425.0, 7635.0, 7881.0, 7701.0, 7567.0, 7762.0, 8008.0, 8166.0, 8046.0, 8286.0, 8547.0, 8419.0, 8671.0, 8839.0, 9179.0, 9345.0, 9594.0, 9840.0, 10120.0, 10278.0]
```
