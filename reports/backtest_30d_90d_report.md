# Autonomous Options Strategy Backtest Report: Last 30 Days & Last 90 Days

- **Default Robinhood Account:** `MCP` (`••••0422`, Limited Margin, Option Level 3)
- **Base Portfolio Allocation:** `$5,000.00` (40%–50% Idle Cash Floor Enforced)
- **Data Source:** TradingView MCP (`1D` / `4H` OHLCV for `CBOE:VIX`, `CBOE:VIX3M`, `AMZN`, `AAPL`, `TSLA`, `META`) + Unusual Whales Flow Rules

## 1. Executive Performance Summary (30-Day vs. 90-Day)

| Metric | Last 30 Days (`2026-10-17` to `2026-11-16`) | Last 90 Days (`2026-08-18` to `2026-11-16`) |
| :--- | :---: | :---: |
| **Starting Capital** | `$5,000.00` | `$5,000.00` |
| **Ending Capital (Policy D - Active)** | **`$5,666.00`** | **`$6,693.00`** |
| **Net Realized P&L (Policy D)** | **`$+666.00` (`+13.32%`)** | **`$+1,693.00` (`+33.86%`)** |
| **Win Rate** | `83.33%` (5W / 1L) | `85.71%` (12W / 2L) |
| **Profit Factor** | `5.90x` | `7.32x` |
| **Max Drawdown** | `$136.00` (`2.48%`) | `$136.00` (`2.33%`) |
| **Annualized Sharpe Ratio** | `7.83` | `8.42` |
| **Min / Avg Cash Reserve Observed** | `46.4%` / `54.2%` | `43.2%` / `51.8%` |

## 2. Exit Policy Counterfactual Comparison (30D vs. 90D)

| Exit Policy | Last 30 Days Net P&L | Last 90 Days Net P&L | Alpha vs. Tight Cut (90D) |
| :--- | :---: | :---: | :---: |
| **`Policy D (Loose-First + VIX + ML Early Cut)`** | **`$+666.00`** | **`$+1,693.00`** | **`+$1,138.00`** |
| `Policy B (Loose Structural Stop Only)` | `$+506.00` | `$+1,381.00` | `+$826.00` |
| `Policy C (21-DTE / 0.45 Delta Roll)` | `$+429.00` | `$+1,224.00` | `+$669.00` |
| `Policy A (Tight 0.40 Delta Stop)` | `$+102.00` | `$+555.00` | `Baseline` |

## 3. Trade-by-Trade Execution Tape (90-Day & 30-Day Windows)

| ID | Window | Dates | Ticker & Strategy | Strikes & Size | VIX / Regime | Policy D Action | Policy D P&L | Policy A P&L | Policy B P&L |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `BT_01_AMZN_BP` | `90D_ONLY` | `2026-08-19 -> 2026-09-01` | **AMZN** `BULL_PUT_SPREAD` | 2x $250/$245 Put | `14.89` (`REGIME_1_LOW_VOL`) | `50PCT_GTC_TAKE_PROFIT` | **`$+158.00`** | `$+158.00` | `$+158.00` |
| `BT_02_AAPL_PMCC_C1` | `90D_ONLY` | `2026-08-27 -> 2026-09-18` | **AAPL** `PMCC_DIAGONAL` | 1x Long $295C (240d) / Short $325C (36d) | `14.51` (`REGIME_1_LOW_VOL`) | `PMCC_CYCLE_HARVEST` | **`$+185.00`** | `$+185.00` | `$+185.00` |
| `BT_03_TSLA_BP` | `90D_ONLY` | `2026-09-02 -> 2026-09-15` | **TSLA** `BULL_PUT_SPREAD` | 2x $340/$335 Put | `16.34` (`REGIME_2_NORMAL`) | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+164.00`** | `$-132.00` | `$+164.00` |
| `BT_04_META_BP` | `90D_ONLY` | `2026-09-11 -> 2026-09-24` | **META** `BULL_PUT_SPREAD` | 2x $600/$595 Put | `15.72` (`REGIME_2_NORMAL`) | `50PCT_GTC_TAKE_PROFIT` | **`$+160.00`** | `$+160.00` | `$+160.00` |
| `BT_05_AMZN_BP_BRK` | `90D_ONLY` | `2026-09-16 -> 2026-09-22` | **AMZN** `BULL_PUT_SPREAD` | 2x $248/$243 Put | `17.84` (`REGIME_2_NORMAL`) | `EARLY_CUT_ML_OVERRIDE` | **`$-132.00`** | `$-126.00` | `$-284.00` |
| `BT_06_META_BC` | `90D_ONLY` | `2026-09-30 -> 2026-10-06` | **META** `BEAR_CALL_SPREAD` | 2x $785/$790 Call | `15.67` (`REGIME_2_NORMAL`) | `50PCT_GTC_TAKE_PROFIT` | **`$+162.00`** | `$+162.00` | `$+162.00` |
| `BT_07_AAPL_PMCC_C2` | `90D_ONLY` | `2026-09-25 -> 2026-10-15` | **AAPL** `PMCC_DIAGONAL` | 1x Long $295C / Rolled Short $345C | `14.81` (`REGIME_1_LOW_VOL`) | `PMCC_CYCLE_HARVEST` | **`$+170.00`** | `$+170.00` | `$+170.00` |
| `BT_08_TSLA_BP_FAKEOUT` | `90D_ONLY` | `2026-10-05 -> 2026-10-12` | **TSLA** `BULL_PUT_SPREAD` | 2x $345/$340 Put | `16.07` (`REGIME_2_NORMAL`) | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+160.00`** | `$-124.00` | `$+160.00` |
| `BT_09_AMZN_BP_VIX_SPIKE` | `LAST_30D` | `2026-10-19 -> 2026-10-28` | **AMZN** `BULL_PUT_SPREAD` | 2x $245/$240 Put | `18.44` (`REGIME_2_NORMAL`) | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+162.00`** | `$-134.00` | `$+162.00` |
| `BT_10_META_BC_TREND` | `LAST_30D` | `2026-10-21 -> 2026-10-27` | **META** `BEAR_CALL_SPREAD` | 2x $715/$720 Call | `16.68` (`REGIME_2_NORMAL`) | `50PCT_GTC_TAKE_PROFIT` | **`$+158.00`** | `$+158.00` | `$+158.00` |
| `BT_11_AAPL_BP` | `LAST_30D` | `2026-10-26 -> 2026-11-04` | **AAPL** `BULL_PUT_SPREAD` | 2x $330/$325 Put | `18.39` (`REGIME_2_NORMAL`) | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+156.00`** | `$-120.00` | `$+156.00` |
| `BT_12_TSLA_BP_BACKWARDATION_SAVE` | `LAST_30D` | `2026-11-02 -> 2026-11-05` | **TSLA** `BULL_PUT_SPREAD` | 2x $370/$365 Put | `21.15` (`REGIME_3_ELEVATED_FEAR`) | `EARLY_CUT_ML_OVERRIDE` | **`$-136.00`** | `$-128.00` | `$-296.00` |
| `BT_13_AMZN_BP_VIX_EXHAUSTION` | `LAST_30D` | `2026-11-06 -> 2026-11-11` | **AMZN** `BULL_PUT_SPREAD` | 2x $245/$240 Put | `25.64` (`REGIME_3_ELEVATED_FEAR`) | `50PCT_GTC_TAKE_PROFIT` | **`$+168.00`** | `$+168.00` | `$+168.00` |
| `BT_14_AAPL_BC_CEILING` | `LAST_30D` | `2026-11-12 -> 2026-11-16` | **AAPL** `BEAR_CALL_SPREAD` | 2x $365/$370 Call | `18.70` (`REGIME_2_NORMAL`) | `50PCT_GTC_TAKE_PROFIT` | **`$+158.00`** | `$+158.00` | `$+158.00` |
