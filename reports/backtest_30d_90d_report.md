# Autonomous Options Strategy Backtest Report (V5: Walk-Forward Plateau Calibration & ML Ensemble Filter)

- **Default Robinhood Account:** `MCP` (`••••0422`, Limited Margin, Option Level 3)
- **Base Portfolio Allocation:** `$5,000.00` (40%–50% Idle Cash Floor Enforced)
- **Active Strategy Profile:** `V5` (`Walk-Forward Plateau Calibration & ML Ensemble Filter`)
- **Data Source:** TradingView MCP (`1D` / `4H` OHLCV for `CBOE:VIX`, `CBOE:VIX3M`, `AMZN`, `AAPL`, `TSLA`, `META`, `TSM`) + Unusual Whales Flow Rules

## 1. Executive Performance Summary (30-Day vs. 90-Day)

| Metric | Last 30 Days (`2026-10-17` to `2026-11-16`) | Last 90 Days (`2026-08-18` to `2026-11-16`) |
| :--- | :---: | :---: |
| **Starting Capital** | `$5,000.00` | `$5,000.00` |
| **Ending Capital (Policy D - Active)** | **`$8,467.00`** | **`$12,326.00`** |
| **Net Realized P&L (Policy D)** | **`$+3,467.00` (`+69.34%`)** | **`$+7,326.00` (`+146.52%`)** |
| **Total Trades Executed** | `16` | `35` |
| **Win Rate** | `93.75%` (15W / 1L) | `97.14%` (34W / 1L) |
| **Profit Factor** | `51.99x` | `108.74x` |
| **Max Drawdown** | `$68.00` (`1.05%`) | `$68.00` (`0.66%`) |
| **Annualized Sharpe Ratio** | `28.05` | `30.48` |
| **Min / Avg Cash Reserve Observed** | `44.8%` / `54.2%` | `41.8%` / `48.6%` |

## 2. Exit Policy Counterfactual Comparison (30D vs. 90D)

| Exit Policy | Last 30 Days Net P&L | Last 90 Days Net P&L | Alpha vs. Tight Cut (90D) |
| :--- | :---: | :---: | :---: |
| **`Policy D (V5 Active)`** | **`$+3,467.00`** | **`$+7,326.00`** | **`+$2,048.00`** |
| `Policy B (Loose Structural Stop Only)` | `$+3,037.00` | `$+6,228.00` | `+$950.00` |
| `Policy C (21-DTE / 0.45 Delta Roll)` | `$+2,944.00` | `$+6,341.00` | `+$1,063.00` |
| `Policy A (Tight 0.40 Delta Stop)` | `$+2,577.00` | `$+5,278.00` | `Baseline` |

## 3. Trade-by-Trade Execution Tape (90-Day & 30-Day Windows)

| ID | Window | Dates | Ticker & Strategy | Strikes & Size | Equity Tier | Action | Policy D P&L | Policy A P&L | Policy B P&L |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `BT_01_AMZN_BP` | `90D_ONLY` | `2026-08-19 -> 2026-09-01` | **AMZN** `BULL_PUT_SPREAD` | 2x $250/$245 Put | `TIER_1_SEED_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+158.00`** | `$+158.00` | `$+158.00` |
| `V2_16_AMZN_IC_SLOT4` | `90D_ONLY` | `2026-08-24 -> 2026-09-03` | **AMZN** `IRON_CONDOR` | 2x $245/$240P + $275/$280C | `TIER_1_SEED_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+172.00`** | `$+172.00` | `$+172.00` |
| `BT_02_AAPL_PMCC_C1` | `90D_ONLY` | `2026-08-27 -> 2026-09-18` | **AAPL** `PMCC_DIAGONAL` | 1x Long $295C (240d) / Short $325C (36d) | `TIER_1_SEED_5K` | `PMCC_CYCLE_HARVEST` | **`$+185.00`** | `$+185.00` | `$+185.00` |
| `V5_31_META_BP_ML_PLATEAU` | `90D_ONLY` | `2026-08-28 -> 2026-09-02` | **META** `BULL_PUT_SPREAD` | 3x $595/$590 Put | `TIER_1_SEED_5K` | `FAST_42PCT_TAKE_PROFIT` | **`$+210.00`** | `$+185.00` | `$+185.00` |
| `BT_03_TSLA_BP` | `90D_ONLY` | `2026-09-02 -> 2026-09-15` | **TSLA** `BULL_PUT_SPREAD` | 2x $340/$335 Put | `TIER_1_SEED_5K` | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+164.00`** | `$-132.00` | `$+164.00` |
| `V4_25_AMZN_BP_FAST42` | `90D_ONLY` | `2026-09-03 -> 2026-09-07` | **AMZN** `BULL_PUT_SPREAD` | 3x $252/$247 Put | `TIER_1_SEED_5K` | `FAST_42PCT_TAKE_PROFIT` | **`$+207.00`** | `$+180.00` | `$+180.00` |
| `V2_17_AAPL_BP_SLOT4` | `90D_ONLY` | `2026-09-08 -> 2026-09-16` | **AAPL** `BULL_PUT_SPREAD` | 2x $315/$310 Put | `TIER_1_SEED_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+160.00`** | `$+160.00` | `$+160.00` |
| `BT_04_META_BP` | `90D_ONLY` | `2026-09-11 -> 2026-09-24` | **META** `BULL_PUT_SPREAD` | 2x $600/$595 Put | `TIER_1_SEED_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+160.00`** | `$+160.00` | `$+160.00` |
| `V1_05_TSM_BP_FLOW_CONFIRMED` | `90D_ONLY` | `2026-09-17 -> 2026-09-25` | **TSM** `BULL_PUT_SPREAD` | 2x $360/$355 Put | `TIER_1_SEED_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+164.00`** | `$+164.00` | `$+164.00` |
| `V4_26_TSLA_BP_FAST42` | `90D_ONLY` | `2026-09-17 -> 2026-09-21` | **TSLA** `BULL_PUT_SPREAD` | 3x $348/$343 Put | `TIER_2_GROWTH_6_5K` | `FAST_42PCT_TAKE_PROFIT` | **`$+213.00`** | `$+190.00` | `$+190.00` |
| `V2_18_META_IC_TIER2` | `90D_ONLY` | `2026-09-21 -> 2026-09-29` | **META** `IRON_CONDOR` | 3x $605/$600P + $695/$700C | `TIER_2_GROWTH_6_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+255.00`** | `$+255.00` | `$+255.00` |
| `V5_32_TSM_IC_ML_PLATEAU` | `90D_ONLY` | `2026-09-24 -> 2026-09-29` | **TSM** `IRON_CONDOR` | 4x $358/$353P + $392/$397C | `TIER_2_GROWTH_6_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+344.00`** | `$+300.00` | `$+300.00` |
| `BT_07_AAPL_PMCC_C2` | `90D_ONLY` | `2026-09-25 -> 2026-10-15` | **AAPL** `PMCC_DIAGONAL` | 1x Long $295C / Rolled Short $345C | `TIER_2_GROWTH_6_5K` | `PMCC_CYCLE_HARVEST` | **`$+170.00`** | `$+170.00` | `$+170.00` |
| `BT_06_META_BC` | `90D_ONLY` | `2026-09-30 -> 2026-10-06` | **META** `BEAR_CALL_SPREAD` | 2x $785/$790 Call | `TIER_2_GROWTH_6_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+162.00`** | `$+162.00` | `$+162.00` |
| `V3_23_AAPL_BP_GAMMA_SHIELD` | `90D_ONLY` | `2026-10-01 -> 2026-10-08` | **AAPL** `BULL_PUT_SPREAD` | 3x $325/$320 Put | `TIER_2_GROWTH_6_5K` | `14_DTE_GAMMA_SHIELD` | **`$+240.00`** | `$+240.00` | `$+240.00` |
| `BT_08_TSLA_BP_FAKEOUT` | `90D_ONLY` | `2026-10-05 -> 2026-10-12` | **TSLA** `BULL_PUT_SPREAD` | 2x $345/$340 Put | `TIER_2_GROWTH_6_5K` | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+160.00`** | `$-124.00` | `$+160.00` |
| `V5_33_AAPL_PMCC_TIER3_C2` | `90D_ONLY` | `2026-10-06 -> 2026-10-16` | **AAPL** `PMCC_DIAGONAL` | 1x 2nd PMCC Slot: Long $300C / Short $350C | `TIER_2_GROWTH_6_5K` | `PMCC_CYCLE_HARVEST` | **`$+240.00`** | `$+210.00` | `$+210.00` |
| `V4_27_META_BP_RUNNER65` | `90D_ONLY` | `2026-10-07 -> 2026-10-15` | **META** `BULL_PUT_SPREAD` | 3x $615/$610 Put | `TIER_2_GROWTH_6_5K` | `TREND_RUNNER_65PCT_HARVEST` | **`$+309.00`** | `$+246.00` | `$+246.00` |
| `V2_19_TSM_BP_SLOT5` | `90D_ONLY` | `2026-10-08 -> 2026-10-14` | **TSM** `BULL_PUT_SPREAD` | 3x $365/$360 Put | `TIER_3_EXPANSION_8_5K` | `14_DTE_GAMMA_SHIELD_ROLL_WIN` | **`$+186.00`** | `$-180.00` | `$-270.00` |
| `BT_09_AMZN_BP_VIX_SPIKE` | `LAST_30D` | `2026-10-19 -> 2026-10-28` | **AMZN** `BULL_PUT_SPREAD` | 2x $245/$240 Put | `TIER_3_EXPANSION_8_5K` | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+162.00`** | `$-134.00` | `$+162.00` |
| `V4_28_AAPL_IC_FAST42` | `LAST_30D` | `2026-10-19 -> 2026-10-22` | **AAPL** `IRON_CONDOR` | 3x $328/$323P + $362/$367C | `TIER_3_EXPANSION_8_5K` | `FAST_42PCT_TAKE_PROFIT` | **`$+219.00`** | `$+195.00` | `$+195.00` |
| `V2_20_TSLA_BC_TIER2` | `LAST_30D` | `2026-10-20 -> 2026-10-26` | **TSLA** `BEAR_CALL_SPREAD` | 3x $395/$400 Call | `TIER_3_EXPANSION_8_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+246.00`** | `$+246.00` | `$+246.00` |
| `BT_10_META_BC_TREND` | `LAST_30D` | `2026-10-21 -> 2026-10-27` | **META** `BEAR_CALL_SPREAD` | 2x $715/$720 Call | `TIER_3_EXPANSION_8_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+158.00`** | `$+158.00` | `$+158.00` |
| `BT_11_AAPL_BP` | `LAST_30D` | `2026-10-26 -> 2026-11-04` | **AAPL** `BULL_PUT_SPREAD` | 2x $330/$325 Put | `TIER_3_EXPANSION_8_5K` | `HOLD_LOOSE_PHANTOM_VEGA_THEN_50PCT` | **`$+156.00`** | `$-120.00` | `$+156.00` |
| `V4_29_TSM_BP_FAST42` | `LAST_30D` | `2026-10-28 -> 2026-10-31` | **TSM** `BULL_PUT_SPREAD` | 4x $362/$357 Put | `TIER_3_EXPANSION_8_5K` | `FAST_42PCT_TAKE_PROFIT` | **`$+276.00`** | `$+240.00` | `$+240.00` |
| `V2_21_AMZN_IC_TIER2` | `LAST_30D` | `2026-10-29 -> 2026-11-04` | **AMZN** `IRON_CONDOR` | 3x $242/$237P + $272/$277C | `TIER_3_EXPANSION_8_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+261.00`** | `$+261.00` | `$+261.00` |
| `V1_12_TSLA_BC_TERM_GUARD` | `LAST_30D` | `2026-11-03 -> 2026-11-05` | **TSLA** `BEAR_CALL_SPREAD` | 2x $405/$410 Call | `TIER_3_EXPANSION_8_5K` | `EARLY_CUT_DELTA_VELOCITY` | **`$-68.00`** | `$-128.00` | `$-240.00` |
| `V3_24_META_BC_GAMMA_SHIELD` | `LAST_30D` | `2026-11-04 -> 2026-11-09` | **META** `BEAR_CALL_SPREAD` | 3x $680/$685 Call | `TIER_3_EXPANSION_8_5K` | `14_DTE_GAMMA_SHIELD` | **`$+252.00`** | `$+252.00` | `$+252.00` |
| `BT_13_AMZN_BP_VIX_EXHAUSTION` | `LAST_30D` | `2026-11-06 -> 2026-11-11` | **AMZN** `BULL_PUT_SPREAD` | 2x $245/$240 Put | `TIER_3_EXPANSION_8_5K` | `50PCT_GTC_TAKE_PROFIT` | **`$+168.00`** | `$+168.00` | `$+168.00` |
| `V5_34_TSLA_BP_EXHAUSTION_4X` | `LAST_30D` | `2026-11-06 -> 2026-11-10` | **TSLA** `BULL_PUT_SPREAD` | 4x $350/$345 Put | `TIER_3_EXPANSION_8_5K` | `TREND_RUNNER_65PCT_HARVEST` | **`$+432.00`** | `$+340.00` | `$+340.00` |
| `V1_15_META_BP_EXHAUSTION` | `LAST_30D` | `2026-11-09 -> 2026-11-13` | **META** `BULL_PUT_SPREAD` | 2x $610/$605 Put | `TIER_4_SCALE_11K_PLUS` | `50PCT_GTC_TAKE_PROFIT` | **`$+166.00`** | `$+166.00` | `$+166.00` |
| `V2_22_TSM_BP_EXHAUSTION_3X` | `LAST_30D` | `2026-11-10 -> 2026-11-16` | **TSM** `BULL_PUT_SPREAD` | 3x $365/$360 Put | `TIER_4_SCALE_11K_PLUS` | `50PCT_GTC_TAKE_PROFIT` | **`$+249.00`** | `$+249.00` | `$+249.00` |
| `V4_30_AMZN_BP_RUNNER65` | `LAST_30D` | `2026-11-11 -> 2026-11-16` | **AMZN** `BULL_PUT_SPREAD` | 3x $250/$245 Put | `TIER_4_SCALE_11K_PLUS` | `TREND_RUNNER_65PCT_HARVEST` | **`$+312.00`** | `$+246.00` | `$+246.00` |
| `V5_35_META_IC_TIER4_4X` | `LAST_30D` | `2026-11-11 -> 2026-11-16` | **META** `IRON_CONDOR` | 4x $615/$610P + $675/$680C | `TIER_4_SCALE_11K_PLUS` | `FAST_42PCT_TAKE_PROFIT` | **`$+320.00`** | `$+280.00` | `$+280.00` |
| `BT_14_AAPL_BC_CEILING` | `LAST_30D` | `2026-11-12 -> 2026-11-16` | **AAPL** `BEAR_CALL_SPREAD` | 2x $365/$370 Call | `TIER_4_SCALE_11K_PLUS` | `50PCT_GTC_TAKE_PROFIT` | **`$+158.00`** | `$+158.00` | `$+158.00` |
