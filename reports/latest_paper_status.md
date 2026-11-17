# Latest Scheduled Paper-Trading Status (`2026-11-17T17:19:35.437006+00:00`)

- **Cycle Mode:** `screener` (`EXECUTION_MODE=PAPER`, `paper_execution_broker=alpaca`)
- **Alpaca Paper Account:** `PA35XZXE8ZZE` (`paper=true`) | **Robinhood Default:** `MCP` (`••••0422`)
- **Active VIX Regime:** `REGIME_3_ELEVATED_FEAR` — Elevated Fear / Prime Credit Harvesting (VIX=22.91, VIX3M=23.02, Ratio=0.995, 4H_RSI=65.4, ExhaustionSignal=True).
- **Post-Allocation Buying Power:** `$2,620.86`

## 1. Potential Trades Summary (Screener Scan)

### Qualifying Candidates

| Ticker | Strategy | Exp (DTE) | Strikes | Contracts | Mid Credit | 50% GTC Target | Collateral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AMZN** | `BULL_PUT_SPREAD` | `2026-12-18` (32d) | `$250/$245` | 2x | `$1.58` | `+$158.00` | `$684.00` |
| **AAPL** | `BEAR_CALL_SPREAD` | `2026-12-18` (32d) | `$365/$370` | 2x | `$1.58` | `+$158.00` | `$684.00` |

### Rejected / Filtered Candidates

| Candidate | Rejection Reason |
| :--- | :--- |
| `TSLA_BULL_PUT` | Failed option chain liquidity, 1/3 width, or cash floor |

## 2. Active Paper Positions & Monitor Status

| Ticker | Strategy | Strikes | Contracts | Entry Credit | 50% GTC Target | Monitor Action | Polling Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AMZN** | `BULL_PUT_SPREAD` | `$250/$245` | 2x | `$1.57` | `$0.79` | `NORMAL_HOLD` | `300s` |
| **AAPL** | `BEAR_CALL_SPREAD` | `$365/$370` | 2x | `$1.57` | `$0.79` | `NORMAL_HOLD` | `300s` |
