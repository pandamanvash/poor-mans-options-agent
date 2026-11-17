# Poor Man's Options Agent (`poor-mans-options-agent`)

Autonomous Options Screener, Smart Mid-Price Execution & Weekly Machine-Learning Risk Platform for a **$5,000 Options Portfolio**, integrating **Unusual Whales MCP**, **TradingView MCP**, and **Robinhood MCP**.

## Key Quantitative & Risk Architecture

1. **Portfolio Allocation & Contract Scaling (`src/config/portfolio_rules.py`):**
   - **40%–50% Cash Reserve Floor:** Keeps at least $2,000–$2,500 in idle cash (60% in `Regime 4` panic/backwardation).
   - **Dynamic `>=$100` 50%-Target Scaling:** Scales vertical credit spreads (typically **2 contracts** on a $5-wide spread collecting ~$1.58–$1.60 credit = $316–$320 total credit $\rightarrow$ **+$158–$160 realized profit at the 50% GTC target**) so `0.50 * credit * 100 * contracts >= $100.00`.
   - **PMCC Golden Rule Assertion:** Enforces `net_debit_paid < (short_strike - long_strike)` when `IV Rank < 30%` and `VIX < 18`.
2. **Multi-MCP Screening & `CBOE:VIX` / `CBOE:VIX3M` Pipeline (`src/workers/screener_job.py`):**
   - **Step 1 (`TradingView MCP`):** Classifies `VIX` & `VIX3M` across Regimes 1–4, detecting the **`VIX` Exhaustion Signal** (`VIX 4H RSI > 65` + rejection wick) and **`VIX` Backwardation** (`VIX / VIX3M > 1.02`).
   - **Step 2 (`Unusual Whales MCP`):** Excludes tickers with earnings within 40 days and filters for `IV Rank > 40%`, ask-side sweeps, and dark pool accumulation/distribution.
   - **Step 3 (`TradingView MCP`):** Anchors Short Put strikes safely below horizontal support / POC (`4H RSI` in `35–48`) and Short Call strikes above multi-week resistance (`4H RSI > 68`).
3. **Fault-Tolerant Execution Worker (`src/workers/execution_worker.py`):**
   - Supports `PAPER`, `SEMI_AUTO`, and `FULL_AUTO` modes.
   - **Smart Mid-Price Limit Walker:** Steps limit price by `$0.01–$0.02` down to `min_credit_floor` (max 4 steps).
   - **Automatic 50% GTC Take-Profit Staging:** Stages a closing GTC limit order at `round(fill_credit * 0.50, 2)` immediately on entry fill.
   - **Cancel-First Lock & Aggressive Chaser:** Cancels open GTC orders and confirms unlock before walking defensive exits/rolls by `$0.02` every 15s.
4. **Intra-Week Risk Monitor & Weekly ML Calibration (`src/workers/monitor_worker.py`, `src/ml_engine/`):**
   - Switches to 60s high-frequency polling at `>= 0.38 Delta`, holds through **Phantom Vega Drawdowns** (`VIX 4H RSI > 70` while 1H support holds), cuts early on `VIX` backwardation or confirmed Unusual Whales breakouts (`>= 0.43 Delta`), and retrains a `GradientBoostingClassifier` every Friday at 4:30 PM ET across Policies A, B, C, and D.

## Running Tests & Weekly Paper Simulation

```bash
pytest -v
python -m src.workers.paper_trading_runner
```
