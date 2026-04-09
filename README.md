# Quantitative Portfolio Analyzer & Optimizer (NIFTY / BankNifty)

An institutional-grade pairs trading and volatility system specifically engineered for the Indian markets (NIFTY & BankNifty futures). This repository demonstrates a complete, end-to-end algorithmic pipeline that prioritizes rigorous mathematics, pristine data handling, paranoid risk engine formulation, and high-frequency execution.

## 🧠 Mathematical Models Implemented

This project structurally avoids relying on elementary retail indicators (like simple moving averages or static RSI levels) and instead employs institutional statistical models to discover genuine, mathematically verifiable alpha.

### 1. Engle-Granger & Johansen Cointegration
Instead of relying on basic correlation (which can decouple and blow up an account when structural regimes change), the core of the strategy relies heavily on **Cointegration**. 
- **Purpose**: We mathematically prove that a linear combination of two assets (the *spread*) is stationary—meaning it will reliably revert to its historical mean. 
- **Implementation**: The system runs Augmented Dickey-Fuller (ADF) tests on the residual spread vectors to determine optimal trading pairs across the NIFTY50 grid with sub `0.05` P-values. 

### 2. GARCH(1,1) Volatility Modeling
A static (+/- $2\sigma$) entry threshold fails when market volatility structurally shifts (e.g., black swan events). To account for this, the system shifts boundary entry conditions actively.
- **Purpose**: Adjusting trade execution dynamically dependent on rolling volatility limits.
- **Implementation**: Uses **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** scaling, measuring recent market variance to widen or tighten acceptable deviations on Z-score spreads. If the environment becomes excessively volatile, the threshold geometrically expands to avoid false breaks.

### 3. Kalman Filters (Dynamic Hedge Ratios)
Using an Standard Ordinary Least Squares (OLS) regression algorithm across a multi-year lookback generates a static hedge ratio that inevitably goes stale.
- **Purpose**: Updating the hedging requirement continually on every incoming price tick. 
- **Implementation**: We model the relationship between two assets as a hidden state, continually predicting the structural hedge ratio and refining it securely based on measurement errors via the `pykalman` library. 

### 4. 99% Historical Value-at-Risk (VaR)
This models the literal worst-case mathematical threshold limits that capital logic must safely navigate.
- **Purpose**: Informing algorithmic boundaries to dictate strict margin requirements dynamically based on historical worst-loss iterations.

## 🚀 System Architecture overview

This project fundamentally abandons retail setups (like plain vector-based loops and simple correlation mapping) in favor of dynamic mathematical relationships and event-driven backtesting execution.

### Phase 1: Data Infrastructure
- **Institutional-Grade Fetching:** Designed around tick-level API data streams.
- **Time-Series Storage:** Prepared for integration with specialized engines such as TimescaleDB or ArcticDB logic.
- **Micro-scrubber:** Includes strict alignment and data ingestion scrubbing to resolve forward-fills and market halts seamlessly.

### Phase 2: Alpha Modeling & Math
- Continual evaluation driven strictly by the algorithmic models listed above (Cointegration setups, variance estimation through GARCH, and real-time execution bounds resolved dynamically with Kalman filter recursion).

### Phase 3: The Event-Driven Backtester
- Bypasses traditional Pandas data frame lookahead biases.
- Accurately models market friction, including correct deduction of STT, transaction charges, limits, and random slippage latency to confirm viable alpha parameters physically.

### Phase 4: Risk Management Engine
- Historical tracking for Peak-to-Trough Drawdowns and overall Calmar Ratios.
- Calculates automated pipeline outputs leveraging `QuantStats` and `Pyfolio` execution sheets.

## 💻 Tech Stack
- **Backend Analytics Engine:** Python 3.11+, Pandas, NumPy, Statsmodels, Scikit-learn, Arch, PyKalman.
- **UI Terminal:** `Next.js` (React) + `TypeScript` mapped to a professional **Bloomberg-style aesthetic** layout (dense layout, strict monospace fonts, high contrast execution dashboards).
- **Orchestration:** Managed directly via integrated Helm charts and configured Docker nodes for maximum elasticity and component stability.

## 📈 Dashboard Layout Structure
The repository UI follows strict financial standards emphasizing minimal distractions:
1. **Spread Charts:** Focuses exclusively on Real Spread Z-Scores crossed with GARCH volatility bands.
2. **Cointegration Matrix:** Real-time pairing values mapping half-life values and p-values simultaneously.
3. **Execution Log Terminal:** Active raw log pipelines tracking order routings dynamically.
4. **Live Risk Feed:** Unobtrusive updates monitoring VaR, Unrealized PnL, and current Exposure utilization directly across the ticker grid.
