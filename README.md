# Quantitative Portfolio Analyzer & Optimizer (NIFTY / BankNifty)

An institutional-grade pairs trading and volatility system specifically engineered for the Indian markets (NIFTY & BankNifty futures). This repository demonstrates a complete, end-to-end algorithmic pipeline that prioritizes rigorous mathematics, pristine data handling, paranoid risk engine formulation, and high-frequency execution.

## 🚀 System Architecture overview

This project fundamentally abandons retail setups (like plain vector-based loops and simple correlation mapping) in favor of dynamic mathematical relationships and event-driven backtesting execution.

### Phase 1: Data Infrastructure
- **Institutional-Grade Fetching:** Designed around tick-level API data streams.
- **Time-Series Storage:** Prepared for integration with specialized engines such as TimescaleDB or ArcticDB logic.
- **Micro-scrubber:** Includes strict alignment and data ingestion scrubbing to resolve forward-fills and market halts seamlessly.

### Phase 2: Alpha Modeling & Math
- **Cointegration Testing:** Utilizes `statsmodels` implementations of Engle-Granger and Johansen cointegration to lock solid mean-reverting structures rather than naive correlation.
- **Dynamic Volatility Control:** Deploys **GARCH(1,1)** models (`arch` library) to trigger spreads accurately based on real-time volatility thresholds instead of static standard deviation.
- **Kalman Filtering:** Leverages tick-by-tick hedge ratio adjustments through `pykalman`, adapting trade conditions synchronously.

### Phase 3: The Event-Driven Backtester
- Bypasses traditional Pandas data frame lookahead biases.
- Accurately models market friction, including correct deduction of STT, transaction charges, limits, and random slippage latency to confirm viable alpha parameters physically.

### Phase 4: Risk Management Engine
- Historical tracking for Peak-to-Trough Drawdowns and overall Calmar Ratios.
- Calculates baseline `99% Historical Value-at-Risk (VaR)` models out-of-the-box.
- Built to generate sophisticated execution tearsheets detailing Sharpe, Sortino ratios, and monthly heatmaps.

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
