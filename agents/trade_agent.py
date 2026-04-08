"""
Phase 4 – Step 15: Trade Agent
================================
Converts macro + quant analysis into an actionable trade signal.

Responsibilities:
  1. Run Phase 3 risk engine (PortfolioVaREngine + GJR-GARCH) on the ticker
  2. Generate a position-sized trade signal: {action, size_pct, entry, stop, target}
  3. Apply volatility-halt rule (halt if GARCH vol > threshold)
  4. Compute position sizing via Kelly fraction with GARCH vol adjustment

This agent bridges the Synthetix Alpha quant output with Aladdin Phase 3 risk
infrastructure. It uses the existing risk.var_engine and risk.garch_vol modules.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
from langchain_core.messages import AIMessage

# Make synthetix-alpha importable (for config)
_SYNTH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "synthetix-alpha")
if _SYNTH_ROOT not in sys.path:
    sys.path.insert(0, _SYNTH_ROOT)

from config import settings

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
VOLATILITY_HALT_THRESHOLD = 40.0    # annualised vol % above which no trade
MAX_KELLY_FRACTION = 0.25           # cap Kelly fraction at 25%
MIN_TRADE_SIZE_PCT = 0.01           # minimum position: 1% of portfolio
MAX_TRADE_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT",
                                     str(settings.max_position_size_pct))) / 100.0


def _compute_kelly_fraction(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    garch_annual_vol: float,
) -> float:
    """
    Half-Kelly criterion adjusted for volatility.

    Kelly f* = (p * b - q) / b
    where b = avg_win / avg_loss, p = win_rate, q = 1 - p

    Then scale by (baseline_vol / current_vol) to reduce size in high-vol regimes.
    """
    if avg_loss_pct <= 0 or avg_win_pct <= 0:
        return MIN_TRADE_SIZE_PCT
    b = avg_win_pct / avg_loss_pct
    q = 1 - win_rate
    kelly = (win_rate * b - q) / b
    kelly = max(0.0, kelly) / 2.0     # half-Kelly for safety

    # Vol-adjust: scale down if vol > 15% (normal regime)
    baseline_vol = 15.0
    if garch_annual_vol > 0:
        vol_scale = min(1.0, baseline_vol / garch_annual_vol)
    else:
        vol_scale = 1.0

    return min(max(kelly * vol_scale, MIN_TRADE_SIZE_PCT), MAX_KELLY_FRACTION)


def trade_agent_node(state: AladdinAgentState) -> dict:
    """
    Trade Agent: converts scores into a sized trade signal with stop/target levels.

    Pipeline:
      1. Load stock_data from state
      2. Run GJR-GARCH for current vol + regime
      3. Run Phase 3 VaR engine for portfolio risk
      4. Apply volatility halt if needed
      5. Size position via adjusted Kelly criterion
      6. Compute entry/stop/target based on GARCH vol
    """
    ticker = state.get("ticker", "UNKNOWN")
    stock_data = state.get("stock_data", {})
    confidence = state.get("confidence", 0.5)
    portfolio_value = state.get("portfolio_value", 10_000_000.0)

    log.info("[TradeAgent] Starting for %s, confidence=%.2f", ticker, confidence)

    close_prices = stock_data.get("close", [])
    returns = stock_data.get("returns", [])
    log_returns = stock_data.get("log_returns", [])

    # ── Default signal (safe fallback) ────────────────────────────────────────
    trade_signal: Dict[str, Any] = {
        "action": "HOLD",
        "size_pct": 0.0,
        "size_inr": 0.0,
        "entry_price": close_prices[-1] if close_prices else 0.0,
        "stop_loss_price": 0.0,
        "target_price": 0.0,
        "risk_reward_ratio": 0.0,
        "garch_vol_annual_pct": 0.0,
        "var_99_pct": 0.0,
        "regime": "unknown",
        "halt_reason": None,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    garch_out = None
    var_out = None

    if not close_prices or len(returns) < 30:
        trade_signal["halt_reason"] = "insufficient_data"
        return _return(trade_signal, garch_out, var_out, ticker)

    # ── 1. GJR-GARCH volatility ───────────────────────────────────────────────
    try:
        import pandas as pd
        from risk.garch_vol import GARCHVolEngine
        r_series = pd.Series(log_returns)
        engine = GARCHVolEngine()
        garch_result = engine.fit_and_forecast(r_series, ticker=ticker)
        garch_out = {
            "current_vol_daily_pct":  garch_result.current_vol_daily_pct,
            "current_vol_annual_pct": garch_result.current_vol_annual_pct,
            "current_regime":         garch_result.current_regime,
            "persistence":            garch_result.params.persistence,
            "leverage_gamma":         garch_result.params.gamma,
            "forecasts": [
                {"horizon": f.horizon_days, "vol_annual": f.vol_pct_annual,
                 "var_95": f.var_95_pct, "var_99": f.var_99_pct}
                for f in garch_result.forecasts
            ],
        }
        trade_signal["garch_vol_annual_pct"] = garch_result.current_vol_annual_pct
        trade_signal["regime"] = garch_result.current_regime
        log.info("[TradeAgent] GARCH: annual_vol=%.2f%%, regime=%s",
                 garch_result.current_vol_annual_pct, garch_result.current_regime)
    except Exception as exc:
        log.warning("[TradeAgent] GARCH failed: %s", exc)
        garch_result = None

    # ── 2. Volatility halt ────────────────────────────────────────────────────
    if garch_out and garch_out["current_vol_annual_pct"] > VOLATILITY_HALT_THRESHOLD:
        trade_signal["halt_reason"] = (
            f"volatility_halt: {garch_out['current_vol_annual_pct']:.1f}% "
            f"> {VOLATILITY_HALT_THRESHOLD}% threshold"
        )
        log.warning("[TradeAgent] VOLATILITY HALT for %s: %.1f%%",
                    ticker, garch_out["current_vol_annual_pct"])
        return _return(trade_signal, garch_out, var_out, ticker)

    # ── 3. Portfolio VaR ──────────────────────────────────────────────────────
    try:
        import pandas as pd
        from risk.var_engine import PortfolioVaREngine, PortfolioInput, Position
        prices_df = pd.DataFrame(
            {"Close": close_prices},
            index=pd.bdate_range(end="today", periods=len(close_prices))
        ).rename(columns={"Close": ticker})
        portfolio = PortfolioInput(
            name=f"Single-{ticker}",
            portfolio_value=portfolio_value,
            positions=[Position(ticker, 1.0)],
        )
        var_engine = PortfolioVaREngine()
        var_result = var_engine.compute(portfolio, prices_df)
        var_out = {
            "historical_var_99": var_result.historical.confidence_99,
            "parametric_var_99": var_result.parametric.confidence_99,
            "mc_var_99":         var_result.monte_carlo.confidence_99,
            "historical_es_99":  var_result.historical.es_99,
        }
        trade_signal["var_99_pct"] = var_result.historical.confidence_99
        log.info("[TradeAgent] VaR 99%%: %.3f%%", var_result.historical.confidence_99)
    except Exception as exc:
        log.warning("[TradeAgent] VaR engine failed: %s", exc)

    # ── 4. Determine action from confidence + macro ───────────────────────────
    # Map Synthetix confidence (0-1) to BUY/HOLD/SELL
    # Also read the Consensus action from report if available
    report = state.get("report", {})
    action = "HOLD"
    if report:
        action = (report.get("executive_summary", {}) or {}).get("action", "HOLD")
    else:
        if confidence >= 0.65:
            action = "BUY"
        elif confidence <= 0.35:
            action = "SELL"
        else:
            action = "HOLD"

    # ── 5. Position sizing via adjusted Kelly ─────────────────────────────────
    r_arr = np.array(returns)
    wins  = r_arr[r_arr > 0]
    losses = r_arr[r_arr < 0]
    win_rate   = len(wins) / max(len(r_arr), 1)
    avg_win    = float(np.mean(wins)) if len(wins) else 0.005
    avg_loss   = float(abs(np.mean(losses))) if len(losses) else 0.005
    garch_vol  = garch_out["current_vol_annual_pct"] if garch_out else 20.0

    kelly_fraction = _compute_kelly_fraction(win_rate, avg_win, avg_loss, garch_vol)
    # Scale by confidence (0→0, 1→full Kelly)
    size_pct = kelly_fraction * confidence if action != "HOLD" else 0.0
    size_pct = min(max(size_pct, 0.0), MAX_TRADE_SIZE_PCT)

    # ── 6. Entry / Stop / Target ──────────────────────────────────────────────
    entry = close_prices[-1]
    daily_vol = garch_result.current_vol_daily_pct / 100.0 if garch_out else 0.015
    stop_dist  = 2.0 * daily_vol * entry    # 2σ stop
    target_dist = 3.0 * daily_vol * entry   # 3σ target (1.5 RR minimum)

    if action == "BUY":
        stop_price   = entry - stop_dist
        target_price = entry + target_dist
    elif action == "SELL":
        stop_price   = entry + stop_dist
        target_price = entry - target_dist
    else:
        stop_price = target_price = entry

    rr = (target_dist / stop_dist) if stop_dist > 0 else 0.0

    trade_signal.update({
        "action":            action,
        "size_pct":          round(size_pct * 100, 3),
        "size_inr":          round(size_pct * portfolio_value, 2),
        "entry_price":       round(entry, 2),
        "stop_loss_price":   round(stop_price, 2),
        "target_price":      round(target_price, 2),
        "risk_reward_ratio": round(rr, 2),
        "kelly_fraction":    round(kelly_fraction, 4),
        "win_rate":          round(win_rate, 3),
    })

    log.info(
        "[TradeAgent] %s: action=%s, size=%.2f%% (₹%.0f), RR=%.2f",
        ticker, action, size_pct * 100, size_pct * portfolio_value, rr
    )
    return _return(trade_signal, garch_out, var_out, ticker)


def _return(trade_signal, garch_out, var_out, ticker) -> dict:
    return {
        "trade_signal":  trade_signal,
        "garch_result":  garch_out,
        "var_result":    var_out,
        "messages": [AIMessage(content=(
            f"[Trade Agent] {ticker}: {trade_signal['action']} "
            f"{trade_signal['size_pct']:.2f}% "
            f"(halt={trade_signal.get('halt_reason', 'none')})"
        ))],
    }
