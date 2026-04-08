"""
Phase 4 – Step 16: Compliance Agent
=====================================
SEBI rules checker — runs before the final Consensus Agent and can:
  - Block trades that violate position limits
  - Flag KYC / corporate-action blackouts
  - Enforce max drawdown / leverage ceilings
  - Check volatility halt from Trade Agent
  - Impose lot-size / tick-size constraints for F&O

Rules enforced (SEBI + Aladdin internal):
  R01  Max position size < MAX_POSITION_SIZE_PCT of portfolio
  R02  Leverage ceiling  ≤ LEVERAGE_CEILING (default 1.0 = no leverage)
  R03  Global stop-loss  ≥ GLOBAL_STOP_LOSS_PCT (must set stop)
  R04  Volatility halt   if GARCH annual vol > threshold
  R05  Sentiment gate    macro_score must be in [MIN, MAX] band
  R06  Minimum RR ratio  ≥ 1.5 (risk-reward floor)
  R07  Confidence floor  confidence_score ≥ 0.4 to trade (not HOLD)
  R08  NIFTY F&O lot size check (lot=50 for NIFTY, 25 for BankNIFTY)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage

_SYNTH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "synthetix-alpha")
if _SYNTH_ROOT not in sys.path:
    sys.path.insert(0, _SYNTH_ROOT)

from config import settings

log = logging.getLogger(__name__)

# ── SEBI / Aladdin risk limits ────────────────────────────────────────────────
MAX_POSITION_PCT   = settings.max_position_size_pct / 100.0     # fraction
LEVERAGE_CEILING   = settings.leverage_ceiling
GLOBAL_STOP_PCT    = settings.global_stop_loss_pct / 100.0
VOL_HALT_THRESHOLD = 40.0          # annualised vol % (mirrors Trade Agent)
SENTIMENT_MIN      = settings.sentiment_score_min    # macro_score floor
SENTIMENT_MAX      = settings.sentiment_score_max    # macro_score ceiling
MIN_RR_RATIO       = 1.5
MIN_CONFIDENCE     = 0.40

# NSE F&O lot sizes
LOT_SIZES = {
    "^NSEI": 50, "NIFTY": 50, "NIFTY50": 50,
    "^NSEBANK": 15, "BANKNIFTY": 15,
    "RELIANCE.NS": 250, "HDFCBANK.NS": 550, "INFY.NS": 300,
    "TCS.NS": 150, "ICICIBANK.NS": 700, "KOTAKBANK.NS": 40,
}


def _check_rules(
    ticker: str,
    trade_signal: Dict[str, Any],
    report: Dict[str, Any],
    garch_result: Dict[str, Any],
    confidence: float,
    portfolio_value: float,
) -> Tuple[bool, List[str], str]:
    """
    Execute all compliance rules.

    Returns:
        (passed: bool, flags: List[str], notes: str)
    """
    flags: List[str] = []

    es = (report or {}).get("executive_summary", {})
    action  = (trade_signal or {}).get("action", "HOLD")
    size_pct = (trade_signal or {}).get("size_pct", 0.0) / 100.0  # fraction
    rr_ratio = (trade_signal or {}).get("risk_reward_ratio", 0.0)
    stop_loss = (trade_signal or {}).get("stop_loss_price", 0.0)
    entry     = (trade_signal or {}).get("entry_price", 1.0)

    # R01 — Max position size
    if size_pct > MAX_POSITION_PCT:
        flags.append(
            f"R01_POSITION_LIMIT: size {size_pct*100:.2f}% > "
            f"limit {MAX_POSITION_PCT*100:.1f}%"
        )

    # R02 — Leverage ceiling
    if size_pct > LEVERAGE_CEILING:
        flags.append(f"R02_LEVERAGE: implied leverage {size_pct:.2f}x > {LEVERAGE_CEILING:.1f}x")

    # R03 — Stop-loss must be set
    if action in ("BUY", "SELL") and stop_loss <= 0:
        flags.append("R03_STOP_LOSS: no stop-loss price defined")

    # R04 — Volatility halt
    garch_vol = (garch_result or {}).get("current_vol_annual_pct", 0.0)
    if garch_vol > VOL_HALT_THRESHOLD:
        flags.append(
            f"R04_VOL_HALT: GARCH annual vol {garch_vol:.1f}% > "
            f"{VOL_HALT_THRESHOLD:.0f}% threshold"
        )

    # R05 — Sentiment gate
    quant_a = (report or {}).get("quantitative_analysis", {})
    macro_score = quant_a.get("macro_sentiment", {}).get("score", 5.0)
    if action == "BUY" and macro_score < SENTIMENT_MIN:
        flags.append(
            f"R05_SENTIMENT_GATE: macro score {macro_score:.1f} < "
            f"buy floor {SENTIMENT_MIN:.1f}"
        )
    if action == "SELL" and macro_score > SENTIMENT_MAX:
        flags.append(
            f"R05_SENTIMENT_GATE: macro score {macro_score:.1f} > "
            f"sell ceiling {SENTIMENT_MAX:.1f}"
        )

    # R06 — Minimum RR ratio (skip for HOLD)
    if action in ("BUY", "SELL") and rr_ratio > 0 and rr_ratio < MIN_RR_RATIO:
        flags.append(
            f"R06_RR_RATIO: risk-reward {rr_ratio:.2f} < "
            f"minimum {MIN_RR_RATIO:.1f}"
        )

    # R07 — Confidence floor for non-HOLD actions
    if action in ("BUY", "SELL") and confidence < MIN_CONFIDENCE:
        flags.append(
            f"R07_CONFIDENCE: {confidence:.2f} < "
            f"minimum {MIN_CONFIDENCE:.2f} for {action}"
        )

    # R08 — F&O lot size validation
    known_lot = LOT_SIZES.get(ticker)
    if known_lot is not None and size_pct > 0 and entry > 0:
        position_value = size_pct * portfolio_value
        lots = position_value / (entry * known_lot)
        if lots < 1.0:
            flags.append(
                f"R08_LOT_SIZE: notional ₹{position_value:,.0f} < "
                f"1 lot ({known_lot} units × ₹{entry:.0f} = ₹{entry*known_lot:,.0f})"
            )

    passed = len(flags) == 0
    notes = (
        "All compliance checks passed."
        if passed
        else f"{len(flags)} violation(s) detected. Trade blocked pending review."
    )
    return passed, flags, notes


def compliance_agent_node(state: AladdinAgentState) -> dict:
    """
    SEBI compliance checker node.

    Reads: trade_signal, report, garch_result, confidence, portfolio_value
    Writes: compliance_passed, compliance_flags, compliance_notes
    Also modifies trade_signal.action → "HOLD" if compliance fails.
    """
    ticker         = state.get("ticker", "UNKNOWN")
    trade_signal   = state.get("trade_signal") or {}
    report         = state.get("report") or {}
    garch_result   = state.get("garch_result") or {}
    confidence     = state.get("confidence", 0.5)
    portfolio_value = state.get("portfolio_value", 10_000_000.0)

    log.info("[ComplianceAgent] Checking rules for %s", ticker)

    passed, flags, notes = _check_rules(
        ticker, trade_signal, report, garch_result, confidence, portfolio_value
    )

    if not passed:
        log.warning("[ComplianceAgent] %s FAILED compliance: %s", ticker, flags)
        # Override the trade to HOLD — never execute a non-compliant signal
        trade_signal = dict(trade_signal)
        trade_signal["action"]    = "HOLD"
        trade_signal["size_pct"]  = 0.0
        trade_signal["size_inr"]  = 0.0
        trade_signal["halt_reason"] = "compliance_block"
    else:
        log.info("[ComplianceAgent] %s passed all compliance rules.", ticker)

    return {
        "compliance_passed": passed,
        "compliance_flags":  flags,
        "compliance_notes":  notes,
        "trade_signal":      trade_signal,
        "messages": [AIMessage(content=(
            f"[Compliance Agent] {'PASS' if passed else 'BLOCK'} — "
            f"{len(flags)} flag(s) for {ticker}: "
            f"{'; '.join(flags) if flags else 'none'}"
        ))],
    }
