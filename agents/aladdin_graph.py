"""
Phase 4 – Step 13: Aladdin LangGraph Pipeline
================================================
6-agent state graph:
  START → fetch_data → macro → quant → trade → compliance → consensus → END
"""

from __future__ import annotations

import importlib.util as _ilu
import logging
import os
import sys
from typing import List

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

log = logging.getLogger(__name__)

# ── Locate roots ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SYNTH_ROOT   = os.path.join(_PROJECT_ROOT, "synthetix-alpha")

# Ensure synthetix-alpha is importable for config, models, data, core
if _SYNTH_ROOT not in sys.path:
    sys.path.insert(0, _SYNTH_ROOT)


def _load(name: str, path: str):
    """Load a Python module directly from its file path."""
    spec = _ilu.spec_from_file_location(name, path)
    mod  = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Load synthetix-alpha agents directly (avoids package name conflict) ───────
_sa = os.path.join(_SYNTH_ROOT, "agents")
_synth_state     = _load("synth_agents.state",     os.path.join(_sa, "state.py"))
_synth_macro     = _load("synth_agents.macro",     os.path.join(_sa, "macro_agent.py"))
_synth_quant     = _load("synth_agents.quant",     os.path.join(_sa, "quant_agent.py"))
_synth_consensus = _load("synth_agents.consensus", os.path.join(_sa, "consensus_agent.py"))

macro_agent_node     = _synth_macro.macro_agent_node
quant_agent_node     = _synth_quant.quant_agent_node
consensus_agent_node = _synth_consensus.consensus_agent_node

# ── Load Aladdin-specific agents from project root ─────────────────────────────
_aa = os.path.join(_PROJECT_ROOT, "agents")
_aladdin_state     = _load("aladdin_agents.state",       os.path.join(_aa, "aladdin_state.py"))
_aladdin_trade     = _load("aladdin_agents.trade",       os.path.join(_aa, "trade_agent.py"))
_aladdin_compliance = _load("aladdin_agents.compliance", os.path.join(_aa, "compliance_agent.py"))

AladdinAgentState    = _aladdin_state.AladdinAgentState
trade_agent_node     = _aladdin_trade.trade_agent_node
compliance_agent_node = _aladdin_compliance.compliance_agent_node


# ── Synthetix fetch_data node (re-used verbatim) ──────────────────────────────
def fetch_data_node(state: AladdinAgentState) -> dict:
    """Fetch stock data from yfinance and populate state.stock_data."""
    from data.fetcher import fetch_stock_data

    ticker = state.get("ticker", "RELIANCE.NS")
    try:
        df = fetch_stock_data(ticker, period="1y")
        stock_data = {
            "close":        df["Close"].tolist(),
            "returns":      df["Returns"].tolist(),
            "log_returns":  df["LogReturns"].tolist(),
            "dates":        [str(d) for d in df.index.tolist()],
        }
        return {
            "stock_data": stock_data,
            "messages": [HumanMessage(
                content=f"[FetchData] {ticker}: {len(df)} days, "
                        f"latest close ₹{df['Close'].iloc[-1]:.2f}"
            )],
        }
    except Exception as exc:
        return {
            "stock_data": {},
            "messages": [HumanMessage(content=f"[FetchData] Failed for {ticker}: {exc}")],
        }


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_aladdin_graph() -> StateGraph:
    """
    Build the 6-agent Aladdin pipeline as a compiled LangGraph StateGraph.

    Pipeline:
        fetch_data → macro → quant → trade → compliance → consensus
    """
    builder = StateGraph(AladdinAgentState)

    builder.add_node("fetch_data",  fetch_data_node)
    builder.add_node("macro",       macro_agent_node)
    builder.add_node("quant",       quant_agent_node)
    builder.add_node("trade",       trade_agent_node)
    builder.add_node("compliance",  compliance_agent_node)
    builder.add_node("consensus",   consensus_agent_node)

    builder.add_edge(START,        "fetch_data")
    builder.add_edge("fetch_data", "macro")
    builder.add_edge("macro",      "quant")
    builder.add_edge("quant",      "trade")
    builder.add_edge("trade",      "compliance")
    builder.add_edge("compliance", "consensus")
    builder.add_edge("consensus",  END)

    return builder.compile()


# Pre-built singleton
aladdin_app = build_aladdin_graph()


# ── Convenience runner ────────────────────────────────────────────────────────
def run_pipeline(
    ticker: str,
    portfolio_value: float = 10_000_000.0,
    portfolio_id: str | None = None,
) -> dict:
    """
    Run the full Aladdin agent pipeline for a single ticker.

    Returns the final state as a dict.
    """
    log.info("[Aladdin] Running pipeline for %s (₹%.0f)", ticker, portfolio_value)
    initial_state = {
        "ticker":          ticker,
        "portfolio_value": portfolio_value,
        "portfolio_id":    portfolio_id,
        "llm_available":   True,
        "messages":        [],
        # Initialise optional fields
        "compliance_passed": False,
        "compliance_flags":  [],
        "compliance_notes":  "",
    }
    result = aladdin_app.invoke(initial_state)
    log.info(
        "[Aladdin] Pipeline complete: action=%s, confidence=%.2f, compliance=%s",
        (result.get("trade_signal") or {}).get("action", "?"),
        result.get("confidence", 0.0),
        result.get("compliance_passed", False),
    )
    return result
