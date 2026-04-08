"""
Phase 4 – Agent Layer Package
================================
Integrates the Synthetix Alpha multi-agent system into the Aladdin stack.

Architecture (LangGraph StateGraph):
  START
    → fetch_data        (yfinance price fetch)
    → macro_agent       (RAG + Ollama Llama3 macro sentiment)
    → quant_agent       (GJR-GARCH + MC VaR + Z-score → risk score)
    → trade_agent       (signal generation + position sizing)
    → compliance_agent  (SEBI rules + position limits)
    → consensus_agent   (BUY/SELL/HOLD + structured report)
  END

Sourced from:  github.com/007rudra007/synthetix-alpha  (v2.4.0-stable)
Extended with: trade_agent, compliance_agent, Phase 3 risk integration
"""

# Re-export the compiled LangGraph app from synthetix-alpha
import sys
import os

# Make synthetix-alpha importable as a subpackage
_SYNTH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "synthetix-alpha")
if _SYNTH_ROOT not in sys.path:
    sys.path.insert(0, _SYNTH_ROOT)

from agents.aladdin_graph import build_aladdin_graph, aladdin_app
from agents.trade_agent import trade_agent_node
from agents.compliance_agent import compliance_agent_node
from agents.aladdin_state import AladdinAgentState

__all__ = [
    "build_aladdin_graph",
    "aladdin_app",
    "trade_agent_node",
    "compliance_agent_node",
    "AladdinAgentState",
]
