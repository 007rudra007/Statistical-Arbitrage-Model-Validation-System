"""
Phase 4 – Extended AgentState for Aladdin Integration
======================================================
Extends the Synthetix Alpha AgentState with Phase 3 risk engine outputs,
trade signal, compliance flags, and portfolio context.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, List, Optional

# Load AgentState from synthetix-alpha without triggering the agbuilds 'agents' package
_SYNTH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "synthetix-alpha")
if _SYNTH_ROOT not in sys.path:
    sys.path.insert(0, _SYNTH_ROOT)

# Import directly from synthetix-alpha's agents submodule
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "synth_agents_state",
    os.path.join(_SYNTH_ROOT, "agents", "state.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AgentState = _mod.AgentState


class AladdinAgentState(AgentState):
    """
    Extended state that flows through the full Aladdin 6-agent pipeline.

    Inherits from Synthetix Alpha AgentState:
        ticker: str
        stock_data: dict
        macro_analysis: str
        quant_analysis: str
        trade_thesis: str
        confidence: float
        report: dict

    Extended fields:
    """
    # Phase 3 risk engine outputs (computed by trade_agent using risk.var_engine)
    var_result: Optional[Dict[str, Any]]        # PortfolioVaROutput as dict
    stress_results: Optional[List[Dict]]        # list of StressResult dicts
    garch_result: Optional[Dict[str, Any]]      # GARCHVolResult as dict

    # Trade Agent outputs
    trade_signal: Optional[Dict[str, Any]]      # {action, size, entry, stop, target}

    # Compliance Agent outputs
    compliance_passed: bool
    compliance_flags: List[str]                 # list of SEBI rule violations
    compliance_notes: str

    # Portfolio context (for multi-ticker orchestration)
    portfolio_id: Optional[str]
    portfolio_value: float

    # Ollama availability flag (set during graph init)
    llm_available: bool
