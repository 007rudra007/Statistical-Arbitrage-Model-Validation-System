"""
Phase 4 – Agent API Router
===========================
FastAPI endpoints for the AI multi-agent layer (Phase 4).

  POST /agent/run          – Run full 6-agent pipeline for a ticker
  POST /agent/run/batch    – Run pipeline for multiple tickers (async)
  GET  /agent/status       – Ollama + LangGraph health check
  GET  /agent/graph        – Return pipeline graph topology
  POST /agent/macro        – Run only macro agent (RAG analysis)
  POST /agent/quant        – Run only quant agent (risk models)
  POST /agent/compliance   – Check SEBI compliance for a signal
  GET  /agent/scenarios    – List loaded macro news documents
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

# Project root first — so c:\agbuilds\agents\ takes priority over synthetix-alpha\agents\
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Make synthetix-alpha importable (second priority — appended)
_SYNTH_ROOT = os.path.join(_PROJECT_ROOT, "synthetix-alpha")
if _SYNTH_ROOT not in sys.path:
    sys.path.append(_SYNTH_ROOT)


# ── Request / Response models ─────────────────────────────────────────────────
class AgentRunRequest(BaseModel):
    ticker: str = Field(default="RELIANCE.NS", description="NSE ticker (e.g. RELIANCE.NS)")
    portfolio_value: float = Field(default=10_000_000.0, gt=0)
    portfolio_id: Optional[str] = None


class BatchRunRequest(BaseModel):
    tickers: List[str] = Field(min_length=1, max_length=20)
    portfolio_value: float = Field(default=10_000_000.0, gt=0)


class MacroRequest(BaseModel):
    ticker: str = "RELIANCE.NS"
    query_override: Optional[str] = None


class QuantRequest(BaseModel):
    ticker: str = "RELIANCE.NS"
    period: str = "1y"


class ComplianceCheckRequest(BaseModel):
    ticker: str = "RELIANCE.NS"
    action: str = Field(default="BUY", pattern="^(BUY|SELL|HOLD)$")
    size_pct: float = Field(default=5.0, ge=0, le=100)
    entry_price: float = Field(gt=0)
    stop_loss_price: float = Field(ge=0)
    target_price: float = Field(ge=0)
    risk_reward_ratio: float = Field(default=0.0, ge=0)
    garch_vol_annual_pct: float = Field(default=15.0, ge=0)
    confidence: float = Field(default=0.7, ge=0, le=1)
    portfolio_value: float = Field(default=10_000_000.0, gt=0)
    macro_score: float = Field(default=5.0, ge=0, le=10)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_ollama() -> Dict[str, Any]:
    """Check if Ollama is reachable."""
    import urllib.request
    from config import settings
    try:
        url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        resp = urllib.request.urlopen(url, timeout=3)
        import json
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return {"status": "ok", "models": models, "url": settings.ollama_base_url}
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc),
                "url": settings.ollama_base_url,
                "hint": "Run: ollama serve && ollama pull llama3"}


def _format_result(result: dict) -> Dict[str, Any]:
    """Extract the key fields from a pipeline result for API response."""
    report = result.get("report") or {}
    es = report.get("executive_summary", {})
    qa = report.get("quantitative_analysis", {})
    trade = result.get("trade_signal") or {}

    return {
        "ticker":  result.get("ticker"),
        "action":  trade.get("action", es.get("action", "HOLD")),
        "confidence": result.get("confidence", 0.0),
        "position_sizing": es.get("position_sizing", "CONSERVATIVE"),
        "thesis": es.get("thesis_statement", ""),
        "trade_signal": trade,
        "compliance": {
            "passed": result.get("compliance_passed", False),
            "flags":  result.get("compliance_flags", []),
            "notes":  result.get("compliance_notes", ""),
        },
        "risk": {
            "garch_vol_annual": (result.get("garch_result") or {}).get("current_vol_annual_pct"),
            "regime":           (result.get("garch_result") or {}).get("current_regime"),
            "var_99_pct":       (result.get("var_result") or {}).get("historical_var_99"),
        },
        "macro_analysis":  (result.get("macro_analysis") or "")[:500] + "…",
        "quant_analysis":  (result.get("quant_analysis") or "")[:500] + "…",
        "report_metadata": report.get("report_metadata", {}),
        "quantitative_analysis": qa,
        "investment_pillars":    report.get("investment_pillars", {}),
        "risk_matrix":           report.get("risk_matrix", {}),
        "message_count": len(result.get("messages", [])),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/status", summary="Ollama + LangGraph health check")
def agent_status() -> Dict[str, Any]:
    """
    Check availability of:
    - Ollama LLM service + loaded models
    - LangGraph pipeline (compile check)
    - sentence-transformers (RAG embedding)
    - Phase 3 risk engine modules
    """
    status: Dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat()
    }

    status["ollama"] = _check_ollama()

    try:
        from agents.aladdin_graph import aladdin_app
        status["langgraph"] = {"status": "ok", "nodes": list(aladdin_app.nodes.keys())}
    except Exception as exc:
        status["langgraph"] = {"status": "error", "detail": str(exc)}

    try:
        from sentence_transformers import SentenceTransformer
        status["sentence_transformers"] = "available"
    except ImportError:
        status["sentence_transformers"] = "unavailable"

    try:
        from risk.var_engine import PortfolioVaREngine
        from risk.garch_vol import GARCHVolEngine
        status["risk_engine"] = "available"
    except ImportError:
        status["risk_engine"] = "unavailable"

    all_ok = (
        status["ollama"]["status"] == "ok"
        and status.get("langgraph", {}).get("status") == "ok"
    )
    status["overall"] = "ready" if all_ok else "degraded"
    return status


@router.get("/graph", summary="Pipeline graph topology")
def get_graph_topology() -> Dict[str, Any]:
    """Returns the directed graph structure of the 6-agent pipeline."""
    return {
        "pipeline": "Aladdin Multi-Agent (Phase 4)",
        "nodes": [
            {"id": "fetch_data",  "description": "yfinance price fetch (1y OHLCV)"},
            {"id": "macro",       "description": "RAG + Ollama macro sentiment analysis"},
            {"id": "quant",       "description": "GJR-GARCH + MC VaR + Z-score risk scoring"},
            {"id": "trade",       "description": "Kelly-sized trade signal with vol halt"},
            {"id": "compliance",  "description": "SEBI rules checker (8 rules)"},
            {"id": "consensus",   "description": "BUY/SELL/HOLD thesis + structured report"},
        ],
        "edges": [
            "START → fetch_data",
            "fetch_data → macro",
            "macro → quant",
            "quant → trade",
            "trade → compliance",
            "compliance → consensus",
            "consensus → END",
        ],
        "source_repo": "https://github.com/007rudra007/synthetix-alpha",
        "version": "2.4.0-stable + Aladdin Phase 4 extensions",
    }


@router.post("/run", summary="Run full 6-agent pipeline for a ticker")
async def run_agent(req: AgentRunRequest) -> Dict[str, Any]:
    """
    Execute the complete Aladdin AI pipeline:
    fetch_data → macro (RAG) → quant (GARCH/VaR) → trade → compliance → consensus

    **Requires Ollama** running locally (`ollama serve && ollama pull llama3`).
    If Ollama is unavailable, agents return graceful fallback values.

    Typical latency: 30–120s (depends on LLM inference time).
    """
    try:
        from agents.aladdin_graph import run_pipeline
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail=f"Agent graph not available: {exc}")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_pipeline(req.ticker, req.portfolio_value, req.portfolio_id)
        )
    except Exception as exc:
        log.exception("Agent pipeline failed for %s", req.ticker)
        raise HTTPException(status_code=500, detail=str(exc))

    return _format_result(result)


@router.post("/run/batch", summary="Run pipeline for multiple tickers concurrently")
async def run_batch(req: BatchRunRequest) -> Dict[str, Any]:
    """
    Run the full Aladdin pipeline concurrently for up to 20 tickers.
    Returns a portfolio-level summary + per-ticker results.
    """
    try:
        from agents.aladdin_graph import run_pipeline
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    async def _run_one(ticker: str) -> Dict[str, Any]:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_pipeline(ticker, req.portfolio_value)
            )
            return {"ticker": ticker, "status": "ok", **_format_result(result)}
        except Exception as exc:
            return {"ticker": ticker, "status": "error", "detail": str(exc)}

    tasks = [_run_one(t) for t in req.tickers]
    results = await asyncio.gather(*tasks)

    # Portfolio-level summary
    buys  = [r for r in results if r.get("action") == "BUY"  and r.get("compliance", {}).get("passed")]
    sells = [r for r in results if r.get("action") == "SELL" and r.get("compliance", {}).get("passed")]
    holds = [r for r in results if r.get("action") == "HOLD"]

    return {
        "portfolio_value": req.portfolio_value,
        "tickers_analysed": len(req.tickers),
        "summary": {
            "buy_signals":    len(buys),
            "sell_signals":   len(sells),
            "hold_signals":   len(holds),
            "compliance_blocks": sum(1 for r in results if not r.get("compliance", {}).get("passed")),
        },
        "results":  results,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/macro", summary="Run only the Macro Agent (RAG analysis)")
async def run_macro(req: MacroRequest) -> Dict[str, Any]:
    """
    Run just the Macro Agent — RAG retrieval + Ollama LLM macro sentiment.
    Useful for ad-hoc market research without running the full pipeline.
    """
    try:
        from agents.macro_agent import macro_agent_node
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    state = {"ticker": req.ticker, "messages": [], "stock_data": {}}
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: macro_agent_node(state)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": req.ticker,
        "macro_analysis": result.get("macro_analysis", ""),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/quant", summary="Run only the Quant Agent (GJR-GARCH + MC VaR)")
async def run_quant(req: QuantRequest) -> Dict[str, Any]:
    """
    Fetch price data and run GJR-GARCH, Monte Carlo VaR, and Z-score.
    Returns a structured quantitative risk report without LLM involvement.
    """
    try:
        from agents.aladdin_graph import fetch_data_node
        from agents.quant_agent import quant_agent_node
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    state: Dict[str, Any] = {
        "ticker": req.ticker,
        "messages": [],
        "stock_data": {},
    }
    try:
        state.update(await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_data_node(state)
        ))
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: quant_agent_node(state)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": req.ticker,
        "quant_analysis": result.get("quant_analysis", ""),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/compliance", summary="Check SEBI compliance for a trade signal")
def check_compliance(req: ComplianceCheckRequest) -> Dict[str, Any]:
    """
    Standalone SEBI compliance checker — no LLM required.

    Submit a hypothetical trade and get pass/fail + all rule violations.
    """
    try:
        from agents.compliance_agent import _check_rules
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    trade_signal = {
        "action":            req.action,
        "size_pct":          req.size_pct,
        "entry_price":       req.entry_price,
        "stop_loss_price":   req.stop_loss_price,
        "target_price":      req.target_price,
        "risk_reward_ratio": req.risk_reward_ratio,
    }
    garch_result = {"current_vol_annual_pct": req.garch_vol_annual_pct}
    report = {
        "quantitative_analysis": {
            "macro_sentiment": {"score": req.macro_score}
        }
    }

    passed, flags, notes = _check_rules(
        req.ticker, trade_signal, report, garch_result,
        req.confidence, req.portfolio_value
    )

    return {
        "ticker":   req.ticker,
        "action":   req.action,
        "passed":   passed,
        "flags":    flags,
        "notes":    notes,
        "rules_checked": 8,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/scenarios", summary="List macro news documents in RAG store")
def list_macro_scenarios(
    k: int = Query(default=5, ge=1, le=20, description="Top-k docs to retrieve"),
    query: str = Query(default="India market outlook macro risk"),
) -> Dict[str, Any]:
    """
    Retrieve the top-k macro news documents from the RAG vector store.
    Shows what context is available to the Macro Agent.
    """
    try:
        from data.fetcher import get_vectorstore
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}")

    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query": query,
        "k": k,
        "documents": [
            {
                "source":   d.metadata.get("source", "Unknown"),
                "date":     d.metadata.get("date", ""),
                "category": d.metadata.get("category", ""),
                "excerpt":  d.page_content[:200] + ("…" if len(d.page_content) > 200 else ""),
            }
            for d in docs
        ],
        "total_docs": len(docs),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
