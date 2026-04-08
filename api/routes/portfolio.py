"""
Phase 5 – Portfolio & Trading API Routes
=========================================
FastAPI endpoints for:

  Portfolio Optimization
    POST /portfolio/optimize            Full CVXPY optimization
    POST /portfolio/optimize/quick      Equal-weight + SEBI compliance check

  Order Execution
    POST /portfolio/orders/place        Place a single order (paper/live)
    POST /portfolio/orders/batch        Place a batch of orders
    GET  /portfolio/orders/{id}         Get order status
    DELETE /portfolio/orders/{id}       Cancel order
    GET  /portfolio/positions           Current positions

  Backtesting
    POST /portfolio/backtest/pairs      Pairs trading backtest (yfinance)
    POST /portfolio/backtest/momentum   Momentum strategy backtest
    POST /portfolio/backtest/agent      Replay agent signals backtest

  Compliance
    POST /portfolio/compliance          Portfolio-level SEBI check (10 rules)
    GET  /portfolio/compliance/rules    List all SEBI rules
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["portfolio"])


# ── Request models ────────────────────────────────────────────────────────────
class AssetInput(BaseModel):
    ticker:         str
    sector:         str = "Unknown"
    beta:           float = 1.0
    current_weight: float = 0.0


class OptimizeRequest(BaseModel):
    tickers:          Optional[List[str]] = None   # None = NSE_UNIVERSE
    portfolio_value:  float = Field(default=10_000_000.0, gt=0)
    objective:        str   = Field(default="min_variance",
                                    pattern="^(min_variance|max_sharpe|risk_parity)$")
    assets:           Optional[List[AssetInput]] = None
    lookback_days:    int   = Field(default=252, ge=30, le=1260)
    max_weight:       float = Field(default=0.15, ge=0.01, le=0.50)
    max_sector_weight: float = Field(default=0.30, ge=0.10, le=0.50)
    max_beta:         float = Field(default=0.95, ge=0.5, le=1.5)


class PlaceOrderRequest(BaseModel):
    ticker:       str
    side:         str = Field(pattern="^(BUY|SELL)$")
    quantity:     int = Field(gt=0)
    order_type:   str = Field(default="MARKET",
                               pattern="^(MARKET|LIMIT|SL|SL-M)$")
    product:      str = Field(default="CNC", pattern="^(CNC|MIS|NRML)$")
    price:        float = 0.0
    trigger_price: float = 0.0
    exchange:     str = Field(default="NSE", pattern="^(NSE|BSE|NFO)$")


class BacktestRequest(BaseModel):
    tickers:        List[str] = Field(min_length=1, max_length=20)
    start_date:     str = "2022-01-01"
    end_date:       str = "2024-12-31"
    initial_capital: float = Field(default=10_000_000.0, gt=0)
    position_size_pct: float = Field(default=5.0, ge=0.5, le=25.0)
    slippage_ticks: float = Field(default=1.0, ge=0.1, le=5.0)


class PairsBacktestRequest(BaseModel):
    ticker_a: str = "HDFCBANK.NS"
    ticker_b: str = "ICICIBANK.NS"
    start_date: str = "2022-01-01"
    end_date:   str = "2024-12-31"
    initial_capital: float = Field(default=10_000_000.0, gt=0)
    entry_z:   float = Field(default=2.0,  ge=0.5, le=5.0)
    exit_z:    float = Field(default=0.5,  ge=0.0, le=2.0)
    lookback:  int   = Field(default=60,   ge=20,  le=252)


class ComplianceCheckRequest(BaseModel):
    weights:    Dict[str, float]          # ticker → weight (0-1)
    sectors:    Dict[str, str] = {}       # ticker → sector
    betas:      Dict[str, float] = {}     # ticker → beta
    fno_tickers: List[str] = []
    cash_pct:   float = Field(default=5.0, ge=0.0, le=100.0)
    leverage:   float = Field(default=1.0, ge=0.0, le=5.0)
    portfolio_value: float = Field(default=10_000_000.0, gt=0)
    current_drawdown_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    daily_turnover_pct:   float = Field(default=0.0, ge=0.0, le=100.0)
    blocked_tickers: List[str] = []
    max_single_stock_pct: float = Field(default=15.0, ge=1.0, le=50.0)
    max_sector_pct:       float = Field(default=35.0, ge=10.0, le=100.0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fetch_price_df(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices from yfinance."""
    try:
        import yfinance as yf
        df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                         progress=False, threads=True)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])
        return df.dropna(how="all")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"yfinance fetch failed: {exc}")


def _compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().dropna()


# ── Portfolio optimization ────────────────────────────────────────────────────
@router.post("/optimize", summary="CVXPY mean-variance portfolio optimization")
async def optimize_portfolio(req: OptimizeRequest) -> Dict[str, Any]:
    """
    Optimize portfolio weights using CVXPY.

    Objectives:
    - **min_variance**: Global minimum variance (default)
    - **max_sharpe**: Maximum Sharpe ratio
    - **risk_parity**: Equal risk contribution

    Constraints: single-stock cap, sector cap, beta ceiling, long-only.
    Falls back to NumPy minimum-variance if CVXPY is not installed.
    """
    from portfolio.optimizer import (
        PortfolioOptimizer, OptimizationInput, Asset, NSE_UNIVERSE
    )

    # Build asset list
    if req.assets:
        assets = [Asset(a.ticker, a.sector, a.beta, a.current_weight) for a in req.assets]
    elif req.tickers:
        assets = [Asset(t, "Unknown") for t in req.tickers]
    else:
        assets = NSE_UNIVERSE

    tickers = [a.ticker for a in assets]

    # Fetch historical prices
    price_df = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_price_df(
            tickers,
            str((datetime.now() - pd.Timedelta(days=req.lookback_days)).date()),
            str(datetime.now().date()),
        )
    )

    # Align tickers to what yfinance returned
    available = [t for t in tickers if t in price_df.columns]
    if len(available) < 2:
        raise HTTPException(status_code=422,
                            detail=f"Need ≥2 tickers with data. Got: {available}")

    assets    = [a for a in assets if a.ticker in available]
    price_df  = price_df[available].dropna()
    ret_matrix = _compute_returns(price_df).values

    inp = OptimizationInput(
        assets=assets,
        returns_matrix=ret_matrix,
        portfolio_value=req.portfolio_value,
        objective=req.objective,
        max_weight=req.max_weight,
        max_sector_weight=req.max_sector_weight,
        max_portfolio_beta=req.max_beta,
    )

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: PortfolioOptimizer().optimize(inp)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "objective":    req.objective,
        "solver_status": result.solver_status,
        "n_assets":     result.n_assets,
        "effective_n":  result.effective_n,
        "weights":      result.weights,
        "metrics": {
            "expected_annual_return": f"{result.expected_annual_return*100:.2f}%",
            "expected_annual_vol":    f"{result.expected_annual_vol*100:.2f}%",
            "sharpe_ratio":           result.sharpe_ratio,
            "portfolio_beta":         result.portfolio_beta,
        },
        "sector_weights":       result.sector_weights,
        "rebalancing_trades":   result.rebalancing_trades,
        "transaction_cost_inr": result.transaction_cost_inr,
        "computed_at":          datetime.now(timezone.utc).isoformat(),
    }


# ── Order execution ───────────────────────────────────────────────────────────
@router.post("/orders/place", summary="Place a single order (paper/Kite/FIX)")
async def place_order(req: PlaceOrderRequest) -> Dict[str, Any]:
    """
    Place an order via configured router (KITE_PAPER=1 by default → paper trading).
    Set env ORDER_ROUTER=kite for live Zerodha Kite execution.
    """
    from portfolio.order_router import (
        get_order_router, Order, OrderSide, OrderType, ProductType, Exchange
    )
    router_inst = get_order_router()
    order = Order(
        ticker      = req.ticker,
        side        = OrderSide(req.side),
        quantity    = req.quantity,
        order_type  = OrderType(req.order_type),
        product     = ProductType(req.product),
        exchange    = Exchange(req.exchange),
        price       = req.price,
        trigger_price = req.trigger_price,
    )
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: router_inst.place_order(order)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "order_id":   resp.order_id,
        "status":     resp.status.value,
        "message":    resp.message,
        "fill_price": resp.fill_price,
        "filled_qty": resp.filled_qty,
        "router":     type(router_inst).__name__,
        "placed_at":  datetime.now(timezone.utc).isoformat(),
    }


@router.post("/orders/batch", summary="Place a batch of orders")
async def place_batch_orders(orders: List[PlaceOrderRequest]) -> Dict[str, Any]:
    """Place up to 50 orders concurrently (paper mode)."""
    if len(orders) > 50:
        raise HTTPException(status_code=422, detail="Max 50 orders per batch")

    from portfolio.order_router import (
        get_order_router, Order, OrderSide, OrderType, ProductType, Exchange
    )
    router_inst = get_order_router()

    async def _place(req: PlaceOrderRequest):
        order = Order(
            ticker=req.ticker, side=OrderSide(req.side),
            quantity=req.quantity, order_type=OrderType(req.order_type),
            product=ProductType(req.product), exchange=Exchange(req.exchange),
            price=req.price, trigger_price=req.trigger_price,
        )
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: router_inst.place_order(order)
            )
            return {"order_id": resp.order_id, "status": resp.status.value,
                    "message": resp.message, "fill_price": resp.fill_price}
        except Exception as exc:
            return {"order_id": "error", "status": "REJECTED", "message": str(exc)}

    results = await asyncio.gather(*[_place(o) for o in orders])
    fills   = [r for r in results if r["status"] == "COMPLETE"]
    return {
        "total_orders": len(orders),
        "filled":   len(fills),
        "rejected": len(orders) - len(fills),
        "results":  list(results),
        "placed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/positions", summary="Get current positions")
async def get_positions() -> Dict[str, Any]:
    from portfolio.order_router import get_order_router
    router_inst = get_order_router()
    positions = await asyncio.get_event_loop().run_in_executor(
        None, router_inst.get_positions
    )
    return {
        "positions": [
            {"ticker": p.ticker, "quantity": p.quantity,
             "avg_price": p.avg_price, "product": p.product.value,
             "pnl": p.pnl, "value": p.value}
            for p in positions
        ],
        "n_positions": len(positions),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Backtesting ───────────────────────────────────────────────────────────────
@router.post("/backtest/pairs", summary="Pairs trading backtest (yfinance)")
async def backtest_pairs(req: PairsBacktestRequest) -> Dict[str, Any]:
    """
    Runs the event-driven pairs trading backtest using the existing
    BacktestEngine from backtester.engine (Phase 2 module).
    Entry: |Z| > entry_z  Exit: |Z| < exit_z
    """
    from backtester.multi_strategy import (
        MultiStrategyBacktester, MomentumStrategy
    )

    tickers  = [req.ticker_a, req.ticker_b]
    price_df = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_price_df(tickers, req.start_date, req.end_date)
    )
    if price_df.shape[1] < 2:
        raise HTTPException(status_code=422, detail="Could not fetch both tickers")

    # Use simple Z-score pairs strategy via the new backtester
    from backtester.multi_strategy import (
        MultiStrategyBacktester, Portfolio, Signal,
    )

    class SimplePairsStrategy:
        def __init__(self, ta, tb, lookback, entry_z, exit_z):
            self.ta = ta; self.tb = tb; self.lb = lookback
            self.ez = entry_z; self.xz = exit_z; self._pos = 0

        def on_bar(self, bar, hist):
            if bar.ticker != self.ta:
                return []
            pa = hist.get(self.ta); pb = hist.get(self.tb)
            if pa is None or pb is None or len(pa) < self.lb:
                return []
            spread = pa.iloc[-self.lb:].values - pb.iloc[-self.lb:].values
            z = (spread[-1] - spread.mean()) / (spread.std() + 1e-10)
            sigs = []
            if z < -self.ez and self._pos != 1:
                sigs = [Signal(bar.timestamp, self.ta, "BUY",  min(abs(z)/4, 1), "pairs"),
                        Signal(bar.timestamp, self.tb, "SELL", min(abs(z)/4, 1), "pairs")]
                self._pos = 1
            elif z > self.ez and self._pos != -1:
                sigs = [Signal(bar.timestamp, self.ta, "SELL", min(abs(z)/4, 1), "pairs"),
                        Signal(bar.timestamp, self.tb, "BUY",  min(abs(z)/4, 1), "pairs")]
                self._pos = -1
            elif abs(z) < self.xz and self._pos != 0:
                sigs = [Signal(bar.timestamp, self.ta, "EXIT", 1.0, "pairs"),
                        Signal(bar.timestamp, self.tb, "EXIT", 1.0, "pairs")]
                self._pos = 0
            for s in sigs:
                s.meta = {"z_score": round(float(z), 3)}
            return sigs

    strat = SimplePairsStrategy(
        req.ticker_a, req.ticker_b, req.lookback, req.entry_z, req.exit_z
    )
    bt = MultiStrategyBacktester(price_df, strat, req.initial_capital,
                                  size_pct=0.10, slippage_ticks=1.0)

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, bt.run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "strategy":    "pairs_trading",
        "ticker_a":    req.ticker_a,
        "ticker_b":    req.ticker_b,
        "date_range":  f"{req.start_date} → {req.end_date}",
        "metrics":     result["metrics"],
        "n_trades":    result["metrics"].get("n_trades", 0),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/backtest/momentum", summary="12-1 month momentum strategy backtest")
async def backtest_momentum(req: BacktestRequest) -> Dict[str, Any]:
    """Cross-sectional 12-1 month momentum backtest on NSE tickers."""
    from backtester.multi_strategy import MultiStrategyBacktester, MomentumStrategy

    price_df = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_price_df(req.tickers, req.start_date, req.end_date)
    )
    tickers  = [t for t in req.tickers if t in price_df.columns]
    strat    = MomentumStrategy(tickers, formation=252, skip=21, top_pct=0.3)
    bt       = MultiStrategyBacktester(
        price_df[tickers], strat, req.initial_capital,
        size_pct=req.position_size_pct / 100, slippage_ticks=req.slippage_ticks
    )
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, bt.run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "strategy":    "momentum_12_1",
        "tickers":     tickers,
        "date_range":  f"{req.start_date} → {req.end_date}",
        "metrics":     result["metrics"],
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/backtest/agent", summary="Replay Phase 4 agent signals backtest")
async def backtest_agent(
    req: BacktestRequest,
    signals: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Backtest the Phase 4 AI multi-agent layer's alpha by replaying its signals.
    POST body should include `signals` list:
      [{"timestamp": "2024-01-02", "ticker": "RELIANCE.NS", "action": "BUY", "confidence": 0.8}]
    """
    from backtester.multi_strategy import MultiStrategyBacktester, AgentSignalStrategy

    if not signals:  # catches both None and empty list
        raise HTTPException(status_code=422,
                            detail="Provide 'signals' list in request body")

    sig_df   = pd.DataFrame(signals)
    price_df = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fetch_price_df(req.tickers, req.start_date, req.end_date)
    )
    tickers  = [t for t in req.tickers if t in price_df.columns]
    strat    = AgentSignalStrategy(sig_df)
    bt       = MultiStrategyBacktester(
        price_df[tickers], strat, req.initial_capital,
        size_pct=req.position_size_pct / 100, slippage_ticks=req.slippage_ticks
    )
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, bt.run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "strategy":    "agent_signal_replay",
        "tickers":     tickers,
        "date_range":  f"{req.start_date} → {req.end_date}",
        "n_signals":   len(signals),
        "metrics":     result["metrics"],
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Portfolio compliance ──────────────────────────────────────────────────────
@router.post("/compliance", summary="Portfolio-level SEBI compliance check (10 rules)")
def check_portfolio_compliance(req: ComplianceCheckRequest) -> Dict[str, Any]:
    """
    Run 10 SEBI portfolio-level compliance checks:
    P01 Single-stock cap, P02 Sector cap, P03 Beta, P04 Cash floor,
    P05 Leverage, P06 Diversification, P07 Turnover, P08 Drawdown halt,
    P09 F&O cap, P10 Blocked/UPSI tickers.
    """
    from portfolio.sebi_compliance import (
        SEBIComplianceEngine, ComplianceConfig, PortfolioSnapshot
    )

    cfg = ComplianceConfig(
        max_single_stock_pct=req.max_single_stock_pct,
        max_sector_pct=req.max_sector_pct,
        blocked_tickers=set(req.blocked_tickers),
    )
    engine = SEBIComplianceEngine(cfg)
    snap   = PortfolioSnapshot(
        weights          = req.weights,
        sectors          = req.sectors,
        betas            = req.betas,
        fno_tickers      = set(req.fno_tickers),
        cash_pct         = req.cash_pct,
        leverage         = req.leverage,
        portfolio_value  = req.portfolio_value,
        current_drawdown_pct = req.current_drawdown_pct,
        daily_turnover_pct   = req.daily_turnover_pct,
    )
    report = engine.check(snap)
    fixes  = engine.suggest_fixes(report, snap)

    return {
        "passed":       report.passed,
        "halt_trading": report.halt_trading,
        "summary":      report.summary,
        "checks_run":   report.checks_run,
        "errors": [
            {"rule": v.rule_id, "message": v.message,
             "value": v.value, "threshold": v.threshold}
            for v in report.errors
        ],
        "warnings": [
            {"rule": v.rule_id, "message": v.message,
             "value": v.value, "threshold": v.threshold}
            for v in report.warnings
        ],
        "suggested_fixes": fixes,
        "portfolio_value":  req.portfolio_value,
        "checked_at":       report.checked_at,
    }


@router.get("/compliance/rules", summary="List all SEBI compliance rules")
def list_compliance_rules() -> Dict[str, Any]:
    return {
        "rules": [
            {"id": "P01", "name": "Single-Stock Weight Cap",   "threshold": "15%",  "severity": "ERROR"},
            {"id": "P02", "name": "Sector Concentration Cap",  "threshold": "35%",  "severity": "ERROR"},
            {"id": "P03", "name": "Portfolio Beta Ceiling",    "threshold": "1.00", "severity": "WARNING"},
            {"id": "P04", "name": "Cash Floor",                "threshold": "2%",   "severity": "ERROR"},
            {"id": "P05", "name": "Leverage Ceiling",          "threshold": "1.0×", "severity": "ERROR"},
            {"id": "P06", "name": "Min Diversification (Eff N)","threshold": "4",   "severity": "WARNING"},
            {"id": "P07", "name": "Daily Turnover Cap",        "threshold": "5%",   "severity": "WARNING"},
            {"id": "P08", "name": "Drawdown Circuit Breaker",  "threshold": "15%",  "severity": "ERROR + HALT"},
            {"id": "P09", "name": "F&O Concentration Cap",     "threshold": "20%",  "severity": "WARNING"},
            {"id": "P10", "name": "Blocked/UPSI Tickers",      "threshold": "0%",   "severity": "ERROR + HALT"},
        ],
        "total_rules": 10,
        "framework":   "SEBI LODR + SEBI/HO/MRD/DRMNP/CIR/P/2020/234",
        "updated_at":  "2025-01-01",
    }
