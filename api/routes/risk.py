"""
Phase 3 – Risk API Router
==========================
FastAPI router exposing all Phase 3 risk engine endpoints:

  POST /risk/var          – Portfolio VaR (hist + parametric + MC)
  POST /risk/stress       – Stress test one or all scenarios
  GET  /risk/stress/scenarios – List available scenarios
  POST /risk/vol          – GJR-GARCH volatility forecast
  POST /risk/options/price   – BSM / QuantLib option pricing
  POST /risk/options/iv      – Implied volatility solver
  GET  /risk/options/chain   – NIFTY option chain
  POST /risk/xva          – XVA for IRS / FX Forward portfolio
  GET  /risk/health       – Risk engine status
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["risk"])


# ── Request Models ─────────────────────────────────────────────────────────────
class PositionIn(BaseModel):
    ticker: str
    weight: float = Field(ge=0.0, le=1.0)
    notional: float = 0.0


class PortfolioIn(BaseModel):
    name: str = "Portfolio"
    positions: List[PositionIn]
    portfolio_value: float = Field(default=10_000_000.0, gt=0)
    lookback_years: int = Field(default=2, ge=1, le=10)


class StressRequest(BaseModel):
    positions: Dict[str, float]       # {ticker: weight}
    portfolio_value: float = Field(default=10_000_000.0, gt=0)
    scenario_id: str = "covid_crash"  # or "all" or "custom"
    custom_shocks: Optional[Dict[str, float]] = None


class VolRequest(BaseModel):
    ticker: str = "^NSEI"
    lookback_years: int = Field(default=3, ge=1, le=10)
    horizons: List[int] = Field(default=[1, 5, 21])


class OptionRequest(BaseModel):
    spot: float = Field(gt=0, description="Underlying price (e.g. NIFTY = 22000)")
    strike: float = Field(gt=0)
    maturity_days: int = Field(ge=1, le=365)
    vol: float = Field(gt=0, le=5.0, description="Implied vol (decimal, e.g. 0.18)")
    rate: float = Field(default=0.065, gt=0)
    option_type: str = Field(default="call", pattern="^(call|put)$")
    dividend_yield: float = Field(default=0.013, ge=0)
    ticker: str = "NIFTY"
    lot_size: int = 50


class IVRequest(BaseModel):
    market_price: float = Field(gt=0)
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    maturity_days: int = Field(ge=1, le=365)
    rate: float = Field(default=0.065)
    option_type: str = Field(default="call", pattern="^(call|put)$")


class XVAPortfolioRequest(BaseModel):
    instruments: List[Dict[str, Any]]
    market_rate: float = Field(default=0.07, description="Current IRS market fixed rate")
    spot_fx: float = Field(default=84.0, description="USD/INR spot")
    counterparty_rating: str = Field(default="BBB")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fetch_prices(tickers: List[str], years: int) -> "pd.DataFrame":
    """Fetch aligned historical prices for a list of tickers via yfinance."""
    import yfinance as yf
    import pandas as pd
    from datetime import timedelta

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365)
    frames = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end,
                             auto_adjust=True, progress=False, multi_level_index=False)
            frames[t] = df["Close"].squeeze()
        except Exception as exc:
            log.warning("Could not fetch %s: %s", t, exc)
    if not frames:
        raise HTTPException(status_code=502,
                            detail="No price data available. Check tickers and internet.")
    prices = pd.DataFrame(frames).dropna()
    if len(prices) < 30:
        raise HTTPException(status_code=422,
                            detail=f"Insufficient data: only {len(prices)} days available.")
    return prices


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health", summary="Risk engine status")
def risk_health() -> Dict[str, Any]:
    """Check availability of QuantLib, arch, and scipy."""
    status: Dict[str, Any] = {
        "status": "ok",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import QuantLib as ql
        status["quantlib"] = "available"
    except ImportError:
        status["quantlib"] = "unavailable (BSM fallback active)"

    try:
        from arch import arch_model
        status["arch_garch"] = "available"
    except ImportError:
        status["arch_garch"] = "unavailable"

    try:
        from scipy.stats import norm
        status["scipy"] = "available"
    except ImportError:
        status["scipy"] = "unavailable"

    return status


@router.post("/var", summary="Portfolio VaR — historical, parametric, Monte Carlo")
async def compute_var(req: PortfolioIn) -> Dict[str, Any]:
    """
    Compute 95% and 99% VaR + Expected Shortfall using all three methods.

    - **historical**: ranks actual portfolio P&L scenarios (no distribution assumption)
    - **parametric**: Gaussian assumption, full covariance matrix
    - **monte_carlo**: 10,000 correlated return simulations (Cholesky)

    Returns component VaR, correlation matrix, worst historical days.
    """
    from risk.var_engine import (
        PortfolioVaREngine, PortfolioInput, Position
    )
    import dataclasses

    tickers = [p.ticker for p in req.positions]
    try:
        prices = _fetch_prices(tickers, req.lookback_years)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    portfolio = PortfolioInput(
        name=req.name,
        portfolio_value=req.portfolio_value,
        positions=[Position(p.ticker, p.weight) for p in req.positions],
    )

    try:
        engine = PortfolioVaREngine()
        result = engine.compute(portfolio, prices)
    except Exception as exc:
        log.exception("VaR computation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    def _fmt(r):
        return {
            "method": r.method,
            "var_95_pct": r.confidence_95,
            "var_99_pct": r.confidence_99,
            "es_95_pct":  r.es_95,
            "es_99_pct":  r.es_99,
            "var_95_inr": r.confidence_95_inr,
            "var_99_inr": r.confidence_99_inr,
            "es_99_inr":  r.es_99_inr,
            "portfolio_vol_annual_pct": r.portfolio_vol_annual,
            "n_simulations": r.n_simulations,
            "component_var": r.component_var,
        }

    return {
        "portfolio": req.name,
        "portfolio_value": req.portfolio_value,
        "n_assets": result.n_assets,
        "lookback_days": result.lookback_days,
        "historical": _fmt(result.historical),
        "parametric": _fmt(result.parametric),
        "monte_carlo": _fmt(result.monte_carlo),
        "worst_10_days": result.worst_scenarios,
        "correlation_matrix": result.correlation_matrix,
        "computed_at": result.computed_at,
    }


@router.get("/stress/scenarios", summary="List available stress scenarios")
def list_scenarios() -> Dict[str, Any]:
    """Returns all built-in Aladdin-style stress scenarios with descriptions."""
    from risk.stress_engine import SCENARIOS
    return {
        "count": len(SCENARIOS),
        "scenarios": {
            sid: {
                "name": sc["name"],
                "description": sc["description"],
                "factor_shocks": sc.get("factor_shocks", {}),
            }
            for sid, sc in SCENARIOS.items()
        },
    }


@router.post("/stress", summary="Run stress test scenarios on a portfolio")
async def run_stress(req: StressRequest) -> Dict[str, Any]:
    """
    Apply one or all stress scenarios to a portfolio.

    - **scenario_id = "all"**: runs every built-in scenario
    - **scenario_id = "custom"**: use `custom_shocks` dict ({ticker: shock})
    - Otherwise: runs the named scenario (covid_crash, rate_hike_200bps, etc.)
    """
    from risk.stress_engine import StressTestEngine

    engine = StressTestEngine()

    # Normalise weights
    total = sum(req.positions.values())
    positions = {t: w / total for t, w in req.positions.items()}

    def _fmt(r):
        return {
            "scenario_id":         r.scenario_id,
            "scenario_name":       r.scenario_name,
            "description":         r.description,
            "portfolio_pnl_pct":   r.portfolio_pnl_pct,
            "portfolio_pnl_inr":   r.portfolio_pnl_inr,
            "recovery_capital_inr": r.recovery_capital_inr,
            "position_pnl":        r.position_pnl,
            "factor_shocks":       r.factor_shocks_applied,
            "computed_at":         r.computed_at,
        }

    try:
        if req.scenario_id == "all":
            results = engine.run_all_scenarios(positions, req.portfolio_value)
            return {
                "portfolio_value": req.portfolio_value,
                "scenarios_run": len(results),
                "worst_scenario": _fmt(results[0]) if results else None,
                "results": [_fmt(r) for r in results],
            }
        else:
            result = engine.run_scenario(
                req.scenario_id, positions, req.portfolio_value,
                custom_shocks=req.custom_shocks
            )
            return _fmt(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Stress test failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/vol", summary="GJR-GARCH volatility forecast")
async def compute_vol(req: VolRequest) -> Dict[str, Any]:
    """
    Fit GJR-GARCH(1,1) on the requested ticker's return series.
    Returns conditional volatility, regime, and multi-step forecasts.

    Handles the leverage effect: negative return shocks amplify variance more
    than positive shocks (gamma term).
    """
    from risk.garch_vol import GARCHVolEngine
    import yfinance as yf
    from datetime import timedelta

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=req.lookback_years * 365)
    try:
        df = yf.download(req.ticker, start=start, end=end,
                         auto_adjust=True, progress=False, multi_level_index=False)
        if df.empty:
            raise ValueError(f"No data for {req.ticker}")
        prices = df["Close"].squeeze()
        returns = prices.pct_change().dropna()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {exc}")

    try:
        engine = GARCHVolEngine()
        result = engine.fit_and_forecast(returns, ticker=req.ticker,
                                         horizons=req.horizons)
    except Exception as exc:
        log.exception("GARCH fit failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": result.ticker,
        "model": result.model_type,
        "params": {
            "omega": result.params.omega,
            "alpha_arch": result.params.alpha,
            "gamma_leverage": result.params.gamma,
            "beta_garch": result.params.beta,
            "persistence": result.params.persistence,
            "aic": result.params.aic,
            "bic": result.params.bic,
        },
        "current_vol_daily_pct":  result.current_vol_daily_pct,
        "current_vol_annual_pct": result.current_vol_annual_pct,
        "current_regime":         result.current_regime,
        "historical_mean_vol_annual_pct": result.historical_mean_vol_annual,
        "forecasts": [
            {
                "horizon_days":    f.horizon_days,
                "vol_daily_pct":   f.vol_pct_daily,
                "vol_annual_pct":  f.vol_pct_annual,
                "var_95_pct":      f.var_95_pct,
                "var_99_pct":      f.var_99_pct,
                "regime":          f.regime,
            }
            for f in result.forecasts
        ],
        "vol_cone": result.vol_cone,
        "computed_at": result.computed_at,
    }


@router.post("/options/price", summary="Black-Scholes / QuantLib option pricing")
def price_option(req: OptionRequest) -> Dict[str, Any]:
    """
    Price a European option using QuantLib BSM engine (falls back to NumPy BSM).
    Returns premium, intrinsic value, time value, and full Greeks.
    """
    from risk.quantlib_pricer import OptionPricer, OptionSpec

    spec = OptionSpec(
        spot=req.spot, strike=req.strike, maturity_days=req.maturity_days,
        vol=req.vol, rate=req.rate, option_type=req.option_type,
        dividend_yield=req.dividend_yield, ticker=req.ticker, lot_size=req.lot_size,
    )
    try:
        result = OptionPricer().price(spec)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": req.ticker,
        "option_type": req.option_type,
        "spot": req.spot, "strike": req.strike,
        "maturity_days": req.maturity_days,
        "vol_input_pct": round(req.vol * 100, 2),
        "pricing_engine": result.engine,
        "premium": result.premium,
        "premium_lot": result.premium_lot,
        "intrinsic": result.intrinsic,
        "time_value": result.time_value,
        "greeks": {
            "delta": result.greeks.delta,
            "gamma": result.greeks.gamma,
            "vega":  result.greeks.vega,
            "theta": result.greeks.theta,
            "rho":   result.greeks.rho,
        },
        "computed_at": result.computed_at,
    }


@router.post("/options/iv", summary="Implied volatility solver")
def solve_iv(req: IVRequest) -> Dict[str, Any]:
    """Solve for implied volatility given a market option price."""
    from risk.quantlib_pricer import OptionPricer, OptionSpec

    spec = OptionSpec(
        spot=req.spot, strike=req.strike,
        maturity_days=req.maturity_days, vol=0.20,
        rate=req.rate, option_type=req.option_type,
    )
    try:
        result = OptionPricer().implied_vol(req.market_price, spec)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "market_price": req.market_price,
        "implied_vol_pct": round(result.implied_vol * 100, 4),
        "converged": result.converged,
        "iterations": result.iterations,
    }


@router.get("/options/chain", summary="NIFTY option chain (calls + puts)")
def option_chain(
    spot: float = Query(default=22000, gt=0),
    expiry_days: int = Query(default=30, ge=1, le=365),
    vol: float = Query(default=0.18, gt=0, le=3.0),
    rate: float = Query(default=0.065),
    n_strikes: int = Query(default=11, ge=3, le=25),
    step: int = Query(default=100, ge=10),
) -> Dict[str, Any]:
    """Full NIFTY option chain: ATM ± n_strikes/2 × step."""
    from risk.quantlib_pricer import price_nifty_option_chain
    try:
        chain = price_nifty_option_chain(spot, expiry_days, vol, rate, n_strikes, step)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "spot": spot,
        "expiry_days": expiry_days,
        "vol_input_pct": round(vol * 100, 2),
        "n_strikes": len(chain),
        "chain": chain,
    }


@router.post("/xva", summary="XVA (CVA/DVA/FVA) for derivatives portfolio")
def compute_xva(req: XVAPortfolioRequest) -> Dict[str, Any]:
    """
    Compute CVA, DVA, and FVA for a portfolio of IRS / FX Forwards.

    Instrument format for IRS:
        {"type": "irs", "notional": 10000000, "fixed_rate": 0.07, "tenor_years": 5}
    Instrument format for FX Forward:
        {"type": "fx_forward", "notional_usd": 100000, "forward_rate": 85.0, "tenor_days": 90}
    """
    from risk.xva_engine import (
        XVAEngine, IRSwapSpec, FXForwardSpec
    )

    engine = XVAEngine()
    results = []
    total_cva = total_dva = total_fva = total_mtm = 0.0

    for inst in req.instruments:
        try:
            itype = inst.get("type", "irs").lower()
            if itype == "irs":
                spec = IRSwapSpec(
                    notional=float(inst.get("notional", 10_000_000)),
                    fixed_rate=float(inst.get("fixed_rate", 0.07)),
                    tenor_years=int(inst.get("tenor_years", 5)),
                    pay_fixed=bool(inst.get("pay_fixed", True)),
                    counterparty=inst.get("counterparty", "Bank_A"),
                    collateralised=bool(inst.get("collateralised", False)),
                )
                r = engine.compute_irs_xva(
                    spec, req.market_rate,
                    counterparty_rating=req.counterparty_rating
                )
            elif itype == "fx_forward":
                spec = FXForwardSpec(
                    notional_usd=float(inst.get("notional_usd", 100_000)),
                    forward_rate=float(inst.get("forward_rate", req.spot_fx)),
                    tenor_days=int(inst.get("tenor_days", 90)),
                    counterparty=inst.get("counterparty", "Bank_B"),
                )
                r = engine.compute_fx_forward_xva(
                    spec, req.spot_fx,
                    counterparty_rating=req.counterparty_rating
                )
            else:
                continue

            results.append({
                "id": r.instrument_id,
                "type": r.instrument_type,
                "mtm": r.mtm,
                "cva": r.cva,
                "dva": r.dva,
                "fva": r.fva,
                "xva_total": r.xva_total,
                "mtm_net_xva": r.mtm_net_xva,
                "epe": r.epe,
                "ene": r.ene,
                "pd_counterparty": r.pd_counterparty,
            })
            total_mtm += r.mtm
            total_cva += r.cva
            total_dva += r.dva
            total_fva += r.fva

        except Exception as exc:
            log.warning("XVA failed for %s: %s", inst, exc)
            results.append({"error": str(exc), "instrument": inst})

    return {
        "portfolio_xva": {
            "total_mtm": round(total_mtm, 2),
            "total_cva": round(total_cva, 2),
            "total_dva": round(total_dva, 2),
            "total_fva": round(total_fva, 2),
            "total_xva": round(total_cva + total_dva + total_fva, 2),
            "mtm_net_all_xva": round(total_mtm + total_cva + total_dva + total_fva, 2),
        },
        "counterparty_rating": req.counterparty_rating,
        "instruments": results,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
