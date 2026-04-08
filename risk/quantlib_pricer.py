"""
Phase 3 – Step 8: QuantLib Derivatives Pricer
================================================
Black-Scholes-Merton option pricing for NIFTY / NSE equity options.
Also provides:
  - Greeks: Delta, Gamma, Vega, Theta, Rho
  - Implied Volatility (Newton-Raphson solver)
  - Put-Call Parity check
  - Volatility surface from multiple strikes/expiries

Uses the `QuantLib` Python bindings (pip install QuantLib).
Falls back to pure-NumPy BSM if QuantLib is not installed.

Usage:
    from risk.quantlib_pricer import OptionPricer, OptionSpec
    pricer = OptionPricer()
    result = pricer.price(OptionSpec(
        spot=22000, strike=22500, maturity_days=30,
        vol=0.18, rate=0.065, option_type="call"
    ))
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Suppress QuantLib deprecation notices only.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="QuantLib")
log = logging.getLogger(__name__)

# ── Try to import QuantLib (optional) ────────────────────────────────────────
try:
    import QuantLib as ql
    _QL_AVAILABLE = True
    log.info("QuantLib %s available.", ql.__version__ if hasattr(ql, "__version__") else "?")
except ImportError:
    _QL_AVAILABLE = False
    log.warning("QuantLib not installed. Using NumPy BSM fallback. "
                "Install with: pip install QuantLib")


# ── Input / Output Models ─────────────────────────────────────────────────────
@dataclass
class OptionSpec:
    spot: float              # Current underlying price (e.g. NIFTY = 22000)
    strike: float            # Option strike price
    maturity_days: int       # Calendar days to expiry
    vol: float               # Implied/historical vol (decimal, e.g. 0.18)
    rate: float              # Risk-free rate (decimal, e.g. 0.065 = 6.5%)
    option_type: str         # "call" or "put"
    dividend_yield: float = 0.013   # NIFTY dividend yield ~1.3%
    lot_size: int = 50              # NIFTY lot size (50 units)
    ticker: str = "NIFTY"


@dataclass
class Greeks:
    delta: float     # dV/dS
    gamma: float     # d²V/dS²
    vega: float      # dV/dσ (per 1% vol move)
    theta: float     # dV/dt (per calendar day)
    rho: float       # dV/dr (per 1% rate move)


@dataclass
class OptionResult:
    spec: OptionSpec
    premium: float          # option price per unit
    premium_lot: float      # option price × lot_size
    intrinsic: float        # max(0, S-K) or max(0, K-S)
    time_value: float       # premium - intrinsic
    greeks: Greeks
    engine: str             # "QuantLib" or "BSM_numpy"
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ImpliedVolResult:
    market_price: float
    implied_vol: float      # solved IV
    converged: bool
    iterations: int


# ── Pure-NumPy BSM (always available) ────────────────────────────────────────
def _bsm_price(S, K, T, sigma, r, q, call: bool) -> float:
    """Black-Scholes-Merton price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(0, S - K) if call else max(0, K - S)
        return intrinsic
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return (S * math.exp(-q * T) * norm.cdf(d1)
                - K * math.exp(-r * T) * norm.cdf(d2))
    else:
        return (K * math.exp(-r * T) * norm.cdf(-d2)
                - S * math.exp(-q * T) * norm.cdf(-d1))


def _bsm_greeks(S, K, T, sigma, r, q, call: bool) -> Tuple[float, ...]:
    """Returns (delta, gamma, vega, theta, rho)."""
    if T <= 0:
        return (1.0 if call else -1.0), 0.0, 0.0, 0.0, 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    e_rt  = math.exp(-r * T)
    e_qt  = math.exp(-q * T)

    gamma = e_qt * pdf_d1 / (S * sigma * math.sqrt(T))
    vega  = S * e_qt * pdf_d1 * math.sqrt(T) / 100   # per 1% vol

    if call:
        delta = e_qt * cdf_d1
        theta = (-(S * e_qt * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 - r * K * e_rt * cdf_d2
                 + q * S * e_qt * cdf_d1) / 365
        rho   = K * T * e_rt * cdf_d2 / 100
    else:
        delta = -e_qt * norm.cdf(-d1)
        theta = (-(S * e_qt * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 + r * K * e_rt * norm.cdf(-d2)
                 - q * S * e_qt * norm.cdf(-d1)) / 365
        rho   = -K * T * e_rt * norm.cdf(-d2) / 100

    return delta, gamma, vega, theta, rho


# ── QuantLib-based pricing ────────────────────────────────────────────────────
def _ql_price(spec: OptionSpec) -> Tuple[float, Greeks]:
    """Price with QuantLib BSM engine (European option)."""
    import QuantLib as ql

    T_years = spec.maturity_days / 365.0
    today = ql.Date.todaysDate()
    expiry = today + ql.Period(spec.maturity_days, ql.Days)
    ql.Settings.instance().evaluationDate = today

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if spec.option_type.lower() == "call" else ql.Option.Put,
        spec.strike,
    )
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle    = ql.QuoteHandle(ql.SimpleQuote(spec.spot))
    rate_ts        = ql.YieldTermStructureHandle(
        ql.FlatForward(today, spec.rate, ql.Actual365Fixed()))
    div_ts         = ql.YieldTermStructureHandle(
        ql.FlatForward(today, spec.dividend_yield, ql.Actual365Fixed()))
    vol_ts         = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), spec.vol, ql.Actual365Fixed()))

    process = ql.BlackScholesMertonProcess(spot_handle, div_ts, rate_ts, vol_ts)
    engine  = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    premium = float(option.NPV())
    greeks  = Greeks(
        delta=float(option.delta()),
        gamma=float(option.gamma()),
        vega=float(option.vega()) / 100,
        theta=float(option.theta()) / 365,
        rho=float(option.rho()) / 100,
    )
    return premium, greeks


# ── Main Pricer ───────────────────────────────────────────────────────────────
class OptionPricer:
    """
    European option pricer supporting QuantLib (preferred) and NumPy BSM (fallback).

    Example — price a NIFTY 22500 CE expiring in 30 days:
        pricer = OptionPricer()
        spec   = OptionSpec(spot=22000, strike=22500, maturity_days=30,
                            vol=0.18, rate=0.065, option_type="call")
        result = pricer.price(spec)
        print(f"Premium: {result.premium:.2f}  Delta: {result.greeks.delta:.4f}")
    """

    def price(self, spec: OptionSpec) -> OptionResult:
        """Price a single option."""
        T = spec.maturity_days / 365.0
        call = spec.option_type.lower() == "call"
        intrinsic = max(0, spec.spot - spec.strike) if call else max(0, spec.strike - spec.spot)

        if _QL_AVAILABLE:
            try:
                premium, greeks = _ql_price(spec)
                engine_used = "QuantLib"
            except Exception as exc:
                log.warning("QuantLib pricing failed (%s), falling back to BSM", exc)
                premium, greeks = self._numpy_price(spec, T, call)
                engine_used = "BSM_numpy_fallback"
        else:
            premium, greeks = self._numpy_price(spec, T, call)
            engine_used = "BSM_numpy"

        return OptionResult(
            spec=spec,
            premium=round(premium, 4),
            premium_lot=round(premium * spec.lot_size, 2),
            intrinsic=round(intrinsic, 4),
            time_value=round(max(0, premium - intrinsic), 4),
            greeks=greeks,
            engine=engine_used,
        )

    @staticmethod
    def _numpy_price(spec: OptionSpec, T: float, call: bool) -> Tuple[float, Greeks]:
        premium = _bsm_price(spec.spot, spec.strike, T, spec.vol,
                              spec.rate, spec.dividend_yield, call)
        d, g, v, th, rh = _bsm_greeks(spec.spot, spec.strike, T, spec.vol,
                                        spec.rate, spec.dividend_yield, call)
        return premium, Greeks(delta=round(d, 6), gamma=round(g, 8),
                               vega=round(v, 6), theta=round(th, 6), rho=round(rh, 6))

    def implied_vol(
        self,
        market_price: float,
        spec: OptionSpec,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> ImpliedVolResult:
        """
        Solve for implied volatility via Brent-q root finder.
        target: BSM(sigma) - market_price = 0
        """
        T = spec.maturity_days / 365.0
        call = spec.option_type.lower() == "call"

        def objective(sigma):
            return _bsm_price(spec.spot, spec.strike, T, sigma,
                               spec.rate, spec.dividend_yield, call) - market_price

        try:
            iv, result = brentq(objective, 1e-4, 5.0, xtol=tol,
                                maxiter=max_iter, full_output=True)
            return ImpliedVolResult(
                market_price=market_price,
                implied_vol=round(float(iv), 6),
                converged=result.converged,
                iterations=result.iterations,
            )
        except ValueError:
            return ImpliedVolResult(
                market_price=market_price,
                implied_vol=float("nan"),
                converged=False,
                iterations=max_iter,
            )

    def build_vol_surface(
        self,
        spot: float,
        strikes: List[float],
        maturities_days: List[int],
        market_prices: Optional[List[List[float]]] = None,   # [maturity][strike]
        base_vol: float = 0.18,
        rate: float = 0.065,
        option_type: str = "call",
    ) -> Dict:
        """
        Build a simplified implied volatility surface.
        If market_prices provided → solve IV for each point.
        Otherwise → use simplified skew model (vol smile).
        """
        surface = {}
        for i, mat in enumerate(maturities_days):
            surface[f"{mat}d"] = {}
            for j, K in enumerate(strikes):
                moneyness = K / spot
                if market_prices and i < len(market_prices) and j < len(market_prices[i]):
                    mp = market_prices[i][j]
                    spec = OptionSpec(spot=spot, strike=K, maturity_days=mat,
                                      vol=base_vol, rate=rate, option_type=option_type)
                    iv_result = self.implied_vol(mp, spec)
                    iv = iv_result.implied_vol
                else:
                    # Simplified skew: vol increases for OTM puts (put skew)
                    skew  = 0.05 * (1.0 - moneyness)   # +5% vol per 10% OTM
                    smile = 0.02 * (moneyness - 1.0)**2 * 10  # quadratic smile
                    iv = base_vol + skew + smile

                surface[f"{mat}d"][str(round(K))] = round(float(iv) * 100, 2)

        return {
            "spot": spot,
            "rate": rate,
            "strikes": [str(round(k)) for k in strikes],
            "maturities": [f"{m}d" for m in maturities_days],
            "implied_vols_pct": surface,
        }


# ── Convenience: NIFTY Option Chain ──────────────────────────────────────────
def price_nifty_option_chain(
    spot: float,
    expiry_days: int = 30,
    vol: float = 0.18,
    rate: float = 0.065,
    n_strikes: int = 11,
    step: int = 100,
) -> List[Dict]:
    """
    Price a NIFTY option chain: ATM ± n_strikes/2 strikes.
    Returns list of {strike, call_premium, put_premium, call_delta, put_delta, ...}
    """
    atm = round(spot / step) * step
    strikes = [atm + (i - n_strikes // 2) * step for i in range(n_strikes)]
    pricer = OptionPricer()
    chain = []

    for K in strikes:
        call_spec = OptionSpec(spot=spot, strike=K, maturity_days=expiry_days,
                               vol=vol, rate=rate, option_type="call")
        put_spec  = OptionSpec(spot=spot, strike=K, maturity_days=expiry_days,
                               vol=vol, rate=rate, option_type="put")
        c = pricer.price(call_spec)
        p = pricer.price(put_spec)
        chain.append({
            "strike": K,
            "moneyness": round(K / spot, 4),
            "call_premium": c.premium,
            "put_premium":  p.premium,
            "call_delta":   c.greeks.delta,
            "put_delta":    p.greeks.delta,
            "gamma":        c.greeks.gamma,
            "vega":         c.greeks.vega,
            "call_theta":   c.greeks.theta,
            "put_theta":    p.greeks.theta,
            "intrinsic_call": c.intrinsic,
            "intrinsic_put":  p.intrinsic,
            "pcp_check": round(c.premium - p.premium - (spot - K * math.exp(-rate * expiry_days / 365)), 4),
        })

    return chain
