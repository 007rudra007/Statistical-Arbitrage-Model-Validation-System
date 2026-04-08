"""
Phase 5 – Step 18: Portfolio Optimizer
========================================
CVXPY mean-variance optimizer for NSE universe.

Features:
  - Markowitz mean-variance (min-variance + max-Sharpe + risk-parity)
  - Constraints: beta < 1, sector caps, long-only, full-investment
  - Transaction cost model (roundtrip brokerage + STT)
  - Rebalancing signal vs current weights
  - Output: optimal weights + expected return/vol/Sharpe

Endpoint: POST /portfolio/optimize
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE  = 0.065          # 6.5% RBI repo rate (annual)
TRADING_DAYS    = 252
ROUNDTRIP_COST  = 0.0020         # 20bps total (brokerage + STT + stamp)
MAX_WEIGHT      = 0.15           # 15% single-stock cap
MIN_WEIGHT      = 0.00           # no short selling
MAX_SECTOR_WEIGHT = 0.30         # 30% sector cap
MAX_BETA        = 0.95           # portfolio beta ceiling


@dataclass
class Asset:
    ticker: str
    sector: str
    beta: float = 1.0
    current_weight: float = 0.0


@dataclass
class OptimizationInput:
    assets: List[Asset]
    returns_matrix: np.ndarray        # shape (T, N) — daily returns
    portfolio_value: float = 10_000_000.0
    objective: str = "min_variance"   # 'min_variance' | 'max_sharpe' | 'risk_parity'
    max_weight: float = MAX_WEIGHT
    min_weight: float = MIN_WEIGHT
    max_sector_weight: float = MAX_SECTOR_WEIGHT
    max_portfolio_beta: float = MAX_BETA
    risk_free_rate: float = RISK_FREE_RATE
    transaction_cost: float = ROUNDTRIP_COST


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_annual_return: float
    expected_annual_vol: float
    sharpe_ratio: float
    portfolio_beta: float
    sector_weights: Dict[str, float]
    rebalancing_trades: List[Dict]   # [{ticker, from_w, to_w, delta_inr, direction}]
    transaction_cost_inr: float
    solver_status: str
    objective: str
    n_assets: int
    effective_n: float               # 1 / sum(w²) — diversification measure


# ── Core optimizer ────────────────────────────────────────────────────────────
class PortfolioOptimizer:
    """
    CVXPY-based mean-variance portfolio optimizer for NSE equities.
    Falls back to numpy closed-form min-variance if CVXPY is unavailable.
    """

    def optimize(self, inp: OptimizationInput) -> OptimizationResult:
        n = len(inp.assets)
        tickers = [a.ticker for a in inp.assets]
        sectors  = [a.sector for a in inp.assets]
        betas    = np.array([a.beta for a in inp.assets])
        cur_w    = np.array([a.current_weight for a in inp.assets])

        # Annualise moments
        R = inp.returns_matrix                      # (T, N)
        mu = R.mean(axis=0) * TRADING_DAYS          # (N,) annual returns
        Sigma = np.cov(R.T) * TRADING_DAYS          # (N,N) annual cov

        try:
            import cvxpy as cp
            w_opt, status = self._cvxpy_solve(
                cp, n, mu, Sigma, betas, sectors,
                inp.objective, inp.max_weight, inp.min_weight,
                inp.max_sector_weight, inp.max_portfolio_beta,
                inp.risk_free_rate,
            )
        except ImportError:
            log.warning("[Optimizer] CVXPY not installed — using numpy fallback")
            w_opt, status = self._numpy_min_variance(n, Sigma, inp.max_weight)

        w_opt = np.maximum(w_opt, 0)
        w_opt /= w_opt.sum()                        # normalise

        # ── Performance metrics ───────────────────────────────────────────────
        exp_ret = float(mu @ w_opt)
        exp_vol = float(np.sqrt(w_opt @ Sigma @ w_opt))
        sharpe  = (exp_ret - inp.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
        port_beta = float(betas @ w_opt)
        eff_n   = 1.0 / float(np.sum(w_opt**2)) if np.sum(w_opt**2) > 1e-12 else 1.0

        # ── Sector aggregation ────────────────────────────────────────────────
        sector_w: Dict[str, float] = {}
        for i, s in enumerate(sectors):
            sector_w[s] = sector_w.get(s, 0.0) + float(w_opt[i])

        # ── Rebalancing trades ────────────────────────────────────────────────
        trades = []
        total_cost = 0.0
        for i, t in enumerate(tickers):
            delta = float(w_opt[i]) - float(cur_w[i])
            if abs(delta) > 0.001:
                notional = abs(delta) * inp.portfolio_value
                cost     = notional * inp.transaction_cost
                total_cost += cost
                trades.append({
                    "ticker":    t,
                    "from_weight": round(float(cur_w[i]), 4),
                    "to_weight":   round(float(w_opt[i]), 4),
                    "delta_weight": round(delta, 4),
                    "notional_inr": round(notional, 2),
                    "cost_inr":    round(cost, 2),
                    "direction":   "BUY" if delta > 0 else "SELL",
                })

        return OptimizationResult(
            weights         = {t: round(float(w_opt[i]), 6) for i, t in enumerate(tickers)},
            expected_annual_return = round(exp_ret, 6),
            expected_annual_vol    = round(exp_vol, 6),
            sharpe_ratio           = round(sharpe, 4),
            portfolio_beta         = round(port_beta, 4),
            sector_weights         = {k: round(v, 4) for k, v in sector_w.items()},
            rebalancing_trades     = trades,
            transaction_cost_inr   = round(total_cost, 2),
            solver_status          = status,
            objective              = inp.objective,
            n_assets               = n,
            effective_n            = round(eff_n, 2),
        )

    # ── CVXPY solvers ─────────────────────────────────────────────────────────
    def _cvxpy_solve(
        self, cp, n, mu, Sigma, betas, sectors,
        objective, max_w, min_w, max_sector_w, max_beta, rf,
    ) -> Tuple[np.ndarray, str]:

        w = cp.Variable(n)

        # Base constraints
        constraints = [
            cp.sum(w) == 1,
            w >= min_w,
            w <= max_w,
            betas @ w <= max_beta,
        ]

        # Sector constraints
        unique_sectors = list(set(sectors))
        for sec in unique_sectors:
            idx = [i for i, s in enumerate(sectors) if s == sec]
            constraints.append(cp.sum(w[idx]) <= max_sector_w)

        portfolio_variance = cp.quad_form(w, Sigma)
        portfolio_return   = mu @ w

        if objective == "max_sharpe":
            # Maximise Sharpe via Dinkelbach-style rescaling (auxiliary variable κ)
            kappa = cp.Variable(pos=True)
            y     = kappa * w
            constraints_sr = [
                cp.sum(y) == kappa,
                y >= 0,
                y <= max_w * kappa,
                betas @ y <= max_beta * kappa,
            ]
            for sec in unique_sectors:
                idx = [i for i, s in enumerate(sectors) if s == sec]
                constraints_sr.append(cp.sum(y[idx]) <= max_sector_w * kappa)
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(y, Sigma)),
                constraints_sr + [(mu - rf) @ y >= 1],
            )
            prob.solve(solver=cp.CLARABEL, warm_start=True)
            if prob.status in ("optimal", "optimal_inaccurate") and kappa.value > 1e-9:
                return y.value / kappa.value, prob.status
            # fall through to min_variance
            log.warning("[Optimizer] max_sharpe solve failed (%s) — fallback to min_variance", prob.status)

        if objective == "risk_parity":
            return self._risk_parity_numpy(n, Sigma, max_w), "risk_parity_newton"

        # Default: min variance
        prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            log.warning("[Optimizer] CVXPY status=%s — using numpy fallback", prob.status)
            return self._numpy_min_variance(n, Sigma, max_w)
        return w.value, prob.status

    # ── Pure-numpy fallbacks ──────────────────────────────────────────────────
    def _numpy_min_variance(self, n: int, Sigma: np.ndarray, max_w: float):
        """Closed-form global minimum variance (long-only, equal-weight warm-start)."""
        ones = np.ones(n)
        try:
            Sigma_inv = np.linalg.pinv(Sigma)
            w = Sigma_inv @ ones
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w /= w.sum()
            else:
                w = np.full(n, 1.0 / n)
        except np.linalg.LinAlgError:
            w = np.full(n, 1.0 / n)
        # Clip to max_weight and renormalise
        w = np.clip(w, 0, max_w)
        w /= w.sum()
        return w, "numpy_min_var"

    def _risk_parity_numpy(self, n: int, Sigma: np.ndarray, max_w: float,
                            tol: float = 1e-8, max_iter: int = 500) -> np.ndarray:
        """Risk parity via Newton–Raphson budget equalisation."""
        w = np.full(n, 1.0 / n)
        for _ in range(max_iter):
            Sigma_w  = Sigma @ w
            port_var = w @ Sigma_w
            rc       = w * Sigma_w / port_var           # risk contributions
            grad     = rc - 1.0 / n                     # target: equal RC
            H        = (np.diag(Sigma_w) + np.outer(w, Sigma @ np.ones(n))) / port_var
            w        = w - np.linalg.solve(H + 1e-6 * np.eye(n), grad)
            w        = np.maximum(w, 1e-8)
            w       /= w.sum()
            if np.max(np.abs(grad)) < tol:
                break
        return np.clip(w, 0, max_w) / np.clip(w, 0, max_w).sum()


# ── NSE Universe defaults ─────────────────────────────────────────────────────
NSE_UNIVERSE: List[Asset] = [
    Asset("RELIANCE.NS",  "Energy",       beta=0.95),
    Asset("HDFCBANK.NS",  "Financials",   beta=1.05),
    Asset("ICICIBANK.NS", "Financials",   beta=1.10),
    Asset("INFY.NS",      "IT",           beta=0.85),
    Asset("TCS.NS",       "IT",           beta=0.80),
    Asset("AXISBANK.NS",  "Financials",   beta=1.15),
    Asset("BAJFINANCE.NS","Financials",   beta=1.20),
    Asset("HINDUNILVR.NS","FMCG",         beta=0.65),
    Asset("ITC.NS",       "FMCG",         beta=0.70),
    Asset("LT.NS",        "Industrials",  beta=0.90),
    Asset("SBIN.NS",      "Financials",   beta=1.10),
    Asset("KOTAKBANK.NS", "Financials",   beta=0.95),
    Asset("ASIANPAINT.NS","FMCG",         beta=0.75),
    Asset("MARUTI.NS",    "Auto",         beta=0.85),
    Asset("TITAN.NS",     "Consumer",     beta=0.90),
]
