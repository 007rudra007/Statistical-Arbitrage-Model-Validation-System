"""
Phase 3 – Step 9: Portfolio VaR Engine
========================================
Three methods identical to Aladdin SDK risk:
  1. Historical Simulation   – empirical distribution of portfolio P&L
  2. Parametric (Delta-Normal)– Gaussian assumption, covariance matrix
  3. Monte Carlo             – 10,000 correlated return simulations

All methods return VaR and Expected Shortfall (CVaR) at 95% and 99%.

Usage:
    from risk.var_engine import PortfolioVaREngine, PortfolioInput
    engine = PortfolioVaREngine()
    result = engine.compute(portfolio, prices_df)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# Targeted suppression only: scipy/numpy runtime warnings from normal distribution tails.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
log = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class Position:
    ticker: str
    weight: float           # portfolio weight 0-1
    notional: float = 0.0  # absolute notional (INR). Derived if 0.


@dataclass
class PortfolioInput:
    """
    Input: a list of positions + total portfolio value.
    Weights must sum to ~1.0 (we normalise).
    """
    positions: List[Position]
    portfolio_value: float = 10_000_000.0   # ₹1 Cr default
    currency: str = "INR"
    name: str = "Portfolio"

    def normalise_weights(self) -> None:
        total = sum(p.weight for p in self.positions)
        if abs(total - 1.0) > 0.01:
            log.warning("Weights sum to %.4f — normalising.", total)
            for p in self.positions:
                p.weight /= total


@dataclass
class VaRResult:
    method: str
    confidence_95: float        # VaR at 95% (positive = loss)
    confidence_99: float        # VaR at 99% (positive = loss)
    es_95: float                # Expected Shortfall at 95%
    es_99: float                # Expected Shortfall at 99%
    confidence_95_inr: float    # Dollar VaR at 95%
    confidence_99_inr: float    # Dollar VaR at 99%
    es_99_inr: float
    n_simulations: int = 0      # for MC
    portfolio_vol_annual: float = 0.0
    component_var: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioVaROutput:
    portfolio_name: str
    portfolio_value: float
    n_assets: int
    lookback_days: int
    historical: VaRResult
    parametric: VaRResult
    monte_carlo: VaRResult
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_matrix: Optional[Dict] = None
    worst_scenarios: Optional[List[Dict]] = None


# ── Engine ────────────────────────────────────────────────────────────────────
class PortfolioVaREngine:
    """
    Aladdin-style multi-method VaR engine for a weighted portfolio.

    Accepts a PortfolioInput and a DataFrame of historical prices
    (one column per ticker, DatetimeIndex).
    """

    N_MC: int = 10_000          # Monte Carlo paths
    LOOKBACK_DAYS: int = 252 * 2  # 2-year history for historical VaR

    def compute(
        self,
        portfolio: PortfolioInput,
        prices: pd.DataFrame,
        lookback_days: int = LOOKBACK_DAYS,
    ) -> PortfolioVaROutput:
        """
        Main entry point. Runs all three VaR methods.

        Args:
            portfolio: PortfolioInput with positions + total value
            prices: DataFrame[ticker → daily close], DatetimeIndex sorted
            lookback_days: history window for historical simulation

        Returns:
            PortfolioVaROutput with historical, parametric, MC results
        """
        portfolio.normalise_weights()

        # Align tickers
        tickers = [p.ticker for p in portfolio.positions]
        weights = np.array([p.weight for p in portfolio.positions])
        available = [t for t in tickers if t in prices.columns]
        if not available:
            raise ValueError(f"None of {tickers} found in prices DataFrame")

        weights, tickers = self._align_weights(tickers, weights, available)
        px = prices[tickers].dropna().tail(lookback_days)

        log.info("VaR engine: %d assets, %d days history, value=%.0f",
                 len(tickers), len(px), portfolio.portfolio_value)

        returns = px.pct_change().dropna()
        port_returns = returns @ weights          # weighted portfolio returns

        hist = self._historical_var(
            port_returns, tickers, weights, returns, portfolio.portfolio_value
        )
        param = self._parametric_var(
            returns, weights, tickers, portfolio.portfolio_value
        )
        mc = self._monte_carlo_var(
            returns, weights, tickers, portfolio.portfolio_value
        )

        # Correlation matrix (for UI)
        corr = returns.corr().round(4).to_dict()

        # Worst 10 historical days
        worst = (
            port_returns.sort_values()
            .head(10)
            .reset_index()
        )
        worst_scenarios = [
            {"date": str(row.iloc[0])[:10], "return_pct": round(float(row.iloc[1]) * 100, 3)}
            for _, row in worst.iterrows()
        ]

        return PortfolioVaROutput(
            portfolio_name=portfolio.name,
            portfolio_value=portfolio.portfolio_value,
            n_assets=len(tickers),
            lookback_days=len(px),
            historical=hist,
            parametric=param,
            monte_carlo=mc,
            correlation_matrix=corr,
            worst_scenarios=worst_scenarios,
        )

    # ── 1. Historical Simulation ──────────────────────────────────────────────
    def _historical_var(
        self,
        port_returns: pd.Series,
        tickers: List[str],
        weights: np.ndarray,
        asset_returns: pd.DataFrame,
        value: float,
    ) -> VaRResult:
        """
        Historical VaR: rank historical portfolio P&L scenarios.
        No distribution assumption — uses actual return distribution.
        """
        r = port_returns.dropna().values

        var95 = float(-np.percentile(r, 5))
        var99 = float(-np.percentile(r, 1))
        tail95 = r[r <= -var95]
        tail99 = r[r <= -var99]
        es95 = float(-tail95.mean()) if len(tail95) else var95
        es99 = float(-tail99.mean()) if len(tail99) else var99

        # Component VaR (marginal contribution)
        comp = self._component_var(asset_returns, weights, tickers, var99)

        return VaRResult(
            method="historical_simulation",
            confidence_95=round(var95 * 100, 4),
            confidence_99=round(var99 * 100, 4),
            es_95=round(es95 * 100, 4),
            es_99=round(es99 * 100, 4),
            confidence_95_inr=round(var95 * value, 2),
            confidence_99_inr=round(var99 * value, 2),
            es_99_inr=round(es99 * value, 2),
            component_var=comp,
        )

    # ── 2. Parametric (Delta-Normal) ──────────────────────────────────────────
    def _parametric_var(
        self,
        asset_returns: pd.DataFrame,
        weights: np.ndarray,
        tickers: List[str],
        value: float,
    ) -> VaRResult:
        """
        Parametric VaR: assumes multivariate normal returns.
        Uses full covariance matrix.
        portfolio_var = w' Σ w
        VaR = -portfolio_vol * z_alpha * sqrt(1)  (daily, 1-day holding)
        """
        cov = asset_returns.cov().values
        port_var = float(weights @ cov @ weights)
        port_vol = float(np.sqrt(port_var))
        port_vol_annual = port_vol * np.sqrt(252)

        z95, z99 = norm.ppf(0.05), norm.ppf(0.01)
        var95 = float(-z95 * port_vol)
        var99 = float(-z99 * port_vol)

        # ES = phi(z) / (1-c) * port_vol
        es95 = float(norm.pdf(z95) / 0.05 * port_vol)
        es99 = float(norm.pdf(z99) / 0.01 * port_vol)

        comp = self._component_var(asset_returns, weights, tickers, var99)

        return VaRResult(
            method="parametric_delta_normal",
            confidence_95=round(var95 * 100, 4),
            confidence_99=round(var99 * 100, 4),
            es_95=round(es95 * 100, 4),
            es_99=round(es99 * 100, 4),
            confidence_95_inr=round(var95 * value, 2),
            confidence_99_inr=round(var99 * value, 2),
            es_99_inr=round(es99 * value, 2),
            portfolio_vol_annual=round(port_vol_annual * 100, 4),
            component_var=comp,
        )

    # ── 3. Monte Carlo ────────────────────────────────────────────────────────
    def _monte_carlo_var(
        self,
        asset_returns: pd.DataFrame,
        weights: np.ndarray,
        tickers: List[str],
        value: float,
        n_sims: int = N_MC,
    ) -> VaRResult:
        """
        Monte Carlo VaR: simulate N_MC correlated 1-day return paths
        using Cholesky decomposition of the covariance matrix.
        """
        cov = asset_returns.cov().values
        means = asset_returns.mean().values
        n = len(tickers)

        # Cholesky decomposition for correlated normals
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Add small ridge if matrix is not positive definite
            cov += np.eye(n) * 1e-8
            L = np.linalg.cholesky(cov)

        rng = np.random.default_rng(seed=42)
        Z = rng.standard_normal((n_sims, n))
        sim_returns = means + (Z @ L.T)          # shape (N_MC, n)
        port_sim = sim_returns @ weights          # shape (N_MC,)

        var95 = float(-np.percentile(port_sim, 5))
        var99 = float(-np.percentile(port_sim, 1))
        tail95 = port_sim[port_sim <= -var95]
        tail99 = port_sim[port_sim <= -var99]
        es95 = float(-tail95.mean()) if len(tail95) else var95
        es99 = float(-tail99.mean()) if len(tail99) else var99

        comp = self._component_var(asset_returns, weights, tickers, var99)

        return VaRResult(
            method="monte_carlo",
            confidence_95=round(var95 * 100, 4),
            confidence_99=round(var99 * 100, 4),
            es_95=round(es95 * 100, 4),
            es_99=round(es99 * 100, 4),
            confidence_95_inr=round(var95 * value, 2),
            confidence_99_inr=round(var99 * value, 2),
            es_99_inr=round(es99 * value, 2),
            n_simulations=n_sims,
            component_var=comp,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _align_weights(
        tickers: List[str], weights: np.ndarray, available: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        idx = [tickers.index(t) for t in available]
        w = weights[idx]
        w = w / w.sum()
        return w, available

    @staticmethod
    def _component_var(
        returns: pd.DataFrame,
        weights: np.ndarray,
        tickers: List[str],
        portfolio_var_decimal: float,
    ) -> Dict[str, float]:
        """
        Euler decomposition of portfolio VaR into per-asset contributions.

        Beta_i = w_i * (Σw)_i / (w'Σw)           → sums to exactly 1
        Component VaR_i (in %) = Beta_i × VaR_pct → sums to VaR_pct

        Args:
            portfolio_var_decimal: total 1-day portfolio VaR in decimal
                                   (e.g. 0.029 = 2.9%)
        """
        cov = returns.cov().values
        sigma_w = cov @ weights                    # Σw  (n-vector, decimal²/decimal)
        port_var = float(weights @ sigma_w)        # w'Σw (scalar, decimal²)
        if port_var <= 0:
            return {t: 0.0 for t in tickers}

        # Fractional contribution per asset — guaranteed to sum to 1
        beta = weights * sigma_w / port_var        # shape (n,), sums to 1

        var_pct = portfolio_var_decimal * 100       # convert to %
        comp = {
            t: round(float(beta[i]) * var_pct, 4)
            for i, t in enumerate(tickers)
        }
        return comp


