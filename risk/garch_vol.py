"""
Phase 3 – Step 11: GJR-GARCH Volatility Service
=================================================
GJR-GARCH(1,1) on NIFTY / equity returns with:
  - Asymmetric leverage effect (negative shocks increase variance more)
  - Multi-step forecasts: 1-day, 5-day, 21-day (1W, 1M)
  - Annualised volatility cone
  - Volatility regime classification (low/mid/high/crisis)
  - GARCH VaR (volatility-scaled quantile)

Different from alpha/garch_model.py which fits GARCH on the pair spread.
This module is specifically for NIFTY return forecasting for risk management.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from arch.__future__ import reindexing

# Only suppress arch's convergence noise; preserve all other warnings.
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arch")
log = logging.getLogger(__name__)


# ── Regime thresholds (annualised vol, %) ─────────────────────────────────────
REGIME_THRESHOLDS = {
    "low":    (0.0,  12.0),
    "normal": (12.0, 20.0),
    "high":   (20.0, 30.0),
    "crisis": (30.0, 999.0),
}


# ── Result Types ──────────────────────────────────────────────────────────────
@dataclass
class GARCHParams:
    omega: float
    alpha: float        # ARCH term
    gamma: float        # GJR leverage term
    beta: float         # GARCH term
    persistence: float  # alpha + gamma/2 + beta
    dist: str = "normal"
    aic: float = 0.0
    bic: float = 0.0
    log_likelihood: float = 0.0


@dataclass
class VolForecast:
    horizon_days: int
    vol_pct_daily: float        # daily vol %
    vol_pct_annual: float       # annualised vol %
    var_95_pct: float           # 1-day VaR at 95%
    var_99_pct: float           # 1-day VaR at 99%
    regime: str                 # low / normal / high / crisis


@dataclass
class GARCHVolResult:
    ticker: str
    model_type: str
    params: GARCHParams
    current_vol_daily_pct: float
    current_vol_annual_pct: float
    current_regime: str
    forecasts: List[VolForecast]        # 1d, 5d, 21d
    vol_cone: Dict[str, float]          # percentile → annualised vol
    historical_mean_vol_annual: float
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── GJR-GARCH Engine ──────────────────────────────────────────────────────────
class GARCHVolEngine:
    """
    Fit GJR-GARCH(1,1) on return series and forecast volatility.

    GJR (Glosten-Jagannathan-Runkle) adds an asymmetric leverage term:
        sigma²(t) = omega + (alpha + gamma * I_{eps<0}) * eps²(t-1) + beta * sigma²(t-1)

    where gamma > 0 captures the leverage effect:
    negative shocks inflate variance more than positive shocks.
    """

    def fit_and_forecast(
        self,
        returns: pd.Series,
        ticker: str = "Asset",
        horizons: Optional[List[int]] = None,
    ) -> GARCHVolResult:
        """
        Fit GJR-GARCH and produce multi-step vol forecasts.

        Args:
            returns: daily return series (decimal, e.g. 0.01 = 1%)
            ticker: label for the result
            horizons: forecast horizons in trading days

        Returns:
            GARCHVolResult
        """
        if horizons is None:
            horizons = [1, 5, 21]
        r = returns.dropna()
        # Scale to % to improve numerical conditioning
        r_pct = r * 100

        log.info("[GJR-GARCH] Fitting on %s: %d obs", ticker, len(r_pct))

        model = arch_model(
            r_pct,
            vol="GARCH",
            p=1, o=1, q=1,    # o=1 → GJR asymmetric term
            mean="Constant",
            dist="normal",
        )
        res = model.fit(disp="off", show_warning=False)

        params = self._extract_params(res)

        # Current conditional volatility (in %)
        cond_vol_pct = float(res.conditional_volatility.iloc[-1])
        cond_vol_annual = cond_vol_pct * np.sqrt(252)

        regime = self._classify_regime(cond_vol_annual)

        # Multi-horizon forecasts
        forecasts = []
        for h in horizons:
            fc = res.forecast(horizon=h, reindex=False)
            # variance forecast is cumulative for h days
            h_var = float(fc.variance.iloc[-1, -1])    # variance in % units
            h_vol_daily = np.sqrt(h_var / h)           # per-day vol
            h_vol_annual = h_vol_daily * np.sqrt(252)
            var95 = float(1.645 * h_vol_daily)
            var99 = float(2.326 * h_vol_daily)
            forecasts.append(VolForecast(
                horizon_days=h,
                vol_pct_daily=round(h_vol_daily, 4),
                vol_pct_annual=round(h_vol_annual, 4),
                var_95_pct=round(var95, 4),
                var_99_pct=round(var99, 4),
                regime=self._classify_regime(h_vol_annual),
            ))

        # Volatility cone (historical percentiles of rolling 21d vol)
        cone = self._vol_cone(r_pct)

        hist_mean_vol = float(res.conditional_volatility.mean() * np.sqrt(252))

        return GARCHVolResult(
            ticker=ticker,
            model_type="GJR-GARCH(1,1)",
            params=params,
            current_vol_daily_pct=round(cond_vol_pct, 4),
            current_vol_annual_pct=round(cond_vol_annual, 4),
            current_regime=regime,
            forecasts=forecasts,
            vol_cone=cone,
            historical_mean_vol_annual=round(hist_mean_vol, 4),
        )

    @staticmethod
    def _extract_params(res) -> GARCHParams:
        p = res.params
        alpha = float(p.get("alpha[1]", 0))
        gamma = float(p.get("gamma[1]", 0))
        beta  = float(p.get("beta[1]", 0))
        persistence = alpha + gamma / 2 + beta
        return GARCHParams(
            omega=round(float(p.get("omega", 0)), 8),
            alpha=round(alpha, 6),
            gamma=round(gamma, 6),
            beta=round(beta, 6),
            persistence=round(persistence, 6),
            aic=round(float(res.aic), 4),
            bic=round(float(res.bic), 4),
            log_likelihood=round(float(res.loglikelihood), 4),
        )

    @staticmethod
    def _classify_regime(annual_vol_pct: float) -> str:
        for regime, (lo, hi) in REGIME_THRESHOLDS.items():
            if lo <= annual_vol_pct < hi:
                return regime
        return "crisis"

    @staticmethod
    def _vol_cone(r_pct: pd.Series, window: int = 21) -> Dict[str, float]:
        """
        Rolling 21-day realised vol (annualised %) — historical percentiles.
        This forms the 'volatility cone' used on Bloomberg vol surfaces.
        """
        rolling_vol = r_pct.rolling(window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) == 0:
            return {}
        return {
            "p10": round(float(rolling_vol.quantile(0.10)), 4),
            "p25": round(float(rolling_vol.quantile(0.25)), 4),
            "p50": round(float(rolling_vol.quantile(0.50)), 4),
            "p75": round(float(rolling_vol.quantile(0.75)), 4),
            "p90": round(float(rolling_vol.quantile(0.90)), 4),
            "current": round(float(rolling_vol.iloc[-1]), 4),
            "mean":    round(float(rolling_vol.mean()), 4),
        }


# ── NIFTY-specific convenience wrapper ───────────────────────────────────────
def compute_nifty_vol(
    prices: Optional[pd.Series] = None,
    lookback_years: int = 3,
) -> GARCHVolResult:
    """
    Fetch NIFTY 50 prices (yfinance) and run GJR-GARCH.
    If prices are provided, uses those instead of fetching.
    """
    if prices is None:
        import yfinance as yf
        from datetime import timedelta
        end = datetime.now()
        start = end - timedelta(days=lookback_years * 365)
        df = yf.download("^NSEI", start=start, end=end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        prices = df["Close"].squeeze()

    returns = prices.pct_change().dropna()
    engine = GARCHVolEngine()
    return engine.fit_and_forecast(returns, ticker="^NSEI (NIFTY 50)")
