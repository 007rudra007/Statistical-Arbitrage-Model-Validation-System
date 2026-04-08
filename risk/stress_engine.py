"""
Phase 3 – Step 10: Stress Test Engine
========================================
Aladdin-style scenario shocks applied to a portfolio.

Pre-built scenarios:
  - COVID Crash (Mar 2020):   Equity -30%, Credit spread +300bps, VIX ×3
  - Rate Hike +200bps:        Bond proxy -15%, Rate-sensitive equity -12%
  - Rate Hike +400bps:        Severe version of rate shock
  - Oil Spike +40%:           Energy +15%, Transport -10%, broader -5%
  - INR Depreciation -10%:    IT/Export +8%, Import-sensitive (oils) -5%
  - Nifty Flash Crash -15%:   Uniform equity -15% (1987-style)
  - Custom:                   User-defined sector/factor shocks

Each scenario maps ticker-level and factor-level shocks.
The engine also computes scenario P&L, VaR breach probability,
and recovery capital requirement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Scenario Definitions ──────────────────────────────────────────────────────
SCENARIOS: Dict[str, Dict] = {
    "covid_crash": {
        "name": "COVID-19 Market Crash (Mar 2020)",
        "description": "Replicates the 30% sell-off seen in Mar 2020. Equities crushed, volatility spikes, credit spreads blow out.",
        "factor_shocks": {
            "equity_broad": -0.30,
            "financials":   -0.35,
            "it":           -0.20,
            "fmcg":         -0.10,
            "pharma":       +0.05,
            "energy":       -0.40,
            "metals":       -0.35,
            "rate_10y_bps": +100,
            "usd_inr":      +0.08,
        },
        "ticker_overrides": {
            "HDFCBANK.NS":   -0.38, "ICICIBANK.NS":  -0.40, "KOTAKBANK.NS":  -0.32,
            "AXISBANK.NS":   -0.42, "SBIN.NS":       -0.45, "RELIANCE.NS":   -0.28,
            "INFY.NS":       -0.22, "TCS.NS":        -0.20, "WIPRO.NS":      -0.25,
            "HINDUNILVR.NS": -0.12, "ITC.NS":        -0.15, "SUNPHARMA.NS":  +0.04,
            "TITAN.NS":      -0.35, "MARUTI.NS":     -0.30, "BAJFINANCE.NS": -0.50,
            "LT.NS":         -0.30, "ASIANPAINT.NS": -0.25, "ULTRACEMCO.NS": -0.30,
            "BHARTIARTL.NS": -0.20, "LICI.NS":       -0.35,
            "^NSEI":         -0.30, "^NSEBANK":      -0.38,
        },
    },
    "rate_hike_200bps": {
        "name": "RBI Rate Hike +200 bps",
        "description": "Sudden 200bps rate increase causes bond proxies to sell off sharply. NBFCs and rate-sensitive names hit hard.",
        "factor_shocks": {
            "equity_broad": -0.08,
            "financials":   -0.12,
            "nbfc":         -0.18,
            "real_estate":  -0.20,
            "it":           -0.05,
            "utilities":    -0.15,
            "rate_10y_bps": +200,
            "usd_inr":      +0.02,
        },
        "ticker_overrides": {
            "HDFCBANK.NS":   -0.10, "ICICIBANK.NS":  -0.09, "KOTAKBANK.NS":  -0.08,
            "BAJFINANCE.NS": -0.18, "AXISBANK.NS":   -0.11, "SBIN.NS":       -0.07,
            "RELIANCE.NS":   -0.06, "INFY.NS":       -0.04, "TCS.NS":        -0.04,
            "HINDUNILVR.NS": -0.06, "ITC.NS":        -0.05, "SUNPHARMA.NS":  -0.04,
            "TITAN.NS":      -0.12, "MARUTI.NS":     -0.08, "BHARTIARTL.NS": -0.06,
            "^NSEI":         -0.08, "^NSEBANK":      -0.12,
        },
    },
    "rate_hike_400bps": {
        "name": "Severe Rate Shock +400 bps (Volcker-style)",
        "description": "Extreme monetary tightening, similar to 2022 US Fed cycle but applied to India.",
        "factor_shocks": {
            "equity_broad": -0.20,
            "financials":   -0.25,
            "nbfc":         -0.35,
            "real_estate":  -0.40,
            "rate_10y_bps": +400,
            "usd_inr":      +0.05,
        },
        "ticker_overrides": {
            "HDFCBANK.NS":   -0.22, "ICICIBANK.NS":  -0.20, "KOTAKBANK.NS":  -0.18,
            "BAJFINANCE.NS": -0.38, "AXISBANK.NS":   -0.24, "SBIN.NS":       -0.18,
            "RELIANCE.NS":   -0.14, "INFY.NS":       -0.10, "TCS.NS":        -0.10,
            "HINDUNILVR.NS": -0.12, "TITAN.NS":      -0.25, "MARUTI.NS":     -0.18,
            "^NSEI":         -0.20, "^NSEBANK":      -0.25,
        },
    },
    "oil_spike": {
        "name": "Oil Price Spike +40%",
        "description": "Supply shock drives Brent crude up 40%. India is a net importer: current account widens, INR weakens.",
        "factor_shocks": {
            "equity_broad": -0.05,
            "energy":       +0.15,
            "transport":    -0.10,
            "paints":       -0.15,
            "fmcg":         -0.06,
            "usd_inr":      +0.04,
            "rate_10y_bps": +50,
        },
        "ticker_overrides": {
            "RELIANCE.NS":   +0.10, "HDFCBANK.NS":   -0.05, "ICICIBANK.NS":  -0.04,
            "ASIANPAINT.NS": -0.18, "HINDUNILVR.NS": -0.07, "ITC.NS":        -0.04,
            "MARUTI.NS":     -0.12, "TITAN.NS":      -0.06, "INFY.NS":       +0.02,
            "TCS.NS":        +0.02, "BHARTIARTL.NS": -0.04,
            "^NSEI":         -0.05, "^NSEBANK":      -0.06,
        },
    },
    "inr_crash": {
        "name": "INR Depreciation -10% (EM Crisis)",
        "description": "Emerging market sell-off. Capital flight causes INR to depreciate 10% vs USD.",
        "factor_shocks": {
            "equity_broad": -0.08,
            "it":           +0.08,
            "pharma":       +0.05,
            "energy":       -0.05,
            "financials":   -0.10,
            "rate_10y_bps": +150,
            "usd_inr":      +0.10,
        },
        "ticker_overrides": {
            "INFY.NS":       +0.08, "TCS.NS":        +0.09, "WIPRO.NS":      +0.07,
            "SUNPHARMA.NS":  +0.05, "RELIANCE.NS":   -0.04, "HDFCBANK.NS":   -0.12,
            "ICICIBANK.NS":  -0.11, "BAJFINANCE.NS": -0.14, "MARUTI.NS":     -0.08,
            "^NSEI":         -0.08, "^NSEBANK":      -0.12,
        },
    },
    "nifty_flash_crash": {
        "name": "Nifty Flash Crash -15%",
        "description": "Rapid 15% intraday sell-off across all sectors. Liquidity dries up.",
        "factor_shocks": {
            "equity_broad": -0.15,
            "rate_10y_bps": +30,
        },
        "ticker_overrides": {},   # uniform shock applied below
        "uniform_shock": -0.15,
    },
    "geopolitical_shock": {
        "name": "Geopolitical Conflict / War Premium",
        "description": "Regional conflict triggers risk-off. Defence + commodities gain; financials and consumption fall.",
        "factor_shocks": {
            "equity_broad": -0.12,
            "energy":       +0.20,
            "defence":      +0.15,
            "financials":   -0.15,
            "it":           -0.05,
            "rate_10y_bps": +80,
            "usd_inr":      +0.06,
        },
        "ticker_overrides": {
            "RELIANCE.NS":   +0.08, "HDFCBANK.NS":   -0.14, "INFY.NS":       -0.05,
            "TCS.NS":        -0.05, "BAJFINANCE.NS": -0.18, "HINDUNILVR.NS": -0.08,
            "BHARTIARTL.NS": -0.08, "TITAN.NS":      -0.15,
            "^NSEI":         -0.12, "^NSEBANK":      -0.16,
        },
    },
}

# SEBI sector classifications for factor-to-ticker mapping
SECTOR_MAP: Dict[str, List[str]] = {
    "financials": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
                   "SBIN.NS", "BANKBARODA.NS", "INDUSINDBK.NS", "FEDERALBNK.NS",
                   "IDFCFIRSTB.NS", "BANDHANBNK.NS", "PNB.NS", "AUBANK.NS",
                   "^NSEBANK"],
    "nbfc":       ["BAJFINANCE.NS"],
    "it":         ["INFY.NS", "TCS.NS", "WIPRO.NS"],
    "fmcg":       ["HINDUNILVR.NS", "ITC.NS"],
    "pharma":     ["SUNPHARMA.NS"],
    "energy":     ["RELIANCE.NS"],
    "metals":     ["TATASTEEL.NS", "HINDALCO.NS"],
    "paints":     ["ASIANPAINT.NS"],
    "auto":       ["MARUTI.NS"],
    "telecom":    ["BHARTIARTL.NS"],
    "insurance":  ["LICI.NS"],
    "consumer":   ["TITAN.NS"],
    "cement":     ["ULTRACEMCO.NS"],
    "infra":      ["LT.NS"],
    "equity_broad": ["^NSEI"],
}


# ── Stress Engine ─────────────────────────────────────────────────────────────
@dataclass
class StressResult:
    scenario_id: str
    scenario_name: str
    description: str
    portfolio_pnl_pct: float
    portfolio_pnl_inr: float
    position_pnl: Dict[str, Dict]
    factor_shocks_applied: Dict[str, float]
    recovery_capital_inr: float     # capital needed to recover to starting NAV
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class StressTestEngine:
    """
    Apply pre-built or custom scenario shocks to a portfolio.

    Algorithm:
      1. For each position, find ticker-specific shock (override)
      2. If no override, use sector-based factor shock
      3. If no sector, apply equity_broad shock
      4. Aggregate to portfolio P&L
    """

    def run_scenario(
        self,
        scenario_id: str,
        positions: Dict[str, float],    # {ticker: weight}
        portfolio_value: float,
        custom_shocks: Optional[Dict[str, float]] = None,
    ) -> StressResult:
        """
        Run a single named scenario.

        Args:
            scenario_id: key from SCENARIOS dict, or 'custom'
            positions: {ticker: portfolio_weight}
            portfolio_value: total NAV in INR
            custom_shocks: for 'custom' scenario — {ticker_or_factor: shock}

        Returns:
            StressResult
        """
        if scenario_id == "custom":
            if not custom_shocks:
                raise ValueError("custom_shocks required for custom scenario")
            scenario = {
                "name": "Custom Scenario",
                "description": "User-defined shocks",
                "factor_shocks": custom_shocks,
                "ticker_overrides": custom_shocks,
            }
        else:
            if scenario_id not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario_id}. "
                                 f"Valid: {list(SCENARIOS.keys())}")
            scenario = SCENARIOS[scenario_id]

        position_pnl: Dict[str, Dict] = {}
        total_pnl_pct = 0.0

        for ticker, weight in positions.items():
            shock = self._resolve_shock(ticker, scenario)
            pnl_pct = shock * weight            # contribution to portfolio
            pnl_inr = shock * weight * portfolio_value
            position_pnl[ticker] = {
                "weight": round(weight * 100, 2),
                "shock_pct": round(shock * 100, 2),
                "contribution_pct": round(pnl_pct * 100, 4),
                "pnl_inr": round(pnl_inr, 2),
            }
            total_pnl_pct += pnl_pct

        total_pnl_inr = total_pnl_pct * portfolio_value
        recovery = abs(total_pnl_inr) if total_pnl_inr < 0 else 0.0

        return StressResult(
            scenario_id=scenario_id,
            scenario_name=scenario["name"],
            description=scenario["description"],
            portfolio_pnl_pct=round(total_pnl_pct * 100, 4),
            portfolio_pnl_inr=round(total_pnl_inr, 2),
            position_pnl=position_pnl,
            factor_shocks_applied=scenario.get("factor_shocks", {}),
            recovery_capital_inr=round(recovery, 2),
        )

    def run_all_scenarios(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
    ) -> List[StressResult]:
        """Run all pre-built scenarios and return sorted by severity."""
        results = []
        for sid in SCENARIOS:
            try:
                r = self.run_scenario(sid, positions, portfolio_value)
                results.append(r)
            except Exception as exc:
                log.warning("Scenario %s failed: %s", sid, exc)
        results.sort(key=lambda r: r.portfolio_pnl_pct)
        return results

    @staticmethod
    def _resolve_shock(ticker: str, scenario: Dict) -> float:
        """Priority: ticker override > sector factor > equity_broad > 0."""
        # 1. Uniform shock (flash crash)
        if "uniform_shock" in scenario and ticker not in scenario.get("ticker_overrides", {}):
            return scenario["uniform_shock"]

        # 2. Ticker-specific override
        overrides = scenario.get("ticker_overrides", {})
        if ticker in overrides:
            return float(overrides[ticker])

        # 3. Sector factor
        factors = scenario.get("factor_shocks", {})
        for sector, tickers in SECTOR_MAP.items():
            if ticker in tickers and sector in factors:
                return float(factors[sector])

        # 4. Broad equity default
        return float(factors.get("equity_broad", 0.0))
