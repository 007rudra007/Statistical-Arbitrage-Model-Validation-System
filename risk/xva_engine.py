"""
Phase 3 – Step 12: ORE/XVA Engine (QuantLib-ORE via Docker)
=============================================================
XVA = Credit Valuation Adjustment + Debt Valuation Adjustment + CVA
This module:
  1. Builds a simplified XVA calculator using QuantLib instruments
     (Pure Python path — no ORE binary needed for CVA/DVA estimates)
  2. Defines the Docker service interface to the full QuantLib-ORE engine
  3. Provides REST client to call the ORE container
  4. Includes IRS and FX Forward pricing for XVA computation

QuantLib-ORE Docker image: quant/ore (community build)

Local run:
    docker-compose up ore -d
    curl http://localhost:8080/xva/compute -d @payload.json

Full ORE docs: https://www.opensourcerisk.org/
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm

# Suppress QuantLib/scipy integration noise only.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="QuantLib")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
log = logging.getLogger(__name__)

# Try QuantLib for proper instrument pricing
try:
    import QuantLib as ql
    _QL_AVAILABLE = True
except ImportError:
    _QL_AVAILABLE = False


# ── Instrument Specs ──────────────────────────────────────────────────────────
@dataclass
class IRSwapSpec:
    """Interest Rate Swap specification (pay fixed / receive float)."""
    notional: float             # INR notional
    fixed_rate: float           # fixed coupon rate (decimal)
    tenor_years: int            # swap tenor in years
    pay_fixed: bool = True      # True = pay fixed, receive float
    floating_spread: float = 0.0
    currency: str = "INR"
    counterparty: str = "Bank_A"
    collateralised: bool = False


@dataclass
class FXForwardSpec:
    """FX Forward (USD/INR)."""
    notional_usd: float
    forward_rate: float     # agreed forward USD/INR
    tenor_days: int
    buy_usd: bool = True    # True = buy USD at forward_rate
    counterparty: str = "Bank_B"


# ── XVA Results ───────────────────────────────────────────────────────────────
@dataclass
class XVAResult:
    instrument_id: str
    instrument_type: str
    mtm: float              # Clean MTM value (INR)
    cva: float              # Credit Valuation Adjustment (loss from cpty default)
    dva: float              # Debt Valuation Adjustment (gain from own default)
    fva: float              # Funding Valuation Adjustment (simplified)
    xva_total: float        # CVA + DVA + FVA
    mtm_net_xva: float      # MTM - CVA + DVA - FVA
    pd_counterparty: float  # 1-yr PD of counterparty
    lgd: float              # Loss Given Default
    epe: float              # Expected Positive Exposure
    ene: float              # Expected Negative Exposure
    engine: str = "QuantLib_BSM"
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Simplified XVA Calculator ─────────────────────────────────────────────────
class XVAEngine:
    """
    Simplified CVA/DVA calculator using:
      CVA ≈ -LGD × PD_cpty × EPE
      DVA ≈ +LGD_self × PD_self × ENE

    For full XVA (including KVA, MVA, ColVA), use the ORE Docker service.
    """

    # CDS-implied PDs for common counterparty ratings (1-year, %)
    RATING_PD = {
        "AAA": 0.0005, "AA": 0.001, "A": 0.003, "BBB": 0.010,
        "BB":  0.030,  "B": 0.060, "CCC": 0.150, "D": 0.500,
    }
    OWN_PD = 0.002        # self 1-yr PD (assume BBB-equivalent)
    OWN_LGD = 0.40

    def compute_irs_xva(
        self,
        swap: IRSwapSpec,
        market_rate: float,     # current market fixed rate for same tenor
        yield_curve_rate: float = 0.065,
        counterparty_rating: str = "BBB",
        vol_rate: float = 0.012,    # rate vol for Monte Carlo EPE
        n_sims: int = 5000,
    ) -> XVAResult:
        """
        XVA for an Interest Rate Swap.

        MTM: discount_factor × (fixed_rate - market_rate) × notional × annuity
        EPE: Monte Carlo simulation of future positive exposure
        CVA: -LGD × PD × EPE
        """
        tenor = swap.tenor_years
        r = yield_curve_rate

        # Annuity factor (PV of 1 per year for tenor years)
        annuity = sum(math.exp(-r * t) for t in range(1, tenor + 1))

        # MTM of IRS (USD/INR swap — simplified)
        direction = 1 if swap.pay_fixed else -1
        mtm = direction * (market_rate - swap.fixed_rate) * swap.notional * annuity

        # Monte Carlo EPE / ENE
        epe, ene = self._mc_exposure(
            mtm, swap.notional, tenor, r, vol_rate, n_sims
        )

        pd_cpty = self.RATING_PD.get(counterparty_rating.upper(), 0.01)
        lgd = 0.40      # Basel standard LGD for uncollateralised

        cva = -lgd * pd_cpty * epe
        dva = self.OWN_LGD * self.OWN_PD * ene
        fva = -0.001 * epe if not swap.collateralised else 0.0   # simplified 10bps funding cost

        return XVAResult(
            instrument_id=f"IRS_{swap.counterparty}_{tenor}Y",
            instrument_type="InterestRateSwap",
            mtm=round(mtm, 2),
            cva=round(cva, 2),
            dva=round(dva, 2),
            fva=round(fva, 2),
            xva_total=round(cva + dva + fva, 2),
            mtm_net_xva=round(mtm + cva + dva + fva, 2),
            pd_counterparty=round(pd_cpty, 6),
            lgd=lgd,
            epe=round(epe, 2),
            ene=round(ene, 2),
        )

    def compute_fx_forward_xva(
        self,
        fwd: FXForwardSpec,
        spot_rate: float,           # current USD/INR
        yield_inr: float = 0.065,
        yield_usd: float = 0.045,
        vol_fx: float = 0.07,       # USD/INR vol ~7%
        counterparty_rating: str = "A",
        n_sims: int = 5000,
    ) -> XVAResult:
        """XVA for a USD/INR FX Forward."""
        T = fwd.tenor_days / 365.0

        # Fair forward rate
        fair_fwd = spot_rate * math.exp((yield_inr - yield_usd) * T)

        # MTM: PV of (agreed_rate - fair_fwd) × notional
        direction = 1 if fwd.buy_usd else -1
        mtm_usd = direction * (fwd.forward_rate - fair_fwd) * fwd.notional_usd
        mtm_inr = mtm_usd * spot_rate * math.exp(-yield_inr * T)

        # EPE / ENE via Black model
        epe, ene = self._fx_exposure(
            fwd.notional_usd * spot_rate, T, vol_fx, n_sims
        )

        pd_cpty = self.RATING_PD.get(counterparty_rating.upper(), 0.003)
        lgd = 0.40

        cva = -lgd * pd_cpty * epe
        dva =  self.OWN_LGD * self.OWN_PD * ene
        fva =  0.0

        return XVAResult(
            instrument_id=f"FXFWD_{fwd.counterparty}_{fwd.tenor_days}d",
            instrument_type="FXForward",
            mtm=round(mtm_inr, 2),
            cva=round(cva, 2),
            dva=round(dva, 2),
            fva=round(fva, 2),
            xva_total=round(cva + dva + fva, 2),
            mtm_net_xva=round(mtm_inr + cva + dva + fva, 2),
            pd_counterparty=round(pd_cpty, 6),
            lgd=lgd,
            epe=round(epe, 2),
            ene=round(ene, 2),
        )

    @staticmethod
    def _mc_exposure(
        current_mtm: float,
        notional: float,
        tenor_years: int,
        rate: float,
        vol: float,
        n_sims: int,
    ):
        """
        Monte Carlo simulation of future IRS exposure profile.
        Returns (EPE, ENE) in INR.
        """
        rng = np.random.default_rng(seed=42)
        steps = tenor_years * 4    # quarterly
        dt = tenor_years / steps
        mtm_paths = np.zeros((n_sims, steps))
        mtm_paths[:, 0] = current_mtm

        for t in range(1, steps):
            shock = rng.standard_normal(n_sims)
            mtm_paths[:, t] = (
                mtm_paths[:, t - 1] * math.exp(-0.1 * dt)  # mean reversion
                + notional * vol * shock * math.sqrt(dt)
            )

        exposures = np.maximum(mtm_paths, 0)
        neg_exp   = np.maximum(-mtm_paths, 0)
        epe = float(exposures.mean())
        ene = float(neg_exp.mean())
        return epe, ene

    @staticmethod
    def _fx_exposure(notional_inr: float, T: float, vol: float, n_sims: int):
        """Monte Carlo FX forward exposure."""
        rng = np.random.default_rng(seed=42)
        z = rng.standard_normal(n_sims)
        mtm_sims = notional_inr * (np.exp(vol * math.sqrt(T) * z - 0.5 * vol**2 * T) - 1)
        epe = float(np.maximum(mtm_sims, 0).mean())
        ene = float(np.maximum(-mtm_sims, 0).mean())
        return epe, ene


# ── ORE REST client (for Docker service) ─────────────────────────────────────
class OREClient:
    """
    REST client for the QuantLib-ORE Docker service.
    Used when full XVA (KVA, MVA, ColVA, SA-CCR) is needed.
    Falls back gracefully if ORE container is not running.
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict:
        try:
            import urllib.request
            req = urllib.request.urlopen(f"{self.base_url}/health", timeout=3)
            return {"status": "ok", "endpoint": self.base_url}
        except Exception as exc:
            return {"status": "unavailable", "endpoint": self.base_url, "error": str(exc)}

    def compute_xva(self, payload: Dict) -> Dict:
        """
        POST to ORE /xva/compute endpoint.
        Payload schema: {instruments: [...], netting_sets: [...], market_data: {...}}
        """
        import json, urllib.request
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/xva/compute",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            return json.loads(resp.read())
        except Exception as exc:
            log.warning("ORE service unavailable (%s) – using simplified XVA.", exc)
            return {"error": str(exc), "fallback": "simplified_xva"}
