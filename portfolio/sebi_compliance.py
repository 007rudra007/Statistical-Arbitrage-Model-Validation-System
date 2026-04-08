"""
Phase 5 – Step 21: SEBI Portfolio-Level Compliance Engine
===========================================================
Enforces SEBI LODR, SEBI circular SEBI/HO/MRD/DRMNP/CIR/P/2020/234,
and Aladdin-internal guidelines for portfolio-level compliance.

Rules (portfolio-level, separate from Phase 4 trade-level rules):
  P01  Single-stock weight cap                 (>15% → violation)
  P02  Sector concentration cap                (>35% → violation)
  P03  Portfolio beta ceiling                  (>1.0 → violation)
  P04  Cash floor — must maintain ≥2% cash     (<2% → violation)
  P05  Leverage ceiling                        (>1.0× → violation)
  P06  Min diversification (Effective N)       (<4 stocks → violation)
  P07  Max turnover per day                    (>5% NAV/day → violation)
  P08  Drawdown circuit-breaker               (>15% MDD → position halt)
  P09  F&O position as % of NAV               (>20% → violation)
  P10  Related-party / UPSI flag              (blocked tickers must = 0)

Each check returns: passed (bool), rule_id, severity, message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class ComplianceConfig:
    max_single_stock_pct:  float = 15.0   # P01
    max_sector_pct:        float = 35.0   # P02
    max_portfolio_beta:    float = 1.00   # P03
    min_cash_pct:          float = 2.0    # P04
    max_leverage:          float = 1.00   # P05
    min_effective_n:       float = 4.0    # P06
    max_daily_turnover_pct: float = 5.0   # P07
    max_drawdown_halt_pct: float = 15.0   # P08  (trigger circuit breaker)
    max_fno_pct:           float = 20.0   # P09
    blocked_tickers:       Set[str] = field(default_factory=set)   # P10 UPSI list


@dataclass
class ComplianceViolation:
    rule_id:   str
    severity:  str        # "ERROR" | "WARNING"
    message:   str
    value:     float
    threshold: float
    ticker:    Optional[str] = None


@dataclass
class ComplianceReport:
    passed:          bool
    violations:      List[ComplianceViolation]
    checks_run:      int
    portfolio_value: float
    checked_at:      str
    halt_trading:    bool        # True if circuit breaker triggered
    summary:         str

    @property
    def errors(self):
        return [v for v in self.violations if v.severity == "ERROR"]

    @property
    def warnings(self):
        return [v for v in self.violations if v.severity == "WARNING"]


# ── Portfolio snapshot ────────────────────────────────────────────────────────
@dataclass
class PortfolioSnapshot:
    """Current portfolio state — fed to compliance engine."""
    weights:          Dict[str, float]   # ticker → weight (0-1)
    sectors:          Dict[str, str]     # ticker → sector name
    betas:            Dict[str, float]   # ticker → beta
    fno_tickers:      Set[str]           # F&O position tickers
    cash_pct:         float              # cash as % of NAV
    leverage:         float              # gross exposure / NAV
    portfolio_value:  float              # total NAV in INR
    current_drawdown_pct: float          # current drawdown from peak (%)
    daily_turnover_pct:   float = 0.0   # today's turnover / NAV (%)


# ── Engine ────────────────────────────────────────────────────────────────────
class SEBIComplianceEngine:
    """
    Portfolio-level SEBI compliance checker (10 rules, P01–P10).
    Runs daily before market open and after every rebalance.
    """

    def __init__(self, config: Optional[ComplianceConfig] = None):
        self._cfg = config or ComplianceConfig()
        log.info("[SEBICompliance] Initialised with %d blocked tickers",
                 len(self._cfg.blocked_tickers))

    def check(self, portfolio: PortfolioSnapshot) -> ComplianceReport:
        violations: List[ComplianceViolation] = []
        halt = False

        # P01 – Single-stock weight cap
        for ticker, w in portfolio.weights.items():
            if w * 100 > self._cfg.max_single_stock_pct:
                violations.append(ComplianceViolation(
                    rule_id="P01_SINGLE_STOCK", severity="ERROR",
                    message=f"{ticker}: {w*100:.1f}% > {self._cfg.max_single_stock_pct}% cap",
                    value=w * 100, threshold=self._cfg.max_single_stock_pct, ticker=ticker,
                ))

        # P02 – Sector concentration cap
        sector_w: Dict[str, float] = {}
        for t, w in portfolio.weights.items():
            sec = portfolio.sectors.get(t, "Unknown")
            sector_w[sec] = sector_w.get(sec, 0.0) + w
        for sec, sw in sector_w.items():
            if sw * 100 > self._cfg.max_sector_pct:
                violations.append(ComplianceViolation(
                    rule_id="P02_SECTOR_CONC", severity="ERROR",
                    message=f"Sector '{sec}': {sw*100:.1f}% > {self._cfg.max_sector_pct}%",
                    value=sw * 100, threshold=self._cfg.max_sector_pct,
                ))

        # P03 – Portfolio beta ceiling
        port_beta = sum(portfolio.betas.get(t, 1.0) * w
                        for t, w in portfolio.weights.items())
        if port_beta > self._cfg.max_portfolio_beta:
            violations.append(ComplianceViolation(
                rule_id="P03_BETA", severity="WARNING",
                message=f"Portfolio beta {port_beta:.3f} > {self._cfg.max_portfolio_beta}",
                value=port_beta, threshold=self._cfg.max_portfolio_beta,
            ))

        # P04 – Cash floor
        if portfolio.cash_pct < self._cfg.min_cash_pct:
            violations.append(ComplianceViolation(
                rule_id="P04_CASH_FLOOR", severity="ERROR",
                message=f"Cash {portfolio.cash_pct:.1f}% < {self._cfg.min_cash_pct}% floor",
                value=portfolio.cash_pct, threshold=self._cfg.min_cash_pct,
            ))

        # P05 – Leverage ceiling
        if portfolio.leverage > self._cfg.max_leverage:
            violations.append(ComplianceViolation(
                rule_id="P05_LEVERAGE", severity="ERROR",
                message=f"Leverage {portfolio.leverage:.2f}× > {self._cfg.max_leverage}×",
                value=portfolio.leverage, threshold=self._cfg.max_leverage,
            ))

        # P06 – Diversification (Effective N)
        weights = list(portfolio.weights.values())
        eff_n   = 1.0 / sum(w**2 for w in weights) if weights else 0.0
        if eff_n < self._cfg.min_effective_n:
            violations.append(ComplianceViolation(
                rule_id="P06_DIVERSIFICATION", severity="WARNING",
                message=f"Effective N = {eff_n:.2f} < {self._cfg.min_effective_n} (undiversified)",
                value=eff_n, threshold=self._cfg.min_effective_n,
            ))

        # P07 – Daily turnover
        if portfolio.daily_turnover_pct > self._cfg.max_daily_turnover_pct:
            violations.append(ComplianceViolation(
                rule_id="P07_TURNOVER", severity="WARNING",
                message=f"Daily turnover {portfolio.daily_turnover_pct:.1f}% > {self._cfg.max_daily_turnover_pct}%",
                value=portfolio.daily_turnover_pct, threshold=self._cfg.max_daily_turnover_pct,
            ))

        # P08 – Drawdown circuit breaker
        if portfolio.current_drawdown_pct > self._cfg.max_drawdown_halt_pct:
            violations.append(ComplianceViolation(
                rule_id="P08_DRAWDOWN_HALT", severity="ERROR",
                message=f"Portfolio drawdown {portfolio.current_drawdown_pct:.1f}% > "
                        f"{self._cfg.max_drawdown_halt_pct}% — TRADING HALTED",
                value=portfolio.current_drawdown_pct,
                threshold=self._cfg.max_drawdown_halt_pct,
            ))
            halt = True

        # P09 – F&O concentration
        fno_weight = sum(portfolio.weights.get(t, 0) for t in portfolio.fno_tickers)
        if fno_weight * 100 > self._cfg.max_fno_pct:
            violations.append(ComplianceViolation(
                rule_id="P09_FNO", severity="WARNING",
                message=f"F&O weight {fno_weight*100:.1f}% > {self._cfg.max_fno_pct}%",
                value=fno_weight * 100, threshold=self._cfg.max_fno_pct,
            ))

        # P10 – Blocked / UPSI tickers
        for bt in self._cfg.blocked_tickers:
            if portfolio.weights.get(bt, 0) > 0:
                violations.append(ComplianceViolation(
                    rule_id="P10_BLOCKED_TICKER", severity="ERROR",
                    message=f"BLOCKED ticker {bt} has non-zero weight ({portfolio.weights[bt]*100:.2f}%)",
                    value=portfolio.weights.get(bt, 0) * 100,
                    threshold=0.0, ticker=bt,
                ))
                halt = True  # Hard stop on UPSI

        errors   = [v for v in violations if v.severity == "ERROR"]
        warnings = [v for v in violations if v.severity == "WARNING"]
        passed   = len(errors) == 0

        summary = (
            f"PASSED — {len(warnings)} warnings"
            if passed
            else f"FAILED — {len(errors)} errors, {len(warnings)} warnings"
        )
        if halt:
            summary += " | TRADING HALTED"

        log.info("[SEBICompliance] %s", summary)

        return ComplianceReport(
            passed=passed,
            violations=violations,
            checks_run=10,
            portfolio_value=portfolio.portfolio_value,
            checked_at=datetime.now(timezone.utc).isoformat(),
            halt_trading=halt,
            summary=summary,
        )

    def suggest_fixes(self, report: ComplianceReport,
                      portfolio: PortfolioSnapshot) -> List[str]:
        """Generate human-readable fix recommendations."""
        fixes = []
        for v in report.violations:
            if v.rule_id == "P01_SINGLE_STOCK":
                excess = v.value - v.threshold
                fixes.append(
                    f"Reduce {v.ticker} by {excess:.1f}% (sell "
                    f"₹{excess / 100 * portfolio.portfolio_value:,.0f})"
                )
            elif v.rule_id == "P02_SECTOR_CONC":
                fixes.append(f"Reduce sector concentration: {v.message}")
            elif v.rule_id == "P03_BETA":
                fixes.append(f"Add low-beta assets (FMCG/IT) to reduce portfolio beta below 1.0")
            elif v.rule_id == "P04_CASH_FLOOR":
                need = (v.threshold - v.value) / 100 * portfolio.portfolio_value
                fixes.append(f"Raise cash by ₹{need:,.0f} (sell some equity positions)")
            elif v.rule_id == "P05_LEVERAGE":
                fixes.append("De-leverage: close F&O positions or reduce margin usage")
            elif v.rule_id == "P06_DIVERSIFICATION":
                fixes.append(f"Add {int(v.threshold - v.value) + 1} more uncorrelated positions")
            elif v.rule_id == "P08_DRAWDOWN_HALT":
                fixes.append("CIRCUIT BREAKER: close all positions and await risk committee review")
            elif v.rule_id == "P09_FNO":
                fixes.append(f"Reduce F&O exposure below {v.threshold}% of NAV")
            elif v.rule_id == "P10_BLOCKED_TICKER":
                fixes.append(f"URGENT: exit {v.ticker} immediately — UPSI restriction active")
        return fixes
