# Risk Management Module
# ──────────────────────────────────────────────────────────
# Phase 1 (existing):
#   metrics.py    – Equity-curve VaR, drawdown, Sharpe, Calmar
#   tearsheet.py  – QuantStats HTML + custom matplotlib tearsheet
#
# Phase 3 (new):
#   var_engine.py       – Portfolio VaR: historical, parametric, MC 10k
#   stress_engine.py    – Aladdin-style stress scenarios (COVID, rates, oil…)
#   garch_vol.py        – GJR-GARCH volatility forecast + regime classification
#   quantlib_pricer.py  – QuantLib BSM option pricing + Greeks + IV solver
#   xva_engine.py       – CVA/DVA/FVA for IRS and FX Forwards

from risk.metrics import (
    max_drawdown, calmar_ratio, historical_var, generate_risk_report, print_risk_report
)
from risk.var_engine import (
    PortfolioVaREngine, PortfolioInput, Position, VaRResult, PortfolioVaROutput
)
from risk.stress_engine import StressTestEngine, StressResult, SCENARIOS
from risk.garch_vol import GARCHVolEngine, GARCHVolResult, compute_nifty_vol
from risk.quantlib_pricer import OptionPricer, OptionSpec, OptionResult
from risk.xva_engine import XVAEngine, XVAResult, IRSwapSpec, FXForwardSpec

__all__ = [
    # Phase 1
    "max_drawdown", "calmar_ratio", "historical_var",
    "generate_risk_report", "print_risk_report",
    # Phase 3 – VaR
    "PortfolioVaREngine", "PortfolioInput", "Position",
    "VaRResult", "PortfolioVaROutput",
    # Phase 3 – Stress
    "StressTestEngine", "StressResult", "SCENARIOS",
    # Phase 3 – Vol
    "GARCHVolEngine", "GARCHVolResult", "compute_nifty_vol",
    # Phase 3 – Options
    "OptionPricer", "OptionSpec", "OptionResult",
    # Phase 3 – XVA
    "XVAEngine", "XVAResult", "IRSwapSpec", "FXForwardSpec",
]
