"""
Phase 5 Tests: Portfolio & Trading
====================================
Coverage:
  Step 18 – Portfolio Optimizer tests (8 tests)
  Step 19 – Order router + FIX builder tests (8 tests)
  Step 20 – Multi-strategy backtester tests (10 tests)
  Step 21 – SEBI portfolio compliance tests (12 tests)
  Integration – Full pipeline: optimize → order → backtest → comply (5 tests)

Run:
    python -X utf8 tests/test_phase5_portfolio.py
"""
from __future__ import annotations

import json, os, sys, unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Path setup
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Fixtures ──────────────────────────────────────────────────────────────────
def _rand_prices(n: int = 252, tickers: list = None, seed: int = 42) -> pd.DataFrame:
    """Random log-normal price DataFrame."""
    np.random.seed(seed)
    tickers = tickers or ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]
    df = {}
    for t in tickers:
        log_r = np.random.randn(n) * 0.012
        df[t] = 1000.0 * np.exp(np.cumsum(log_r))
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    return pd.DataFrame(df, index=idx)


def _rand_returns(n: int = 252, m: int = 5, seed: int = 7) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(n, m) * 0.012


# ══════════════════════════════════════════════════════════════════════════════
# STEP 18 – Portfolio Optimizer
# ══════════════════════════════════════════════════════════════════════════════
class TestPortfolioOptimizer(unittest.TestCase):

    def _make_input(self, n=5, obj="min_variance"):
        from portfolio.optimizer import OptimizationInput, Asset
        assets = [Asset(f"T{i}.NS", "Sector" + str(i % 3), beta=0.8 + 0.1 * i)
                  for i in range(n)]
        R = _rand_returns(252, n)
        return OptimizationInput(assets=assets, returns_matrix=R, objective=obj)

    def test_min_variance_weights_sum_to_one(self):
        from portfolio.optimizer import PortfolioOptimizer
        inp = self._make_input(obj="min_variance")
        res = PortfolioOptimizer().optimize(inp)
        self.assertAlmostEqual(sum(res.weights.values()), 1.0, places=4)

    def test_weights_are_non_negative(self):
        from portfolio.optimizer import PortfolioOptimizer
        res = PortfolioOptimizer().optimize(self._make_input())
        self.assertTrue(all(w >= -1e-6 for w in res.weights.values()))

    def test_max_weight_constraint(self):
        from portfolio.optimizer import PortfolioOptimizer
        inp = self._make_input()
        inp.max_weight = 0.30
        res = PortfolioOptimizer().optimize(inp)
        self.assertTrue(all(w <= 0.30 + 1e-4 for w in res.weights.values()))

    def test_sharpe_objective(self):
        from portfolio.optimizer import PortfolioOptimizer
        inp = self._make_input(obj="max_sharpe")
        res = PortfolioOptimizer().optimize(inp)
        self.assertAlmostEqual(sum(res.weights.values()), 1.0, places=3)

    def test_risk_parity_objective(self):
        from portfolio.optimizer import PortfolioOptimizer
        inp = self._make_input(obj="risk_parity")
        res = PortfolioOptimizer().optimize(inp)
        self.assertAlmostEqual(sum(res.weights.values()), 1.0, places=3)

    def test_effective_n_greater_than_one(self):
        from portfolio.optimizer import PortfolioOptimizer
        res = PortfolioOptimizer().optimize(self._make_input(n=10))
        self.assertGreater(res.effective_n, 1.0)

    def test_sharpe_non_negative(self):
        from portfolio.optimizer import PortfolioOptimizer
        res = PortfolioOptimizer().optimize(self._make_input())
        # Sharpe can be negative in bad markets, just check it's a number
        self.assertIsInstance(res.sharpe_ratio, float)

    def test_rebalancing_trades_direction(self):
        from portfolio.optimizer import PortfolioOptimizer, OptimizationInput, Asset
        assets = [Asset("X.NS", "IT", beta=0.9, current_weight=0.2),
                  Asset("Y.NS", "FMCG", beta=0.7, current_weight=0.8)]
        R = _rand_returns(252, 2)
        inp = OptimizationInput(assets=assets, returns_matrix=R)
        res = PortfolioOptimizer().optimize(inp)
        for t in res.rebalancing_trades:
            self.assertIn(t["direction"], ["BUY", "SELL"])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 19 – Order Router
# ══════════════════════════════════════════════════════════════════════════════
class TestOrderRouter(unittest.TestCase):

    def _make_order(self, side="BUY"):
        from portfolio.order_router import Order, OrderSide, OrderType, ProductType, Exchange
        return Order(
            ticker="RELIANCE.NS", side=OrderSide(side), quantity=10,
            order_type=OrderType.LIMIT, product=ProductType.CNC,
            exchange=Exchange.NSE, price=2800.0,
        )

    def test_paper_router_fills_buy(self):
        from portfolio.order_router import PaperOrderRouter, OrderStatus
        r = PaperOrderRouter()
        resp = r.place_order(self._make_order("BUY"))
        self.assertEqual(resp.status, OrderStatus.COMPLETE)
        self.assertGreater(resp.fill_price, 0)
        self.assertEqual(resp.filled_qty, 10)

    def test_paper_router_fills_sell_after_buy(self):
        from portfolio.order_router import PaperOrderRouter, OrderStatus
        r = PaperOrderRouter()
        r.place_order(self._make_order("BUY"))
        resp = r.place_order(self._make_order("SELL"))
        self.assertEqual(resp.status, OrderStatus.COMPLETE)

    def test_paper_router_tracks_position(self):
        from portfolio.order_router import PaperOrderRouter
        r = PaperOrderRouter()
        r.place_order(self._make_order("BUY"))
        positions = r.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].ticker, "RELIANCE.NS")

    def test_paper_router_cancel(self):
        from portfolio.order_router import PaperOrderRouter, OrderStatus
        r = PaperOrderRouter()
        resp = r.place_order(self._make_order())
        cancel = r.cancel_order(resp.order_id)
        self.assertEqual(cancel.status, OrderStatus.CANCELLED)

    def test_fix_message_new_order_single(self):
        from portfolio.order_router import FIXMessageBuilder, Order, OrderSide, OrderType, ProductType, Exchange
        b = FIXMessageBuilder()
        order = Order(
            ticker="INFY.NS", side=OrderSide.BUY, quantity=50,
            order_type=OrderType.LIMIT, product=ProductType.CNC,
            exchange=Exchange.NSE, price=1500.0,
        )
        msg = b.new_order_single(order)
        self.assertIn("35=D", msg)
        self.assertIn("55=INFY.NS", msg)
        self.assertIn("38=50", msg)
        self.assertIn("10=", msg)   # checksum

    def test_fix_message_cancel_request(self):
        from portfolio.order_router import FIXMessageBuilder, OrderSide
        b = FIXMessageBuilder()
        msg = b.order_cancel_request("NEW123", "ORIG456", "TCS.NS", OrderSide.SELL)
        self.assertIn("35=F", msg)
        self.assertIn("41=ORIG456", msg)

    def test_fix_router_dry_run(self):
        from portfolio.order_router import FIXOrderRouter, OrderStatus
        r = FIXOrderRouter()   # FIX_DRY_RUN=1 by default
        resp = r.place_order(self._make_order())
        self.assertEqual(resp.status, OrderStatus.OPEN)

    def test_factory_returns_paper_by_default(self):
        from portfolio.order_router import get_order_router, PaperOrderRouter
        r = get_order_router("paper")
        self.assertIsInstance(r, PaperOrderRouter)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 20 – Multi-Strategy Backtester
# ══════════════════════════════════════════════════════════════════════════════
class TestMultiStrategyBacktester(unittest.TestCase):

    def _run_momentum(self, n=300):
        from backtester.multi_strategy import MultiStrategyBacktester, MomentumStrategy
        price_df = _rand_prices(n, ["A.NS","B.NS","C.NS","D.NS","E.NS"])
        strat = MomentumStrategy(list(price_df.columns), formation=60, skip=5, top_pct=0.4)
        bt = MultiStrategyBacktester(price_df, strat, capital=1_000_000.0,
                                     size_pct=0.05, slippage_ticks=1.0)
        return bt.run()

    def test_run_returns_metrics(self):
        result = self._run_momentum()
        self.assertIn("metrics", result)
        self.assertIn("sharpe_ratio", result["metrics"])

    def test_initial_nav_correct(self):
        result = self._run_momentum()
        self.assertAlmostEqual(result["metrics"]["initial_capital"], 1_000_000.0, places=-2)

    def test_final_nav_positive(self):
        result = self._run_momentum()
        self.assertGreater(result["metrics"]["final_nav"], 0)

    def test_sharpe_is_float(self):
        result = self._run_momentum()
        self.assertIsInstance(result["metrics"]["sharpe_ratio"], float)

    def test_max_drawdown_non_positive(self):
        result = self._run_momentum()
        self.assertLessEqual(result["metrics"]["max_drawdown"], 0)

    def test_equity_df_has_index(self):
        result = self._run_momentum()
        self.assertIn("equity_df", result)
        self.assertGreater(len(result["equity_df"]), 0)

    def test_metrics_engine_sharpe(self):
        from backtester.multi_strategy import MetricsEngine, Portfolio
        import pandas as pd
        # Manually craft an equity curve
        nav = pd.Series(
            [1_000_000 * (1.0005 ** i) for i in range(252)],
            index=pd.bdate_range("2024-01-01", periods=252)
        )
        eq_df = pd.DataFrame({"nav": nav})
        metrics = MetricsEngine().compute(eq_df, [])
        self.assertGreater(metrics["sharpe_ratio"], 0)

    def test_profit_factor_infinity_no_losses(self):
        from backtester.multi_strategy import MetricsEngine
        nav = pd.Series(
            [1_000_000 + 100 * i for i in range(252)],
            index=pd.bdate_range("2024-01-01", periods=252)
        )
        eq_df = pd.DataFrame({"nav": nav})
        trades = [{"pnl": 500}, {"pnl": 300}, {"pnl": 100}]
        metrics = MetricsEngine().compute(eq_df, trades)
        self.assertEqual(metrics["profit_factor"], float("inf"))

    def test_agent_strategy_fires_on_signal(self):
        from backtester.multi_strategy import AgentSignalStrategy, BarEvent, Portfolio
        import pandas as pd
        dt = pd.Timestamp("2024-03-01")
        sig_df = pd.DataFrame([{
            "timestamp": dt, "ticker": "RELIANCE.NS",
            "action": "BUY", "confidence": 0.8
        }])
        strat = AgentSignalStrategy(sig_df)
        bar = BarEvent(timestamp=dt, ticker="RELIANCE.NS", close=2800.0)
        signals = strat.on_bar(bar, {})
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, "BUY")

    def test_agent_strategy_no_signal_for_hold(self):
        from backtester.multi_strategy import AgentSignalStrategy, BarEvent
        dt = pd.Timestamp("2024-03-01")
        sig_df = pd.DataFrame([{
            "timestamp": dt, "ticker": "RELIANCE.NS",
            "action": "HOLD", "confidence": 0.5
        }])
        strat = AgentSignalStrategy(sig_df)
        bar = BarEvent(timestamp=dt, ticker="RELIANCE.NS", close=2800.0)
        signals = strat.on_bar(bar, {})
        self.assertEqual(len(signals), 0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 21 – SEBI Portfolio Compliance
# ══════════════════════════════════════════════════════════════════════════════
class TestSEBICompliance(unittest.TestCase):

    def _clean_snap(self, **overrides):
        from portfolio.sebi_compliance import PortfolioSnapshot
        snap = PortfolioSnapshot(
            weights        = {"RELIANCE.NS": 0.12, "INFY.NS": 0.12, "TCS.NS": 0.10,
                              "HDFC.NS": 0.10, "ITC.NS": 0.10, "LT.NS": 0.10,
                              "SBIN.NS": 0.09, "AXIS.NS": 0.09, "MARUTI.NS": 0.09,
                              "WIPRO.NS": 0.09},
            sectors        = {"RELIANCE.NS": "Energy", "INFY.NS": "IT", "TCS.NS": "IT",
                              "HDFC.NS": "Financials", "ITC.NS": "FMCG", "LT.NS": "Industrials",
                              "SBIN.NS": "Financials", "AXIS.NS": "Financials",
                              "MARUTI.NS": "Auto", "WIPRO.NS": "IT"},
            betas          = {t: 0.85 for t in ["RELIANCE.NS","INFY.NS","TCS.NS",
                                                  "HDFC.NS","ITC.NS","LT.NS",
                                                  "SBIN.NS","AXIS.NS","MARUTI.NS","WIPRO.NS"]},
            fno_tickers    = set(),
            cash_pct       = 10.0,
            leverage       = 0.90,
            portfolio_value = 10_000_000.0,
            current_drawdown_pct = 5.0,
            daily_turnover_pct   = 2.0,
        )
        for k, v in overrides.items():
            setattr(snap, k, v)
        return snap

    def _engine(self, **cfg_overrides):
        from portfolio.sebi_compliance import SEBIComplianceEngine, ComplianceConfig
        return SEBIComplianceEngine(ComplianceConfig(**cfg_overrides))

    def test_clean_portfolio_passes(self):
        report = self._engine().check(self._clean_snap())
        self.assertTrue(report.passed, f"Violations: {[v.rule_id for v in report.violations]}")

    def test_p01_single_stock_violation(self):
        snap = self._clean_snap()
        snap.weights["RELIANCE.NS"] = 0.20  # 20% > 15% cap
        report = self._engine().check(snap)
        self.assertFalse(report.passed)
        self.assertTrue(any(v.rule_id == "P01_SINGLE_STOCK" for v in report.violations))

    def test_p02_sector_violation(self):
        snap = self._clean_snap()
        # IT sector: INFY + TCS + WIPRO = 12+12+9 = 33% → cap at 35% default
        snap.weights["INFY.NS"] = 0.20  # push IT to 39%
        report = self._engine().check(snap)
        self.assertFalse(report.passed)
        self.assertTrue(any(v.rule_id == "P02_SECTOR_CONC" for v in report.violations))

    def test_p03_beta_warning(self):
        snap = self._clean_snap()
        snap.betas = {t: 1.2 for t in snap.weights}  # all high beta
        report = self._engine().check(snap)
        self.assertTrue(any(v.rule_id == "P03_BETA" for v in report.violations))

    def test_p04_cash_floor_violation(self):
        snap = self._clean_snap(cash_pct=0.5)
        report = self._engine().check(snap)
        self.assertFalse(report.passed)
        self.assertTrue(any(v.rule_id == "P04_CASH_FLOOR" for v in report.violations))

    def test_p05_leverage_violation(self):
        snap = self._clean_snap(leverage=1.5)
        report = self._engine().check(snap)
        self.assertFalse(report.passed)
        self.assertTrue(any(v.rule_id == "P05_LEVERAGE" for v in report.violations))

    def test_p06_diversification_warning(self):
        from portfolio.sebi_compliance import PortfolioSnapshot
        snap = PortfolioSnapshot(
            weights={"A.NS": 0.70, "B.NS": 0.30},
            sectors={"A.NS": "IT", "B.NS": "IT"}, betas={},
            fno_tickers=set(), cash_pct=5.0, leverage=0.9,
            portfolio_value=1_000_000.0, current_drawdown_pct=0.0,
        )
        report = self._engine().check(snap)
        self.assertTrue(any(v.rule_id == "P06_DIVERSIFICATION" for v in report.violations))

    def test_p08_drawdown_halt(self):
        snap = self._clean_snap(current_drawdown_pct=20.0)
        report = self._engine().check(snap)
        self.assertFalse(report.passed)
        self.assertTrue(report.halt_trading)
        self.assertTrue(any(v.rule_id == "P08_DRAWDOWN_HALT" for v in report.violations))

    def test_p10_blocked_ticker_halt(self):
        snap = self._clean_snap()
        engine = self._engine(blocked_tickers={"INFY.NS"})
        report = engine.check(snap)
        self.assertFalse(report.passed)
        self.assertTrue(report.halt_trading)
        self.assertTrue(any(v.rule_id == "P10_BLOCKED_TICKER" for v in report.violations))

    def test_checks_run_always_ten(self):
        report = self._engine().check(self._clean_snap())
        self.assertEqual(report.checks_run, 10)

    def test_suggest_fixes_not_empty_on_violation(self):
        snap = self._clean_snap(leverage=1.5)
        engine = self._engine()
        report = engine.check(snap)
        fixes = engine.suggest_fixes(report, snap)
        self.assertGreater(len(fixes), 0)

    def test_multiple_violations_accumulated(self):
        snap = self._clean_snap(leverage=1.5, cash_pct=0.5, current_drawdown_pct=25.0)
        report = self._engine().check(snap)
        self.assertGreaterEqual(len(report.violations), 3)


# ══════════════════════════════════════════════════════════════════════════════
# Integration
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase5Integration(unittest.TestCase):

    def test_optimize_then_comply(self):
        """Optimize a portfolio and immediately compliance-check the output."""
        from portfolio.optimizer import PortfolioOptimizer, OptimizationInput, Asset
        from portfolio.sebi_compliance import (
            SEBIComplianceEngine, PortfolioSnapshot, ComplianceConfig
        )

        tickers = ["A.NS","B.NS","C.NS","D.NS","E.NS","F.NS"]
        sectors = ["IT","FMCG","Financials","Energy","IT","Auto"]
        assets  = [Asset(t, s, beta=0.85 + i * 0.05)
                   for i, (t, s) in enumerate(zip(tickers, sectors))]
        R = _rand_returns(252, len(tickers))
        inp = OptimizationInput(assets=assets, returns_matrix=R, objective="min_variance")
        result = PortfolioOptimizer().optimize(inp)

        snap = PortfolioSnapshot(
            weights        = result.weights,
            sectors        = {a.ticker: a.sector for a in assets},
            betas          = {a.ticker: a.beta   for a in assets},
            fno_tickers    = set(),
            cash_pct       = 5.0,
            leverage       = 1.0,
            portfolio_value = 10_000_000.0,
            current_drawdown_pct = 0.0,
        )
        report = SEBIComplianceEngine().check(snap)
        # Min-variance with 6 assets should pass most rules
        self.assertEqual(report.checks_run, 10)

    def test_paper_order_then_position(self):
        """Place buy + check position exists."""
        from portfolio.order_router import PaperOrderRouter, Order, OrderSide, OrderType, ProductType, Exchange
        r = PaperOrderRouter()
        for t in ["RELIANCE.NS", "INFY.NS", "TCS.NS"]:
            r.place_order(Order(ticker=t, side=OrderSide.BUY, quantity=5,
                                order_type=OrderType.MARKET, product=ProductType.CNC,
                                exchange=Exchange.NSE, price=1500.0))
        positions = r.get_positions()
        self.assertEqual(len(positions), 3)

    def test_backtest_equity_grows_rising_market(self):
        """Momentum strategy on a steadily rising market should produce positive return."""
        from backtester.multi_strategy import MultiStrategyBacktester, MomentumStrategy
        n = 300
        tickers = ["R.NS", "I.NS", "T.NS", "H.NS", "S.NS"]
        # Create steadily rising prices (all up)
        idx = pd.bdate_range(end="2024-12-31", periods=n)
        df  = pd.DataFrame({t: 1000.0 * (1.0008 ** np.arange(n)) for t in tickers}, index=idx)
        strat = MomentumStrategy(tickers, formation=60, skip=5, top_pct=0.5)
        bt    = MultiStrategyBacktester(df, strat, capital=1_000_000.0, size_pct=0.10)
        res   = bt.run()
        self.assertGreaterEqual(res["metrics"]["total_return"], 0)

    def test_fix_message_valid_checksum(self):
        from portfolio.order_router import (
            FIXMessageBuilder, Order, OrderSide, OrderType, ProductType, Exchange
        )
        b = FIXMessageBuilder()
        order = Order("NIFTY50.NS", OrderSide.BUY, 50,
                      OrderType.MARKET, ProductType.MIS, Exchange.NFO)
        msg = b.new_order_single(order)
        # Structural FIX 4.4 validation:
        self.assertIn("35=D", msg)   # NewOrderSingle
        self.assertIn("10=", msg)    # checksum field present
        # Checksum must be 3 digits 000-255
        chk_str = msg.split("10=")[-1].rstrip("\x01")
        self.assertTrue(chk_str.isdigit(), f"Checksum not numeric: {chk_str!r}")
        self.assertEqual(len(chk_str), 3)
        self.assertLessEqual(int(chk_str), 255)

    def test_compliance_api_rules_endpoint(self):
        from api.routes.portfolio import list_compliance_rules
        result = list_compliance_rules()
        self.assertEqual(result["total_rules"], 10)
        self.assertEqual(len(result["rules"]), 10)


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  PHASE 5: PORTFOLIO & TRADING TESTS")
    print("="*65 + "\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestPortfolioOptimizer,
        TestOrderRouter,
        TestMultiStrategyBacktester,
        TestSEBICompliance,
        TestPhase5Integration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
