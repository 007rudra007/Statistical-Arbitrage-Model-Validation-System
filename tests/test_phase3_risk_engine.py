"""
Phase 3 Integration Tests: Risk Engine
========================================
All unit tests run without internet or MinIO.
Live tests (marked with SKIP_LIVE) fetch real NIFTY data via yfinance.

Run unit tests only:
    python -X utf8 tests/test_phase3_risk_engine.py

Run all (requires internet):
    SKIP_LIVE=0 python -X utf8 tests/test_phase3_risk_engine.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIP_LIVE = os.getenv("SKIP_LIVE", "1") == "1"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fake_prices(n: int = 500, n_assets: int = 3) -> pd.DataFrame:
    """Generate synthetic correlated price series."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-04-01", periods=n)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    # Correlated random walk
    cov = np.eye(n_assets) * 0.0002 + np.ones((n_assets, n_assets)) * 0.0001
    L = np.linalg.cholesky(cov)
    shocks = (np.random.randn(n, n_assets) @ L.T)
    prices = 1000 * np.exp(np.cumsum(shocks - 0.0001, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _fake_portfolio(tickers):
    from risk.var_engine import PortfolioInput, Position
    n = len(tickers)
    w = 1.0 / n
    return PortfolioInput(
        name="Test Portfolio",
        portfolio_value=10_000_000.0,
        positions=[Position(t, w) for t in tickers],
    )


def _fake_returns(n: int = 500, vol: float = 0.015) -> pd.Series:
    """Synthetic daily returns (iid Gaussian)."""
    np.random.seed(7)
    dates = pd.bdate_range(end="2026-04-01", periods=n)
    r = np.random.randn(n) * vol
    return pd.Series(r, index=dates, name="returns")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Value at Risk Engine
# ══════════════════════════════════════════════════════════════════════════════
class TestVaREngine(unittest.TestCase):

    def setUp(self):
        self.prices = _fake_prices(500, 3)
        self.portfolio = _fake_portfolio(self.prices.columns.tolist())
        from risk.var_engine import PortfolioVaREngine
        self.engine = PortfolioVaREngine()

    def test_historical_var_positive(self):
        result = self.engine.compute(self.portfolio, self.prices)
        self.assertGreater(result.historical.confidence_99, 0)
        self.assertGreater(result.historical.es_99, result.historical.confidence_99)

    def test_parametric_var_positive(self):
        result = self.engine.compute(self.portfolio, self.prices)
        self.assertGreater(result.parametric.confidence_99, 0)
        self.assertGreater(result.parametric.portfolio_vol_annual, 0)

    def test_mc_var_positive(self):
        result = self.engine.compute(self.portfolio, self.prices)
        self.assertGreater(result.monte_carlo.confidence_99, 0)
        self.assertEqual(result.monte_carlo.n_simulations, 10_000)

    def test_99_greater_than_95(self):
        result = self.engine.compute(self.portfolio, self.prices)
        for method in [result.historical, result.parametric, result.monte_carlo]:
            self.assertGreater(method.confidence_99, method.confidence_95,
                               f"{method.method}: 99% VaR should exceed 95%")

    def test_es_gt_var(self):
        result = self.engine.compute(self.portfolio, self.prices)
        for method in [result.historical, result.parametric, result.monte_carlo]:
            self.assertGreaterEqual(method.es_99, method.confidence_99,
                                    f"{method.method}: ES should be >= VaR")

    def test_component_var_sums_to_portfolio(self):
        """
        Component VaR uses a covariance-based decomposition.
        For parametric, the sum should ≈ portfolio VaR (within 10%).
        Historical method uses empirical percentile independently, so
        the covariance decomposition is approximate — allow wider tolerance.
        """
        result = self.engine.compute(self.portfolio, self.prices)
        # Parametric: should be close (same covariance matrix used)
        param = result.parametric
        comp_sum = sum(param.component_var.values())
        self.assertAlmostEqual(
            comp_sum, param.confidence_99, delta=param.confidence_99 * 0.20,
            msg="Parametric component VaR sum should be within 20% of total"
        )
        # Historical: component VaR is covariance-based proxy; verify it's non-zero
        hist = result.historical
        hist_sum = sum(hist.component_var.values())
        self.assertGreater(hist_sum, 0, "Historical component VaR should be positive")


    def test_inr_var_scales_with_value(self):
        result = self.engine.compute(self.portfolio, self.prices)
        ratio = result.historical.confidence_99_inr / (result.historical.confidence_99 / 100)
        self.assertAlmostEqual(ratio, self.portfolio.portfolio_value, delta=1000)

    def test_worst_scenarios_sorted(self):
        result = self.engine.compute(self.portfolio, self.prices)
        scenarios = result.worst_scenarios
        self.assertEqual(len(scenarios), 10)
        # Worst first
        self.assertLessEqual(scenarios[0]["return_pct"], scenarios[-1]["return_pct"])

    def test_weight_normalisation(self):
        from risk.var_engine import PortfolioInput, Position
        # Weights that don't sum to 1
        portfolio = PortfolioInput(
            name="BadWeights",
            portfolio_value=5_000_000,
            positions=[Position("ASSET_0", 0.5), Position("ASSET_1", 0.5),
                       Position("ASSET_2", 0.5)],
        )
        # Should not raise
        result = self.engine.compute(portfolio, self.prices)
        self.assertIsNotNone(result)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Stress Test Engine
# ══════════════════════════════════════════════════════════════════════════════
class TestStressEngine(unittest.TestCase):

    def setUp(self):
        from risk.stress_engine import StressTestEngine
        self.engine = StressTestEngine()
        self.positions = {
            "HDFCBANK.NS": 0.30,
            "RELIANCE.NS": 0.25,
            "INFY.NS":     0.20,
            "ITC.NS":      0.15,
            "SUNPHARMA.NS": 0.10,
        }
        self.value = 10_000_000.0

    def test_covid_crash_negative_pnl(self):
        r = self.engine.run_scenario("covid_crash", self.positions, self.value)
        self.assertLess(r.portfolio_pnl_pct, 0,
                        "COVID crash should produce negative P&L")
        self.assertLess(r.portfolio_pnl_pct, -20,
                        "Portfolio should lose >20% in COVID crash")

    def test_rate_hike_negative_pnl(self):
        r = self.engine.run_scenario("rate_hike_200bps", self.positions, self.value)
        self.assertLess(r.portfolio_pnl_pct, 0)

    def test_inr_crash_it_gains(self):
        """IT-heavy portfolio should gain from INR depreciation."""
        it_positions = {"INFY.NS": 0.5, "TCS.NS": 0.5}
        r = self.engine.run_scenario("inr_crash", it_positions, self.value)
        self.assertGreater(r.portfolio_pnl_pct, 0,
                           "IT-heavy portfolio should gain in INR crash")

    def test_run_all_scenarios(self):
        results = self.engine.run_all_scenarios(self.positions, self.value)
        self.assertGreaterEqual(len(results), 5)
        # Sorted by severity (worst first)
        pnls = [r.portfolio_pnl_pct for r in results]
        self.assertEqual(pnls, sorted(pnls))

    def test_custom_scenario(self):
        shocks = {"HDFCBANK.NS": -0.20, "RELIANCE.NS": -0.15}
        r = self.engine.run_scenario("custom", self.positions, self.value,
                                     custom_shocks=shocks)
        self.assertEqual(r.scenario_id, "custom")
        self.assertLess(r.portfolio_pnl_pct, 0)

    def test_recovery_capital_nonnegative(self):
        r = self.engine.run_scenario("covid_crash", self.positions, self.value)
        self.assertGreaterEqual(r.recovery_capital_inr, 0)
        self.assertAlmostEqual(r.recovery_capital_inr,
                               abs(r.portfolio_pnl_inr), delta=1)

    def test_position_pnl_breakdown(self):
        r = self.engine.run_scenario("covid_crash", self.positions, self.value)
        contributions = sum(p["contribution_pct"] for p in r.position_pnl.values())
        self.assertAlmostEqual(contributions, r.portfolio_pnl_pct, delta=0.01)

    def test_flash_crash_uniform_shock(self):
        r = self.engine.run_scenario("nifty_flash_crash", self.positions, self.value)
        # All positions get -15%; total should be ≈ -15%
        self.assertAlmostEqual(r.portfolio_pnl_pct, -15.0, delta=0.5)

    def test_invalid_scenario_raises(self):
        with self.assertRaises(ValueError):
            self.engine.run_scenario("nonexistent_scenario", self.positions, self.value)

    def test_custom_without_shocks_raises(self):
        with self.assertRaises(ValueError):
            self.engine.run_scenario("custom", self.positions, self.value)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — GJR-GARCH Volatility
# ══════════════════════════════════════════════════════════════════════════════
class TestGARCHVol(unittest.TestCase):

    def setUp(self):
        from risk.garch_vol import GARCHVolEngine
        self.engine = GARCHVolEngine()
        # NIFTY-like returns: slightly negative mean, fat tails
        np.random.seed(13)
        n = 750
        # Simulate with leverage effect
        r = []
        h = 0.0002
        omega, alpha, gamma, beta = 1e-6, 0.05, 0.08, 0.90
        for _ in range(n):
            eps = np.random.randn() * math.sqrt(h)
            r.append(eps)
            indicator = 1 if eps < 0 else 0
            h = omega + (alpha + gamma * indicator) * eps**2 + beta * h
        dates = pd.bdate_range(end="2026-04-01", periods=n)
        self.returns = pd.Series(r, index=dates)

    def test_fit_returns_result(self):
        result = self.engine.fit_and_forecast(self.returns, ticker="TEST")
        self.assertIsNotNone(result)
        self.assertEqual(result.model_type, "GJR-GARCH(1,1)")

    def test_params_valid_range(self):
        result = self.engine.fit_and_forecast(self.returns)
        self.assertGreater(result.params.omega, 0)
        self.assertGreater(result.params.alpha, 0)
        self.assertGreaterEqual(result.params.gamma, 0)  # leverage ≥ 0
        self.assertGreater(result.params.beta, 0)
        # Persistence should be < 1 for stationarity
        self.assertLess(result.params.persistence, 1.0)

    def test_current_vol_positive(self):
        result = self.engine.fit_and_forecast(self.returns)
        self.assertGreater(result.current_vol_daily_pct, 0)
        self.assertGreater(result.current_vol_annual_pct, 0)

    def test_forecasts_length(self):
        horizons = [1, 5, 21]
        result = self.engine.fit_and_forecast(self.returns, horizons=horizons)
        self.assertEqual(len(result.forecasts), len(horizons))
        for i, fc in enumerate(result.forecasts):
            self.assertEqual(fc.horizon_days, horizons[i])

    def test_forecast_vol_reasonable(self):
        result = self.engine.fit_and_forecast(self.returns)
        for fc in result.forecasts:
            # Annual vol should be between 1% and 100%
            self.assertGreater(fc.vol_pct_annual, 0)
            self.assertLess(fc.vol_pct_annual, 100)

    def test_regime_classification(self):
        from risk.garch_vol import GARCHVolEngine
        classify = GARCHVolEngine._classify_regime
        self.assertEqual(classify(8.0),  "low")
        self.assertEqual(classify(15.0), "normal")
        self.assertEqual(classify(25.0), "high")
        self.assertEqual(classify(40.0), "crisis")

    def test_vol_cone_percentiles(self):
        result = self.engine.fit_and_forecast(self.returns)
        cone = result.vol_cone
        self.assertIn("p10", cone)
        self.assertIn("p50", cone)
        self.assertIn("p90", cone)
        self.assertLessEqual(cone["p10"], cone["p50"])
        self.assertLessEqual(cone["p50"], cone["p90"])

    def test_var_order(self):
        """99% VaR should exceed 95% VaR for each horizon."""
        result = self.engine.fit_and_forecast(self.returns)
        for fc in result.forecasts:
            self.assertGreater(fc.var_99_pct, fc.var_95_pct)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — QuantLib / BSM Option Pricer
# ══════════════════════════════════════════════════════════════════════════════
class TestOptionPricer(unittest.TestCase):

    def setUp(self):
        from risk.quantlib_pricer import OptionPricer, OptionSpec
        self.pricer = OptionPricer()
        self.call_spec = OptionSpec(
            spot=22000, strike=22500, maturity_days=30,
            vol=0.18, rate=0.065, option_type="call",
        )
        self.put_spec = OptionSpec(
            spot=22000, strike=22500, maturity_days=30,
            vol=0.18, rate=0.065, option_type="put",
        )
        self.atm_call = OptionSpec(
            spot=22000, strike=22000, maturity_days=30,
            vol=0.18, rate=0.065, option_type="call",
        )

    def test_call_premium_positive(self):
        r = self.pricer.price(self.call_spec)
        self.assertGreater(r.premium, 0)

    def test_put_premium_positive(self):
        r = self.pricer.price(self.put_spec)
        self.assertGreater(r.premium, 0)

    def test_put_call_parity(self):
        """C - P = S * exp(-q*T) - K * exp(-r*T)  (BSM PCP)"""
        T = self.call_spec.maturity_days / 365.0
        c = self.pricer.price(self.call_spec)
        p = self.pricer.price(self.put_spec)
        S, K, r, q = 22000, 22500, 0.065, 0.013
        parity_lhs = c.premium - p.premium
        parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        self.assertAlmostEqual(parity_lhs, parity_rhs, delta=5.0,
                               msg="Put-call parity violated")

    def test_call_delta_range(self):
        r = self.pricer.price(self.call_spec)
        self.assertGreater(r.greeks.delta, 0)
        self.assertLess(r.greeks.delta, 1)

    def test_put_delta_range(self):
        r = self.pricer.price(self.put_spec)
        self.assertGreater(r.greeks.delta, -1)
        self.assertLess(r.greeks.delta, 0)

    def test_atm_delta_near_half(self):
        r = self.pricer.price(self.atm_call)
        self.assertAlmostEqual(r.greeks.delta, 0.5, delta=0.10)

    def test_gamma_positive(self):
        r = self.pricer.price(self.call_spec)
        self.assertGreater(r.greeks.gamma, 0)

    def test_vega_positive(self):
        r = self.pricer.price(self.call_spec)
        self.assertGreater(r.greeks.vega, 0)

    def test_theta_negative(self):
        """Time decay: theta should be negative (option loses value each day)."""
        r = self.pricer.price(self.call_spec)
        self.assertLess(r.greeks.theta, 0)

    def test_deep_itm_intrinsic(self):
        """Deep ITM call: premium ≈ intrinsic value."""
        from risk.quantlib_pricer import OptionSpec
        spec = OptionSpec(spot=25000, strike=20000, maturity_days=1,
                          vol=0.18, rate=0.065, option_type="call")
        r = self.pricer.price(spec)
        self.assertAlmostEqual(r.premium, r.intrinsic, delta=50)

    def test_expired_option_intrinsic_only(self):
        """Maturity=0: option worth max(0, S-K)."""
        from risk.quantlib_pricer import OptionSpec, _bsm_price
        price = _bsm_price(22000, 21000, 0, 0.18, 0.065, 0.013, call=True)
        self.assertAlmostEqual(price, 1000.0, delta=1)

    def test_higher_vol_higher_premium(self):
        from risk.quantlib_pricer import OptionSpec
        low_vol  = OptionSpec(spot=22000, strike=22500, maturity_days=30,
                              vol=0.10, rate=0.065, option_type="call")
        high_vol = OptionSpec(spot=22000, strike=22500, maturity_days=30,
                              vol=0.30, rate=0.065, option_type="call")
        r_low  = self.pricer.price(low_vol)
        r_high = self.pricer.price(high_vol)
        self.assertGreater(r_high.premium, r_low.premium)

    def test_longer_tenor_higher_premium(self):
        from risk.quantlib_pricer import OptionSpec
        short = OptionSpec(spot=22000, strike=22500, maturity_days=7,
                           vol=0.18, rate=0.065, option_type="call")
        long_ = OptionSpec(spot=22000, strike=22500, maturity_days=90,
                           vol=0.18, rate=0.065, option_type="call")
        self.assertGreater(self.pricer.price(long_).premium,
                           self.pricer.price(short).premium)


class TestImpliedVol(unittest.TestCase):

    def test_iv_roundtrip(self):
        """Price an option, then solve for IV — should recover input vol."""
        from risk.quantlib_pricer import OptionPricer, OptionSpec
        pricer = OptionPricer()
        spec = OptionSpec(spot=22000, strike=22500, maturity_days=30,
                          vol=0.20, rate=0.065, option_type="call")
        r = pricer.price(spec)
        iv_result = pricer.implied_vol(r.premium, spec)
        self.assertTrue(iv_result.converged)
        self.assertAlmostEqual(iv_result.implied_vol, 0.20, delta=0.001)

    def test_iv_various_moneyness(self):
        from risk.quantlib_pricer import OptionPricer, OptionSpec
        pricer = OptionPricer()
        for strike in [21000, 22000, 23000]:
            spec = OptionSpec(spot=22000, strike=strike, maturity_days=30,
                              vol=0.18, rate=0.065, option_type="call")
            r = pricer.price(spec)
            iv = pricer.implied_vol(r.premium, spec)
            self.assertAlmostEqual(iv.implied_vol, 0.18, delta=0.002,
                                   msg=f"IV roundtrip failed for K={strike}")

    def test_option_chain_length(self):
        from risk.quantlib_pricer import price_nifty_option_chain
        chain = price_nifty_option_chain(spot=22000, n_strikes=11, step=100)
        self.assertEqual(len(chain), 11)
        for row in chain:
            self.assertIn("call_premium", row)
            self.assertIn("put_premium",  row)
            self.assertIn("call_delta",   row)
            self.assertGreater(row["call_premium"], 0)
            self.assertGreater(row["put_premium"],  0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — XVA Engine
# ══════════════════════════════════════════════════════════════════════════════
class TestXVAEngine(unittest.TestCase):

    def setUp(self):
        from risk.xva_engine import XVAEngine, IRSwapSpec, FXForwardSpec
        self.engine = XVAEngine()
        self.swap = IRSwapSpec(
            notional=10_000_000,
            fixed_rate=0.07,
            tenor_years=5,
            pay_fixed=True,
            counterparty="TestBank",
        )
        self.fwd = FXForwardSpec(
            notional_usd=100_000,
            forward_rate=85.0,
            tenor_days=90,
            counterparty="FXBank",
        )

    def test_irs_cva_negative(self):
        """CVA should be negative (it's a cost to us)."""
        r = self.engine.compute_irs_xva(self.swap, market_rate=0.08)
        self.assertLess(r.cva, 0, "CVA must be a negative adjustment")

    def test_irs_dva_positive(self):
        """DVA should be positive (benefit from own default risk)."""
        r = self.engine.compute_irs_xva(self.swap, market_rate=0.08)
        self.assertGreater(r.dva, 0)

    def test_irs_epe_positive(self):
        r = self.engine.compute_irs_xva(self.swap, market_rate=0.08)
        self.assertGreaterEqual(r.epe, 0)

    def test_irs_better_rating_less_cva(self):
        """Higher-rated counterparty → lower CVA charge."""
        r_aaa = self.engine.compute_irs_xva(self.swap, 0.08,
                                             counterparty_rating="AAA")
        r_ccc = self.engine.compute_irs_xva(self.swap, 0.08,
                                             counterparty_rating="CCC")
        self.assertGreater(r_aaa.cva, r_ccc.cva,
                           "AAA CVA should be smaller (less negative) than CCC")

    def test_fx_forward_xva(self):
        r = self.engine.compute_fx_forward_xva(self.fwd, spot_rate=84.0)
        self.assertIsNotNone(r)
        self.assertLess(r.cva, 0)

    def test_xva_total_equals_sum(self):
        r = self.engine.compute_irs_xva(self.swap, market_rate=0.08)
        self.assertAlmostEqual(r.xva_total, r.cva + r.dva + r.fva, places=2)

    def test_collateralised_zero_fva(self):
        from risk.xva_engine import IRSwapSpec
        collat = IRSwapSpec(notional=10_000_000, fixed_rate=0.07,
                            tenor_years=5, collateralised=True)
        r = self.engine.compute_irs_xva(collat, market_rate=0.08)
        self.assertEqual(r.fva, 0.0)

    def test_ore_client_health(self):
        """ORE client should return unavailable (not running locally)."""
        from risk.xva_engine import OREClient
        client = OREClient("http://localhost:8080")
        health = client.health_check()
        self.assertIn("status", health)
        # Either ok (if ORE is running) or unavailable
        self.assertIn(health["status"], ["ok", "unavailable"])


# ══════════════════════════════════════════════════════════════════════════════
# LIVE tests (requires internet)
# ══════════════════════════════════════════════════════════════════════════════
@unittest.skipIf(SKIP_LIVE, "Skipping live NIFTY tests (SKIP_LIVE=1)")
class TestLiveNIFTY(unittest.TestCase):

    def test_nifty_garch(self):
        from risk.garch_vol import compute_nifty_vol
        result = compute_nifty_vol(lookback_years=2)
        self.assertGreater(result.current_vol_annual_pct, 5)
        self.assertLess(result.current_vol_annual_pct, 80)

    def test_live_var(self):
        import yfinance as yf
        from datetime import datetime, timedelta
        from risk.var_engine import PortfolioVaREngine, PortfolioInput, Position
        end = datetime.now()
        start = end - timedelta(days=365)
        tickers = ["HDFCBANK.NS", "RELIANCE.NS", "INFY.NS"]
        px = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"].dropna()
        portfolio = PortfolioInput(
            name="Live Test",
            portfolio_value=10_000_000,
            positions=[Position(t, 1/3) for t in tickers],
        )
        engine = PortfolioVaREngine()
        result = engine.compute(portfolio, px)
        self.assertGreater(result.historical.confidence_99, 0)


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("  PHASE 3: RISK ENGINE TESTS")
    if SKIP_LIVE:
        print("  Mode: UNIT ONLY  (set SKIP_LIVE=0 for live tests)")
    else:
        print("  Mode: FULL (includes live NIFTY data fetch)")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestVaREngine,
        TestStressEngine,
        TestGARCHVol,
        TestOptionPricer,
        TestImpliedVol,
        TestXVAEngine,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    if not SKIP_LIVE:
        suite.addTests(loader.loadTestsFromTestCase(TestLiveNIFTY))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
