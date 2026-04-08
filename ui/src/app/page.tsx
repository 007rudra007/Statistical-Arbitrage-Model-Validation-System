"use client";

import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8000';

// Types
interface OptimizeParams {
  objective: string;
  portfolio_value: number;
  max_weight: number;
  max_sector_weight: number;
  max_beta: number;
  lookback_days: number;
}

interface OptimizationResult {
  objective: string;
  solver_status: string;
  n_assets: number;
  effective_n: number;
  weights: Record<string, number>;
  metrics: {
    expected_annual_return: string;
    expected_annual_vol: string;
    sharpe_ratio: number;
    portfolio_beta: number;
  };
  sector_weights: Record<string, number>;
  rebalancing_trades: { ticker: string; direction: string; notional_inr: number; delta_weight: number }[];
  computed_at: string;
}

interface ComplianceReport {
  passed: boolean;
  halt_trading: boolean;
  summary: string;
  checks_run: number;
  errors: { rule: string; message: string }[];
  warnings: { rule: string; message: string }[];
  suggested_fixes: string[];
}

export default function PortfolioDashboard() {
  const [params, setParams] = useState<OptimizeParams>({
    objective: 'max_sharpe',
    portfolio_value: 10000000,
    max_weight: 0.15,
    max_sector_weight: 0.30,
    max_beta: 0.95,
    lookback_days: 252,
  });

  const [loading, setLoading] = useState(false);
  const [optResult, setOptResult] = useState<OptimizationResult | null>(null);
  const [compResult, setCompResult] = useState<ComplianceReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Aesthetic: Animated glow effect based on status
  const [statusColor, setStatusColor] = useState('rgba(0,191,255,0.4)');

  useEffect(() => {
    if (loading) {
      setStatusColor('rgba(255,176,0,0.6)');
    } else if (error) {
      setStatusColor('rgba(255,59,59,0.5)');
    } else if (compResult && !compResult.passed) {
      setStatusColor('rgba(255,59,59,0.5)');
    } else if (compResult && compResult.passed) {
      setStatusColor('rgba(0,255,136,0.6)');
    } else {
      setStatusColor('rgba(0,191,255,0.4)');
    }
  }, [loading, error, compResult]);

  const runPipeline = async () => {
    setLoading(true);
    setError(null);
    setOptResult(null);
    setCompResult(null);

    try {
      // 1. Optimize
      const optRes = await fetch(`${API}/portfolio/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          portfolio_value: params.portfolio_value,
          objective: params.objective,
          lookback_days: params.lookback_days,
          max_weight: params.max_weight,
          max_sector_weight: params.max_sector_weight,
          max_beta: params.max_beta
        })
      });

      if (!optRes.ok) {
        throw new Error(`Optimization failed: ${optRes.statusText}`);
      }

      const optData: OptimizationResult = await optRes.json();
      setOptResult(optData);

      // 2. Compliance
      // We will map the optimal weights back to compliance.
      // Assuming a generic sector/beta dictionary for the SEBI check, or backend handles it natively if we pass it blank.
      const compRes = await fetch(`${API}/portfolio/compliance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          weights: optData.weights,
          sectors: {}, // Mock default if the backend looks it up, otherwise backend allows empty
          betas: {},
          fno_tickers: [],
          cash_pct: 2.5,
          leverage: 1.0,
          portfolio_value: params.portfolio_value
        })
      });

      if (!compRes.ok) {
        throw new Error(`Compliance failed: ${compRes.statusText}`);
      }

      const compData: ComplianceReport = await compRes.json();
      setCompResult(compData);

    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pd-container">
      <div className="pd-bg-glow" style={{ background: `radial-gradient(circle at top right, ${statusColor}, transparent 40%)` }} />
      <div className="pd-bg-glow" style={{ background: `radial-gradient(circle at bottom left, rgba(0, 191, 255, 0.1), transparent 50%)` }} />

      <header className="pd-header">
        <h1>ALADDIN<span className="pd-accent">PORTFOLIO</span></h1>
        <div className="pd-status">
          <div className={`pd-dot ${loading ? 'pulsing' : ''}`} style={{ backgroundColor: statusColor.replace(/0\.[0-9]+\)/, '1)') }} />
          <span>{loading ? 'PROCESSING...' : 'SYSTEM READY'}</span>
        </div>
      </header>

      <main className="pd-main">
        {/* SIDEBAR: CONTROLS */}
        <aside className="pd-sidebar glass-panel">
          <h2>OPTIMIZATION PARAMETERS</h2>
          <div className="pd-form-group">
            <label>OBJECTIVE</label>
            <select
              value={params.objective}
              onChange={e => setParams(p => ({ ...p, objective: e.target.value }))}
            >
              <option value="max_sharpe">Maximum Sharpe</option>
              <option value="min_variance">Minimum Variance</option>
              <option value="risk_parity">Risk Parity</option>
            </select>
          </div>

          <div className="pd-form-group">
            <label>PORTFOLIO VALUE (INR)</label>
            <input
              type="number"
              value={params.portfolio_value}
              onChange={e => setParams(p => ({ ...p, portfolio_value: Number(e.target.value) }))}
            />
          </div>

          <div className="pd-form-group">
            <label>LOOKBACK DAYS</label>
            <input
              type="number"
              value={params.lookback_days}
              onChange={e => setParams(p => ({ ...p, lookback_days: Number(e.target.value) }))}
            />
          </div>

          <div className="pd-form-group">
            <label>MAX SINGLE WEIGHT</label>
            <input
              type="number"
              step="0.01"
              value={params.max_weight}
              onChange={e => setParams(p => ({ ...p, max_weight: Number(e.target.value) }))}
            />
          </div>

          <div className="pd-form-group">
            <label>MAX SECTOR WEIGHT</label>
            <input
              type="number"
              step="0.01"
              value={params.max_sector_weight}
              onChange={e => setParams(p => ({ ...p, max_sector_weight: Number(e.target.value) }))}
            />
          </div>

          <div className="pd-form-group">
            <label>MAX BETA</label>
            <input
              type="number"
              step="0.01"
              value={params.max_beta}
              onChange={e => setParams(p => ({ ...p, max_beta: Number(e.target.value) }))}
            />
          </div>

          <button className="pd-btn-run" onClick={runPipeline} disabled={loading}>
            {loading ? <span className="pd-spinner"></span> : 'EXECUTE PIPELINE'}
          </button>
        </aside>

        {/* CONTENT AREA */}
        <section className="pd-content">
          {error && (
            <div className="pd-alert pd-alert-error glass-panel">
              <strong>EXECUTION FAILED:</strong> {error}
            </div>
          )}

          {!optResult && !loading && !error && (
            <div className="pd-empty-state">
              <div className="pd-icon-grid"></div>
              <p>Configure parameters and execute pipeline to view portfolio analysis.</p>
            </div>
          )}

          {optResult && (
            <div className="pd-results-grid">
              
              {/* METRICS CARD */}
              <div className="pd-card glass-panel fade-in">
                <h3>METRICS</h3>
                <div className="pd-stats-grid">
                  <div className="pd-stat-box">
                    <span>SHARPE</span>
                    <strong>{optResult.metrics.sharpe_ratio}</strong>
                  </div>
                  <div className="pd-stat-box">
                    <span>EXP RETURN</span>
                    <strong>{optResult.metrics.expected_annual_return}</strong>
                  </div>
                  <div className="pd-stat-box">
                    <span>EXP VOLATILITY</span>
                    <strong>{optResult.metrics.expected_annual_vol}</strong>
                  </div>
                  <div className="pd-stat-box">
                    <span>PORTFOLIO β</span>
                    <strong>{optResult.metrics.portfolio_beta}</strong>
                  </div>
                </div>
              </div>

              {/* COMPLIANCE CARD */}
              {compResult && (
                <div className={`pd-card glass-panel fade-in delay-1 ${compResult.passed ? 'border-green' : 'border-red'}`}>
                  <h3>SEBI COMPLIANCE</h3>
                  <div className="pd-comp-summary">
                    {compResult.passed ? (
                      <strong className="text-green">PASS — ALL RULES CLEARED</strong>
                    ) : (
                      <strong className="text-red">FAIL — {compResult.errors.length} VIOLATIONS</strong>
                    )}
                    <span className="text-muted">{compResult.checks_run} CHECKS RUN</span>
                  </div>
                  
                  {compResult.errors.length > 0 && (
                    <div className="pd-violation-list">
                      {compResult.errors.map((e, idx) => (
                        <div key={idx} className="pd-comp-item pd-error">
                          <span className="pd-badge">ERROR</span> {e.message}
                        </div>
                      ))}
                    </div>
                  )}

                  {compResult.warnings.length > 0 && (
                    <div className="pd-violation-list">
                      {compResult.warnings.map((w, idx) => (
                        <div key={idx} className="pd-comp-item pd-warn">
                          <span className="pd-badge-warn">WARN</span> {w.message}
                        </div>
                      ))}
                    </div>
                  )}

                  {compResult.suggested_fixes && compResult.suggested_fixes.length > 0 && (
                    <div className="pd-fixes">
                      <strong>AI SUGGESTIONS:</strong>
                      <ul>
                        {compResult.suggested_fixes.map((fix, idx) => <li key={idx}>{fix}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* WEIGHTS TABLE */}
              <div className="pd-card glass-panel fade-in delay-2 pd-col-span-full">
                <h3>OPTIMIZED ALLOCATIONS</h3>
                <div className="pd-table-wrapper">
                  <table className="pd-table">
                    <thead>
                      <tr>
                        <th>TICKER</th>
                        <th>TARGET WEIGHT</th>
                        <th>ACTION</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(optResult.weights).map(([ticker, weight]) => (
                        <tr key={ticker}>
                          <td className="text-bright">{ticker}</td>
                          <td>{(weight * 100).toFixed(2)}%</td>
                          <td>
                            {weight > 0 ? (
                              <span className="pd-tag pd-tag-buy">ALLOCATE</span>
                            ) : (
                              <span className="pd-tag pd-tag-ignore">IGNORE</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

            </div>
          )}
        </section>
      </main>
    </div>
  );
}
