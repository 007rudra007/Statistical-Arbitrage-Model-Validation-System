"""
Tasks 10-11: Risk Metrics
===========================
Task 10: Maximum Drawdown & Calmar Ratio
Task 11: Value at Risk (VaR) - 99% Historical

These metrics prove to capital allocators that your strategy is safe.
"""

import sys
import os
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# -----------------------------------------------------------------
# Task 10: Maximum Drawdown & Calmar Ratio
# -----------------------------------------------------------------
def max_drawdown(equity_curve: pd.Series) -> dict:
    """
    Calculate Maximum Drawdown from an equity curve.
    
    Maximum Drawdown = (Trough - Peak) / Peak
    This is the WORST peak-to-trough decline your strategy experienced.
    
    Returns dict with:
        - max_dd: Maximum drawdown as a negative percentage
        - peak_date: Date of the peak before the worst drawdown
        - trough_date: Date of the trough  
        - recovery_date: Date when equity recovered (or None if not recovered)
        - drawdown_series: Full drawdown time series
    """
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    
    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    
    # Find the peak date (when was the high before the trough?)
    peak_value_at_trough = peak.loc[trough_idx]
    peak_dates = equity_curve[equity_curve == peak_value_at_trough].index
    peak_date = peak_dates[peak_dates <= trough_idx][-1] if len(peak_dates[peak_dates <= trough_idx]) > 0 else None
    
    # Find recovery date (when did equity first exceed the previous peak?)
    post_trough = equity_curve.loc[trough_idx:]
    recovered = post_trough[post_trough >= peak_value_at_trough]
    recovery_date = recovered.index[0] if len(recovered) > 0 else None
    
    # Duration
    if peak_date and recovery_date:
        recovery_days = (recovery_date - peak_date).days
    else:
        recovery_days = None
    
    return {
        'max_dd': max_dd * 100,  # as percentage
        'peak_date': peak_date,
        'trough_date': trough_idx,
        'recovery_date': recovery_date,
        'recovery_days': recovery_days,
        'drawdown_series': drawdown * 100
    }


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar Ratio.
    
    Calmar Ratio = Annualized Return / |Maximum Drawdown|
    
    A Calmar > 1 is generally considered good.
    A Calmar > 3 is excellent.
    Institutional allocators often require Calmar > 0.5.
    """
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_periods = len(equity_curve)
    n_years = n_periods / periods_per_year
    
    if n_years <= 0:
        return 0.0
    
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    
    dd = max_drawdown(equity_curve)
    max_dd = abs(dd['max_dd'] / 100)  # Convert back to decimal
    
    if max_dd == 0:
        return float('inf')
    
    return ann_return / max_dd


# -----------------------------------------------------------------
# Task 11: Value at Risk (VaR) - 99% Historical
# -----------------------------------------------------------------
def historical_var(returns: pd.Series, confidence: float = 0.99) -> dict:
    """
    Calculate Historical Value at Risk (VaR).
    
    VaR answers: "What is the maximum expected loss on the worst
    X% of trading days?"
    
    99% VaR: On the worst 1% of days, you lose AT LEAST this much.
    
    Args:
        returns: Daily returns series
        confidence: Confidence level (default 0.99 = 99%)
    
    Returns dict with:
        - var: VaR as a negative percentage
        - expected_shortfall: Average loss beyond VaR (CVaR/ES)
        - worst_day: Worst single day return
        - var_dollar: Dollar VaR (if initial capital known)
    """
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        return {'var': 0, 'expected_shortfall': 0, 'worst_day': 0}
    
    # Historical VaR: the (1-confidence) percentile of returns
    var = np.percentile(clean_returns, (1 - confidence) * 100)
    
    # Expected Shortfall (CVaR): average of returns worse than VaR
    tail_returns = clean_returns[clean_returns <= var]
    if len(tail_returns) > 0:
        expected_shortfall = tail_returns.mean()
    else:
        expected_shortfall = var
    
    worst_day = clean_returns.min()
    best_day = clean_returns.max()
    
    return {
        'var': var * 100,  # as percentage
        'expected_shortfall': expected_shortfall * 100,
        'worst_day': worst_day * 100,
        'best_day': best_day * 100,
        'confidence': confidence,
        'n_observations': len(clean_returns),
        'var_breaches': (clean_returns < var).sum()
    }


# -----------------------------------------------------------------
# Comprehensive Risk Report
# -----------------------------------------------------------------
def generate_risk_report(equity_curve: pd.Series, 
                         initial_capital: float = 1_000_000) -> dict:
    """Generate a comprehensive risk report."""
    returns = equity_curve.pct_change().dropna()
    
    # Task 10 metrics
    dd = max_drawdown(equity_curve)
    calmar = calmar_ratio(equity_curve)
    
    # Task 11 metrics
    var = historical_var(returns, confidence=0.99)
    var_95 = historical_var(returns, confidence=0.95)
    
    # Additional metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    n_days = len(equity_curve)
    n_years = n_days / 252
    ann_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/max(n_years, 0.01)) - 1) * 100
    
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # Sortino uses only downside deviation
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(252)
        else:
            sortino = float('inf')
    else:
        sharpe = 0
        sortino = 0
    
    # Kurtosis of returns
    kurtosis = returns.kurtosis()
    skew = returns.skew()
    
    # Win rate
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    win_rate = winning_days / max(winning_days + losing_days, 1) * 100
    
    report = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': dd['max_dd'],
        'max_dd_peak_date': dd['peak_date'],
        'max_dd_trough_date': dd['trough_date'],
        'max_dd_recovery_date': dd['recovery_date'],
        'max_dd_recovery_days': dd['recovery_days'],
        'var_99': var['var'],
        'expected_shortfall_99': var['expected_shortfall'],
        'var_95': var_95['var'],
        'worst_day': var['worst_day'],
        'best_day': var['best_day'],
        'kurtosis': kurtosis,
        'skewness': skew,
        'win_rate': win_rate,
        'winning_days': winning_days,
        'losing_days': losing_days,
        'var_99_dollar': initial_capital * abs(var['var'] / 100),
        'drawdown_series': dd['drawdown_series']
    }
    
    return report


def print_risk_report(report: dict, initial_capital: float = 1_000_000):
    """Print the risk report in Bloomberg terminal style."""
    print("\n" + "=" * 60)
    print("  RISK MANAGEMENT REPORT")
    print("=" * 60)
    
    print(f"\n  -- Performance --")
    print(f"  Total Return:       {report['total_return']:>11.2f}%")
    print(f"  Annualized Return:  {report['annualized_return']:>11.2f}%")
    print(f"  Sharpe Ratio:       {report['sharpe_ratio']:>11.4f}")
    print(f"  Sortino Ratio:      {report['sortino_ratio']:>11.4f}")
    print(f"  Calmar Ratio:       {report['calmar_ratio']:>11.4f}")
    
    print(f"\n  -- Drawdown Analysis (Task 10) --")
    print(f"  Max Drawdown:       {report['max_drawdown']:>11.2f}%")
    if report['max_dd_peak_date']:
        print(f"  Peak Date:          {str(report['max_dd_peak_date'])[:10]}")
    if report['max_dd_trough_date']:
        print(f"  Trough Date:        {str(report['max_dd_trough_date'])[:10]}")
    if report['max_dd_recovery_date']:
        print(f"  Recovery Date:      {str(report['max_dd_recovery_date'])[:10]}")
    if report['max_dd_recovery_days']:
        print(f"  Recovery Duration:  {report['max_dd_recovery_days']} days")
    
    print(f"\n  -- Value at Risk (Task 11) --")
    print(f"  99% VaR:            {report['var_99']:>11.2f}%")
    print(f"  99% VaR (Rs.):      Rs.{report['var_99_dollar']:>9,.0f}")
    print(f"  99% CVaR/ES:        {report['expected_shortfall_99']:>11.2f}%")
    print(f"  95% VaR:            {report['var_95']:>11.2f}%")
    print(f"  Worst Day:          {report['worst_day']:>11.2f}%")
    print(f"  Best Day:           {report['best_day']:>11.2f}%")
    
    print(f"\n  -- Distribution --")
    print(f"  Kurtosis:           {report['kurtosis']:>11.4f}")
    print(f"  Skewness:           {report['skewness']:>11.4f}")
    print(f"  Win Rate:           {report['win_rate']:>11.1f}%")
    print(f"  Winning Days:       {report['winning_days']:>11d}")
    print(f"  Losing Days:        {report['losing_days']:>11d}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  RISK METRICS TEST")
    print("#" * 60)
    
    # Generate synthetic equity curve for testing
    np.random.seed(42)
    n_days = 1000
    returns = np.random.normal(0.0003, 0.015, n_days)
    equity = 1_000_000 * np.cumprod(1 + returns)
    dates = pd.bdate_range(start='2020-01-01', periods=n_days)
    equity_curve = pd.Series(equity, index=dates)
    
    report = generate_risk_report(equity_curve)
    print_risk_report(report)
