"""
Phase 5 – Step 20: Multi-Strategy Backtester (Phase 5 extension)
==================================================================
Extends the existing Phase 2 BacktestEngine with:
  - Multi-ticker price DataFrame support
  - MomentumStrategy + AgentSignalStrategy
  - Unified MetricsEngine (Sharpe, Sortino, Calmar, MDD, profit factor)
  - Integration with Phase 4 agent signals
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS   = 252
RISK_FREE_RATE = 0.065
BROKERAGE      = 0.0020   # 20bps round-trip


# ── Events ─────────────────────────────────────────────────────────────────────
@dataclass
class BarEvent:
    timestamp: pd.Timestamp
    ticker: str
    close: float
    open: float = 0.0
    high: float = 0.0
    low: float  = 0.0
    volume: float = 0.0


@dataclass
class Signal:
    timestamp: pd.Timestamp
    ticker: str
    direction: str    # BUY | SELL | EXIT | HOLD
    strength: float   # 0-1
    strategy: str
    meta: Dict = field(default_factory=dict)


@dataclass
class Fill:
    timestamp: pd.Timestamp
    ticker: str
    direction: str
    quantity: int
    fill_price: float
    slippage: float
    commission: float

    @property
    def cost(self):
        return self.commission + self.slippage * self.quantity

    @property
    def notional(self):
        return self.fill_price * self.quantity


# ── Portfolio ──────────────────────────────────────────────────────────────────
class Portfolio:
    def __init__(self, capital: float = 10_000_000.0):
        self.capital   = capital
        self.cash      = capital
        self.positions: Dict[str, int]   = {}
        self.avg_cost:  Dict[str, float] = {}
        self.equity_curve: List[Dict]    = []
        self.trade_log:    List[Dict]    = []

    def on_fill(self, fill: Fill):
        t = fill.ticker; q = fill.quantity; p = fill.fill_price
        if fill.direction == "BUY":
            old_q  = self.positions.get(t, 0)
            old_c  = self.avg_cost.get(t, 0.0)
            new_q  = old_q + q
            self.avg_cost[t]   = (old_c * old_q + p * q) / new_q if new_q else 0.0
            self.positions[t]  = new_q
            self.cash         -= p * q + fill.cost
        else:
            self.positions[t]  = self.positions.get(t, 0) - q
            entry              = self.avg_cost.get(t, p)
            pnl                = (p - entry) * q - fill.cost
            self.cash         += p * q - fill.cost
            self.trade_log.append({
                "timestamp": fill.timestamp, "ticker": t,
                "direction": fill.direction, "quantity": q,
                "fill_price": p, "pnl": pnl, "commission": fill.cost,
            })

    def mark(self, prices: Dict[str, float], ts: pd.Timestamp):
        pos_val = sum(q * prices.get(t, self.avg_cost.get(t, 0))
                      for t, q in self.positions.items() if q > 0)
        nav = self.cash + pos_val
        self.equity_curve.append({"timestamp": ts, "nav": nav, "cash": self.cash,
                                   "positions": pos_val})
        return nav

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve).set_index("timestamp")


# ── Metrics ────────────────────────────────────────────────────────────────────
class MetricsEngine:
    def compute(self, eq_df: pd.DataFrame, trade_log: List[Dict]) -> Dict[str, Any]:
        nav = eq_df["nav"]
        ret = nav.pct_change().dropna()
        if len(ret) < 2:
            return {"error": "insufficient data"}
        n    = len(ret)
        cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / n) - 1)
        vol  = float(ret.std() * np.sqrt(TRADING_DAYS))
        sr   = float((ret.mean() * TRADING_DAYS - RISK_FREE_RATE) / (vol + 1e-10))
        neg  = ret[ret < 0]
        dvol = float(neg.std() * np.sqrt(TRADING_DAYS)) if len(neg) > 1 else vol
        so   = float((ret.mean() * TRADING_DAYS - RISK_FREE_RATE) / (dvol + 1e-10))
        mdd  = float(((nav - nav.cummax()) / nav.cummax()).min())
        cal  = float(cagr / abs(mdd)) if mdd != 0 else 0.0
        trades = pd.DataFrame(trade_log)
        if trades.empty:
            wr = pf = aw = al = 0.0; nt = 0
        else:
            pnl = trades["pnl"]; nt = len(pnl)
            wins = pnl[pnl > 0]; loss = pnl[pnl < 0]
            wr = len(wins) / nt if nt else 0.0
            aw = float(wins.mean()) if len(wins) else 0.0
            al = float(loss.mean()) if len(loss) else 0.0
            pf = float(wins.sum() / abs(loss.sum())) if len(loss) and loss.sum() != 0 else float("inf")
        return {
            "cagr": round(cagr, 4), "annual_vol": round(vol, 4),
            "sharpe_ratio": round(sr, 4), "sortino_ratio": round(so, 4),
            "calmar_ratio": round(cal, 4), "max_drawdown": round(mdd, 4),
            "win_rate": round(wr, 4), "profit_factor": round(pf, 4),
            "avg_win_inr": round(aw, 2), "avg_loss_inr": round(al, 2),
            "n_trades": nt, "final_nav": round(float(nav.iloc[-1]), 2),
            "initial_capital": round(float(nav.iloc[0]), 2),
            "total_return": round(float(nav.iloc[-1] / nav.iloc[0] - 1), 4),
            "n_days": n,
        }


# ── Strategies ─────────────────────────────────────────────────────────────────
class MomentumStrategy:
    """12-1 month cross-sectional momentum with monthly rebalance."""

    def __init__(self, tickers: List[str], formation: int = 252,
                 skip: int = 21, top_pct: float = 0.3):
        self.tickers   = tickers
        self.formation = formation
        self.skip      = skip
        self.top_pct   = top_pct
        self._last_rebal: Optional[pd.Timestamp] = None

    def on_bar(self, bar: BarEvent, hist: Dict[str, pd.Series]) -> List[Signal]:
        if self._last_rebal and (bar.timestamp - self._last_rebal).days < 21:
            return []
        scores = {}
        for t in self.tickers:
            px = hist.get(t)
            if px is None or len(px) < self.formation + self.skip:
                continue
            scores[t] = float(px.iloc[-(self.formation + self.skip)] /
                               px.iloc[-self.skip] - 1)
        if not scores:
            return []
        ranked   = sorted(scores, key=scores.get, reverse=True)
        n_long   = max(1, int(len(ranked) * self.top_pct))
        long_set = set(ranked[:n_long])
        signals  = [Signal(
            timestamp=bar.timestamp, ticker=t,
            direction="BUY" if t in long_set else "EXIT",
            strength=min(abs(scores.get(t, 0)), 1.0),
            strategy="momentum",
            meta={"score": round(scores.get(t, 0), 4)},
        ) for t in self.tickers]
        self._last_rebal = bar.timestamp
        return signals


class AgentSignalStrategy:
    """Replays pre-computed Phase 4 agent signals."""

    def __init__(self, df: pd.DataFrame):
        # columns: timestamp, ticker, action, confidence
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        self._idx = df.set_index(["timestamp", "ticker"])

    def on_bar(self, bar: BarEvent, hist: Dict[str, pd.Series]) -> List[Signal]:
        try:
            row = self._idx.loc[(bar.timestamp, bar.ticker)]
            act = row.get("action", "HOLD")
            if act == "HOLD":
                return []
            return [Signal(
                timestamp=bar.timestamp, ticker=bar.ticker,
                direction=act, strength=float(row.get("confidence", 0.5)),
                strategy="agent",
            )]
        except KeyError:
            return []


# ── Execution handler ──────────────────────────────────────────────────────────
class ExecutionHandler:
    def __init__(self, portfolio: Portfolio, size_pct: float = 0.05,
                 slippage_ticks: float = 1.0):
        self._p = portfolio
        self._size = size_pct
        try:
            from backtester.slippage import SlippageModel
            self._slip = SlippageModel(0.5, slippage_ticks * 2, seed=42)
        except Exception:
            self._slip = None

    def execute(self, sig: Signal, price: float) -> Optional[Fill]:
        if price <= 0 or sig.direction == "HOLD":
            return None
        qty = self._p.positions.get(sig.ticker, 0)
        if sig.direction == "EXIT":
            if qty <= 0:
                return None
            fill_qty = qty; fill_side = "SELL"
        elif sig.direction == "BUY":
            nav      = self._p.cash + sum(
                q * self._p.avg_cost.get(t, 0)
                for t, q in self._p.positions.items()
            )
            notional = nav * self._size * sig.strength
            fill_qty = max(1, int(notional / price)); fill_side = "BUY"
        else:
            fill_qty  = max(1, qty // 2) if qty > 0 else 0
            fill_side = "SELL"
            if fill_qty == 0:
                return None
        if self._slip:
            adj, slip, _ = self._slip.apply_slippage(price, fill_side)
        else:
            adj = price; slip = 0.0
        return Fill(
            timestamp=sig.timestamp, ticker=sig.ticker,
            direction=fill_side, quantity=fill_qty,
            fill_price=adj, slippage=slip,
            commission=price * fill_qty * BROKERAGE,
        )


# ── Main backtester class ──────────────────────────────────────────────────────
class MultiStrategyBacktester:
    """
    Phase 5 event-driven multi-strategy backtester.

    Usage:
        bt = MultiStrategyBacktester(price_df, strategy, capital=10_000_000)
        result = bt.run()
    """

    def __init__(self, price_df: pd.DataFrame, strategy,
                 capital: float = 10_000_000.0,
                 size_pct: float = 0.05,
                 slippage_ticks: float = 1.0):
        self._prices   = price_df.sort_index()
        self._strategy = strategy
        self._port     = Portfolio(capital)
        self._exec     = ExecutionHandler(self._port, size_pct, slippage_ticks)
        self._tickers  = list(price_df.columns)
        self._hist: Dict[str, pd.Series] = {t: pd.Series(dtype=float) for t in self._tickers}

    def run(self) -> Dict[str, Any]:
        log.info("[Backtester] %d tickers, %d bars", len(self._tickers), len(self._prices))
        for ts, row in self._prices.iterrows():
            prices = row.to_dict()
            for t in self._tickers:
                if t in prices and not np.isnan(prices[t]):
                    self._hist[t] = pd.concat([
                        self._hist[t], pd.Series([prices[t]], index=[ts])
                    ])
            for t in self._tickers:
                if t not in prices or np.isnan(prices[t]):
                    continue
                bar = BarEvent(timestamp=ts, ticker=t, close=prices[t],
                               open=prices[t], high=prices[t], low=prices[t])
                for sig in self._strategy.on_bar(bar, self._hist):
                    fill = self._exec.execute(sig, prices.get(sig.ticker, 0))
                    if fill:
                        self._port.on_fill(fill)
            self._port.mark(prices, ts)

        eq   = self._port.equity_df()
        mets = MetricsEngine().compute(eq, self._port.trade_log)
        log.info("[Backtester] Sharpe=%.3f MDD=%.2f%% CAGR=%.2f%%",
                 mets.get("sharpe_ratio", 0),
                 mets.get("max_drawdown", 0) * 100,
                 mets.get("cagr", 0) * 100)
        return {
            "metrics":        mets,
            "equity_df":      eq,
            "trade_log":      pd.DataFrame(self._port.trade_log),
            "strategy":       type(self._strategy).__name__,
            "tickers":        self._tickers,
            "n_days":         len(self._prices),
        }
