"""
Task 9: Slippage & Latency Model
===================================
If your strategy survives randomized slippage, you have REAL alpha.
This module simulates the reality that your limit orders will NOT
always fill at the exact price you requested.

Models:
1. Random slippage: 0.5 to 1.5 ticks per order
2. Latency impact: delayed fills in fast-moving markets
3. Partial fills: not all quantity gets filled (configurable)
"""

import sys
import os
import numpy as np

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# -----------------------------------------------------------------
# Slippage Model
# -----------------------------------------------------------------
class SlippageModel:
    """
    Models realistic order execution slippage.
    
    In real markets:
    - BUY orders fill ABOVE the theoretical price
    - SELL orders fill BELOW the theoretical price
    - The magnitude depends on volatility and order size
    """
    
    def __init__(self, min_ticks: float = 0.5, max_ticks: float = 1.5,
                 tick_size: float = 0.05, seed: int = None):
        """
        Args:
            min_ticks: Minimum slippage in ticks
            max_ticks: Maximum slippage in ticks
            tick_size: Price tick size (NSE tick = Rs.0.05)
            seed: Random seed for reproducibility
        """
        self.min_ticks = min_ticks
        self.max_ticks = max_ticks
        self.tick_size = tick_size
        self.rng = np.random.RandomState(seed)
    
    def apply_slippage(self, price: float, side: str, 
                       volatility_multiplier: float = 1.0) -> tuple:
        """
        Apply random slippage to a fill price.
        
        Args:
            price: Theoretical fill price
            side: 'BUY' or 'SELL'
            volatility_multiplier: Scales slippage in high-vol regimes (>1.0)
        
        Returns:
            (adjusted_price, slippage_amount, slippage_ticks)
        """
        # Random number of ticks (uniform between min and max)
        n_ticks = self.rng.uniform(self.min_ticks, self.max_ticks)
        n_ticks *= volatility_multiplier
        
        slippage = n_ticks * self.tick_size
        
        if side.upper() == 'BUY':
            adjusted_price = price + slippage  # Buy at a WORSE (higher) price
        elif side.upper() == 'SELL':
            adjusted_price = price - slippage  # Sell at a WORSE (lower) price
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")
        
        # Round to tick size
        adjusted_price = round(adjusted_price / self.tick_size) * self.tick_size
        actual_slippage = abs(adjusted_price - price)
        
        return adjusted_price, actual_slippage, n_ticks


# -----------------------------------------------------------------
# Latency Model
# -----------------------------------------------------------------
class LatencyModel:
    """
    Models execution latency — the delay between signal generation
    and order reaching the exchange.
    
    In reality:
    - Signal generated → 50ms
    - Order sent to broker API → 100ms
    - Order reaches exchange → 50ms
    - Total: ~200ms minimum
    
    During this delay, the price can move against you.
    """
    
    def __init__(self, base_latency_ms: float = 200.0,
                 jitter_ms: float = 100.0, seed: int = None):
        """
        Args:
            base_latency_ms: Base latency in milliseconds
            jitter_ms: Random jitter (+/- this amount)
        """
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.rng = np.random.RandomState(seed)
    
    def get_latency(self) -> float:
        """Get a randomized latency value in milliseconds."""
        jitter = self.rng.uniform(-self.jitter_ms, self.jitter_ms)
        return max(0, self.base_latency_ms + jitter)
    
    def price_impact(self, price: float, volatility_per_ms: float) -> float:
        """
        Estimate price drift during latency period.
        
        Args:
            price: Current price
            volatility_per_ms: Expected price movement per millisecond
        
        Returns:
            Expected adverse price movement
        """
        latency = self.get_latency()
        # Price can move sqrt(latency) * vol (random walk)
        drift = np.sqrt(latency) * volatility_per_ms * price
        return drift


# -----------------------------------------------------------------
# Combined Execution Model
# -----------------------------------------------------------------
class ExecutionModel:
    """
    Combines slippage + latency into a single execution model.
    This is what gets applied to every simulated order fill.
    """
    
    def __init__(self, slippage_min_ticks: float = 0.5,
                 slippage_max_ticks: float = 1.5,
                 tick_size: float = 0.05,
                 latency_ms: float = 200.0,
                 latency_jitter_ms: float = 100.0,
                 seed: int = 42):
        
        self.slippage = SlippageModel(slippage_min_ticks, slippage_max_ticks,
                                       tick_size, seed)
        self.latency = LatencyModel(latency_ms, latency_jitter_ms, seed + 1)
        self.fill_log = []
    
    def execute(self, price: float, quantity: int, side: str,
                volatility_multiplier: float = 1.0) -> dict:
        """
        Simulate order execution with slippage and latency.
        
        Returns:
            dict with fill details
        """
        # Apply slippage
        fill_price, slippage_amt, slippage_ticks = self.slippage.apply_slippage(
            price, side, volatility_multiplier
        )
        
        # Apply latency
        latency = self.latency.get_latency()
        
        fill = {
            'theoretical_price': price,
            'fill_price': fill_price,
            'quantity': quantity,
            'side': side,
            'slippage': slippage_amt,
            'slippage_ticks': slippage_ticks,
            'latency_ms': latency,
            'adverse_cost': slippage_amt * quantity
        }
        
        self.fill_log.append(fill)
        return fill
    
    def get_statistics(self) -> dict:
        """Get execution statistics from all fills."""
        if not self.fill_log:
            return {}
        
        slippages = [f['slippage'] for f in self.fill_log]
        latencies = [f['latency_ms'] for f in self.fill_log]
        costs = [f['adverse_cost'] for f in self.fill_log]
        
        return {
            'total_fills': len(self.fill_log),
            'avg_slippage': np.mean(slippages),
            'max_slippage': np.max(slippages),
            'avg_latency_ms': np.mean(latencies),
            'total_adverse_cost': np.sum(costs),
            'avg_adverse_cost': np.mean(costs)
        }


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  SLIPPAGE & LATENCY MODEL TEST")
    print("#" * 60)
    
    exec_model = ExecutionModel(seed=42)
    
    # Simulate 10 trades
    prices = [1650, 1655, 1648, 1660, 1645, 1670, 1635, 1675, 1640, 1680]
    sides = ['BUY', 'SELL'] * 5
    
    for i, (price, side) in enumerate(zip(prices, sides)):
        fill = exec_model.execute(price, 100, side)
        print(f"  [{i+1:02d}] {side:4s} @ Rs.{price:.2f} -> Fill: Rs.{fill['fill_price']:.2f} "
              f"(slip: {fill['slippage']:.2f}, {fill['slippage_ticks']:.1f} ticks, "
              f"lat: {fill['latency_ms']:.0f}ms)")
    
    stats = exec_model.get_statistics()
    print(f"\n  --- STATISTICS ---")
    print(f"  Total fills:       {stats['total_fills']}")
    print(f"  Avg slippage:      Rs.{stats['avg_slippage']:.4f}")
    print(f"  Max slippage:      Rs.{stats['max_slippage']:.4f}")
    print(f"  Avg latency:       {stats['avg_latency_ms']:.0f}ms")
    print(f"  Total adverse cost: Rs.{stats['total_adverse_cost']:.2f}")
