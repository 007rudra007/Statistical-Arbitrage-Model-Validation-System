"""
Task 8: Indian Market Transaction Costs
=========================================
Model real-world friction for Indian equity markets.
Your backtester MUST deduct these costs on every trade:
- STT (Securities Transaction Tax)
- Exchange Transaction Charges
- SEBI Turnover Fees
- Stamp Duty
- GST on brokerage and charges
"""

import sys
import os

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# -----------------------------------------------------------------
# Indian Market Cost Model (NSE Equity Delivery)
# -----------------------------------------------------------------
class IndianMarketCosts:
    """
    Models all statutory and exchange costs for Indian equity markets (NSE).
    
    These are the REAL costs that eat into your returns.
    Amateur quants ignore them. Final bosses model them precisely.
    
    All rates are as of 2024 (update as SEBI changes them).
    """
    
    # STT - Securities Transaction Tax
    # Buy: 0.1% on turnover (delivery-based equity)
    # Sell: 0.1% on turnover (delivery-based equity)
    STT_BUY_RATE = 0.001       # 0.1%
    STT_SELL_RATE = 0.001      # 0.1%
    
    # Exchange Transaction Charges (NSE)
    NSE_TXN_CHARGE_RATE = 0.0000345  # 0.00345%
    
    # SEBI Turnover Fee
    SEBI_TURNOVER_FEE_RATE = 0.000001  # Rs.1 per crore (0.0001%)
    
    # Stamp Duty (varies by state, using Maharashtra as reference)
    # Buy side only, 0.015% or Rs.1500 per crore
    STAMP_DUTY_RATE = 0.00015  # 0.015%
    
    # GST on brokerage and transaction charges
    GST_RATE = 0.18  # 18%
    
    # Brokerage (discount broker like Zerodha: Rs.20 per order or 0.03%)
    BROKERAGE_PER_ORDER = 20.0  # Rs.20 flat
    BROKERAGE_RATE = 0.0003     # 0.03% (whichever is lower)
    
    def __init__(self, brokerage_per_order=None, brokerage_rate=None):
        if brokerage_per_order is not None:
            self.BROKERAGE_PER_ORDER = brokerage_per_order
        if brokerage_rate is not None:
            self.BROKERAGE_RATE = brokerage_rate
    
    def calculate_buy_costs(self, quantity: int, price: float) -> dict:
        """Calculate all costs for a BUY order."""
        turnover = quantity * price
        
        brokerage = min(self.BROKERAGE_PER_ORDER, turnover * self.BROKERAGE_RATE)
        stt = turnover * self.STT_BUY_RATE
        exchange_txn = turnover * self.NSE_TXN_CHARGE_RATE
        sebi_fee = turnover * self.SEBI_TURNOVER_FEE_RATE
        stamp_duty = turnover * self.STAMP_DUTY_RATE
        gst = (brokerage + exchange_txn + sebi_fee) * self.GST_RATE
        
        total = brokerage + stt + exchange_txn + sebi_fee + stamp_duty + gst
        
        return {
            'turnover': turnover,
            'brokerage': brokerage,
            'stt': stt,
            'exchange_txn': exchange_txn,
            'sebi_fee': sebi_fee,
            'stamp_duty': stamp_duty,
            'gst': gst,
            'total': total,
            'cost_pct': (total / turnover) * 100
        }
    
    def calculate_sell_costs(self, quantity: int, price: float) -> dict:
        """Calculate all costs for a SELL order."""
        turnover = quantity * price
        
        brokerage = min(self.BROKERAGE_PER_ORDER, turnover * self.BROKERAGE_RATE)
        stt = turnover * self.STT_SELL_RATE
        exchange_txn = turnover * self.NSE_TXN_CHARGE_RATE
        sebi_fee = turnover * self.SEBI_TURNOVER_FEE_RATE
        stamp_duty = 0  # No stamp duty on sell side
        gst = (brokerage + exchange_txn + sebi_fee) * self.GST_RATE
        
        total = brokerage + stt + exchange_txn + sebi_fee + stamp_duty + gst
        
        return {
            'turnover': turnover,
            'brokerage': brokerage,
            'stt': stt,
            'exchange_txn': exchange_txn,
            'sebi_fee': sebi_fee,
            'stamp_duty': stamp_duty,
            'gst': gst,
            'total': total,
            'cost_pct': (total / turnover) * 100
        }
    
    def round_trip_cost(self, quantity: int, buy_price: float, 
                        sell_price: float) -> dict:
        """Calculate total round-trip costs (buy + sell)."""
        buy_costs = self.calculate_buy_costs(quantity, buy_price)
        sell_costs = self.calculate_sell_costs(quantity, sell_price)
        
        total = buy_costs['total'] + sell_costs['total']
        total_turnover = buy_costs['turnover'] + sell_costs['turnover']
        
        return {
            'buy_costs': buy_costs,
            'sell_costs': sell_costs,
            'total_costs': total,
            'total_cost_pct': (total / total_turnover) * 100
        }


def print_cost_breakdown(costs: dict, side: str = "BUY"):
    """Pretty-print a cost breakdown."""
    print(f"\n  [{side}] Cost Breakdown:")
    print(f"    Turnover:         Rs.{costs['turnover']:>12,.2f}")
    print(f"    Brokerage:        Rs.{costs['brokerage']:>12,.2f}")
    print(f"    STT:              Rs.{costs['stt']:>12,.2f}")
    print(f"    Exchange Txn:     Rs.{costs['exchange_txn']:>12,.2f}")
    print(f"    SEBI Fee:         Rs.{costs['sebi_fee']:>12,.2f}")
    print(f"    Stamp Duty:       Rs.{costs['stamp_duty']:>12,.2f}")
    print(f"    GST (18%):        Rs.{costs['gst']:>12,.2f}")
    print(f"    -----------------------------------")
    print(f"    TOTAL:            Rs.{costs['total']:>12,.2f} ({costs['cost_pct']:.4f}%)")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  INDIAN MARKET COST MODEL")
    print("#" * 60)
    
    costs = IndianMarketCosts()
    
    # Example: Buy 100 shares of HDFCBANK at Rs.1650
    buy = costs.calculate_buy_costs(100, 1650)
    print_cost_breakdown(buy, "BUY")
    
    # Example: Sell 100 shares at Rs.1680
    sell = costs.calculate_sell_costs(100, 1680)
    print_cost_breakdown(sell, "SELL")
    
    # Round trip
    rt = costs.round_trip_cost(100, 1650, 1680)
    print(f"\n  Round-trip total: Rs.{rt['total_costs']:.2f} ({rt['total_cost_pct']:.4f}%)")
    print(f"  P&L before costs: Rs.{(1680-1650)*100:.2f}")
    print(f"  P&L after costs:  Rs.{(1680-1650)*100 - rt['total_costs']:.2f}")
