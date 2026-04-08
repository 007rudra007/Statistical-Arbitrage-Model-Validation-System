"""
Task 1: Data Fetcher (Abstraction Layer)
==========================================
An abstraction layer for data sources.
Currently uses yfinance, but designed to be swappable
with broker APIs (Zerodha, Truedata, XTS) when going live.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import yfinance as yf

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# -----------------------------------------------------------------
# Abstract Data Source
# -----------------------------------------------------------------
class DataSource:
    """Base class for data sources. Override fetch() for different APIs."""
    
    def fetch(self, ticker: str, start: datetime, end: datetime,
              interval: str = '1d') -> pd.DataFrame:
        raise NotImplementedError


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source (free, good for development)."""
    
    def fetch(self, ticker: str, start: datetime, end: datetime,
              interval: str = '1d') -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            ticker: Yahoo Finance ticker (e.g., 'HDFCBANK.NS')
            start: Start date
            end: End date
            interval: '1m', '5m', '15m', '1h', '1d', '1wk'
        
        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
        """
        data = yf.download(ticker, start=start, end=end,
                          interval=interval, auto_adjust=True,
                          progress=False)
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data


# -----------------------------------------------------------------
# Multi-Ticker Data Manager
# -----------------------------------------------------------------
class DataManager:
    """
    Manages data for multiple tickers.
    Handles alignment, caching, and source abstraction.
    """
    
    def __init__(self, source: DataSource = None):
        self.source = source or YahooFinanceSource()
        self.cache = {}
    
    def fetch_pair(self, ticker_a: str, ticker_b: str,
                   years: int = 5) -> pd.DataFrame:
        """Fetch aligned price data for a pair of tickers."""
        end = datetime.now()
        start = end - timedelta(days=years * 365)
        
        print(f"[DATA MANAGER] Fetching pair: {ticker_a} / {ticker_b}")
        
        data_a = self.source.fetch(ticker_a, start, end)
        data_b = self.source.fetch(ticker_b, start, end)
        
        # Align on common dates
        df = pd.DataFrame({
            ticker_a: data_a['Close'].squeeze(),
            ticker_b: data_b['Close'].squeeze()
        }).dropna()
        
        # Cache it
        self.cache[f"{ticker_a}_{ticker_b}"] = df
        
        print(f"[DATA MANAGER] {len(df)} aligned trading days")
        return df
    
    def fetch_universe(self, tickers: List[str], years: int = 5) -> pd.DataFrame:
        """Fetch aligned price data for an entire universe of tickers."""
        end = datetime.now()
        start = end - timedelta(days=years * 365)
        
        print(f"[DATA MANAGER] Fetching universe of {len(tickers)} tickers...")
        
        all_data = {}
        for ticker in tickers:
            try:
                data = self.source.fetch(ticker, start, end)
                all_data[ticker] = data['Close'].squeeze()
                print(f"  [OK] {ticker}: {len(data)} bars")
            except Exception as e:
                print(f"  [FAIL] {ticker}: {e}")
        
        df = pd.DataFrame(all_data).dropna()
        print(f"[DATA MANAGER] {len(df)} aligned trading days across {len(df.columns)} tickers")
        
        return df


# -----------------------------------------------------------------
# NIFTY 50 Top Constituents (for pair scanning)
# -----------------------------------------------------------------
NIFTY_TOP_CONSTITUENTS = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'TCS.NS', 'KOTAKBANK.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS',
    'SBIN.NS', 'BAJFINANCE.NS', 'ITC.NS', 'LICI.NS',
    'AXISBANK.NS', 'LT.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
]


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  DATA FETCHER TEST")
    print("#" * 60)
    
    dm = DataManager()
    df = dm.fetch_pair('HDFCBANK.NS', 'ICICIBANK.NS', years=2)
    print(f"\n  Shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\n  Sample data (last 5 rows):")
    print(df.tail().to_string())
