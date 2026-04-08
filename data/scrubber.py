"""
Task 3: Data Scrubber
=======================
Detects missing data packets, forward-fills missing minute bars,
and aligns timestamps across all assets to handle market halts
and illiquid ticks.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Optional

# Force UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# -----------------------------------------------------------------
# Data Scrubber
# -----------------------------------------------------------------
class DataScrubber:
    """
    Cleans and validates time-series data for quantitative analysis.
    
    Operations:
    1. Detect and report missing data packets
    2. Forward-fill missing bars (configurable max gap)
    3. Align timestamps across multiple assets
    4. Detect and handle outliers / bad ticks
    5. Validate data integrity
    """
    
    def __init__(self, max_ffill_periods: int = 5, 
                 outlier_std: float = 5.0):
        """
        Args:
            max_ffill_periods: Maximum consecutive bars to forward-fill
            outlier_std: Number of std devs to flag as outlier
        """
        self.max_ffill_periods = max_ffill_periods
        self.outlier_std = outlier_std
        self.report = {}
    
    def scrub(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full scrubbing pipeline on a DataFrame.
        
        Args:
            df: DataFrame with DatetimeIndex and numeric columns
        
        Returns:
            Cleaned DataFrame
        """
        print(f"\n[SCRUBBER] Starting data scrub...")
        print(f"           Shape: {df.shape}")
        print(f"           Date range: {df.index[0]} to {df.index[-1]}")
        
        original_shape = df.shape
        
        # Step 1: Detect missing data
        missing = self._detect_missing(df)
        
        # Step 2: Handle outliers
        df, outliers_fixed = self._handle_outliers(df)
        
        # Step 3: Forward-fill gaps
        df, ffills = self._forward_fill(df)
        
        # Step 4: Drop remaining NaN rows
        pre_drop = len(df)
        df = df.dropna()
        dropped = pre_drop - len(df)
        
        # Step 5: Validate
        self._validate(df)
        
        # Report
        self.report = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'missing_detected': missing,
            'outliers_fixed': outliers_fixed,
            'forward_fills': ffills,
            'rows_dropped': dropped
        }
        
        print(f"\n[SCRUBBER] Complete:")
        print(f"           Original: {original_shape}")
        print(f"           Final:    {df.shape}")
        print(f"           Missing:  {missing}")
        print(f"           Outliers: {outliers_fixed}")
        print(f"           F-Fills:  {ffills}")
        print(f"           Dropped:  {dropped}")
        
        return df
    
    def _detect_missing(self, df: pd.DataFrame) -> int:
        """Detect missing data points."""
        total_missing = df.isnull().sum().sum()
        
        if total_missing > 0:
            print(f"\n[SCRUBBER] Missing data detected:")
            for col in df.columns:
                col_missing = df[col].isnull().sum()
                if col_missing > 0:
                    pct = col_missing / len(df) * 100
                    print(f"           {col}: {col_missing} ({pct:.1f}%)")
        
        return total_missing
    
    def _handle_outliers(self, df: pd.DataFrame) -> tuple:
        """Detect and clip outlier values."""
        df = df.copy()
        total_outliers = 0
        
        for col in df.select_dtypes(include=[np.number]).columns:
            returns = df[col].pct_change()
            mean_ret = returns.mean()
            std_ret = returns.std()
            
            if std_ret > 0:
                # Flag returns beyond outlier_std standard deviations
                outlier_mask = returns.abs() > (mean_ret + self.outlier_std * std_ret)
                n_outliers = outlier_mask.sum()
                
                if n_outliers > 0:
                    # Replace outlier values with interpolated values
                    df.loc[outlier_mask, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')
                    total_outliers += n_outliers
                    print(f"[SCRUBBER] {col}: {n_outliers} outliers interpolated")
        
        return df, total_outliers
    
    def _forward_fill(self, df: pd.DataFrame) -> tuple:
        """Forward-fill missing values up to max_ffill_periods."""
        df = df.copy()
        total_fills = df.isnull().sum().sum()
        
        df = df.ffill(limit=self.max_ffill_periods)
        
        remaining = df.isnull().sum().sum()
        actual_fills = total_fills - remaining
        
        return df, actual_fills
    
    def _validate(self, df: pd.DataFrame):
        """Validate data integrity."""
        issues = []
        
        # Check for non-positive prices
        for col in df.select_dtypes(include=[np.number]).columns:
            if (df[col] <= 0).any():
                n_bad = (df[col] <= 0).sum()
                issues.append(f"{col}: {n_bad} non-positive values")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            n_dups = df.index.duplicated().sum()
            issues.append(f"{n_dups} duplicate timestamps")
        
        # Check monotonic index
        if not df.index.is_monotonic_increasing:
            issues.append("Index is not sorted chronologically")
        
        if issues:
            print(f"\n[SCRUBBER] Validation issues:")
            for issue in issues:
                print(f"           WARNING: {issue}")
        else:
            print(f"[SCRUBBER] Validation: PASSED")
    
    def align_timestamps(self, *dataframes) -> list:
        """Align multiple DataFrames to common timestamps."""
        if len(dataframes) < 2:
            return list(dataframes)
        
        # Find common index
        common_idx = dataframes[0].index
        for df in dataframes[1:]:
            common_idx = common_idx.intersection(df.index)
        
        print(f"[SCRUBBER] Aligned {len(dataframes)} DataFrames to {len(common_idx)} common timestamps")
        
        return [df.loc[common_idx] for df in dataframes]


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  DATA SCRUBBER TEST")
    print("#" * 60)
    
    # Generate test data with known issues
    np.random.seed(42)
    dates = pd.bdate_range(start='2023-01-01', periods=500)
    
    data = pd.DataFrame({
        'HDFC': 1500 + np.cumsum(np.random.randn(500) * 3),
        'ICICI': 900 + np.cumsum(np.random.randn(500) * 2)
    }, index=dates)
    
    # Inject issues
    data.iloc[50, 0] = np.nan    # missing value
    data.iloc[100:103, 1] = np.nan  # gap
    data.iloc[200, 0] = data.iloc[199, 0] * 1.5  # outlier spike
    
    scrubber = DataScrubber(max_ffill_periods=3, outlier_std=4.0)
    clean = scrubber.scrub(data)
    
    print(f"\n  Scrub report: {scrubber.report}")
