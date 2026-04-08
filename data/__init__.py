# Data Infrastructure Module
# ──────────────────────────
# Phase 1: DataManager, DataScrubber  (data fetching + cleaning)
# Phase 2: NSEIngestion Pipeline, StorageClient (MinIO/S3, Parquet, portfolios)

from data.fetcher import DataManager, YahooFinanceSource, DataSource, NIFTY_TOP_CONSTITUENTS
from data.scrubber import DataScrubber
from data.storage import StorageClient, get_storage_client, StorageConfig

__all__ = [
    # Fetching
    "DataManager",
    "YahooFinanceSource",
    "DataSource",
    "NIFTY_TOP_CONSTITUENTS",
    # Cleaning
    "DataScrubber",
    # Storage
    "StorageClient",
    "get_storage_client",
    "StorageConfig",
]
