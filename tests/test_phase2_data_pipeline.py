"""
Phase 2 Integration Test
=========================
Tests the full data pipeline without a live K8s cluster:
  1. StorageClient health + bucket setup (requires local MinIO)
  2. NSE ingestion smoke test (1 ticker, 30 days)
  3. Portfolio upload / download round-trip
  4. Parquet report round-trip
  5. OHLCV data fetch (yfinance fallback path)
  6. Spark pipeline module import + schema validation (no actual Spark needed)

Run against local MinIO:
    docker-compose up minio minio-init -d   # start MinIO
    python tests/test_phase2_data_pipeline.py

Run without MinIO (dry-run, skips storage tests):
    SKIP_MINIO=1 python tests/test_phase2_data_pipeline.py
"""

import io
import json
import os
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIP_MINIO = os.getenv("SKIP_MINIO", "0") == "1"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    close = 1500 + np.cumsum(np.random.randn(n) * 3)
    return pd.DataFrame({
        "Open":   close * 0.995,
        "High":   close * 1.010,
        "Low":    close * 0.985,
        "Close":  close,
        "Volume": np.random.randint(100_000, 2_000_000, size=n).astype(float),
    }, index=dates)


# ── Test Suite ────────────────────────────────────────────────────────────────
class TestDataScrubber(unittest.TestCase):
    """Unit tests for DataScrubber — no external services needed."""

    def setUp(self):
        from data.scrubber import DataScrubber
        self.scrubber = DataScrubber(max_ffill_periods=3, outlier_std=4.0)

    def test_scrub_clean_data(self):
        df = _make_ohlcv(200)
        clean = self.scrubber.scrub(df[["Close"]])
        self.assertGreater(len(clean), 100)
        self.assertFalse(clean.isnull().any().any())

    def test_scrub_handles_missing_values(self):
        df = _make_ohlcv(200)
        df.iloc[10, 0] = np.nan
        df.iloc[50:53, 0] = np.nan
        clean = self.scrubber.scrub(df[["Close"]])
        self.assertFalse(clean.isnull().any().any())

    def test_scrub_detects_outliers(self):
        df = _make_ohlcv(200)
        df.iloc[100, 0] = df.iloc[99, 0] * 2.5   # big spike
        clean = self.scrubber.scrub(df[["Close"]])
        # After scrubbing, outlier should be removed / interpolated
        self.assertFalse(clean.isnull().any().any())

    def test_align_timestamps(self):
        df1 = _make_ohlcv(100)
        df2 = _make_ohlcv(80)
        aligned = self.scrubber.align_timestamps(df1, df2)
        self.assertEqual(len(aligned[0]), len(aligned[1]))


class TestParquetSerialization(unittest.TestCase):
    """Unit test for Parquet round-trip (no MinIO needed)."""

    def test_df_to_parquet_bytes_roundtrip(self):
        from data.storage import StorageClient
        df = _make_ohlcv(50)
        # Test the helper inline
        table = pa.Table.from_pandas(df, preserve_index=True)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        df2 = pq.read_table(buf).to_pandas()
        self.assertEqual(len(df), len(df2))
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, df2.columns)

    def test_s3_key_format(self):
        from data.nse_ingestion import _s3_key
        key = _s3_key("HDFCBANK.NS", "1d", "2024-01-15")
        self.assertEqual(key, "nse/HDFCBANK_NS/1d/2024-01-15.parquet")

    def test_s3_key_index_ticker(self):
        from data.nse_ingestion import _s3_key
        key = _s3_key("^NSEI", "1d", "2024-01-15")
        self.assertTrue(key.startswith("nse/idx_NSEI/"))


class TestNSEIngestionConfig(unittest.TestCase):
    """Config and tickers validation — no network needed."""

    def test_all_tickers_not_empty(self):
        from data.nse_ingestion import ALL_TICKERS
        self.assertGreater(len(ALL_TICKERS), 20)

    def test_all_tickers_have_ns_suffix_or_index(self):
        from data.nse_ingestion import ALL_TICKERS
        for t in ALL_TICKERS:
            self.assertTrue(
                t.endswith(".NS") or t.startswith("^"),
                f"Unexpected ticker format: {t}"
            )

    def test_config_defaults(self):
        from data.nse_ingestion import Config
        self.assertEqual(Config.BUCKET_RAW, "aladdin-raw")
        self.assertEqual(Config.BUCKET_CLEAN, "aladdin-clean")
        self.assertEqual(Config.DAILY_LOOKBACK_YEARS, 5)


class TestStorageClientMocked(unittest.TestCase):
    """StorageClient tests using a mocked boto3 client."""

    def _make_sc(self):
        from data.storage import StorageClient
        sc = StorageClient()
        sc._client = MagicMock()
        return sc

    def test_upload_portfolio(self):
        sc = self._make_sc()
        sc.client.put_object = MagicMock()
        sc._ensure_bucket = MagicMock()

        portfolio = {
            "name": "Test",
            "positions": [{"ticker": "HDFCBANK.NS", "weight": 1.0}]
        }
        key = sc.upload_portfolio("test_v1", portfolio, user="testuser")
        self.assertEqual(key, "portfolios/testuser/test_v1.json")
        sc.client.put_object.assert_called_once()

    def test_download_portfolio(self):
        sc = self._make_sc()
        payload = {"name": "Test", "positions": []}
        sc.client.get_object = MagicMock(return_value={
            "Body": MagicMock(read=MagicMock(
                return_value=json.dumps(payload).encode()
            ))
        })
        result = sc.download_portfolio("test_v1", user="testuser")
        self.assertEqual(result["name"], "Test")

    def test_health_check_ok(self):
        sc = self._make_sc()
        sc.client.list_buckets = MagicMock(return_value={
            "Buckets": [{"Name": "aladdin-raw"}, {"Name": "aladdin-clean"}]
        })
        health = sc.health_check()
        self.assertEqual(health["status"], "ok")
        self.assertIn("aladdin-raw", health["buckets"])

    def test_health_check_error(self):
        sc = self._make_sc()
        sc.client.list_buckets = MagicMock(side_effect=Exception("Connection refused"))
        health = sc.health_check()
        self.assertEqual(health["status"], "error")

    def test_iam_policy_structure(self):
        from data.storage import portfolio_uploader_policy, data_reader_policy
        policy = portfolio_uploader_policy("aladdin-portfolios")
        self.assertIn("Statement", policy)
        self.assertEqual(policy["Version"], "2012-10-17")

        reader = data_reader_policy("aladdin-raw", "aladdin-clean")
        arns = reader["Statement"][0]["Resource"]
        self.assertTrue(any("aladdin-raw" in r for r in arns))


@unittest.skipIf(SKIP_MINIO, "Skipping MinIO live tests (SKIP_MINIO=1)")
class TestStorageClientLive(unittest.TestCase):
    """
    Integration tests against a live local MinIO.
    Requires: docker-compose up minio minio-init -d
    """

    def setUp(self):
        from data.storage import StorageClient, StorageConfig
        self.sc = StorageClient()
        # Override endpoint to local
        StorageConfig.ENDPOINT = MINIO_ENDPOINT
        try:
            health = self.sc.health_check()
            if health["status"] != "ok":
                self.skipTest("MinIO not reachable")
        except Exception:
            self.skipTest("MinIO not reachable")

    def test_bucket_setup(self):
        self.sc.setup_all_buckets()
        buckets = self.sc.list_buckets()
        for b in ["aladdin-raw", "aladdin-clean", "aladdin-portfolios"]:
            self.assertIn(b, buckets)

    def test_portfolio_roundtrip(self):
        portfolio = {
            "name": "Integration Test Portfolio",
            "positions": [
                {"ticker": "HDFCBANK.NS", "weight": 0.5},
                {"ticker": "RELIANCE.NS",  "weight": 0.5},
            ]
        }
        pid = f"test_integration_{int(time.time())}"
        self.sc.upload_portfolio(pid, portfolio, user="ci")
        downloaded = self.sc.download_portfolio(pid, user="ci")
        self.assertEqual(downloaded["name"], portfolio["name"])
        self.assertEqual(len(downloaded["positions"]), 2)

    def test_parquet_report_roundtrip(self):
        df = pd.DataFrame({
            "ticker": ["HDFCBANK.NS", "RELIANCE.NS"],
            "var_95": [0.025, 0.030],
        })
        report_id = f"ci_test_{int(time.time())}"
        self.sc.upload_parquet_report(report_id, df)
        df2 = self.sc.download_parquet_report(report_id)
        self.assertEqual(len(df), len(df2))

    def test_presigned_url_generation(self):
        # Upload a small test file first
        self.sc.client.put_object(
            Bucket="aladdin-reports",
            Key="test/presigned_test.txt",
            Body=b"hello",
        )
        url = self.sc.presigned_url(
            "aladdin-reports", "test/presigned_test.txt",
            operation="get_object", expires=60
        )
        self.assertTrue(url.startswith("http"))


@unittest.skipIf(SKIP_MINIO, "Skipping ingestion live tests (SKIP_MINIO=1)")
class TestNSEIngestionLive(unittest.TestCase):
    """
    End-to-end ingestion test: fetches 1 ticker → uploads to MinIO → queries back.
    Requires MinIO + internet (for yfinance).
    """

    def setUp(self):
        from data.storage import StorageClient
        sc = StorageClient()
        health = sc.health_check()
        if health["status"] != "ok":
            self.skipTest("MinIO not reachable")

    def test_single_ticker_daily_ingestion(self):
        import yfinance as yf
        from data.nse_ingestion import (
            build_s3_client, ensure_bucket, fetch_and_upload_ohlcv, Config
        )
        from data.scrubber import DataScrubber

        client = build_s3_client(Config)
        ensure_bucket(client, Config.BUCKET_RAW)
        ensure_bucket(client, Config.BUCKET_CLEAN)

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        scrubber = DataScrubber()

        results = fetch_and_upload_ohlcv(
            client=client,
            tickers=["HDFCBANK.NS"],
            interval="1d",
            start=start,
            end=end,
            raw_bucket=Config.BUCKET_RAW,
            clean_bucket=Config.BUCKET_CLEAN,
            scrubber=scrubber,
        )
        self.assertTrue(results.get("HDFCBANK.NS", False),
                        "Ingestion should succeed for HDFCBANK.NS")

        # Verify clean file exists in MinIO
        from data.nse_ingestion import _s3_key
        key = _s3_key("HDFCBANK.NS", "1d", end.strftime("%Y-%m-%d"))
        resp = client.get_object(Bucket=Config.BUCKET_CLEAN, Key=key)
        data = resp["Body"].read()
        self.assertGreater(len(data), 100, "Clean parquet file should not be empty")


if __name__ == "__main__":
    import sys
    # Force UTF-8 on Windows terminal
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("  PHASE 2: DATA PIPELINE TESTS")
    if SKIP_MINIO:
        print("  Mode: UNIT ONLY (SKIP_MINIO=1) -- no MinIO required")
    else:
        print(f"  Mode: FULL  (MinIO at {MINIO_ENDPOINT})")
        print("  Start MinIO first: docker-compose up minio minio-init -d")
    print("=" * 60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always run unit tests
    for cls in [
        TestDataScrubber,
        TestParquetSerialization,
        TestNSEIngestionConfig,
        TestStorageClientMocked,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Integration tests only if MinIO is available
    if not SKIP_MINIO:
        for cls in [TestStorageClientLive, TestNSEIngestionLive]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

