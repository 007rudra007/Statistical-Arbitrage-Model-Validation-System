"""
Phase 2 – Step 5: NSE Ingestion Pipeline
=========================================
Fetches NIFTY 50 constituents + BankNifty OHLCV data via yfinance,
cleans it with DataScrubber, and persists Parquet files to MinIO / AWS S3.

Usage (local MinIO):
    python -m data.nse_ingestion --mode daily

Usage (AWS S3):
    MINIO_ENDPOINT=https://s3.amazonaws.com \
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
    python -m data.nse_ingestion --mode daily
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from botocore.exceptions import ClientError
import yfinance as yf

from data.fetcher import DataManager, YahooFinanceSource, NIFTY_TOP_CONSTITUENTS
from data.scrubber import DataScrubber

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("nse_ingestion")

# Force UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Configuration (from environment, with sane defaults) ─────────────────────
class Config:
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    BUCKET_RAW: str = os.getenv("BUCKET_RAW", "aladdin-raw")
    BUCKET_CLEAN: str = os.getenv("BUCKET_CLEAN", "aladdin-clean")
    REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
    # 1m intraday data limited to last 7 days by yfinance free tier
    INTRADAY_LOOKBACK_DAYS: int = int(os.getenv("INTRADAY_LOOKBACK_DAYS", "7"))
    DAILY_LOOKBACK_YEARS: int = int(os.getenv("DAILY_LOOKBACK_YEARS", "5"))


# ── Tickers ───────────────────────────────────────────────────────────────────
# BankNifty constituents (top 12 by weight)
BANKNIFTY_CONSTITUENTS: List[str] = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "BANKBARODA.NS", "INDUSINDBK.NS", "AUBANK.NS",
    "FEDERALBNK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "PNB.NS",
]

# Index proxies
INDEX_TICKERS: List[str] = ["^NSEI", "^NSEBANK"]

ALL_TICKERS: List[str] = list(
    dict.fromkeys(NIFTY_TOP_CONSTITUENTS + BANKNIFTY_CONSTITUENTS + INDEX_TICKERS)
)


# ── MinIO / S3 Client ─────────────────────────────────────────────────────────
def build_s3_client(cfg: Config = Config) -> boto3.client:
    """
    Build a boto3 S3 client pointing at MinIO (local) or AWS S3 (prod).
    When MINIO_ENDPOINT starts with http://localhost, uses path-style addressing.
    """
    is_local = cfg.MINIO_ENDPOINT.startswith("http://localhost") or \
               cfg.MINIO_ENDPOINT.startswith("http://minio")

    kwargs: Dict = dict(
        endpoint_url=cfg.MINIO_ENDPOINT,
        aws_access_key_id=cfg.MINIO_ACCESS_KEY,
        aws_secret_access_key=cfg.MINIO_SECRET_KEY,
        region_name=cfg.REGION,
    )
    if is_local:
        from botocore.config import Config as BotoConfig
        kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

    return boto3.client("s3", **kwargs)


def ensure_bucket(client, bucket: str) -> None:
    """Create bucket if it does not exist."""
    try:
        client.head_bucket(Bucket=bucket)
        log.debug("Bucket %s already exists.", bucket)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            log.info("Creating bucket: %s", bucket)
            client.create_bucket(Bucket=bucket)
        else:
            raise


# ── Parquet I/O ───────────────────────────────────────────────────────────────
def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialise DataFrame → Parquet bytes (snappy compressed)."""
    table = pa.Table.from_pandas(df, preserve_index=True)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    return buf.read()


def upload_parquet(
    client,
    bucket: str,
    key: str,
    df: pd.DataFrame,
) -> None:
    """Upload a DataFrame as a Parquet file to MinIO/S3."""
    data = df_to_parquet_bytes(df)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType="application/octet-stream",
        Metadata={
            "rows": str(len(df)),
            "columns": ",".join(df.columns.tolist()),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    log.info("Uploaded s3://%s/%s  (%d rows, %.1f KB)", bucket, key, len(df), len(data) / 1024)


def download_parquet(client, bucket: str, key: str) -> pd.DataFrame:
    """Download a Parquet file from MinIO/S3 into a DataFrame."""
    response = client.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(response["Body"].read())
    return pq.read_table(buf).to_pandas()


# ── Ingestion Logic ───────────────────────────────────────────────────────────
def _s3_key(ticker: str, interval: str, date_str: str) -> str:
    """Partition key: nse/{ticker}/{interval}/YYYY-MM-DD.parquet"""
    safe_ticker = ticker.replace("^", "idx_").replace(".", "_")
    return f"nse/{safe_ticker}/{interval}/{date_str}.parquet"


def fetch_and_upload_ohlcv(
    client,
    tickers: List[str],
    interval: str,
    start: datetime,
    end: datetime,
    raw_bucket: str,
    clean_bucket: str,
    scrubber: DataScrubber,
) -> Dict[str, bool]:
    """
    Core function: fetch → scrub → upload raw + clean Parquet per ticker.

    Returns dict of ticker → success.
    """
    date_str = end.strftime("%Y-%m-%d")
    results: Dict[str, bool] = {}

    for ticker in tickers:
        try:
            log.info("[%s] Fetching %s  %s → %s", ticker, interval,
                     start.date(), end.date())

            raw_df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if raw_df.empty:
                log.warning("[%s] No data returned – skipping.", ticker)
                results[ticker] = False
                continue

            # Flatten multi-level columns (yfinance quirk)
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)

            # ── Upload RAW ────────────────────────────────────────────────────
            raw_key = _s3_key(ticker, interval, f"raw_{date_str}")
            upload_parquet(client, raw_bucket, raw_key, raw_df)

            # ── Clean ─────────────────────────────────────────────────────────
            try:
                clean_df = scrubber.scrub(raw_df[["Close"]].copy())
            except Exception as exc:
                log.warning("[%s] Scrubber failed (%s) – uploading raw close.", ticker, exc)
                clean_df = raw_df[["Close"]].dropna()

            # Full OHLCV clean: re-attach other columns aligned to clean index
            ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw_df.columns]
            clean_ohlcv = raw_df[ohlcv_cols].loc[clean_df.index]

            # ── Upload CLEAN ──────────────────────────────────────────────────
            clean_key = _s3_key(ticker, interval, date_str)
            upload_parquet(client, clean_bucket, clean_key, clean_ohlcv)

            results[ticker] = True

        except Exception as exc:
            log.error("[%s] Ingestion failed: %s", ticker, exc, exc_info=True)
            results[ticker] = False

    return results


# ── Public API ────────────────────────────────────────────────────────────────
def run_daily_ingestion(cfg: Config = Config) -> None:
    """
    Daily EOD ingestion: last 5 years of daily OHLCV for all tickers.
    Idempotent – re-running overwrites today's partition.
    """
    log.info("=" * 60)
    log.info("NSE DAILY INGESTION  (interval=1d)")
    log.info("=" * 60)

    client = build_s3_client(cfg)
    ensure_bucket(client, cfg.BUCKET_RAW)
    ensure_bucket(client, cfg.BUCKET_CLEAN)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg.DAILY_LOOKBACK_YEARS * 365)
    scrubber = DataScrubber(max_ffill_periods=5, outlier_std=5.0)

    results = fetch_and_upload_ohlcv(
        client=client,
        tickers=ALL_TICKERS,
        interval="1d",
        start=start,
        end=end,
        raw_bucket=cfg.BUCKET_RAW,
        clean_bucket=cfg.BUCKET_CLEAN,
        scrubber=scrubber,
    )

    ok = sum(v for v in results.values())
    log.info("Daily ingestion complete: %d/%d tickers succeeded.", ok, len(results))
    _print_summary(results)


def run_intraday_ingestion(cfg: Config = Config) -> None:
    """
    Intraday ingestion: last N days of 1-minute OHLCV.
    yfinance free tier: max 7 days of 1m data.
    """
    log.info("=" * 60)
    log.info("NSE INTRADAY INGESTION  (interval=1m)")
    log.info("=" * 60)

    client = build_s3_client(cfg)
    ensure_bucket(client, cfg.BUCKET_RAW)
    ensure_bucket(client, cfg.BUCKET_CLEAN)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg.INTRADAY_LOOKBACK_DAYS)
    scrubber = DataScrubber(max_ffill_periods=10, outlier_std=4.0)

    results = fetch_and_upload_ohlcv(
        client=client,
        tickers=ALL_TICKERS,
        interval="1m",
        start=start,
        end=end,
        raw_bucket=cfg.BUCKET_RAW,
        clean_bucket=cfg.BUCKET_CLEAN,
        scrubber=scrubber,
    )

    ok = sum(v for v in results.values())
    log.info("Intraday ingestion complete: %d/%d tickers succeeded.", ok, len(results))
    _print_summary(results)


def _print_summary(results: Dict[str, bool]) -> None:
    print("\n" + "─" * 50)
    print(f"  {'TICKER':<20} {'STATUS'}")
    print("─" * 50)
    for ticker, ok in sorted(results.items()):
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {ticker:<20} {status}")
    print("─" * 50 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aladdin NSE Ingestion Pipeline")
    parser.add_argument(
        "--mode",
        choices=["daily", "intraday", "both"],
        default="daily",
        help="Ingestion mode: daily EOD, 1m intraday, or both",
    )
    args = parser.parse_args()

    if args.mode in ("daily", "both"):
        run_daily_ingestion()
    if args.mode in ("intraday", "both"):
        run_intraday_ingestion()
