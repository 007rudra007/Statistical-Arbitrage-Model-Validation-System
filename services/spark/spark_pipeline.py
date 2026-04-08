"""
Phase 2 – Step 6: Spark Pipeline (Dockerized)
===============================================
PySpark job that:
  1. Reads raw Parquet tick data from MinIO / S3 (aladdin-raw bucket)
  2. Computes OHLCV aggregations (1m → 5m, 15m, 1h, 1d)
  3. Calculates intraday features: VWAP, returns, rolling vol
  4. Writes Delta-Lake-compatible Parquet files to aladdin-clean bucket

Designed to run as a Kubernetes CronJob (see infra/k8s/spark-cronjob.yaml).

Local debug:
    docker run --rm \\
      -e MINIO_ENDPOINT=http://host.docker.internal:9000 \\
      aladdin-spark python services/spark/spark_pipeline.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("spark_pipeline")

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Config from environment ───────────────────────────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
BUCKET_RAW     = os.getenv("BUCKET_RAW", "aladdin-raw")
BUCKET_DELTA   = os.getenv("BUCKET_DELTA", "aladdin-delta")
RUN_DATE       = os.getenv("RUN_DATE", datetime.now(timezone.utc).strftime("%Y-%m-%d"))


# ── SparkSession Factory ──────────────────────────────────────────────────────
def build_spark(app_name: str = "AladdinSparkPipeline") -> SparkSession:
    """
    Create SparkSession configured for MinIO (S3A) access.
    The hadoop-aws and aws-java-sdk jars are baked into the Docker image.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        # Delta Lake (write _delta_log metadata alongside Parquet)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Performance
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ── Pipeline Steps ────────────────────────────────────────────────────────────
def read_raw_parquet(spark: SparkSession, ticker: str, interval: str) -> DataFrame:
    """Read all raw Parquet partitions for a ticker from MinIO."""
    safe_ticker = ticker.replace("^", "idx_").replace(".", "_")
    path = f"s3a://{BUCKET_RAW}/nse/{safe_ticker}/{interval}/"
    log.info("Reading raw: %s", path)
    df = spark.read.parquet(path)
    return df


def add_ohlcv_features(df: DataFrame) -> DataFrame:
    """
    Compute derived OHLCV features:
      - log_return: log(Close_t / Close_{t-1})
      - typical_price: (H + L + C) / 3
      - vwap: cumulative VWAP from start of day
      - rolling_vol_20: 20-bar rolling std of log_return
      - intraday_range_pct: (High - Low) / Open * 100
    """
    w_order = Window.orderBy("__index_level_0__")
    w_vol = Window.orderBy("__index_level_0__").rowsBetween(-19, 0)

    df = df.withColumn(
        "log_return",
        F.log(F.col("Close") / F.lag("Close", 1).over(w_order))
    )
    df = df.withColumn(
        "typical_price",
        (F.col("High") + F.col("Low") + F.col("Close")) / 3.0
    )
    # VWAP = Σ(typical_price × volume) / Σ(volume)
    df = df.withColumn(
        "tp_vol", F.col("typical_price") * F.col("Volume")
    )
    w_cum = Window.orderBy("__index_level_0__").rowsBetween(Window.unboundedPreceding, 0)
    df = df.withColumn(
        "vwap",
        F.sum("tp_vol").over(w_cum) / F.sum("Volume").over(w_cum)
    )
    df = df.withColumn("vwap", F.col("vwap").cast(DoubleType()))
    df = df.drop("tp_vol")

    # 20-bar rolling volatility (annualised if daily, raw if intraday)
    df = df.withColumn(
        "rolling_vol_20",
        F.stddev("log_return").over(w_vol)
    )
    df = df.withColumn(
        "intraday_range_pct",
        ((F.col("High") - F.col("Low")) / F.col("Open")) * 100.0
    )

    return df


def resample_to_bar(df: DataFrame, minutes: int) -> DataFrame:
    """
    Re-aggregate 1m OHLCV DataFrame into N-minute bars.
    Uses floor(timestamp / N-minutes) as the bar key.
    """
    ts_col = "__index_level_0__"
    bar_seconds = minutes * 60

    df = df.withColumn(
        "bar_ts",
        (F.unix_timestamp(F.col(ts_col)) / bar_seconds).cast("long") * bar_seconds
    )
    df = df.withColumn("bar_ts", F.to_timestamp(F.col("bar_ts")))

    agg = df.groupBy("bar_ts").agg(
        F.first("Open").alias("Open"),
        F.max("High").alias("High"),
        F.min("Low").alias("Low"),
        F.last("Close").alias("Close"),
        F.sum("Volume").alias("Volume"),
    ).orderBy("bar_ts")

    return agg


def write_delta(df: DataFrame, ticker: str, interval_label: str) -> None:
    """Write DataFrame in Delta Lake format to MinIO aladdin-delta bucket."""
    safe_ticker = ticker.replace("^", "idx_").replace(".", "_")
    path = f"s3a://{BUCKET_DELTA}/nse/{safe_ticker}/{interval_label}/"
    log.info("Writing Delta: %s", path)

    (
        df.write
        .format("delta")
        .mode("overwrite")          # idempotent daily re-run
        .option("overwriteSchema", "true")
        .save(path)
    )
    log.info("Delta write complete → %s", path)


# ── Main Orchestration ────────────────────────────────────────────────────────
def run_pipeline(tickers: list[str] | None = None) -> None:
    """
    Full pipeline:
      1. Read 1m raw Parquet  (from aladdin-raw)
      2. Compute features      (log_return, VWAP, rolling vol, …)
      3. Write 1m Delta        (to aladdin-delta)
      4. Resample → 5m, 15m, 1h, 1d and write each as Delta
    """
    from data.nse_ingestion import ALL_TICKERS
    tickers = tickers or ALL_TICKERS

    spark = build_spark()
    log.info("Spark version: %s", spark.version)
    log.info("Run date: %s | Tickers: %d", RUN_DATE, len(tickers))

    for ticker in tickers:
        try:
            log.info("──────────────────────── %s ────────────────────────", ticker)

            # 1. Read
            df_1m = read_raw_parquet(spark, ticker, "1m")
            row_count = df_1m.count()
            log.info("[%s] Raw 1m rows: %d", ticker, row_count)

            if row_count == 0:
                log.warning("[%s] No 1m data – skipping.", ticker)
                continue

            # 2. Features on 1m
            df_1m_feat = add_ohlcv_features(df_1m)

            # 3. Write 1m Delta
            write_delta(df_1m_feat, ticker, "1m")

            # 4. Resample & write higher timeframes
            for mins, label in [(5, "5m"), (15, "15m"), (60, "1h")]:
                df_resampled = resample_to_bar(df_1m, mins)
                write_delta(df_resampled, ticker, label)

            # Daily read from raw (already have daily from ingestion, just feature-engineer it)
            try:
                df_1d = read_raw_parquet(spark, ticker, "1d")
                df_1d_feat = add_ohlcv_features(df_1d)
                write_delta(df_1d_feat, ticker, "1d")
            except Exception as exc:
                log.warning("[%s] Daily feature pass failed: %s", ticker, exc)

        except Exception as exc:
            log.error("[%s] Pipeline failed: %s", ticker, exc, exc_info=True)

    spark.stop()
    log.info("Spark pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
