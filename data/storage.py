"""
Phase 2 – Step 7: MinIO Storage Client
========================================
boto3-based client for:
  - Portfolio JSON uploads/downloads (aladdin-portfolios bucket)
  - IAM policy helpers for bucket-level isolation
  - Presigned URL generation for UI access

This module also documents the MinIO Helm installation procedure in its
module docstring. See helm/minio/ chart and infra/k8s/minio-secrets.yaml.

MinIO Helm install (one-time, run in the cluster):
    helm repo add minio https://charts.min.io/
    helm repo update
    helm install minio minio/minio \\
      --namespace aladdin \\
      --create-namespace \\
      -f helm/minio/values.yaml

Verify:
    kubectl port-forward svc/minio 9000:9000 -n aladdin &
    python -c "from data.storage import StorageClient; StorageClient().health_check()"
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
class StorageConfig:
    """
    Reads from environment. Safe defaults point to local MinIO.
    In AWS prod, set MINIO_ENDPOINT=https://s3.amazonaws.com
    and use real IAM credentials.
    """
    ENDPOINT: str     = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    ACCESS_KEY: str   = os.getenv("MINIO_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
    SECRET_KEY: str   = os.getenv("MINIO_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    REGION: str       = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

    # Buckets
    BUCKET_RAW: str         = os.getenv("BUCKET_RAW",         "aladdin-raw")
    BUCKET_CLEAN: str       = os.getenv("BUCKET_CLEAN",        "aladdin-clean")
    BUCKET_DELTA: str       = os.getenv("BUCKET_DELTA",        "aladdin-delta")
    BUCKET_PORTFOLIOS: str  = os.getenv("BUCKET_PORTFOLIOS",   "aladdin-portfolios")
    BUCKET_REPORTS: str     = os.getenv("BUCKET_REPORTS",      "aladdin-reports")

    ALL_BUCKETS: List[str] = [
        BUCKET_RAW, BUCKET_CLEAN, BUCKET_DELTA,
        BUCKET_PORTFOLIOS, BUCKET_REPORTS,
    ]


# ── IAM Policy Templates ──────────────────────────────────────────────────────
def portfolio_uploader_policy(portfolio_bucket: str) -> Dict:
    """
    Least-privilege IAM policy for a service that only uploads portfolios.
    Attach to: ECR task role / Kubernetes ServiceAccount via IRSA.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PortfolioUpload",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject"],
                "Resource": f"arn:aws:s3:::{portfolio_bucket}/portfolios/*",
            },
            {
                "Sid": "ListBucket",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{portfolio_bucket}",
                "Condition": {"StringLike": {"s3:prefix": ["portfolios/*"]}},
            },
        ],
    }


def data_reader_policy(*buckets: str) -> Dict:
    """Read-only policy across the data lake buckets."""
    resources = [f"arn:aws:s3:::{b}" for b in buckets] + \
                [f"arn:aws:s3:::{b}/*" for b in buckets]
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DataLakeRead",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": resources,
            }
        ],
    }


# ── StorageClient ─────────────────────────────────────────────────────────────
class StorageClient:
    """
    High-level client wrapping MinIO / AWS S3 for Aladdin data operations.

    Example usage:
        sc = StorageClient()
        sc.setup_all_buckets()

        # Upload a portfolio
        portfolio = {"name": "Test", "positions": [{"ticker": "HDFCBANK.NS", "weight": 0.3}]}
        sc.upload_portfolio("my_portfolio", portfolio)

        # Download it back
        pf = sc.download_portfolio("my_portfolio")

        # Save risk report as Parquet
        df = pd.DataFrame(...)
        sc.upload_parquet_report("var_report_2024-01-01", df)
    """

    def __init__(self, cfg: type = StorageConfig):
        self.cfg = cfg
        self._client: Optional[boto3.client] = None

    @property
    def client(self) -> boto3.client:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> boto3.client:
        is_local = (
            "localhost" in self.cfg.ENDPOINT or
            "minio" in self.cfg.ENDPOINT
        )
        kwargs: Dict[str, Any] = dict(
            endpoint_url=self.cfg.ENDPOINT,
            aws_access_key_id=self.cfg.ACCESS_KEY,
            aws_secret_access_key=self.cfg.SECRET_KEY,
            region_name=self.cfg.REGION,
        )
        if is_local:
            kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})
        return boto3.client("s3", **kwargs)

    # ── Bucket Management ─────────────────────────────────────────────────────
    def setup_all_buckets(self) -> None:
        """Create all required buckets if they don't exist. Call once on startup."""
        for bucket in self.cfg.ALL_BUCKETS:
            self._ensure_bucket(bucket)

    def _ensure_bucket(self, bucket: str) -> None:
        try:
            self.client.head_bucket(Bucket=bucket)
            log.debug("Bucket exists: %s", bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                self.client.create_bucket(Bucket=bucket)
                log.info("Created bucket: %s", bucket)
            else:
                raise

    def list_buckets(self) -> List[str]:
        """Return list of all bucket names."""
        resp = self.client.list_buckets()
        return [b["Name"] for b in resp.get("Buckets", [])]

    # ── Portfolio Upload / Download ───────────────────────────────────────────
    def upload_portfolio(
        self,
        portfolio_id: str,
        portfolio: Dict[str, Any],
        user: str = "system",
    ) -> str:
        """
        Upload a portfolio JSON dict to MinIO.

        Args:
            portfolio_id: Unique identifier (e.g. 'my_portfolio_v1')
            portfolio: Dict with at minimum {"name": str, "positions": [...]}
            user: Optional owner tag

        Returns:
            S3 key of the uploaded object.
        """
        key = f"portfolios/{user}/{portfolio_id}.json"
        body = json.dumps(portfolio, indent=2, default=str).encode("utf-8")
        self.client.put_object(
            Bucket=self.cfg.BUCKET_PORTFOLIOS,
            Key=key,
            Body=body,
            ContentType="application/json",
            Metadata={
                "portfolio_id": portfolio_id,
                "owner": user,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        log.info("Portfolio uploaded → s3://%s/%s", self.cfg.BUCKET_PORTFOLIOS, key)
        return key

    def download_portfolio(
        self,
        portfolio_id: str,
        user: str = "system",
    ) -> Dict[str, Any]:
        """Download a portfolio JSON from MinIO."""
        key = f"portfolios/{user}/{portfolio_id}.json"
        resp = self.client.get_object(Bucket=self.cfg.BUCKET_PORTFOLIOS, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))

    def list_portfolios(self, user: str = "system") -> List[str]:
        """List all portfolio IDs for a user."""
        prefix = f"portfolios/{user}/"
        resp = self.client.list_objects_v2(
            Bucket=self.cfg.BUCKET_PORTFOLIOS, Prefix=prefix
        )
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        return [k.replace(prefix, "").replace(".json", "") for k in keys]

    # ── Parquet Reports ───────────────────────────────────────────────────────
    def upload_parquet_report(self, report_id: str, df: pd.DataFrame) -> str:
        """Upload a DataFrame as Parquet to the reports bucket."""
        table = pa.Table.from_pandas(df, preserve_index=True)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        key = f"reports/{report_id}.parquet"
        self.client.put_object(
            Bucket=self.cfg.BUCKET_REPORTS,
            Key=key,
            Body=buf.read(),
            ContentType="application/octet-stream",
        )
        log.info("Report uploaded → s3://%s/%s", self.cfg.BUCKET_REPORTS, key)
        return key

    def download_parquet_report(self, report_id: str) -> pd.DataFrame:
        """Download a Parquet report from the reports bucket."""
        key = f"reports/{report_id}.parquet"
        resp = self.client.get_object(Bucket=self.cfg.BUCKET_REPORTS, Key=key)
        buf = io.BytesIO(resp["Body"].read())
        return pq.read_table(buf).to_pandas()

    # ── Presigned URLs ────────────────────────────────────────────────────────
    def presigned_url(
        self,
        bucket: str,
        key: str,
        operation: str = "get_object",
        expires: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for browser/UI direct access.
        operation: 'get_object' | 'put_object'
        """
        url = self.client.generate_presigned_url(
            ClientMethod=operation,
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
        return url

    # ── Health Check ──────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        """
        Verify MinIO connectivity. Returns status dict.
        Used by FastAPI /data/health endpoint.
        """
        try:
            buckets = self.list_buckets()
            return {
                "status": "ok",
                "endpoint": self.cfg.ENDPOINT,
                "buckets": buckets,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            return {
                "status": "error",
                "endpoint": self.cfg.ENDPOINT,
                "error": str(exc),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }


# ── Module-level singleton ────────────────────────────────────────────────────
_default_client: Optional[StorageClient] = None


def get_storage_client() -> StorageClient:
    """FastAPI dependency: returns the shared StorageClient singleton."""
    global _default_client
    if _default_client is None:
        _default_client = StorageClient()
    return _default_client


# ── CLI / smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    sc = StorageClient()

    print("\n" + "═" * 60)
    print("  STORAGE CLIENT SMOKE TEST")
    print("═" * 60)

    # Health check
    health = sc.health_check()
    print(f"\n  Health: {health['status'].upper()}")
    print(f"  Endpoint: {health['endpoint']}")

    if health["status"] != "ok":
        print("\n  MinIO not reachable. Start it with:")
        print("    docker-compose -f docker-compose.yml up minio -d")
        sys.exit(1)

    # Setup buckets
    sc.setup_all_buckets()
    print(f"\n  Buckets: {sc.list_buckets()}")

    # Portfolio round-trip
    test_portfolio = {
        "name": "Test NIFTY Portfolio",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "positions": [
            {"ticker": "HDFCBANK.NS", "weight": 0.30, "sector": "Financials"},
            {"ticker": "RELIANCE.NS",  "weight": 0.25, "sector": "Energy"},
            {"ticker": "INFY.NS",      "weight": 0.20, "sector": "IT"},
            {"ticker": "TCS.NS",       "weight": 0.15, "sector": "IT"},
            {"ticker": "ITC.NS",       "weight": 0.10, "sector": "FMCG"},
        ],
    }

    key = sc.upload_portfolio("test_portfolio_v1", test_portfolio)
    print(f"\n  Portfolio uploaded: {key}")

    downloaded = sc.download_portfolio("test_portfolio_v1")
    print(f"  Portfolio downloaded: {downloaded['name']}")
    print(f"  Positions: {len(downloaded['positions'])}")

    # Parquet round-trip
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        "ticker": ["HDFCBANK.NS", "RELIANCE.NS"],
        "var_95": [0.028, 0.031],
        "var_99": [0.042, 0.048],
        "expected_shortfall": [0.055, 0.062],
    })
    sc.upload_parquet_report("test_var_report", df)
    df2 = sc.download_parquet_report("test_var_report")
    print(f"\n  Report round-trip:  {df2.shape}  ✓")

    print("\n  All storage tests PASSED ✓\n")
