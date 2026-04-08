"""
Phase 2 – Data Router
======================
FastAPI router exposing data pipeline endpoints:
  GET  /data/health               – MinIO connectivity check
  GET  /data/tickers              – List available tickers
  POST /data/ingest               – Trigger ad-hoc ingestion for a ticker list
  GET  /data/ohlcv/{ticker}       – Fetch OHLCV parquet from clean bucket
  GET  /data/portfolios           – List portfolios for an owner
  POST /data/portfolios           – Upload a new portfolio
  GET  /data/portfolios/{pid}     – Get a specific portfolio
  DELETE /data/portfolios/{pid}   – Delete a portfolio
  GET  /data/presigned/{bucket}/{key:path}  – Presigned download URL
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])

# ── Lazy import storage to avoid crash if MinIO unreachable at startup ─────────
def _get_storage():
    try:
        from data.storage import get_storage_client
        return get_storage_client()
    except Exception as exc:
        log.warning("StorageClient unavailable: %s", exc)
        return None


# ── Models ────────────────────────────────────────────────────────────────────
class Position(BaseModel):
    ticker: str
    weight: float = Field(ge=0.0, le=1.0)
    sector: Optional[str] = None
    notional: Optional[float] = None


class PortfolioUploadRequest(BaseModel):
    portfolio_id: str = Field(..., min_length=3, max_length=64,
                              pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str
    owner: str = "system"
    currency: str = "INR"
    positions: List[Position]
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    tickers: Optional[List[str]] = None      # None → all NIFTY50 + BankNifty
    mode: str = Field(default="daily", pattern="^(daily|intraday|both)$")


class IngestResponse(BaseModel):
    triggered: bool
    mode: str
    tickers: List[str]
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_storage():
    sc = _get_storage()
    if sc is None:
        raise HTTPException(
            status_code=503,
            detail="Storage backend (MinIO/S3) is not reachable. "
                   "Ensure MINIO_ENDPOINT env var is set and MinIO is running."
        )
    return sc


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health", summary="MinIO / S3 connectivity check")
def data_health() -> Dict[str, Any]:
    """
    Returns connectivity status of the object storage backend.
    Used by Kubernetes readiness probes and Grafana dashboards.
    """
    sc = _get_storage()
    if sc is None:
        return {"status": "unavailable", "endpoint": "unknown"}
    return sc.health_check()


@router.get("/tickers", summary="List available tickers in the data lake")
def list_tickers() -> Dict[str, Any]:
    """
    Returns the full universe of tickers the ingestion pipeline covers:
    NIFTY50 top constituents + BankNifty constituents + index proxies.
    """
    from data.nse_ingestion import ALL_TICKERS, BANKNIFTY_CONSTITUENTS, INDEX_TICKERS
    from data.fetcher import NIFTY_TOP_CONSTITUENTS
    return {
        "total": len(ALL_TICKERS),
        "tickers": ALL_TICKERS,
        "groups": {
            "nifty50": NIFTY_TOP_CONSTITUENTS,
            "banknifty": BANKNIFTY_CONSTITUENTS,
            "indices": INDEX_TICKERS,
        },
    }


@router.post("/ingest", response_model=IngestResponse,
             summary="Trigger ad-hoc NSE data ingestion")
def trigger_ingest(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Triggers the NSE ingestion pipeline for the requested tickers and mode.
    Runs asynchronously in a background task so the request returns immediately.

    - **mode=daily**: Fetch EOD OHLCV (last 5 years)
    - **mode=intraday**: Fetch 1m bars (last 7 days)
    - **mode=both**: Both of the above
    """
    from data.nse_ingestion import (
        ALL_TICKERS, run_daily_ingestion, run_intraday_ingestion, Config
    )

    tickers = req.tickers or ALL_TICKERS

    def _run():
        try:
            cfg = Config
            if req.mode in ("daily", "both"):
                run_daily_ingestion(cfg)
            if req.mode in ("intraday", "both"):
                run_intraday_ingestion(cfg)
        except Exception as exc:
            log.error("Background ingestion error: %s", exc, exc_info=True)

    background_tasks.add_task(_run)

    return IngestResponse(
        triggered=True,
        mode=req.mode,
        tickers=tickers,
        message=f"Ingestion started in background for {len(tickers)} tickers "
                f"(mode={req.mode}). Check server logs for progress.",
    )


@router.get("/ohlcv/{ticker}", summary="Fetch OHLCV from clean data lake")
def get_ohlcv(
    ticker: str,
    interval: str = Query(default="1d", pattern="^(1m|5m|15m|1h|1d)$"),
    limit: int = Query(default=252, ge=1, le=5000),
) -> Dict[str, Any]:
    """
    Returns the latest N rows of OHLCV data for a ticker from the clean bucket.
    Data is fetched from MinIO aladdin-clean / aladdin-delta.

    Falls back to live yfinance fetch if the data lake is unavailable.
    """
    sc = _get_storage()

    # First try data lake
    if sc and sc.health_check()["status"] == "ok":
        try:
            from data.nse_ingestion import _s3_key
            import pyarrow.parquet as pq
            import io

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            key = _s3_key(ticker, interval, date_str)
            resp = sc.client.get_object(Bucket=sc.cfg.BUCKET_CLEAN, Key=key)
            import pandas as pd
            import pyarrow.parquet as pq
            buf = io.BytesIO(resp["Body"].read())
            df = pq.read_table(buf).to_pandas()
            df = df.tail(limit)
            rows = df.reset_index().to_dict(orient="records")
            return {
                "ticker": ticker,
                "interval": interval,
                "source": "data_lake",
                "rows": len(rows),
                "data": [{k: (str(v) if hasattr(v, 'isoformat') else v)
                          for k, v in r.items()} for r in rows],
            }
        except Exception as exc:
            log.warning("[%s] Data lake miss (%s) → falling back to yfinance", ticker, exc)

    # Fallback: live yfinance
    try:
        import yfinance as yf
        from datetime import timedelta
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7 if interval == "1m" else 365 * 5)
        df = yf.download(ticker, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.tail(limit)
        rows = df.reset_index().to_dict(orient="records")
        return {
            "ticker": ticker,
            "interval": interval,
            "source": "yfinance_live",
            "rows": len(rows),
            "data": [{k: (str(v) if hasattr(v, 'isoformat') else
                          (float(v) if hasattr(v, '__float__') else v))
                      for k, v in r.items()} for r in rows],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OHLCV fetch failed: {exc}")


# ── Portfolio endpoints ───────────────────────────────────────────────────────

@router.get("/portfolios", summary="List portfolios for an owner")
def list_portfolios(owner: str = "system") -> Dict[str, Any]:
    sc = _require_storage()
    try:
        sc.setup_all_buckets()
        ids = sc.list_portfolios(owner)
        return {"owner": owner, "count": len(ids), "portfolio_ids": ids}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/portfolios", summary="Upload a new portfolio")
def upload_portfolio(req: PortfolioUploadRequest) -> Dict[str, Any]:
    sc = _require_storage()
    try:
        sc.setup_all_buckets()
        payload = {
            "name": req.name,
            "currency": req.currency,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "positions": [p.model_dump(exclude_none=True) for p in req.positions],
            "metadata": req.metadata or {},
        }
        key = sc.upload_portfolio(req.portfolio_id, payload, user=req.owner)
        return {
            "status": "uploaded",
            "portfolio_id": req.portfolio_id,
            "owner": req.owner,
            "s3_key": key,
            "positions": len(req.positions),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/portfolios/{portfolio_id}", summary="Get a specific portfolio")
def get_portfolio(portfolio_id: str, owner: str = "system") -> Dict[str, Any]:
    sc = _require_storage()
    try:
        portfolio = sc.download_portfolio(portfolio_id, user=owner)
        return {"portfolio_id": portfolio_id, "owner": owner, **portfolio}
    except sc.client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/portfolios/{portfolio_id}", summary="Delete a portfolio")
def delete_portfolio(portfolio_id: str, owner: str = "system") -> Dict[str, Any]:
    sc = _require_storage()
    try:
        key = f"portfolios/{owner}/{portfolio_id}.json"
        sc.client.delete_object(Bucket=sc.cfg.BUCKET_PORTFOLIOS, Key=key)
        return {"status": "deleted", "portfolio_id": portfolio_id, "owner": owner}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/presigned/{bucket}/{key:path}", summary="Generate presigned URL")
def get_presigned_url(
    bucket: str,
    key: str,
    expires: int = Query(default=3600, ge=60, le=86400),
) -> Dict[str, Any]:
    """Generate a short-lived presigned URL for browser-side direct downloads."""
    sc = _require_storage()
    # Safety: only allow aladdin-* buckets
    if not bucket.startswith("aladdin-"):
        raise HTTPException(status_code=400, detail="Only aladdin-* buckets are allowed.")
    try:
        url = sc.presigned_url(bucket, key, operation="get_object", expires=expires)
        return {
            "url": url,
            "expires_in_seconds": expires,
            "bucket": bucket,
            "key": key,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
