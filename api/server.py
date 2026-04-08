import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timezone

# ── Import Routers ─────────────────────────────────────────────────
from api.routes.data import router as data_router
from api.routes.risk import router as risk_router
from api.routes.agent import router as agent_router
from api.routes.portfolio import router as portfolio_router

app = FastAPI(
    title="Aladdin System API",
    version="6.0.0",
    description="Quantitative trading platform: portfolio optimizer, compliance engine, AI agents.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Routers ──────────────────────────────────────────────────
app.include_router(data_router)
app.include_router(risk_router)
app.include_router(agent_router)
app.include_router(portfolio_router)

# ── Prometheus Telemetry (Phase 6) ──────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
    print("[API] Prometheus metrics enabled at /metrics")
except ImportError:
    print("[API] Warning: prometheus_fastapi_instrumentator missing.")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "6.0.0",
        "time": datetime.now(timezone.utc).isoformat(),
        "modules": ["data", "risk", "agent", "portfolio"]
    }

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  ALADDIN SYSTEM API - http://localhost:8000")
    print("  Docs -> http://localhost:8000/docs")
    print("=" * 50 + "\n")
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000)
