"""
ORE XVA Service — FastAPI entry point
"""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.risk import router as risk_router

app = FastAPI(
    title="Aladdin ORE / XVA Service",
    version="1.0.0",
    description="QuantLib-based XVA microservice: CVA, DVA, FVA for IRS and FX Forwards.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Re-expose only the /risk/xva and /risk/options routes
from api.routes.risk import router as risk_router
app.include_router(risk_router)

@app.get("/health")
def health():
    from datetime import datetime, timezone
    return {"status": "ok", "service": "ore-xva",
            "time": datetime.now(timezone.utc).isoformat()}
