"""
Aladdin Auth Service — FastAPI + JWT
Endpoints:
  POST /auth/login   → returns signed JWT
  GET  /auth/me      → validates bearer token, returns user info
  GET  /auth/health  → liveness probe
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-aws-ssm")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# ── Password helpers (using bcrypt directly — avoids passlib 1.7/bcrypt 4.x conflict) ──
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def verify_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed)

# ── In-memory user store — lazy-initialized to avoid module-load-time bcrypt cost ──
_RAW_USERS = {
    "admin":  {"password": "aladdin123", "role": "admin",  "full_name": "Aladdin Admin"},
    "trader": {"password": "trader123",  "role": "trader", "full_name": "Quant Trader"},
}

def _build_users() -> dict[str, dict]:
    return {
        username: {
            "username": username,
            "hashed_password": hash_password(data["password"]),
            "role": data["role"],
            "full_name": data["full_name"],
        }
        for username, data in _RAW_USERS.items()
    }

# Built once at startup (not at import time)
USERS: dict[str, dict] = {}

# ── Pydantic models ───────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    role: str

class UserInfo(BaseModel):
    username: str
    full_name: str
    role: str

# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload["exp"] = expire
    payload["iat"] = datetime.now(timezone.utc)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aladdin Auth Service",
    description="JWT authentication microservice for the Aladdin trading platform",
    version="1.0.0",
    docs_url="/auth/docs",
    redoc_url="/auth/redoc",
)

@app.on_event("startup")
async def startup_event() -> None:
    """Build hashed user table once on startup (not at import time)."""
    global USERS
    USERS = _build_users()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ── Dependency: current user from bearer token ────────────────────────────────
def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> dict:
    payload = decode_token(credentials.credentials)
    username: str | None = payload.get("sub")
    if username is None or username not in USERS:
        raise HTTPException(status_code=401, detail="User not found")
    return USERS[username]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/auth/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "service": "auth", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(body: LoginRequest) -> TokenResponse:
    user = USERS.get(body.username)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return TokenResponse(access_token=token, role=user["role"])

@app.get("/auth/me", response_model=UserInfo, tags=["auth"])
def me(current_user: Annotated[dict, Depends(get_current_user)]) -> UserInfo:
    return UserInfo(
        username=current_user["username"],
        full_name=current_user["full_name"],
        role=current_user["role"],
    )

@app.get("/auth/verify", tags=["auth"])
def verify(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
    """Used by API gateway / other services to validate tokens internally."""
    return {"valid": True, "username": current_user["username"], "role": current_user["role"]}
