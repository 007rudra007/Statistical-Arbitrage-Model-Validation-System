# ── ORE (QuantLib-ORE) XVA Microservice ──────────────────────────────────────
# Wraps the QuantLib-ORE Python-based XVA engine as a FastAPI service.
# Runs standalone so the main API container stays lightweight.
#
# Build:  docker build -f services/ore/Dockerfile -t aladdin-ore .
# Run:    docker-compose up ore -d
# Docs:   http://localhost:8080/docs

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# QuantLib Python wheel (pre-built for Python 3.12)
# QuantLib-ORE is large; we use the pure-NumPy fallback
# and optionally the community ORE wheel when available.
RUN pip install --no-cache-dir \
        fastapi>=0.111.0 \
        uvicorn[standard]>=0.30.0 \
        pydantic>=2.7.0 \
        numpy>=1.24 \
        scipy>=1.10 \
        QuantLib>=1.34 \
        pandas>=2.0 \
        arch>=6.0

COPY risk/         ./risk/
COPY services/ore/ ./services/ore/

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "services.ore.main:app", "--host", "0.0.0.0", "--port", "8080"]
