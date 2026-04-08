# Use official low-footprint Python image
FROM python:3.12.5-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

# Install system dependencies (required for some math libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN addgroup --system appgroup && adduser --system --group appuser

# Copy backend codebase
COPY --chown=appuser:appgroup alpha/ ./alpha/
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup backtester/ ./backtester/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup risk/ ./risk/

# Expose FastAPI port
EXPOSE 8000

# Switch to non-root user
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server via Uvicorn explicitly
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
