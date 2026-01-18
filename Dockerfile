# Stock Clustering Pipeline - Docker Image
# Supports both one-shot and interactive modes

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2 and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for layer caching)
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[full]" \
    && pip cache purge

# Copy application code
COPY src/ ./src/
COPY cli.py .
COPY scripts/ ./scripts/

# Copy config if it exists
COPY config/ ./config/ 2>/dev/null || true

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Default environment variables (can be overridden)
ENV ENABLE_CACHE=true
ENV ENABLE_DB_EXPORT=true

# Default command: run full pipeline (one-shot mode)
# For interactive mode: docker run -it ... python cli.py
CMD ["python", "-c", "from price_correlation.pipeline import run_pipeline; run_pipeline()"]
