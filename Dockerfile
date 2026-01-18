# Stock Clustering Pipeline - Docker Image
# Supports web server, one-shot, and interactive CLI modes

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2 and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy all source files needed for installation
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies (non-editable for Docker)
RUN pip install --no-cache-dir ".[full]" \
    && pip cache purge

# Copy remaining application files
COPY cli.py .
COPY scripts/ ./scripts/

# Create output and config directories (config is volume-mounted in docker-compose)
RUN mkdir -p /app/output /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Default environment variables (can be overridden)
ENV ENABLE_CACHE=true
ENV ENABLE_DB_EXPORT=true
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=5000

# Expose web port
EXPOSE 5000

# Default command: run web server
# For CLI: docker run -it ... python cli.py
# For one-shot pipeline: docker run ... python -c "from price_correlation.pipeline import run_pipeline; run_pipeline()"
# Single worker to share pipeline state, multiple threads for concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", "--timeout", "600", "price_correlation.web:app"]
