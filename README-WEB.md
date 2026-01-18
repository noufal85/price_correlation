# Stock Clustering Web Interface

Web-based interface for running the stock clustering pipeline and viewing results.

## Quick Start

```bash
# 1. Create .env file with your settings
cp .env.example .env
# Edit .env with your FMP_API_KEY and database settings

# 2. Start the web server
docker-compose up -d

# 3. Open browser
http://localhost:5000
```

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Pipeline status, trigger runs, quick stats |
| Clusters | `/clusters` | Browse all clusters and their member stocks |
| Correlations | `/correlations` | View highly correlated stock pairs |
| Stock Detail | `/stock/AAPL` | Detailed view of individual stock with FMP data |
| Charts | `/charts` | Analytics visualizations |

## Dashboard Features

- **Run Pipeline**: Click "Run Pipeline" button to start a new clustering analysis
- **Pipeline Status**: Shows if pipeline is idle, running, or last run time
- **Quick Stats**: Number of stocks, clusters, and average correlation
- **Market Overview**: Sector performance from FMP API

## Cluster View

- Lists all clusters with member count
- Click a cluster to see all stocks in that cluster
- Cluster size distribution chart
- Each stock links to its detail page

## Correlation View

- Shows pairs of stocks with high correlation (>0.8 by default)
- Filter by minimum correlation threshold
- Search for specific tickers
- Click any ticker to view stock details

## Stock Detail Page

Access via `/stock/TICKER` (e.g., `/stock/AAPL`)

Displays:
- **Company Profile**: Name, sector, industry, description
- **Current Quote**: Price, change, volume, market cap
- **Key Ratios**: P/E, P/B, ROE, debt/equity
- **Similar Stocks**: Peers from FMP
- **Price Chart**: 1-year price history
- **Recent News**: Latest news articles

## API Endpoints

### Pipeline Control

```bash
# Check pipeline status
GET /api/pipeline/status

# Run pipeline
POST /api/pipeline/run

# Response:
{
  "status": "running" | "idle",
  "last_run": "2024-01-15T10:30:00",
  "message": "Pipeline started"
}
```

### Database Queries

```bash
# List analysis runs
GET /api/runs
GET /api/runs?limit=10

# Get clusters for a run
GET /api/clusters?date=2024-01-15

# Get correlations
GET /api/correlations?date=2024-01-15&min_corr=0.85

# Get stock history
GET /api/stock/AAPL/clusters
GET /api/stock/AAPL/correlations
```

### FMP Data

```bash
# Company profile
GET /api/fmp/profile/AAPL

# Current quote
GET /api/fmp/quote/AAPL

# Company news
GET /api/fmp/news/AAPL

# Financial ratios
GET /api/fmp/ratios/AAPL

# Peer companies
GET /api/fmp/peers/AAPL

# Sector performance
GET /api/fmp/sector-performance

# Price history
GET /api/fmp/historical/AAPL
```

### Cache Control

```bash
# Clear all cached data
POST /api/cache/clear
```

### Chart Data

```bash
GET /api/charts/cluster-sizes
GET /api/charts/correlation-distribution
GET /api/charts/silhouette-history
```

## Configuration

Environment variables in `.env`:

```bash
# Required
FMP_API_KEY=your_api_key_here

# Database (TimescaleDB)
TIMESCALE_HOST=192.168.68.88
TIMESCALE_PORT=5432
TIMESCALE_DB=timescaledb
TIMESCALE_USER=postgres
TIMESCALE_PASSWORD=password

# Cache (Redis)
REDIS_HOST=192.168.68.88
REDIS_PORT=6379

# Feature toggles
ENABLE_CACHE=true
ENABLE_DB_EXPORT=true

# Web server
WEB_HOST=0.0.0.0
WEB_PORT=5000
```

## Docker Commands

```bash
# Start web server
docker-compose up -d

# View logs
docker-compose logs -f web

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Run CLI instead of web
docker-compose run --rm web python cli.py
```

## Network Configuration

The web server needs to connect to Redis and TimescaleDB. If these services are on your host machine (not in Docker):

**Option 1: Host networking** (recommended for local development)

Edit `docker-compose.yml`:
```yaml
services:
  web:
    # Comment out ports when using host network
    # ports:
    #   - "5000:5000"
    network_mode: host
```

**Option 2: Use host.docker.internal** (macOS/Windows)

In `.env`:
```bash
REDIS_HOST=host.docker.internal
TIMESCALE_HOST=host.docker.internal
```

## Troubleshooting

### Pipeline won't start
- Check logs: `docker-compose logs -f web`
- Verify FMP_API_KEY is set in `.env`

### No data showing
- Run the pipeline first (click "Run Pipeline" on dashboard)
- Check if TimescaleDB is accessible
- Run schema init: `psql -h HOST -U postgres -d timescaledb -f scripts/init_timescaledb.sql`

### FMP data not loading
- Verify FMP_API_KEY is valid
- Check browser console for API errors
- Free FMP tier has rate limits

### Cannot connect to Redis/TimescaleDB
- Verify services are running
- Check host/port in `.env`
- Try `network_mode: host` in docker-compose.yml

## Browser Support

Modern browsers with JavaScript enabled:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
