# Stock Price Correlation Clustering

Identify groups of correlated stocks across NYSE/NASDAQ using ML clustering algorithms on historical price data.

## Overview

Stock clustering groups stocks that move together in the market. When two stocks are "correlated," their prices tend to rise and fall at similar times. This system:

1. **Fetches stock universe** - Gets all NYSE/NASDAQ stocks via FMP API with configurable filters
2. **Downloads price history** - Retrieves 18 months of adjusted close prices
3. **Computes correlations** - Calculates pairwise correlation matrix
4. **Clusters stocks** - Groups similar stocks using ML algorithms
5. **Exports results** - Saves to JSON, Parquet, and TimescaleDB

**Use Cases:**
- **Portfolio diversification**: Avoid holding multiple stocks from the same cluster
- **Pairs trading**: Find highly correlated pairs for mean-reversion strategies
- **Sector analysis**: Discover hidden relationships beyond traditional sector classifications
- **Risk management**: Understand which stocks move together during market stress

## Features

### Clustering Algorithms
| Method | Description |
|--------|-------------|
| **Hierarchical** | Agglomerative clustering with dendrogram cutting |
| **HDBSCAN** | Density-based clustering, handles noise well |
| **K-Means PCA** | K-Means on PCA-reduced correlation space |
| **Louvain** | Community detection on correlation graph |
| **DBSCAN** | Classic density-based spatial clustering |
| **Run All** | Execute all methods and compare results |

### Data Sources
- **FMP API** - Full NYSE/NASDAQ universe with filtering (market cap, volume, sector)
- **yfinance** - Quick testing with sample stocks

### Infrastructure
- **Redis Caching** - Cache API responses to avoid repeated fetches
- **TimescaleDB** - Store historical clustering results for analysis
- **Docker** - Containerized deployment with docker-compose

### Web Interface
Modern web dashboard with:
- Sidebar navigation
- Real-time system status (Cache/DB connectivity)
- Interactive charts and visualizations
- Stock detail modals with price charts

## Quick Start

### Docker (Recommended)

```bash
# Clone repository
git clone git@github.com:noufal85/price_correlation.git
cd price_correlation

# Configure environment
cp .env.example .env
# Edit .env - add FMP_API_KEY and configure Redis/TimescaleDB hosts

# Start web server
docker-compose up -d --build

# Open browser
open http://localhost:5000
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev,full]"

# Set API key
export FMP_API_KEY=your_key_here

# Launch CLI
python cli.py
```

## Web Interface

Access at http://localhost:5000 after starting Docker.

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | System overview, recent runs, quick stats |
| **Pipeline** | One-click full pipeline execution with configuration |
| **Steps** | Step-by-step pipeline control with status indicators |
| **Clusters** | View clusters with insights panel and detail modals |
| **Correlations** | Search correlated pairs with stock detail popups |
| **Charts** | Visualizations: cluster sizes, correlation distribution |

### Clusters Page Features
- **Insights Panel** - Silhouette score, cluster count, size stats, execution time
- **Clustering Method Dropdown** - Switch between results from different algorithms
- **Clickable Cluster Titles** - Opens modal with sector distribution and member list

### Correlations Page Features
- **Search** - Find stocks and view their top correlated pairs
- **Stock Modal** - Click any ticker to see profile, price chart, cluster info, peers
- **Dual Price Chart** - Compare two correlated stocks normalized to 100

### Steps Page Features
- **Pipeline Control** - Run individual steps or full pipeline
- **Status Indicators** - Green/red status lines showing completion time and duration
- **Configuration Panel** - Set data source, clustering method, filters
- **Execution Logs** - Real-time progress updates

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/clusters` | GET | Cluster assignments with insights |
| `/api/clusters?method=hdbscan` | GET | Results for specific clustering method |
| `/api/stock/<ticker>/correlations` | GET | Correlated pairs for a stock |
| `/api/profiles/batch?tickers=AAPL,MSFT` | GET | Batch profile fetch |
| `/api/pipeline/run` | POST | Trigger pipeline run |
| `/api/pipeline/status` | GET | Current pipeline status |
| `/api/steps/run/<step>` | POST | Run specific pipeline step |
| `/api/steps/session/<id>` | GET | Session status with step details |
| `/api/cache/status` | GET | Redis connection status |
| `/api/db/status` | GET | TimescaleDB connection status |

## Configuration

### Environment Variables (.env)

```bash
# API Key (required for FMP data source)
FMP_API_KEY=your_api_key_here

# Redis Cache
REDIS_HOST=host.docker.internal  # Use actual IP if not using Docker
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# TimescaleDB
TIMESCALE_HOST=host.docker.internal
TIMESCALE_PORT=5432
TIMESCALE_DB=timescaledb
TIMESCALE_USER=postgres
TIMESCALE_PASSWORD=password

# Feature Toggles
ENABLE_CACHE=true
ENABLE_DB_EXPORT=true
```

### Pipeline Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `data_source` | fmp | Data source: `fmp`, `fmp_filtered`, `yfinance` |
| `period_months` | 18 | Price history lookback period |
| `clustering_method` | hierarchical | Algorithm to use |
| `clustering_methods` | [hierarchical] | List of methods (for multi-method runs) |
| `min_history_pct` | 0.90 | Filter stocks with insufficient data |
| `correlation_threshold` | 0.7 | Threshold for pair export |
| `max_stocks` | 0 | Limit universe size (0 = no limit) |

### FMP Filters (config/*.yaml)

```yaml
filters:
  market_cap:
    min: 1000000000    # $1B minimum
    max: null
  volume:
    min: 100000        # 100K daily volume
  exchanges:
    - NYSE
    - NASDAQ
  sectors:
    - Technology
    - Healthcare
  is_actively_trading: true
```

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `equity_clusters.json` | JSON | Per-ticker cluster assignments |
| `equity_clusters.parquet` | Parquet | Database-ready cluster data |
| `pair_correlations.json` | JSON | Highly correlated pairs |
| `pair_correlations.parquet` | Parquet | Pairs for analytics |
| `stock_clusters.json` | JSON | Clusters grouped by ID |

### Sample Output

**equity_clusters.json:**
```json
[
  {"ticker": "AAPL", "cluster_id": 0, "analysis_date": "2026-01-21"},
  {"ticker": "MSFT", "cluster_id": 0, "analysis_date": "2026-01-21"},
  {"ticker": "JPM", "cluster_id": 1, "analysis_date": "2026-01-21"}
]
```

**pair_correlations.json:**
```json
[
  {"ticker_a": "GOOGL", "ticker_b": "META", "correlation": 0.89},
  {"ticker_a": "JPM", "ticker_b": "BAC", "correlation": 0.85}
]
```

## Database Schema (TimescaleDB)

```sql
-- Cluster assignments (hypertable)
CREATE TABLE equity_clusters (
    analysis_date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    cluster_id INTEGER NOT NULL,
    clustering_method VARCHAR(50) DEFAULT 'hierarchical',
    PRIMARY KEY (analysis_date, ticker, clustering_method)
);

-- Correlated pairs (hypertable)
CREATE TABLE pair_correlations (
    analysis_date DATE NOT NULL,
    ticker_a VARCHAR(20) NOT NULL,
    ticker_b VARCHAR(20) NOT NULL,
    correlation DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (analysis_date, ticker_a, ticker_b)
);

-- Pipeline run metadata
CREATE TABLE analysis_runs (
    analysis_date DATE PRIMARY KEY,
    n_stocks_processed INTEGER,
    n_clusters INTEGER,
    silhouette_score DOUBLE PRECISION,
    clustering_method VARCHAR(50),
    execution_time_seconds DOUBLE PRECISION
);
```

### Sample Queries

```sql
-- Find stocks in same cluster as AAPL
SELECT ticker FROM equity_clusters
WHERE cluster_id = (
    SELECT cluster_id FROM equity_clusters
    WHERE ticker = 'AAPL'
    ORDER BY analysis_date DESC LIMIT 1
)
AND analysis_date = (SELECT MAX(analysis_date) FROM equity_clusters);

-- Top correlated pairs from latest run
SELECT ticker_a, ticker_b, correlation
FROM pair_correlations
WHERE analysis_date = (SELECT MAX(analysis_date) FROM analysis_runs)
ORDER BY correlation DESC
LIMIT 20;

-- Cluster history for a ticker
SELECT analysis_date, cluster_id, clustering_method
FROM equity_clusters
WHERE ticker = 'AAPL'
ORDER BY analysis_date DESC;
```

## Project Structure

```
price_correlation/
├── cli.py                      # Interactive CLI
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # Container definition
├── .env.example                # Environment template
├── pyproject.toml              # Package config
├── README.md                   # This file
├── CLAUDE.md                   # Development instructions
│
├── config/                     # FMP filter configurations
│   ├── default.yaml
│   └── sample_filtered.yaml
│
├── scripts/
│   └── init_timescaledb.sql    # Database schema
│
├── src/price_correlation/
│   ├── __init__.py
│   ├── pipeline.py             # Main orchestrator
│   ├── pipeline_steps.py       # Step-by-step execution
│   ├── pipeline_state.py       # State management
│   ├── clustering.py           # All clustering algorithms
│   ├── correlation.py          # Correlation computation
│   ├── preprocess.py           # Data preprocessing
│   ├── ingestion.py            # Price data fetching
│   ├── universe.py             # Stock universe
│   ├── fmp_client.py           # FMP API client
│   ├── export.py               # JSON/Parquet export
│   ├── cache.py                # Redis caching
│   ├── db.py                   # TimescaleDB client
│   ├── web.py                  # Flask web server
│   └── templates/              # HTML templates
│       ├── base.html           # Layout with sidebar
│       ├── index.html          # Dashboard
│       ├── pipeline.html       # One-click pipeline
│       ├── steps.html          # Step-by-step control
│       ├── clusters.html       # Cluster viewer
│       ├── correlations.html   # Correlation search
│       ├── charts.html         # Visualizations
│       └── stock_detail.html   # Stock page
│
├── tests/
│   └── test_integration.py     # Integration tests
│
└── output/                     # Generated results
```

## Docker Commands

```bash
# Start web server
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after code changes
docker-compose down && docker-compose up -d --build

# Run CLI interactively
docker-compose run --rm web python cli.py
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,full]"

# Run tests
pytest tests/ -v

# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/
```

## Dependencies

**Core:**
- Flask, Gunicorn (web server)
- pandas, numpy, scipy (data processing)
- scikit-learn (ML algorithms)
- hdbscan, python-louvain (additional clustering)
- yfinance, requests (data fetching)
- redis (caching)
- psycopg2-binary (PostgreSQL/TimescaleDB)
- pyarrow (Parquet support)

**Optional:**
- matplotlib (visualization)
- statsmodels (statistical tests)
- tslearn (time series clustering)

## License

MIT
