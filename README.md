# Stock Price Correlation Clustering

Identify groups of correlated stocks across NYSE/NASDAQ using ML clustering algorithms on historical price data.

## What is Stock Clustering?

Stock clustering groups stocks that move together in the market. When two stocks are "correlated," their prices tend to rise and fall at similar times. This system:

1. **Calculates correlation** - Measures how similarly each pair of stocks moves over the past 6 months
2. **Converts to distance** - Highly correlated stocks are "close" to each other; uncorrelated stocks are "far apart"
3. **Groups into clusters** - Uses machine learning algorithms to find natural groupings of similar stocks

**Why is this useful?**
- **Portfolio diversification**: Avoid holding multiple stocks from the same cluster (they'll all drop together)
- **Pairs trading**: Find highly correlated pairs for mean-reversion strategies
- **Sector analysis**: Discover hidden relationships beyond traditional sector classifications
- **Risk management**: Understand which stocks are likely to move together during market stress

## Features

- **Full Stock Universe**: Fetch ALL NYSE/NASDAQ stocks via FMP API with configurable filters
- **Flexible Filtering**: Filter by market cap, volume, sector, industry, exchange
- **Multiple Data Sources**: FMP API (full universe) or yfinance (quick samples)
- **ML Clustering**: DBSCAN and Hierarchical clustering with auto-tuning
- **Export Formats**: JSON and Parquet with DuckDB-ready schemas
- **TimescaleDB Integration**: Store clustering results in TimescaleDB for historical analysis
- **Redis Caching**: Cache API responses to avoid repeated fetches
- **Docker Support**: Run the pipeline in a container (one-shot or interactive)
- **Web Interface**: Browser-based dashboard to trigger runs, view results, and explore data
- **Visualizations**: t-SNE 2D cluster plots

## Web Interface (Recommended)

The easiest way to use the system is via the web interface. Start with Docker:

```bash
# Clone and setup
git clone git@github.com:noufal85/price_correlation.git
cd price_correlation

# Configure environment
cp .env.example .env
# Edit .env and add your FMP_API_KEY

# Start web server
docker-compose up -d

# Open in browser
open http://localhost:5000
```

### Web Features

| Page | Description |
|------|-------------|
| **Dashboard** | Pipeline status, run trigger, market overview, sector performance |
| **Clusters** | View cluster assignments, member stocks, size distribution |
| **Correlations** | Browse correlated pairs, filter by threshold |
| **Stock Detail** | Company profile, price chart, news, financial ratios (from FMP) |
| **Charts** | Cluster sizes, correlation distribution, silhouette history |

### API Endpoints

The web server exposes REST APIs for programmatic access:

| Endpoint | Description |
|----------|-------------|
| `GET /api/runs` | List of analysis runs |
| `GET /api/clusters` | Current cluster assignments |
| `GET /api/correlations` | Correlated pairs |
| `POST /api/pipeline/run` | Trigger pipeline run |
| `GET /api/fmp/profile/<ticker>` | Company profile from FMP |
| `GET /api/fmp/quote/<ticker>` | Real-time quote from FMP |
| `GET /api/fmp/news/<ticker>` | Stock news from FMP |

## Quick Start

```bash
# Clone the repository
git clone git@github.com:noufal85/price_correlation.git
cd price_correlation

# Create virtual environment (one-time setup)
python -m venv venv

# Activate virtual environment (REQUIRED before every session)
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies (one-time setup)
pip install -e ".[dev,full]"

# Launch interactive menu
python cli.py

# For FMP data source, set API key first
export FMP_API_KEY=your_key_here
python cli.py

# When done, deactivate virtual environment
deactivate
```

> **Important**: Always activate the virtual environment (`source venv/bin/activate`) before running any commands.

## CLI Usage

Interactive menu-based CLI for all operations with detailed logging and step-by-step execution.

### Interactive Menu

Simply run `python cli.py` to launch the interactive menu:

```
============================================================
       STOCK CLUSTERING PIPELINE - INTERACTIVE MENU
============================================================

  Current Settings:
    Source:      YFINANCE
    Mode:        Sample (50)
    Days:        180
    Method:      hierarchical
    Output:      ./output

────────────────────────────────────────────────────────────

  Pipeline Steps:

    1  ○  Fetch Universe       - Get list of stocks to analyze
    2  ○  Fetch Prices         - Download historical price data
    3  ○  Preprocess           - Compute returns & normalize
    4  ○  Correlations         - Build correlation matrix
    5  ○  Cluster              - Run clustering algorithm
    6     Export               - Save results to files

────────────────────────────────────────────────────────────

  Actions:

    7     Run Full Pipeline    - Execute all steps (1-6)
    8     Settings             - Change configuration
    9     Clear State          - Reset pipeline state
    0     Exit

============================================================

  Enter choice:
```

### Menu Options

| Key | Action | Description |
|-----|--------|-------------|
| 1-6 | Pipeline Steps | Run individual steps sequentially |
| 7 | Full Pipeline | Run all steps automatically |
| 8 | Settings | Configure data source, method, etc. |
| 9 | Clear State | Reset and start fresh |
| 0 | Exit | Quit the program |

### Settings Menu

Press `8` to access settings where you can configure:

- **Data Source**: `yfinance` (quick test) or `fmp` (full universe)
- **Days of History**: Price data lookback period (default: 180)
- **Clustering Method**: `hierarchical` or `dbscan`
- **Min History %**: Filter stocks with insufficient data
- **Output Directory**: Where to save results
- **Universe Mode** (yfinance): Full or Sample size
- **Market Cap Min** (FMP): Filter by market capitalization
- **Config File** (FMP): Use YAML config for advanced filters

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (choose your platform)
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows PowerShell
venv\Scripts\activate.bat       # Windows CMD
```

### Install Package

```bash
# Basic installation
pip install -e .

# With development tools (pytest, ruff)
pip install -e ".[dev]"

# With all optional dependencies (matplotlib, statsmodels)
pip install -e ".[dev,full]"
```

## Usage

### Running the CLI

```bash
# Step 1: Activate virtual environment (ALWAYS do this first)
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Step 2: Launch interactive menu
python cli.py

# Step 3: When done, deactivate
deactivate
```

### Typical Workflow

```bash
# Start a new session
source venv/bin/activate

# Launch CLI and select options from menu
python cli.py
# → Press 8 to change settings (source, method, etc.)
# → Press 7 to run full pipeline, OR
# → Press 1-6 to run individual steps

# Exit CLI with 0, then deactivate
deactivate
```

### Python API

```python
from price_correlation import run_pipeline, PipelineConfig

# Simple run with defaults
result = run_pipeline()

# Custom configuration
config = PipelineConfig(
    tickers=["AAPL", "MSFT", "GOOGL", "JPM", "BAC"],
    period_months=12,
    clustering_method="hierarchical",  # or "dbscan"
    output_dir="./output",
    visualize=True,
)
result = run_pipeline(config)

print(f"Found {result['n_clusters']} clusters")
print(f"Silhouette score: {result['silhouette_score']:.3f}")
```

### Quick Test

```python
from price_correlation import run_sample_pipeline

# Run on 30 stocks with 6 months data
result = run_sample_pipeline()
```

## Output

After running the pipeline, find results in the output directory. Export formats (JSON/Parquet) are configurable via Settings menu (options 9 and A).

### Output Files

| File | Format | Description |
|------|--------|-------------|
| `stock_clusters.json` | JSON | Clusters grouped by cluster ID with member lists |
| `equity_clusters.json` | JSON | Per-ticker cluster assignments |
| `pair_correlations.json` | JSON | Highly correlated pairs (>0.7) |
| `equity_clusters.parquet` | Parquet | Database-ready cluster assignments |
| `pair_correlations.parquet` | Parquet | Correlated pairs for DuckDB/analytics |
| `cluster_visualization.png` | PNG | t-SNE 2D visualization |

### JSON Formats

**stock_clusters.json** - Grouped by cluster:
```json
[
  {"cluster": 0, "members": ["AAPL", "MSFT", "GOOGL", ...], "size": 15},
  {"cluster": 1, "members": ["JPM", "BAC", "GS", ...], "size": 12},
  {"cluster": -1, "members": ["ODDSTOCK", ...], "size": 3}
]
```

**equity_clusters.json** - Per-ticker assignments:
```json
[
  {"ticker": "AAPL", "cluster_id": 0, "analysis_date": "2024-01-15"},
  {"ticker": "MSFT", "cluster_id": 0, "analysis_date": "2024-01-15"},
  {"ticker": "JPM", "cluster_id": 1, "analysis_date": "2024-01-15"}
]
```

**pair_correlations.json** - Correlated pairs:
```json
[
  {"ticker_a": "GOOGL", "ticker_b": "META", "correlation": 0.89},
  {"ticker_a": "JPM", "ticker_b": "BAC", "correlation": 0.85},
  {"ticker_a": "XOM", "ticker_b": "CVX", "correlation": 0.82}
]
```

### Query with DuckDB

```sql
-- Find stocks in same cluster as AAPL
SELECT ticker FROM 'equity_clusters.parquet'
WHERE cluster_id = (
  SELECT cluster_id FROM 'equity_clusters.parquet'
  WHERE ticker = 'AAPL'
);

-- Find highly correlated pairs
SELECT * FROM 'pair_correlations.parquet'
WHERE correlation > 0.9
ORDER BY correlation DESC;
```

## Running Tests

```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests (uses real data, may take a few minutes)
pytest tests/ -v

# Run specific test
pytest tests/test_integration.py::TestFullPipeline -v

# Deactivate when done
deactivate
```

## Project Structure

```
price_correlation/
├── cli.py                 # Interactive menu CLI
├── pyproject.toml         # Package configuration
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Docker build exclusions
├── .env.example           # Environment template
├── README.md              # This file
├── CLAUDE.md              # Development instructions
├── config/
│   ├── default.yaml       # Default config (full universe)
│   ├── sample_filtered.yaml  # Large cap + sectors example
│   └── sample_smallcap.yaml  # Small cap example
├── scripts/
│   └── init_timescaledb.sql  # Database schema
├── docs/
│   ├── DESIGN.md          # System architecture
│   └── tasks/             # Task breakdown
├── src/price_correlation/
│   ├── fmp_client.py      # FMP API client
│   ├── universe.py        # Ticker list fetching (yfinance)
│   ├── ingestion.py       # Price data (yfinance)
│   ├── preprocess.py      # Returns, normalization
│   ├── correlation.py     # Correlation matrices
│   ├── clustering.py      # DBSCAN, hierarchical
│   ├── validation.py      # Quality metrics
│   ├── export.py          # JSON/Parquet output
│   ├── pipeline.py        # Main orchestrator
│   ├── cache.py           # Redis caching layer
│   ├── db.py              # TimescaleDB client
│   ├── web.py             # Flask web server
│   └── templates/         # HTML templates for web UI
├── tests/
│   ├── test_integration.py
│   ├── test_cache_integration.py
│   └── test_db_integration.py
└── output/                # Generated results
```

## FMP Configuration

Create or modify YAML config files in `config/` directory.

### Filter Options

```yaml
filters:
  # Market cap (USD)
  market_cap:
    min: 1000000000      # $1B minimum
    max: null            # No maximum

  # Average daily volume
  volume:
    min: 100000          # 100K minimum
    max: null

  # Exchanges
  exchanges:
    - NYSE
    - NASDAQ

  # Sectors (null = all)
  sectors:
    - Technology
    - Healthcare
    - Financial Services

  # Only active stocks
  is_actively_trading: true
```

### Example Configs

| Config | Description |
|--------|-------------|
| `default.yaml` | Full universe, no filters |
| `sample_filtered.yaml` | Large cap ($2B+), Tech & Healthcare |
| `sample_smallcap.yaml` | Small cap ($300M-$2B) |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `period_months` | 18 | Lookback period for price data |
| `min_history_pct` | 0.90 | Minimum required data (filter sparse tickers) |
| `clustering_method` | "hierarchical" | "hierarchical" or "dbscan" |
| `remove_market_factor` | False | Remove beta via PCA |
| `correlation_threshold` | 0.7 | Threshold for pair export |
| `visualize` | True | Generate t-SNE plot |

## Dependencies

**Core:**
- numpy, pandas, scipy, scikit-learn
- pyarrow (Parquet support)
- yfinance (price data)
- redis (caching)
- psycopg2-binary (TimescaleDB)

**Optional:**
- matplotlib (visualization)
- statsmodels (cointegration tests)
- tslearn (DTW clustering)

## Docker Usage

Run the pipeline in a Docker container with automatic Redis caching and TimescaleDB export.

### Prerequisites

- Docker and Docker Compose installed
- Redis and TimescaleDB running (optional, for caching and DB export)

### Build and Run

```bash
# Copy environment template and configure
cp .env.example .env
# Edit .env with your FMP_API_KEY and service addresses

# Build the Docker image
docker build -t price-correlation:latest .

# One-shot mode (runs full pipeline)
docker run --rm \
    --env-file .env \
    -v $(pwd)/output:/app/output \
    --network=host \
    price-correlation:latest

# Interactive CLI mode
docker run -it --rm \
    --env-file .env \
    -v $(pwd)/output:/app/output \
    --network=host \
    price-correlation:latest \
    python cli.py
```

### Docker Compose

```bash
# One-shot mode
docker-compose up --build

# Interactive mode
docker-compose run price-correlation python cli.py
```

## Redis Caching

Redis caching reduces API calls by storing fetched data.

### Configuration

Set in `.env` or environment:

```bash
REDIS_HOST=192.168.68.88
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=           # Leave empty if no password
ENABLE_CACHE=true         # Set to false to disable
```

### Cache Behavior

| Data | TTL | Description |
|------|-----|-------------|
| Universe | 24 hours | Stock ticker lists |
| Prices | 6 hours | Historical price data |

### CLI Commands

- Press `C` in the main menu to clear all cache entries
- Toggle caching in Settings menu (option `B`)

### Graceful Degradation

If Redis is unavailable, the pipeline continues without caching - it just fetches from APIs directly.

## TimescaleDB Integration

Store clustering results in TimescaleDB for historical analysis and querying.

### Configuration

Set in `.env` or environment:

```bash
TIMESCALE_HOST=192.168.68.88
TIMESCALE_PORT=5432
TIMESCALE_DB=timescaledb
TIMESCALE_USER=postgres
TIMESCALE_PASSWORD=password
ENABLE_DB_EXPORT=true     # Set to false to disable
```

### Database Schema

Initialize the database with the provided schema:

```bash
psql -h 192.168.68.88 -U postgres -d timescaledb -f scripts/init_timescaledb.sql
```

### Tables

| Table | Description |
|-------|-------------|
| `equity_clusters` | Cluster assignments per analysis run (hypertable) |
| `pair_correlations` | Highly correlated pairs per run (hypertable) |
| `analysis_runs` | Pipeline execution metadata |

### Sample Queries

```sql
-- Latest cluster for AAPL
SELECT cluster_id FROM equity_clusters
WHERE ticker = 'AAPL'
ORDER BY analysis_date DESC LIMIT 1;

-- All stocks in same cluster as AAPL (latest run)
WITH latest AS (
    SELECT cluster_id, analysis_date FROM equity_clusters
    WHERE ticker = 'AAPL' ORDER BY analysis_date DESC LIMIT 1
)
SELECT ec.ticker FROM equity_clusters ec
JOIN latest l ON ec.cluster_id = l.cluster_id AND ec.analysis_date = l.analysis_date;

-- Top correlated pairs (latest run)
SELECT ticker_a, ticker_b, correlation
FROM pair_correlations
WHERE analysis_date = (SELECT MAX(analysis_date) FROM analysis_runs)
ORDER BY correlation DESC LIMIT 10;

-- Cluster history for a ticker
SELECT analysis_date, cluster_id FROM equity_clusters
WHERE ticker = 'AAPL' ORDER BY analysis_date;
```

### CLI Commands

- Press `D` in the main menu to manually trigger database export
- Toggle DB export in Settings menu (option `C`)

### Graceful Degradation

If TimescaleDB is unavailable, the pipeline logs a warning and continues - results are still exported to files.

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `FMP_API_KEY` | - | Financial Modeling Prep API key |
| `REDIS_HOST` | 192.168.68.88 | Redis server host |
| `REDIS_PORT` | 6379 | Redis server port |
| `REDIS_DB` | 0 | Redis database number |
| `REDIS_PASSWORD` | - | Redis password (optional) |
| `TIMESCALE_HOST` | 192.168.68.88 | TimescaleDB host |
| `TIMESCALE_PORT` | 5432 | TimescaleDB port |
| `TIMESCALE_DB` | timescaledb | TimescaleDB database name |
| `TIMESCALE_USER` | postgres | TimescaleDB username |
| `TIMESCALE_PASSWORD` | password | TimescaleDB password |
| `ENABLE_CACHE` | true | Enable/disable Redis caching |
| `ENABLE_DB_EXPORT` | true | Enable/disable TimescaleDB export |

## License

MIT
