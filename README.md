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
- **Visualizations**: t-SNE 2D cluster plots

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
├── README.md              # This file
├── CLAUDE.md              # Development instructions
├── config/
│   ├── default.yaml       # Default config (full universe)
│   ├── sample_filtered.yaml  # Large cap + sectors example
│   └── sample_smallcap.yaml  # Small cap example
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
│   └── pipeline.py        # Main orchestrator
├── tests/
│   └── test_integration.py
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

**Optional:**
- matplotlib (visualization)
- statsmodels (cointegration tests)
- tslearn (DTW clustering)

## License

MIT
