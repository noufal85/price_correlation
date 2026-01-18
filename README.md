# Stock Price Correlation Clustering

Identify groups of correlated stocks across NYSE/NASDAQ using ML clustering algorithms on historical price data.

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

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -e ".[dev,full]"

# Quick test with sample data (yfinance)
python run.py

# --- OR ---

# Full universe with FMP (requires API key)
export FMP_API_KEY=your_key_here    # Get free key at financialmodelingprep.com
python run_fmp.py
```

## Data Sources

### Option 1: FMP API (Recommended for Full Universe)

Uses [Financial Modeling Prep](https://financialmodelingprep.com) API to fetch the complete stock universe with filters.

```bash
# Get your free API key
# https://financialmodelingprep.com/developer

# Set API key
export FMP_API_KEY=your_key_here

# Run with full universe
python run_fmp.py

# Run with market cap filter ($1B+ stocks)
python run_fmp.py --market-cap-min 1000000000

# Use custom config
python run_fmp.py --config config/sample_filtered.yaml
```

### Option 2: yfinance (Quick Testing)

Uses yfinance for smaller samples (S&P 500, NASDAQ-100).

```bash
# Sample run (50 stocks)
python run.py

# Full S&P 500 + NASDAQ-100
python run.py --full
```

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

### Command Line

```bash
# Activate virtual environment first
source venv/bin/activate

# Sample run (50 stocks, 6 months) - quick test
python run.py

# Full universe (S&P 500 + NASDAQ-100, 18 months)
python run.py --full

# Custom tickers
python run.py --tickers AAPL MSFT GOOGL JPM BAC XOM CVX

# Custom lookback period
python run.py --months 12

# Use DBSCAN instead of hierarchical clustering
python run.py --method dbscan

# Custom output directory
python run.py --output ./results

# Combine options
python run.py --full --method dbscan --output ./results
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

After running the pipeline, find results in the output directory:

| File | Description |
|------|-------------|
| `stock_clusters.json` | Human-readable cluster assignments |
| `equity_clusters.parquet` | Database-ready format with metadata |
| `pair_correlations.parquet` | Highly correlated pairs (>0.7) |
| `cluster_visualization.png` | t-SNE 2D visualization |

### JSON Format

```json
[
  {"cluster": 0, "members": ["AAPL", "MSFT", "GOOGL", ...], "size": 15},
  {"cluster": 1, "members": ["JPM", "BAC", "GS", ...], "size": 12},
  {"cluster": -1, "members": ["ODDSTOCK", ...], "size": 3}
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
# Activate virtual environment
source venv/bin/activate

# Run all tests (uses real data, may take a few minutes)
pytest tests/ -v

# Run specific test
pytest tests/test_integration.py::TestFullPipeline -v
```

## Project Structure

```
price_correlation/
├── run.py                 # CLI runner (yfinance)
├── run_fmp.py             # CLI runner (FMP full universe)
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
│   ├── universe.py        # Ticker list fetching
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
