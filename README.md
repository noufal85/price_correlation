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

# Quick test with sample data (50 stocks)
python cli.py run

# Full universe with FMP (requires API key)
export FMP_API_KEY=your_key_here
python cli.py run --source fmp
```

## CLI Usage

Single unified CLI for all operations with detailed logging.

### Full Pipeline

```bash
# Quick test (50 stocks via yfinance)
python cli.py run

# Full universe (all NYSE/NASDAQ via FMP)
export FMP_API_KEY=your_key_here
python cli.py run --source fmp

# Large cap only ($1B+)
python cli.py run --source fmp --market-cap-min 1000000000

# Use config file
python cli.py run --source fmp --config config/sample_filtered.yaml

# DBSCAN instead of hierarchical
python cli.py run --method dbscan
```

### Step-by-Step Execution

Run individual steps and inspect results between each:

```bash
# Step 1: Fetch universe
python cli.py universe --source fmp

# Step 2: Fetch prices
python cli.py prices --days 180

# Step 3: Preprocess (compute returns)
python cli.py preprocess

# Step 4: Compute correlations
python cli.py correlate

# Step 5: Cluster
python cli.py cluster --method hierarchical

# Step 6: Export results
python cli.py export
```

### CLI Options

```bash
python cli.py --help           # Show all commands
python cli.py run --help       # Show run options

# Common options
--source fmp|yfinance          # Data source (default: yfinance)
--config PATH                  # Config file for FMP filters
--output DIR                   # Output directory (default: ./output)
--days N                       # Days of price history (default: 180)
--method hierarchical|dbscan   # Clustering method
--market-cap-min N             # Min market cap in USD (FMP)
--market-cap-max N             # Max market cap in USD (FMP)
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
├── cli.py                 # Unified CLI (run, universe, prices, etc.)
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
