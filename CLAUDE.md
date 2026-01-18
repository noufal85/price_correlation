# Claude Code Instructions for price_correlation

## Project Overview

Stock clustering system that identifies groups of correlated stocks across NYSE/NASDAQ using ML clustering algorithms on 18-month price history.

---

## Git Workflow

**Repository**: `git@github.com:noufal85/price_correlation.git`

### Rules
- ALWAYS commit changes after completing a task or logical unit of work
- ALWAYS push to remote after commits
- Use descriptive commit messages summarizing the "why"
- Never leave uncommitted work at end of session

### Commit Flow
```
1. Complete task/feature
2. git add <files>
3. git commit -m "descriptive message"
4. git push origin <branch>
```

---

## Task Management

### Rules
- ALWAYS use TodoWrite tool to track tasks before starting work
- Update task status in REAL-TIME as you progress:
  - `pending` → `in_progress` when starting
  - `in_progress` → `completed` immediately when done
- Break complex work into granular, trackable items
- Never batch task completions - mark done as you go

### Example
```
Starting "Implement correlation engine":
  1. Mark "Implement correlation engine" as in_progress
  2. Do the work
  3. Mark as completed IMMEDIATELY
  4. Move to next task
```

---

## Testing Philosophy

### Core Principles

1. **NO MOCKS - EVER**
   - Tests use REAL data, real API calls, real computations
   - If testing data ingestion, fetch actual stock prices
   - If testing clustering, run actual clustering on real returns

2. **INTEGRATION OVER UNIT**
   - Prefer fewer tests that exercise multiple functions together
   - One test should validate an entire workflow, not isolated pieces
   - Tests should mirror how code is used in production

3. **SEPARATE TEST PACKAGE**
   - Tests live in `/tests/` directory (separate package)
   - Not alongside source code
   - Import from main package as external user would

### Test Structure
```
price_correlation/
├── src/
│   └── price_correlation/
│       ├── __init__.py
│       ├── ingestion.py
│       ├── preprocessing.py
│       └── clustering.py
└── tests/                    # Separate package
    ├── __init__.py
    └── test_pipeline.py      # Integration tests
```

### Test Style
```python
# BAD - Too many isolated tests with mocks
def test_fetch_single_ticker():
    mock_api.return_value = fake_data  # NO MOCKS
    ...

def test_compute_returns():
    ...

def test_normalize():
    ...

# GOOD - One integration test, real data, multiple functions
def test_full_preprocessing_pipeline():
    """
    Fetch real prices → compute returns → normalize → validate output
    Tests: fetch_prices, compute_log_returns, zscore_normalize
    """
    # Use real tickers, real API
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = fetch_prices(tickers, "2024-01-01", "2024-06-01")

    returns = compute_log_returns(prices)
    normalized = zscore_normalize(returns)

    # Validate the whole chain
    assert normalized.shape[0] > 100  # Has data
    assert abs(normalized.mean().mean()) < 0.01  # Properly normalized
```

### What to Test
- End-to-end workflows (ingestion → preprocessing → clustering → export)
- Critical edge cases only (empty data, single stock, etc.)
- Output format correctness

### What NOT to Test
- Individual helper functions in isolation
- Mock-based scenarios
- Every possible input combination

---

## Design Documentation

### Rules
- MINIMAL CODE in design docs
- Prefer: flowcharts, diagrams, pseudo-code, logic descriptions
- Design docs explain WHAT and WHY, not exact implementation

### Preferred Formats

**Flow Diagrams (ASCII)**
```
[Input] → [Process A] → [Process B] → [Output]
              ↓
         [Side Effect]
```

**Pseudo-code (not Python)**
```
FUNCTION do_something(input):
    FOR EACH item IN input:
        PROCESS item
        IF condition THEN
            STORE result
    RETURN results
```

**Logic Descriptions**
```
1. Fetch universe of tickers from API
2. For each ticker, retrieve 18 months of adjusted close
3. Compute pairwise correlations
4. Apply DBSCAN with auto-tuned epsilon
5. Export cluster assignments to Parquet
```

### Avoid in Design Docs
```python
# Don't include implementation details like:
def compute_correlation(df: pd.DataFrame) -> np.ndarray:
    returns = df.pct_change().dropna()
    return np.corrcoef(returns.T)
```

---

## Project Structure

```
price_correlation/
├── CLAUDE.md              # This file
├── README.md              # User documentation
├── pyproject.toml         # Package config
├── docs/
│   ├── DESIGN.md          # Architecture & pseudo-code
│   └── tasks/             # Individual task files
│       ├── 00_OVERVIEW.md
│       ├── 01_package_structure.md
│       ├── 02_dependencies.md
│       ├── ... (11 task files)
│       └── 11_tests.md
├── src/
│   └── price_correlation/
│       ├── __init__.py
│       ├── universe.py    # Universe management
│       ├── ingestion.py   # Data fetching
│       ├── preprocess.py  # Returns, normalization
│       ├── correlation.py # Distance matrices
│       ├── clustering.py  # DBSCAN, hierarchical
│       ├── validation.py  # Silhouette, visualization
│       └── export.py      # Parquet, JSON output
├── tests/
│   ├── __init__.py
│   └── test_integration.py
└── output/                # Generated files
    ├── equity_clusters.parquet
    └── stock_clusters.json
```

---

## Code Style

- Type hints on public functions
- Docstrings for modules and classes (brief)
- No excessive comments - code should be self-explanatory
- Prefer composition over inheritance
- Keep functions focused and small

---

## Dependencies

Core:
- numpy, pandas, scipy, scikit-learn
- pyarrow (Parquet)
- yfinance (data ingestion)

Optional:
- tslearn (DTW clustering)
- statsmodels (cointegration)
- matplotlib (visualization)

---

## Setup & Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install package with dev dependencies
pip install -e ".[dev]"

# Install with all optional dependencies (visualization, etc.)
pip install -e ".[dev,full]"
```

## Common Commands

```bash
# IMPORTANT: Always activate venv first
source venv/bin/activate

# Run full pipeline (sample data via yfinance)
python cli.py run

# Run with FMP data source (full universe)
export FMP_API_KEY=your_key
python cli.py run --source fmp

# Run with market cap filter
python cli.py run --source fmp --market-cap-min 1000000000

# Run individual steps
python cli.py universe --source fmp
python cli.py prices
python cli.py preprocess
python cli.py correlate
python cli.py cluster --method dbscan
python cli.py export

# Run tests (real data, may take time)
pytest tests/ -v

# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Deactivate when done
deactivate
```

---

## Reminders for Claude

1. **Tasks**: Update TodoWrite status in real-time
2. **Git**: Commit and push after completing work
3. **Tests**: Real data only, combine functions, separate package
4. **Design**: Diagrams and pseudo-code, minimal actual code
5. **Focus**: Keep implementations simple, avoid over-engineering
