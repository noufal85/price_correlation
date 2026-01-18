# Task 13: Unified CLI

**Phase**: Interface
**Status**: ✅ Completed
**Depends on**: All previous tasks

---

## Objective

Create single unified command-line interface for all pipeline operations.

## Implementation

File: `cli.py`

## Commands

```
python cli.py run          # Full pipeline
python cli.py universe     # Step 1: Fetch universe
python cli.py prices       # Step 2: Fetch prices
python cli.py preprocess   # Step 3: Compute returns
python cli.py correlate    # Step 4: Correlation matrix
python cli.py cluster      # Step 5: Clustering
python cli.py export       # Step 6: Export results
```

## Features

- Step-by-step execution with state persistence
- Colored terminal output
- Progress bars for long operations
- Detailed statistics at each step
- Support for both yfinance and FMP data sources
- Config file support for FMP filters

## Key Options

```
--source fmp|yfinance      # Data source
--config PATH              # Config file
--days N                   # Price history days
--method hierarchical|dbscan
--market-cap-min N         # FMP filter
--output DIR               # Output directory
```

## Acceptance Criteria

- [x] Single entry point for all operations
- [x] Step-by-step execution works
- [x] State persists between steps
- [x] Colored output and progress bars
- [x] Detailed logging at each step
- [x] Both data sources work
