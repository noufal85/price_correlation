# Task 10: Pipeline Orchestrator

**Phase**: Output
**Status**: ⬜ Not Started
**Depends on**: Task 09

---

## Objective

Create main entry point that orchestrates all pipeline stages.

## Module

`src/price_correlation/pipeline.py`

## Functions

### run_pipeline(config) → dict
```
Main orchestrator function

Input: config dict with all parameters
Output: summary dict with results

Flow diagram:
  ┌─────────────────────────────────────────────┐
  │               START                          │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  1. Load config                              │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  2. Fetch universe                           │
  │     get_full_universe()                      │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  3. Ingest prices                            │
  │     fetch_price_history(tickers, dates)      │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  4. Preprocess                               │
  │     clean → log_returns → zscore             │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  5. Compute correlations                     │
  │     correlation_matrix → distance_matrix     │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  6. Cluster                                  │
  │     DBSCAN or Hierarchical                   │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  7. Validate                                 │
  │     silhouette, stats, visualization         │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  8. Export                                   │
  │     JSON / Parquet                           │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │               DONE                           │
  │     Return summary                           │
  └─────────────────────────────────────────────┘
```

### load_config(path) → dict
```
Load configuration from YAML or JSON file

Supports:
  - config.yaml
  - config.json
  - Environment variable overrides
```

## Configuration Schema

```yaml
# config.yaml

# Data settings
data:
  start_date: "2024-07-01"
  end_date: "2025-12-31"
  min_history_pct: 0.90

# Preprocessing
preprocessing:
  remove_market_factor: true
  n_components_remove: 1

# Clustering
clustering:
  method: "dbscan"           # or "hierarchical"
  linkage: "average"         # for hierarchical
  n_clusters: null           # auto-detect if null

# Output
output:
  format: "parquet"          # or "json"
  path: "./output"
  correlation_threshold: 0.7

# Visualization
visualization:
  enabled: true
  tsne_perplexity: 30
```

## Return Value

```python
{
    "n_stocks_input": 5500,
    "n_stocks_processed": 4800,
    "n_clusters": 15,
    "n_noise": 342,
    "silhouette_score": 0.28,
    "output_files": [
        "output/stock_clusters.json",
        "output/equity_clusters.parquet"
    ],
    "execution_time_seconds": 145.3
}
```

## CLI Entry Point

```
# Optional: Add CLI via click or argparse

python -m price_correlation.pipeline --config config.yaml

# Or with inline args
python -m price_correlation.pipeline \
    --start-date 2024-07-01 \
    --end-date 2025-12-31 \
    --method dbscan
```

## Error Handling

```
Pipeline should handle:
  - API failures → retry or skip tickers
  - Empty universe → raise clear error
  - Clustering failure → fall back to hierarchical
  - Export failure → log and continue
```

## Acceptance Criteria

- [ ] Runs end-to-end without manual intervention
- [ ] Produces output files
- [ ] Returns meaningful summary
- [ ] Logs progress at each step
- [ ] Handles errors gracefully

## Usage Example

```python
from price_correlation.pipeline import run_pipeline

config = {
    "data": {"start_date": "2024-07-01", "end_date": "2025-12-31"},
    "clustering": {"method": "dbscan"},
    "output": {"format": "parquet", "path": "./output"}
}

result = run_pipeline(config)
print(f"Found {result['n_clusters']} clusters")
```
