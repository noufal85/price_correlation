# Task Breakdown - Stock Clustering System

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION PHASES                         │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 1: Setup    →  Phase 2: Core    →  Phase 3: ML    →  Phase 4 │
│  (Foundation)         (Data Pipeline)     (Clustering)     (Output) │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Project Setup

### 1.1 Package Structure
```
Create directory layout:
  src/price_correlation/
  tests/
  output/

Create files:
  pyproject.toml (package config, dependencies)
  src/price_correlation/__init__.py
  tests/__init__.py
```

**Acceptance**: `pip install -e .` succeeds

### 1.2 Dependencies Configuration
```
Core deps:
  numpy, pandas, scipy, scikit-learn, pyarrow, yfinance

Dev deps:
  pytest, ruff

Optional:
  tslearn, statsmodels, matplotlib
```

**Acceptance**: All imports work without error

---

## Phase 2: Data Pipeline

### 2.1 Universe Manager
```
Module: src/price_correlation/universe.py

Functions needed:
  get_exchange_tickers(exchange) → list[str]
    - Fetch all tickers for NYSE or NASDAQ
    - Filter to common stocks only (no ETFs, ADRs)

  get_full_universe() → list[str]
    - Combine NYSE + NASDAQ
    - Remove duplicates

Flow:
  [API Call] → [Parse Response] → [Filter by Type] → [Return Tickers]
```

**Acceptance**: Returns 5000+ tickers from real API

### 2.2 Price Ingestion
```
Module: src/price_correlation/ingestion.py

Functions needed:
  fetch_price_history(tickers, start, end) → DataFrame
    - Fetch adjusted close for all tickers
    - Handle failures gracefully (skip bad tickers)
    - Return DataFrame: index=date, columns=tickers

  fetch_single_ticker(ticker, start, end) → Series
    - Helper for single ticker fetch
    - Used by parallel executor

Flow:
  [Ticker List] → [Parallel Fetch] → [Combine] → [Aligned DataFrame]

Parallelization:
  Use concurrent.futures.ThreadPoolExecutor
  Batch requests to avoid rate limits
```

**Acceptance**: Fetches 100+ tickers in parallel, returns aligned DataFrame

### 2.3 Preprocessing
```
Module: src/price_correlation/preprocess.py

Functions needed:
  clean_price_data(df, min_history_pct=0.9) → DataFrame
    - Forward-fill missing values
    - Drop tickers with >10% missing data
    - Align all series to common dates

  compute_log_returns(prices) → DataFrame
    - r_t = ln(P_t) - ln(P_{t-1})
    - Drop first row (NaN)

  zscore_normalize(returns) → DataFrame
    - For each ticker: (x - mean) / std
    - Result has mean≈0, std≈1 per column

  remove_market_factor(returns, n_components=1) → DataFrame
    - PCA to extract market mode
    - Return residuals after removing first PC

Flow:
  [Raw Prices] → [Clean] → [Log Returns] → [Z-Score] → [Optional: De-market]
```

**Acceptance**: Output DataFrame has ~0 mean, ~1 std per column

---

## Phase 3: Correlation & Clustering

### 3.1 Correlation Engine
```
Module: src/price_correlation/correlation.py

Functions needed:
  compute_correlation_matrix(returns_df) → ndarray
    - Pearson correlation between all pairs
    - Shape: (N, N)

  correlation_to_distance(corr_matrix, method="sqrt") → ndarray
    - "sqrt": d = sqrt(2 * (1 - rho))
    - "simple": d = 1 - rho

  get_condensed_distance(returns_matrix) → ndarray
    - Use scipy.spatial.distance.pdist
    - Returns flattened upper triangle
    - More memory efficient

Flow:
  [Returns Matrix] → [Correlation] → [Distance Conversion] → [Distance Matrix]
```

**Acceptance**: Distance matrix is symmetric, diagonal=0, values in [0, 2]

### 3.2 Clustering Engine
```
Module: src/price_correlation/clustering.py

Functions needed:
  cluster_dbscan(distance_matrix, eps, min_samples) → ndarray
    - Returns cluster labels
    - -1 = noise (unclustered)

  find_optimal_eps(distance_matrix, k=5) → float
    - K-distance graph method
    - Find elbow point

  cluster_hierarchical(condensed_dist, method="average") → ndarray
    - Returns scipy linkage matrix Z

  cut_dendrogram(Z, n_clusters=None, threshold=None) → ndarray
    - Extract flat cluster labels

  find_optimal_k(Z, distance_matrix, max_k=30) → int
    - Silhouette score optimization
    - Return best k

Flow (DBSCAN):
  [Distance Matrix] → [Find eps] → [DBSCAN] → [Labels]

Flow (Hierarchical):
  [Condensed Dist] → [Linkage] → [Find k] → [Cut Tree] → [Labels]
```

**Acceptance**: Produces 5-30 clusters with silhouette > 0.1

### 3.3 Validation
```
Module: src/price_correlation/validation.py

Functions needed:
  compute_silhouette(distance_matrix, labels) → float
    - Cluster quality score [-1, 1]

  compute_cluster_stats(labels, tickers) → dict
    - Number of clusters
    - Size distribution
    - Noise count (label=-1)

  generate_tsne_plot(distance_matrix, labels, output_path)
    - 2D visualization
    - Color by cluster
    - Save to file

Flow:
  [Labels + Data] → [Compute Metrics] → [Visualize] → [Report]
```

**Acceptance**: Returns valid silhouette score, generates plot file

---

## Phase 4: Output & Export

### 4.1 Export Module
```
Module: src/price_correlation/export.py

Functions needed:
  export_clusters_json(labels, tickers, path)
    - Format: [{"cluster": 1, "members": [...], "size": N}, ...]

  export_clusters_parquet(labels, tickers, metadata, path)
    - Columns: analysis_date, ticker, cluster_id, listing_status

  export_correlations_parquet(corr_matrix, tickers, threshold, path)
    - Sparse format: only pairs above threshold
    - Columns: ticker_a, ticker_b, correlation

Flow:
  [Labels + Tickers] → [Format] → [Write File]
```

**Acceptance**: Files readable by pandas/DuckDB

### 4.2 Main Pipeline
```
Module: src/price_correlation/pipeline.py

Functions needed:
  run_pipeline(config) → dict
    - Orchestrates all steps
    - Returns summary stats

  load_config(path) → dict
    - Load from YAML/JSON

Config structure:
  start_date, end_date
  clustering_method (dbscan/hierarchical)
  output_format (json/parquet)
  output_path

Flow:
  [Config] → [Universe] → [Ingest] → [Preprocess] →
  [Correlate] → [Cluster] → [Validate] → [Export] → [Summary]
```

**Acceptance**: End-to-end run produces output files

---

## Phase 5: Testing

### 5.1 Integration Tests
```
Location: tests/test_integration.py

Test 1: test_data_pipeline
  - Fetch real prices for 10 tickers
  - Preprocess through z-score
  - Verify output shape and statistics

Test 2: test_correlation_clustering
  - Use real preprocessed data
  - Compute correlation matrix
  - Run both DBSCAN and hierarchical
  - Verify valid cluster labels

Test 3: test_full_pipeline
  - Run complete pipeline on small universe (50 tickers)
  - Verify output files created
  - Verify file contents parseable

Notes:
  - All tests use REAL DATA (no mocks)
  - Combine multiple functions per test
  - May take several minutes due to API calls
```

**Acceptance**: All tests pass with real data

---

## Task Checklist

### Phase 1: Setup
- [ ] 1.1 Create package structure
- [ ] 1.2 Configure dependencies in pyproject.toml

### Phase 2: Data Pipeline
- [ ] 2.1 Implement universe.py
- [ ] 2.2 Implement ingestion.py
- [ ] 2.3 Implement preprocess.py

### Phase 3: Clustering
- [ ] 3.1 Implement correlation.py
- [ ] 3.2 Implement clustering.py
- [ ] 3.3 Implement validation.py

### Phase 4: Output
- [ ] 4.1 Implement export.py
- [ ] 4.2 Implement pipeline.py (orchestrator)

### Phase 5: Testing
- [ ] 5.1 Write integration tests

### Final
- [ ] Run full pipeline on complete universe
- [ ] Generate final output files
- [ ] Commit and push

---

## Dependency Graph

```
                    ┌──────────────┐
                    │   universe   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ingestion   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  preprocess  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ correlation  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ DBSCAN   │ │Hierarchic│ │   DTW    │
        │          │ │   al     │ │(optional)│
        └────┬─────┘ └────┬─────┘ └──────────┘
             │            │
             └─────┬──────┘
                   ▼
            ┌──────────────┐
            │  validation  │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │    export    │
            └──────────────┘
```

---

## Estimated Effort

| Phase | Tasks | Complexity |
|-------|-------|------------|
| 1. Setup | 2 | Low |
| 2. Data Pipeline | 3 | Medium |
| 3. Clustering | 3 | Medium-High |
| 4. Output | 2 | Low |
| 5. Testing | 1 | Medium |

**Critical Path**: Universe → Ingestion → Preprocess → Correlation → Clustering
