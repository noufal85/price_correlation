# Task 09: Export Module

**Phase**: Output
**Status**: ⬜ Not Started
**Depends on**: Task 08

---

## Objective

Export clustering results to JSON and Parquet formats for database consumption.

## Module

`src/price_correlation/export.py`

## Functions

### export_clusters_json(labels, tickers, output_path)
```
Output format:
  [
    {"cluster": 0, "members": ["AAPL", "MSFT", ...], "size": 45},
    {"cluster": 1, "members": ["JPM", "BAC", ...], "size": 32},
    {"cluster": -1, "members": ["ODD1", ...], "size": 128}
  ]

Flow:
  [Group tickers by label]
           ↓
  [Build list of dicts]
           ↓
  [Write JSON with indent=2]
```

### export_clusters_parquet(labels, tickers, metadata, output_path)
```
Schema:
  ┌──────────────┬─────────┬────────────────────────────┐
  │ Column       │ Type    │ Description                │
  ├──────────────┼─────────┼────────────────────────────┤
  │ analysis_date│ DATE    │ When clustering was run    │
  │ ticker       │ STRING  │ Stock symbol               │
  │ cluster_id   │ INT     │ Cluster assignment (-1=noise)│
  │ listing_status│ STRING │ Active/Delisted (optional) │
  │ sector       │ STRING  │ GICS sector (optional)     │
  └──────────────┴─────────┴────────────────────────────┘

Flow:
  [Build DataFrame from labels + tickers]
           ↓
  [Add metadata columns if provided]
           ↓
  [Write Parquet with snappy compression]
```

### export_correlations_parquet(corr_matrix, tickers, threshold, output_path)
```
Sparse export: only pairs above correlation threshold

Schema:
  ┌──────────────┬─────────┬────────────────────────────┐
  │ Column       │ Type    │ Description                │
  ├──────────────┼─────────┼────────────────────────────┤
  │ ticker_a     │ STRING  │ First stock                │
  │ ticker_b     │ STRING  │ Second stock               │
  │ correlation  │ FLOAT   │ Pearson correlation        │
  └──────────────┴─────────┴────────────────────────────┘

Flow:
  [Iterate upper triangle of matrix]
           ↓
  [Keep pairs where |corr| >= threshold]
           ↓
  [Build DataFrame]
           ↓
  [Write Parquet]

Threshold = 0.7 recommended
  → Reduces millions of pairs to thousands
```

## File Size Comparison

```
Format         Clusters (5000 stocks)    Correlations (sparse)
─────────────────────────────────────────────────────────────
JSON           ~200 KB                   Not recommended
CSV            ~100 KB                   ~5 MB
Parquet        ~30 KB                    ~1 MB

Parquet advantages:
  - Columnar (fast queries)
  - Compressed (smaller)
  - Schema enforced (no type errors)
```

## Query Examples (DuckDB)

```sql
-- Find all stocks in same cluster as AAPL
SELECT ticker
FROM 'equity_clusters.parquet'
WHERE cluster_id = (
  SELECT cluster_id
  FROM 'equity_clusters.parquet'
  WHERE ticker = 'AAPL'
);

-- Find highly correlated pairs
SELECT *
FROM 'pair_correlations.parquet'
WHERE correlation > 0.9
ORDER BY correlation DESC;
```

## Acceptance Criteria

- [ ] JSON output is valid and readable
- [ ] Parquet files load correctly in pandas
- [ ] Parquet files queryable via DuckDB
- [ ] Sparse correlation export reduces file size significantly
- [ ] Schema matches documented structure

## Output Files

```
output/
├── stock_clusters.json         # Human readable
├── equity_clusters.parquet     # Database ready
└── pair_correlations.parquet   # Sparse correlations
```
