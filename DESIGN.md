# Stock Clustering System Design Document

## Document Comparison Summary

| Aspect | gemini_research.md | Clustering Stocks.txt |
|--------|-------------------|----------------------|
| **Scope** | Comprehensive institutional-grade system | Focused step-by-step guide |
| **Clustering** | DBSCAN (primary) + Hierarchical + K-Means | Hierarchical only |
| **Distance Metric** | Correlation + DTW options | Correlation distance √(2(1-ρ)) |
| **Preprocessing** | Log-returns + Z-score + PCA | Log-returns only |
| **Data Export** | Apache Parquet + SQL schema | JSON |
| **Survivorship Bias** | Addressed via LISTING_STATUS API | Not explicitly addressed |
| **Cointegration** | Hybrid approach (cluster then test) | Not covered |
| **Validation** | Silhouette + Cophenetic + Economic | Manual/domain inspection |

---

## Requirements Specification

### Functional Requirements

1. **Data Acquisition**
   - Retrieve daily adjusted close prices for all NYSE/NASDAQ stocks
   - Cover 18-month historical window
   - Handle survivorship bias by including delisted stocks
   - Handle corporate actions (splits, dividends) via adjusted prices

2. **Data Preprocessing**
   - Align all time series to common trading day index
   - Handle missing data (forward-fill strategy)
   - Filter stocks with insufficient history (>10% missing data)
   - Transform prices to log-returns for stationarity
   - Apply Z-score normalization

3. **Correlation/Distance Computation**
   - Compute pairwise Pearson correlation matrix
   - Convert to distance matrix: `d = √(2(1-ρ))` or `d = 1-ρ`
   - Support optional DTW distance for lead-lag detection

4. **Clustering**
   - Primary: DBSCAN for noise filtering (stocks that don't belong anywhere)
   - Secondary: Hierarchical clustering for interpretable dendrogram
   - Support for determining optimal cluster count (silhouette, elbow)

5. **Validation**
   - Calculate silhouette scores
   - Optional: Compare clusters to GICS sector codes
   - Generate visualization (t-SNE 2D plot)

6. **Output/Persistence**
   - Export cluster assignments to Parquet and/or JSON
   - Store pairwise correlations (sparse, significant pairs only)
   - Schema suitable for SQL querying

### Non-Functional Requirements

- Handle 5,000-7,000 stocks efficiently
- Distance matrix computation parallelized
- Memory-efficient for large N (use condensed distance format)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STOCK CLUSTERING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   DATA       │───▶│  PREPROCESS  │───▶│  CORRELATION │       │
│  │   INGESTION  │    │  MODULE      │    │  ENGINE      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Universe    │    │  Returns     │    │  Distance    │       │
│  │  Definition  │    │  Matrix      │    │  Matrix      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                      ┌──────────────────────────────────────┐   │
│                      │         CLUSTERING ENGINE            │   │
│                      │  ┌─────────┐  ┌─────────┐  ┌───────┐│   │
│                      │  │ DBSCAN  │  │ Hierarch│  │ Valid ││   │
│                      │  │         │  │ ical    │  │ ation ││   │
│                      │  └─────────┘  └─────────┘  └───────┘│   │
│                      └──────────────────────────────────────┘   │
│                                         │                        │
│                                         ▼                        │
│                      ┌──────────────────────────────────────┐   │
│                      │         OUTPUT MODULE                │   │
│                      │   Parquet / JSON / SQL Export        │   │
│                      └──────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Design (Pseudo-code)

### Module 1: Universe Definition

```
CLASS UniverseManager:

    FUNCTION get_universe(start_date, end_date):
        # Fetch listing status as of start_date to avoid survivorship bias
        active_tickers = fetch_listing_status(date=start_date, status="active")
        delisted_tickers = fetch_listing_status(date=start_date, status="delisted")

        # Combine for complete point-in-time universe
        universe = active_tickers UNION delisted_tickers

        # Filter by exchange
        universe = FILTER universe WHERE exchange IN ("NYSE", "NASDAQ")

        # Filter out non-equity types (ETFs, ADRs if desired)
        universe = FILTER universe WHERE asset_type == "Stock"

        RETURN universe

    FUNCTION fetch_listing_status(date, status):
        # Call Alpha Vantage LISTING_STATUS endpoint or similar
        # Returns: list of {ticker, name, exchange, ipo_date, delisting_date, status}
        CALL API with date parameter
        RETURN parsed_result
```

### Module 2: Data Ingestion

```
CLASS DataIngestion:

    FUNCTION fetch_price_data(tickers, start_date, end_date):
        price_data = EMPTY DataFrame indexed by date

        FOR EACH ticker IN tickers (PARALLEL):
            TRY:
                daily_prices = fetch_adjusted_close(ticker, start_date, end_date)
                price_data[ticker] = daily_prices
            CATCH DataNotFound:
                LOG warning for ticker
                CONTINUE

        RETURN price_data

    FUNCTION fetch_adjusted_close(ticker, start, end):
        # Use yfinance, Alpha Vantage, or Databento
        # MUST use adjusted close to handle splits/dividends
        raw_data = API_CALL(ticker, start, end)
        RETURN raw_data["Adj Close"]
```

### Module 3: Preprocessing

```
CLASS Preprocessor:

    FUNCTION clean_and_transform(price_df, min_history_pct=0.90):

        # Step 1: Align to common trading day index
        trading_days = get_trading_calendar(start_date, end_date)
        price_df = price_df.reindex(trading_days)

        # Step 2: Forward-fill missing values (no look-ahead bias)
        price_df = price_df.forward_fill()

        # Step 3: Filter stocks with insufficient data
        required_days = LENGTH(trading_days) * min_history_pct
        valid_stocks = []
        FOR EACH ticker IN price_df.columns:
            non_null_count = COUNT non-null values for ticker
            IF non_null_count >= required_days:
                valid_stocks.APPEND(ticker)

        price_df = price_df[valid_stocks]

        # Step 4: Compute log returns
        # r_t = ln(P_t) - ln(P_{t-1})
        log_prices = LOG(price_df)
        returns_df = log_prices.DIFF().DROP_FIRST_ROW()

        # Step 5: Z-score normalize each stock's return series
        FOR EACH ticker IN returns_df.columns:
            mean = MEAN(returns_df[ticker])
            std = STD(returns_df[ticker])
            returns_df[ticker] = (returns_df[ticker] - mean) / std

        # Step 6: Drop any remaining NaN (edge cases)
        returns_df = returns_df.DROP_NA(axis=1, how="any")

        RETURN returns_df

    FUNCTION apply_pca_demarket(returns_matrix, n_components_remove=1):
        # Remove market factor (first principal component)
        # This isolates idiosyncratic correlations

        pca = FIT_PCA(returns_matrix, n_components=n_components_remove)
        market_component = pca.TRANSFORM(returns_matrix)
        reconstructed = pca.INVERSE_TRANSFORM(market_component)

        residual_returns = returns_matrix - reconstructed

        RETURN residual_returns
```

### Module 4: Correlation & Distance Engine

```
CLASS CorrelationEngine:

    FUNCTION compute_correlation_matrix(returns_df):
        # returns_df: columns are tickers, rows are dates
        # Compute NxN Pearson correlation matrix

        returns_matrix = returns_df.VALUES.TRANSPOSE()  # Shape: (N_stocks, T_days)
        corr_matrix = NUMPY.CORRCOEF(returns_matrix)    # Shape: (N, N)

        RETURN corr_matrix

    FUNCTION compute_distance_matrix(corr_matrix, method="sqrt"):
        # Convert correlation to distance

        IF method == "sqrt":
            # d = sqrt(2 * (1 - rho))  -- proper metric
            dist_matrix = SQRT(2 * (1 - corr_matrix))
        ELSE IF method == "simple":
            # d = 1 - rho
            dist_matrix = 1 - corr_matrix

        RETURN dist_matrix

    FUNCTION compute_condensed_distance(returns_matrix, metric="correlation"):
        # For scipy hierarchical clustering, use condensed form
        # More memory efficient: N*(N-1)/2 instead of N*N

        condensed = SCIPY.PDIST(returns_matrix, metric=metric)
        RETURN condensed

    FUNCTION compute_dtw_distance(returns_matrix, window=10):
        # Dynamic Time Warping for lead-lag detection
        # Use FastDTW for efficiency

        N = NUMBER_OF_STOCKS(returns_matrix)
        dist_matrix = ZEROS(N, N)

        FOR i FROM 0 TO N-1 (PARALLEL):
            FOR j FROM i+1 TO N-1:
                distance = FASTDTW(returns_matrix[i], returns_matrix[j],
                                   radius=window)
                dist_matrix[i,j] = distance
                dist_matrix[j,i] = distance

        RETURN dist_matrix
```

### Module 5: Clustering Engine

```
CLASS ClusteringEngine:

    FUNCTION cluster_dbscan(distance_matrix, eps, min_samples):
        # DBSCAN: density-based, handles noise
        # Stocks in low-density regions labeled as -1 (noise)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = dbscan.FIT_PREDICT(distance_matrix)

        # Labels: cluster_id (0,1,2,...) or -1 for noise
        RETURN labels

    FUNCTION find_optimal_eps(distance_matrix, k=5):
        # Elbow method for DBSCAN eps parameter
        # Compute k-distance graph

        k_distances = []
        FOR EACH point IN distance_matrix:
            sorted_distances = SORT(point)
            k_dist = sorted_distances[k]  # k-th nearest neighbor distance
            k_distances.APPEND(k_dist)

        k_distances = SORT(k_distances, descending=True)

        # Find elbow point (maximum curvature)
        elbow_index = find_elbow(k_distances)
        optimal_eps = k_distances[elbow_index]

        RETURN optimal_eps

    FUNCTION cluster_hierarchical(condensed_distance, method="average"):
        # Agglomerative hierarchical clustering
        # Returns linkage matrix Z

        # method options: 'single', 'complete', 'average', 'ward'
        Z = SCIPY.HIERARCHY.LINKAGE(condensed_distance, method=method)

        RETURN Z

    FUNCTION cut_dendrogram(Z, n_clusters=None, threshold=None):
        # Extract flat clusters from linkage matrix

        IF n_clusters IS NOT None:
            labels = SCIPY.HIERARCHY.FCLUSTER(Z, t=n_clusters, criterion="maxclust")
        ELSE IF threshold IS NOT None:
            labels = SCIPY.HIERARCHY.FCLUSTER(Z, t=threshold, criterion="distance")

        RETURN labels

    FUNCTION find_optimal_clusters(Z, max_k=30):
        # Use silhouette score to find optimal number of clusters

        best_k = 2
        best_score = -1

        FOR k FROM 2 TO max_k:
            labels = cut_dendrogram(Z, n_clusters=k)
            score = SILHOUETTE_SCORE(distance_matrix, labels, metric="precomputed")

            IF score > best_score:
                best_score = score
                best_k = k

        RETURN best_k, best_score
```

### Module 6: Validation

```
CLASS ClusterValidator:

    FUNCTION compute_silhouette(distance_matrix, labels):
        # Silhouette score: [-1, 1], higher is better
        score = SKLEARN.SILHOUETTE_SCORE(distance_matrix, labels,
                                          metric="precomputed")
        RETURN score

    FUNCTION compute_cophenetic(condensed_distance, Z):
        # How well dendrogram preserves pairwise distances
        coph_dist = SCIPY.HIERARCHY.COPHENET(Z)
        correlation = PEARSON_CORR(condensed_distance, coph_dist)
        RETURN correlation

    FUNCTION validate_against_sectors(labels, tickers, sector_map):
        # Compare clusters to GICS sectors
        # sector_map: {ticker: sector_name}

        cluster_sector_matrix = {}

        FOR cluster_id IN UNIQUE(labels):
            cluster_tickers = tickers WHERE label == cluster_id
            sectors_in_cluster = [sector_map[t] FOR t IN cluster_tickers]

            # Count sector distribution
            sector_counts = COUNT_VALUES(sectors_in_cluster)
            dominant_sector = MAX_KEY(sector_counts)
            purity = sector_counts[dominant_sector] / LEN(cluster_tickers)

            cluster_sector_matrix[cluster_id] = {
                "dominant_sector": dominant_sector,
                "purity": purity,
                "sector_distribution": sector_counts
            }

        RETURN cluster_sector_matrix

    FUNCTION visualize_tsne(distance_matrix, labels, tickers):
        # 2D visualization using t-SNE

        tsne = TSNE(n_components=2, metric="precomputed", perplexity=30)
        embedding = tsne.FIT_TRANSFORM(distance_matrix)

        PLOT_SCATTER(embedding[:,0], embedding[:,1],
                     color=labels, labels=tickers)
        SAVE_PLOT("cluster_visualization.png")
```

### Module 7: Cointegration Testing (Optional)

```
CLASS CointegrationAnalyzer:

    FUNCTION test_pairs_in_cluster(price_df, cluster_tickers):
        # Run Engle-Granger cointegration test on pairs within cluster
        # This identifies pairs suitable for statistical arbitrage

        cointegrated_pairs = []

        FOR i FROM 0 TO LEN(cluster_tickers)-1:
            FOR j FROM i+1 TO LEN(cluster_tickers)-1:
                ticker_a = cluster_tickers[i]
                ticker_b = cluster_tickers[j]

                price_a = price_df[ticker_a]
                price_b = price_df[ticker_b]

                # Engle-Granger test
                p_value = STATSMODELS.COINT(price_a, price_b)[1]

                IF p_value < 0.05:  # Significant at 5%
                    cointegrated_pairs.APPEND({
                        "ticker_a": ticker_a,
                        "ticker_b": ticker_b,
                        "p_value": p_value
                    })

        RETURN cointegrated_pairs
```

### Module 8: Output/Export

```
CLASS OutputManager:

    FUNCTION export_to_json(labels, tickers, output_path):
        # Group tickers by cluster

        clusters = {}
        FOR i, ticker IN ENUMERATE(tickers):
            cluster_id = labels[i]
            IF cluster_id NOT IN clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].APPEND(ticker)

        # Format as list of cluster objects
        output = []
        FOR cluster_id, members IN clusters.ITEMS():
            output.APPEND({
                "cluster": cluster_id,
                "members": members,
                "size": LEN(members)
            })

        WRITE_JSON(output, output_path)

    FUNCTION export_to_parquet(labels, tickers, metadata, output_path):
        # Create structured DataFrame for database import

        records = []
        analysis_date = TODAY()

        FOR i, ticker IN ENUMERATE(tickers):
            records.APPEND({
                "analysis_date": analysis_date,
                "ticker": ticker,
                "cluster_id": labels[i],
                "listing_status": metadata[ticker].get("status", "Active"),
                "sector": metadata[ticker].get("sector", None)
            })

        df = DataFrame(records)
        df.TO_PARQUET(output_path, compression="snappy")

    FUNCTION export_correlations_sparse(corr_matrix, tickers, threshold, output_path):
        # Only store pairs with correlation above threshold
        # Avoids storing millions of weak correlations

        records = []
        N = LEN(tickers)

        FOR i FROM 0 TO N-1:
            FOR j FROM i+1 TO N-1:
                corr = corr_matrix[i, j]
                IF ABS(corr) >= threshold:
                    records.APPEND({
                        "ticker_a": tickers[i],
                        "ticker_b": tickers[j],
                        "correlation": corr
                    })

        df = DataFrame(records)
        df.TO_PARQUET(output_path, compression="snappy")
```

### Main Pipeline Orchestrator

```
CLASS StockClusteringPipeline:

    FUNCTION run(config):
        # Configuration
        start_date = config.start_date
        end_date = config.end_date
        clustering_method = config.clustering_method  # "dbscan" or "hierarchical"
        output_format = config.output_format          # "json" or "parquet"

        LOG("Step 1: Fetching universe...")
        universe_mgr = UniverseManager()
        tickers = universe_mgr.get_universe(start_date, end_date)

        LOG("Step 2: Ingesting price data...")
        ingestion = DataIngestion()
        price_df = ingestion.fetch_price_data(tickers, start_date, end_date)

        LOG("Step 3: Preprocessing...")
        preprocessor = Preprocessor()
        returns_df = preprocessor.clean_and_transform(price_df)
        valid_tickers = returns_df.columns.TO_LIST()

        # Optional: Remove market factor
        IF config.remove_market_factor:
            returns_matrix = returns_df.VALUES.TRANSPOSE()
            returns_matrix = preprocessor.apply_pca_demarket(returns_matrix)

        LOG("Step 4: Computing correlation matrix...")
        corr_engine = CorrelationEngine()
        returns_matrix = returns_df.VALUES.TRANSPOSE()
        corr_matrix = corr_engine.compute_correlation_matrix(returns_df)

        LOG("Step 5: Computing distance matrix...")
        dist_matrix = corr_engine.compute_distance_matrix(corr_matrix)
        condensed_dist = corr_engine.compute_condensed_distance(returns_matrix)

        LOG("Step 6: Clustering...")
        cluster_engine = ClusteringEngine()

        IF clustering_method == "dbscan":
            optimal_eps = cluster_engine.find_optimal_eps(dist_matrix)
            labels = cluster_engine.cluster_dbscan(dist_matrix,
                                                    eps=optimal_eps,
                                                    min_samples=5)
        ELSE IF clustering_method == "hierarchical":
            Z = cluster_engine.cluster_hierarchical(condensed_dist, method="average")
            optimal_k, _ = cluster_engine.find_optimal_clusters(Z)
            labels = cluster_engine.cut_dendrogram(Z, n_clusters=optimal_k)

        LOG("Step 7: Validation...")
        validator = ClusterValidator()
        silhouette = validator.compute_silhouette(dist_matrix, labels)
        LOG(f"Silhouette Score: {silhouette}")

        IF config.visualize:
            validator.visualize_tsne(dist_matrix, labels, valid_tickers)

        LOG("Step 8: Exporting results...")
        output_mgr = OutputManager()

        IF output_format == "json":
            output_mgr.export_to_json(labels, valid_tickers, "stock_clusters.json")
        ELSE:
            output_mgr.export_to_parquet(labels, valid_tickers, {},
                                          "equity_clusters.parquet")
            output_mgr.export_correlations_sparse(corr_matrix, valid_tickers,
                                                   threshold=0.7,
                                                   "pair_correlations.parquet")

        # Optional: Cointegration analysis
        IF config.run_cointegration:
            LOG("Step 9: Cointegration testing...")
            coint_analyzer = CointegrationAnalyzer()
            FOR cluster_id IN UNIQUE(labels):
                IF cluster_id == -1:  # Skip noise
                    CONTINUE
                cluster_tickers = valid_tickers WHERE labels == cluster_id
                pairs = coint_analyzer.test_pairs_in_cluster(price_df, cluster_tickers)
                # Store results...

        LOG("Pipeline complete.")
        RETURN {
            "n_stocks": LEN(valid_tickers),
            "n_clusters": LEN(UNIQUE(labels)),
            "silhouette_score": silhouette,
            "labels": labels,
            "tickers": valid_tickers
        }
```

---

## Configuration Schema

```
config = {
    "start_date": "2024-07-01",
    "end_date": "2025-12-31",
    "data_source": "yfinance",          # or "alpha_vantage", "databento"
    "api_key": "YOUR_API_KEY",

    "min_history_pct": 0.90,            # Require 90% of trading days
    "remove_market_factor": True,       # Apply PCA to remove beta

    "clustering_method": "dbscan",      # or "hierarchical"
    "linkage_method": "average",        # for hierarchical
    "n_clusters": None,                 # Auto-detect if None

    "output_format": "parquet",         # or "json"
    "correlation_threshold": 0.7,       # For sparse export

    "visualize": True,
    "run_cointegration": False
}
```

---

## Key Implementation Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary clustering | DBSCAN | Handles noise (unclustered stocks) naturally |
| Fallback clustering | Hierarchical (average linkage) | Proven in academic literature, no preset k needed |
| Distance metric | √(2(1-ρ)) | Proper metric space, standard in finance literature |
| Data transform | Log returns + Z-score | Stationarity + scale normalization |
| Missing data | Forward-fill | No look-ahead bias |
| Output format | Parquet (primary), JSON (alt) | Columnar, compressed, schema-enforced |
| Parallelization | joblib/multiprocessing | GIL bypass for CPU-bound distance calc |

---

## Dependencies

```
numpy
pandas
scipy
scikit-learn
tslearn          # Optional: for DTW-based clustering
statsmodels      # Optional: for cointegration tests
pyarrow          # For Parquet export
yfinance         # For data ingestion (or alpha_vantage, databento)
matplotlib       # For visualization
```

---

## Output Schema

### equity_clusters.parquet

| Column | Type | Description |
|--------|------|-------------|
| analysis_date | DATE | Date of analysis |
| ticker | STRING | Stock symbol |
| cluster_id | INT | Cluster assignment (-1 = noise) |
| listing_status | STRING | Active/Delisted |
| sector | STRING | GICS sector (optional) |

### pair_correlations.parquet (sparse)

| Column | Type | Description |
|--------|------|-------------|
| ticker_a | STRING | First stock |
| ticker_b | STRING | Second stock |
| correlation | FLOAT | Pearson correlation |

### stock_clusters.json

```json
[
  {"cluster": 0, "members": ["AAPL", "MSFT", "GOOGL", ...], "size": 45},
  {"cluster": 1, "members": ["JPM", "BAC", "WFC", ...], "size": 32},
  {"cluster": -1, "members": ["ODDBALL1", ...], "size": 128}
]
```
