# Task 12: Clustering Optimization for Large Stock Populations

## Problem Statement

When clustering large populations (1000+ stocks), current clustering approaches fail:

**Observed Issues:**
- **DBSCAN**: With 3800+ stocks, produces only 2 clusters (1 main cluster + 1 noise point)
- **Hierarchical**: Similar issue - all stocks collapse into 1-2 clusters
- **Root cause**: The distance/correlation space becomes too dense at scale

---

## Part 1: Technical Deep-Dive - How Correlation is Computed

### Step 1: Price Data Ingestion

```
INPUT:  Raw adjusted close prices for N stocks over T trading days
OUTPUT: DataFrame of shape (T, N) - rows=dates, columns=tickers

Example:
           AAPL    MSFT    GOOGL   ...
2024-01-02  185.5   375.2   140.3
2024-01-03  186.2   376.8   141.1
2024-01-04  184.9   374.5   139.8
...
```

### Step 2: Data Cleaning (`clean_price_data`)

```
PROCESS:
1. Forward-fill missing values (holidays, halts)
   - If AAPL missing on day T, use day T-1 price

2. Filter sparse tickers
   - Require min_history_pct (default 90%) non-null values
   - Stock with 300/365 days valid = 82% → EXCLUDED
   - Stock with 340/365 days valid = 93% → INCLUDED

OUTPUT: Cleaned DataFrame with only tickers having sufficient history
```

### Step 3: Log Returns Computation (`compute_log_returns`)

```
FORMULA:
    r_t = ln(P_t) - ln(P_{t-1})

    where:
    - r_t = log return at time t
    - P_t = price at time t
    - ln = natural logarithm

WHY LOG RETURNS?
1. Additive over time: r_total = r_1 + r_2 + ... + r_n
2. Symmetric: +10% and -10% have equal magnitude
3. Approximately normal distribution
4. Better statistical properties for correlation

EXAMPLE:
    Day 1: Price = $100
    Day 2: Price = $105

    Simple return: (105-100)/100 = 5%
    Log return: ln(105) - ln(100) = ln(1.05) = 4.88%

OUTPUT: Returns DataFrame of shape (T-1, N)
```

### Step 4: Z-Score Normalization (`zscore_normalize`)

```
FORMULA:
    z_i = (r_i - μ) / σ

    where:
    - z_i = normalized return
    - r_i = raw log return
    - μ = mean of returns for that stock
    - σ = standard deviation of returns

PURPOSE:
1. Centers each stock's returns around 0
2. Scales to unit variance (σ = 1)
3. Makes stocks comparable regardless of volatility
4. Removes bias from high-volatility stocks dominating correlations

FILTER:
- Exclude stocks with σ ≈ 0 (no variance = dead stock)
- Threshold: σ > 1e-10

OUTPUT: Normalized returns, each column has mean≈0, std≈1
```

### Step 5: Correlation Matrix Computation (`compute_correlation_matrix`)

```
FORMULA: Pearson Correlation Coefficient

    ρ(X,Y) = Σ[(x_i - μ_x)(y_i - μ_y)] / [(n-1) * σ_x * σ_y]

    Simplified (after z-score normalization where μ=0, σ=1):

    ρ(X,Y) = (1/n) * Σ(x_i * y_i)

IMPLEMENTATION:
    returns_matrix = returns_df.values.T    # Shape: (N, T)
    corr = np.corrcoef(returns_matrix)      # Shape: (N, N)

PROPERTIES:
- Symmetric: ρ(A,B) = ρ(B,A)
- Diagonal = 1: ρ(A,A) = 1 (perfect self-correlation)
- Range: [-1, +1]
  - +1: Perfect positive correlation (move together)
  -  0: No linear relationship
  - -1: Perfect negative correlation (move opposite)

MATRIX SIZE:
- 500 stocks  →  250,000 pairs  (500 × 500)
- 1000 stocks →  1,000,000 pairs
- 3000 stocks →  9,000,000 pairs
- Memory: 3000×3000×8 bytes = 72 MB for correlation matrix alone
```

### Step 6: Distance Matrix Conversion (`correlation_to_distance`)

```
FORMULA (sqrt method - default):
    d(A,B) = √[2 × (1 - ρ(A,B))]

WHY THIS FORMULA?
- Maps correlation [-1, +1] to distance [0, 2]
- ρ = +1.0 (perfect correlation)  → d = 0 (identical)
- ρ =  0.0 (uncorrelated)         → d = √2 ≈ 1.414
- ρ = -1.0 (anti-correlated)      → d = 2 (maximally different)

ALTERNATIVE (simple method):
    d(A,B) = 1 - ρ(A,B)

    - ρ = +1.0 → d = 0
    - ρ =  0.0 → d = 1
    - ρ = -1.0 → d = 2

THE SQRT METHOD IS PREFERRED because it satisfies the triangle inequality,
making it a proper metric for clustering algorithms.
```

### Step 7: Condensed Distance Array (`get_condensed_distance`)

```
PURPOSE: Memory-efficient format for scipy hierarchical clustering

FULL MATRIX (redundant):
    [0.0  0.5  0.8  1.2]
    [0.5  0.0  0.3  0.9]
    [0.8  0.3  0.0  0.6]
    [1.2  0.9  0.6  0.0]

CONDENSED (upper triangle only):
    [0.5, 0.8, 1.2, 0.3, 0.9, 0.6]

    Order: d(0,1), d(0,2), d(0,3), d(1,2), d(1,3), d(2,3)

SIZE:
    Full matrix:    N × N values
    Condensed:      N × (N-1) / 2 values

    For N=3000: 9M → 4.5M values (50% reduction)

IMPLEMENTATION:
    Uses scipy.spatial.distance.pdist with metric="correlation"
    Directly computes d = 1 - ρ for efficiency
```

---

## Part 2: Current Clustering Problem Analysis

### Why DBSCAN Fails at Scale

```
DBSCAN PARAMETERS:
- eps: Maximum distance between two points to be neighbors
- min_samples: Minimum points to form a dense region (cluster)

CURRENT AUTO-TUNING (find_optimal_eps):
1. For each point, find distance to k-th nearest neighbor
2. Sort these distances descending
3. Find "elbow" (maximum rate of change)
4. Use elbow distance as eps

THE PROBLEM:
- With 3000+ stocks, the distance distribution becomes nearly uniform
- Most stocks have correlation 0.3-0.7 → distances 0.8-1.2
- The "elbow" occurs at a very small eps (e.g., 0.3)
- At eps=0.3, only perfectly correlated stocks cluster
- Result: 1 giant cluster of "similar enough" + noise

DISTANCE DISTRIBUTION (typical 3000 stocks):
    Distance 0.0-0.5:  ~2% of pairs (highly correlated)
    Distance 0.5-1.0:  ~25% of pairs
    Distance 1.0-1.5:  ~50% of pairs (weakly correlated)
    Distance 1.5-2.0:  ~23% of pairs (anti-correlated)
```

### Why Hierarchical Clustering Fails at Scale

```
SILHOUETTE SCORE PROBLEM:
- Silhouette measures cluster separation vs cohesion
- At scale, all silhouette scores converge to similar values
- Algorithm picks k=2 or k=3 because it's "technically" optimal
- But this is meaningless for analysis

EXAMPLE SCORES (3000 stocks):
    k=2:  silhouette = 0.15
    k=5:  silhouette = 0.14
    k=10: silhouette = 0.13
    k=50: silhouette = 0.11

All scores are within 0.04 - algorithm picks k=2 despite being useless
```

---

## Part 3: Proposed Solution - Multi-Method Clustering

### New Architecture

After correlation step completes, run MULTIPLE clustering methods in parallel and let user compare results.

```
                    ┌─────────────────────────────────────┐
                    │     CORRELATION MATRIX (N×N)        │
                    │     + DISTANCE MATRIX               │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  HIERARCHICAL   │    │    HDBSCAN      │    │   K-MEANS++     │
│  (forced k)     │    │  (auto density) │    │  (fixed k)      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ k = sqrt(N)     │    │ min_cluster=10  │    │ k = N/50        │
│ linkage=ward    │    │ min_samples=5   │    │ on PCA-reduced  │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Clusters A      │    │ Clusters B      │    │ Clusters C      │
│ + Silhouette    │    │ + Silhouette    │    │ + Silhouette    │
│ + Distribution  │    │ + Distribution  │    │ + Distribution  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────────────────┐
                    │   COMPARISON VIEW IN UI             │
                    │   - Side-by-side cluster stats      │
                    │   - User selects preferred method   │
                    │   - Can save multiple views         │
                    └─────────────────────────────────────┘
```

### Method 1: Forced-K Hierarchical

```
APPROACH:
- Don't auto-detect k, force a reasonable value
- k = max(20, sqrt(N))  → 3000 stocks = ~55 clusters

PARAMETERS:
- linkage: "ward" (minimizes variance within clusters)
- k: sqrt(N) or user-specified

PROS:
- Guaranteed cluster count
- Hierarchical structure preserved
- Interpretable dendrograms

CONS:
- Arbitrary k choice
- May split natural clusters
```

### Method 2: HDBSCAN (Hierarchical DBSCAN)

```
APPROACH:
- Variable density clustering
- Automatically finds clusters of different densities
- Better noise handling than DBSCAN

PARAMETERS:
- min_cluster_size: 10 (minimum stocks per cluster)
- min_samples: 5 (core point definition)
- cluster_selection_method: "eom" (excess of mass)

FORMULA:
    Instead of fixed eps, HDBSCAN builds a hierarchy:
    1. Build minimum spanning tree of mutual reachability distances
    2. Extract clusters at different density levels
    3. Select most stable clusters across levels

PROS:
- No eps parameter to tune
- Handles varying cluster densities
- Natural noise detection

CONS:
- More computational cost
- May still produce few clusters if data is uniform
```

### Method 3: K-Means++ on PCA-Reduced Space

```
APPROACH:
- Reduce dimensionality first with PCA
- Run K-Means in reduced space
- Much faster and often better separated

STEPS:
1. Take normalized returns matrix (N stocks × T days)
2. Apply PCA to reduce to 50 dimensions
3. Run K-Means++ with k = N/50 clusters
4. Map back to original tickers

PARAMETERS:
- n_components: 50 (PCA dimensions)
- k: N/50 (e.g., 60 clusters for 3000 stocks)
- init: "k-means++" (smart initialization)
- n_init: 10 (run 10 times, pick best)

PROS:
- Very fast (O(n) instead of O(n²))
- PCA removes noise, improves separation
- Guaranteed k clusters

CONS:
- Loses some information in PCA
- K-Means assumes spherical clusters
```

### Method 4: Spectral Clustering

```
APPROACH:
- Use eigenvalues of similarity matrix
- Better for non-convex clusters

STEPS:
1. Build similarity matrix: S = exp(-d²/2σ²)
2. Compute graph Laplacian: L = D - S
3. Find k smallest eigenvectors
4. Cluster eigenvector coordinates with K-Means

PARAMETERS:
- n_clusters: sqrt(N)
- affinity: "precomputed" (use our correlation-based similarity)
- assign_labels: "kmeans"

PROS:
- Can find non-spherical clusters
- Uses global structure

CONS:
- O(n³) eigenvalue computation
- Memory intensive for large N
```

### Method 5: Louvain Community Detection

```
APPROACH:
- Treat correlation matrix as weighted graph
- Apply network community detection

STEPS:
1. Build graph: nodes=stocks, edges=correlations above threshold
2. Optimize modularity function
3. Communities = clusters

PARAMETERS:
- threshold: 0.5 (only connect stocks with ρ > 0.5)
- resolution: 1.0 (controls cluster granularity)

PROS:
- No k parameter needed
- Handles different cluster sizes naturally
- Very fast (O(n log n))

CONS:
- Threshold selection matters
- May create many small clusters
```

---

## Part 4: Implementation Plan

### Phase 1: Add HDBSCAN Support

```python
# clustering.py - add new method

def cluster_hdbscan(
    distance_matrix: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> np.ndarray:
    """
    HDBSCAN clustering with precomputed distances.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(distance_matrix)
```

### Phase 2: Add K-Means on PCA

```python
# clustering.py - add new method

def cluster_kmeans_pca(
    returns_df: pd.DataFrame,
    n_clusters: int | None = None,
    n_components: int = 50,
) -> np.ndarray:
    """
    K-Means clustering on PCA-reduced return space.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Reduce dimensions
    pca = PCA(n_components=min(n_components, returns_df.shape[0] - 1))
    reduced = pca.fit_transform(returns_df.values.T)

    # Auto-select k if not provided
    if n_clusters is None:
        n_clusters = max(20, len(returns_df.columns) // 50)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10)
    return kmeans.fit_predict(reduced)
```

### Phase 3: Add Louvain Community Detection

```python
# clustering.py - add new method

def cluster_louvain(
    corr_matrix: np.ndarray,
    threshold: float = 0.5,
    resolution: float = 1.0,
) -> np.ndarray:
    """
    Louvain community detection on correlation graph.
    """
    import networkx as nx
    from community import community_louvain

    n = corr_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges for correlations above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=corr_matrix[i, j])

    # Detect communities
    partition = community_louvain.best_partition(G, resolution=resolution)
    return np.array([partition[i] for i in range(n)])
```

### Phase 4: Multi-Clustering Pipeline Step

```python
# pipeline_steps.py - modify run_step_clustering

def run_step_clustering(config, state_manager, progress_callback=None):
    """Run multiple clustering methods and store all results."""

    methods_to_run = config.clustering_methods  # ["hierarchical", "hdbscan", "kmeans_pca", "louvain"]

    results = {}

    for method in methods_to_run:
        log(f"Running {method} clustering...")

        if method == "hierarchical":
            labels = cluster_hierarchical_forced_k(...)
        elif method == "hdbscan":
            labels = cluster_hdbscan(...)
        elif method == "kmeans_pca":
            labels = cluster_kmeans_pca(...)
        elif method == "louvain":
            labels = cluster_louvain(...)

        # Compute stats for this method
        silhouette = compute_silhouette(dist_matrix, labels)
        stats = compute_cluster_stats(labels, tickers)

        results[method] = {
            "labels": labels,
            "silhouette": silhouette,
            "n_clusters": stats["n_clusters"],
            "n_noise": stats["n_noise"],
            "cluster_sizes": stats["cluster_sizes"],
        }

    return results
```

### Phase 5: UI Updates

```
STEPS PAGE:
- Add multi-select for clustering methods
- Show progress for each method

CLUSTERS PAGE:
- Add dropdown to switch between clustering results
- Side-by-side comparison view
- Metrics table: silhouette, n_clusters, size distribution

COMPARISON VIEW:
┌─────────────────────────────────────────────────────────────┐
│ Clustering Comparison                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Method        │ Clusters │ Noise │ Silhouette │ Max Size  │
│  ─────────────────────────────────────────────────────────  │
│  Hierarchical  │    55    │   0   │   0.12     │   245     │
│  HDBSCAN       │    42    │  89   │   0.18     │   312     │
│  K-Means PCA   │    60    │   0   │   0.15     │   198     │
│  Louvain       │    78    │   0   │   0.09     │   156     │
│                                                             │
│  [View Hierarchical] [View HDBSCAN] [View K-Means] [View L] │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 5: Dependencies to Add

```toml
# pyproject.toml additions

dependencies = [
    # existing...
    "hdbscan>=0.8.33",           # HDBSCAN clustering
    "python-louvain>=0.16",      # Louvain community detection
    "networkx>=3.0",             # Graph operations for Louvain
]
```

---

## Part 6: Configuration Options

```python
@dataclass
class ClusteringConfig:
    # Methods to run (can select multiple)
    methods: list[str] = field(default_factory=lambda: ["hierarchical", "hdbscan"])

    # Hierarchical settings
    hierarchical_k: int | None = None  # None = auto (sqrt(N))
    hierarchical_linkage: str = "ward"

    # HDBSCAN settings
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5

    # K-Means PCA settings
    kmeans_n_clusters: int | None = None  # None = auto (N/50)
    kmeans_pca_components: int = 50

    # Louvain settings
    louvain_threshold: float = 0.5
    louvain_resolution: float = 1.0
```

---

## Part 7: Next Steps

1. **Immediate**: Add HDBSCAN to existing clustering.py
2. **Short-term**: Add K-Means PCA method
3. **Medium-term**: Add Louvain community detection
4. **UI**: Add method comparison view
5. **Testing**: Benchmark all methods on 500, 1000, 3000 stocks

---

## Appendix: Mathematical Formulas Reference

### Pearson Correlation
```
ρ(X,Y) = cov(X,Y) / (σ_X × σ_Y)

       = Σ[(x_i - μ_X)(y_i - μ_Y)] / √[Σ(x_i - μ_X)² × Σ(y_i - μ_Y)²]
```

### Silhouette Score
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
- a(i) = average distance to points in same cluster
- b(i) = minimum average distance to points in other clusters
```

### Modularity (Louvain)
```
Q = (1/2m) × Σ[A_ij - (k_i × k_j)/(2m)] × δ(c_i, c_j)

where:
- A_ij = edge weight between i and j
- k_i = sum of weights of edges attached to i
- m = total edge weight
- c_i = community of node i
- δ = 1 if same community, 0 otherwise
```
