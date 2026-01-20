# Task 12: Clustering Optimization for Large Stock Populations

## Problem Statement

When clustering large populations (1000+ stocks), the current hierarchical clustering approach becomes:
1. **Computationally expensive** - O(n^2) or O(n^3) complexity for correlation matrix and clustering
2. **Less effective** - All stocks may end up in very few clusters despite having distinct behavior patterns
3. **Memory intensive** - Correlation matrices grow quadratically (3000 stocks = 9M pairs)

## Current Approach

```
1. Fetch 18-month price history for all stocks
2. Compute log returns
3. Z-score normalize returns
4. Build pairwise correlation matrix (n x n)
5. Convert to distance matrix (1 - correlation)
6. Run hierarchical clustering with optimal k search
7. Select k via silhouette score optimization
```

### Current Issues

1. **Silhouette score optimization fails at scale**
   - With 3000+ stocks, optimal k determination becomes unreliable
   - Silhouette scores become flat or pick extremes (k=2 or k=n)

2. **Correlation matrix includes noise**
   - Many stock pairs have near-zero correlation (noise)
   - Low correlations don't indicate similar behavior

3. **Single clustering pass over entire universe**
   - Treats mega-caps same as small-caps
   - Doesn't account for sector/industry structure

---

## Optimization Strategies to Consider

### Strategy 1: Pre-Filtering by Sector/Industry

**Concept**: Cluster within sectors first, then find cross-sector relationships

```
STEPS:
1. Group stocks by GICS sector (11 sectors)
2. Run clustering within each sector
3. Find cross-sector correlations at cluster-centroid level
4. Merge similar clusters across sectors
```

**Pros**:
- Reduces problem size (300 stocks/sector vs 3000 total)
- Sector-aware clusters make more intuitive sense
- Faster computation

**Cons**:
- May miss cross-sector correlations early
- Requires sector data

---

### Strategy 2: Hierarchical Sampling + Expansion

**Concept**: Cluster a representative sample, then assign remaining stocks

```
STEPS:
1. Sample 500 stocks (stratified by market cap / sector)
2. Cluster the sample → get k cluster centers
3. For remaining stocks, assign to nearest cluster center
4. Refine with k-means using assigned clusters
```

**Pros**:
- Much faster (cluster 500 instead of 3000)
- Can handle very large universes (10k+)

**Cons**:
- Sample may miss unique behavior patterns
- Two-phase approach adds complexity

---

### Strategy 3: Graph-Based Community Detection

**Concept**: Build correlation graph, use network algorithms

```
STEPS:
1. Build correlation matrix
2. Create graph: node = stock, edge = correlation > threshold
3. Use Louvain/Leiden algorithm for community detection
4. Communities = clusters
```

**Pros**:
- Naturally handles different cluster sizes
- No need to specify k
- Scales well with sparse graphs

**Cons**:
- Threshold selection matters
- May create many small clusters

---

### Strategy 4: DBSCAN with Better Distance Metric

**Concept**: Use DBSCAN (already in code) with optimized parameters

```
IMPROVEMENTS:
1. Use Dynamic Time Warping (DTW) distance instead of correlation
2. Auto-tune epsilon based on distribution of distances
3. Use HDBSCAN (hierarchical DBSCAN) for variable density
```

**Pros**:
- Handles noise (uncorrelated stocks) naturally
- No need to specify k
- DTW captures lagged correlations

**Cons**:
- DTW is slow for large datasets
- Parameter tuning still needed

---

### Strategy 5: Multi-Level Hierarchical Approach

**Concept**: Build hierarchy of clusters

```
STEPS:
1. First pass: Create 50-100 macro-clusters
2. Second pass: Split large clusters (>100 stocks) into sub-clusters
3. Build hierarchy: Sector → Macro-cluster → Sub-cluster
4. Allow stocks to be viewed at any level
```

**Pros**:
- Natural zoom in/out capability
- Handles scale well
- User can explore at different granularities

**Cons**:
- More complex implementation
- Requires UI changes for hierarchy display

---

### Strategy 6: Dimensionality Reduction Before Clustering

**Concept**: Reduce feature space before clustering

```
STEPS:
1. Compute return series for all stocks (n x 365 matrix)
2. Apply PCA/UMAP to reduce to k dimensions (e.g., k=50)
3. Cluster in reduced space
4. Much faster distance computations
```

**Pros**:
- Handles very large datasets
- Removes noise in return patterns
- Faster clustering

**Cons**:
- May lose important information
- PCA components may not be interpretable

---

### Strategy 7: Rolling Window Clusters

**Concept**: Cluster based on recent behavior, not full 18 months

```
IMPROVEMENTS:
1. Weight recent returns higher (exponential decay)
2. Use 3-month rolling window for "current" clusters
3. Track cluster stability over time
4. Flag when stock changes clusters
```

**Pros**:
- More responsive to regime changes
- Captures current relationships
- Useful for trading signals

**Cons**:
- Less stable clusters
- May miss longer-term patterns

---

## Performance Benchmarks Needed

| Dataset Size | Current Time | Target Time |
|-------------|--------------|-------------|
| 500 stocks  | ~30s         | ~10s        |
| 1000 stocks | ~2min        | ~30s        |
| 3000 stocks | ~15min+      | ~2min       |
| 5000 stocks | OOM/timeout  | ~5min       |

---

## Recommended Evaluation Criteria

1. **Cluster Quality**
   - Silhouette score (intra vs inter cluster distance)
   - Within-cluster correlation average
   - Cluster size distribution (not too skewed)

2. **Interpretability**
   - Clusters should have sector/theme coherence
   - User should understand why stocks grouped together

3. **Stability**
   - Clusters shouldn't change dramatically day-to-day
   - Core members should remain consistent

4. **Performance**
   - Linear or near-linear scaling with stock count
   - Memory usage within reasonable bounds

---

## Questions for Brainstorming

1. What is the target number of clusters for 3000+ stocks?
   - Fixed (e.g., 50)?
   - Dynamic based on data?
   - User configurable?

2. Should clusters have hierarchy (sectors → sub-clusters)?

3. Is cross-sector clustering important?
   - Tech + Healthcare stocks moving together

4. Time horizon for correlation:
   - 18 months? 6 months? 3 months?
   - Rolling vs fixed window?

5. Should we support multiple clustering views?
   - By correlation
   - By sector
   - By market cap tier

---

## Next Steps

- [ ] Benchmark current implementation with 500, 1000, 3000 stocks
- [ ] Implement Strategy 1 (sector pre-grouping) as experiment
- [ ] Try HDBSCAN as alternative to hierarchical
- [ ] Add cluster quality metrics to output
- [ ] User selects preferred approach after testing
