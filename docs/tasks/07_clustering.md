# Task 07: Clustering Engine

**Phase**: Clustering
**Status**: ⬜ Not Started
**Depends on**: Task 06

---

## Objective

Implement DBSCAN and Hierarchical clustering with automatic parameter tuning.

## Module

`src/price_correlation/clustering.py`

## Functions

### cluster_dbscan(distance_matrix, eps, min_samples) → ndarray
```
Input:
  distance_matrix - precomputed NxN distances
  eps             - neighborhood radius
  min_samples     - minimum points to form cluster

Output:
  labels array, length N
  Values: 0, 1, 2, ... for clusters
          -1 for noise points

Flow:
  [Create DBSCAN with metric="precomputed"]
           ↓
  [Fit on distance matrix]
           ↓
  [Return labels]
```

### find_optimal_eps(distance_matrix, k=5) → float
```
Elbow method for DBSCAN epsilon

Flow:
  [For each point, find k-th nearest neighbor distance]
           ↓
  [Sort distances descending]
           ↓
  [Plot curve, find elbow (max curvature)]
           ↓
  [Return eps at elbow]

Visual:
  │
  │\
  │ \
  │  \_____  ← elbow point
  │        \____
  └──────────────
        points
```

### cluster_hierarchical(condensed_dist, method="average") → ndarray
```
Input:
  condensed_dist - from scipy.pdist
  method         - linkage method

Output:
  Linkage matrix Z, shape (N-1, 4)

Linkage methods:
  "single"   - min distance (can chain)
  "complete" - max distance (compact clusters)
  "average"  - mean distance (balanced)
  "ward"     - minimize variance
```

### cut_dendrogram(Z, n_clusters=None, threshold=None) → ndarray
```
Extract flat clusters from hierarchy

Mode 1: By count
  n_clusters=10 → exactly 10 clusters

Mode 2: By threshold
  threshold=0.5 → cut tree at distance 0.5

Output:
  labels array, length N
```

### find_optimal_k(Z, distance_matrix, max_k=30) → tuple[int, float]
```
Find best cluster count using silhouette score

Flow:
  FOR k FROM 2 TO max_k:
    [Cut dendrogram at k clusters]
    [Compute silhouette score]
    [Track best]
           ↓
  [Return (best_k, best_score)]
```

## Algorithm Comparison

```
┌─────────────────┬─────────────────┬─────────────────┐
│     DBSCAN      │  Hierarchical   │    K-Means      │
├─────────────────┼─────────────────┼─────────────────┤
│ Handles noise   │ No noise label  │ No noise label  │
│ (-1 labels)     │                 │                 │
├─────────────────┼─────────────────┼─────────────────┤
│ Auto # clusters │ Choose k or     │ Must specify k  │
│                 │ threshold       │                 │
├─────────────────┼─────────────────┼─────────────────┤
│ Density-based   │ Hierarchy/tree  │ Centroid-based  │
├─────────────────┼─────────────────┼─────────────────┤
│ Works with      │ Works with      │ Needs raw       │
│ precomputed     │ condensed dist  │ features        │
└─────────────────┴─────────────────┴─────────────────┘

Recommendation: DBSCAN primary, Hierarchical secondary
```

## DBSCAN Parameter Selection

```
eps too small:
  → Everything is noise (-1)
  → Very few/no clusters

eps too large:
  → Everything in one cluster
  → No separation

Optimal eps:
  → Natural clusters emerge
  → Noise points are true outliers
```

## Acceptance Criteria

- [ ] DBSCAN returns valid labels including -1 for noise
- [ ] find_optimal_eps finds reasonable epsilon
- [ ] Hierarchical returns valid linkage matrix
- [ ] cut_dendrogram produces k clusters when requested
- [ ] find_optimal_k returns k with max silhouette

## Expected Output

For ~5000 stocks:
- 5-30 meaningful clusters
- 100-500 noise points (outliers)
- Silhouette score > 0.1
