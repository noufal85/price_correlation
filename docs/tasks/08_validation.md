# Task 08: Validation

**Phase**: Clustering
**Status**: ⬜ Not Started
**Depends on**: Task 07

---

## Objective

Evaluate clustering quality and generate visualizations.

## Module

`src/price_correlation/validation.py`

## Functions

### compute_silhouette(distance_matrix, labels) → float
```
Input:
  distance_matrix - precomputed distances
  labels          - cluster assignments

Output:
  Score in range [-1, 1]

Interpretation:
  +1.0  Perfect separation
  +0.5  Good clustering
   0.0  Overlapping clusters
  -0.5  Wrong assignments
  -1.0  Completely wrong

Flow:
  [Call sklearn.metrics.silhouette_score]
           ↓
  [Use metric="precomputed"]
           ↓
  [Return score]
```

### compute_cluster_stats(labels, tickers) → dict
```
Input:
  labels  - cluster assignments
  tickers - stock symbols

Output:
  {
    "n_clusters": int,
    "n_noise": int,
    "cluster_sizes": {id: count},
    "largest_cluster": int,
    "smallest_cluster": int
  }

Flow:
  [Count unique labels]
           ↓
  [Count -1 (noise) separately]
           ↓
  [Compute size per cluster]
           ↓
  [Return stats dict]
```

### get_cluster_members(labels, tickers) → dict
```
Output:
  {
    0: ["AAPL", "MSFT", ...],
    1: ["JPM", "BAC", ...],
    -1: ["ODDBALL1", ...]
  }
```

### generate_tsne_plot(distance_matrix, labels, tickers, output_path)
```
2D visualization using t-SNE

Flow:
  [Fit t-SNE on distance matrix]
           ↓
  [Get 2D coordinates]
           ↓
  [Color points by cluster label]
           ↓
  [Save plot to file]

Settings:
  n_components = 2
  metric = "precomputed"
  perplexity = 30
```

## Visualization Example

```
    Cluster 0 (Tech)         Cluster 1 (Banks)
         ●●●                      ▲▲▲
        ●●●●●                    ▲▲▲▲
         ●●●                      ▲▲

              ■■■■
             ■■■■■■   Cluster 2 (Energy)
              ■■■■

    ○  ○    ○   ○   Noise points (unclustered)
```

## Quality Interpretation

```
┌─────────────────────────────────────────────────┐
│              Clustering Quality                 │
├──────────────────┬──────────────────────────────┤
│ Silhouette > 0.5 │ Strong structure found       │
│ Silhouette 0.2-0.5│ Reasonable clusters         │
│ Silhouette < 0.2 │ Weak/overlapping structure  │
├──────────────────┼──────────────────────────────┤
│ Noise < 10%      │ Most stocks clustered        │
│ Noise 10-30%     │ Some outliers (expected)     │
│ Noise > 30%      │ eps might be too small       │
├──────────────────┼──────────────────────────────┤
│ 5-20 clusters    │ Reasonable for US market     │
│ < 5 clusters     │ Too coarse                   │
│ > 50 clusters    │ Likely overfitting           │
└──────────────────┴──────────────────────────────┘
```

## Optional: Sector Validation

```
validate_against_sectors(labels, tickers, sector_map) → dict

Compare clusters to GICS sectors:
  - Compute "purity" per cluster
  - Purity = % of dominant sector

Example:
  Cluster 0: 80% Technology → purity = 0.80
  Cluster 1: 90% Financials → purity = 0.90

High purity = algorithm learned real structure
```

## Acceptance Criteria

- [ ] Silhouette score computed correctly
- [ ] Cluster stats returned as dict
- [ ] t-SNE plot generated and saved
- [ ] Colors distinguish clusters visually
- [ ] Handles noise points (-1) in visualization

## Output Files

- `output/cluster_stats.json` - quality metrics
- `output/cluster_visualization.png` - t-SNE plot
