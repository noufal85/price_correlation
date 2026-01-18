# Task 06: Correlation Engine

**Phase**: Clustering
**Status**: ⬜ Not Started
**Depends on**: Task 05

---

## Objective

Compute pairwise correlation matrix and convert to distance matrix for clustering.

## Module

`src/price_correlation/correlation.py`

## Functions

### compute_correlation_matrix(returns_df) → ndarray
```
Input:  DataFrame (rows=dates, columns=tickers)
Output: ndarray shape (N, N)

Properties:
  - Symmetric: corr[i,j] = corr[j,i]
  - Diagonal = 1: corr[i,i] = 1
  - Range: [-1, 1]

Flow:
  [Transpose to (N_tickers, T_days)]
           ↓
  [numpy.corrcoef()]
           ↓
  [Return NxN matrix]
```

### correlation_to_distance(corr_matrix, method="sqrt") → ndarray
```
Input:  correlation matrix
Output: distance matrix

Methods:
  "sqrt"   → d = sqrt(2 * (1 - rho))    [proper metric]
  "simple" → d = 1 - rho                [faster, common]

Distance properties:
  - Range: [0, 2]
  - d(i,i) = 0
  - Higher correlation = smaller distance
```

### get_condensed_distance(returns_matrix) → ndarray
```
Input:  returns matrix (N, T)
Output: condensed distance array, length N*(N-1)/2

Uses scipy.spatial.distance.pdist with metric='correlation'

Memory advantage:
  Full matrix: N² elements
  Condensed:   N*(N-1)/2 elements

  For N=5000: 25M vs 12.5M
```

## Distance Matrix Visualization

```
         Stock1  Stock2  Stock3  Stock4
Stock1    0.00    0.15    0.82    1.34
Stock2    0.15    0.00    0.71    1.28
Stock3    0.82    0.71    0.00    0.95
Stock4    1.34    1.28    0.95    0.00

Interpretation:
  Stock1-Stock2: very close (0.15) → high correlation
  Stock1-Stock4: far apart (1.34) → low/negative correlation
```

## Correlation to Distance Mapping

```
Correlation (ρ)    Distance (sqrt)    Distance (simple)
     +1.00              0.00               0.00
     +0.75              0.71               0.25
     +0.50              1.00               0.50
      0.00              1.41               1.00
     -0.50              1.73               1.50
     -1.00              2.00               2.00
```

## Acceptance Criteria

- [ ] Correlation matrix is symmetric
- [ ] Diagonal values = 1 for correlation
- [ ] Diagonal values = 0 for distance
- [ ] Distance values in range [0, 2]
- [ ] Condensed format works with scipy.hierarchy

## Performance Notes

- Use `numpy.corrcoef` (vectorized, fast)
- Avoid Python loops for large N
- Condensed distance saves ~50% memory
