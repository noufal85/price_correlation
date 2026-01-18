# Task 05: Preprocessing

**Phase**: Data Pipeline
**Status**: ⬜ Not Started
**Depends on**: Task 04

---

## Objective

Transform raw price data into normalized log returns suitable for clustering.

## Module

`src/price_correlation/preprocess.py`

## Functions

### clean_price_data(price_df, min_history_pct=0.9) → DataFrame
```
Input:
  price_df        - DataFrame of raw prices
  min_history_pct - minimum required data (0.9 = 90%)

Flow:
  [Forward-fill missing values]
           ↓
  [Count non-null per ticker]
           ↓
  [Drop tickers below threshold]
           ↓
  [Return cleaned DataFrame]

Example:
  If 400 trading days, require 360+ non-null values
  Tickers with more gaps are dropped
```

### compute_log_returns(prices) → DataFrame
```
Input:  DataFrame of prices
Output: DataFrame of log returns

Formula:
  r_t = ln(P_t) - ln(P_{t-1})

Flow:
  [Take natural log of prices]
           ↓
  [Compute difference (shift by 1)]
           ↓
  [Drop first row (NaN)]
           ↓
  [Return returns DataFrame]
```

### zscore_normalize(returns) → DataFrame
```
Input:  DataFrame of returns
Output: DataFrame of z-scored returns

Formula per column:
  z = (x - mean) / std

Result:
  Each column has mean ≈ 0, std ≈ 1
```

### remove_market_factor(returns, n_components=1) → DataFrame
```
Optional: Remove systematic market movement

Flow:
  [Fit PCA with n_components]
           ↓
  [Transform data to PC space]
           ↓
  [Inverse transform (reconstruct)]
           ↓
  [Subtract reconstruction from original]
           ↓
  [Return residuals]

Effect:
  Removes "beta" / market mode
  Isolates idiosyncratic correlations
```

## Complete Preprocessing Pipeline

```
┌─────────────────┐
│   Raw Prices    │
│  (N tickers)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Forward Fill   │
│  (handle gaps)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Filter Tickers  │
│ (>90% history)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Log Returns    │
│ ln(Pt/Pt-1)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Z-Score       │
│ (mean=0,std=1)  │
└────────┬────────┘
         │
         ▼ (optional)
┌─────────────────┐
│ Remove Market   │
│ Factor (PCA)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ready for       │
│ Clustering      │
└─────────────────┘
```

## Acceptance Criteria

- [ ] Forward-fill handles missing data correctly
- [ ] Filters out tickers with insufficient history
- [ ] Log returns computed correctly
- [ ] Z-score output: mean ≈ 0, std ≈ 1 per column
- [ ] PCA removal is optional and configurable

## Validation

```
After z-score:
  assert abs(df.mean().mean()) < 0.01
  assert abs(df.std().mean() - 1.0) < 0.01
```
