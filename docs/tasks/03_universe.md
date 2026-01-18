# Task 03: Universe Manager

**Phase**: Data Pipeline
**Status**: ⬜ Not Started
**Depends on**: Task 02

---

## Objective

Create module to fetch complete list of NYSE/NASDAQ stock tickers.

## Module

`src/price_correlation/universe.py`

## Functions

### get_exchange_tickers(exchange) → list[str]
```
Input:  exchange name ("NYSE" or "NASDAQ")
Output: list of ticker symbols

Flow:
  [Call yfinance or similar API]
       ↓
  [Parse response]
       ↓
  [Filter: common stocks only, exclude ETFs/ADRs]
       ↓
  [Return sorted list]
```

### get_full_universe() → list[str]
```
Flow:
  [get_exchange_tickers("NYSE")]
       ↓
  [get_exchange_tickers("NASDAQ")]
       ↓
  [Combine and deduplicate]
       ↓
  [Return sorted list]
```

## Data Sources

Option 1: yfinance
- Use screen or ticker lookup functions

Option 2: External API
- Alpha Vantage LISTING_STATUS endpoint
- Returns active + delisted tickers

Option 3: Static file
- Download CSV from exchange website
- Parse and return tickers

## Acceptance Criteria

- [ ] Returns 5000+ unique tickers
- [ ] No duplicates in output
- [ ] Only common stocks (filter out ETFs, preferred shares)
- [ ] Works with real API (no mocks)

## Edge Cases

- API rate limits → implement retry with backoff
- Network errors → raise with clear message
- Empty response → raise error
