# Task 04: Price Ingestion

**Phase**: Data Pipeline
**Status**: ⬜ Not Started
**Depends on**: Task 03

---

## Objective

Fetch historical adjusted close prices for all tickers in parallel.

## Module

`src/price_correlation/ingestion.py`

## Functions

### fetch_price_history(tickers, start_date, end_date) → DataFrame
```
Input:
  tickers    - list of stock symbols
  start_date - "YYYY-MM-DD"
  end_date   - "YYYY-MM-DD"

Output:
  DataFrame with:
    - index: trading dates
    - columns: ticker symbols
    - values: adjusted close prices

Flow:
  [Split tickers into batches]
           ↓
  [Parallel fetch using ThreadPoolExecutor]
           ↓
  [Combine results into single DataFrame]
           ↓
  [Align all series to common date index]
           ↓
  [Return DataFrame]
```

### fetch_single_ticker(ticker, start, end) → Series
```
Helper function for parallel fetching

Flow:
  [Call yfinance.download(ticker)]
           ↓
  [Extract "Adj Close" column]
           ↓
  [Return as Series with date index]

Error handling:
  - Return empty Series on failure
  - Log warning with ticker name
```

## Parallelization Strategy

```
┌─────────────────────────────────────────┐
│            Ticker List (5000)           │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ↓             ↓             ↓
┌───────┐    ┌───────┐    ┌───────┐
│Batch 1│    │Batch 2│    │Batch N│
│ 100   │    │ 100   │    │ 100   │
└───┬───┘    └───┬───┘    └───┬───┘
    │             │             │
    ↓             ↓             ↓
  Thread        Thread        Thread
  Pool          Pool          Pool
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ↓
         ┌───────────────┐
         │ Combined DF   │
         └───────────────┘
```

## Acceptance Criteria

- [ ] Fetches 100+ tickers in parallel
- [ ] Returns aligned DataFrame (same dates for all)
- [ ] Uses adjusted close (handles splits/dividends)
- [ ] Gracefully skips failed tickers
- [ ] Respects API rate limits

## Configuration

- `batch_size`: 100 (tickers per batch)
- `max_workers`: 10 (concurrent threads)
- `retry_count`: 3 (retries on failure)

## Notes

- yfinance allows batch downloads: `yf.download(["AAPL", "MSFT", ...])`
- Consider adding progress indicator for long fetches
