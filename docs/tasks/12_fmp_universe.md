# Task 12: FMP Universe Builder

**Phase**: Data Pipeline (Enhanced)
**Status**: ⬜ Not Started
**Depends on**: Task 02

---

## Objective

Build complete stock universe using Financial Modeling Prep (FMP) API with configurable filters.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FMP UNIVERSE BUILDER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Config File] ──→ [FMP Stock Screener] ──→ [Filter & Validate] │
│                           │                         │            │
│                           ▼                         ▼            │
│                    [Paginated Fetch]         [Universe List]    │
│                    (incremental)                                 │
│                           │                                      │
│                           ▼                                      │
│                    [FMP Historical Prices] ──→ [Price DataFrame]│
│                    (180 days, batched)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## FMP API Endpoints

### 1. Stock Screener
```
GET https://financialmodelingprep.com/api/v3/stock-screener

Parameters:
  - marketCapMoreThan: Minimum market cap (e.g., 1000000000 for $1B)
  - marketCapLowerThan: Maximum market cap
  - volumeMoreThan: Minimum average volume
  - exchange: NYSE, NASDAQ, AMEX
  - isActivelyTrading: true
  - limit: Number per page (max ~10000)
  - apikey: Your API key

Response:
  [
    {
      "symbol": "AAPL",
      "companyName": "Apple Inc.",
      "marketCap": 2800000000000,
      "sector": "Technology",
      "industry": "Consumer Electronics",
      "exchange": "NASDAQ",
      "volume": 50000000,
      ...
    },
    ...
  ]
```

### 2. Historical Prices
```
GET https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}

Parameters:
  - from: Start date (YYYY-MM-DD)
  - to: End date (YYYY-MM-DD)
  - apikey: Your API key

Response:
  {
    "symbol": "AAPL",
    "historical": [
      {"date": "2024-01-15", "close": 185.50, "adjClose": 185.50, ...},
      ...
    ]
  }
```

### 3. Batch Historical Prices
```
GET https://financialmodelingprep.com/api/v3/historical-price-full/{symbols}

Where {symbols} = "AAPL,MSFT,GOOGL" (comma-separated, max ~5 per request)
```

## Configuration Schema

```yaml
# config/filters.yaml

# API Configuration
api:
  key: "${FMP_API_KEY}"        # From environment variable
  base_url: "https://financialmodelingprep.com/api/v3"
  rate_limit: 300              # Requests per minute (free tier)
  batch_size: 5                # Symbols per price request

# Universe Filters
filters:
  # Market cap filter (in USD)
  market_cap:
    min: 100000000             # $100M minimum (null for no minimum)
    max: null                  # No maximum

  # Volume filter
  volume:
    min: 100000                # 100K average daily volume
    max: null

  # Exchange filter
  exchanges:
    - NYSE
    - NASDAQ
    # - AMEX                   # Uncomment to include

  # Active trading only
  is_actively_trading: true

  # Sector filter (null = all sectors)
  sectors: null
  # sectors:
  #   - Technology
  #   - Healthcare
  #   - Financial Services

  # Country filter
  country: "US"

# Price Data Settings
price_data:
  days: 180                    # Last N days of price history
  use_adjusted: true           # Use adjusted close prices
```

## Incremental Fetching Strategy

```
1. Initial Screener Call
   [Request stock-screener with filters]
           ↓
   [Get total count from response]
           ↓
2. Paginated Fetching (if needed)
   FOR offset IN range(0, total, page_size):
       [Fetch page with offset]
       [Append to universe list]
       [Rate limit delay]
           ↓
3. Price Data Fetching
   FOR batch IN chunk(universe, batch_size):
       [Fetch historical prices for batch]
       [Merge into price DataFrame]
       [Progress logging]
       [Rate limit delay]
```

## Module Functions

### universe.py (updated)

```
get_fmp_universe(config) → list[dict]
  - Fetch all stocks matching filters
  - Return list with symbol, name, marketCap, sector, etc.

get_fmp_universe_tickers(config) → list[str]
  - Wrapper returning just ticker symbols
```

### ingestion.py (updated)

```
fetch_fmp_prices(tickers, days, api_key) → DataFrame
  - Fetch historical prices from FMP
  - Batch requests to respect rate limits
  - Return aligned DataFrame
```

## Acceptance Criteria

- [ ] Fetches complete universe with pagination
- [ ] Respects rate limits (configurable delay)
- [ ] Filters work correctly (market cap, volume, exchange)
- [ ] Price data fetched for all universe stocks
- [ ] Progress logging during fetch
- [ ] Config file parsed correctly
- [ ] Environment variable for API key works

## Error Handling

```
- API rate limit exceeded → exponential backoff
- Invalid API key → clear error message
- Ticker not found → skip and log warning
- Network timeout → retry with backoff
```

## Notes

- FMP free tier: 250 requests/day
- FMP starter tier: 300 requests/minute
- Batch price requests reduce API calls significantly
- Cache universe list to avoid repeated screener calls
