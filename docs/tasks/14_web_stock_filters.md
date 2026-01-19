# Task 14: Web Interface Stock Filters (Market Cap & Volume)

**Phase**: Web Interface Enhancement
**Status**: Pending Review
**Depends on**: FMP Client (fmp_client.py), Web Interface (web.py)

---

## Objective

Add custom filtering controls to the web interface allowing users to filter stocks by market cap and volume before running the pipeline. Include a "Preview Universe" feature to see stock counts before running.

## Current State

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT FLOW                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Web UI: [Sample Size] [Days] [Method]                      │
│              │                                               │
│              ▼                                               │
│  Pipeline: use_sample=True                                   │
│              │                                               │
│              ▼                                               │
│  get_sample_tickers(n) ──→ Hardcoded list of 50 tickers     │
│              │                                               │
│              ▼                                               │
│  yfinance: fetch prices ──→ Cluster ──→ Export              │
│                                                              │
│  LIMITATION: No market cap or volume filtering              │
└─────────────────────────────────────────────────────────────┘
```

## Proposed Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    NEW FLOW                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Web UI: [Data Source ▼] [Market Cap ▼] [Volume ▼]         │
│          [Preview Universe] [Max Stocks] [Days] [Method]    │
│              │                                               │
│              ▼                                               │
│  ┌─────────────────────────────────────────────┐            │
│  │ Data Source = "Sample"                       │            │
│  │   └─→ Hardcoded tickers (current behavior)  │            │
│  │                                              │            │
│  │ Data Source = "FMP All Stocks"               │            │
│  │   └─→ Full universe, auto-split ranges      │            │
│  │                                              │            │
│  │ Data Source = "FMP Filtered"                 │            │
│  │   └─→ FMP API with market cap/volume filters │            │
│  └─────────────────────────────────────────────┘            │
│              │                                               │
│              ▼                                               │
│  [Preview] ──→ Show count before running                    │
│              │                                               │
│              ▼                                               │
│  [Run] ──→ Fetch Prices ──→ Cluster ──→ Export              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## UI Design

### Filter Controls (Pipeline Page)

```
┌─────────────────────────────────────────────────────────────┐
│  Configuration                                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Source                                                 │
│  ┌─────────────────────────────────────┐                    │
│  │ ○ Sample (50 popular stocks)        │ ← Fast, no API     │
│  │ ○ FMP All Stocks (no filter)        │ ← Full universe    │
│  │ ● FMP Filtered (custom filters)     │ ← With filters     │
│  └─────────────────────────────────────┘                    │
│                                                              │
│  ─── Market Cap ─────────────────────── (FMP modes only)    │
│                                                              │
│  Preset: [Large Cap ▼]                                      │
│    • All Caps (no filter)                                   │
│    • Large Cap ($10B+)                                      │
│    • Mid Cap ($2B - $10B)                                   │
│    • Small Cap ($300M - $2B)                                │
│    • Micro Cap (<$300M)                                     │
│    • Custom...                                              │
│                                                              │
│  Min: [$________] Max: [$________]  (shown if Custom)       │
│                                                              │
│  ─── Volume ───────────────────────────                     │
│                                                              │
│  Min Daily Volume: [Any ▼]                                  │
│    • Any (no filter)                                        │
│    • 100K+                                                  │
│    • 500K+                                                  │
│    • 1M+                                                    │
│    • Custom...                                              │
│                                                              │
│  ─── Limits ───────────────────────────                     │
│                                                              │
│  Max Stocks: [500_____] (0 = no limit)                      │
│  History:    [180 days ▼]                                   │
│  Method:     [Hierarchical ▼]                               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ [🔍 Preview Universe]  ← Shows count before running   │   │
│  │                                                       │   │
│  │  Preview Result:                                      │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ Found: 1,247 stocks matching filters            │ │   │
│  │  │ NYSE: 623  |  NASDAQ: 624                       │ │   │
│  │  │ Will process: 500 (limited by Max Stocks)       │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  [▶ Run Pipeline]                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Source Options

| Option | Description | API Usage |
|--------|-------------|-----------|
| Sample | Hardcoded 50 popular stocks | None |
| FMP All Stocks | Complete NYSE/NASDAQ universe, no filters | High (iterative) |
| FMP Filtered | Apply market cap and/or volume filters | Medium |

### Market Cap Presets

| Preset | Min | Max | Typical Count |
|--------|-----|-----|---------------|
| All Caps | - | - | ~6000+ |
| Large Cap | $10B | - | ~500-700 |
| Mid Cap | $2B | $10B | ~800-1000 |
| Small Cap | $300M | $2B | ~1500-2000 |
| Micro Cap | - | $300M | ~3000+ |

### Volume Presets

| Preset | Min Volume | Description |
|--------|------------|-------------|
| Any | 0 | No filter (include all) |
| 100K+ | 100,000 | Liquid |
| 500K+ | 500,000 | Very liquid |
| 1M+ | 1,000,000 | High volume |

---

## Auto-Split Logic for "All Stocks" Mode

When fetching with no filters (or wide filters), use iterative range splitting:

```
SPLIT_THRESHOLD = 475  (slightly below API limit of 500)

FUNCTION fetch_all_stocks():
    ranges_queue = [
        ($1T+), ($100B-$1T), ($50B-$100B), ($10B-$50B),
        ($5B-$10B), ($2B-$5B), ($1B-$2B), ($500M-$1B),
        ($300M-$500M), ($100M-$300M), ($50M-$100M),
        ($10M-$50M), (<$10M)
    ]

    all_stocks = []

    WHILE ranges_queue not empty:
        range = ranges_queue.pop()
        stocks = fetch_screener(range)

        IF count(stocks) >= SPLIT_THRESHOLD:
            # Split range in half and re-queue
            (range_lower, range_upper) = split_range(range)
            ranges_queue.prepend(range_lower, range_upper)
            LOG "Range {range} returned {count} stocks, splitting..."
        ELSE:
            all_stocks.append(stocks)
            LOG "Range {range}: +{count} stocks"

    RETURN deduplicate(all_stocks)
```

This ensures we capture ALL stocks even when a single range would exceed API limits.

---

## API Changes

### GET /api/universe/preview

**New endpoint** - Preview universe count without running pipeline.

**Request:**
```json
{
  "data_source": "fmp_filtered",
  "filters": {
    "market_cap_min": 10000000000,
    "market_cap_max": null,
    "volume_min": 100000,
    "volume_max": null
  }
}
```

**Response:**
```json
{
  "total_count": 1247,
  "by_exchange": {
    "NYSE": 623,
    "NASDAQ": 624
  },
  "filters_applied": {
    "market_cap": ">= $10B",
    "volume": ">= 100K"
  },
  "sample_tickers": ["AAPL", "MSFT", "GOOGL", "..."]
}
```

### POST /api/pipeline/run

**Updated Parameters:**
```json
{
  "data_source": "fmp_all",       // "sample" | "fmp_all" | "fmp_filtered"
  "filters": {
    "market_cap_min": null,       // null = no minimum
    "market_cap_max": null,       // null = no maximum
    "volume_min": null,           // null = no minimum
    "volume_max": null            // null = no maximum
  },
  "max_stocks": 500,              // 0 = no limit
  "days": 180,
  "method": "hierarchical"
}
```

### Data Source Values

| Value | Behavior |
|-------|----------|
| `sample` | Use hardcoded 50 tickers (fast, no API) |
| `fmp_all` | Fetch ALL stocks using iterative range splitting |
| `fmp_filtered` | Fetch stocks matching filter criteria |

### Backward Compatibility

- `use_sample: true` → treated as `data_source: "sample"`
- Missing `data_source` → defaults to `"sample"`

---

## Implementation Tasks

### Phase 1: Backend API (web.py)

| # | Task | Description |
|---|------|-------------|
| 1.1 | Add `/api/universe/preview` endpoint | Return stock count for given filters |
| 1.2 | Update `/api/pipeline/run` | Accept new filter parameters |
| 1.3 | Add `fmp_all` data source support | Use `get_full_universe_iterative()` |
| 1.4 | Add `fmp_filtered` data source support | Use `get_stock_screener()` with filters |
| 1.5 | Handle max_stocks limit | Truncate universe after fetching |
| 1.6 | Update split threshold | Set to 475 in fmp_client.py |

### Phase 2: Frontend UI (pipeline.html)

| # | Task | Description |
|---|------|-------------|
| 2.1 | Add data source radio buttons | Sample / FMP All / FMP Filtered |
| 2.2 | Add market cap preset dropdown | Including "All Caps" option |
| 2.3 | Add custom market cap inputs | Min/Max fields (shown for Custom) |
| 2.4 | Add volume preset dropdown | Including "Any" option |
| 2.5 | Add "Preview Universe" button | Calls preview API, shows results |
| 2.6 | Add preview results display | Stock count, exchange breakdown |
| 2.7 | Show/hide filters based on data source | Hide for Sample mode |
| 2.8 | Update form submission | Send new parameters |

### Phase 3: Pipeline Integration

| # | Task | Description |
|---|------|-------------|
| 3.1 | Add filter parameters to PipelineConfig | Support filter dict |
| 3.2 | Update pipeline.py Step 1 | Use FMP when data_source is fmp_* |
| 3.3 | Show filter summary in logs | "Filters: All Caps, Volume 100K+" |
| 3.4 | Show progress during universe fetch | "Fetching range $10B-$50B..." |
| 3.5 | Store stock metadata | Market cap, sector for each ticker |

### Phase 4: FMP Client Updates

| # | Task | Description |
|---|------|-------------|
| 4.1 | Update split_threshold default | Change from 500 to 475 |
| 4.2 | Add preview method | `preview_universe()` - count only, no price fetch |
| 4.3 | Improve progress callbacks | More granular progress updates |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/price_correlation/web.py` | New endpoints, filter parameter handling |
| `src/price_correlation/templates/pipeline.html` | Filter UI, preview button |
| `src/price_correlation/pipeline.py` | Accept filters, use FMP client |
| `src/price_correlation/fmp_client.py` | Update split_threshold to 475, add preview |

---

## Flow Diagram

```
User opens Pipeline page
         │
         ▼
┌─────────────────────────────┐
│ Select Data Source          │
│   ○ Sample                  │
│   ○ FMP All Stocks          │
│   ○ FMP Filtered            │
└─────────────────────────────┘
         │
         ▼ (if FMP mode)
┌─────────────────────────────┐
│ Configure Filters           │
│   Market Cap: [Large Cap ▼] │
│   Volume: [100K+ ▼]         │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ [🔍 Preview Universe]       │──────────────────┐
└─────────────────────────────┘                  │
         │                                        ▼
         │                            ┌─────────────────────┐
         │                            │ GET /api/universe/  │
         │                            │     preview         │
         │                            │                     │
         │                            │ Returns: 1,247      │
         │                            │ stocks found        │
         │                            └─────────────────────┘
         │                                        │
         │◄───────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ User reviews count          │
│ Adjusts filters if needed   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ [▶ Run Pipeline]            │
└─────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                     PIPELINE EXECUTION                      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  IF data_source == "sample":                               │
│      tickers = hardcoded_50_tickers                        │
│                                                             │
│  ELIF data_source == "fmp_all":                            │
│      tickers = get_full_universe_iterative(                │
│          split_threshold=475                               │
│      )                                                     │
│      # Auto-splits ranges when count >= 475                │
│                                                             │
│  ELIF data_source == "fmp_filtered":                       │
│      tickers = get_stock_screener(                         │
│          market_cap_min, market_cap_max,                   │
│          volume_min, volume_max                            │
│      )                                                     │
│                                                             │
│  IF max_stocks > 0:                                        │
│      tickers = tickers[:max_stocks]                        │
│                                                             │
│  prices = fetch_prices(tickers)                            │
│  clusters = run_clustering(prices)                         │
│  export(clusters)                                          │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Example Scenarios

### Scenario 1: Full Universe Run
```
Data Source: FMP All Stocks
Market Cap: All Caps (no filter)
Volume: Any (no filter)
Max Stocks: 0 (no limit)

→ Uses iterative range splitting
→ Fetches ~6000+ stocks across all market caps
→ Auto-splits any range returning >= 475 stocks
```

### Scenario 2: Large Cap Only
```
Data Source: FMP Filtered
Market Cap: Large Cap ($10B+)
Volume: 100K+
Max Stocks: 500

→ Single API call to screener
→ Expected: ~500-700 stocks
→ Returns first 500 if more found
```

### Scenario 3: Preview Before Run
```
1. User selects "FMP Filtered"
2. Sets Market Cap: Mid Cap ($2B-$10B)
3. Sets Volume: 500K+
4. Clicks "Preview Universe"
5. Sees: "Found 847 stocks"
6. Adjusts Max Stocks to 300
7. Clicks "Run Pipeline"
```

### Scenario 4: Quick Test
```
Data Source: Sample
Max Stocks: 50

→ Uses hardcoded list
→ No API calls
→ Fastest option
```

---

## Notes

- **API Limits**: High API call cap available - no need to worry about rate limiting
- **Split Threshold**: 475 (below API's 500 limit to ensure complete data)
- **Preview is fast**: Only counts stocks, doesn't fetch prices
- **Metadata preserved**: Stock market cap/sector stored for analysis
