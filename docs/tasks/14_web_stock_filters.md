# Task 14: Web Interface Stock Filters (Market Cap & Volume)

**Phase**: Web Interface Enhancement
**Status**: Pending Review
**Depends on**: FMP Client (fmp_client.py), Web Interface (web.py)

---

## Objective

Add custom filtering controls to the web interface allowing users to filter stocks by market cap and volume before running the pipeline.

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
│  Web UI: [Data Source ▼] [Market Cap Range ▼] [Volume ▼]   │
│          [Max Stocks] [Days] [Method]                       │
│              │                                               │
│              ▼                                               │
│  ┌─────────────────────────────────────────────┐            │
│  │ Data Source = "Sample"                       │            │
│  │   └─→ Hardcoded tickers (current behavior)  │            │
│  │                                              │            │
│  │ Data Source = "FMP Filtered"                 │            │
│  │   └─→ FMP API with market cap/volume filters │            │
│  └─────────────────────────────────────────────┘            │
│              │                                               │
│              ▼                                               │
│  FMPClient.get_stock_screener(                              │
│      market_cap_min = X,                                    │
│      market_cap_max = Y,                                    │
│      volume_min = Z,                                        │
│  )                                                          │
│              │                                               │
│              ▼                                               │
│  FMPClient.get_batch_historical_prices(tickers)             │
│              │                                               │
│              ▼                                               │
│  Cluster ──→ Export (with stock metadata)                   │
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
│  │ ● FMP Filtered (custom filters)     │ ← Requires API key │
│  └─────────────────────────────────────┘                    │
│                                                              │
│  ─── Market Cap ───────────────────────                     │
│                                                              │
│  Preset: [Large Cap ▼]                                      │
│    • Large Cap ($10B+)                                      │
│    • Mid Cap ($2B - $10B)                                   │
│    • Small Cap ($300M - $2B)                                │
│    • Micro Cap (<$300M)                                     │
│    • All Caps                                               │
│    • Custom...                                              │
│                                                              │
│  Min: [$________] Max: [$________]  (shown if Custom)       │
│                                                              │
│  ─── Volume ───────────────────────────                     │
│                                                              │
│  Min Daily Volume: [100,000 ▼]                              │
│    • Any                                                    │
│    • 100K+                                                  │
│    • 500K+                                                  │
│    • 1M+                                                    │
│    • Custom...                                              │
│                                                              │
│  ─── Limits ───────────────────────────                     │
│                                                              │
│  Max Stocks: [200_____]                                     │
│  History:    [180 days ▼]                                   │
│  Method:     [Hierarchical ▼]                               │
│                                                              │
│  [▶ Run Pipeline]                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Market Cap Presets

| Preset | Min | Max | Typical Count |
|--------|-----|-----|---------------|
| Large Cap | $10B | - | ~500-700 |
| Mid Cap | $2B | $10B | ~800-1000 |
| Small Cap | $300M | $2B | ~1500-2000 |
| Micro Cap | - | $300M | ~3000+ |
| All Caps | - | - | ~6000+ |

### Volume Presets

| Preset | Min Volume | Description |
|--------|------------|-------------|
| Any | 0 | Include all |
| 100K+ | 100,000 | Liquid |
| 500K+ | 500,000 | Very liquid |
| 1M+ | 1,000,000 | High volume |

---

## API Changes

### POST /api/pipeline/run

**Current Parameters:**
```json
{
  "use_sample": true,
  "sample_size": 50,
  "days": 180,
  "method": "hierarchical"
}
```

**New Parameters:**
```json
{
  "data_source": "fmp_filtered",  // "sample" | "fmp_filtered"
  "filters": {
    "market_cap_min": 10000000000,   // $10B (null = no minimum)
    "market_cap_max": null,           // No maximum
    "volume_min": 100000,             // 100K daily volume
    "volume_max": null                // No maximum
  },
  "max_stocks": 200,                  // Limit results
  "days": 180,
  "method": "hierarchical"
}
```

### Backward Compatibility

- If `use_sample: true` is sent (old format), treat as `data_source: "sample"`
- If `data_source` is not provided, default to `"sample"`

---

## Implementation Tasks

### Phase 1: Backend API (web.py)

| # | Task | Description |
|---|------|-------------|
| 1.1 | Update `/api/pipeline/run` | Accept new filter parameters |
| 1.2 | Add FMP integration to pipeline runner | Call FMPClient when `data_source="fmp_filtered"` |
| 1.3 | Add `/api/fmp/status` endpoint | Check if FMP API key is configured |
| 1.4 | Handle max_stocks limit | Truncate universe after filtering |

### Phase 2: Frontend UI (pipeline.html)

| # | Task | Description |
|---|------|-------------|
| 2.1 | Add data source radio buttons | Sample vs FMP Filtered |
| 2.2 | Add market cap preset dropdown | With Custom option |
| 2.3 | Add volume preset dropdown | With Custom option |
| 2.4 | Add max stocks input | Limit universe size |
| 2.5 | Show/hide filters based on data source | Only show when FMP selected |
| 2.6 | Show FMP API status indicator | Green/red badge |
| 2.7 | Update form submission | Send new parameters |

### Phase 3: Pipeline Integration

| # | Task | Description |
|---|------|-------------|
| 3.1 | Add filter parameters to PipelineConfig | Support filter dict |
| 3.2 | Update pipeline.py to use FMP | When filters provided |
| 3.3 | Show filter summary in logs | "Filtering: Large Cap, 100K+ volume" |
| 3.4 | Store stock metadata | Market cap, sector for each ticker |

### Phase 4: Results Enhancement

| # | Task | Description |
|---|------|-------------|
| 4.1 | Include filter info in results | What filters were applied |
| 4.2 | Show stock metadata in clusters view | Market cap, sector |
| 4.3 | Export metadata to database | Store with cluster assignments |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/price_correlation/web.py` | New API parameters, FMP integration |
| `src/price_correlation/templates/pipeline.html` | Filter UI controls |
| `src/price_correlation/pipeline.py` | Accept and apply filters |
| `src/price_correlation/fmp_client.py` | (No changes - already supports filtering) |

## Files to Create

None - all functionality builds on existing modules.

---

## Flow Diagram

```
User clicks "Run Pipeline"
         │
         ▼
┌─────────────────────────┐
│ data_source = ?         │
└─────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 sample    fmp_filtered
    │         │
    ▼         ▼
┌────────┐  ┌─────────────────────────────────────┐
│Hardcode│  │ FMPClient.get_stock_screener(       │
│50 ticks│  │   market_cap_min = filters.mcap_min │
└────────┘  │   market_cap_max = filters.mcap_max │
    │       │   volume_min = filters.vol_min      │
    │       │ )                                   │
    │       └─────────────────────────────────────┘
    │                    │
    │                    ▼
    │         ┌──────────────────────┐
    │         │ Truncate to max_stocks│
    │         └──────────────────────┘
    │                    │
    └────────┬───────────┘
             ▼
    ┌─────────────────┐
    │ Fetch Prices    │
    │ (yfinance/FMP)  │
    └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Run Clustering  │
    └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Export Results  │
    └─────────────────┘
```

---

## Validation Checks

Before running with FMP:
1. Check FMP_API_KEY is set
2. Validate filter ranges (min < max)
3. Warn if expected stock count is very large (>1000)
4. Show estimated API calls needed

---

## Error Handling

| Error | User Message |
|-------|--------------|
| No FMP API key | "FMP API key required. Set FMP_API_KEY environment variable." |
| Invalid filter range | "Invalid filter: minimum must be less than maximum" |
| FMP rate limit | "API rate limit reached. Try again in X seconds." |
| No stocks match filters | "No stocks found matching your filters. Try relaxing criteria." |

---

## Example Scenarios

### Scenario 1: Large Cap Technology Stocks
```
Data Source: FMP Filtered
Market Cap: Large Cap ($10B+)
Volume: 500K+
Max Stocks: 100

→ Expected: ~100 large tech companies
→ API calls: 1 (screener) + 100 (prices) = 101
```

### Scenario 2: Small Cap High Volume
```
Data Source: FMP Filtered
Market Cap: Small Cap ($300M - $2B)
Volume: 1M+
Max Stocks: 200

→ Expected: ~150-200 liquid small caps
→ API calls: 1 (screener) + 200 (prices) = 201
```

### Scenario 3: Quick Sample Run
```
Data Source: Sample
Max Stocks: 50

→ Uses hardcoded list, no API calls
→ Fastest option for testing
```

---

## Notes

- FMP free tier: 250 requests/day - sufficient for 1 full run with ~200 stocks
- Consider adding sector filter in future phase
- Stock metadata could enable post-clustering analysis (e.g., "which sectors cluster together")
