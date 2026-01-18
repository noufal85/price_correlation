# Task 11: Integration Tests

**Phase**: Testing
**Status**: ⬜ Not Started
**Depends on**: Task 10

---

## Objective

Write integration tests using REAL data (no mocks) that validate the complete pipeline.

## Location

`tests/test_integration.py`

---

## Testing Philosophy

```
┌────────────────────────────────────────────────────────────┐
│                    TESTING RULES                           │
├────────────────────────────────────────────────────────────┤
│  ✓ Use REAL data from APIs                                │
│  ✓ Combine multiple functions per test                    │
│  ✓ Validate end-to-end flows                              │
│  ✗ NO mocks, NO fixtures with fake data                   │
│  ✗ NO testing isolated helper functions                   │
└────────────────────────────────────────────────────────────┘
```

---

## Test Cases

### Test 1: test_data_pipeline
```
Purpose: Validate data ingestion and preprocessing

Flow:
  [Fetch real prices for 10-20 tickers]
           ↓
  [Run preprocessing pipeline]
           ↓
  [Validate output]

Tickers to use:
  ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
   "JPM", "BAC", "XOM", "JNJ", "PG"]

Assertions:
  - Output DataFrame has expected shape
  - Returns have mean ≈ 0 (after z-score)
  - Returns have std ≈ 1 (after z-score)
  - No NaN values in output
```

### Test 2: test_correlation_and_clustering
```
Purpose: Validate correlation matrix and clustering

Flow:
  [Use preprocessed data from Test 1 pattern]
           ↓
  [Compute correlation matrix]
           ↓
  [Convert to distance]
           ↓
  [Run DBSCAN]
           ↓
  [Run Hierarchical]
           ↓
  [Validate outputs]

Assertions:
  - Correlation matrix is symmetric
  - Distance matrix diagonal = 0
  - Labels array has correct length
  - At least 1 cluster formed (label >= 0)
  - Silhouette score is valid number
```

### Test 3: test_full_pipeline_small
```
Purpose: End-to-end pipeline on small universe

Flow:
  [Run pipeline with ~50 tickers]
           ↓
  [Verify output files created]
           ↓
  [Verify file contents]

Config:
  - Use subset of universe (50 tickers)
  - Short date range (3 months)
  - Both JSON and Parquet output

Assertions:
  - Output files exist
  - JSON is valid and parseable
  - Parquet loads into DataFrame
  - Cluster assignments present for all tickers
```

---

## Test Structure

```python
# tests/test_integration.py

class TestDataPipeline:

    def test_data_pipeline(self):
        """
        Fetch real prices → preprocess → validate
        Tests: fetch_price_history, clean_price_data,
               compute_log_returns, zscore_normalize
        """
        # Real API call
        # Real computation
        # Real validation

class TestClustering:

    def test_correlation_and_clustering(self):
        """
        Compute correlations → cluster → validate
        Tests: compute_correlation_matrix, correlation_to_distance,
               cluster_dbscan, cluster_hierarchical, compute_silhouette
        """
        # Chain of real operations

class TestFullPipeline:

    def test_full_pipeline_small(self):
        """
        Run complete pipeline on small dataset
        Tests: run_pipeline (orchestrator)
        """
        # End-to-end with real data
```

---

## Running Tests

```bash
# Run all integration tests
pytest tests/ -v

# Run specific test
pytest tests/test_integration.py::TestDataPipeline -v

# Run with output
pytest tests/ -v -s

# Note: Tests may take several minutes due to API calls
```

---

## Expected Test Duration

```
┌─────────────────────────┬──────────────┐
│ Test                    │ Duration     │
├─────────────────────────┼──────────────┤
│ test_data_pipeline      │ 10-30 sec    │
│ test_correlation_cluster│ 5-15 sec     │
│ test_full_pipeline_small│ 30-60 sec    │
├─────────────────────────┼──────────────┤
│ Total                   │ 1-2 min      │
└─────────────────────────┴──────────────┘

Note: API rate limits may cause variability
```

---

## Acceptance Criteria

- [ ] All 3 tests pass with real data
- [ ] No mocks or fake fixtures used
- [ ] Tests are in separate `tests/` package
- [ ] Tests combine multiple functions per test case
- [ ] Tests complete in reasonable time (<5 min total)

---

## What NOT to Test

```
✗ Individual helper functions in isolation
✗ Edge cases that require mocking
✗ Every possible input combination
✗ Internal implementation details
```

---

## Test Data Cleanup

```python
# Use pytest fixtures for cleanup if needed

@pytest.fixture(autouse=True)
def cleanup_output():
    yield
    # Remove generated test files after test
    for f in Path("output").glob("test_*"):
        f.unlink()
```
