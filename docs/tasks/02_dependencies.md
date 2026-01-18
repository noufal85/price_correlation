# Task 02: Dependencies Configuration

**Phase**: Setup
**Status**: ⬜ Not Started
**Depends on**: Task 01

---

## Objective

Configure all project dependencies in `pyproject.toml`.

## Dependencies

### Core (Required)
```
numpy          - Array operations
pandas         - DataFrames
scipy          - Distance calculations, clustering
scikit-learn   - DBSCAN, silhouette
pyarrow        - Parquet export
yfinance       - Stock price data
```

### Development
```
pytest         - Testing
ruff           - Linting and formatting
```

### Optional
```
tslearn        - DTW-based clustering
statsmodels    - Cointegration tests
matplotlib     - Visualization
```

## pyproject.toml Structure

```
[project]
  name, version, description
  dependencies = [core deps]

[project.optional-dependencies]
  dev = [pytest, ruff]
  full = [tslearn, statsmodels, matplotlib]

[build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

[tool.ruff]
  line-length = 88
```

## Acceptance Criteria

- [ ] `pip install -e .` installs all core deps
- [ ] `pip install -e ".[dev]"` adds dev tools
- [ ] All imports work:
  - `import numpy, pandas, scipy, sklearn, pyarrow, yfinance`

## Notes

- Pin minimum versions for reproducibility
- Use hatchling as build backend (simple, modern)
