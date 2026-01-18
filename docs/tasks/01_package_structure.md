# Task 01: Package Structure

**Phase**: Setup
**Status**: ⬜ Not Started
**Depends on**: None

---

## Objective

Create the directory layout and initial files for the Python package.

## Directory Structure

```
price_correlation/
├── src/
│   └── price_correlation/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── output/
│   └── .gitkeep
├── pyproject.toml
└── README.md
```

## Steps

1. Create `src/price_correlation/` directory
2. Create `tests/` directory
3. Create `output/` directory for generated files
4. Add `__init__.py` files to make packages
5. Create minimal `pyproject.toml`
6. Create basic `README.md`

## Acceptance Criteria

- [ ] Directory structure exists
- [ ] `pip install -e .` runs without error
- [ ] `import price_correlation` works in Python

## Notes

- Use `src/` layout (modern Python packaging)
- Keep `__init__.py` files minimal initially
