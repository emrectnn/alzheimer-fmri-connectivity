# Test Suite

Regression tests for the pipeline components, using synthetic fixtures.
The tests do not depend on real ADNI data and run in under 30 seconds.

## Running

```bash
# All tests
pytest tests/ -v

# A single file
pytest tests/test_classification.py -v

# The critical leakage test
pytest tests/test_classification.py::test_no_leakage_scaler_fit_called_per_fold -v
```

## Coverage

| File | What it tests |
|---|---|
| `test_connectivity.py` | Pearson matrix symmetry, Fisher z finiteness, density thresholding, `TangentSpaceTransformer` CV-safety |
| `test_graph_metrics.py` | `binary_to_graph`, no NaNs in global metrics, nodal vector length, density-to-edge monotonicity |
| `test_classification.py` | `make_imb_pipeline` structure, **per-fold `scaler.fit` regression (leakage test)**, confound-regressor and ComBat CV-safety, clinical features not leaking |
| `test_features.py`, `test_integration.py` | ALFF/ReHo and NBS features, extended feature sets, and the leaderboard build |

## Critical test

`test_no_leakage_scaler_fit_called_per_fold` counts how many times
`StandardScaler.fit` is called via monkeypatching. In a 5-fold CV it must be
called exactly five times. If a refactor reintroduces
`scaler.fit_transform(X_full)`, the counter drops to one and the test fails,
guarding against data leakage.

## Synthetic fixtures (`conftest.py`)

- 20 subjects x 100 TR x 30 ROI, with three modular clusters to emulate FC structure.
- A 30-subject x 20-feature synthetic DataFrame (HC=12, MCI=10, AD=8) for imaging-only tests.
- Fixed seed (42) so the tests are deterministic.
