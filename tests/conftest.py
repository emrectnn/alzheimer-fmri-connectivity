"""Shared pytest fixtures providing synthetic data for the test suite."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

KOD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if KOD_DIR not in sys.path:
    sys.path.insert(0, KOD_DIR)

SEED = 42


@pytest.fixture(scope="session")
def rng():
    """Session-wide deterministic random generator (seed=42)."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def synthetic_timeseries_list(rng):
    """List of synthetic ROI time series with modular structure."""
    n_subj, n_tr, n_roi = 20, 100, 30
    subjects = []
    for _ in range(n_subj):
        ts = np.zeros((n_tr, n_roi))
        for cluster in range(3):
            factor = rng.standard_normal(n_tr)
            for roi in range(cluster * 10, (cluster + 1) * 10):
                ts[:, roi] = factor * 0.6 + rng.standard_normal(n_tr) * 0.4
        subjects.append(ts)
    return subjects


@pytest.fixture(scope="session")
def synthetic_corr_matrix(synthetic_timeseries_list):
    """Symmetric, zero-diagonal correlation matrix fixture."""
    ts = synthetic_timeseries_list[0]
    ts_std = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-12)
    corr = (ts_std.T @ ts_std) / (ts.shape[0] - 1)
    np.fill_diagonal(corr, 0)
    return corr


@pytest.fixture(scope="session")
def synthetic_binary_matrix(synthetic_corr_matrix):
    """15%-density thresholded binary matrix fixture."""
    n = synthetic_corr_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vals = np.abs(synthetic_corr_matrix[triu_idx])
    threshold = np.quantile(vals, 1 - 0.15)
    binary = (np.abs(synthetic_corr_matrix) >= threshold).astype(int)
    np.fill_diagonal(binary, 0)
    return binary


@pytest.fixture(scope="session")
def synthetic_graph(synthetic_binary_matrix):
    """NetworkX graph built from the synthetic binary matrix."""
    import networkx as nx
    return nx.from_numpy_array(synthetic_binary_matrix)


@pytest.fixture(scope="session")
def synthetic_feature_df(rng):
    """Synthetic feature DataFrame with label/group/subject_id columns."""
    groups_config = [("HC", 12, 0), ("MCI", 10, 1), ("AD", 8, 2)]
    rows, group_col, subj_ids = [], [], []
    idx = 0
    for gname, n, shift in groups_config:
        for _ in range(n):
            signal = rng.standard_normal(10) + shift * 0.3
            noise = rng.standard_normal(10)
            rows.append(np.concatenate([signal, noise]))
            group_col.append(gname)
            subj_ids.append(f"sub-{idx:03d}")
            idx += 1
    X = np.stack(rows)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(20)])
    df["group"] = group_col
    df["label"] = df["group"].map({"HC": 0, "MCI": 1, "AD": 2})
    df["subject_id"] = subj_ids
    return df
