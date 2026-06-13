"""Tests for connectivity matrix construction and thresholding."""

import numpy as np
from importlib import import_module

conn = import_module("02a_connectivity")


def test_corr_matrix_properties(synthetic_timeseries_list):
    """Correlation matrix is symmetric, zero-diagonal and bounded by 1."""
    mat = conn.compute_connectivity(synthetic_timeseries_list[0])
    n = mat.shape[0]
    assert mat.shape == (n, n)
    np.testing.assert_allclose(mat, mat.T, atol=1e-10)
    assert np.all(np.diag(mat) == 0)
    assert np.all(np.abs(mat) <= 1.0 + 1e-9)


def test_fisher_z_transform_finite(synthetic_corr_matrix):
    """Fisher z-transform is finite with a zero diagonal."""
    z = conn.fisher_z_transform(synthetic_corr_matrix)
    assert np.all(np.isfinite(z))
    assert z.shape == synthetic_corr_matrix.shape
    np.testing.assert_allclose(np.diag(z), 0, atol=1e-12)


def test_threshold_by_density_target_density(synthetic_corr_matrix):
    """Density thresholding hits the requested edge density."""
    target = 0.15
    binary = conn.threshold_by_density(synthetic_corr_matrix, target)
    n = binary.shape[0]
    off_diag = binary.copy()
    np.fill_diagonal(off_diag, 0)
    actual = off_diag.sum() / (n * (n - 1))
    assert abs(actual - target) < 0.02, f"target={target}, actual={actual}"
    assert np.all(np.diag(binary) == 0)


def test_tangent_transformer_shape_and_cv_safety(synthetic_timeseries_list):
    """Tangent transformer output shape is correct and deterministic on test data."""
    ts = synthetic_timeseries_list
    X_train, X_test = ts[:15], ts[15:]
    t = conn.TangentSpaceTransformer()
    t.fit(X_train)
    assert hasattr(t, "measure_")
    n = X_train[0].shape[1]
    expected_d = n * (n - 1) // 2
    out_train = t.transform(X_train)
    out_test = t.transform(X_test)
    assert out_train.shape == (len(X_train), expected_d)
    assert out_test.shape == (len(X_test), expected_d)
    out_test2 = t.transform(X_test)
    np.testing.assert_allclose(out_test, out_test2, atol=1e-10)
