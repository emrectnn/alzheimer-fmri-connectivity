"""Tests for the extended feature sets and metrics."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

KOD = Path(__file__).resolve().parents[1]
if str(KOD) not in sys.path:
    sys.path.insert(0, str(KOD))


def test_alff_falff_shape(rng):
    """ALFF/fALFF have the right shape and valid ranges."""
    ar = import_module('02b_alff_reho')
    T, n_roi = 150, 20
    ts = rng.standard_normal((T, n_roi))
    alff, falff = ar.compute_alff_falff(ts, tr=3.0, band=(0.01, 0.08))
    assert alff.shape == (n_roi,)
    assert falff.shape == (n_roi,)
    assert np.all(alff >= 0)
    assert np.all((falff >= 0) & (falff <= 1))
    assert not np.any(np.isnan(alff))


def test_reho_roi_shape(rng):
    """ROI-level ReHo has the right shape and valid range."""
    ar = import_module('02b_alff_reho')
    T, n_roi = 120, 15
    ts = rng.standard_normal((T, n_roi))
    reho = ar.compute_reho_roi(ts, k_neighbors=4)
    assert reho.shape == (n_roi,)
    assert np.all((reho >= 0) & (reho <= 1.01))
    assert not np.any(np.isnan(reho))


def test_nbs_edge_selector_cv_safe(rng):
    """NBS edge selector returns a valid boolean support mask."""
    nbs = import_module('06b_nbs')
    n_subj, n_roi = 30, 15
    n_edges = n_roi * (n_roi - 1) // 2
    X = rng.standard_normal((n_subj, n_edges))
    y = np.array([0] * 15 + [1] * 15)
    X[y == 1, :5] += 2.0
    sel = nbs.NBSEdgeSelector(
        n_roi=n_roi, thresh=2.0, n_perm=50, alpha=0.10,
        random_state=42, fallback_topk=10)
    X_sel = sel.fit_transform(X, y)
    assert X_sel.shape[0] == n_subj
    assert X_sel.shape[1] >= 1
    mask = sel.get_support()
    assert mask.dtype == bool
    assert mask.shape == (n_edges,)


def test_ordinal_regression_smoke(synthetic_feature_df):
    """Ordinal model trains and predicts valid 3-class labels."""
    ec = import_module('08e_run')
    if not ec.HAS_MORD:
        pytest.skip('mord yuklu degil')
    df = synthetic_feature_df
    X = df[[c for c in df.columns if c.startswith('feat_')]].values
    y = df['label'].values
    models = ec._build_models(X.shape[1], task='3class')
    assert 'OrdinalAT' in models, 'OrdinalAT _build_models icinde olmali'
    pipe = models['OrdinalAT']
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_cv_score_per_class_keys(synthetic_feature_df):
    """cv_score returns AUC/accuracy, per-class recalls and a confusion matrix."""
    ec = import_module('08e_run')
    from sklearn.linear_model import LogisticRegression
    df = synthetic_feature_df
    X = df[[c for c in df.columns if c.startswith('feat_')]].values
    y = df['label'].values
    model = ec.make_imb_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    out = ec.cv_score(X, y, model, n_splits=3, n_repeats=2, random_state=42)
    assert 'auc' in out and 'acc' in out
    pc_keys = [k for k in out if k.startswith('recall_cls')]
    assert len(pc_keys) >= 2
    assert 'confusion_matrix' in out
    cm = np.asarray(out['confusion_matrix'])
    assert cm.shape == (3, 3)
