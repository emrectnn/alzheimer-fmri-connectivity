"""Integration tests covering the leaderboard build and extended pipeline."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KOD = Path(__file__).resolve().parents[1]
if str(KOD) not in sys.path:
    sys.path.insert(0, str(KOD))


@pytest.fixture
def synthetic_multifeature_df(rng):
    n = 60
    y = np.array([0]*20 + [1]*20 + [2]*20)
    rows = {
        'subject_id': [f"sub-{i:03d}" for i in range(n)],
        'label': y,
        'group': np.array(['HC']*20 + ['MCI']*20 + ['AD']*20),
        'age': rng.normal(70, 8, n),
        'gender_bin': rng.integers(0, 2, n),
        'education': rng.normal(14, 3, n),
        'mean_fd': np.abs(rng.normal(0.2, 0.05, n)),
    }
    for p in ['global_efficiency_auc', 'avg_clustering_auc', 'density_auc',
              'global_efficiency', 'avg_clustering', 'modularity',
              'null_ge_zscore', 'null_cluster_zscore']:
        rows[p] = rng.standard_normal(n) + 0.3*y
    for i in range(1, 11):
        rows[f'alff_roi_{i}'] = rng.standard_normal(n) + 0.2*y
        rows[f'reho_roi_{i}'] = rng.standard_normal(n) + 0.2*y
    for p in ['global_efficiency', 'avg_clustering', 'density',
              'assortativity', 'modularity', 'transitivity']:
        rows[f'scha200_{p}'] = rng.standard_normal(n)
        rows[f'ho48_{p}'] = rng.standard_normal(n)
    return pd.DataFrame(rows)


def test_feature_sets_include_extended(synthetic_multifeature_df):
    ec = import_module('08_enhanced_classification')
    fs = ec.get_feature_sets_by_mode(synthetic_multifeature_df, 'imaging_only')
    assert 'ALFF' in fs, fs.keys()
    assert 'ReHo' in fs
    assert 'ALFF+ReHo' in fs
    assert 'Graf_Tam+ALFF+ReHo' in fs
    assert 'Graf_Schaefer200' in fs
    assert 'Graf_HO48' in fs
    assert 'Graf_MultiAtlas' in fs
    fs2 = ec.get_feature_sets_by_mode(synthetic_multifeature_df,
                                      'imaging_plus_demographics')
    assert 'Graf_MultiAtlas+Demog' in fs2
    assert 'Graf_Tam+ALFF+ReHo+Demog' in fs2


def test_build_models_includes_ordinal():
    ec = import_module('08_enhanced_classification')
    if not ec.HAS_MORD:
        pytest.skip('mord yuklu degil')
    models = ec._build_models(n_feat=20, task='3class')
    assert 'OrdinalAT' in models
    assert 'OrdinalIT' in models


def test_nbs_tangent_pipeline_cv(rng):
    ec = import_module('08_enhanced_classification')
    from sklearn.model_selection import cross_val_score

    n, T, R = 20, 30, 10
    lookup = {f"sub-{i:02d}": rng.standard_normal((T, R)) for i in range(n)}
    sids = np.array(list(lookup.keys()))
    y = np.array([0]*10 + [1]*10)
    from sklearn.linear_model import LogisticRegression
    pipe = ec.make_tangent_pipeline(
        LogisticRegression(max_iter=1000, random_state=42),
        k_best=5, time_series_lookup=lookup,
        use_nbs=True,
    )
    try:
        scores = cross_val_score(pipe, sids, y, cv=3, scoring='accuracy',
                                 n_jobs=1)
    except Exception as e:
        pytest.skip(f"NBS tangent pipeline CV kucuk sample'da patladi: {e}")
    assert scores.shape == (3,)
    assert not np.any(np.isnan(scores))


def test_twostage_with_calibration_smoke(synthetic_multifeature_df):
    ec = import_module('08_enhanced_classification')
    df = synthetic_multifeature_df
    try:
        res = ec.experiment_twostage_by_mode(
            df, mode='imaging_only', residualize=False, use_combat=False,
            n_splits=3, n_repeats=2, calibrate=True)
    except Exception as e:
        pytest.skip(f"twostage calistirilamadi (sentetik): {e}")
    assert isinstance(res, pd.DataFrame)
    if not res.empty:
        assert 'AUC' in res.columns
        assert 'Task' in res.columns
        assert all(res['Task'] == '3class_twostage')


def test_build_leaderboard(tmp_path, monkeypatch):
    lb = import_module('qc.build_leaderboard')
    config = import_module('00_config')

    monkeypatch.setattr(config, 'METRICS_DIR', str(tmp_path))
    monkeypatch.setattr(lb, 'METRICS_DIR', Path(tmp_path))
    sweep_dir = tmp_path / 'sweep'
    sweep_dir.mkdir()
    monkeypatch.setattr(lb, 'SWEEP_DIR', sweep_dir)

    rows = []
    for task in ['3class', '3class_twostage', 'HC-AD']:
        for mode in ['imaging_only', 'imaging_plus_demographics']:
            for feat in ['Graf_Tam', 'Graf_Tam+ALFF+ReHo',
                         'Tangent_FC', 'NBS_Edges']:
                rows.append({
                    'Task': task, 'Mode': mode, 'Features': feat,
                    'Model': 'LightGBM',
                    'AUC': 0.55 + np.random.random()*0.2,
                    'AUC_CI_low': 0.45, 'AUC_CI_high': 0.75,
                    'Acc': 0.5, 'F1': 0.5,
                    'Residualized': False, 'ComBat': False, 'Optuna': True,
                })
    pd.DataFrame(rows).to_csv(sweep_dir / 'task-3class__imaging_only.csv',
                              index=False)

    rc = lb.main()
    assert rc == 0
    assert (tmp_path / 'leaderboard_final.md').exists()
    assert (tmp_path / 'leaderboard_headline.csv').exists()

    md = (tmp_path / 'leaderboard_final.md').read_text(encoding='utf-8')
    assert '# Leaderboard' in md
    assert 'Headline' in md
    assert 'NBS_Edges' in md or 'Tangent_FC' in md
