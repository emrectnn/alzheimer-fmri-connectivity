"""Tests for the classification pipeline, including cross-validation leakage checks."""

import numpy as np
from importlib import import_module


def test_make_imb_pipeline_has_scaler_and_smote():
    clf_mod = import_module("08_enhanced_classification")
    from sklearn.linear_model import LogisticRegression
    pipe = clf_mod.make_imb_pipeline(LogisticRegression(max_iter=500),
                                     k_best=5, use_smote=True)
    step_names = [name for name, _ in pipe.steps]
    assert step_names[0] == "imputer"
    assert step_names[-1] == "clf"
    assert "scaler" in step_names
    assert "select" in step_names
    if clf_mod.HAS_IMBLEARN:
        assert "smote" in step_names
        smote_i = step_names.index("smote")
        clf_i = step_names.index("clf")
        select_i = step_names.index("select")
        assert select_i < smote_i < clf_i


def test_no_leakage_scaler_fit_called_per_fold(monkeypatch):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate

    original_fit = StandardScaler.fit
    call_count = {"n": 0}

    def counting_fit(self, X, y=None):
        call_count["n"] += 1
        return original_fit(self, X, y)

    monkeypatch.setattr(StandardScaler, "fit", counting_fit)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 10))
    y = rng.integers(0, 3, size=60)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=500))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validate(pipe, X, y, cv=cv, scoring="accuracy")

    assert call_count["n"] == 5, (
        f"Scaler.fit {call_count['n']} kez cagrildi, beklenen 5. "
        "Muhtemelen kodda tum veri uzerinde fit yapiliyor (LEAKAGE)."
    )


def test_get_feature_sets_by_mode_no_clinical_leak(synthetic_feature_df):
    clf_mod = import_module("08_enhanced_classification")
    df = synthetic_feature_df.copy()
    df["mmse"] = 28.0
    df["cdrsb"] = 0.5
    df["cdglobal"] = 0.5
    df["age"] = 70.0
    df["gender_bin"] = 1.0
    df["education"] = 12.0
    df["dev_clustering"] = 0.1
    df["dev_path_length"] = 0.1
    df["dev_global_eff"] = 0.1
    df["dev_sigma"] = 0.1
    for mode in ("imaging_only", "imaging_plus_demographics", "imaging_plus_clinical"):
        sets = clf_mod.get_feature_sets_by_mode(df, mode)
        all_cols = [c for cols in sets.values() for c in cols]
        if mode == "imaging_only":
            for forbidden in ("mmse", "cdrsb", "cdglobal", "age",
                              "gender_bin", "education"):
                assert forbidden not in all_cols, (
                    f"{mode} icinde yasak feature var: {forbidden}"
                )
        elif mode == "imaging_plus_demographics":
            assert "age" in all_cols
            for forbidden in ("mmse", "cdrsb", "cdglobal"):
                assert forbidden not in all_cols, (
                    f"{mode} icinde klinik leakage: {forbidden}"
                )
        elif mode == "imaging_plus_clinical":
            assert "mmse" in all_cols or "cdrsb" in all_cols or "cdglobal" in all_cols


def test_confound_regressor_leakage(monkeypatch):
    import numpy as np
    from importlib import import_module
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.linear_model import LogisticRegression

    clf_mod = import_module("08_enhanced_classification")

    original_fit = LinearRegression.fit
    counter = {"n": 0}

    def counting_fit(self, X, y=None, **kw):
        counter["n"] += 1
        return original_fit(self, X, y)

    monkeypatch.setattr(LinearRegression, "fit", counting_fit)

    rng = np.random.default_rng(0)
    n, p, k_conf = 60, 8, 2
    X_feat = rng.standard_normal((n, p))
    C = rng.standard_normal((n, k_conf))
    X = np.concatenate([X_feat, C], axis=1)
    y = rng.integers(0, 2, size=n)

    pipe = clf_mod.make_imb_pipeline(
        LogisticRegression(max_iter=500),
        n_confounds=k_conf, use_smote=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validate(pipe, X, y, cv=cv, scoring="accuracy")

    expected = 5
    assert counter["n"] == expected, (
        f"LinearRegression.fit {counter['n']} kez cagrildi, beklenen {expected}. "
        "1 ise LEAKAGE (tum X uzerinde fit); 0 ise ConfoundRegressor devre disi."
    )


def test_experiment_binary_by_mode_smoke(synthetic_feature_df):
    from importlib import import_module
    clf_mod = import_module("08_enhanced_classification")
    df = synthetic_feature_df.copy()
    df["age"] = 70.0
    df["gender_bin"] = 1.0
    df["mean_fd"] = 0.1

    res = clf_mod.experiment_binary_by_mode(
        df, mode='imaging_only', pair=('HC', 'AD'),
        residualize=False, time_series_lookup=None)
    assert res is not None
    assert not res.empty, "Binary deneyden sonuc gelmedi"
    for col in ("Task", "Mode", "Residualized", "Features",
                "Model", "Acc", "AUC", "F1"):
        assert col in res.columns, f"Eksik kolon: {col}"
    assert (res["Task"] == "HC-AD").all()


def test_tangent_pipeline_cv_safe(rng):
    import numpy as np
    from importlib import import_module
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression

    clf_mod = import_module("08_enhanced_classification")

    n_subj, n_tr, n_roi = 10, 50, 10
    ts_lookup = {}
    ids = []
    y = []
    for i in range(n_subj):
        sid = f"sub-{i:03d}"
        ids.append(sid)
        shift = (i % 2) * 0.4
        ts = rng.standard_normal((n_tr, n_roi)) + shift
        ts_lookup[sid] = ts
        y.append(i % 2)
    X_ids = np.array(ids)
    y = np.array(y)

    pipe = clf_mod.make_tangent_pipeline(
        LogisticRegression(max_iter=500),
        k_best=min(10, (n_roi * (n_roi - 1)) // 2),
        time_series_lookup=ts_lookup,
        use_smote=False,
    )
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_ids, y, cv=cv, scoring="accuracy")
    assert len(scores) == 2
    assert np.all(np.isfinite(scores)), f"Non-finite skor: {scores}"


def test_repeated_kfold_fit_count():
    import numpy as np
    from importlib import import_module
    from sklearn.linear_model import LogisticRegression

    clf_mod = import_module("08_enhanced_classification")
    rng = np.random.default_rng(0)
    n, p = 40, 8
    X = rng.standard_normal((n, p))
    y = (rng.random(n) > 0.5).astype(int)

    counter = {"n": 0}
    orig_fit = LogisticRegression.fit

    def counting_fit(self, X, y, sample_weight=None):
        counter["n"] += 1
        return orig_fit(self, X, y, sample_weight=sample_weight)

    LogisticRegression.fit = counting_fit
    try:
        pipe = clf_mod.make_imb_pipeline(
            LogisticRegression(max_iter=200), k_best=None, use_smote=False)
        _ = clf_mod.cv_score(X, y, pipe, n_splits=5, n_repeats=3)
    finally:
        LogisticRegression.fit = orig_fit

    assert counter["n"] == 20, f"Beklenen 20 fit, olan {counter['n']}"


def test_combat_transformer_cv_safe():
    import numpy as np
    import pytest
    from importlib import import_module

    clf_mod = import_module("08_enhanced_classification")
    if not getattr(clf_mod, "HAS_NH", False):
        pytest.skip("neuroHarmonize yuklu degil — test atlaniyor.")

    rng = np.random.default_rng(1)
    n, p = 30, 6
    X_feat = rng.standard_normal((n, p))
    site = np.array([0] * 15 + [1] * 15, dtype=float).reshape(-1, 1)
    X = np.concatenate([X_feat, site], axis=1)
    y = (rng.random(n) > 0.5).astype(int)

    counter = {"n": 0}
    orig_learn = clf_mod.harmonizationLearn

    def counting_learn(features, covars, **kwargs):
        counter["n"] += 1
        return orig_learn(features, covars, **kwargs)

    clf_mod.harmonizationLearn = counting_learn
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        pipe = clf_mod.make_imb_pipeline(
            LogisticRegression(max_iter=200),
            k_best=None, use_smote=False,
            n_site_cols=1, n_bio_cols=0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        _ = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    finally:
        clf_mod.harmonizationLearn = orig_learn

    assert counter["n"] == 3, f"Beklenen 3 fit, olan {counter['n']}"


def test_optuna_objective_smoke():
    import numpy as np
    import pytest
    from importlib import import_module

    clf_mod = import_module("08_enhanced_classification")
    if not getattr(clf_mod, "HAS_OPTUNA", False):
        pytest.skip("Optuna yuklu degil — test atlaniyor.")
    if not getattr(clf_mod, "HAS_LGBM", False):
        pytest.skip("LightGBM yuklu degil — test atlaniyor.")

    rng = np.random.default_rng(2)
    n, p = 30, 20
    X = rng.standard_normal((n, p))
    y = (rng.random(n) > 0.5).astype(int)

    models = clf_mod._build_models_optuna(
        X, y, task='binary', n_trials=5,
        n_confounds=None, n_site_cols=0, n_bio_cols=0)
    assert isinstance(models, dict)
    assert len(models) >= 1
    assert 'LightGBM' in models or any('LightGBM' in k for k in models)
