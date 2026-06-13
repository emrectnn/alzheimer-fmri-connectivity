"""Per-task, per-mode classification experiments.

Runs the 3-class, binary (pairwise) and two-stage experiments for each feature
mode, returning result DataFrames. Reuses feature sets (08a), pipelines (08b)
and the model roster (08c).
"""

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_validate, cross_val_score, GridSearchCV)
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    LGBMClassifier = None

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None
    TPESampler = None

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")

MAX_FEATURES_HARD_LIMIT = 30

# Reuse feature sets, pipelines and models from the sibling modules.
_feat = import_module("08a_features")
_tf = import_module("08b_transformers")
_models = import_module("08c_models")
load_all_features = _feat.load_all_features
get_feature_sets = _feat.get_feature_sets
get_feature_sets_by_mode = _feat.get_feature_sets_by_mode
_prepare_feature_matrix = _feat._prepare_feature_matrix
make_imb_pipeline = _tf.make_imb_pipeline
make_tangent_pipeline = _tf.make_tangent_pipeline
load_timeseries_lookup = _tf.load_timeseries_lookup
cv_score = _models.cv_score
_build_models = _models._build_models
_build_models_optuna = _models._build_models_optuna


def experiment_3class(df):
    """Legacy 3-class experiment over the default feature sets.

    Args:
        df: Merged feature DataFrame.

    Returns:
        Result DataFrame ranked by AUC.
    """
    print("DENEY 1: 3-SINIF (HC vs MCI vs AD)")

    feat_sets = get_feature_sets(df)
    y = df['label'].values

    results = []

    for fs_name, cols in feat_sets.items():
        X = df[cols].values
        if X.shape[1] == 0:
            continue

        models = {
            'SVM_RBF': make_imb_pipeline(
                SVC(kernel='rbf', C=1, probability=True, random_state=42)),
            'SVM_RBF_C10': make_imb_pipeline(
                SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            'RF':        make_imb_pipeline(RandomForestClassifier(200, random_state=42)),
            'GBM': make_imb_pipeline(
                GradientBoostingClassifier(n_estimators=100, random_state=42)),
            'LogReg':    make_imb_pipeline(LogisticRegression(C=1, max_iter=2000, random_state=42)),
            'ElasticNet': make_imb_pipeline(LogisticRegression(
                penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga',
                max_iter=5000, random_state=42)),
            'PCA20+SVM': make_imb_pipeline(
                SVC(kernel='rbf', C=10, probability=True, random_state=42),
                n_pca=min(20, X.shape[1] - 1)),
            'Top15+SVM': make_imb_pipeline(
                SVC(kernel='rbf', C=10, probability=True, random_state=42),
                k_best=min(15, X.shape[1])),
            'Top15+RF': make_imb_pipeline(
                RandomForestClassifier(200, random_state=42),
                k_best=min(15, X.shape[1])),
        }

        if HAS_XGB:
            models['XGBoost'] = make_imb_pipeline(XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, eval_metric='mlogloss', verbosity=0))

        for m_name, model in models.items():
            try:
                r = cv_score(X, y, model)
                results.append({
                    'Features': fs_name,
                    'Model': m_name,
                    'n_feat': X.shape[1],
                    'Acc': r['acc'],
                    'Acc_std': r['acc_std'],
                    'AUC': r['auc'],
                    'AUC_std': r['auc_std'],
                    'F1': r['f1'],
                    'Train_Acc': r['train_acc'],
                })
            except Exception as e:
                pass

    res_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    cols = ['Features', 'Model', 'n_feat', 'Acc', 'AUC', 'F1', 'Train_Acc']
    print(res_df[cols].head(15).to_string(index=False))
    return res_df


def experiment_3class_by_mode(df, mode, residualize=False,
                              time_series_lookup=None, use_combat=False,
                              use_optuna=False, n_repeats=None):
    """Run the 3-class experiment for one feature mode.

    Args:
        df: Merged feature DataFrame.
        mode: Active feature mode.
        residualize: Regress out confounds in-fold when True.
        time_series_lookup: id-to-array map for tangent features.
        use_combat: Apply ComBat harmonization when True.
        use_optuna: Tune hyperparameters with Optuna when True.
        n_repeats: CV repeats.

    Returns:
        Result DataFrame for this mode.
    """
    banner = {
        'imaging_only': "IMAGING-ONLY (leakage-siz, gercek bilimsel sonuc)",
        'imaging_plus_demographics':
            "IMAGING + YAS/CINSIYET/EGITIM (konfound kontrol, leakage yok)",
        'imaging_plus_clinical': "IMAGING + MMSE/CDR (DIKKAT: TARGET LEAKAGE, upper-bound)",
    }[mode]
    print(f"DENEY 1 - 3-SINIF [{mode}]{' (residualize)' if residualize else ''}")
    print(banner)

    feat_sets = get_feature_sets_by_mode(df, mode)
    y = df['label'].values
    results = []

    for fs_name, cols in feat_sets.items():
        if not cols:
            continue

        X, n_conf, n_bio, n_site = _prepare_feature_matrix(
            df, cols, residualize, mode, use_combat=use_combat)
        if X.shape[1] == 0:
            continue

        n_feat = X.shape[1]
        n_extra = n_conf + n_bio + n_site
        if use_optuna and HAS_OPTUNA:
            models = _build_models_optuna(
                X, y, task='3class',
                n_confounds=(n_conf if n_conf > 0 else None),
                n_site_cols=n_site, n_bio_cols=n_bio,
                n_trials=config.OPTUNA_N_TRIALS)
        else:
            models = _build_models(
                n_feat, task='3class',
                n_confounds=(n_conf if n_conf > 0 else None),
                n_site_cols=n_site, n_bio_cols=n_bio)
        reported_nfeat = n_feat - n_extra

        for m_name, model in models.items():
            try:
                r = cv_score(X, y, model, n_repeats=n_repeats)
                results.append({
                    'Task': '3class',
                    'Mode': mode,
                    'Residualized': bool(residualize and n_conf > 0),
                    'ComBat': bool(use_combat and n_site > 0),
                    'Optuna': bool(use_optuna and HAS_OPTUNA),
                    'Features': fs_name,
                    'Model': m_name,
                    'n_feat': reported_nfeat,
                    'Acc': r['acc'],
                    'Acc_std': r['acc_std'],
                    'Acc_CI_low': r.get('acc_ci_low', np.nan),
                    'Acc_CI_high': r.get('acc_ci_high', np.nan),
                    'AUC': r['auc'],
                    'AUC_std': r['auc_std'],
                    'AUC_CI_low': r.get('auc_ci_low', np.nan),
                    'AUC_CI_high': r.get('auc_ci_high', np.nan),
                    'F1': r['f1'],
                    'Train_Acc': r['train_acc'],
                })
            except Exception:
                pass

    if (time_series_lookup is not None and
        mode in ('imaging_only', 'imaging_plus_demographics')):
        tangent_rows = _run_tangent_experiment(
            df, y, time_series_lookup, mode, task='3class',
            n_repeats=n_repeats)
        results.extend(tangent_rows)
        try:
            nbs_rows = _run_tangent_experiment(
                df, y, time_series_lookup, mode, task='3class',
                n_repeats=n_repeats, use_nbs=True,
                feature_set_name='NBS_Edges')
            results.extend(nbs_rows)
        except Exception as e:
            print(f"    [WARN] NBS_Edges hesaplanamadi: {e}")

    res_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    if not res_df.empty:
        print(res_df[['Features', 'Model', 'n_feat', 'Acc', 'AUC', 'F1', 'Train_Acc']]
              .head(10).to_string(index=False))
    return res_df


def _run_tangent_experiment(df, y, time_series_lookup, mode,
                            task='3class', pair=None, n_repeats=None,
                            use_nbs=False, feature_set_name=None):
    """Run the tangent-space (and NBS) experiments for one task.

    Args:
        df: Feature DataFrame.
        y: Labels.
        time_series_lookup: id-to-array map.
        mode: Active feature mode.
        task: Task name.
        pair: Group pair for binary tasks (None for 3-class).
        n_repeats: CV repeats.
        use_nbs: Include the NBS variant when True.
        feature_set_name: Label for the produced rows.

    Returns:
        List of result rows.
    """
    rows = []
    subject_ids = df['subject_id'].astype(str).values
    mask = np.array([sid in time_series_lookup for sid in subject_ids])
    if mask.sum() < 10:
        return rows
    X_ids = subject_ids[mask]
    y_sel = y[mask]

    if task == '3class':
        scoring = ['accuracy', 'f1_macro', 'roc_auc_ovr_weighted']
        auc_key, f1_key = 'test_roc_auc_ovr_weighted', 'test_f1_macro'
    else:
        scoring = ['accuracy', 'f1', 'roc_auc']
        auc_key, f1_key = 'test_roc_auc', 'test_f1'

    candidates = {
        'Tangent+LogReg': LogisticRegression(C=1, max_iter=2000,
                                              random_state=42),
        'Tangent+ElasticNet': LogisticRegression(
            penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga',
            max_iter=5000, random_state=42),
        'Tangent+SVM_RBF': SVC(kernel='rbf', C=10, probability=True,
                                random_state=42),
    }
    if HAS_LGBM:
        candidates['Tangent+LightGBM'] = LGBMClassifier(
            n_estimators=200, num_leaves=15, learning_rate=0.05,
            random_state=42, verbose=-1)

    if n_repeats and n_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=n_repeats, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for m_name, clf in candidates.items():
        try:
            pipe = make_tangent_pipeline(
                clf, k_best=MAX_FEATURES_HARD_LIMIT,
                time_series_lookup=time_series_lookup,
                smote_k_neighbors=3,
                use_nbs=use_nbs,
            )
            res = cross_validate(pipe, X_ids, y_sel, cv=cv,
                                 scoring=scoring, return_train_score=True,
                                 error_score=np.nan)
            acc_scores = np.asarray(res['test_accuracy'], dtype=float)
            auc_scores = np.asarray(res[auc_key], dtype=float)
            rows.append({
                'Task': '3class' if task == '3class' else f"{pair[0]}-{pair[1]}",
                'Mode': mode,
                'Residualized': False,
                'ComBat': False,
                'Optuna': False,
                'Features': feature_set_name or ('NBS_Edges' if use_nbs else 'Tangent_FC'),
                'Model': m_name,
                'n_feat': MAX_FEATURES_HARD_LIMIT,
                'Acc': float(np.nanmean(acc_scores)),
                'Acc_std': float(np.nanstd(acc_scores)),
                'Acc_CI_low': float(np.nanpercentile(acc_scores, 2.5)),
                'Acc_CI_high': float(np.nanpercentile(acc_scores, 97.5)),
                'AUC': float(np.nanmean(auc_scores)),
                'AUC_std': float(np.nanstd(auc_scores)),
                'AUC_CI_low': float(np.nanpercentile(auc_scores, 2.5)),
                'AUC_CI_high': float(np.nanpercentile(auc_scores, 97.5)),
                'F1': float(np.nanmean(res[f1_key])),
                'Train_Acc': float(np.nanmean(res['train_accuracy'])),
            })
        except Exception as e:
            print(f"    [WARN] Tangent {m_name} hata: {e}")
    return rows


def experiment_binary_by_mode(df, mode, pair, residualize=False,
                              time_series_lookup=None, use_combat=False,
                              use_optuna=False, n_repeats=None):
    """Run a binary (pairwise) experiment for one feature mode.

    Args:
        df: Merged feature DataFrame.
        mode: Active feature mode.
        pair: Group pair, e.g. ('HC', 'AD').
        residualize: Regress out confounds in-fold when True.
        time_series_lookup: id-to-array map for tangent features.
        use_combat: Apply ComBat harmonization when True.
        use_optuna: Tune hyperparameters with Optuna when True.
        n_repeats: CV repeats.

    Returns:
        Result DataFrame for this pair and mode.
    """
    assert len(pair) == 2 and pair[0] != pair[1], "pair iki farkli sinif olmali"
    task_label = f"{pair[0]}-{pair[1]}"
    banner = {
        'imaging_only': "IMAGING-ONLY (leakage-siz)",
        'imaging_plus_demographics': "IMAGING + YAS/CINSIYET/EGITIM",
        'imaging_plus_clinical': "IMAGING + MMSE/CDR (LEAKAGE upper-bound)",
    }[mode]
    print(f"DENEY 2 - BINARY {task_label} [{mode}]"
          f"{' (residualize)' if residualize else ''}")
    print(banner)

    df2 = df[df['group'].isin(pair)].copy()
    if df2.empty or df2['group'].nunique() < 2:
        print(f"  [SKIP] pair {pair} icin yeterli veri yok")
        return pd.DataFrame()

    y = (df2['group'].values == pair[1]).astype(int)
    print(f"  {pair[0]}: {(y==0).sum()}, {pair[1]}: {(y==1).sum()}")

    feat_sets = get_feature_sets_by_mode(df2, mode)
    results = []

    for fs_name, cols in feat_sets.items():
        if not cols:
            continue
        X, n_conf, n_bio, n_site = _prepare_feature_matrix(
            df2, cols, residualize, mode, use_combat=use_combat)
        if X.shape[1] == 0:
            continue

        n_feat = X.shape[1]
        n_extra = n_conf + n_bio + n_site
        if use_optuna and HAS_OPTUNA:
            models = _build_models_optuna(
                X, y, task='binary',
                n_confounds=(n_conf if n_conf > 0 else None),
                n_site_cols=n_site, n_bio_cols=n_bio,
                n_trials=config.OPTUNA_N_TRIALS)
        else:
            models = _build_models(
                n_feat, task='binary',
                n_confounds=(n_conf if n_conf > 0 else None),
                n_site_cols=n_site, n_bio_cols=n_bio)
        reported_nfeat = n_feat - n_extra

        if n_repeats and n_repeats > 1:
            cv = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=n_repeats, random_state=42)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for m_name, model in models.items():
            try:
                r = cross_validate(
                    model, X, y, cv=cv,
                    scoring=['accuracy', 'f1', 'roc_auc'],
                    return_train_score=True, error_score=np.nan,
                )
                acc_scores = np.asarray(r['test_accuracy'], dtype=float)
                auc_scores = np.asarray(r['test_roc_auc'], dtype=float)
                results.append({
                    'Task': task_label,
                    'Mode': mode,
                    'Residualized': bool(residualize and n_conf > 0),
                    'ComBat': bool(use_combat and n_site > 0),
                    'Optuna': bool(use_optuna and HAS_OPTUNA),
                    'Features': fs_name,
                    'Model': m_name,
                    'n_feat': reported_nfeat,
                    'Acc': float(np.nanmean(acc_scores)),
                    'Acc_std': float(np.nanstd(acc_scores)),
                    'Acc_CI_low': float(np.nanpercentile(acc_scores, 2.5)),
                    'Acc_CI_high': float(np.nanpercentile(acc_scores, 97.5)),
                    'AUC': float(np.nanmean(auc_scores)),
                    'AUC_std': float(np.nanstd(auc_scores)),
                    'AUC_CI_low': float(np.nanpercentile(auc_scores, 2.5)),
                    'AUC_CI_high': float(np.nanpercentile(auc_scores, 97.5)),
                    'F1': float(np.nanmean(r['test_f1'])),
                    'Train_Acc': float(np.nanmean(r['train_accuracy'])),
                })
            except Exception:
                pass

    if (time_series_lookup is not None and
        mode in ('imaging_only', 'imaging_plus_demographics')):
        tangent_rows = _run_tangent_experiment(
            df2, y, time_series_lookup, mode, task='binary', pair=pair,
            n_repeats=n_repeats)
        results.extend(tangent_rows)
        try:
            nbs_rows = _run_tangent_experiment(
                df2, y, time_series_lookup, mode, task='binary', pair=pair,
                n_repeats=n_repeats, use_nbs=True,
                feature_set_name='NBS_Edges')
            results.extend(nbs_rows)
        except Exception as e:
            print(f"    [WARN] NBS_Edges (binary {pair}) hesaplanamadi: {e}")

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values('AUC', ascending=False)
        print(res_df[['Features', 'Model', 'n_feat', 'Acc', 'AUC',
                      'F1', 'Train_Acc']].head(10).to_string(index=False))
    return res_df


def experiment_binary_hcad(df):
    """Convenience wrapper for the HC-vs-AD binary experiment.

    Args:
        df: Merged feature DataFrame.

    Returns:
        Result DataFrame for HC vs AD.
    """
    return experiment_binary_by_mode(df, mode='imaging_only',
                                     pair=('HC', 'AD'), residualize=False)


def experiment_twostage(df):
    """Legacy two-stage (hierarchical) 3-class experiment.

    Args:
        df: Merged feature DataFrame.

    Returns:
        Result DataFrame.
    """
    print("DENEY 3: 2-ASAMALI (HC vs Hasta -> MCI vs AD)")

    feat_sets = get_feature_sets(df)
    img_sets = get_feature_sets_by_mode(df, 'imaging_only')
    cols = img_sets.get('Graf_AUC+Null', img_sets['Graf_AUC'])
    X = df[cols].values
    y = df['label'].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stage1_accs, stage2_accs, combined_accs = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        y_bin_tr = (y_tr > 0).astype(int)
        y_bin_te = (y_te > 0).astype(int)
        pipe1 = make_imb_pipeline(
            SVC(kernel='rbf', C=10, probability=True, random_state=42)
        )
        pipe1.fit(X_tr, y_bin_tr)
        pred_bin = pipe1.predict(X_te)
        stage1_accs.append(accuracy_score(y_bin_te, pred_bin))

        dis_tr = (y_tr > 0)
        dis_te = (y_te > 0) & (pred_bin == 1)
        final_pred = pred_bin.copy()

        if dis_tr.sum() > 3 and dis_te.sum() > 0:
            X_dis_tr = X_tr[dis_tr]
            y_dis_tr = y_tr[dis_tr]
            pipe2 = make_imb_pipeline(
                SVC(kernel='rbf', C=10, probability=True, random_state=42),
                smote_k_neighbors=2
            )
            pipe2.fit(X_dis_tr, y_dis_tr)
            pred_dis = pipe2.predict(X_te[dis_te])

            full_pred = np.zeros(len(y_te), dtype=int)
            full_pred[pred_bin == 0] = 0
            diseased_indices = np.where(pred_bin == 1)[0]
            for i, idx in enumerate(diseased_indices):
                full_pred[idx] = pred_dis[i] if i < len(pred_dis) else 1
            stage2_accs.append(accuracy_score(y_te[dis_te], pred_dis))
            combined_accs.append(accuracy_score(y_te, full_pred))
        else:
            stage2_accs.append(np.nan)
            combined_accs.append(np.nan)

    print(f"  Stage 1 (HC vs Sick): Acc = {np.mean(stage1_accs):.3f} +/- {np.std(stage1_accs):.3f}")
    print(f"  Stage 2 (MCI vs AD):  Acc = {np.nanmean(stage2_accs):.3f} "
          f"+/- {np.nanstd(stage2_accs):.3f}")
    print(f"  Combined 3-class:     Acc = {np.nanmean(combined_accs):.3f} "
          f"+/- {np.nanstd(combined_accs):.3f}")

    return {
        'stage1': np.mean(stage1_accs),
        'stage2': np.nanmean(stage2_accs),
        'combined': np.nanmean(combined_accs)
    }


def experiment_twostage_by_mode(df, mode, residualize=False, use_combat=False,
                                n_splits=5, n_repeats=10, random_state=42,
                                calibrate=True):
    """Two-stage HC-vs-impaired then MCI-vs-AD experiment.

    Args:
        df: Merged feature DataFrame.
        mode: Active feature mode.
        residualize: Regress out confounds in-fold when True.
        use_combat: Apply ComBat harmonization when True.
        n_splits: Folds per repeat.
        n_repeats: CV repeats.
        random_state: Seed for reproducibility.
        calibrate: Apply probability calibration when True.

    Returns:
        Result DataFrame for the two-stage classifier.
    """
    feat_map = get_feature_sets_by_mode(df, mode)
    y = df['label'].values
    groups = df['group'].values
    rows = []
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for feat_name, cols in feat_map.items():
        if not cols:
            continue
        X, n_conf, n_bio, n_site = _prepare_feature_matrix(
            df, cols, residualize=residualize, mode=mode, use_combat=use_combat)

        fold_auc, fold_acc, fold_f1 = [], [], []
        y_pred_all = np.full(len(y), -1, dtype=int)

        for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]

            y1_tr = (y_tr > 0).astype(int)
            pipe1 = make_imb_pipeline(
                LGBMClassifier(n_estimators=200, num_leaves=15,
                               learning_rate=0.05, random_state=42, verbose=-1)
                if HAS_LGBM else
                SVC(kernel='rbf', C=10, probability=True, random_state=42),
                k_best=(MAX_FEATURES_HARD_LIMIT if X.shape[1] > MAX_FEATURES_HARD_LIMIT else None),
                n_confounds=n_conf, n_site_cols=n_site, n_bio_cols=n_bio,
            )
            try:
                if calibrate:
                    from sklearn.calibration import CalibratedClassifierCV
                    pipe1 = CalibratedClassifierCV(pipe1, method='isotonic', cv=3)
                pipe1.fit(X_tr, y1_tr)
                p1_te = pipe1.predict(X_te)
                prob1_te = pipe1.predict_proba(X_te)[:, 1]
            except Exception:
                continue

            mask_dis_tr = y_tr > 0
            if mask_dis_tr.sum() < 6:
                continue
            X_dis_tr = X_tr[mask_dis_tr]
            y_dis_tr = y_tr[mask_dis_tr]
            try:
                pipe2 = make_imb_pipeline(
                    LGBMClassifier(n_estimators=200, num_leaves=15,
                                   learning_rate=0.05, random_state=42, verbose=-1)
                    if HAS_LGBM else
                    SVC(kernel='rbf', C=10, probability=True, random_state=42),
                    k_best=(MAX_FEATURES_HARD_LIMIT
                            if X.shape[1] > MAX_FEATURES_HARD_LIMIT else None),
                    smote_k_neighbors=2,
                    n_confounds=n_conf, n_site_cols=n_site, n_bio_cols=n_bio,
                )
                if calibrate:
                    from sklearn.calibration import CalibratedClassifierCV
                    pipe2 = CalibratedClassifierCV(pipe2, method='isotonic', cv=3)
                pipe2.fit(X_dis_tr, y_dis_tr)
                prob2_te = pipe2.predict_proba(X_te)
            except Exception:
                continue

            proba3 = np.zeros((len(y_te), 3))
            proba3[:, 0] = 1.0 - prob1_te
            proba3[:, 1] = prob1_te * prob2_te[:, 0]
            proba3[:, 2] = prob1_te * prob2_te[:, 1]
            y_hat = np.argmax(proba3, axis=1)
            y_pred_all[te] = y_hat

            try:
                from sklearn.metrics import roc_auc_score, f1_score
                auc_f = roc_auc_score(
                    y_te, proba3, multi_class='ovr', average='weighted',
                    labels=[0, 1, 2])
            except Exception:
                auc_f = np.nan
            fold_auc.append(auc_f)
            fold_acc.append(accuracy_score(y_te, y_hat))
            fold_f1.append(f1_score(y_te, y_hat, average='macro'))

        if not fold_auc:
            continue
        auc_arr = np.asarray(fold_auc, dtype=float)
        acc_arr = np.asarray(fold_acc, dtype=float)
        f1_arr = np.asarray(fold_f1, dtype=float)
        rows.append({
            'Task': '3class_twostage',
            'Mode': mode,
            'Features': feat_name,
            'Model': 'TwoStage_LGBM' if HAS_LGBM else 'TwoStage_SVM',
            'n_feat': X.shape[1],
            'Acc': float(np.nanmean(acc_arr)),
            'AUC': float(np.nanmean(auc_arr)),
            'F1': float(np.nanmean(f1_arr)),
            'AUC_CI_low': float(np.nanpercentile(auc_arr, 2.5)),
            'AUC_CI_high': float(np.nanpercentile(auc_arr, 97.5)),
            'Acc_CI_low': float(np.nanpercentile(acc_arr, 2.5)),
            'Acc_CI_high': float(np.nanpercentile(acc_arr, 97.5)),
            'Train_Acc': np.nan,
            'Residualized': residualize,
        })

    return pd.DataFrame(rows).sort_values('AUC', ascending=False) if rows else pd.DataFrame()


def experiment_gridsearch(df):
    """Small grid-search experiment kept for reference.

    Args:
        df: Merged feature DataFrame.

    Returns:
        Result DataFrame.
    """
    print("DENEY 4: GRID SEARCH (SVM + RF en iyi parametreler)")

    img_sets = get_feature_sets_by_mode(df, 'imaging_only')
    cols = img_sets.get('Graf_AUC+Null', img_sets['Graf_AUC'])
    X = df[cols].values
    y = df['label'].values

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    pipe = make_imb_pipeline(
        SVC(kernel='rbf', probability=True, random_state=42)
    )

    svm_param = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__gamma': ['scale', 'auto', 0.01, 0.1],
    }
    svm_gs = GridSearchCV(
        pipe, svm_param, cv=cv_inner,
        scoring='roc_auc_ovr_weighted', n_jobs=-1
    )

    accs, aucs = [], []
    for train_idx, test_idx in cv_outer.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        svm_gs.fit(X_tr, y_tr)

        pred = svm_gs.predict(X_te)
        prob = svm_gs.predict_proba(X_te)
        accs.append(accuracy_score(y_te, pred))
        try:
            aucs.append(roc_auc_score(y_te, prob, multi_class='ovr', average='weighted'))
        except Exception:
            aucs.append(np.nan)

    print(f"  Nested CV SVM (GridSearch, CV-safe): "
          f"Acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}, "
          f"AUC={np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}")
    print(f"  Son fold'daki en iyi parametreler: {svm_gs.best_params_}")

    return {'acc': np.mean(accs), 'auc': np.nanmean(aucs)}
