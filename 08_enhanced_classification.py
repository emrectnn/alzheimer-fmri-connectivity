"""Classification sweep.

imbalanced-learn pipeline combining ComBat harmonization, confound regression,
feature selection, SMOTE, repeated stratified cross-validation, and Optuna tuning
across tasks (3-class and binary) and feature modes (imaging / +demographics / +clinical).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    ImbPipeline = Pipeline
    SMOTE = None
    print("[!] imbalanced-learn yuklu degil. SMOTE atlanacak. "
          "pip install imbalanced-learn ile kurun.")

from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, cross_validate,
    cross_val_score, GridSearchCV,
)

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] XGBoost yuklu degil. pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[!] LightGBM yuklu degil. pip install lightgbm")

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    HAS_NH = True
except ImportError:
    HAS_NH = False
    harmonizationLearn = None
    harmonizationApply = None
    print("[!] neuroHarmonize yuklu degil. ComBat atlanacak.")

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None
    TPESampler = None
    print("[!] Optuna yuklu degil. pip install optuna")

try:
    import mord
    HAS_MORD = True
except ImportError:
    HAS_MORD = False
    mord = None
    print("[!] mord yuklu degil. Ordinal regression atlanacak. pip install mord")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")

MAX_FEATURES_HARD_LIMIT = 30


class ConfoundRegressor(BaseEstimator, TransformerMixin):

    """Regress confounds (age, sex, motion) out of features within each CV fold."""
    def __init__(self, n_confounds=0):
        self.n_confounds = int(n_confounds) if n_confounds else 0

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.n_confounds <= 0:
            self.coef_ = None
            self.intercept_ = None
            return self
        p_total = X.shape[1]
        n_conf = self.n_confounds
        if n_conf >= p_total:
            raise ValueError(
                f"ConfoundRegressor: n_confounds={n_conf} >= n_features={p_total}"
            )
        p_feat = p_total - n_conf
        C = X[:, p_feat:]
        F = X[:, :p_feat]
        lr = LinearRegression()
        lr.fit(C, F)
        self.coef_ = lr.coef_
        self.intercept_ = lr.intercept_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.n_confounds <= 0 or self.coef_ is None:
            return X
        p_total = X.shape[1]
        n_conf = self.n_confounds
        p_feat = p_total - n_conf
        C = X[:, p_feat:]
        F = X[:, :p_feat]
        F_resid = F - (C @ self.coef_.T + self.intercept_)
        return F_resid


class ComBatTransformer(BaseEstimator, TransformerMixin):

    """CV-safe ComBat site/protocol harmonization transformer."""
    def __init__(self, n_site_cols=1, n_bio_cols=0):
        self.n_site_cols = int(n_site_cols) if n_site_cols else 0
        self.n_bio_cols = int(n_bio_cols) if n_bio_cols else 0

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        n_total = X.shape[1]
        n_site = self.n_site_cols
        n_bio = self.n_bio_cols
        n_feat = n_total - n_site - n_bio
        if n_feat <= 0:
            raise ValueError(
                f"ComBatTransformer: site/bio kolonlari ({n_site}+{n_bio}) "
                f"toplam feature sayisindan ({n_total}) fazla/esit."
            )
        F = X[:, :n_feat]
        B = X[:, n_feat:n_feat + n_bio] if n_bio > 0 else None
        S = X[:, n_feat + n_bio:] if n_site > 0 else None
        return F, B, S

    def _build_covars(self, B, S, n_rows):
        covars = pd.DataFrame(index=np.arange(n_rows))
        if S is not None:
            covars['SITE'] = S[:, 0].astype(int).astype(str)
        if B is not None:
            for j in range(B.shape[1]):
                covars[f'bio_{j}'] = B[:, j]
        return covars

    def fit(self, X, y=None):
        if not HAS_NH or self.n_site_cols <= 0:
            self.model_ = None
            return self
        F, B, S = self._split(X)
        covars = self._build_covars(B, S, F.shape[0])
        if covars['SITE'].nunique() < 2:
            self.model_ = None
            return self
        site_counts = covars['SITE'].value_counts()
        if (site_counts < 2).any():
            self.model_ = None
            return self
        try:
            self.model_, _adj = harmonizationLearn(F, covars)
        except Exception as exc:
            print(f"  [WARN] ComBat fit basarisiz ({exc}); pass-through.")
            self.model_ = None
        return self

    def transform(self, X):
        if not HAS_NH or self.n_site_cols <= 0:
            F, _B, _S = self._split(X) if (self.n_site_cols or self.n_bio_cols) else (np.asarray(X, dtype=float), None, None)
            return F
        F, B, S = self._split(X)
        if self.model_ is None:
            return F
        covars = self._build_covars(B, S, F.shape[0])
        try:
            F_adj = harmonizationApply(F, covars, self.model_)
        except Exception as exc:
            print(f"  [WARN] ComBat apply basarisiz ({exc}); raw feature dondurulur.")
            return F
        return F_adj


def load_all_features():
    """Load and merge all feature tables into a single DataFrame."""
    gm = pd.read_csv(os.path.join(config.METRICS_DIR, "global_metrics.csv"))

    null = pd.read_csv(os.path.join(config.METRICS_DIR, "null_model_erdos_renyi.csv"))

    meta = pd.read_csv(os.path.join(config.NIFTI_DIR, "subject_metadata.csv"))

    df = gm.merge(null[['subject_id','dev_clustering','dev_path_length',
                          'dev_global_eff','dev_sigma']], on='subject_id', how='left')
    df = df.merge(meta[['subject_id','mmse','cdrsb','cdglobal','age',
                          'gender','education']], on='subject_id', how='left')

    df['gender_bin'] = (df['gender'] == 'Male').astype(float)

    motion_qc_path = os.path.join(config.METRICS_DIR, "motion_qc.csv")
    if os.path.exists(motion_qc_path):
        try:
            mq = pd.read_csv(motion_qc_path)
            if 'subject_id' in mq.columns and 'mean_fd' in mq.columns:
                df = df.merge(mq[['subject_id', 'mean_fd']],
                              on='subject_id', how='left')
        except Exception as e:
            print(f"[WARN] motion_qc.csv okunamadi: {e}")

    for kind in ('alff', 'reho'):
        cand = os.path.join(config.RESULTS_DIR, kind, f"{kind}_AAL3.csv")
        if os.path.exists(cand):
            try:
                d = pd.read_csv(cand)
                d = d.rename(columns={c: f"{kind}_{c}" for c in d.columns
                                      if c != 'subject_id'})
                df = df.merge(d, on='subject_id', how='left')
            except Exception as e:
                print(f"[WARN] {cand} okunamadi: {e}")

    for atlas_key, prefix in (('Schaefer200', 'scha200_'), ('HO48', 'ho48_')):
        cand = os.path.join(config.METRICS_DIR, f"global_{atlas_key}.csv")
        if os.path.exists(cand):
            try:
                d = pd.read_csv(cand, dtype={'subject_id': str})
                if 'group' in d.columns:
                    d = d.drop(columns=['group'])
                d = d.rename(columns={c: f"{prefix}{c}" for c in d.columns
                                      if c != 'subject_id'})
                df = df.merge(d, on='subject_id', how='left')
            except Exception as e:
                print(f"[WARN] {cand} okunamadi: {e}")


    return df


def get_feature_sets(df):
    graph_cols = [c for c in df.columns if c not in
                  ['subject_id','label','group','n_nodes','n_nodes_lc',
                   'n_edges','density','C_rand','L_rand','auc_n_nodes',
                   'auc_n_nodes_lc','auc_n_edges','auc_density',
                   'auc_C_rand','auc_L_rand','gender','age','education',
                   'mmse','cdrsb','cdglobal','gender_bin',
                   'dev_clustering','dev_path_length','dev_global_eff','dev_sigma']]

    auc_cols = [c for c in graph_cols if c.startswith('auc_')]

    clinical_cols = ['mmse','cdrsb','cdglobal','age','gender_bin','education']

    null_cols = ['dev_clustering','dev_path_length','dev_global_eff','dev_sigma']

    feature_sets = {
        'Graf_AUC': auc_cols,
        'Graf_Tam': graph_cols,
        'Klinik': clinical_cols,
        'Null_Model': null_cols,
        'Graf_AUC+Klinik': auc_cols + clinical_cols,
        'Graf_Tam+Klinik': graph_cols + clinical_cols,
        'Graf_AUC+Null': auc_cols + null_cols,
        'Hepsi': graph_cols + clinical_cols + null_cols,
    }

    for name, cols in feature_sets.items():
        feature_sets[name] = [c for c in cols if c in df.columns]

    return feature_sets


def get_feature_sets_by_mode(df, mode):
    """Return feature-set definitions for a given mode (imaging / +demographics / +clinical)."""
    base = get_feature_sets(df)
    graph_only = base['Graf_Tam']
    auc_only = base['Graf_AUC']
    null_only = base['Null_Model']
    demographics = [c for c in ['age', 'gender_bin', 'education'] if c in df.columns]
    clinical_only = [c for c in ['mmse', 'cdrsb', 'cdglobal'] if c in df.columns]

    alff_cols = [c for c in df.columns if c.startswith('alff_roi_')]
    reho_cols = [c for c in df.columns if c.startswith('reho_roi_')]
    scha200_cols = [c for c in df.columns if c.startswith('scha200_')]
    ho48_cols = [c for c in df.columns if c.startswith('ho48_')]

    if mode == 'imaging_only':
        fs = {
            'Graf_AUC': auc_only,
            'Graf_Tam': graph_only,
            'Null_Model': null_only,
            'Graf_AUC+Null': auc_only + null_only,
            'Graf_Tam+Null': graph_only + null_only,
        }
        if alff_cols:
            fs['ALFF'] = alff_cols
        if reho_cols:
            fs['ReHo'] = reho_cols
        if alff_cols and reho_cols:
            fs['ALFF+ReHo'] = alff_cols + reho_cols
            fs['Graf_Tam+ALFF+ReHo'] = graph_only + alff_cols + reho_cols
        if scha200_cols:
            fs['Graf_Schaefer200'] = scha200_cols
        if ho48_cols:
            fs['Graf_HO48'] = ho48_cols
        if scha200_cols and ho48_cols:
            fs['Graf_MultiAtlas'] = graph_only + scha200_cols + ho48_cols
        return fs
    elif mode == 'imaging_plus_demographics':
        fs = {
            'Graf_AUC+Demog': auc_only + demographics,
            'Graf_Tam+Demog': graph_only + demographics,
            'Graf_AUC+Null+Demog': auc_only + null_only + demographics,
            'Graf_Tam+Null+Demog': graph_only + null_only + demographics,
        }
        if scha200_cols and ho48_cols:
            fs['Graf_MultiAtlas+Demog'] = (graph_only + scha200_cols
                                           + ho48_cols + demographics)
        if alff_cols and reho_cols:
            fs['Graf_Tam+ALFF+ReHo+Demog'] = (graph_only + alff_cols
                                              + reho_cols + demographics)
        return fs
    elif mode == 'imaging_plus_clinical':
        full_clinical = clinical_only + demographics
        return {
            'Graf_AUC+Klinik': auc_only + full_clinical,
            'Graf_Tam+Klinik': graph_only + full_clinical,
            'Graf_AUC+Null+Klinik': auc_only + null_only + full_clinical,
            'Hepsi': graph_only + null_only + full_clinical,
        }
    else:
        raise ValueError(
            f"Bilinmeyen mode: {mode!r}. Beklenen: 'imaging_only', "
            "'imaging_plus_demographics', 'imaging_plus_clinical'."
        )


def cv_score(X, y, model, n_splits=5, random_state=42, n_repeats=None):
    if n_repeats and n_repeats > 1:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
    results = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'f1_macro', 'roc_auc_ovr_weighted'],
        return_train_score=True
    )
    acc_scores = np.asarray(results['test_accuracy'], dtype=float)
    auc_scores = np.asarray(results['test_roc_auc_ovr_weighted'], dtype=float)

    per_class = {}
    try:
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import classification_report, confusion_matrix as _cm
        pc_cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state)
        y_pred = cross_val_predict(model, X, y, cv=pc_cv, n_jobs=1)
        report = classification_report(
            y, y_pred, output_dict=True, zero_division=0)
        classes = sorted(np.unique(y).tolist())
        for cls in classes:
            key = str(cls)
            if key in report:
                per_class[f'precision_cls{cls}'] = float(report[key]['precision'])
                per_class[f'recall_cls{cls}'] = float(report[key]['recall'])
                per_class[f'f1_cls{cls}'] = float(report[key]['f1-score'])
        cm = _cm(y, y_pred, labels=classes)
        per_class['confusion_matrix'] = cm.tolist()
    except Exception as _e:
        per_class['per_class_error'] = str(_e)

    out = {
        'acc':  float(np.mean(acc_scores)),
        'acc_std': float(np.std(acc_scores)),
        'acc_ci_low':  float(np.nanpercentile(acc_scores, 2.5)),
        'acc_ci_high': float(np.nanpercentile(acc_scores, 97.5)),
        'f1':   float(np.mean(results['test_f1_macro'])),
        'auc':  float(np.mean(auc_scores)),
        'auc_std': float(np.std(auc_scores)),
        'auc_ci_low':  float(np.nanpercentile(auc_scores, 2.5)),
        'auc_ci_high': float(np.nanpercentile(auc_scores, 97.5)),
        'train_acc': float(np.mean(results['train_accuracy'])),
    }
    out.update(per_class)
    return out


def make_pipeline(clf, k_best=None, n_pca=None):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    steps.append(('scaler', StandardScaler()))
    if k_best:
        steps.append(('select', SelectKBest(f_classif, k=k_best)))
    if n_pca:
        steps.append(('pca', PCA(n_components=n_pca, random_state=42)))
    steps.append(('clf', clf))
    return Pipeline(steps)


def make_imb_pipeline(clf, k_best=None, n_pca=None, use_smote=True,
                     smote_k_neighbors=3, n_confounds=None,
                     n_site_cols=0, n_bio_cols=0):
    """Build the imbalanced-learn pipeline (impute, scale, select, SMOTE, classifier)."""
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    if n_site_cols and n_site_cols > 0:
        steps.append(('combat', ComBatTransformer(
            n_site_cols=n_site_cols, n_bio_cols=n_bio_cols)))
    steps.append(('scaler', StandardScaler()))
    if n_confounds and n_confounds > 0:
        steps.append(('confound', ConfoundRegressor(n_confounds=n_confounds)))
    if k_best:
        steps.append(('select', SelectKBest(f_classif, k=k_best)))
    if n_pca:
        steps.append(('pca', PCA(n_components=n_pca, random_state=42)))
    if use_smote and HAS_IMBLEARN:
        steps.append(('smote', SMOTE(k_neighbors=smote_k_neighbors, random_state=42)))
    steps.append(('clf', clf))
    return ImbPipeline(steps) if HAS_IMBLEARN else Pipeline(
        [(n, s) for n, s in steps if n != 'smote']
    )


def load_timeseries_lookup(subject_ids, data_dir=None, min_rois=None):
    if data_dir is None:
        data_dir = config.PREPROCESSED_DIR

    raw = {}
    for sid in subject_ids:
        path = os.path.join(data_dir, f"{sid}_timeseries.npy")
        if not os.path.exists(path):
            continue
        ts = np.load(path)
        if ts.ndim != 2 or ts.shape[0] < 10:
            continue
        if min_rois is not None and ts.shape[1] < min_rois:
            continue
        raw[sid] = ts

    if not raw:
        return {}

    common_n = min(ts.shape[1] for ts in raw.values())
    lookup = {sid: ts[:, :common_n] for sid, ts in raw.items()}
    return lookup


def make_tangent_pipeline(clf, k_best=MAX_FEATURES_HARD_LIMIT,
                          n_confounds=None, time_series_lookup=None,
                          use_smote=True, smote_k_neighbors=3,
                          use_nbs=False, nbs_n_roi=None,
                          nbs_thresh=None, nbs_n_perm=None,
                          nbs_alpha=None):
    connectivity_mod = import_module("02_connectivity")
    tangent_cls = connectivity_mod.TangentSpaceTransformer

    steps = [
        ('tangent', tangent_cls(time_series_lookup=time_series_lookup)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    if n_confounds and n_confounds > 0:
        steps.append(('confound', ConfoundRegressor(n_confounds=n_confounds)))
    if use_nbs:
        nbs_mod = import_module('06b_nbs')
        inferred_n = nbs_n_roi
        if inferred_n is None and time_series_lookup:
            first_ts = next(iter(time_series_lookup.values()))
            inferred_n = first_ts.shape[1]
        if inferred_n is None:
            inferred_n = 166
        steps.append(('nbs', nbs_mod.NBSEdgeSelector(
            n_roi=inferred_n,
            thresh=nbs_thresh if nbs_thresh is not None else config.NBS_THRESH,
            n_perm=nbs_n_perm if nbs_n_perm is not None else config.NBS_N_PERM,
            alpha=nbs_alpha if nbs_alpha is not None else config.NBS_ALPHA,
            random_state=42,
            fallback_topk=k_best or 30,
        )))
    if k_best:
        steps.append(('select', SelectKBest(f_classif, k=k_best)))
    if use_smote and HAS_IMBLEARN:
        steps.append(('smote', SMOTE(k_neighbors=smote_k_neighbors,
                                     random_state=42)))
    steps.append(('clf', clf))
    return ImbPipeline(steps) if HAS_IMBLEARN else Pipeline(
        [(n, s) for n, s in steps if n != 'smote']
    )


def _build_models(n_feat, task='3class', n_confounds=None,
                  n_site_cols=0, n_bio_cols=0):
    auto_k = MAX_FEATURES_HARD_LIMIT if n_feat > MAX_FEATURES_HARD_LIMIT else None
    kw = {
        'k_best': auto_k,
        'n_confounds': n_confounds,
        'n_site_cols': n_site_cols,
        'n_bio_cols': n_bio_cols,
    }

    models = {
        'SVM_RBF_C10': make_imb_pipeline(
            SVC(kernel='rbf', C=10, probability=True, random_state=42), **kw),
        'RF': make_imb_pipeline(
            RandomForestClassifier(200, random_state=42), **kw),
        'GBM': make_imb_pipeline(
            GradientBoostingClassifier(n_estimators=100, random_state=42), **kw),
        'LogReg': make_imb_pipeline(
            LogisticRegression(C=1, max_iter=2000, random_state=42), **kw),
        'ElasticNet': make_imb_pipeline(
            LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0,
                               solver='saga', max_iter=5000, random_state=42),
            **kw),
        'Top15+SVM': make_imb_pipeline(
            SVC(kernel='rbf', C=10, probability=True, random_state=42),
            k_best=min(15, n_feat), n_confounds=n_confounds,
            n_site_cols=n_site_cols, n_bio_cols=n_bio_cols),
    }

    if HAS_XGB:
        eval_metric = 'mlogloss' if task == '3class' else 'logloss'
        max_depth = 4 if task == '3class' else 3
        models['XGBoost'] = make_imb_pipeline(XGBClassifier(
            n_estimators=100, max_depth=max_depth, learning_rate=0.1,
            random_state=42, eval_metric=eval_metric, verbosity=0), **kw)

    if HAS_LGBM:
        models['LightGBM'] = make_imb_pipeline(LGBMClassifier(
            n_estimators=200, num_leaves=15, learning_rate=0.05,
            random_state=42, verbose=-1), **kw)

    if HAS_LGBM and HAS_XGB:
        base_estimators = [
            ('en', LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
                                       C=1.0, solver='saga', max_iter=5000,
                                       random_state=42)),
            ('svc', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            ('lgbm', LGBMClassifier(n_estimators=200, num_leaves=15,
                                     learning_rate=0.05, random_state=42,
                                     verbose=-1)),
        ]
        stacker = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=42),
            cv=5, passthrough=False, n_jobs=1,
        )
        models['Stacking'] = make_imb_pipeline(stacker, **kw)

    if HAS_MORD and task == '3class':
        try:
            models['OrdinalAT'] = make_imb_pipeline(
                mord.LogisticAT(alpha=1.0), **kw)
            models['OrdinalIT'] = make_imb_pipeline(
                mord.LogisticIT(alpha=1.0), **kw)
        except Exception as e:
            print(f"[WARN] mord model olusturulamadi: {e}")

    return models


def _make_clf_from_params(model_name, params, task):
    if model_name == 'LightGBM' and HAS_LGBM:
        return LGBMClassifier(random_state=42, verbose=-1, **params)
    if model_name == 'XGBoost' and HAS_XGB:
        eval_metric = 'mlogloss' if task == '3class' else 'logloss'
        return XGBClassifier(random_state=42, eval_metric=eval_metric,
                             verbosity=0, **params)
    if model_name == 'ElasticNet':
        return LogisticRegression(
            penalty='elasticnet', solver='saga', max_iter=5000,
            random_state=42, **params)
    if model_name == 'SVM_RBF':
        return SVC(kernel='rbf', probability=True, random_state=42, **params)
    if model_name == 'Stacking' and HAS_LGBM:
        base = [
            ('en', LogisticRegression(
                penalty='elasticnet', l1_ratio=params.get('en_l1', 0.5),
                C=params.get('en_C', 1.0), solver='saga',
                max_iter=5000, random_state=42)),
            ('svc', SVC(kernel='rbf', C=params.get('svc_C', 10),
                        probability=True, random_state=42)),
            ('lgbm', LGBMClassifier(
                n_estimators=params.get('lgbm_n', 200),
                num_leaves=params.get('lgbm_leaves', 15),
                learning_rate=params.get('lgbm_lr', 0.05),
                random_state=42, verbose=-1)),
        ]
        return StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(
                C=params.get('final_C', 1.0), max_iter=2000,
                random_state=42),
            cv=5, passthrough=False, n_jobs=1)
    raise ValueError(f"Bilinmeyen / desteklenmeyen model: {model_name}")


def _optuna_suggest(trial, model_name):
    if model_name == 'LightGBM':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 7, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
    if model_name == 'XGBoost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }
    if model_name == 'ElasticNet':
        return {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
        }
    if model_name == 'SVM_RBF':
        return {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        }
    if model_name == 'Stacking':
        return {
            'lgbm_n': trial.suggest_int('lgbm_n', 100, 300),
            'lgbm_leaves': trial.suggest_int('lgbm_leaves', 7, 31),
            'lgbm_lr': trial.suggest_float('lgbm_lr', 0.02, 0.2, log=True),
            'svc_C': trial.suggest_float('svc_C', 1.0, 30.0, log=True),
            'en_C': trial.suggest_float('en_C', 0.1, 5.0, log=True),
            'en_l1': trial.suggest_float('en_l1', 0.2, 0.8),
            'final_C': trial.suggest_float('final_C', 0.1, 5.0, log=True),
        }
    return {}


def _build_models_optuna(X, y, task='3class', n_confounds=None,
                         n_site_cols=0, n_bio_cols=0, n_trials=50):
    if not HAS_OPTUNA:
        return _build_models(X.shape[1], task=task, n_confounds=n_confounds,
                             n_site_cols=n_site_cols, n_bio_cols=n_bio_cols)

    fixed = _build_models(X.shape[1], task=task, n_confounds=n_confounds,
                          n_site_cols=n_site_cols, n_bio_cols=n_bio_cols)
    top_set = set(config.OPTUNA_TOP_MODELS)
    tuned = {name: pipe for name, pipe in fixed.items() if name not in top_set}

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = 'roc_auc_ovr_weighted' if task == '3class' else 'roc_auc'

    auto_k = MAX_FEATURES_HARD_LIMIT if X.shape[1] > MAX_FEATURES_HARD_LIMIT else None

    for model_name in config.OPTUNA_TOP_MODELS:
        if model_name == 'LightGBM' and not HAS_LGBM:
            continue
        if model_name == 'XGBoost' and not HAS_XGB:
            continue
        if model_name == 'Stacking' and not HAS_LGBM:
            continue

        def objective(trial, _name=model_name):
            params = _optuna_suggest(trial, _name)
            try:
                clf = _make_clf_from_params(_name, params, task)
                pipe = make_imb_pipeline(
                    clf, k_best=auto_k, n_confounds=n_confounds,
                    n_site_cols=n_site_cols, n_bio_cols=n_bio_cols,
                )
                scores = cross_val_score(
                    pipe, X, y, cv=inner_cv, scoring=scoring,
                    error_score='raise')
                return float(np.mean(scores))
            except Exception as exc:
                raise optuna.exceptions.TrialPruned(str(exc))

        study = optuna.create_study(
            direction='maximize', sampler=TPESampler(seed=42))
        try:
            study.optimize(objective, n_trials=n_trials,
                           show_progress_bar=False, n_jobs=1,
                           catch=(Exception,))
        except Exception as exc:
            print(f"    [WARN] Optuna {model_name} hata: {exc}")
            continue

        if not study.best_trial or study.best_value is None:
            continue
        best_params = study.best_trial.params
        try:
            clf_best = _make_clf_from_params(model_name, best_params, task)
            tuned[model_name] = make_imb_pipeline(
                clf_best, k_best=auto_k, n_confounds=n_confounds,
                n_site_cols=n_site_cols, n_bio_cols=n_bio_cols,
            )
            print(f"    [Optuna {model_name}] best_auc={study.best_value:.4f} "
                  f"best_params={best_params}")
        except Exception as exc:
            print(f"    [WARN] Optuna {model_name} best rebuild hata: {exc}")

    return tuned


def experiment_3class(df):
    print("DENEY 1: 3-SINIF (HC vs MCI vs AD)")

    feat_sets = get_feature_sets(df)
    y = df['label'].values

    results = []

    for fs_name, cols in feat_sets.items():
        X = df[cols].values
        if X.shape[1] == 0:
            continue

        models = {
            'SVM_RBF':   make_imb_pipeline(SVC(kernel='rbf', C=1, probability=True, random_state=42)),
            'SVM_RBF_C10': make_imb_pipeline(SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            'RF':        make_imb_pipeline(RandomForestClassifier(200, random_state=42)),
            'GBM':       make_imb_pipeline(GradientBoostingClassifier(n_estimators=100, random_state=42)),
            'LogReg':    make_imb_pipeline(LogisticRegression(C=1, max_iter=2000, random_state=42)),
            'ElasticNet': make_imb_pipeline(LogisticRegression(
                penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga',
                max_iter=5000, random_state=42)),
            'PCA20+SVM': make_imb_pipeline(SVC(kernel='rbf', C=10, probability=True, random_state=42), n_pca=min(20, X.shape[1]-1)),
            'Top15+SVM': make_imb_pipeline(SVC(kernel='rbf', C=10, probability=True, random_state=42), k_best=min(15, X.shape[1])),
            'Top15+RF':  make_imb_pipeline(RandomForestClassifier(200, random_state=42), k_best=min(15, X.shape[1])),
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
    print(res_df[['Features','Model','n_feat','Acc','AUC','F1','Train_Acc']].head(15).to_string(index=False))
    return res_df


def _prepare_feature_matrix(df, cols, residualize, mode, use_combat=False):
    X_feat = df[cols].values.astype(float)
    n_conf = 0
    n_bio = 0
    n_site = 0
    parts = [X_feat]

    if residualize and mode != 'imaging_plus_clinical':
        conf_cols = [c for c in ('age', 'gender_bin', 'mean_fd')
                     if c in df.columns and c not in cols]
        if conf_cols:
            parts.append(df[conf_cols].values.astype(float))
            n_conf = len(conf_cols)

    if use_combat and HAS_NH and mode != 'imaging_plus_clinical':
        if 'protocol' in df.columns:
            site_vals = (df['protocol'].astype(str).str.strip() == 'MB').astype(int).values
            if len(np.unique(site_vals)) >= 2:
                bio_cols = []
                if not residualize:
                    for c in ('age', 'gender_bin'):
                        if c in df.columns and c not in cols:
                            bio_cols.append(c)
                if bio_cols:
                    parts.append(df[bio_cols].values.astype(float))
                    n_bio = len(bio_cols)
                parts.append(site_vals.reshape(-1, 1).astype(float))
                n_site = 1

    if len(parts) == 1:
        return X_feat, 0, 0, 0
    X_stacked = np.concatenate(parts, axis=1)
    return X_stacked, n_conf, n_bio, n_site


def experiment_3class_by_mode(df, mode, residualize=False,
                              time_series_lookup=None, use_combat=False,
                              use_optuna=False, n_repeats=None):
    """Run the 3-class experiment for one feature mode."""
    banner = {
        'imaging_only': "IMAGING-ONLY (leakage-siz, gercek bilimsel sonuc)",
        'imaging_plus_demographics': "IMAGING + YAS/CINSIYET/EGITIM (konfound kontrol, leakage yok)",
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
    """Run a binary (pairwise) experiment for one feature mode."""
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
    return experiment_binary_by_mode(df, mode='imaging_only',
                                     pair=('HC', 'AD'), residualize=False)


def experiment_twostage(df):
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
    print(f"  Stage 2 (MCI vs AD):  Acc = {np.nanmean(stage2_accs):.3f} +/- {np.nanstd(stage2_accs):.3f}")
    print(f"  Combined 3-class:     Acc = {np.nanmean(combined_accs):.3f} +/- {np.nanstd(combined_accs):.3f}")

    return {
        'stage1': np.mean(stage1_accs),
        'stage2': np.nanmean(stage2_accs),
        'combined': np.nanmean(combined_accs)
    }


def experiment_twostage_by_mode(df, mode, residualize=False, use_combat=False,
                                n_splits=5, n_repeats=10, random_state=42,
                                calibrate=True):
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
                    k_best=(MAX_FEATURES_HARD_LIMIT if X.shape[1] > MAX_FEATURES_HARD_LIMIT else None),
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

    print(f"  Nested CV SVM (GridSearch, CV-safe): Acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}, AUC={np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}")
    print(f"  Son fold'daki en iyi parametreler: {svm_gs.best_params_}")

    return {'acc': np.mean(accs), 'auc': np.nanmean(aucs)}


def plot_results(res_3class, res_binary, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top15 = res_3class.head(15)
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top15)))
    bars = ax.barh(
        range(len(top15)),
        top15['AUC'].values,
        color=colors, edgecolor='gray', linewidth=0.5
    )
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([f"{r['Features'][:12]}\n{r['Model']}" for _, r in top15.iterrows()], fontsize=8)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Sanslilik (0.5)')
    ax.axvline(x=0.33, color='orange', linestyle=':', alpha=0.7, label='Rastgele')
    ax.set_xlabel('AUC (weighted OvR)')
    ax.set_title('3-Sinif HC/MCI/AD - AUC Karsilastirmasi\n(Ust 15 kombinasyon)')
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 0.85)

    top15b = res_binary.head(15)
    ax2 = axes[1]
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top15b)))
    ax2.barh(
        range(len(top15b)),
        top15b['AUC'].values,
        color=colors2, edgecolor='gray', linewidth=0.5
    )
    ax2.set_yticks(range(len(top15b)))
    ax2.set_yticklabels([f"{r['Features'][:12]}\n{r['Model']}" for _, r in top15b.iterrows()], fontsize=8)
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Sanslilik (0.5)')
    ax2.set_xlabel('AUC (binary HC vs AD)')
    ax2.set_title('Binary HC vs AD - AUC Karsilastirmasi\n(Ust 15 kombinasyon)')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.3, 0.95)

    plt.tight_layout()
    path = os.path.join(output_dir, 'enhanced_classification_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGorsel kaydedildi: {path}")


def plot_confusion_matrix(df, output_dir):
    img_sets = get_feature_sets_by_mode(df, 'imaging_only')
    cols = img_sets.get('Graf_AUC+Null', img_sets['Graf_AUC'])
    X = df[cols].values
    y = df['label'].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds, all_true = [], []
    for train_idx, test_idx in cv.split(X, y):
        pipe = make_imb_pipeline(
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        )
        pipe.fit(X[train_idx], y[train_idx])
        all_preds.extend(pipe.predict(X[test_idx]))
        all_true.extend(y[test_idx])

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=['HC', 'MCI', 'AD'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix - GBM (imaging_only, CV-safe)\n(Graf_AUC+Null, 5-fold CV)')
    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix_best.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix kaydedildi: {path}")


def _slug(task, mode, residualize, variant=None):
    t = 'task-' + (task if isinstance(task, str) else f"{task[0]}-{task[1]}")
    r = 'resid' if residualize else 'raw'
    base = f"{t}__{mode}__{r}"
    return f"{base}__{variant}" if variant else base


def build_leaderboard(results_dict, out_dir, variant=None):
    """Aggregate sweep results into a ranked leaderboard table."""
    rows = []
    for (task, mode, resid), df_res in results_dict.items():
        if df_res is None or df_res.empty:
            continue
        df_nn = df_res.dropna(subset=['AUC'])
        if df_nn.empty:
            continue
        top = df_nn.sort_values('AUC', ascending=False).iloc[0]
        rows.append({
            'Task': task if isinstance(task, str) else f"{task[0]}-{task[1]}",
            'Mode': mode,
            'Residualized': bool(resid),
            'ComBat': bool(top.get('ComBat', False)),
            'Optuna': bool(top.get('Optuna', False)),
            'Best_Features': top.get('Features', ''),
            'Best_Model': top.get('Model', ''),
            'n_feat': int(top.get('n_feat', 0)),
            'Acc': float(top.get('Acc', np.nan)),
            'AUC': float(top.get('AUC', np.nan)),
            'AUC_CI_low': float(top.get('AUC_CI_low', np.nan)),
            'AUC_CI_high': float(top.get('AUC_CI_high', np.nan)),
            'F1': float(top.get('F1', np.nan)),
            'Train_Acc': float(top.get('Train_Acc', np.nan)),
            'Leakage_Flag': (mode == 'imaging_plus_clinical'),
        })
    lb = pd.DataFrame(rows)
    if lb.empty:
        print("[WARN] Leaderboard bos — hic sonuc toplanmadi.")
        return lb

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{variant}" if variant else ""
    csv_path = os.path.join(out_dir, f"leaderboard{suffix}.csv")
    lb.to_csv(csv_path, index=False)
    print(f"\nLeaderboard CSV: {csv_path}")

    title = "Leaderboard (tuned)" if variant else "Leaderboard (fixed)"
    subtitle = {
        'fixed': "RepeatedStratifiedKFold + ComBat (sabit hiperparametreler)",
        'optuna': "RepeatedStratifiedKFold + ComBat + Optuna Bayesian HPO",
    }.get(variant, "")
    md_lines = [
        f"# {title}",
        "",
        subtitle,
        "",
        "Her (Task, Mode, Residualized) icin AUC en yuksek model.",
        "",
        "**UYARI:** `imaging_plus_clinical` modunda MMSE/CDR feature'lari target",
        "leakage icerir (tani tanimlari). Sadece upper-bound referans; tez ana",
        "sonuclari **imaging_only** satirlarindandir.",
        "",
    ]
    for task_lbl, g in lb.groupby('Task'):
        md_lines.append(f"## {task_lbl}")
        md_lines.append("")
        md_lines.append("| Mode | Residualized | ComBat | Optuna | Features | Model | n_feat | Acc | AUC [95% CI] | F1 | Train_Acc | Leakage |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        g_sorted = g.sort_values(['Leakage_Flag', 'Mode', 'Residualized'])
        for _, r in g_sorted.iterrows():
            flag = "**LEAKAGE**" if r['Leakage_Flag'] else ""
            ci_low = r.get('AUC_CI_low', np.nan)
            ci_high = r.get('AUC_CI_high', np.nan)
            if pd.notna(ci_low) and pd.notna(ci_high):
                auc_str = f"**{r['AUC']:.3f}** [{ci_low:.3f}-{ci_high:.3f}]"
            else:
                auc_str = f"**{r['AUC']:.3f}**"
            md_lines.append(
                f"| {r['Mode']} | {r['Residualized']} |"
                f" {'Y' if r.get('ComBat') else '-'} |"
                f" {'Y' if r.get('Optuna') else '-'} |"
                f" {r['Best_Features']} | {r['Best_Model']} |"
                f" {r['n_feat']} | {r['Acc']:.3f} | {auc_str} |"
                f" {r['F1']:.3f} | {r['Train_Acc']:.3f} | {flag} |"
            )
        md_lines.append("")
    md_path = os.path.join(out_dir, f"leaderboard{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Leaderboard MD : {md_path}")

    return lb


def _run_sweep_pass(df, ts_lookup, tasks, modes, sweep_dir,
                    use_combat, use_optuna, n_repeats, tag):
    print(f"  Pass: {tag} | ComBat={use_combat} | Optuna={use_optuna}"
          f" | n_repeats={n_repeats}")
    results = {}
    for task in tasks:
        for mode in modes:
            resid_opts = [False] if mode == 'imaging_plus_clinical' else [False, True]
            for resid in resid_opts:
                if task == '3class':
                    r = experiment_3class_by_mode(
                        df, mode, residualize=resid,
                        time_series_lookup=ts_lookup,
                        use_combat=use_combat, use_optuna=use_optuna,
                        n_repeats=n_repeats)
                elif task == '3class_twostage':
                    if mode == 'imaging_plus_clinical':
                        continue
                    try:
                        r = experiment_twostage_by_mode(
                            df, mode, residualize=resid,
                            use_combat=use_combat, n_repeats=n_repeats)
                    except Exception as e:
                        print(f"[WARN] twostage {mode} resid={resid} basarisiz: {e}")
                        r = pd.DataFrame()
                else:
                    r = experiment_binary_by_mode(
                        df, mode, pair=task, residualize=resid,
                        time_series_lookup=ts_lookup,
                        use_combat=use_combat, use_optuna=use_optuna,
                        n_repeats=n_repeats)
                results[(task, mode, resid)] = r
                slug = _slug(task, mode, resid, variant=tag)
                if r is not None and not r.empty:
                    r.to_csv(os.path.join(sweep_dir, f"{slug}.csv"),
                             index=False)
    return results


def _write_comparison(lb_fixed, lb_optuna, out_dir, baseline_ref=None):
    md_path = os.path.join(out_dir, "leaderboard_comparison.md")
    merged = lb_fixed.rename(columns={
        'AUC': 'AUC_S3_fixed',
        'AUC_CI_low': 'CI_low_S3_fixed',
        'AUC_CI_high': 'CI_high_S3_fixed',
    })[['Task', 'Mode', 'Residualized', 'AUC_S3_fixed',
        'CI_low_S3_fixed', 'CI_high_S3_fixed']].copy()
    if not lb_optuna.empty:
        opt = lb_optuna.rename(columns={
            'AUC': 'AUC_S3_optuna',
            'AUC_CI_low': 'CI_low_S3_optuna',
            'AUC_CI_high': 'CI_high_S3_optuna',
        })[['Task', 'Mode', 'Residualized', 'AUC_S3_optuna',
            'CI_low_S3_optuna', 'CI_high_S3_optuna']]
        merged = merged.merge(opt, on=['Task', 'Mode', 'Residualized'],
                              how='outer')
    if baseline_ref is not None and not baseline_ref.empty:
        s2 = baseline_ref.rename(columns={'AUC': 'AUC_S2'})
        if 'Residualized' not in s2.columns:
            s2['Residualized'] = False
        merged = merged.merge(
            s2[['Task', 'Mode', 'Residualized', 'AUC_S2']],
            on=['Task', 'Mode', 'Residualized'], how='left')
    if 'AUC_S2' in merged.columns:
        merged['Delta_S2_to_S3fixed'] = merged['AUC_S3_fixed'] - merged['AUC_S2']
    if 'AUC_S3_optuna' in merged.columns:
        merged['Delta_S3fixed_to_S3optuna'] = merged['AUC_S3_optuna'] - merged['AUC_S3_fixed']
    merged.to_csv(md_path.replace('.md', '.csv'), index=False)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# AUC Comparison\n\n")
        f.write(merged.to_markdown(index=False, floatfmt=".3f"))
    print(f"Comparison: {md_path}")


def run_enhanced_classification():
    """Run the full classification sweep and write the leaderboard tables."""
    print("Classification sweep — RepeatedKFold + ComBat + Optuna")

    df = load_all_features()
    print(f"Yuklendi: {len(df)} ozne, {len(df.columns)} kolon")
    print(f"Grup dagilimi: {df['group'].value_counts().to_dict()}")

    ts_lookup = None
    try:
        ts_lookup = load_timeseries_lookup(df['subject_id'].astype(str).tolist())
        print(f"  Tangent lookup: {len(ts_lookup)} denek, "
              f"{next(iter(ts_lookup.values())).shape if ts_lookup else 'N/A'}")
    except Exception as e:
        print(f"  [WARN] Tangent lookup yuklenemedi: {e}")

    tasks = ['3class', '3class_twostage',
             ('HC', 'AD'), ('HC', 'MCI'), ('MCI', 'AD')]
    modes = ['imaging_only', 'imaging_plus_demographics',
             'imaging_plus_clinical']

    sweep_dir = os.path.join(config.METRICS_DIR, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    use_combat = bool(getattr(config, 'USE_COMBAT', True)) and HAS_NH
    n_repeats = int(getattr(config, 'N_REPEATS', 10))

    results_fixed = _run_sweep_pass(
        df, ts_lookup, tasks, modes, sweep_dir,
        use_combat=use_combat, use_optuna=False,
        n_repeats=n_repeats, tag='fixed')

    results_optuna = {}
    print("[i] Optuna pass devre disi (SKIP_OPTUNA=1).")

    results = results_fixed

    def _get(task, mode, resid):
        return results.get((task, mode, resid), pd.DataFrame())

    res_img   = _get('3class', 'imaging_only', False)
    res_demog = _get('3class', 'imaging_plus_demographics', False)
    res_clin  = _get('3class', 'imaging_plus_clinical', False)

    if not res_img.empty:
        res_img.to_csv(os.path.join(config.METRICS_DIR,
                                    "results_imaging_only.csv"), index=False)
    if not res_demog.empty:
        res_demog.to_csv(os.path.join(config.METRICS_DIR,
                                      "results_imaging_plus_demographics.csv"),
                         index=False)
    if not res_clin.empty:
        res_clin.to_csv(os.path.join(config.METRICS_DIR,
                                     "results_imaging_plus_clinical_LEAKAGE.csv"),
                        index=False)

    res_3class_concat = pd.concat(
        [v for k, v in results.items() if k[0] == '3class' and v is not None and not v.empty],
        ignore_index=True)
    res_binary_concat = pd.concat(
        [v for k, v in results.items()
         if isinstance(k[0], tuple) and k[0] == ('HC', 'AD')
         and v is not None and not v.empty],
        ignore_index=True)
    if not res_3class_concat.empty:
        res_3class_concat.sort_values('AUC', ascending=False).to_csv(
            os.path.join(config.METRICS_DIR, "enhanced_3class_results.csv"),
            index=False)
    if not res_binary_concat.empty:
        res_binary_concat.sort_values('AUC', ascending=False).to_csv(
            os.path.join(config.METRICS_DIR, "enhanced_binary_results.csv"),
            index=False)

    lb_fixed = build_leaderboard(results_fixed, config.METRICS_DIR,
                                 variant='fixed')
    lb_optuna = (build_leaderboard(results_optuna, config.METRICS_DIR,
                                   variant='optuna')
                 if results_optuna else pd.DataFrame())
    lb = build_leaderboard(results_fixed, config.METRICS_DIR)
    try:
        _write_comparison(lb_fixed, lb_optuna, config.METRICS_DIR)
    except Exception as exc:
        print(f"[WARN] comparison table failed: {exc}")

    try:
        output_dir = os.path.join(config.PROJECT_ROOT, "results", "figures")
        if not res_3class_concat.empty and not res_binary_concat.empty:
            plot_results(res_3class_concat.sort_values('AUC', ascending=False),
                         res_binary_concat.sort_values('AUC', ascending=False),
                         output_dir)
        plot_confusion_matrix(df, output_dir)
    except Exception as e:
        print(f"[WARN] Gorsel uretimi atlandi: {e}")

    print("OZET — Headline (imaging_only, residualize=False)")
    for task in tasks:
        r = _get(task, 'imaging_only', False)
        tlbl = task if isinstance(task, str) else f"{task[0]}-{task[1]}"
        if r is None or r.empty:
            print(f"  {tlbl:<12}: sonuc yok")
            continue
        top = r.dropna(subset=['AUC']).sort_values('AUC', ascending=False)
        if top.empty:
            print(f"  {tlbl:<12}: tum deneyler NaN")
            continue
        t = top.iloc[0]
        print(f"  {tlbl:<12}: Acc={t['Acc']:.3f} AUC={t['AUC']:.3f} "
              f"F1={t['F1']:.3f} ({t['Model']} / {t['Features']})")

    print(f"\nSweep CSVs : {sweep_dir}")
    print(f"Leaderboard fixed : {os.path.join(config.METRICS_DIR, 'leaderboard_fixed.md')}")
    if results_optuna:
        print(f"Leaderboard optuna: {os.path.join(config.METRICS_DIR, 'leaderboard_optuna.md')}")
    return {'fixed': results_fixed, 'optuna': results_optuna}


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    run_enhanced_classification()
