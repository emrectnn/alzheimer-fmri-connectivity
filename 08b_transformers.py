"""Custom scikit-learn transformers and CV-safe pipeline builders.

Contains the confound regressor, the ComBat harmonization transformer, and the
imbalanced-learn / tangent-space pipeline constructors used across the sweep.
All steps are fit only on the training fold to avoid data leakage.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    ImbPipeline = Pipeline
    SMOTE = None

try:
    from neuroHarmonize import harmonizationLearn, harmonizationApply
    HAS_NH = True
except ImportError:
    HAS_NH = False
    harmonizationLearn = None
    harmonizationApply = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")

MAX_FEATURES_HARD_LIMIT = 30


class ConfoundRegressor(BaseEstimator, TransformerMixin):

    """Regress confounds (age, sex, motion) out of features within each CV fold."""
    def __init__(self, n_confounds=0):
        """Store the transformer hyperparameters."""
        self.n_confounds = int(n_confounds) if n_confounds else 0

    def fit(self, X, y=None):
        """Fit the transformer on the training fold only (CV-safe)."""
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
        """Apply the fitted transformation to the input matrix."""
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
        """Store the transformer hyperparameters."""
        self.n_site_cols = int(n_site_cols) if n_site_cols else 0
        self.n_bio_cols = int(n_bio_cols) if n_bio_cols else 0

    def _split(self, X):
        """Split the input matrix into feature, biological and site columns."""
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
        """Build the covariate DataFrame ComBat needs (site + biological)."""
        covars = pd.DataFrame(index=np.arange(n_rows))
        if S is not None:
            covars['SITE'] = S[:, 0].astype(int).astype(str)
        if B is not None:
            for j in range(B.shape[1]):
                covars[f'bio_{j}'] = B[:, j]
        return covars

    def fit(self, X, y=None):
        """Fit the transformer on the training fold only (CV-safe)."""
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
        """Apply the fitted transformation to the input matrix."""
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


def make_pipeline(clf, k_best=None, n_pca=None):
    """Build a plain scikit-learn pipeline (impute, scale, select, classifier)."""
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
    """Build the imbalanced-learn pipeline (impute, scale, select, SMOTE, classifier).

    Every step is fit inside the pipeline, so it only ever sees the training fold
    and cross-validation stays leakage-free. The order is deliberate: impute ->
    ComBat (site harmonization, before scaling) -> scale -> confound regression
    (after scaling) -> feature selection -> PCA -> SMOTE -> classifier.
    """
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    # ComBat removes scanner/protocol effects and must run before scaling.
    if n_site_cols and n_site_cols > 0:
        steps.append(('combat', ComBatTransformer(
            n_site_cols=n_site_cols, n_bio_cols=n_bio_cols)))
    steps.append(('scaler', StandardScaler()))
    # Regress out confounds (age, sex, motion) once features share a scale.
    if n_confounds and n_confounds > 0:
        steps.append(('confound', ConfoundRegressor(n_confounds=n_confounds)))
    if k_best:
        steps.append(('select', SelectKBest(f_classif, k=k_best)))
    if n_pca:
        steps.append(('pca', PCA(n_components=n_pca, random_state=42)))
    # SMOTE sits in the pipeline so it oversamples training rows only.
    if use_smote and HAS_IMBLEARN:
        steps.append(('smote', SMOTE(k_neighbors=smote_k_neighbors, random_state=42)))
    steps.append(('clf', clf))
    return ImbPipeline(steps) if HAS_IMBLEARN else Pipeline(
        [(n, s) for n, s in steps if n != 'smote']
    )


def load_timeseries_lookup(subject_ids, data_dir=None, min_rois=None):
    """Load subject time series into an id-to-array dictionary."""
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
    """Build a tangent-space (optionally NBS) pipeline ending in the classifier."""
    connectivity_mod = import_module("02a_connectivity")
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
