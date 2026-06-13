"""Classifier roster and hyperparameter search.

Builds the candidate models (SVM, random forest, boosting, logistic, ordinal),
the Optuna-based tuning helpers, and the repeated cross-validation scorer.
Classifiers are wrapped in the leakage-safe pipelines from 08b_transformers.
"""

import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_validate, cross_val_score)
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

try:
    import mord
    HAS_MORD = True
except ImportError:
    HAS_MORD = False
    mord = None

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")

MAX_FEATURES_HARD_LIMIT = 30

# Pipeline builders live in 08b_transformers.
_tf = import_module("08b_transformers")
make_pipeline = _tf.make_pipeline
make_imb_pipeline = _tf.make_imb_pipeline
make_tangent_pipeline = _tf.make_tangent_pipeline


def cv_score(X, y, model, n_splits=5, random_state=42, n_repeats=None):
    """Repeated stratified cross-validation; returns AUC/accuracy/per-class metrics."""
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


def _build_models(n_feat, task='3class', n_confounds=None,
                  n_site_cols=0, n_bio_cols=0):
    """Build the roster of candidate classifiers wrapped in leakage-safe pipelines."""
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
    """Instantiate one classifier from an Optuna parameter dict."""
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
    """Suggest a hyperparameter set for a model from an Optuna trial."""
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
    """Tune the top models with Optuna and return the best pipelines."""
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
            """Optuna objective: mean CV AUC for a sampled hyperparameter set."""
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
