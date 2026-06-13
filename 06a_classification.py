"""Baseline cross-validated HC/MCI/AD classification from graph and nodal features."""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_validate
)
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")


def build_feature_matrix(global_metrics_df, nodal_metrics_list,
                          null_model_df=None, dynamic_df=None,
                          use_nodal=True):
    """Assemble the feature matrix from graph, nodal and null-model features.

    Args:
        global_metrics_df: Per-subject global graph metrics.
        nodal_metrics_list: Per-subject nodal metric dicts.
        null_model_df: Optional null-model deviation features.
        dynamic_df: Optional dynamic-FC features.
        use_nodal: Include nodal features when True.

    Returns:
        Tuple (X, y, feature_names).
    """
    subjects = global_metrics_df["subject_id"].values
    y = global_metrics_df["label"].values
    groups = global_metrics_df["group"].values

    feature_parts = []
    feature_names = []

    global_cols = [c for c in global_metrics_df.columns
                   if c not in ["subject_id", "label", "group"]
                   and global_metrics_df[c].dtype in [np.float64, np.int64, float, int]]

    auc_cols = [c for c in global_cols if c.startswith("auc_")]

    other_global = ["sigma", "modularity", "dmn_clustering",
                     "rich_club_mean", "cb_cx_mean_corr",
                     "consensus_stability"]
    other_global = [c for c in other_global if c in global_cols]

    selected_global = auc_cols + other_global
    if selected_global:
        X_global = global_metrics_df[selected_global].values
        feature_parts.append(X_global)
        feature_names.extend(selected_global)

    if use_nodal and nodal_metrics_list:
        dmn_idx = config.DMN_ROI_INDICES
        X_nodal = []
        nodal_names = []

        for nm in nodal_metrics_list:
            row = []
            for metric in ["degree_centrality", "betweenness_centrality", "clustering"]:
                if metric in nm:
                    vals = nm[metric]
                    dmn_vals = [vals[i] for i in dmn_idx if i < len(vals)]
                    row.extend(dmn_vals)
                    if not nodal_names:
                        for i, idx in enumerate(dmn_idx):
                            if idx < len(vals):
                                nodal_names.append(f"dmn_{metric[:4]}_{idx}")
            X_nodal.append(row)

        if X_nodal:
            X_nodal = np.array(X_nodal)
            feature_parts.append(X_nodal)
            if not any(n.startswith("dmn_") for n in feature_names):
                feature_names.extend(nodal_names)

    if null_model_df is not None:
        dev_cols = [c for c in null_model_df.columns if c.startswith("dev_")]
        if dev_cols:
            null_aligned = null_model_df.set_index("subject_id").loc[subjects]
            X_null = null_aligned[dev_cols].values
            feature_parts.append(X_null)
            feature_names.extend(dev_cols)

    if dynamic_df is not None:
        dyn_cols = [c for c in dynamic_df.columns
                    if c not in ["subject_id", "label", "group"]
                    and dynamic_df[c].dtype in [np.float64, np.int64, float, int]]
        if dyn_cols:
            dyn_aligned = dynamic_df.set_index("subject_id").loc[subjects]
            X_dyn = dyn_aligned[dyn_cols].values
            feature_parts.append(X_dyn)
            feature_names.extend(dyn_cols)

    X = np.hstack(feature_parts)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"[!] {nan_mask.sum()} NaN deger 0 ile degistirildi")
        X = np.nan_to_num(X, nan=0.0)

    while len(feature_names) < X.shape[1]:
        feature_names.append(f"feature_{len(feature_names)}")
    feature_names = feature_names[:X.shape[1]]

    print(f"\nFeature matrisi: {X.shape} (hedef: 50-70 boyut)")
    print(f"  Global: {len(selected_global) if selected_global else 0}")
    n_nodal = X_nodal.shape[1] if isinstance(X_nodal, np.ndarray) and X_nodal.ndim == 2 else 0
    print(f"  Nodal DMN: {n_nodal}")
    print(f"  Null model: {len(dev_cols) if null_model_df is not None and dev_cols else 0}")
    print(f"  Dinamik: {len(dyn_cols) if dynamic_df is not None and dyn_cols else 0}")

    return X, y, groups, np.array(feature_names)


def get_models():
    """Return the dictionary of candidate classifiers."""
    return {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                       probability=True, random_state=config.RANDOM_STATE))
        ]),

        "SVM_Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0,
                       probability=True, random_state=config.RANDOM_STATE))
        ]),

        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=config.RANDOM_STATE))
        ]),

        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=2000,
                random_state=config.RANDOM_STATE))
        ]),

        "RF_FS_SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(
                RandomForestClassifier(100, random_state=config.RANDOM_STATE),
                threshold="median"
            )),
            ("clf", SVC(kernel="rbf", probability=True,
                       random_state=config.RANDOM_STATE))
        ]),
    }


def cross_validate_all(X, y, n_splits=None):
    """Cross-validate every candidate model and collect scores.

    Args:
        X: Feature matrix.
        y: Labels.
        n_splits: Number of CV folds.

    Returns:
        Dict mapping model name to its score summary.
    """
    if n_splits is None:
        n_splits = config.N_FOLDS

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.RANDOM_STATE)
    models = get_models()

    results = {}
    for name, model in models.items():
        print(f"\n  {name}...", end="")

        scoring = ["accuracy", "f1_macro"]

        n_classes = len(np.unique(y))
        if n_classes > 2:
            scoring.append("roc_auc_ovr")
        else:
            scoring.append("roc_auc")

        try:
            scores = cross_validate(
                model, X, y, cv=cv, scoring=scoring,
                return_train_score=True
            )

            auc_key = "test_roc_auc_ovr" if "test_roc_auc_ovr" in scores else "test_roc_auc"

            results[name] = {
                "acc_test": (f"{scores['test_accuracy'].mean():.3f} ± "
                             f"{scores['test_accuracy'].std():.3f}"),
                "f1_test": (f"{scores['test_f1_macro'].mean():.3f} ± "
                            f"{scores['test_f1_macro'].std():.3f}"),
                "auc_test": f"{scores[auc_key].mean():.3f} ± {scores[auc_key].std():.3f}",
                "acc_train": f"{scores['train_accuracy'].mean():.3f}",
                "raw_scores": scores,
            }
            print(f" Acc={results[name]['acc_test']}, AUC={results[name]['auc_test']}")

        except Exception as e:
            print(f" HATA: {e}")
            results[name] = {"error": str(e)}

    return results


def feature_importance_analysis(X, y, feature_names, top_n=20):
    """Rank features by a random-forest importance estimate.

    Args:
        X: Feature matrix.
        y: Labels.
        feature_names: Names aligned with X columns.
        top_n: Number of top features to return.

    Returns:
        Ranked list/table of the most important features.
    """
    rf = RandomForestClassifier(
        n_estimators=500, random_state=config.RANDOM_STATE
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    feature_names = np.array(feature_names)

    idx = np.argsort(importances)[::-1][:top_n]

    print(f"\nEn onemli {top_n} feature:")
    for rank, j in enumerate(idx):
        print(f"  {rank+1:2d}. {feature_names[j]}: {importances[j]:.4f}")

    return feature_names[idx], importances[idx], importances


def run_classification(global_metrics_df, nodal_metrics_list,
                        null_model_df=None, dynamic_df=None):
    """Run the baseline classification pipeline end to end.

    Args:
        global_metrics_df: Per-subject global graph metrics.
        nodal_metrics_list: Per-subject nodal metric dicts.
        null_model_df: Optional null-model deviation features.
        dynamic_df: Optional dynamic-FC features.

    Returns:
        Tuple (cv_results, X, y, feature_names).
    """
    print("SINIFLANDIRMA PIPELINE")

    X, y, groups, feature_names = build_feature_matrix(
        global_metrics_df, nodal_metrics_list,
        null_model_df, dynamic_df
    )

    for label, name in config.GROUP_LABELS.items():
        n = np.sum(y == label)
        print(f"  {name}: {n} denek")

    print("\n--- Cross-Validation Sonuclari ---")
    cv_results = cross_validate_all(X, y)

    results_df = pd.DataFrame({
        k: {m: v for m, v in r.items() if m != "raw_scores"}
        for k, r in cv_results.items() if "error" not in r
    }).T
    print(f"\n{results_df.to_string()}")

    top_features, top_importances, all_importances = feature_importance_analysis(
        X, y, feature_names
    )

    results_path = os.path.join(config.METRICS_DIR, "classification_results.csv")
    results_df.to_csv(results_path)
    print(f"\nSonuclar kaydedildi: {results_path}")

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": all_importances,
    }).sort_values("importance", ascending=False)
    fi_path = os.path.join(config.METRICS_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)

    return cv_results, X, y, feature_names
