"""Functional connectivity construction.

Pearson correlation, Fisher z-transform, density thresholding, AUC multi-threshold
binary graphs, weighted adjacency for GNNs, and tangent-space embedding.
"""

import os
import sys
import numpy as np
import pickle
from nilearn.connectome import ConnectivityMeasure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")


def compute_connectivity(time_series, kind=None):
    """Compute the ROI-by-ROI Pearson connectivity matrix."""
    if kind is None:
        kind = config.CONN_KIND

    measure = ConnectivityMeasure(kind=kind)
    matrix = measure.fit_transform([time_series])[0]

    np.fill_diagonal(matrix, 0)
    return matrix


def fisher_z_transform(matrix):
    """Apply the Fisher r-to-z transform to a correlation matrix."""
    z_matrix = np.arctanh(np.clip(matrix, -0.999, 0.999))

    np.fill_diagonal(z_matrix, 0)
    return z_matrix


def threshold_absolute(matrix, threshold):
    """Binarize a matrix by an absolute correlation threshold."""
    binary = (np.abs(matrix) >= threshold).astype(int)
    np.fill_diagonal(binary, 0)
    return binary


def threshold_by_density(matrix, density):
    """Binarize a matrix keeping the strongest edges at a target density."""
    n = matrix.shape[0]

    upper = np.abs(matrix[np.triu_indices(n, k=1)])

    if density <= 0:
        return np.zeros_like(matrix, dtype=int)
    if density >= 1:
        return (np.abs(matrix) > 0).astype(int)

    percentile = 100 * (1 - density)
    thresh_val = np.percentile(upper, percentile)

    binary = (np.abs(matrix) >= thresh_val).astype(int)
    np.fill_diagonal(binary, 0)
    return binary


def compute_auc_multi_threshold(matrix, density_range=None):
    """Return binary graphs across a range of densities for the AUC strategy."""
    if density_range is None:
        density_range = config.DENSITY_RANGE

    thresholded = {}
    for d in density_range:
        thresholded[d] = threshold_by_density(matrix, d)

    return thresholded


def prepare_weighted_matrix(matrix):
    """Build a non-negative, row-normalized weighted adjacency matrix."""
    z_mat = fisher_z_transform(matrix)

    z_mat = np.maximum(z_mat, 0)

    row_sums = z_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    z_normalized = z_mat / row_sums

    np.fill_diagonal(z_normalized, 0)
    return z_normalized


try:
    from sklearn.base import BaseEstimator, TransformerMixin
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    BaseEstimator = object
    TransformerMixin = object


class TangentSpaceTransformer(BaseEstimator, TransformerMixin):

    """scikit-learn transformer for CV-safe tangent-space connectivity embedding."""
    def __init__(self, vectorize=True, time_series_lookup=None):
        self.vectorize = vectorize
        self.time_series_lookup = time_series_lookup

    def _resolve_time_series(self, X):
        if self.time_series_lookup is not None:
            keys = list(X)
            return [self.time_series_lookup[k] for k in keys]
        return list(X)

    def fit(self, X, y=None):
        ts_list = self._resolve_time_series(X)
        if not ts_list:
            raise ValueError("TangentSpaceTransformer.fit: X bos.")
        self.n_rois_ = ts_list[0].shape[1]
        self.measure_ = ConnectivityMeasure(
            kind='tangent',
            vectorize=self.vectorize,
            discard_diagonal=True,
        )
        self.measure_.fit(ts_list)
        return self

    def transform(self, X):
        if not hasattr(self, 'measure_'):
            raise RuntimeError(
                "TangentSpaceTransformer fit edilmedi. Once .fit(X_train) "
                "veya .fit_transform(X_train) cagirin."
            )
        ts_list = self._resolve_time_series(X)
        return self.measure_.transform(ts_list)

    def fit_transform(self, X, y=None, **fit_params):
        ts_list = self._resolve_time_series(X)
        if not ts_list:
            raise ValueError("TangentSpaceTransformer.fit_transform: X bos.")
        self.n_rois_ = ts_list[0].shape[1]
        self.measure_ = ConnectivityMeasure(
            kind='tangent',
            vectorize=self.vectorize,
            discard_diagonal=True,
        )
        return self.measure_.fit_transform(ts_list)


def load_timeseries_dict(subjects):
    """Load subject time series into an id-to-array dictionary."""
    ts_dict = {}
    for subj in subjects:
        ts_dict[subj["id"]] = np.load(subj["timeseries_path"])
    return ts_dict


def compute_all_matrices(subjects, kind=None, save=True):
    """Compute and save connectivity representations for every subject."""
    if kind is None:
        kind = config.CONN_KIND

    matrices = {}
    print(f"\nBaglanti matrisleri hesaplaniyor ({len(subjects)} denek)...")

    for i, subj in enumerate(subjects):
        subj_id = subj["id"]
        print(f"  [{i+1}/{len(subjects)}] {subj_id} ({subj['group']})", end="")

        ts = np.load(subj["timeseries_path"])

        mat_raw = compute_connectivity(ts, kind=kind)

        mat_z = fisher_z_transform(mat_raw)

        thresholded = compute_auc_multi_threshold(mat_raw)

        mat_weighted = prepare_weighted_matrix(mat_raw)

        matrices[subj_id] = {
            "raw": mat_raw,
            "z": mat_z,
            "thresholded": thresholded,
            "weighted": mat_weighted,
            "label": subj["label"],
            "group": subj["group"],
        }

        density = np.mean(thresholded[0.15] > 0)
        print(f" -- density@0.15: {density:.3f}")

    if save:
        save_path = os.path.join(config.PREPROCESSED_DIR, "connectivity_matrices.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(matrices, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nMatrisler kaydedildi: {save_path}")

    return matrices


def compute_group_means(matrices):
    """Compute the mean connectivity matrix for each clinical group."""
    group_mats = {"HC": [], "MCI": [], "AD": []}

    for subj_id, data in matrices.items():
        grp = data["group"]
        if grp in group_mats:
            group_mats[grp].append(data["raw"])

    means = {}
    for grp, mats in group_mats.items():
        if mats:
            means[grp] = {
                "mean": np.mean(mats, axis=0),
                "std": np.std(mats, axis=0),
                "n": len(mats),
            }
            print(f"  {grp}: n={len(mats)}, "
                  f"mean |r| = {np.mean(np.abs(means[grp]['mean'])):.4f}")

    return means


def threshold_sensitivity_analysis(matrix, subject_id="sample"):
    """Plot how graph density and connectedness vary with the threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    densities = np.arange(0.05, 0.55, 0.05)
    results = []

    for d in densities:
        binary = threshold_by_density(matrix, d)
        G = nx.from_numpy_array(binary)
        n_components = nx.number_connected_components(G)
        actual_density = nx.density(G)
        n_edges = G.number_of_edges()

        results.append({
            "target_density": d,
            "actual_density": actual_density,
            "n_edges": n_edges,
            "n_components": n_components,
        })

    import pandas as pd
    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df["target_density"], df["actual_density"], "b-o", label="Gercek Yogunluk")
    ax1.plot(df["target_density"], df["target_density"], "k--", alpha=0.3, label="Hedef")
    ax1.set_xlabel("Hedef Yogunluk")
    ax1.set_ylabel("Gercek Graf Yogunlugu")
    ax1.legend()
    ax1.set_title("Yogunluk Kalibrasyonu")

    ax2.plot(df["target_density"], df["n_components"], "r-s")
    ax2.set_xlabel("Hedef Yogunluk")
    ax2.set_ylabel("Bagli Bilesen Sayisi")
    ax2.set_title("Graf Baglantililigi")

    plt.suptitle(f"Seuillage Duyarlilik Analizi -- {subject_id}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, f"threshold_sensitivity_{subject_id}.png"), dpi=150)
    plt.close()

    return df


if __name__ == "__main__":
    import glob

    ts_files = sorted(glob.glob(
        os.path.join(config.PREPROCESSED_DIR, "*_timeseries.npy")
    ))

    if not ts_files:
        print("Henuz islenmis zaman serisi yok. Once 01_preprocess.py'yi calistirin.")
    else:
        print(f"{len(ts_files)} zaman serisi dosyasi bulundu.")
