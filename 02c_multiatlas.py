"""Multi-atlas pipeline.

Computes per-atlas (Schaefer-200, Harvard-Oxford-48) time series, connectivity,
and global graph metrics used as additional classification feature sets.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kod"))
config = import_module("00a_config")

try:
    from nilearn import datasets as nl_datasets
    from nilearn.maskers import NiftiLabelsMasker
    HAS_NILEARN = True
except Exception as e:
    HAS_NILEARN = False
    print(f"[WARN] nilearn yok: {e}")

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False


def load_atlas_by_name(atlas_key: str):
    """Load an atlas image, labels and ROI count from the config spec.

    Args:
        atlas_key: Atlas name ('Schaefer200', 'HO48', 'AAL3').

    Returns:
        Tuple (atlas_img, labels, n_rois); atlas_img is None for AAL3.
    """
    if not HAS_NILEARN:
        raise RuntimeError("nilearn gerekli")

    spec = config.ATLASES.get(atlas_key)
    if spec is None:
        raise ValueError(f"Bilinmeyen atlas: {atlas_key}")

    fetcher = spec.get('fetcher')
    if atlas_key == 'Schaefer200' or fetcher == 'schaefer_2018':
        atlas = nl_datasets.fetch_atlas_schaefer_2018(
            n_rois=spec.get('n_rois', 200),
            yeo_networks=spec.get('yeo_networks', 7),
            resolution_mm=spec.get('resolution_mm', 2),
        )
        return atlas.maps, list(atlas.labels), spec['n_rois']
    if atlas_key == 'HO48' or fetcher == 'harvard_oxford':
        atlas = nl_datasets.fetch_atlas_harvard_oxford(
            atlas_name=spec.get('atlas_name', 'cort-maxprob-thr25-2mm'))
        return atlas.maps, list(atlas.labels), spec.get('n_rois', 48)
    if atlas_key == 'AAL3':
        return None, None, spec.get('n_rois', 166)
    raise ValueError(f"Atlas fetch tanimsiz: {atlas_key}")


def extract_timeseries(fmri_path: str, atlas_img, tr: float,
                       n_skip: int = None) -> np.ndarray:
    """Extract atlas-based ROI mean time series from one scan.

    Args:
        fmri_path: Path to the rsfMRI NIfTI.
        atlas_img: Labelled atlas image.
        tr: Repetition time in seconds.
        n_skip: Number of initial volumes to drop.

    Returns:
        (n_timepoints, n_roi) time series.
    """
    if n_skip is None:
        n_skip = getattr(config, 'FIRST_N_VOLUMES',
                         getattr(config, 'N_SKIP_VOLUMES', 5))

    from nilearn.image import index_img, load_img
    img = load_img(fmri_path)
    n_vol = img.shape[3]
    if n_skip >= n_vol:
        n_skip = 0
    img_trimmed = index_img(img, slice(n_skip, n_vol))

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,
        detrend=True,
        low_pass=config.LOW_PASS,
        high_pass=config.HIGH_PASS,
        t_r=tr,
        smoothing_fwhm=config.SMOOTHING_FWHM,
        memory=None, verbose=0,
    )
    ts = masker.fit_transform(img_trimmed)
    return ts


def compute_fc_vector(ts: np.ndarray) -> np.ndarray:
    """Upper-triangle Pearson correlation vector of a time series.

    Args:
        ts: (n_timepoints, n_roi) time series.

    Returns:
        1-D vector of unique edge correlations.
    """
    fc = np.corrcoef(ts.T)
    fc = np.nan_to_num(fc, nan=0.0)
    iu = np.triu_indices(fc.shape[0], k=1)
    return fc[iu]


def compute_global_graph_features(fc_mat: np.ndarray,
                                  density: float = 0.15) -> dict:
    """A few global graph metrics at a fixed density.

    Args:
        fc_mat: (n_roi, n_roi) correlation matrix.
        density: Target graph density for thresholding.

    Returns:
        Dict of global metrics (empty if networkx is unavailable).
    """
    if not HAS_NX:
        return {}
    n = fc_mat.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = fc_mat[iu]
    k = int(len(vals) * density)
    if k < 1:
        return {}
    thresh = np.partition(np.abs(vals), -k)[-k]
    adj = (np.abs(fc_mat) >= thresh).astype(int)
    np.fill_diagonal(adj, 0)

    G = nx.from_numpy_array(adj)
    feats = {}
    try:
        feats['global_efficiency'] = nx.global_efficiency(G)
    except Exception:
        feats['global_efficiency'] = np.nan
    try:
        feats['avg_clustering'] = nx.average_clustering(G)
    except Exception:
        feats['avg_clustering'] = np.nan
    try:
        feats['density'] = nx.density(G)
    except Exception:
        feats['density'] = density
    try:
        feats['assortativity'] = nx.degree_assortativity_coefficient(G)
    except Exception:
        feats['assortativity'] = np.nan
    try:
        feats['modularity'] = _compute_modularity(G)
    except Exception:
        feats['modularity'] = np.nan
    try:
        feats['transitivity'] = nx.transitivity(G)
    except Exception:
        feats['transitivity'] = np.nan
    return feats


def _compute_modularity(G) -> float:
    """Greedy-community modularity (Q) of a graph; NaN on failure."""
    from networkx.algorithms.community import greedy_modularity_communities, modularity
    try:
        comms = list(greedy_modularity_communities(G))
        return modularity(G, comms)
    except Exception:
        return np.nan


def run_atlas(atlas_key: str, skip_existing: bool = True) -> int:
    """Process every subject for one atlas (time series, FC, metrics).

    Args:
        atlas_key: Atlas name to process.
        skip_existing: Skip already-processed subjects when True.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    metadata_csv = Path(config.NIFTI_DIR) / "subject_metadata.csv"
    if not metadata_csv.exists():
        print(f"[ERR] {metadata_csv} yok")
        return 1
    meta = pd.read_csv(metadata_csv, dtype={'subject_id': str})

    try:
        atlas_img, labels, n_rois = load_atlas_by_name(atlas_key)
    except Exception as e:
        print(f"[ERR] {atlas_key} atlas yuklenemedi: {e}")
        return 1
    if atlas_img is None:
        print(f"[SKIP] {atlas_key}: bu modul yalnizca Schaefer200 / HO48 icin. "
              f"AAL3 mevcut pipeline tarafindan islenir.")
        return 0

    out_ts_dir = Path(config.PREPROCESSED_DIR)
    out_fc_dir = Path(config.RESULTS_DIR) / "connectivity"
    out_metric_dir = Path(config.METRICS_DIR)
    out_fc_dir.mkdir(parents=True, exist_ok=True)
    out_metric_dir.mkdir(parents=True, exist_ok=True)

    fc_path = out_fc_dir / f"fc_{atlas_key}.csv"
    global_path = out_metric_dir / f"global_{atlas_key}.csv"

    existing_fc_ids = set()
    existing_global_ids = set()
    if skip_existing:
        if fc_path.exists():
            try:
                existing_fc_ids = set(
                    pd.read_csv(fc_path, usecols=['subject_id'], dtype=str)
                    ['subject_id'].tolist())
            except Exception:
                existing_fc_ids = set()
        if global_path.exists():
            try:
                existing_global_ids = set(
                    pd.read_csv(global_path, usecols=['subject_id'], dtype=str)
                    ['subject_id'].tolist())
            except Exception:
                existing_global_ids = set()

    fc_rows_new = []
    global_rows_new = []
    n_done = 0
    n_skip = 0

    for _, row in meta.iterrows():
        sid = str(row['subject_id'])
        fmri_path = row.get('fmri_path')
        if not fmri_path or pd.isna(fmri_path):
            cand = Path(config.NIFTI_DIR) / sid / f"{sid}_rsfmri.nii.gz"
            fmri_path = str(cand) if cand.exists() else None
        tr = float(row.get('tr', config.TR))

        if sid in existing_fc_ids and sid in existing_global_ids:
            n_skip += 1
            continue

        ts_path = out_ts_dir / f"{sid}_timeseries_{atlas_key}.npy"
        if ts_path.exists():
            ts = np.load(ts_path)
        else:
            if not fmri_path or not Path(fmri_path).exists():
                print(f"[WARN] {sid}: fmri_path yok, atlaniyor")
                continue
            try:
                ts = extract_timeseries(str(fmri_path), atlas_img, tr)
                np.save(ts_path, ts)
            except Exception as e:
                print(f"[ERR] {sid} {atlas_key} extract basarisiz: {e}")
                continue

        if sid not in existing_fc_ids:
            fc_vec = compute_fc_vector(ts)
            fc_rows_new.append([sid] + fc_vec.tolist())

        if sid not in existing_global_ids:
            fc_mat = np.corrcoef(ts.T)
            fc_mat = np.nan_to_num(fc_mat, nan=0.0)
            feats = compute_global_graph_features(fc_mat)
            if feats:
                global_rows_new.append([sid, row.get('group', '')] + list(feats.values()))
        n_done += 1
        print(f"[OK] {sid} ({atlas_key}): TS shape={ts.shape}")

    print(f"[INFO] {atlas_key}: {n_done} yeni islendi, {n_skip} atlandi")

    if fc_rows_new:
        n_edges = len(fc_rows_new[0]) - 1
        cols = ['subject_id'] + [f'edge_{i}' for i in range(n_edges)]
        df_new = pd.DataFrame(fc_rows_new, columns=cols)
        if fc_path.exists():
            df_old = pd.read_csv(fc_path, dtype={'subject_id': str})
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset='subject_id', keep='last')
        else:
            df_all = df_new
        df_all.to_csv(fc_path, index=False)
        print(f"[SAVE] {fc_path}  ({len(df_all)} satir)")

    if global_rows_new:
        sample_feats = compute_global_graph_features(
            np.eye(n_rois))
        metric_cols = ['global_efficiency', 'avg_clustering', 'density',
                       'assortativity', 'modularity', 'transitivity']
        cols = ['subject_id', 'group'] + metric_cols
        df_new = pd.DataFrame(global_rows_new, columns=cols)
        if global_path.exists():
            df_old = pd.read_csv(global_path, dtype={'subject_id': str})
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset='subject_id', keep='last')
        else:
            df_all = df_new
        df_all.to_csv(global_path, index=False)
        print(f"[SAVE] {global_path}  ({len(df_all)} satir)")

    return 0


def main() -> int:
    """Run the multi-atlas pipeline for the requested atlases."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlases', nargs='+', default=['Schaefer200', 'HO48'])
    ap.add_argument('--no-skip-existing', action='store_true')
    args = ap.parse_args()

    rc = 0
    for atlas_key in args.atlases:
        print(f"\n=== Atlas: {atlas_key} ===")
        rc |= run_atlas(atlas_key, skip_existing=not args.no_skip_existing)
    return rc


if __name__ == "__main__":
    sys.exit(main())
