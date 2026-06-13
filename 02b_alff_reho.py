"""Per-ROI ALFF/fALFF (low-frequency amplitude) and ReHo (regional homogeneity)."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kod"))

config = import_module("00a_config")


def compute_alff_falff(ts: np.ndarray, tr: float,
                       band: tuple[float, float] = (0.01, 0.08)) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROI ALFF and fALFF from the Welch power spectrum.

    Args:
        ts: (n_timepoints, n_roi) ROI time series.
        tr: Repetition time in seconds.
        band: (low, high) frequency band in Hz.

    Returns:
        Tuple (alff, falff), each of shape (n_roi,).
    """
    T, n_roi = ts.shape
    fs = 1.0 / tr
    nperseg = min(T, 64)
    alff = np.zeros(n_roi)
    falff = np.zeros(n_roi)
    low, high = band
    for i in range(n_roi):
        x = ts[:, i]
        x = x - np.mean(x)
        if np.std(x) < 1e-8:
            continue
        freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=nperseg)
        amp = np.sqrt(psd)
        total = amp.sum()
        band_mask = (freqs >= low) & (freqs <= high)
        band_amp = amp[band_mask].sum()
        alff[i] = band_amp
        falff[i] = band_amp / total if total > 1e-8 else 0.0
    return alff, falff


def kendalls_w(data: np.ndarray) -> float:
    """Kendall's W coefficient of concordance.

    Args:
        data: (n_raters, n_items) matrix of values.

    Returns:
        Concordance coefficient in [0, 1].
    """
    n_raters, n_items = data.shape
    if n_raters < 2 or n_items < 2:
        return 0.0
    ranks = np.apply_along_axis(rankdata, 1, data)
    R = ranks.sum(axis=0)
    mean_R = R.mean()
    S = ((R - mean_R) ** 2).sum()
    denom = n_raters ** 2 * (n_items ** 3 - n_items) / 12.0
    if denom < 1e-12:
        return 0.0
    return float(S / denom)


def compute_reho_roi(ts: np.ndarray, k_neighbors: int = 6) -> np.ndarray:
    """ROI-level ReHo via concordance with the most-correlated neighbours.

    Args:
        ts: (n_timepoints, n_roi) ROI time series.
        k_neighbors: Number of neighbour ROIs per region.

    Returns:
        (n_roi,) array of ReHo values.
    """
    T, n_roi = ts.shape
    ts_z = (ts - ts.mean(axis=0, keepdims=True)) / (ts.std(axis=0, keepdims=True) + 1e-8)
    fc = np.corrcoef(ts_z.T)
    np.fill_diagonal(fc, -np.inf)
    reho = np.zeros(n_roi)
    for i in range(n_roi):
        neighbors = np.argsort(fc[i])[-k_neighbors:]
        cluster = np.vstack([ts[:, i], ts[:, neighbors].T])
        reho[i] = kendalls_w(cluster)
    return reho


def load_timeseries(sid: str) -> np.ndarray | None:
    """Load a subject's preprocessed ROI time series.

    Args:
        sid: Subject identifier.

    Returns:
        (n_timepoints, n_roi) array, or None if missing/invalid.
    """
    p = Path(config.PREPROCESSED_DIR) / f"{sid}_timeseries.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim != 2:
        return None
    T, N = arr.shape
    if T < N:
        arr = arr.T
    return arr


def main(atlas_name: str = 'AAL3', skip_existing: bool = True) -> int:
    """Compute ALFF/fALFF/ReHo for every subject and write the CSVs.

    Args:
        atlas_name: Atlas key used in the output filenames.
        skip_existing: Reuse already-computed subjects when True.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    metadata_csv = Path(config.NIFTI_DIR) / "subject_metadata.csv"
    if not metadata_csv.exists():
        print(f"[ERR] {metadata_csv} yok")
        return 1
    meta = pd.read_csv(metadata_csv, dtype={'subject_id': str})

    out_alff_dir = Path(config.RESULTS_DIR) / "alff"
    out_reho_dir = Path(config.RESULTS_DIR) / "reho"
    out_alff_dir.mkdir(parents=True, exist_ok=True)
    out_reho_dir.mkdir(parents=True, exist_ok=True)

    alff_path = out_alff_dir / f"alff_{atlas_name}.csv"
    falff_path = out_alff_dir / f"falff_{atlas_name}.csv"
    reho_path = out_reho_dir / f"reho_{atlas_name}.csv"

    existing_alff = {}
    existing_falff = {}
    existing_reho = {}
    if skip_existing and alff_path.exists() and reho_path.exists():
        try:
            ea = pd.read_csv(alff_path, dtype={'subject_id': str})
            ef = pd.read_csv(falff_path, dtype={'subject_id': str}) if falff_path.exists() else ea
            er = pd.read_csv(reho_path, dtype={'subject_id': str})
            existing_alff = {r['subject_id']: r.tolist() for _, r in ea.iterrows()}
            existing_falff = {r['subject_id']: r.tolist() for _, r in ef.iterrows()}
            existing_reho = {r['subject_id']: r.tolist() for _, r in er.iterrows()}
            print(f"[SKIP] Mevcut CSV'de {len(existing_alff)} denek, tekrar hesaplanmayacak")
        except Exception as e:
            print(f"[WARN] mevcut CSV okunamadi: {e}; tum denekler yeniden hesaplanacak")
            existing_alff = existing_falff = existing_reho = {}

    alff_rows, falff_rows, reho_rows = [], [], []
    missing = []
    n_new = 0
    for _, row in meta.iterrows():
        sid = str(row['subject_id'])
        if sid in existing_alff and sid in existing_reho:
            alff_rows.append(existing_alff[sid])
            falff_rows.append(existing_falff.get(sid, existing_alff[sid]))
            reho_rows.append(existing_reho[sid])
            continue
        tr = float(row.get('tr', config.TR))
        ts = load_timeseries(sid)
        if ts is None:
            missing.append(sid)
            continue
        alff, falff = compute_alff_falff(ts, tr, band=config.ALFF_BAND)
        reho = compute_reho_roi(ts, k_neighbors=6)
        alff_rows.append([sid] + alff.tolist())
        falff_rows.append([sid] + falff.tolist())
        reho_rows.append([sid] + reho.tolist())
        n_new += 1
        print(f"[OK] {sid}: ALFF/fALFF/ReHo {len(alff)} ROI (yeni)")
    print(f"[INFO] {n_new} yeni denek hesaplandi, {len(alff_rows) - n_new} mevcut")

    if not alff_rows:
        print("[ERR] hicbir denek islenemedi")
        return 1
    def _pad(rows, max_len):
        """Right-pad rows with NaN so every row has the same column count."""
        padded = []
        for r in rows:
            if len(r) < max_len:
                r = r + [np.nan] * (max_len - len(r))
            padded.append(r)
        return padded
    max_len = max(len(r) for r in alff_rows)
    n_roi = max_len - 1
    cols = ['subject_id'] + [f'roi_{i+1}' for i in range(n_roi)]
    alff_rows = _pad(alff_rows, max_len)
    falff_rows = _pad(falff_rows, max_len)
    reho_rows = _pad(reho_rows, max_len)
    pd.DataFrame(alff_rows, columns=cols).to_csv(alff_path, index=False)
    pd.DataFrame(falff_rows, columns=cols).to_csv(falff_path, index=False)
    pd.DataFrame(reho_rows, columns=cols).to_csv(reho_path, index=False)
    print(f"[SAVE] {alff_path}  ({len(alff_rows)} satir)")
    print(f"[SAVE] {falff_path}")
    print(f"[SAVE] {reho_path}")
    if missing:
        print(f"[WARN] {len(missing)} denek eksik: {missing[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(atlas_name='AAL3'))
