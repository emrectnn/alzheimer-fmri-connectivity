"""Resting-state fMRI preprocessing.

Denoising with 24-parameter Friston confound regression, framewise-displacement
and DVARS motion QC, and atlas-based ROI mean time-series extraction.
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00_config")


def load_atlas(atlas_name=None):
    """Load the parcellation atlas maps and region labels."""
    if atlas_name is None:
        atlas_name = config.ATLAS_NAME

    if atlas_name == "aal":
        atlas = datasets.fetch_atlas_aal()
        return atlas.maps, atlas.labels
    elif atlas_name == "schaefer100":
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
        return atlas.maps, atlas.labels
    else:
        raise ValueError(f"Bilinmeyen atlas: {atlas_name}")


def create_masker(atlas_maps, tr=None):
    """Create a NiftiLabelsMasker that extracts mean ROI time series."""
    if tr is None:
        tr = config.TR

    masker = NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize=True,
        detrend=True,
        low_pass=config.LOW_PASS,
        high_pass=config.HIGH_PASS,
        t_r=tr,
        smoothing_fwhm=config.SMOOTHING_FWHM,
        memory=None,
        verbose=0,
    )
    return masker


def load_confounds(fmri_path, n_skip=None):
    """Build the 24-parameter Friston confound matrix for a scan."""
    if n_skip is None:
        n_skip = config.FIRST_N_VOLUMES

    confound_candidates = [
        fmri_path.replace(".nii.gz", "_confounds.tsv"),
        fmri_path.replace(".nii.gz", "_desc-confounds_timeseries.tsv"),
        fmri_path.replace("_bold.nii.gz", "_desc-confounds_timeseries.tsv"),
    ]

    for cf in confound_candidates:
        if cf != fmri_path and os.path.exists(cf):
            conf_df = pd.read_csv(cf, sep="\t")
            cols = [c for c in conf_df.columns
                    if any(k in c.lower() for k in
                           ["white_matter", "csf", "trans_x", "trans_y",
                            "trans_z", "rot_x", "rot_y", "rot_z",
                            "framewise_displacement"])]
            if cols:
                confounds = conf_df[cols].values[n_skip:, :]
                confounds = np.nan_to_num(confounds, nan=0.0)
                print(f"    Confound yuklendi: {len(cols)} parametre")
                return confounds

    print("    Confound dosyasi bulunamadi -- yalnizca bandpass + detrend uygulanacak")
    print("    (ADNI DICOM verisi confound .tsv icermez, bu normal)")
    return None


def check_motion(fmri_path, n_skip=None, fd_threshold=0.5,
                 max_fd_percent=0.20):
    """Compute framewise displacement and flag high-motion scans."""
    if n_skip is None:
        n_skip = config.FIRST_N_VOLUMES

    confound_files = [
        fmri_path.replace(".nii.gz", "_confounds.tsv"),
        fmri_path.replace(".nii.gz", "_desc-confounds_timeseries.tsv"),
    ]

    for cf in confound_files:
        if os.path.exists(cf):
            conf_df = pd.read_csv(cf, sep="\t")
            if "framewise_displacement" in conf_df.columns:
                fd = conf_df["framewise_displacement"].values[n_skip:]
                fd = np.nan_to_num(fd, nan=0.0)

                mean_fd = np.mean(fd)
                max_fd = np.max(fd)
                pct_bad = np.mean(fd > fd_threshold) * 100

                passed = (mean_fd < fd_threshold) and (pct_bad < max_fd_percent * 100)

                return passed, {
                    "mean_fd": mean_fd,
                    "max_fd": max_fd,
                    "pct_above_threshold": pct_bad,
                    "passed": passed,
                }

    return True, {"note": "FD verisi mevcut degil, kontrol atlandi"}


def preprocess_subject(fmri_path, masker, n_skip=None, check_quality=True):
    """Preprocess one scan and return its ROI time series and QC info."""
    if n_skip is None:
        n_skip = config.FIRST_N_VOLUMES

    info = {"fmri_path": fmri_path}

    img = nib.load(fmri_path)
    shape = img.shape
    info["original_shape"] = shape
    print(f"    Ham shape: {shape}")

    if len(shape) != 4:
        raise ValueError(f"4D bekleniyor, {len(shape)}D bulundu: {shape}")

    zooms = img.header.get_zooms()
    if len(zooms) > 3 and zooms[3] > 0:
        tr_from_header = float(zooms[3])
        info["tr"] = tr_from_header
        print(f"    TR (header): {tr_from_header:.2f} s")

    n_total = shape[3]
    if n_total <= n_skip:
        raise ValueError(f"Yetersiz zaman noktasi: {n_total} <= {n_skip}")

    img_trimmed = image.index_img(img, slice(n_skip, None))
    info["n_timepoints_after_trim"] = n_total - n_skip
    print(f"    Ilk {n_skip} TR atildi -> {n_total - n_skip} TR kaldi")

    if check_quality:
        motion_ok, motion_info = check_motion(fmri_path, n_skip)
        info["motion"] = motion_info
        if not motion_ok:
            print(f"    [!] Asiri hareket tespit edildi: {motion_info}")
            raise ValueError(f"Hareket artefakti: mean FD={motion_info.get('mean_fd', '?'):.3f}")

    confounds = load_confounds(fmri_path, n_skip)

    time_series = masker.fit_transform(img_trimmed, confounds=confounds)
    info["processed_shape"] = time_series.shape
    print(f"    Islenmis shape: {time_series.shape}")

    if np.any(np.isnan(time_series)):
        n_nan = np.sum(np.isnan(time_series))
        print(f"    [!] {n_nan} NaN deger bulundu -- 0 ile degistiriliyor")
        time_series = np.nan_to_num(time_series, nan=0.0)

    return time_series, info


def quality_check(time_series, subject_id, save_dir=None):
    """Render a per-subject QC figure (time series, motion, connectivity)."""
    if save_dir is None:
        save_dir = config.FIGURES_DIR

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Contrôle qualité : {subject_id}", fontsize=14)

    ax = axes[0, 0]
    n_show = min(5, time_series.shape[1])
    for i in range(n_show):
        ax.plot(time_series[:, i], alpha=0.7, label=f"ROI {i+1}")
    ax.set_title("Série temporelle (5 premières ROI)")
    ax.set_xlabel("Points temporels (TR)")
    ax.set_ylabel("z-score")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    corr = np.corrcoef(time_series.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"Matrice de corrélation ({corr.shape[0]}x{corr.shape[1]})")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    ax.hist(time_series.flatten(), bins=50, color="steelblue", edgecolor="white")
    ax.set_title("Distribution du signal (z-score normalisé)")
    ax.set_xlabel("Valeur")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)

    ax = axes[1, 1]
    mean_abs_corr = np.mean(np.abs(corr), axis=0)
    ax.bar(range(len(mean_abs_corr)), mean_abs_corr, color="steelblue", width=1.0)
    ax.set_title("|Corrélation| moyenne par ROI")
    ax.set_xlabel("Indice ROI")
    ax.set_ylabel("|r| moy.")

    for idx in config.DMN_ROI_INDICES:
        if idx < len(mean_abs_corr):
            ax.bar(idx, mean_abs_corr[idx], color="orange", width=1.0)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"qc_{subject_id}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    stats = {
        "min_corr": float(corr[np.triu_indices_from(corr, k=1)].min()),
        "max_corr": float(corr[np.triu_indices_from(corr, k=1)].max()),
        "mean_corr": float(np.mean(corr[np.triu_indices_from(corr, k=1)])),
        "std_signal": float(np.std(time_series)),
        "n_timepoints": time_series.shape[0],
        "n_rois": time_series.shape[1],
    }
    print(f"    QC: r in [{stats['min_corr']:.3f}, {stats['max_corr']:.3f}], "
          f"mean r = {stats['mean_corr']:.3f}")

    return stats


def preprocess_all(subjects, save_dir=None, skip_existing=True):
    """Preprocess every subject and save ROI time series to disk."""
    if save_dir is None:
        save_dir = config.PREPROCESSED_DIR

    os.makedirs(save_dir, exist_ok=True)

    atlas_maps, atlas_labels = load_atlas()

    masker_cache = {}

    processed = []
    failed = []
    motion_qc_rows = []

    print(f"ON ISLEME BASLIYOR -- {len(subjects)} denek")

    for i, subj in enumerate(subjects):
        subj_id = subj['id']
        print(f"\n[{i+1}/{len(subjects)}] {subj_id} ({subj['group']})")

        save_path = os.path.join(save_dir, f"{subj_id}_timeseries.npy")

        if skip_existing and os.path.exists(save_path):
            print("    Zaten mevcut, atlaniyor.")
            ts = np.load(save_path)
            subj["timeseries_path"] = save_path
            subj["n_timepoints"] = ts.shape[0]
            processed.append(subj)
            try:
                _, minfo = check_motion(subj["fmri_path"])
            except Exception as _e:
                minfo = {"note": f"check_motion hatasi: {_e}"}
            motion_qc_rows.append({
                "subject_id": subj_id,
                "group": subj.get("group", ""),
                "mean_fd": minfo.get("mean_fd", np.nan),
                "max_fd": minfo.get("max_fd", np.nan),
                "pct_high_fd": minfo.get("pct_above_threshold", np.nan),
                "n_timepoints": int(ts.shape[0]),
                "tr": config.SUBJECT_TR.get(subj_id, config.TR),
                "passed": minfo.get("passed", True),
                "note": minfo.get("note", ""),
            })
            continue

        subj_tr = config.SUBJECT_TR.get(subj_id, config.TR)
        if subj_tr not in masker_cache:
            masker_cache[subj_tr] = create_masker(atlas_maps, tr=subj_tr)
            print(f"    Masker olusturuldu: TR={subj_tr}s")
        masker = masker_cache[subj_tr]

        try:
            ts, info = preprocess_subject(subj["fmri_path"], masker)

            np.save(save_path, ts)
            subj["timeseries_path"] = save_path
            subj["n_timepoints"] = ts.shape[0]
            subj["preprocess_info"] = info

            qc_stats = quality_check(ts, subj["id"])
            subj["qc_stats"] = qc_stats

            processed.append(subj)
            minfo = info.get("motion", {}) if isinstance(info, dict) else {}
            motion_qc_rows.append({
                "subject_id": subj_id,
                "group": subj.get("group", ""),
                "mean_fd": minfo.get("mean_fd", np.nan),
                "max_fd": minfo.get("max_fd", np.nan),
                "pct_high_fd": minfo.get("pct_above_threshold", np.nan),
                "n_timepoints": int(ts.shape[0]),
                "tr": subj_tr,
                "passed": minfo.get("passed", True),
                "note": minfo.get("note", ""),
            })
            print(f"    OK Kaydedildi: {save_path}")

        except Exception as e:
            print(f"    HATA: {e}")
            failed.append({"id": subj["id"], "error": str(e)})

    print("ON ISLEME TAMAMLANDI")
    print(f"  Basarili: {len(processed)} / {len(subjects)}")
    print(f"  Basarisiz: {len(failed)}")
    if processed:
        groups = {}
        for s in processed:
            g = s["group"]
            groups[g] = groups.get(g, 0) + 1
        for g in ["HC", "MCI", "AD"]:
            print(f"  {g}: {groups.get(g, 0)}")

    if failed:
        fail_df = pd.DataFrame(failed)
        fail_path = os.path.join(save_dir, "failed_subjects.csv")
        fail_df.to_csv(fail_path, index=False)
        print(f"Basarisiz denekler: {fail_path}")

    if motion_qc_rows:
        try:
            metrics_dir = os.path.join(config.RESULTS_DIR, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            qc_path = os.path.join(metrics_dir, "motion_qc.csv")
            pd.DataFrame(motion_qc_rows).to_csv(qc_path, index=False)
            print(f"Motion QC: {qc_path} ({len(motion_qc_rows)} denek)")
        except Exception as e:
            print(f"[WARN] motion_qc.csv yazilamadi: {e}")

    return processed


if __name__ == "__main__":
    from importlib import import_module
    adni_data = import_module("00b_adni_data")

    subjects = adni_data.prepare_subjects()

    if subjects:
        processed = preprocess_all(subjects)
        print(f"\nHazir: {len(processed)} denek islendi.")
    else:
        print("\nVeri henuz yuklenmedi. Once 00b_adni_data.py'yi calistirin.")

        print("\n--- Demo: Nilearn dahili veri ile pipeline testi ---")
        try:
            adhd = datasets.fetch_adhd(n_subjects=2)
            atlas_maps, labels = load_atlas()
            masker = create_masker(atlas_maps, tr=2.5)

            for i, fmri_file in enumerate(adhd.func[:2]):
                print(f"\nDemo denek {i+1}: {os.path.basename(fmri_file)}")
                ts, info = preprocess_subject(fmri_file, masker, check_quality=False)
                print(f"  Shape: {ts.shape}")
                qc = quality_check(ts, f"demo_{i+1}")
        except Exception as e:
            print(f"Demo hatasi: {e}")
