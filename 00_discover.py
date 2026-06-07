"""Discover available subjects on disk and update the in-memory subject lists in the config module."""

import os
import re
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from importlib import import_module
config = import_module("00_config")


RSFMRI_PATTERNS = ["rsfmri", "fcmri"]

EXCLUDE_PATTERNS = ["moco", "perfusion", "relcbf", "_ph_", "phase_map", "phasemap"]

SUBJECT_ID_REGEX = re.compile(r"^\d{3}_S_\d{4,5}$")

ADNI_SEARCH_DIRS = [
    os.path.join(config.PROJECT_ROOT, "veri", "ADNI"),
    os.path.join(config.PROJECT_ROOT, "veri", "ADNI01"),
    os.path.join(config.PROJECT_ROOT, "veri", "ADNI02"),
    os.path.join(config.PROJECT_ROOT, "veri", "ADNI03"),
]

GROUP_REMAP = {
    "CN": "HC", "Normal": "HC", "Control": "HC",
    "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI", "SMC": "MCI",
    "AD": "AD", "Patient": "AD", "Dementia": "AD",
}


def scan_adni_directory(adni_dir=None):
    if adni_dir is None:
        adni_dir = config.DICOM_DIR

    if not os.path.isdir(adni_dir):
        return []

    subjects = []
    for name in sorted(os.listdir(adni_dir)):
        full_path = os.path.join(adni_dir, name)
        if os.path.isdir(full_path) and SUBJECT_ID_REGEX.match(name):
            subjects.append(name)

    return subjects


def scan_all_adni_directories():
    subject_to_dir = {}
    for adni_dir in ADNI_SEARCH_DIRS:
        if not os.path.isdir(adni_dir):
            continue
        for subj_id in scan_adni_directory(adni_dir):
            if subj_id not in subject_to_dir:
                subject_to_dir[subj_id] = adni_dir
    return subject_to_dir


def load_labels_from_csv():
    try:
        import pandas as pd
    except ImportError:
        return {}

    csv_paths = glob.glob(
        os.path.join(config.PROJECT_ROOT, "veri", "*.csv")
    )
    labels = {}
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            if "Subject" not in df.columns or "Group" not in df.columns:
                continue
            for _, row in df.iterrows():
                subj = str(row["Subject"]).strip()
                grp  = str(row["Group"]).strip()
                mapped = GROUP_REMAP.get(grp)
                if mapped and SUBJECT_ID_REGEX.match(subj):
                    labels[subj] = mapped
        except Exception:
            continue
    return labels


def detect_protocol(subject_id, adni_dir=None):
    if adni_dir is None:
        adni_dir = config.DICOM_DIR

    subject_dir = os.path.join(adni_dir, subject_id)
    if not os.path.isdir(subject_dir):
        return {
            "protocol": "unknown",
            "series_name": None,
            "tr": None,
            "excluded": True,
            "exclude_reason": "Dizin bulunamadi",
        }

    try:
        series_dirs = [d for d in os.listdir(subject_dir)
                       if os.path.isdir(os.path.join(subject_dir, d))]
    except OSError:
        return {
            "protocol": "unknown",
            "series_name": None,
            "tr": None,
            "excluded": True,
            "exclude_reason": "Dizin okunamadi",
        }

    rsfmri_series = None
    for s in series_dirs:
        s_lower = s.lower()

        if any(excl in s_lower for excl in EXCLUDE_PATTERNS):
            continue

        if any(pat in s_lower for pat in RSFMRI_PATTERNS):
            rsfmri_series = s
            break

    if rsfmri_series is None:
        return {
            "protocol": "unknown",
            "series_name": None,
            "tr": None,
            "excluded": True,
            "exclude_reason": f"rsfMRI serisi bulunamadi. Seriler: {series_dirs}",
        }

    series_lower = rsfmri_series.lower()

    if "_ph_" in series_lower or "phase_map" in series_lower or "phasemap" in series_lower:
        return {
            "protocol": "unknown",
            "series_name": rsfmri_series,
            "tr": None,
            "excluded": True,
            "exclude_reason": "Phase-rekonstruksiyon serisi (magnitude degil)",
        }

    if "_mb_" in series_lower or "multiband" in series_lower:
        protocol = "multiband"
        tr = 0.607
    else:
        protocol = "standard"
        tr = 3.0

    nifti_path = os.path.join(
        config.NIFTI_DIR, subject_id, f"{subject_id}_rsfmri.nii.gz"
    )
    if os.path.exists(nifti_path):
        try:
            import nibabel as nib
            img = nib.load(nifti_path)
            header_tr = float(img.header.get_zooms()[3])
            if header_tr > 0:
                tr = round(header_tr, 3)
        except Exception:
            pass

    return {
        "protocol": protocol,
        "series_name": rsfmri_series,
        "tr": tr,
        "excluded": False,
        "exclude_reason": None,
    }


def discover_all(adni_dir=None):
    if adni_dir is not None:
        subject_to_dir = {
            s: adni_dir for s in scan_adni_directory(adni_dir)
        }
    else:
        subject_to_dir = scan_all_adni_directories()

    standard   = []
    multiband  = []
    excluded   = []
    subject_tr = {}
    details    = {}
    source_dir = {}

    for subj_id, sdir in sorted(subject_to_dir.items()):
        info = detect_protocol(subj_id, sdir)
        details[subj_id]    = info
        source_dir[subj_id] = sdir

        if info["excluded"]:
            excluded.append((subj_id, info["exclude_reason"]))
        elif info["protocol"] == "standard":
            standard.append(subj_id)
            subject_tr[subj_id] = info["tr"]
        elif info["protocol"] == "multiband":
            multiband.append(subj_id)
            subject_tr[subj_id] = info["tr"]
        else:
            excluded.append((subj_id, "Bilinmeyen protokol"))

    all_valid = standard + multiband

    labels = load_labels_from_csv()

    missing = [s for s in all_valid if s not in labels]
    if missing:
        try:
            adni_data = import_module("00b_adni_data")
            dxsum_df  = adni_data.load_diagnosis()
            for sid in missing:
                dx = adni_data.get_subject_diagnosis(dxsum_df, sid)
                if dx:
                    labels[sid] = dx
        except Exception:
            pass

    return {
        "standard":   standard,
        "multiband":  multiband,
        "excluded":   excluded,
        "subject_tr": subject_tr,
        "all_valid":  all_valid,
        "labels":     labels,
        "source_dir": source_dir,
        "details":    details,
    }


def update_config(discovery_result=None):
    if discovery_result is None:
        discovery_result = discover_all()

    d = discovery_result

    config.STANDARD_SUBJECTS = d["standard"]
    config.MB_SUBJECTS        = d["multiband"]
    config.EXCLUDED_SUBJECTS  = [subj for subj, _ in d["excluded"]]
    config.ALL_SUBJECTS       = d["all_valid"]
    config.SUBJECT_TR         = d["subject_tr"]
    config.SUBJECT_LABELS     = d.get("labels", {})

    return d


def build_subjects_list(discovery_result=None):
    if discovery_result is None:
        discovery_result = discover_all()

    d = discovery_result
    label_to_int = {"HC": 0, "MCI": 1, "AD": 2}
    subjects = []

    for subj_id in d["all_valid"]:
        lbl = d["labels"].get(subj_id)
        if lbl is None:
            continue

        nifti_path = os.path.join(
            config.NIFTI_DIR, subj_id, f"{subj_id}_rsfmri.nii.gz"
        )
        proto = "MB" if subj_id in d["multiband"] else "Standard"

        subjects.append({
            "subject_id": subj_id,
            "label":      lbl,
            "diagnosis":  label_to_int.get(lbl, -1),
            "fmri_path":  nifti_path if os.path.exists(nifti_path) else None,
            "tr":         d["subject_tr"].get(subj_id, config.TR),
            "protocol":   proto,
        })

    return subjects


def run(adni_dir=None, verbose=True):
    result = discover_all(adni_dir)
    update_config(result)

    if verbose:
        n_std = len(result["standard"])
        n_mb  = len(result["multiband"])
        n_exc = len(result["excluded"])
        n_tot = n_std + n_mb

        taranan = adni_dir or " + ".join(
            d for d in ADNI_SEARCH_DIRS if os.path.isdir(d)
        )
        print("OTOMATIK OZNE KESFI")
        print(f"  Taranan: {taranan}")
        print(f"  Standart rsfMRI : {n_std}")
        print(f"  Multiband rsfMRI: {n_mb}")
        print(f"  Haric tutulan   : {n_exc}")
        print(f"  Pipeline toplam : {n_tot}")

        grp = {}
        for sid in result["all_valid"]:
            lbl = result["labels"].get(sid)
            if lbl:
                grp[lbl] = grp.get(lbl, 0) + 1
        if grp:
            print("  Grup dagilimi: " +
                  " | ".join(f"{k}:{grp.get(k,0)}" for k in ["HC","MCI","AD"]))

        unlabeled = [s for s in result["all_valid"]
                     if s not in result["labels"]]
        if unlabeled:
            print(f"  Uyari: {len(unlabeled)} oznede etiket yok")

        if result["excluded"] and verbose:
            print("\n  Haric tutulanlar:")
            for subj, reason in result["excluded"][:10]:
                print(f"    {subj}: {reason}")
            if n_exc > 10:
                print(f"    ... ve {n_exc - 10} tane daha")

        print()

    return result


if __name__ == "__main__":
    result = run()

    print("Config guncellendi:")
    print(f"  config.ALL_SUBJECTS       = {len(config.ALL_SUBJECTS)} ozne")
    print(f"  config.STANDARD_SUBJECTS  = {len(config.STANDARD_SUBJECTS)} ozne")
    print(f"  config.MB_SUBJECTS        = {len(config.MB_SUBJECTS)} ozne")
    print(f"  config.EXCLUDED_SUBJECTS  = {len(config.EXCLUDED_SUBJECTS)} ozne")
    print(f"  config.SUBJECT_TR         = {len(config.SUBJECT_TR)} giris")
    print(f"  config.SUBJECT_LABELS     = {len(config.SUBJECT_LABELS)} etiket")
