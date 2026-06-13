"""Match NIfTI scans with ADNI clinical metadata and build the labelled subject list."""

import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")


def load_diagnosis(csv_path=None):
    """Load the ADNI diagnosis summary table (DXSUM).

    Args:
        csv_path: Optional path override; defaults to the configured location.

    Returns:
        The diagnosis table as a DataFrame.
    """
    if csv_path is None:
        csv_path = config.DXSUM_CSV

    if not os.path.exists(csv_path):
        print(f"[!] DXSUM bulunamadi: {csv_path}")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    df["DIAGNOSIS"] = pd.to_numeric(df["DIAGNOSIS"], errors="coerce")
    df = df.dropna(subset=["DIAGNOSIS"])
    df["DIAGNOSIS"] = df["DIAGNOSIS"].astype(int)

    print(f"DXSUM yuklendi: {len(df)} satir, {df['PTID'].nunique()} benzersiz ozne")
    return df


def get_subject_diagnosis(dxsum_df, subject_id):
    """Return the HC/MCI/AD group for one subject.

    Args:
        dxsum_df: Diagnosis table from load_diagnosis().
        subject_id: ADNI subject identifier.

    Returns:
        Group label string, or None if unavailable.
    """
    subj = dxsum_df[dxsum_df["PTID"] == subject_id]
    if subj.empty:
        return None, None, None

    bl = subj[subj["VISCODE"] == "bl"]
    if not bl.empty:
        row = bl.iloc[0]
    else:
        sc = subj[subj["VISCODE"] == "sc"]
        if not sc.empty:
            row = sc.iloc[0]
        else:
            row = subj.iloc[0]

    dx_code = int(row["DIAGNOSIS"])
    group = config.DIAGNOSIS_MAP.get(dx_code, "Unknown")
    examdate = row.get("EXAMDATE", None)
    return dx_code, group, examdate


def load_demographics(csv_path=None):
    """Load the ADNI demographics table.

    Args:
        csv_path: Optional path override.

    Returns:
        The demographics table as a DataFrame.
    """
    if csv_path is None:
        csv_path = config.PTDEMOG_CSV

    if not os.path.exists(csv_path):
        print(f"[!] PTDEMOG bulunamadi: {csv_path}")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"PTDEMOG yuklendi: {len(df)} satir")
    return df


def get_subject_demographics(demog_df, subject_id, exam_year=None):
    """Return age, sex and education for one subject.

    Args:
        demog_df: Demographics table.
        subject_id: ADNI subject identifier.
        exam_year: Optional exam year for age at scan.

    Returns:
        Dict with age, sex and education (values may be None).
    """
    subj = demog_df[demog_df["PTID"] == subject_id]
    if subj.empty:
        return {}

    sc = subj[subj["VISCODE"].isin(["sc", "bl"])]
    row = sc.iloc[0] if not sc.empty else subj.iloc[0]

    gender_code = pd.to_numeric(row.get("PTGENDER"), errors="coerce")
    gender = "M" if gender_code == 1 else ("F" if gender_code == 2 else "?")

    birth_year = pd.to_numeric(row.get("PTDOBYY"), errors="coerce")
    education = pd.to_numeric(row.get("PTEDUCAT"), errors="coerce")

    age = None
    if pd.notna(birth_year) and exam_year:
        try:
            age = int(exam_year) - int(birth_year)
        except (ValueError, TypeError):
            pass

    return {
        "gender": gender,
        "birth_year": int(birth_year) if pd.notna(birth_year) else None,
        "age": age,
        "education": int(education) if pd.notna(education) else None,
    }


def load_clinical_scores(mmse_path=None, cdr_path=None):
    """Load the MMSE and CDR clinical-score tables.

    Args:
        mmse_path: Optional MMSE path override.
        cdr_path: Optional CDR path override.

    Returns:
        Tuple (mmse_df, cdr_df).
    """
    mmse_df = None
    cdr_df = None

    if mmse_path is None:
        mmse_path = config.MMSE_CSV
    if cdr_path is None:
        cdr_path = config.CDR_CSV

    if os.path.exists(mmse_path):
        mmse_df = pd.read_csv(mmse_path, low_memory=False)
        print(f"MMSE yuklendi: {len(mmse_df)} satir")

    if os.path.exists(cdr_path):
        cdr_df = pd.read_csv(cdr_path, low_memory=False)
        print(f"CDR yuklendi: {len(cdr_df)} satir")

    return mmse_df, cdr_df


def get_subject_scores(mmse_df, cdr_df, subject_id):
    """Return the MMSE and CDR scores for one subject.

    Args:
        mmse_df: MMSE table.
        cdr_df: CDR table.
        subject_id: ADNI subject identifier.

    Returns:
        Dict with the available clinical scores.
    """
    scores = {}

    if mmse_df is not None:
        subj = mmse_df[mmse_df["PTID"] == subject_id]
        bl = subj[subj["VISCODE"].isin(["bl", "sc"])]
        row = bl.iloc[0] if not bl.empty else (subj.iloc[0] if not subj.empty else None)
        if row is not None:
            mmscore = pd.to_numeric(row.get("MMSCORE"), errors="coerce")
            scores["mmse"] = float(mmscore) if pd.notna(mmscore) else None

    if cdr_df is not None:
        subj = cdr_df[cdr_df["PTID"] == subject_id]
        bl = subj[subj["VISCODE"].isin(["bl", "sc"])]
        row = bl.iloc[0] if not bl.empty else (subj.iloc[0] if not subj.empty else None)
        if row is not None:
            cdrsb = pd.to_numeric(row.get("CDRSB"), errors="coerce")
            cdglobal = pd.to_numeric(row.get("CDGLOBAL"), errors="coerce")
            scores["cdrsb"] = float(cdrsb) if pd.notna(cdrsb) else None
            scores["cdglobal"] = float(cdglobal) if pd.notna(cdglobal) else None

    return scores


def load_deleted_scans(csv_path=None):
    """Load the list of scans excluded after quality control.

    Args:
        csv_path: Optional path override.

    Returns:
        Set of excluded subject identifiers.
    """
    if csv_path is None:
        csv_path = config.DELMRSCANS_CSV

    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path, low_memory=False)
    deleted_subjects = set(df["SUBJECTID"].unique())
    print(f"DELMRSCANS: {len(deleted_subjects)} oznenin taramasi silinmis")
    return deleted_subjects


def build_subject_metadata(subject_list=None):
    """Build the subject metadata table from the ADNI CSV tables.

    Args:
        subject_list: Optional subset of subjects; defaults to all discovered.

    Returns:
        DataFrame with diagnosis, demographics and clinical scores per subject.
    """
    if subject_list is None:
        subject_list = config.ALL_SUBJECTS

    print("OZNE METADATA OLUSTURMA")

    dxsum_df = load_diagnosis()
    demog_df = load_demographics()
    mmse_df, cdr_df = load_clinical_scores()
    deleted = load_deleted_scans()

    if dxsum_df is None:
        print("[!] DXSUM yuklenemedi, devam edilemiyor.")
        return None

    records = []
    print(f"\n{len(subject_list)} ozne icin metadata toplaniyor...")

    for subj_id in subject_list:
        dx_code, group, examdate = get_subject_diagnosis(dxsum_df, subj_id)

        if dx_code is None:
            print(f"  [!] {subj_id}: DXSUM'da bulunamadi")
            continue

        exam_year = None
        if examdate and isinstance(examdate, str) and len(examdate) >= 4:
            exam_year = examdate[:4]

        demo = {}
        if demog_df is not None:
            demo = get_subject_demographics(demog_df, subj_id, exam_year)

        scores = get_subject_scores(mmse_df, cdr_df, subj_id)

        is_deleted = subj_id in deleted

        tr = config.SUBJECT_TR.get(subj_id, config.TR)
        protocol = "MB" if subj_id in config.MB_SUBJECTS else "Standard"

        record = {
            "subject_id": subj_id,
            "diagnosis": dx_code,
            "group": group,
            "label": config.LABEL_MAP.get(group, -1),
            "age": demo.get("age"),
            "gender": demo.get("gender"),
            "education": demo.get("education"),
            "mmse": scores.get("mmse"),
            "cdrsb": scores.get("cdrsb"),
            "cdglobal": scores.get("cdglobal"),
            "tr": tr,
            "protocol": protocol,
            "scan_deleted": is_deleted,
            "examdate": examdate,
        }
        records.append(record)

        status = "[SILINMIS]" if is_deleted else ""
        print(f"  {subj_id}: {group} | yas={demo.get('age', '?')} | "
              f"mmse={scores.get('mmse', '?')} | {protocol} {status}")

    metadata_df = pd.DataFrame(records)

    print("\n--- Grup Dagilimi ---")
    for grp in ["HC", "MCI", "AD"]:
        n = len(metadata_df[metadata_df["group"] == grp])
        print(f"  {grp}: {n}")

    if "age" in metadata_df.columns:
        valid_ages = metadata_df["age"].dropna()
        if len(valid_ages) > 0:
            print(f"\nYas: {valid_ages.mean():.1f} +/- {valid_ages.std():.1f}")

    save_path = os.path.join(config.NIFTI_DIR, "subject_metadata.csv")
    metadata_df.to_csv(save_path, index=False)
    print(f"\nMetadata kaydedildi: {save_path}")

    return metadata_df


def match_nifti_files(metadata_df):
    """Attach the matching NIfTI path to each subject row.

    Args:
        metadata_df: Subject metadata table.

    Returns:
        The same table with an added fmri_path column.
    """
    matched = []
    unmatched = []

    for _, row in metadata_df.iterrows():
        subj_id = row["subject_id"]
        nifti_dir = os.path.join(config.NIFTI_DIR, subj_id)

        nifti_files = glob.glob(os.path.join(nifti_dir, "*.nii.gz"))
        if nifti_files:
            row_dict = row.to_dict()
            row_dict["fmri_path"] = nifti_files[0]
            matched.append(row_dict)
        else:
            unmatched.append(subj_id)

    print(f"\nNIfTI eslestirme: {len(matched)} / {len(metadata_df)}")
    if unmatched:
        print(f"  Eslesmeyen: {unmatched}")
        print("  -> Once 00a_dicom2nifti.py calistirilmali")

    return matched


def validate_nifti(fmri_path, min_timepoints=50):
    """Check that a NIfTI exists and has enough time points.

    Args:
        fmri_path: Path to the subject's NIfTI.
        min_timepoints: Minimum acceptable number of volumes.

    Returns:
        True if the file is usable.
    """
    try:
        import nibabel as nib
        img = nib.load(fmri_path)
        shape = img.shape

        if len(shape) != 4:
            return False, f"4D degil: shape={shape}"

        n_timepoints = shape[3]
        if n_timepoints < min_timepoints:
            return False, f"Yetersiz TR: {n_timepoints} < {min_timepoints}"

        header = img.header
        tr = header.get_zooms()[3] if len(header.get_zooms()) > 3 else None

        return True, {
            "shape": shape,
            "n_timepoints": n_timepoints,
            "voxel_size": header.get_zooms()[:3],
            "tr": tr,
        }

    except Exception as e:
        return False, str(e)


def validate_all_subjects(subjects):
    """Drop subjects whose NIfTI is missing or too short.

    Args:
        subjects: List of subject dicts.

    Returns:
        The filtered list of valid subjects.
    """
    from collections import Counter

    valid = []
    invalid = []

    for subj in subjects:
        ok, info = validate_nifti(subj["fmri_path"])
        if ok:
            subj["nifti_info"] = info
            valid.append(subj)
        else:
            print(f"  [!] {subj['subject_id']}: {info}")
            invalid.append(subj)

    print(f"\nGecerli: {len(valid)}, Gecersiz: {len(invalid)}")

    groups = Counter(s["group"] for s in valid)
    for g in ["HC", "MCI", "AD"]:
        print(f"  {g}: {groups.get(g, 0)}")

    return valid


def prepare_subjects(subject_list=None):
    """Build, validate and return the final list of analyzable subjects.

    Args:
        subject_list: Optional subset; defaults to all discovered subjects.

    Returns:
        List of valid subject dicts, or None if none are usable.
    """
    metadata_df = build_subject_metadata(subject_list)
    if metadata_df is None or metadata_df.empty:
        return None

    subjects = match_nifti_files(metadata_df)
    if not subjects:
        print("\n[!] Hic NIfTI dosyasi eslesmedi.")
        print("    Once calistirin: python 00a_dicom2nifti.py")
        return None

    subjects = validate_all_subjects(subjects)

    for s in subjects:
        s["id"] = s["subject_id"]

    print(f"\nHazir: {len(subjects)} ozne pipeline'a girilebilir.")
    return subjects


if __name__ == "__main__":
    print("\n--- ADNI Veri Hazirlama ---\n")
    subjects = prepare_subjects()

    if subjects:
        print(f"\n{len(subjects)} ozne hazir.")
        for s in subjects:
            print(f"  {s['id']}: {s['group']} | {s.get('fmri_path', 'N/A')}")
    else:
        print("\nVeri hazirlama tamamlanamadi.")
        print("1. Once DICOM donusumu yapin: python 00a_dicom2nifti.py")
        print("2. Sonra tekrar calistirin: python 00b_adni_data.py")
