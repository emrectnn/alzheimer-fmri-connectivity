"""Convert raw ADNI DICOM series into NIfTI volumes."""

import os
import sys
import glob
import subprocess
import shutil
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")


EXCLUDE_SERIES = ["MoCoSeries", "Perfusion_Weighted", "relCBF", "phase"]


def check_dcm2niix():
    """Return True if the dcm2niix converter is available on PATH."""
    if shutil.which("dcm2niix") is None:
        print("[!] dcm2niix bulunamadi!")
        print("    Kurulum:")
        print("    conda install -c conda-forge dcm2niix")
        print("    veya: https://github.com/rordenlab/dcm2niix/releases")
        return False
    result = subprocess.run(["dcm2niix", "-v"], capture_output=True, text=True)
    version = result.stdout.strip() or result.stderr.strip()
    print(f"dcm2niix bulundu: {version[:80]}")
    return True


def find_rsfmri_dicom_dir(subject_id):
    """Locate a subject's resting-state fMRI DICOM folder."""
    subject_dir = os.path.join(config.DICOM_DIR, subject_id)
    if not os.path.isdir(subject_dir):
        return None, f"Dizin bulunamadi: {subject_dir}"

    series_dirs = [d for d in os.listdir(subject_dir)
                   if os.path.isdir(os.path.join(subject_dir, d))]

    rsfmri_series = None
    for s in series_dirs:
        s_lower = s.lower()
        if "rsfmri" not in s_lower and "fcmri" not in s_lower:
            continue
        excluded = False
        for excl in EXCLUDE_SERIES:
            if excl.lower() in s_lower:
                if excl.lower() == "phase" and ("rsfmri" in s_lower or "fcmri" in s_lower):
                    continue
                excluded = True
                break
        if excluded:
            continue
        rsfmri_series = s
        break

    if rsfmri_series is None:
        return None, f"rsfMRI serisi bulunamadi. Mevcut seriler: {series_dirs}"

    series_path = os.path.join(subject_dir, rsfmri_series)

    date_dirs = sorted([d for d in os.listdir(series_path)
                        if os.path.isdir(os.path.join(series_path, d))])
    if not date_dirs:
        return None, f"Tarih dizini bulunamadi: {series_path}"

    date_path = os.path.join(series_path, date_dirs[-1])

    image_dirs = sorted([d for d in os.listdir(date_path)
                         if os.path.isdir(os.path.join(date_path, d))])
    if not image_dirs:
        return None, f"ImageID dizini bulunamadi: {date_path}"

    dicom_dir = os.path.join(date_path, image_dirs[-1])

    dcm_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dcm_files:
        return None, f"DICOM dosyasi bulunamadi: {dicom_dir}"

    return dicom_dir, f"Bulundu: {rsfmri_series} ({len(dcm_files)} dcm)"


def convert_subject(subject_id):
    """Convert one subject's resting-state DICOM series to NIfTI."""
    dicom_dir, msg = find_rsfmri_dicom_dir(subject_id)
    if dicom_dir is None:
        print(f"  [!] {subject_id}: {msg}")
        return None

    print(f"  {subject_id}: {msg}")

    output_dir = os.path.join(config.NIFTI_DIR, subject_id)
    os.makedirs(output_dir, exist_ok=True)

    existing = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if existing:
        print(f"    Zaten mevcut: {os.path.basename(existing[0])}")
        return existing[0]

    cmd = [
        "dcm2niix",
        "-z", "y",
        "-f", f"{subject_id}_rsfmri",
        "-o", output_dir,
        "-b", "n",
        dicom_dir,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"    [!] dcm2niix hatasi: {result.stderr[:200]}")
            return None

        nifti_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
        if not nifti_files:
            print("    [!] NIfTI dosyasi olusturulamadi")
            return None

        nifti_path = nifti_files[0]
        print(f"    Donusturuldu: {os.path.basename(nifti_path)}")
        return nifti_path

    except subprocess.TimeoutExpired:
        print("    [!] Zaman asimi (300s)")
        return None
    except Exception as e:
        print(f"    [!] Hata: {e}")
        return None


def verify_conversion(nifti_path, subject_id):
    """Check that a converted NIfTI looks valid (4D with enough volumes)."""
    try:
        import nibabel as nib
        img = nib.load(nifti_path)
        shape = img.shape
        zooms = img.header.get_zooms()

        is_4d = len(shape) == 4
        tr = float(zooms[3]) if len(zooms) > 3 else None

        expected_tr = config.SUBJECT_TR.get(subject_id, config.TR)

        info = {
            "subject_id": subject_id,
            "nifti_path": nifti_path,
            "shape": str(shape),
            "n_volumes": shape[3] if is_4d else 0,
            "voxel_size": str(zooms[:3]),
            "tr_header": tr,
            "tr_expected": expected_tr,
            "valid": is_4d,
        }

        if is_4d:
            print(f"    Dogrulandi: shape={shape}, TR={tr:.3f}s")
        else:
            print(f"    [!] 4D degil: shape={shape}")

        return info

    except Exception as e:
        print(f"    [!] Dogrulama hatasi: {e}")
        return {"subject_id": subject_id, "valid": False, "error": str(e)}


def convert_all(subject_list=None):
    """Convert every subject's DICOM series to NIfTI."""
    if subject_list is None:
        subject_list = config.ALL_SUBJECTS

    print(f"DICOM -> NIfTI DONUSUM ({len(subject_list)} ozne)")

    if not check_dcm2niix():
        return None

    manifest = []
    success = 0
    failed = 0

    for i, subj_id in enumerate(subject_list):
        print(f"\n[{i+1}/{len(subject_list)}] {subj_id}")

        nifti_path = convert_subject(subj_id)
        if nifti_path is None:
            failed += 1
            manifest.append({
                "subject_id": subj_id, "nifti_path": "", "valid": False
            })
            continue

        info = verify_conversion(nifti_path, subj_id)
        manifest.append(info)

        if info.get("valid", False):
            success += 1
        else:
            failed += 1

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(config.NIFTI_DIR, "conversion_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print("DONUSUM TAMAMLANDI")
    print(f"  Basarili: {success} / {len(subject_list)}")
    print(f"  Basarisiz: {failed}")
    print(f"  Manifest: {manifest_path}")

    return manifest_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DICOM -> NIfTI Donusum")
    parser.add_argument("--subject", type=str, default=None,
                       help="Tek ozne donustur (ornek: 003_S_6996)")
    args = parser.parse_args()

    if args.subject:
        if not check_dcm2niix():
            sys.exit(1)
        nifti = convert_subject(args.subject)
        if nifti:
            verify_conversion(nifti, args.subject)
    else:
        convert_all()
