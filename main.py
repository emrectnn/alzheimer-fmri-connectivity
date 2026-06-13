"""Pipeline entry point.

Runs the full analysis end to end: DICOM conversion, metadata matching, preprocessing,
connectivity, graph metrics, null models, statistics, and classification.

Usage:
    python main.py                # full pipeline
    python main.py --convert-only # DICOM -> NIfTI only
    python main.py --demo         # quick test on the nilearn demo dataset
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import argparse
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)


def run_demo():
    """Run the pipeline on the small nilearn demo dataset."""
    from importlib import import_module
    config = import_module("00a_config")
    preprocess = import_module("01_preprocess")
    connectivity = import_module("02a_connectivity")
    graph_metrics = import_module("03_graph_metrics")
    null_models = import_module("04_null_models")
    classification = import_module("06a_classification")
    statistics = import_module("07_statistics")

    from nilearn import datasets

    print("DEMO MOD -- Nilearn ADHD verisi ile pipeline testi")

    print("\n[1/6] Demo veri indiriliyor...")
    adhd = datasets.fetch_adhd(n_subjects=10)

    atlas_maps, labels = preprocess.load_atlas()

    masker = preprocess.create_masker(atlas_maps, tr=2.5)

    print("\n[2/6] On isleme...")
    subjects = []
    for i, fmri_file in enumerate(adhd.func):
        subj_id = f"demo_{i:03d}"
        group = "HC" if i < 5 else "MCI"
        label = 0 if group == "HC" else 1

        try:
            ts, info = preprocess.preprocess_subject(
                fmri_file, masker, check_quality=False
            )
            save_path = os.path.join(config.PREPROCESSED_DIR, f"{subj_id}_timeseries.npy")
            np.save(save_path, ts)
            subjects.append({
                "id": subj_id, "fmri_path": fmri_file,
                "group": group, "label": label,
                "timeseries_path": save_path,
            })
            print(f"  OK {subj_id} ({group}): {ts.shape}")
        except Exception as e:
            print(f"  FAIL {subj_id}: {e}")

    print("\n[3/6] Baglanti matrisleri...")
    matrices = connectivity.compute_all_matrices(subjects, save=True)

    print("\n[4/6] Graf metrikleri...")
    global_df, nodal_list = graph_metrics.compute_all_subjects(matrices)

    print("\n[5/6] Istatistiksel karsilastirma...")
    stats_df = statistics.report_all_metrics(global_df)
    statistics.print_summary(stats_df)

    print("\n[6/6] Siniflandirma...")
    try:
        cv_results, X, y, fnames = classification.run_classification(
            global_df, nodal_list
        )
    except Exception as e:
        print(f"  Siniflandirma hatasi: {e}")

    print("DEMO TAMAMLANDI")
    print(f"Sonuclar: {config.RESULTS_DIR}")


def run_full_pipeline(convert_only=False):
    """Run the full ADNI pipeline from DICOM conversion to classification.

    Args:
        convert_only: Stop after the DICOM-to-NIfTI step when True.

    Returns:
        None.
    """
    from importlib import import_module
    config = import_module("00a_config")
    dicom2nifti = import_module("00c_dicom2nifti")
    adni_data = import_module("00d_adni_data")
    preprocess = import_module("01_preprocess")
    connectivity = import_module("02a_connectivity")
    graph_metrics = import_module("03_graph_metrics")
    null_models = import_module("04_null_models")
    classification = import_module("06a_classification")
    statistics = import_module("07_statistics")

    print("TAM PIPELINE -- ADNI VERISI")

    discover = import_module("00b_discover")
    try:
        discover.update_config()
        print(f"  [OK] Discover (pre-DICOM): {len(config.ALL_SUBJECTS)} denek "
              f"bulundu (Standard={len(config.STANDARD_SUBJECTS)}, "
              f"MB={len(config.MB_SUBJECTS)})")
    except Exception as e:
        print(f"  [WARN] Discover basarisiz, hardcoded liste kullanilacak: {e}")

    print("\n[0/7] DICOM -> NIfTI donusumu...")
    dicom2nifti.convert_all(config.ALL_SUBJECTS)

    if convert_only:
        print("\n--convert-only: Sadece DICOM donusumu yapildi.")
        return

    print("\n[1/7] ADNI veri hazirlama...")

    discover = import_module("00b_discover")
    try:
        discover.update_config()
        print(f"  [OK] Discover: {len(config.ALL_SUBJECTS)} denek bulundu "
              f"(Standard={len(config.STANDARD_SUBJECTS)}, MB={len(config.MB_SUBJECTS)})")
    except Exception as e:
        print(f"  [WARN] Discover basarisiz, hardcoded liste kullanilacak: {e}")

    subjects = adni_data.prepare_subjects(config.ALL_SUBJECTS)
    if subjects is None:
        print("Veri bulunamadi veya NIfTI dosyalari eksik.")
        print("Once DICOM donusumunu calistirin: python main.py --convert-only")
        return

    print("\n[2/7] On isleme...")
    subjects = preprocess.preprocess_all(subjects)

    print("\n[3/7] Baglanti matrisleri...")
    matrices = connectivity.compute_all_matrices(subjects)

    try:
        alff_reho = import_module("02b_alff_reho")
        print("\n[3b/7] ALFF / fALFF / ReHo...")
        alff_reho.main(atlas_name='AAL3')
    except Exception as e:
        print(f"[WARN] ALFF/ReHo hesaplanamadi: {e}")

    if getattr(config, 'MULTI_ATLAS_ENABLED', False):
        try:
            multi_atlas = import_module("02c_multiatlas")
            print("\n[3c/7] Multi-atlas (Schaefer200 + HO48)...")
            for atlas_key in ('Schaefer200', 'HO48'):
                try:
                    multi_atlas.run_atlas(atlas_key, skip_existing=True)
                except Exception as e:
                    print(f"[WARN] {atlas_key} atlas hesaplanamadi: {e}")
        except Exception as e:
            print(f"[WARN] multi-atlas modulu yuklenemedi: {e}")

    print("\n[4/7] Graf metrikleri (AUC multi-seuils dahil)...")
    global_df, nodal_list = graph_metrics.compute_all_subjects(matrices)

    print("\n[5/7] Null model analizi (Rossi et al. 2024)...")
    null_df = null_models.null_model_analysis_all(matrices, model="erdos_renyi")

    print("\n[6/7] Istatistiksel karsilastirma...")
    stats_df = statistics.report_all_metrics(global_df)
    statistics.print_summary(stats_df)
    stats_df.to_csv(os.path.join(config.METRICS_DIR, "statistical_tests.csv"), index=False)

    print("\n[7/7] ML Siniflandirma...")
    enhanced = import_module("08e_run")
    try:
        enhanced.run_enhanced_classification()
    except Exception as e:
        import traceback
        print(f"[ERR] Enhanced classification basarisiz: {e}")
        traceback.print_exc()
        print("[WARN] Falling back to 06_classification baseline")
        cv_results, X, y, fnames = classification.run_classification(
            global_df, nodal_list, null_df
        )

    try:
        lb4 = import_module("qc.build_leaderboard")
        print("\n[8a/7] Leaderboard...")
        lb4.main()
    except Exception as e:
        print(f"[WARN] leaderboard could not be generated: {e}")

    if os.environ.get('SKIP_PERMUTATION', '0') != '1':
        try:
            perm = import_module("qc.permutation_test")
            print("\n[8b/7] Permutation test (top-1 / imaging modes)...")
            perm.main()
        except SystemExit:
            pass
        except Exception as e:
            print(f"[WARN] Permutation test calistirilamadi: {e}")

    print("PIPELINE TAMAMLANDI")
    print(f"Sonuclar: {config.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Connectivity Analysis Pipeline")
    parser.add_argument("--demo", action="store_true", help="Demo veri ile test")
    parser.add_argument("--convert-only", action="store_true",
                       help="Sadece DICOM -> NIfTI donusumu yap")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_full_pipeline(convert_only=args.convert_only)
