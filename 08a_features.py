"""Feature loading and feature-set definitions for the classification sweep.

Loads the merged per-subject feature tables and defines the named feature sets
(imaging-only, +demographics, +clinical) together with the residualize/ComBat-aware
feature-matrix preparation consumed by the experiment functions.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
config = import_module("00a_config")

try:
    import neuroHarmonize  # noqa: F401
    HAS_NH = True
except ImportError:
    HAS_NH = False


def load_all_features():
    """Load and merge all feature tables into a single DataFrame."""
    gm = pd.read_csv(os.path.join(config.METRICS_DIR, "global_metrics.csv"))

    null = pd.read_csv(os.path.join(config.METRICS_DIR, "null_model_erdos_renyi.csv"))

    meta = pd.read_csv(os.path.join(config.NIFTI_DIR, "subject_metadata.csv"))

    df = gm.merge(null[['subject_id','dev_clustering','dev_path_length',
                          'dev_global_eff','dev_sigma']], on='subject_id', how='left')
    df = df.merge(meta[['subject_id','mmse','cdrsb','cdglobal','age',
                          'gender','education']], on='subject_id', how='left')

    df['gender_bin'] = (df['gender'] == 'Male').astype(float)

    motion_qc_path = os.path.join(config.METRICS_DIR, "motion_qc.csv")
    if os.path.exists(motion_qc_path):
        try:
            mq = pd.read_csv(motion_qc_path)
            if 'subject_id' in mq.columns and 'mean_fd' in mq.columns:
                df = df.merge(mq[['subject_id', 'mean_fd']],
                              on='subject_id', how='left')
        except Exception as e:
            print(f"[WARN] motion_qc.csv okunamadi: {e}")

    for kind in ('alff', 'reho'):
        cand = os.path.join(config.RESULTS_DIR, kind, f"{kind}_AAL3.csv")
        if os.path.exists(cand):
            try:
                d = pd.read_csv(cand)
                d = d.rename(columns={c: f"{kind}_{c}" for c in d.columns
                                      if c != 'subject_id'})
                df = df.merge(d, on='subject_id', how='left')
            except Exception as e:
                print(f"[WARN] {cand} okunamadi: {e}")

    for atlas_key, prefix in (('Schaefer200', 'scha200_'), ('HO48', 'ho48_')):
        cand = os.path.join(config.METRICS_DIR, f"global_{atlas_key}.csv")
        if os.path.exists(cand):
            try:
                d = pd.read_csv(cand, dtype={'subject_id': str})
                if 'group' in d.columns:
                    d = d.drop(columns=['group'])
                d = d.rename(columns={c: f"{prefix}{c}" for c in d.columns
                                      if c != 'subject_id'})
                df = df.merge(d, on='subject_id', how='left')
            except Exception as e:
                print(f"[WARN] {cand} okunamadi: {e}")


    return df


def get_feature_sets(df):
    """Return the raw feature-set definitions keyed by name."""
    graph_cols = [c for c in df.columns if c not in
                  ['subject_id','label','group','n_nodes','n_nodes_lc',
                   'n_edges','density','C_rand','L_rand','auc_n_nodes',
                   'auc_n_nodes_lc','auc_n_edges','auc_density',
                   'auc_C_rand','auc_L_rand','gender','age','education',
                   'mmse','cdrsb','cdglobal','gender_bin',
                   'dev_clustering','dev_path_length','dev_global_eff','dev_sigma']]

    auc_cols = [c for c in graph_cols if c.startswith('auc_')]

    clinical_cols = ['mmse','cdrsb','cdglobal','age','gender_bin','education']

    null_cols = ['dev_clustering','dev_path_length','dev_global_eff','dev_sigma']

    feature_sets = {
        'Graf_AUC': auc_cols,
        'Graf_Tam': graph_cols,
        'Klinik': clinical_cols,
        'Null_Model': null_cols,
        'Graf_AUC+Klinik': auc_cols + clinical_cols,
        'Graf_Tam+Klinik': graph_cols + clinical_cols,
        'Graf_AUC+Null': auc_cols + null_cols,
        'Hepsi': graph_cols + clinical_cols + null_cols,
    }

    for name, cols in feature_sets.items():
        feature_sets[name] = [c for c in cols if c in df.columns]

    return feature_sets


def get_feature_sets_by_mode(df, mode):
    """Return feature-set definitions for a given mode (imaging / +demographics / +clinical)."""
    base = get_feature_sets(df)
    graph_only = base['Graf_Tam']
    auc_only = base['Graf_AUC']
    null_only = base['Null_Model']
    demographics = [c for c in ['age', 'gender_bin', 'education'] if c in df.columns]
    clinical_only = [c for c in ['mmse', 'cdrsb', 'cdglobal'] if c in df.columns]

    alff_cols = [c for c in df.columns if c.startswith('alff_roi_')]
    reho_cols = [c for c in df.columns if c.startswith('reho_roi_')]
    scha200_cols = [c for c in df.columns if c.startswith('scha200_')]
    ho48_cols = [c for c in df.columns if c.startswith('ho48_')]

    if mode == 'imaging_only':
        fs = {
            'Graf_AUC': auc_only,
            'Graf_Tam': graph_only,
            'Null_Model': null_only,
            'Graf_AUC+Null': auc_only + null_only,
            'Graf_Tam+Null': graph_only + null_only,
        }
        if alff_cols:
            fs['ALFF'] = alff_cols
        if reho_cols:
            fs['ReHo'] = reho_cols
        if alff_cols and reho_cols:
            fs['ALFF+ReHo'] = alff_cols + reho_cols
            fs['Graf_Tam+ALFF+ReHo'] = graph_only + alff_cols + reho_cols
        if scha200_cols:
            fs['Graf_Schaefer200'] = scha200_cols
        if ho48_cols:
            fs['Graf_HO48'] = ho48_cols
        if scha200_cols and ho48_cols:
            fs['Graf_MultiAtlas'] = graph_only + scha200_cols + ho48_cols
        return fs
    elif mode == 'imaging_plus_demographics':
        fs = {
            'Graf_AUC+Demog': auc_only + demographics,
            'Graf_Tam+Demog': graph_only + demographics,
            'Graf_AUC+Null+Demog': auc_only + null_only + demographics,
            'Graf_Tam+Null+Demog': graph_only + null_only + demographics,
        }
        if scha200_cols and ho48_cols:
            fs['Graf_MultiAtlas+Demog'] = (graph_only + scha200_cols
                                           + ho48_cols + demographics)
        if alff_cols and reho_cols:
            fs['Graf_Tam+ALFF+ReHo+Demog'] = (graph_only + alff_cols
                                              + reho_cols + demographics)
        return fs
    elif mode == 'imaging_plus_clinical':
        full_clinical = clinical_only + demographics
        return {
            'Graf_AUC+Klinik': auc_only + full_clinical,
            'Graf_Tam+Klinik': graph_only + full_clinical,
            'Graf_AUC+Null+Klinik': auc_only + null_only + full_clinical,
            'Hepsi': graph_only + null_only + full_clinical,
        }
    else:
        raise ValueError(
            f"Bilinmeyen mode: {mode!r}. Beklenen: 'imaging_only', "
            "'imaging_plus_demographics', 'imaging_plus_clinical'."
        )


def _prepare_feature_matrix(df, cols, residualize, mode, use_combat=False):
    """Assemble X with optional confound and ComBat site columns appended."""
    X_feat = df[cols].values.astype(float)
    n_conf = 0
    n_bio = 0
    n_site = 0
    parts = [X_feat]

    if residualize and mode != 'imaging_plus_clinical':
        conf_cols = [c for c in ('age', 'gender_bin', 'mean_fd')
                     if c in df.columns and c not in cols]
        if conf_cols:
            parts.append(df[conf_cols].values.astype(float))
            n_conf = len(conf_cols)

    if use_combat and HAS_NH and mode != 'imaging_plus_clinical':
        if 'protocol' in df.columns:
            site_vals = (df['protocol'].astype(str).str.strip() == 'MB').astype(int).values
            if len(np.unique(site_vals)) >= 2:
                bio_cols = []
                if not residualize:
                    for c in ('age', 'gender_bin'):
                        if c in df.columns and c not in cols:
                            bio_cols.append(c)
                if bio_cols:
                    parts.append(df[bio_cols].values.astype(float))
                    n_bio = len(bio_cols)
                parts.append(site_vals.reshape(-1, 1).astype(float))
                n_site = 1

    if len(parts) == 1:
        return X_feat, 0, 0, 0
    X_stacked = np.concatenate(parts, axis=1)
    return X_stacked, n_conf, n_bio, n_site
