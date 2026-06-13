"""Project paths, atlas settings, and analysis constants shared across the pipeline."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DICOM_DIR = os.path.join(PROJECT_ROOT, "veri", "ADNI")

NIFTI_DIR = os.path.join(PROJECT_ROOT, "veri", "nifti")

PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed")

METADATA_DIR = os.path.join(PROJECT_ROOT, "veri", "table")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

for d in [NIFTI_DIR, PREPROCESSED_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)


DXSUM_CSV = os.path.join(METADATA_DIR, "Diagnosis", "DXSUM_12Mar2026.csv")

PTDEMOG_CSV = os.path.join(METADATA_DIR, "Subject_Characteristics", "PTDEMOG_12Mar2026.csv")

MMSE_CSV = os.path.join(METADATA_DIR, "Neuropsychological", "MMSE_12Mar2026.csv")

CDR_CSV = os.path.join(METADATA_DIR, "Neuropsychological", "CDR_12Mar2026.csv")

NEUROBAT_CSV = os.path.join(METADATA_DIR, "Neuropsychological", "NEUROBAT_12Mar2026.csv")

REGISTRY_CSV = os.path.join(METADATA_DIR, "Enrollment", "REGISTRY_12Mar2026.csv")

DELMRSCANS_CSV = os.path.join(METADATA_DIR, "Study_Info", "DELMRSCANS_12Mar2026.csv")


STANDARD_SUBJECTS = [
    "003_S_6996", "011_S_6303", "014_S_6765", "022_S_6069",
    "032_S_4429", "032_S_6600", "109_S_6213", "114_S_6113",
    "116_S_6543", "141_S_0767", "168_S_6634", "168_S_6851",
    "011_S_4827", "035_S_7001", "035_S_7120",
    "003_S_6014", "014_S_6076", "068_S_0127", "082_S_6197",
    "003_S_6258", "003_S_6479", "041_S_6731", "109_S_4380",
]

MB_SUBJECTS = [
    "037_S_6216", "037_S_6620", "052_S_4944",
    "941_S_6854", "941_S_6962",
]

EXCLUDED_SUBJECTS = ["057_S_6869", "135_S_6284", "035_S_6927"]

ALL_SUBJECTS = STANDARD_SUBJECTS + MB_SUBJECTS

SUBJECT_TR = {}
for _s in STANDARD_SUBJECTS:
    SUBJECT_TR[_s] = 3.0
for _s in MB_SUBJECTS:
    SUBJECT_TR[_s] = 0.607

ATLAS_NAME = "aal"
N_ROIS = 166

DMN_ROI_INDICES = [
    18, 19,
    148, 149,
    36, 37,
    38, 39,
    40, 41,
    66, 67,
    68, 69,
    84, 85,
    88, 89,
]

CEREBELLAR_ROI_INDICES = list(range(90, 116))


TR = 3.0

FIRST_N_VOLUMES = 10

LOW_PASS = 0.08
HIGH_PASS = 0.01

SMOOTHING_FWHM = 6.0


CONN_KIND = "correlation"

DENSITY_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

THRESHOLD_ABSOLUTE = 0.15

N_NULL_ITERATIONS = 100
NULL_MODELS = ["erdos_renyi", "configuration"]

LOUVAIN_N_REPETITIONS = 100

WINDOW_LENGTH = 50
STEP_SIZE = 10
N_STATES = 4


N_FOLDS = 5

N_REPEATS = 3
OPTUNA_N_TRIALS = 50
OPTUNA_TOP_MODELS = ['LightGBM', 'XGBoost', 'Stacking', 'ElasticNet', 'SVM_RBF']
COMBAT_BATCH_COL = 'protocol'
USE_COMBAT = True

ATLASES = {
    'AAL3': {'n_rois': 166, 'fetcher': None},
    'Schaefer200': {'n_rois': 200, 'fetcher': 'schaefer_2018',
                    'yeo_networks': 7, 'resolution_mm': 2},
    'HO48': {'n_rois': 48, 'fetcher': 'harvard_oxford', 'atlas_name': 'cort-maxprob-thr25-2mm'},
}
DEFAULT_ATLAS = 'AAL3'
MULTI_ATLAS_ENABLED = True

ALFF_BAND = (0.01, 0.08)

REHO_K = 27

NBS_THRESH = 3.1
NBS_N_PERM = 100
NBS_ALPHA = 0.05

PERM_N_ITER = 1000
PERM_N_REPEATS_FOR_CV = 3

RANDOM_STATE = 42

TEST_SIZE = 0.2

TARGET_FEATURE_DIM = (50, 70)

GROUP_LABELS = {0: "HC", 1: "MCI", 2: "AD"}
GROUP_COLORS = {
    "HC": "#2196F3",
    "MCI": "#FF9800",
    "AD": "#F44336",
}

DIAGNOSIS_MAP = {1: "HC", 2: "MCI", 3: "AD"}
LABEL_MAP = {"HC": 0, "MCI": 1, "AD": 2}

print("Config yuklendi.")
print(f"  Atlas: {ATLAS_NAME} ({N_ROIS} ROI)")
print(f"  Kullanilacak ozne: {len(ALL_SUBJECTS)} "
      f"(standart: {len(STANDARD_SUBJECTS)}, MB: {len(MB_SUBJECTS)})")
print(f"  Density plaji: {DENSITY_RANGE[0]}-{DENSITY_RANGE[-1]}")
