# Alzheimer fMRI Brain Connectivity Analysis

Resting-state fMRI functional connectivity and machine-learning classification
(HC / MCI / AD) on the ADNI cohort. Graph-theoretic metrics, tangent-space
embedding, null-model deviation features, ComBat site harmonization and an
Optuna-tuned classification sweep are combined for three-class and pairwise
(HC–AD, HC–MCI, MCI–AD) diagnosis.

---

## Türkçe

ADNI veri tabanındaki dinlenim-hali fMRI verileri üzerinde fonksiyonel bağlantı
analizi ve makine öğrenmesi tabanlı sınıflandırma (Sağlıklı Kontrol / Hafif
Bilişsel Bozukluk / Alzheimer) yapan bir bitirme projesidir. Pipeline; ön işleme,
fonksiyonel bağlantı matrisleri, graf teorisi metrikleri, null-model sapma
skorları ve harmonizasyon + hiperparametre optimizasyonu içeren bir
sınıflandırma taramasından oluşur.

> **Not:** Ham fMRI verileri ADNI kullanım koşulları gereği bu depoya dahil
> edilmemiştir. Veriye [adni.loni.usc.edu](https://adni.loni.usc.edu) üzerinden
> başvuruyla erişilebilir. Bu depo yalnızca **kaynak kodu** içerir.

### Pipeline

```
DICOM  ──►  NIfTI  ──►  ROI zaman serileri  ──►  bağlantı matrisleri
  ──►  graf metrikleri (+ null model)  ──►  istatistik + sınıflandırma
```

1. **Ön işleme** (`01_preprocess.py`) — 24-parametreli Friston konfound
   regresyonu, FD/DVARS hareket QC, atlas tabanlı ROI zaman serisi çıkarımı.
2. **Bağlantı** (`02a_connectivity.py`) — Pearson korelasyonu, Fisher z, yoğunluk
   eşikleme, tangent-space embedding.
3. **Graf metrikleri** (`03_graph_metrics.py`) — global + nodal metrikler,
   yoğunluk ekseninde AUC stratejisi.
4. **Null model** (`04_null_models.py`) — Erdős–Rényi rastgele ağlara karşı
   sapma skorları.
5. **İstatistik & sınıflandırma** (`07_statistics.py`,
   `08a`–`08e_*.py`) — grup karşılaştırmaları, ComBat
   harmonizasyon, tekrarlı çapraz doğrulama, Optuna ile hiperparametre
   optimizasyonu.

---

## English

A graduation project performing functional connectivity analysis and
machine-learning classification (Healthy Control / Mild Cognitive Impairment /
Alzheimer's Disease) on ADNI resting-state fMRI data. The pipeline covers
preprocessing, connectivity matrices, graph-theoretic metrics, null-model
deviation scores, and a classification sweep with harmonization and
hyperparameter tuning.

> **Note:** Raw fMRI data is **not** included in this repository, in line with
> the ADNI data-use terms. Access can be requested at
> [adni.loni.usc.edu](https://adni.loni.usc.edu). This repository contains the
> **source code only**.

### Pipeline

```
DICOM  ──►  NIfTI  ──►  ROI time series  ──►  connectivity matrices
  ──►  graph metrics (+ null model)  ──►  statistics + classification
```

1. **Preprocessing** (`01_preprocess.py`) — 24-parameter Friston confound
   regression, FD/DVARS motion QC, atlas-based ROI time-series extraction.
2. **Connectivity** (`02a_connectivity.py`) — Pearson correlation, Fisher z,
   density thresholding, tangent-space embedding.
3. **Graph metrics** (`03_graph_metrics.py`) — global and nodal metrics with the
   AUC-over-density strategy.
4. **Null models** (`04_null_models.py`) — deviation scores against
   Erdős–Rényi random networks.
5. **Statistics & classification** (`07_statistics.py`,
   `08a`–`08e_*.py`) — group comparisons, ComBat harmonization,
   repeated cross-validation, Optuna hyperparameter tuning.

---

## Project structure

Files are numbered by pipeline stage. When a stage spans several files they
share the number and are lettered in execution order (e.g. `02a`, `02b`, `02c`).

```
.
├─ 00a_config.py           # paths, atlas settings, constants
├─ 00b_discover.py         # dynamic subject discovery
├─ 00c_dicom2nifti.py      # DICOM -> NIfTI conversion
├─ 00d_adni_data.py        # metadata matching, subject list
├─ 01_preprocess.py        # preprocessing + motion QC
├─ 02a_connectivity.py     # connectivity + tangent embedding
├─ 02b_alff_reho.py        # ALFF / fALFF / ReHo features
├─ 02c_multiatlas.py       # multi-atlas pipeline
├─ 03_graph_metrics.py     # graph-theoretic metrics
├─ 04_null_models.py       # null-model deviation features
├─ 06a_classification.py   # baseline classification
├─ 06b_nbs.py              # network-based statistic
├─ 07_statistics.py        # group statistics
├─ 08a_features.py         # classification: feature sets
├─ 08b_transformers.py     # classification: transformers & pipelines
├─ 08c_models.py           # classification: model roster & Optuna tuning
├─ 08d_experiments.py      # classification: per-task experiments
├─ 08e_run.py              # classification: sweep entry point & leaderboard
├─ main.py                 # full-pipeline entry point
├─ run.py                  # discover + run convenience launcher
├─ qc/                     # quality-control & reporting utilities
│  ├─ build_leaderboard.py
│  └─ permutation_test.py
├─ tests/                  # pytest suite (synthetic fixtures)
├─ requirements.txt
└─ LICENSE
```

## Installation

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is recommended (tested on 3.12).

## Usage

```bash
python main.py                 # full pipeline
python main.py --convert-only  # DICOM -> NIfTI only
python main.py --demo          # quick test on the nilearn demo dataset
pytest tests/ -v               # run the test suite
```

## Results

Headline imaging-only results (no clinical-label leakage), reported with
permutation-test significance:

| Task | AUC | 95% CI | Permutation p |
|---|---|---|---|
| 3-class (OVR weighted) | 0.605 | [0.49, 0.72] | 0.010 |
| HC–AD | 0.699 | [0.52, 0.85] | 0.003 |
| HC–MCI | 0.630 | [0.43, 0.77] | 0.022 |
| MCI–AD | 0.697 | [0.55, 0.86] | 0.002 |

All headline configurations reach permutation p < 0.05 (1000 label shuffles,
null AUC ≈ 0.50), i.e. classification performs significantly above chance.
Clinical scores (MMSE, CDR) are diagnostic criteria and are therefore reported
only as a leakage upper bound, not as a defensible result.

## References

- Rubinov & Sporns 2010 — graph metrics
- Varoquaux et al. 2010; Dadi et al. 2019; Pervaiz et al. 2020 — tangent-space FC
- Power et al. 2012 — framewise displacement QC
- Johnson et al. 2007; Pomponio et al. 2020 — ComBat / neuroHarmonize
- Akiba et al. 2019 — Optuna; Chawla et al. 2002 — SMOTE
- Ke et al. 2017 — LightGBM; Chen & Guestrin 2016 — XGBoost

## License

Released under the MIT License (see [LICENSE](LICENSE)). ADNI data is subject to
its own data-use agreement and is not redistributed here.
