# Alzheimer fMRI Brain Connectivity Analysis

Dinlenim-hali (resting-state) fMRI verisinden fonksiyonel beyin bağlantısı
çıkaran ve bundan **Sağlıklı Kontrol (HC) / Hafif Bilişsel Bozukluk (MCI) /
Alzheimer (AD)** ayrımı yapan bir bitirme projesi. Graf teorisi metrikleri,
tangent-space gömme, null-model sapma skorları, ComBat harmonizasyonu ve
Optuna ile ayarlanan bir sınıflandırma taraması birleştirilir.

> Bu README tek başına projeyi anlatacak şekilde yazılmıştır: aşağıda hem
> kullanılan **yöntem**, hem de **her kod dosyasının ne yaptığı** ayrıntılı
> olarak açıklanır. Kodu açmadan projeyi anlamak mümkündür.

---

## İçindekiler

- [Özet (TR / EN)](#özet)
- [Veri](#veri)
- [Pipeline (genel akış)](#pipeline-genel-akış)
- [Yöntem (ne, neden, nasıl)](#yöntem)
- [Kod yapısı](#kod-yapısı)
- [Modüller — her dosya ne yapıyor?](#modüller--her-dosya-ne-yapıyor)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Testler](#testler)
- [Sonuçlar](#sonuçlar)
- [Referanslar](#referanslar)
- [Lisans](#lisans)

---

## Özet

**Türkçe.** Proje, ADNI veri tabanındaki dinlenim-hali fMRI taramalarını alır;
beyni bir atlasla bölgelere (ROI) ayırıp her bölgenin zaman serisini çıkarır;
bölgeler arası fonksiyonel bağlantı (functional connectivity, FC) matrislerini
hesaplar; bu matrisleri bir **ağ (graf)** olarak yorumlayıp graf teorisi
metrikleri üretir; ardından bu özniteliklerden HC/MCI/AD tanısını tahmin eden
makine öğrenmesi modellerini çapraz doğrulama ile değerlendirir. Sınıflandırma;
site/protokol farklarını gideren ComBat harmonizasyonu, yaş/cinsiyet/hareket
konfound düzeltmesi ve Optuna hiperparametre optimizasyonu içerir.

**English.** This graduation project takes ADNI resting-state fMRI scans,
parcellates the brain into atlas regions (ROIs), extracts each region's time
series, computes functional connectivity (FC) matrices, interprets them as
graphs to derive graph-theoretic metrics, and trains cross-validated machine
learning models to classify HC/MCI/AD from those features. The classifier adds
ComBat site harmonization, confound regression (age/sex/motion) and Optuna
hyperparameter tuning.

---

## Veri

- **Kaynak:** ADNI (Alzheimer's Disease Neuroimaging Initiative) dinlenim-hali
  fMRI + klinik CSV tabloları.
- **Gruplar:** HC (sağlıklı kontrol), MCI (hafif bilişsel bozukluk),
  AD (Alzheimer).
- **Protokoller:** Standard (TR = 3.0 s) ve Multiband (kısa TR ≈ 0.607 s).
  ComBat ile `batch = protokol` harmonizasyonu yapılır.
- **Atlaslar:** AAL3 (166 bölge, ana atlas) + Schaefer-200 + Harvard-Oxford-48
  (çoklu-atlas doğrulaması).

> **Önemli:** Ham fMRI verisi ve klinik tablolar ADNI kullanım koşulları gereği
> bu depoya **dahil edilmemiştir**. Veriye [adni.loni.usc.edu](https://adni.loni.usc.edu)
> üzerinden başvuruyla erişilir. Depo yalnızca **kaynak kodu** içerir; bu yüzden
> kod olduğu gibi çalıştırıldığında veri yollarının ayarlanması gerekir.

---

## Pipeline (genel akış)

```
DICOM  ─(00c)─►  NIfTI  ─(01)─►  ROI zaman serileri  ─(02a)─►  FC matrisleri
   │                                                              │
 (00b/00d: ozne kesfi + klinik etiket eslestirme)                │
                                                                  ▼
        sapma skorlari ◄─(04: null model)──  graf metrikleri ◄─(03)
                                                                  │
                              (02b: ALFF/ReHo, 02c: coklu-atlas, 06b: NBS)
                                                                  ▼
              istatistik (07)            siniflandirma taramasi (08a–08e)
                                                                  │
                                              leaderboard + permutation (qc/)
```

Her adımın çıktısı bir sonrakinin girdisidir. Adımların hangi dosyada olduğu
parantezde verilmiştir.

---

## Yöntem

Bu bölüm projedeki temel kavramları ve **neden** kullanıldıklarını açıklar.

**1. Ön işleme (preprocessing).** Ham fMRI sinyali gürültü içerir. İlk birkaç
hacim atılır (manyetik denge), **24-parametreli Friston** konfound regresyonu ile
kafa hareketi etkileri temizlenir, bant-geçirgen filtre (≈0.01–0.1 Hz) uygulanır
ve atlas her bölge için voxellerin ortalaması alınarak ROI zaman serisi üretilir.
**Hareket QC:** her tarama için Framewise Displacement (FD) ve DVARS hesaplanır;
çok hareketli taramalar elenir (Power et al. 2012).

**2. Fonksiyonel bağlantı (FC).** İki bölgenin zaman serileri benzer
dalgalanıyorsa "fonksiyonel olarak bağlı" sayılır. ROI çiftleri arası **Pearson
korelasyonu** ile FC matrisi kurulur; **Fisher z** dönüşümü değerleri normal
dağılıma yaklaştırır (grup karşılaştırmaları için).

**3. Graf metrikleri + AUC stratejisi.** FC matrisi eşiklenerek bir ağ (graf)
elde edilir; ama tek bir eşik keyfidir. Bu yüzden bir **yoğunluk aralığı**
(ör. %5–%50) boyunca metrikler hesaplanıp **trapez kuralı** ile eğri-altı-alan
(AUC) alınır — eşikten bağımsız tek bir robust skaler. Global metrikler:
kümelenme, karakteristik yol uzunluğu, **small-worldness (σ)**, global/lokal
verimlilik, **modülerite (Louvain)**, rich-club, DMN kümelenmesi. Nodal
metrikler: derece, aracılık (betweenness) ve özvektör merkeziliği.

**4. Tangent-space gömme.** Korelasyon matrisleri Riemann manifoldunda yaşar;
**tangent-space** lineerleştirmesi (Varoquaux 2010; Pervaiz 2020) küçük örneklem
+ çok ROI durumunda ham korelasyondan daha iyi ayrım verir. Referans matris
**yalnızca eğitim fold'unda** hesaplanır (veri sızıntısını önlemek için).

**5. Null-model sapma skorları.** Ölçülen graf özelliklerinin rastgele ağlardan
anlamlı şekilde farklı olup olmadığı **Erdős–Rényi** (ve konfigürasyon /
hiperbolik) null modelleriyle test edilir; gözlemin null dağılıma göre z-skoru
özniteliğe dönüştürülür.

**6. Harmonizasyon ve konfound kontrolü.** Farklı protokol/scanner kaynaklı
sistematik farklar **ComBat** (neuroHarmonize) ile giderilir. Yaş, cinsiyet ve
ortalama FD gibi konfoundlar **fold içinde** regrese edilir. Her iki işlem de
çapraz doğrulamada **sızıntısız** olacak şekilde pipeline adımı olarak kurulur.

**7. Sınıflandırma ve değerlendirme.** Dengesiz sınıflar için **SMOTE**
(yalnızca eğitim fold'unda), öznitelik seçimi (SelectKBest), ve geniş bir model
kadrosu (ElasticNet, SVM-RBF, Random Forest, GBM, LightGBM, XGBoost, Stacking,
ordinal regresyon) kullanılır. Değerlendirme **RepeatedStratifiedKFold** ile
yapılır; **Optuna (TPE)** ile hiperparametre ayarı uygulanır.

**8. Öznitelik modları ve sızıntı (leakage) uyarısı.** Üç mod raporlanır:
- `imaging_only` — yalnızca görüntüleme öznitelikleri. **Sızıntı yok**; tezin
  asıl/savunulabilir sonucu budur.
- `imaging_plus_demographics` — + yaş/cinsiyet/eğitim (konfound, sızıntı yok).
- `imaging_plus_clinical` — + MMSE/CDR. Bu skorlar zaten tanı kriteridir; bu mod
  yalnızca **üst-sınır referansı** olarak, "LEAKAGE" uyarısıyla raporlanır.

**9. Anlamlılık testi (permutation).** Headline AUC değerlerinin şans üstü
olduğu, etiketleri 1000 kez karıştırıp aynı CV'yi tekrarlayarak elde edilen null
dağılıma karşı **ampirik p-değeri** ile gösterilir.

---

## Kod yapısı

Dosyalar pipeline aşamasına göre numaralandırılmıştır. Bir aşama birden çok
dosyaya yayılıyorsa aynı numarayı paylaşır ve çalışma sırasına göre harflenir
(ör. `02a`, `02b`, `02c`).

```
.
├─ 00a_config.py           # yollar, atlas ayarları, sabitler
├─ 00b_discover.py         # diskten dinamik ozne kesfi
├─ 00c_dicom2nifti.py      # DICOM -> NIfTI donusumu
├─ 00d_adni_data.py        # klinik metadata eslestirme, ozne listesi
├─ 01_preprocess.py        # on isleme + hareket QC
├─ 02a_connectivity.py     # baglanti matrisleri + tangent gomme
├─ 02b_alff_reho.py        # ALFF / fALFF / ReHo oznitelikleri
├─ 02c_multiatlas.py       # coklu-atlas (Schaefer200, HO48)
├─ 03_graph_metrics.py     # graf teorisi metrikleri
├─ 04_null_models.py       # null-model sapma oznitelikleri
├─ 06a_classification.py   # temel (baseline) siniflandirma
├─ 06b_nbs.py              # Network-Based Statistic (kenar secimi)
├─ 07_statistics.py        # grup istatistikleri
├─ 08a_features.py         # siniflandirma: oznitelik setleri
├─ 08b_transformers.py     # siniflandirma: transformer + pipeline kuruculari
├─ 08c_models.py           # siniflandirma: model kadrosu + Optuna
├─ 08d_experiments.py      # siniflandirma: gorev-bazli deneyler
├─ 08e_run.py              # siniflandirma: tarama girisi + leaderboard
├─ main.py                 # tam pipeline giris noktasi
├─ run.py                  # kesif + calistir kisayolu
├─ qc/                     # kalite kontrol & raporlama
│  ├─ build_leaderboard.py
│  └─ permutation_test.py
├─ tests/                  # pytest (sentetik veriyle)
├─ requirements.txt
└─ LICENSE
```

---

## Modüller — her dosya ne yapıyor?

### Aşama 0 — Hazırlık ve veri

**`00a_config.py`** — Tüm projenin paylaştığı **ayar merkezi**. Dizin yolları,
aktif atlas ve ROI sayısı, TR, bant-geçirgen filtre sınırları, AUC için yoğunluk
aralığı, çapraz doğrulama fold/tekrar sayıları, ALFF bandı, NBS ve permutation
parametreleri, grup renkleri gibi sabitleri tutar. Diğer her modül bunu
`import_module("00a_config")` ile çağırır.

**`00b_discover.py`** — Diskteki ADNI klasörlerini **otomatik tarar**: hangi
denekler var, her birinin rsfMRI serisi hangi protokol (seri adından Standard mı
Multiband mı), klinik CSV'lerden HC/MCI/AD etiketi nedir? Sonuçları çalışma
anında config'in denek listelerine yazar (`update_config`). Böylece yeni denek
eklendiğinde liste elle güncellenmez. `run()` baştan sona keşfi çalıştırır.

**`00c_dicom2nifti.py`** — Ham **DICOM** rsfMRI serilerini `dcm2niix` ile
**NIfTI**'ye çevirir. Deneğin rsfMRI DICOM klasörünü bulur, dönüşümü yapar ve
çıktının geçerli (4B, yeterli hacim) olduğunu doğrular.

**`00d_adni_data.py`** — NIfTI taramalarını ADNI **klinik CSV tablolarıyla**
eşleştirir: tanı (DXSUM), demografi (yaş/cinsiyet/eğitim), MMSE ve CDR skorları.
HC/MCI/AD etiketli `subject_metadata.csv` üretir ve geçersiz/eksik taramaları
(çok kısa, dosya yok) eler.

### Aşama 1 — Ön işleme

**`01_preprocess.py`** — fMRI **ön işleme** çekirdeği. Atlası yükler, ROI
ortalama zaman serisi çıkaran `NiftiLabelsMasker` kurar; ilk hacimleri atar,
**24-parametreli Friston** konfound matrisini kurup regrese eder, bant-geçirgen
filtre uygular. **Hareket QC**: FD/DVARS hesaplar, eşiği aşan taramaları
işaretler; her denek için 4 panelli bir QC görseli üretir. Çıktı: her denek için
`*_timeseries.npy`.

### Aşama 2 — Bağlantı ve sinyal öznitelikleri

**`02a_connectivity.py`** — Zaman serisinden **fonksiyonel bağlantı** üretir:
Pearson korelasyonu, Fisher z dönüşümü, yoğunluk-bazlı eşikleme, AUC için
çoklu-eşik binary graflar, GNN için ağırlıklı komşuluk matrisi ve
**`TangentSpaceTransformer`** (CV-güvenli tangent gömme, scikit-learn uyumlu).
`compute_all_matrices` tüm deneklerin matrislerini hesaplayıp diske yazar.

**`02b_alff_reho.py`** — Bağlantıdan **bağımsız** sinyal öznitelikleri. **ALFF**
(düşük-frekans salınım genliği, Welch PSD'den) ve **fALFF** (bant-içi/toplam
oranı); **ReHo** (her ROI'nin en çok korele olduğu k komşusuyla Kendall's W
uyumu). Her ROI için CSV üretir.

**`02c_multiatlas.py`** — Aynı işlem hattını alternatif atlaslar
(**Schaefer-200**, **Harvard-Oxford-48**) için çalıştırır: ROI zaman serisi, FC
ve birkaç global graf metriği. Sonuçların atlas seçimine duyarlılığını ölçmek ve
`Graf_MultiAtlas` öznitelik setini beslemek için kullanılır.

### Aşama 3-4 — Ağ analizi

**`03_graph_metrics.py`** — FC matrisini **graf** olarak analiz eder. Global
metrikler (kümelenme, yol uzunluğu, small-worldness σ, global/lokal verimlilik,
Louvain modülerite, rich-club, DMN kümelenmesi, serebellar bağlantı) ve nodal
metrikler (derece, aracılık, özvektör merkeziliği). Hepsi **yoğunluk ekseninde
AUC** ile birleştirilir. `compute_all_subjects` tüm denekler için tablo üretir.

**`04_null_models.py`** — Ölçülen metriklerin rastgele ağlardan farkını test
eder. **Erdős–Rényi** (ayrıca konfigürasyon ve hiperbolik) null grafları üretir,
metriklerin null dağılımını çıkarır ve gözlemin **z-skor sapmasını** öznitelik
olarak verir.

### Aşama 6-7 — Temel sınıflandırma, kenar seçimi, istatistik

**`06a_classification.py`** — **Temel (baseline)** sınıflandırma. Graf + nodal +
null özniteliklerinden bir öznitelik matrisi kurar ve bir model kümesini çapraz
doğrulamayla değerlendirir. Ana sonuçlar gelişmiş hatta (`08*`) üretilir; bu
modül `--demo` ve yedek (fallback) yol için durur.

**`06b_nbs.py`** — **Network-Based Statistic** (Zalesky 2010) ile gruplar arası
gerçekten farklılaşan **kenar kümelerini** seçer: her kenara t-testi, eşik üstü
kenarların bağlı bileşenleri, permutation ile en büyük bileşen boyutunun null
dağılımı, p<α olan bileşenlerin kenarları. CV-güvenli transformer olarak sunulur.

**`07_statistics.py`** — Gruplar arası **istatistiksel karşılaştırma**: t-testi /
ANOVA, etki büyüklüğü (Cohen's d) ve çoklu-karşılaştırma düzeltmesi (Bonferroni);
anlamlı metrikleri özetleyip yazdırır.

### Aşama 8 — Gelişmiş sınıflandırma taraması (5 modüle bölünmüş)

**`08a_features.py`** — Tüm öznitelik tablolarını (graf, ALFF/ReHo, çoklu-atlas,
null, demografi, klinik, hareket) yükleyip birleştirir; **öznitelik setlerini**
moda göre tanımlar (`imaging_only`, `+demographics`, `+clinical`) ve
residualize/ComBat için öznitelik matrisini hazırlar.

**`08b_transformers.py`** — **CV-güvenli transformer'lar ve pipeline kurucuları**:
`ConfoundRegressor` (yaş/cinsiyet/hareketi fold içinde regrese eder),
`ComBatTransformer` (site/protokol harmonizasyonu), `make_imb_pipeline`
(impute → ComBat → ölçekle → konfound → seçim → SMOTE → sınıflandırıcı) ve
`make_tangent_pipeline` (tangent + opsiyonel NBS). Hepsi sızıntısız.

**`08c_models.py`** — **Model kadrosu** (ElasticNet, SVM-RBF, RF, GBM, LightGBM,
XGBoost, Stacking, ordinal `mord`), `cv_score` (RepeatedStratifiedKFold ile AUC,
doğruluk, sınıf-bazlı recall, confusion matrix) ve **Optuna (TPE)** ile
hiperparametre ayarı.

**`08d_experiments.py`** — Görev × mod **deneyleri**: 3-sınıf, ikili (HC-AD,
HC-MCI, MCI-AD), iki-aşamalı (hiyerarşik) ve tangent/NBS deneyleri. Her biri
sonuç DataFrame'i döndürür.

**`08e_run.py`** — **Orkestratör ve giriş noktası** (`run_enhanced_classification`):
tüm taramayı (görev × mod × residualize × fixed/Optuna) çalıştırır, sweep
CSV'lerini ve leaderboard'u yazar, figürleri üretir. Ayrıca dış çağıranlar
(main, qc, testler) için sınıflandırma **public API'sini** tek noktadan dışa verir.

### Giriş noktaları, QC ve testler

**`main.py`** — **Tam pipeline** giriş noktası: DICOM dönüşümünden sınıflandırmaya
kadar tüm adımları sırayla çalıştırır. `--convert-only` (sadece dönüşüm),
`--demo` (nilearn demo verisiyle hızlı test) bayrakları vardır.

**`run.py`** — Kısayol: önce diski tarayıp (`00b_discover`) config'i günceller,
sonra `main.run_full_pipeline()` çağırır.

**`qc/build_leaderboard.py`** — Sweep CSV'lerini okuyup **sıralı leaderboard** ve
ablation tabloları (ALFF/ReHo katkısı, NBS vs Tangent, iki-aşamalı vs düz)
üretir; `leaderboard_final.md` + headline CSV yazar.

**`qc/permutation_test.py`** — Headline konfigürasyonlar için **etiket-permutation
anlamlılık testi**; gerçek AUC vs null dağılım → `permutation_pvalues.csv`.

**`tests/`** — Sentetik veriyle hızlı **pytest** regresyon testleri (gerçek ADNI
verisine ihtiyaç duymaz). Bağlantı, graf metrikleri ve sınıflandırma (özellikle
**veri sızıntısı** kontrolleri: scaler/ComBat/konfound'un fold başına bir kez
fit edilmesi) test edilir. Detay: [`tests/README.md`](tests/README.md).

---

## Kurulum

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ önerilir (3.12 ile test edildi).

## Kullanım

```bash
python main.py                 # tam pipeline
python main.py --convert-only  # sadece DICOM -> NIfTI
python main.py --demo          # nilearn demo verisiyle hızlı test
python run.py                  # dinamik keşif + tam pipeline
pytest tests/ -v               # test paketi
```

> Not: Tam pipeline ADNI verisi ve yapılandırılmış yollar gerektirir; veri
> olmadan `--demo` ve `pytest` çalışır.

## Testler

```bash
pytest tests/ -v
```

Testler sentetik fixture'lar kullanır, 30 saniyenin altında koşar ve gerçek
veriye bağımlı değildir. En kritik test, çapraz doğrulamada **veri sızıntısı**
olmadığını doğrular (ön işleme adımlarının her fold'da yeniden fit edilmesi).

## Sonuçlar

Headline `imaging_only` sonuçları (klinik-etiket sızıntısı **yok**), permutation
anlamlılığıyla:

| Görev | AUC | %95 GA | Permutation p |
|---|---|---|---|
| 3-sınıf (OVR weighted) | 0.605 | [0.49, 0.72] | 0.010 |
| HC–AD | 0.699 | [0.52, 0.85] | 0.003 |
| HC–MCI | 0.630 | [0.43, 0.77] | 0.022 |
| MCI–AD | 0.697 | [0.55, 0.86] | 0.002 |

Tüm headline konfigürasyonları permutation p < 0.05 (1000 etiket karıştırma,
null AUC ≈ 0.50) — yani sınıflandırma **şans üstü, istatistiksel olarak anlamlı**.
MMSE/CDR tanı kriteri olduğundan, klinik öznitelikli mod yalnızca **üst-sınır
referansı** olarak (sızıntı uyarısıyla) verilir.

## Referanslar

- Rubinov & Sporns 2010 — graf metrikleri
- Varoquaux et al. 2010; Dadi et al. 2019; Pervaiz et al. 2020 — tangent-space FC
- Power et al. 2012 — framewise displacement (hareket QC)
- Zalesky et al. 2010 — Network-Based Statistic
- Johnson et al. 2007; Pomponio et al. 2020 — ComBat / neuroHarmonize
- Akiba et al. 2019 — Optuna; Chawla et al. 2002 — SMOTE
- Ke et al. 2017 — LightGBM; Chen & Guestrin 2016 — XGBoost

## Lisans

MIT Lisansı altında yayımlanmıştır (bkz. [LICENSE](LICENSE)). ADNI verisi kendi
kullanım sözleşmesine tabidir ve burada yeniden dağıtılmaz.
