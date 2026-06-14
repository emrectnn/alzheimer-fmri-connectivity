# Veri / Data

## Türkçe

Bu klasör projenin **veri** dosyalarını (ham fMRI, ön işlenmiş zaman serileri,
ara çıktılar) tutar. **Veri dosyaları çok büyük olduğu için bu depoya
yüklenmemiştir** (ayrıca ADNI kullanım koşulları yeniden dağıtıma izin vermez).
Depoda yalnızca bu açıklama dosyası bulunur.

**Veri nereden indirilir?**
Proje verisi ADNI (Alzheimer's Disease Neuroimaging Initiative) veri tabanından
indirilmiştir:

- <https://ida.loni.usc.edu/login.jsp?project=ADNI>

Veriye erişmek için ADNI'ye ücretsiz başvuru yapıp onay almak gerekir. Onay
sonrası dinlenim-hali fMRI taramaları ve ilgili klinik tablolar (tanı, demografi,
MMSE, CDR) yukarıdaki bağlantıdan indirilir.

**Veri nasıl yerleştirilir?**
İndirilen dosyalar kodun beklediği klasör yapısına yerleştirilmelidir; yollar
`00a_config.py` içinde tanımlıdır. Tipik yapı:

```
data/
├─ preprocessed/   # ön işlenmiş ROI zaman serileri (*.npy) — koddan üretilir
veri/              # ham ADNI NIfTI + klinik CSV tabloları (indirilen veri)
```

> Veri olmadan kod yalnızca `--demo` modunda (nilearn örnek verisi) ve
> `pytest` testleriyle (sentetik veri) çalışır.

---

## English

This folder holds the project's **data** files (raw fMRI, preprocessed time
series, intermediate outputs). **The data files are not uploaded to this
repository because they are too large** (and ADNI's terms do not permit
redistribution). Only this explanatory file is kept here.

**Where the data comes from.**
The project data was downloaded from the ADNI (Alzheimer's Disease Neuroimaging
Initiative) database:

- <https://ida.loni.usc.edu/login.jsp?project=ADNI>

Access requires a free ADNI application and approval. Once approved, the
resting-state fMRI scans and the related clinical tables (diagnosis,
demographics, MMSE, CDR) are downloaded from the link above and placed into the
folder layout expected by the code (paths are defined in `00a_config.py`).
