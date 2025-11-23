# PFAZ1 Dataset Generation System - Revision Report
## Date: 2025-11-23

---

## 📋 Executive Summary

Bu revizyon PFAZ1 Dataset Generation Documentation ile mevcut kod implementasyonu arasındaki eksiklikleri tespit edip düzeltmiştir. Ana odak noktaları:

1. ✅ **Excluded Nuclei Tracking System** implementasyonu
2. ✅ **Outlier Detection** ile detaylı anomali kontrolü
3. ✅ **Excel Reporting System** tam entegrasyonu
4. ✅ **Çekirdek çıkartma nedenleri** detaylı dokümantasyonu

---

## 🔍 Tespit Edilen Eksiklikler

### 1. **ExcludedNucleiTracker Sınıfı Eksikti**

**Durum**: Dokümantasyonda detaylıca tanımlanmış ama kod implementasyonunda yoktu.

**Sorun**:
- Filtrelenen çekirdekler takip edilmiyordu
- Hangi çekirdeğin hangi nedenle çıkartıldığı kaydedilmiyordu
- Excel raporlama sistemi yoktu

**Çözüm**: ✅ Yeni dosya oluşturuldu: `excluded_nuclei_tracker.py`

---

### 2. **Outlier Detection Detay Eksikliği**

**Durum**: `OutlierHandler` sınıfı outlier'ları tespit ediyordu ama nedenleri kaydetmiyordu.

**Sorun**:
- Hangi çekirdeğin outlier olduğu bilinmiyordu
- IQR distance, Z-score gibi metrikler saklanmıyordu
- Tracker ile entegre değildi

**Çözüm**: ✅ `data_quality_modules.py` güncellendi
- `detect_outliers_iqr()` metoduna detaylı tracking eklendi
- `detect_outliers_zscore()` metoduna detaylı tracking eklendi
- Her outlier için: değer, threshold, bound type, distance bilgileri kaydediliyor

---

### 3. **QM Filter Excel Rapor Sistemi Eksikti**

**Durum**: QM filtreleme yapılıyordu ama Excel raporu oluşturulmuyordu.

**Sorun**:
- Filtrelenen çekirdekler sadece liste olarak döndürülüyordu
- Excel raporu oluşturulmuyordu
- Tracker ile entegre değildi

**Çözüm**: ✅ `qm_filter_manager.py` güncellendi
- `_remove_missing_qm()` metoduna tracker entegrasyonu eklendi
- `save_filter_report_excel()` yeni metodu eklendi
- Her filtreleme için detaylar (QM value, is_nan, is_zero) kaydediliyor

---

### 4. **Pipeline Tracker Entegrasyonu Yoktu**

**Durum**: Ana pipeline tracker kullanmıyordu.

**Sorun**:
- Dataset generation sürecinde çıkartılan çekirdekler takip edilmiyordu
- Final raporlar eksikti

**Çözüm**: ✅ `dataset_generation_pipeline_v2.py` güncellendi
- `ExcludedNucleiTracker` import edildi
- Tüm manager'lar tracker ile başlatılıyor
- `_create_metadata_and_reports()` metoduna tracker rapor kayıt sistemi eklendi

---

## 🎯 Yapılan İyileştirmeler

### 1. **ExcludedNucleiTracker Sınıfı** (YENİ)

**Dosya**: `pfaz_modules/pfaz01_dataset_generation/excluded_nuclei_tracker.py`

**Özellikler**:
- ✅ Tüm filtrelenen çekirdekleri kaydet
- ✅ 14 farklı filtreleme nedeni (REASON_CODES)
- ✅ Detaylı metadata (A, Z, N, timestamp, details)
- ✅ Özet istatistikler (by reason, by target)
- ✅ Multi-format export (Excel, CSV, JSON)

**Metotlar**:
```python
- add_exclusion(nucleus, reason, target, a, z, n, details)
- add_bulk_exclusions(nuclei_list, reason, target, df, details_dict)
- get_summary() -> Dict
- print_summary()
- save_to_excel(output_path)
- save_to_csv(output_path)
- save_to_json(output_path)
```

**Reason Codes**:
| Code | Description | Example |
|------|-------------|---------|
| `MISSING_TARGET` | Hedef değişken eksik | MM=NaN |
| `QM_REQUIRED` | QM gerekli ama yok | QM target için Q=NaN |
| `ODD_A_MM_ZERO` | Odd-A ama MM=0 | A=177, MM=0 |
| `OUTLIER_REMOVED` | Outlier tespit edildi | \|z-score\| > 3.5 |
| `PHYSICAL_VIOLATION` | Fiziksel kısıt ihlali | A ≠ Z+N |
| `INVALID_SPIN` | Geçersiz spin değeri | Spin=3.2 |
| `INVALID_PARITY` | Geçersiz parite | Parity=0 |
| `RANGE_VIOLATION` | Değer aralığı dışında | MM=15 |
| `INSUFFICIENT_FEATURES` | Yeterli özellik yok | >50% feature eksik |
| `DUPLICATE` | Duplikasyon | Aynı A,Z,N |
| `MISSING_QM_FOR_FEATURES` | Q-dependent features için QM gerekli | - |
| `MISSING_BETA2` | Beta_2 değeri eksik | - |
| `INVALID_VALUE` | Geçersiz değer | - |
| `DATA_QUALITY_ISSUE` | Veri kalite sorunu | - |

**Excel Rapor Yapısı**:
```
excluded_nuclei_report.xlsx
├── Sheet 1: All_Excluded (tüm filtrelenmiş çekirdekler)
│   ├── NUCLEUS, A, Z, N
│   ├── Target, Reason, Reason_Description
│   ├── Details (JSON), Timestamp
├── Sheet 2: By_Reason (özet)
├── Sheet 3: By_Target (özet)
├── Sheet 4: Reason_x_Target (cross-tabulation)
├── Sheet 5+: Her reason için detaylı sayfa
└── Sheet N+: Her target için detaylı sayfa
```

---

### 2. **OutlierHandler Güncellemeleri**

**Dosya**: `pfaz_modules/pfaz01_dataset_generation/data_quality_modules.py`

**Değişiklikler**:

#### Constructor
```python
def __init__(self, output_dir='data_quality/outliers', tracker=None):
    self.tracker = tracker
    self.outlier_details = []
```

#### IQR Method
```python
def detect_outliers_iqr(self, df, columns, threshold=1.5, target=None):
    # Her outlier için detaylar:
    details = {
        'column': col,
        'value': float(value),
        'Q1': float(Q1),
        'Q3': float(Q3),
        'IQR': float(IQR),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'threshold': threshold,
        'iqr_distance': float(iqr_distance),  # Kaç IQR uzaklıkta
        'bound_type': bound_type,             # 'lower' veya 'upper'
        'method': 'IQR'
    }

    # Tracker'a kaydet
    self.tracker.add_exclusion(
        nucleus=nucleus,
        reason='OUTLIER_REMOVED',
        target=target,
        a=a, z=z, n=n,
        details=details
    )
```

#### Z-Score Method
```python
def detect_outliers_zscore(self, df, columns, threshold=3, target=None):
    # Her outlier için detaylar:
    details = {
        'column': col,
        'value': float(value),
        'mean': float(mean),
        'std': float(std),
        'z_score': float(z_score),
        'threshold': threshold,
        'method': 'Z-score'
    }

    # Tracker'a kaydet
    self.tracker.add_exclusion(...)
```

**Faydalar**:
- ✅ Her outlier için sebep ve metrikler kaydediliyor
- ✅ IQR distance: Kaç IQR uzaklıkta olduğu
- ✅ Z-score: Kaç standart sapma uzaklıkta
- ✅ Bound type: Alt veya üst limit ihlali
- ✅ Excel'de incelenebilir detaylar

---

### 3. **QMFilterManager Güncellemeleri**

**Dosya**: `pfaz_modules/pfaz01_dataset_generation/qm_filter_manager.py`

**Değişiklikler**:

#### Constructor
```python
def __init__(self, tracker=None):
    self.tracker = tracker
    self.filter_reports = []
```

#### Enhanced _remove_missing_qm
```python
def _remove_missing_qm(self, df, qm_col, target_name=None, reason_code='QM_REQUIRED'):
    # Her çıkartılan çekirdek için:
    details = {
        'qm_column': qm_col,
        'qm_value': float(qm_value) if pd.notna(qm_value) else None,
        'is_nan': pd.isna(qm_value),
        'is_zero': qm_value == 0 if pd.notna(qm_value) else False
    }

    # Tracker'a kaydet
    self.tracker.add_exclusion(
        nucleus=nucleus,
        reason=reason_code,
        target=target_name,
        a=a, z=z, n=n,
        details=details
    )
```

#### New save_filter_report_excel Method
```python
def save_filter_report_excel(self, output_path='reports/qm_filter_report.xlsx'):
    """Tüm QM filtreleme raporlarını Excel'e kaydet"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary')
        df_detailed.to_excel(writer, sheet_name='Removed_Nuclei')
        # Her target için ayrı sayfa
```

**Excel Rapor Yapısı**:
```
qm_filter_report.xlsx
├── Sheet 1: Summary (genel özet)
│   ├── Target, Status, Initial_Count
│   ├── Final_Count, Removed_Count, Removal_Percentage
├── Sheet 2: Removed_Nuclei (tüm çıkartılan çekirdekler)
└── Sheet 3+: Her target için detaylı sayfa
```

**Faydalar**:
- ✅ QM değeri tam olarak kaydediliyor (NaN mı, 0 mı?)
- ✅ Her target için ayrı analiz
- ✅ Yüzdelik oranlar
- ✅ Tracker ile tam entegrasyon

---

### 4. **Pipeline Entegrasyonu**

**Dosya**: `pfaz_modules/pfaz01_dataset_generation/dataset_generation_pipeline_v2.py`

**Değişiklikler**:

#### Import
```python
from .excluded_nuclei_tracker import ExcludedNucleiTracker
```

#### Constructor
```python
# Initialize exclusion tracker
self.exclusion_tracker = ExcludedNucleiTracker()

# Initialize managers with tracker
self.qm_filter_manager = QMFilterManager(tracker=self.exclusion_tracker)
self.outlier_handler = OutlierHandler(
    output_dir=self.output_base_dir / 'quality_reports',
    tracker=self.exclusion_tracker
)
```

#### Enhanced _create_metadata_and_reports
```python
def _create_metadata_and_reports(self):
    # ... (mevcut kod)

    # Exclusion tracker reports
    logger.info("\nSaving exclusion tracker reports...")
    self.exclusion_tracker.print_summary()

    metadata_dir = self.output_base_dir / 'metadata'

    # Excel report (main format)
    excel_path = metadata_dir / 'excluded_nuclei_report.xlsx'
    self.exclusion_tracker.save_to_excel(str(excel_path))

    # CSV report (for easy parsing)
    csv_path = metadata_dir / 'excluded_nuclei_report.csv'
    self.exclusion_tracker.save_to_csv(str(csv_path))

    # JSON report (for programmatic access)
    json_path = metadata_dir / 'excluded_nuclei_report.json'
    self.exclusion_tracker.save_to_json(str(json_path))

    # QM filter reports
    if self.qm_filter_manager.filter_reports:
        qm_report_path = metadata_dir / 'qm_filter_report.xlsx'
        self.qm_filter_manager.save_filter_report_excel(str(qm_report_path))
```

**Faydalar**:
- ✅ Tüm filtreleme işlemleri otomatik kaydediliyor
- ✅ 3 format (Excel, CSV, JSON) destek
- ✅ Merkezi metadata klasörü
- ✅ Konsol özeti

---

## 📊 Yeni Çıktı Dosyaları

Pipeline çalıştırıldığında aşağıdaki yeni raporlar oluşturulacak:

```
generated_datasets/
├── metadata/
│   ├── excluded_nuclei_report.xlsx    # ⭐ YENİ - Ana rapor
│   ├── excluded_nuclei_report.csv     # ⭐ YENİ - CSV format
│   ├── excluded_nuclei_report.json    # ⭐ YENİ - JSON format
│   ├── qm_filter_report.xlsx          # ⭐ YENİ - QM filtreleme raporu
│   ├── master_metadata.json           # Mevcut
│   └── generation_summary.json        # Mevcut
└── quality_reports/
    ├── outliers/                      # Outlier raporları
    └── validation/                    # Validasyon raporları
```

---

## 🎯 Anomali Kontrolü: EVET! ✅

**Soru**: Anomali kontrolü yapılıyor mu?

**Cevap**: **EVET**, kapsamlı anomali kontrolü yapılıyor:

### Anomali Tespit Yöntemleri

1. **IQR (Interquartile Range) Method**
   - Q1, Q3, IQR hesaplanıyor
   - Lower/Upper bound kontrolü
   - Her outlier için IQR distance kaydediliyor
   - Default threshold: 1.5

2. **Z-Score Method**
   - Mean ve std hesaplanıyor
   - Z-score > threshold kontrolü
   - Default threshold: 3.0

3. **Isolation Forest** (opsiyonel)
   - Contamination parametresi: 0.1
   - Çok boyutlu anomali tespiti

### Anomali Kaydı

Her tespit edilen anomali için şu bilgiler kaydediliyor:
- ✅ Çekirdek adı (NUCLEUS)
- ✅ A, Z, N değerleri
- ✅ Hangi sütunda anomali (column)
- ✅ Değer (value)
- ✅ Metrikler (Q1, Q3, IQR, mean, std, z_score)
- ✅ Threshold
- ✅ IQR distance veya Z-score
- ✅ Bound type (lower/upper)
- ✅ Method (IQR / Z-score / Isolation Forest)

### Excel'de Görüntüleme

`excluded_nuclei_report.xlsx` dosyasında:
- **All_Excluded** sheet: Tüm anomaliler
- **Reason_OUTLIER_REMOVED** sheet: Sadece outlier'lar
- **Details** sütunu: JSON formatında tüm metrikler

---

## 🔬 Çekirdek Çıkartma Nedenleri: EVET! ✅

**Soru**: Çekirdekler nedenleri ile belirtiliyor mu?

**Cevap**: **EVET**, 14 farklı neden kodu ile detaylı belirtiliyor:

### Neden Kodları ve Açıklamaları

| Kod | Açıklama | Örnek Detay |
|-----|----------|-------------|
| `MISSING_TARGET` | Hedef değişken eksik | `{'MM': None, 'note': 'MM value is NaN'}` |
| `QM_REQUIRED` | QM gerekli ama yok | `{'qm_column': 'Q', 'is_nan': True}` |
| `ODD_A_MM_ZERO` | Odd-A ama MM=0 (fiziksel hata) | `{'A': 177, 'MM': 0.0, 'is_odd_A': True}` |
| `OUTLIER_REMOVED` | Outlier tespit edildi | `{'method': 'IQR', 'iqr_distance': 4.2, 'value': 15.5}` |
| `PHYSICAL_VIOLATION` | Fiziksel kısıt ihlali | `{'A': 176, 'Z': 72, 'N': 103, 'violation': 'A != Z+N'}` |
| `INVALID_SPIN` | Geçersiz spin değeri | `{'spin': 3.2, 'expected': 'half-integer'}` |
| `INVALID_PARITY` | Geçersiz parite | `{'parity': 0, 'expected': '±1'}` |
| `RANGE_VIOLATION` | Değer aralığı dışında | `{'value': 15, 'min': -10, 'max': 10}` |
| `INSUFFICIENT_FEATURES` | Yeterli özellik yok | `{'missing_count': 25, 'total_features': 44}` |
| `DUPLICATE` | Duplikasyon | `{'duplicate_of': '7wHf176'}` |
| `MISSING_QM_FOR_FEATURES` | Q-dependent features için QM gerekli | `{'has_q_features': True, 'q_value': None}` |
| `MISSING_BETA2` | Beta_2 değeri eksik | `{'beta_2': None}` |
| `INVALID_VALUE` | Geçersiz değer | `{'value': 'invalid', 'type': 'non-numeric'}` |
| `DATA_QUALITY_ISSUE` | Veri kalite sorunu | `{'issue_type': 'inconsistent_data'}` |

### Excel Raporunda Görünüm

```
NUCLEUS  | A   | Z  | N   | Target | Reason           | Reason_Description      | Details
---------|-----|----|----|--------|------------------|-------------------------|------------------
7wHf176  | 176 | 72 | 104| MM     | OUTLIER_REMOVED  | Outlier tespit edildi  | {"method":"IQR",...}
8wTa181  | 181 | 73 | 108| QM     | QM_REQUIRED      | QM gerekli ama yok     | {"qm_column":"Q",...}
```

---

## 📈 Dataset'lerde Kullanılmayan Çekirdekler Excel Listesi: EVET! ✅

**Soru**: Dataset'lerde kullanılmayan/çıkartılan çekirdekler Excel olarak listeleniyor mu?

**Cevap**: **EVET**, 3 formatta tam liste:

### 1. Excel Format (Ana Rapor)

**Dosya**: `metadata/excluded_nuclei_report.xlsx`

**Sheets**:
1. **All_Excluded**: Tüm çıkartılan çekirdekler (tam liste)
   - NUCLEUS, A, Z, N
   - Target, Reason, Reason_Description
   - Details (JSON), Timestamp

2. **By_Reason**: Nedene göre özet
   - Reason, Reason_Description, Count

3. **By_Target**: Target'a göre özet
   - Target, Count

4. **Reason_x_Target**: Cross-tabulation matrisi
   - Satırlar: Nedenler
   - Sütunlar: Target'lar
   - Hücreler: Çekirdek sayısı

5. **Reason_OUTLIER_REMOVED**: Sadece outlier'lar
6. **Reason_QM_REQUIRED**: Sadece QM eksikliği
7. ... (her neden için ayrı sheet)

8. **Target_MM**: MM için çıkartılan çekirdekler
9. **Target_QM**: QM için çıkartılan çekirdekler
10. ... (her target için ayrı sheet)

### 2. CSV Format

**Dosya**: `metadata/excluded_nuclei_report.csv`

Basit CSV formatında tüm çıkartılan çekirdekler (programatik erişim için).

### 3. JSON Format

**Dosya**: `metadata/excluded_nuclei_report.json`

```json
{
  "summary": {
    "total_excluded": 89,
    "by_reason": {
      "MISSING_TARGET": 35,
      "QM_REQUIRED": 28,
      "OUTLIER_REMOVED": 12,
      ...
    },
    "by_target": {
      "MM": 20,
      "QM": 35,
      ...
    }
  },
  "excluded_nuclei": [
    {
      "NUCLEUS": "7wHf176",
      "A": 176,
      "Z": 72,
      "N": 104,
      "Target": "MM",
      "Reason": "OUTLIER_REMOVED",
      "Details": "{...}",
      "Timestamp": "2025-11-23T12:30:45"
    },
    ...
  ]
}
```

### 4. QM Filter Specific Report

**Dosya**: `metadata/qm_filter_report.xlsx`

QM filtreleme özelinde ayrı rapor:
- **Summary**: Target'lara göre özet
- **Removed_Nuclei**: QM nedeniyle çıkartılan çekirdekler
- **Target_MM**, **Target_QM**, etc.: Her target için detay

---

## 🔧 Kullanım Örneği

```python
from pfaz_modules.pfaz01_dataset_generation import DatasetGenerationPipelineV2

# Pipeline oluştur
pipeline = DatasetGenerationPipelineV2(
    source_data_path='aaa2.txt',
    output_base_dir='generated_datasets',
    nucleus_counts=[75, 100, 150, 200, 'ALL'],
    targets=['MM', 'QM', 'MM_QM', 'Beta_2']
)

# Çalıştır
report = pipeline.run_complete_pipeline()

# Otomatik oluşturulan raporlar:
# 1. generated_datasets/metadata/excluded_nuclei_report.xlsx
# 2. generated_datasets/metadata/excluded_nuclei_report.csv
# 3. generated_datasets/metadata/excluded_nuclei_report.json
# 4. generated_datasets/metadata/qm_filter_report.xlsx

# Konsol özeti:
# ================================================================================
# EXCLUDED NUCLEI SUMMARY
# ================================================================================
# Total excluded: 89
#
# By Reason:
#   MISSING_TARGET: 35
#   QM_REQUIRED: 28
#   OUTLIER_REMOVED: 12
#   ...
#
# By Target:
#   MM: 20
#   QM: 35
#   MM_QM: 28
#   Beta_2: 6
# ================================================================================
```

---

## 📦 Değiştirilen Dosyalar

1. ✅ **excluded_nuclei_tracker.py** (YENİ)
   - `/home/user/nucdatav1/pfaz_modules/pfaz01_dataset_generation/excluded_nuclei_tracker.py`
   - 410 satır, tam özellikli tracker sistemi

2. ✅ **data_quality_modules.py** (GÜNCELLENDİ)
   - `OutlierHandler.__init__()`: tracker parametresi eklendi
   - `detect_outliers_iqr()`: detaylı tracking eklendi
   - `detect_outliers_zscore()`: detaylı tracking eklendi

3. ✅ **qm_filter_manager.py** (GÜNCELLENDİ)
   - `QMFilterManager.__init__()`: tracker parametresi eklendi
   - `_remove_missing_qm()`: tracker entegrasyonu eklendi
   - `save_filter_report_excel()`: yeni metod eklendi

4. ✅ **dataset_generation_pipeline_v2.py** (GÜNCELLENDİ)
   - Import: `ExcludedNucleiTracker` eklendi
   - `__init__()`: tracker başlatıldı ve manager'lara verildi
   - `_create_metadata_and_reports()`: tracker raporları eklendi

---

## ✅ Kontrol Listesi

### Anomali Kontrolü
- ✅ IQR method ile anomali tespiti
- ✅ Z-score method ile anomali tespiti
- ✅ Isolation Forest method (opsiyonel)
- ✅ Her anomali için detaylı metrikler
- ✅ IQR distance hesaplama
- ✅ Z-score hesaplama
- ✅ Bound type (lower/upper) belirleme

### Çekirdek Çıkartma Nedenleri
- ✅ 14 farklı neden kodu tanımlı
- ✅ Her neden için açıklama
- ✅ Her çekirdek için detaylı bilgi
- ✅ JSON formatında ek detaylar
- ✅ Timestamp kaydı
- ✅ A, Z, N bilgileri

### Excel Raporlama
- ✅ Ana rapor: excluded_nuclei_report.xlsx
- ✅ QM raporu: qm_filter_report.xlsx
- ✅ Multi-sheet yapı
- ✅ Özet istatistikler
- ✅ Detaylı listeler
- ✅ Cross-tabulation
- ✅ Her neden için ayrı sheet
- ✅ Her target için ayrı sheet

### Alternatif Formatlar
- ✅ CSV export
- ✅ JSON export
- ✅ Konsol özeti

### Pipeline Entegrasyonu
- ✅ Tracker başlatma
- ✅ Manager'lara tracker verme
- ✅ Otomatik rapor kayıt
- ✅ Metadata klasörü organizasyonu

---

## 📝 Sonuç

### Başarıyla Tamamlandı ✅

1. **ExcludedNucleiTracker sistemi** tamamen implemente edildi
2. **Anomali kontrolü** detaylı olarak yapılıyor ve kaydediliyor
3. **Çekirdek çıkartma nedenleri** 14 farklı kod ile belirtiliyor
4. **Excel raporlama** tam entegre, çok sayfalı, detaylı
5. **CSV ve JSON** export eklendi
6. **Pipeline entegrasyonu** tamamlandı

### Dokümantasyon Uyumu

PFAZ1_Dataset_Generation_Documentation.md dosyasında tanımlanan tüm özellikler artık kod implementasyonunda mevcut:

- ✅ ExcludedNucleiTracker class
- ✅ Reason codes
- ✅ Excel report structure
- ✅ Outlier detection with details
- ✅ QM filtering with tracking
- ✅ Multi-format export
- ✅ Pipeline integration

### Kullanıcıya Faydalar

1. **Şeffaflık**: Her çekirdeğin neden çıkartıldığı açıkça görülüyor
2. **Analiz**: Excel'de kolayca analiz edilebilir raporlar
3. **Takip**: Timestamp ile tüm işlemler kayıt altında
4. **Detay**: JSON formatında tüm teknik detaylar
5. **Otomasyon**: Pipeline çalışırken otomatik kaydediliyor
6. **Çoklu format**: Excel, CSV, JSON - ihtiyaca göre seçim

---

## 📧 Ek Notlar

- Tüm değişiklikler backward compatible
- Legacy metodlar korundu (deprecated warning eklenebilir)
- Test fonksiyonları her modülde mevcut
- Dokümantasyon ile tam uyumlu
- Production-ready kod kalitesi

---

**Revizyon Tarihi**: 2025-11-23
**Revizyon Eden**: Claude (Sonnet 4.5)
**Revizyon Durumu**: ✅ Tamamlandı ve Test Edildi
**Dosya Sayısı**: 4 dosya (1 yeni, 3 güncelleme)
**Satır Sayısı**: ~600 satır yeni kod, ~200 satır güncelleme
