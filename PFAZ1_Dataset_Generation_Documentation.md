# PFAZ 1: Dataset Oluşturma Sistemi - Kapsamlı Dokümantasyon

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Sistem Mimarisi](#sistem-mimarisi)
3. [Veri Kaynağı](#veri-kaynağı)
4. [Teorik Hesaplamalar](#teorik-hesaplamalar)
5. [QM Filtreleme Sistemi](#qm-filtreleme-sistemi)
6. [Dataset Boyutlandırma](#dataset-boyutlandırma)
7. [Kalite Kontrol](#kalite-kontrol)
8. [Veri Ölçeklendirme](#veri-ölçeklendirme)
9. [Çıktılar ve Raporlar](#çıktılar-ve-raporlar)
10. [Kullanım Kılavuzu](#kullanım-kılavuzu)

---

## 🎯 Genel Bakış

PFAZ 1, nükleer fizik AI projesinin temel veri hazırlama fazıdır. Bu faz, ham deneysel verilerden başlayarak teorik hesaplamalarla zenginleştirilmiş, kalite kontrollü ve hedef-tabanlı filtrelenmiş eğitim veri setleri üretir.

### Temel Özellikler

- **267 Çekirdek**: AAA2.txt dosyasından yüklenen toplam nükleus sayısı
- **44+ Özellik**: Temel + teorik hesaplanmış özellikler
- **4 Hedef Değişken**: MM, QM, MM_QM, Beta_2
- **5 Dataset Boyutu**: 75, 100, 150, 200, ve ALL (tüm çekirdekler)
- **Akıllı Filtreleme**: Hedef-bazlı QM filtreleme sistemi
- **Kapsamlı Kalite Kontrol**: Otomatik validasyon ve raporlama

### Proje Hedefleri

```
Ham Veri (aaa2.txt, 267 çekirdek)
    ↓
Teorik Hesaplamalar (SEMF, Shell Model, Deformation, vb.)
    ↓
QM Filtreleme (Target-specific intelligent filtering)
    ↓
Dataset Örnekleme (75, 100, 150, 200, ALL)
    ↓
Kalite Kontrol & Validasyon
    ↓
20 Temiz Dataset (4 target × 5 boyut)
```

---

## 🏗️ Sistem Mimarisi

### Ana Bileşenler

```
DatasetGenerationPipeline
├── DataLoader (veri yükleme)
├── TheoreticalCalculationsManager (teorik hesaplamalar)
│   ├── SEMF Calculator
│   ├── Shell Model Calculator
│   ├── Woods-Saxon Calculator
│   ├── Nilsson Model Calculator
│   ├── Schmidt Moments Calculator
│   ├── Collective Model Calculator
│   └── Deformation Calculator
├── QMFilterManager (QM filtreleme)
├── OutlierHandler (aykırı değer tespiti)
├── DataValidator (veri doğrulama)
└── ReportGenerator (raporlama)
```

### Dosya Yapısı

```
project/
├── aaa2.txt                                    # Ham veri kaynağı
├── dataset_generation_pipeline_v2.py           # Ana pipeline
├── data_loader.py                              # Veri yükleme modülü
├── theoretical_calculations_manager.py         # Teorik hesaplamalar
├── qm_filter_manager.py                        # QM filtreleme
├── data_quality_modules.py                     # Kalite kontrol
├── aaa2_quality_checker.py                     # AAA2 özel kontroller
└── constants_v1_1_0.py                         # Fiziksel sabitler
```

---

## 📊 Veri Kaynağı

### AAA2.txt Dosyası

**Format**: Tab-separated values (TSV)  
**Encoding**: UTF-8  
**Çekirdek Sayısı**: 267  
**Kaynak**: Deneysel nükleer fizik veritabanı

### Temel Sütunlar

| Sütun Adı | Açıklama | Birim | Veri Tipi |
|-----------|----------|-------|-----------|
| `NUCLEUS` | Nükleus adı (örn: 7wHf176) | - | string |
| `A` | Kütle numarası | - | int |
| `Z` | Proton sayısı | - | int |
| `N` | Nötron sayısı | - | int |
| `SPIN` | Nükleus spini | ℏ | float |
| `PARITY` | Parite (+1 veya -1) | - | int |
| `MAGNETIC MOMENT [µ]` | Manyetik moment (MM) | µ_N | float |
| `QUADRUPOLE MOMENT [Q]` | Quadrupole moment (QM) | barn | float |
| `Beta_2` | Deformasyon parametresi | - | float |
| `P-factor` | P faktörü | - | float |
| `Nn` | Nötron sayısı (alternatif) | - | int |
| `Np` | Proton sayısı (alternatif) | - | int |

### Veri Yükleme İşlemi

```python
# Adım 1: Dosyayı yükle
df = pd.read_csv('aaa2.txt', sep='\t', encoding='utf-8')

# Adım 2: Sütun isimlerini temizle
df.columns = df.columns.str.strip()

# Adım 3: Sütun adlarını standardize et
column_mapping = {
    'MAGNETIC MOMENT [µ]': 'MM',
    'QUADRUPOLE MOMENT [Q]': 'Q',
    'P-factor': 'P_FACTOR'
}
df = df.rename(columns=column_mapping)

# Adım 4: Veri tiplerini kontrol et
logger.info(f"✓ {len(df)} çekirdek yüklendi")
logger.info(f"Sütunlar: {list(df.columns)}")
```

### Veri Kalitesi - İlk Kontroller

```python
# Missing value analizi
missing = df.isnull().sum()
logger.info(f"Eksik değerler:\n{missing[missing > 0]}")

# Sıfır değer kontrolü (MM için kritik)
zero_mm = df[df['MM'] == 0]
logger.info(f"MM=0 olan çekirdek sayısı: {len(zero_mm)}")

# A = Z + N tutarlılığı
inconsistent = df[df['A'] != df['Z'] + df['N']]
logger.info(f"A≠Z+N tutarsızlığı: {len(inconsistent)}")
```

---

## 🔬 Teorik Hesaplamalar

### 1. SEMF (Semi-Empirical Mass Formula)

**Amaç**: Bağlanma enerjisini ve bileşenlerini hesaplamak

**Formül**:
```
BE = a_v·A - a_s·A^(2/3) - a_c·Z²/A^(1/3) - a_a·(N-Z)²/A + δ(A,Z)
```

**Parametreler**:
```python
SEMF_PARAMS = {
    'a_v': 15.75,    # Hacim terimi (MeV)
    'a_s': 17.8,     # Yüzey terimi (MeV)
    'a_c': 0.711,    # Coulomb terimi (MeV)
    'a_a': 23.7,     # Asimetri terimi (MeV)
    'a_p': 11.18     # Pairing terimi (MeV)
}
```

**Hesaplanan Özellikler** (11 adet):
- `BE_volume`: Hacim enerjisi
- `BE_surface`: Yüzey enerjisi
- `BE_coulomb`: Coulomb enerjisi
- `BE_asymmetry`: Asimetri enerjisi
- `BE_pairing`: Pairing enerjisi
- `BE_total`: Toplam bağlanma enerjisi
- `BE_per_A`: Nükleon başına bağlanma enerjisi
- `nuclear_radius`: Nükleer yarıçap
- `S_n_approx`: Nötron ayrılma enerjisi (yaklaşık)
- `S_p_approx`: Proton ayrılma enerjisi (yaklaşık)
- `mass_excess`: Kütle fazlası

**Pairing Terimi Hesabı**:
```python
def pairing_delta(row):
    if row['Z'] % 2 == 0 and row['N'] % 2 == 0:
        return a_p / sqrt(row['A'])      # even-even: pozitif
    elif row['Z'] % 2 == 1 and row['N'] % 2 == 1:
        return -a_p / sqrt(row['A'])     # odd-odd: negatif
    else:
        return 0.0                        # even-odd veya odd-even: sıfır
```

### 2. Shell Model (Kabuk Modeli)

**Amaç**: Kabuk yapısı etkilerini belirlemek

**Magic Numbers**:
```python
MAGIC_NUMBERS_Z = [2, 8, 20, 28, 50, 82, 114, 126]  # Proton
MAGIC_NUMBERS_N = [2, 8, 20, 28, 50, 82, 126, 184]  # Nötron
```

**Shell Gaps (MeV)**:
```python
SHELL_GAPS = {
    2: 14.0,
    8: 11.0,
    20: 8.0,
    28: 6.0,
    50: 5.0,
    82: 4.0,
    126: 3.0
}
```

**Hesaplanan Özellikler** (10 adet):
- `Z_nearest_magic`: En yakın magic proton sayısı
- `N_nearest_magic`: En yakın magic nötron sayısı
- `Z_magic_dist`: Magic sayıya uzaklık (Z)
- `N_magic_dist`: Magic sayıya uzaklık (N)
- `Z_shell_gap`: Proton shell gap enerjisi
- `N_shell_gap`: Nötron shell gap enerjisi
- `Z_valence`: Valans proton sayısı
- `N_valence`: Valans nötron sayısı
- `magic_character`: Magic karakter skoru (0-1)
- `is_doubly_magic`: Çift-magic mi? (boolean)

**Magic Character Hesabı**:
```python
# 0 = tam magic, 1 = magic'ten çok uzak
magic_character = 1.0 / (1.0 + Z_magic_dist + N_magic_dist)
```

### 3. Deformation (Deformasyon) Hesaplamaları

**Amaç**: Nükleus şekil parametrelerini hesaplamak

**Hesaplanan Özellikler** (4 adet):
- `Beta_2_estimated`: Tahmin edilen β₂ parametresi
- `deformation_type`: Deformasyon tipi (spherical/prolate/oblate)
- `spherical_index`: Küresellik indeksi (0-1)
- `deformation_magnitude`: Deformasyon büyüklüğü

**Beta_2 Tahmini**:
```python
def estimate_beta2(Z, N):
    magic_z_dist = min([abs(Z - m) for m in MAGIC_NUMBERS_Z])
    magic_n_dist = min([abs(N - m) for m in MAGIC_NUMBERS_N])
    
    if magic_z_dist < 2 and magic_n_dist < 2:
        return 0.0      # Küresel (magic)
    elif magic_z_dist > 10 or magic_n_dist > 10:
        return 0.3      # Yüksek deformasyonlu
    else:
        return 0.15     # Orta seviye deformasyonlu
```

**Deformation Type Sınıflandırması**:
```python
def classify_deformation(beta2):
    if abs(beta2) < 0.05:
        return 'spherical'     # Küresel
    elif beta2 > 0.05:
        return 'prolate'       # Yumurta şekli (uzamış)
    else:
        return 'oblate'        # Basık
```

### 4. Schmidt Moments

**Amaç**: Tek-nükleon magnetik momentlerini hesaplamak

**g-Faktörleri**:
```python
G_FACTORS = {
    'proton': {
        'g_l': 1.0,      # Yörüngesel g-faktörü
        'g_s': 5.586     # Spin g-faktörü
    },
    'neutron': {
        'g_l': 0.0,
        'g_s': -3.826
    }
}
```

**Hesaplanan Özellikler** (2 adet):
- `schmidt_nearest`: En yakın Schmidt değeri
- `schmidt_deviation`: Schmidt sapması

**Schmidt Moment Formülleri**:
```python
# j = l + 1/2 durumu
schmidt_plus = g_l * l_plus + g_s * 0.5

# j = l - 1/2 durumu
schmidt_minus = (g_l * l_minus - g_s * 0.5) * j / (j + 1)

# En yakınını seç
schmidt_nearest = min(schmidt_plus, schmidt_minus, key=lambda x: abs(x - MM_exp))
```

### 5. Collective Model (Kollektif Model)

**Amaç**: Kollektif hareketleri modellemek

**Hesaplanan Özellikler** (4 adet):
- `collective_parameter`: Kollektif parametre
- `rotational_constant`: Rotasyonel sabit
- `vibrational_frequency`: Titreşim frekansı
- `nucleus_collective_type`: Kollektif tip (vibrator/rotor/transitional)

**Rotasyonel Sabit**:
```python
# Even-even çekirdekler için
rotational_constant = ℏ² / (2 * I)  # I: moment of inertia
```

### 6. Woods-Saxon Potential (Opsiyonel)

**Amaç**: Nükleer potansiyel derinliğini hesaplamak

**Parametreler**:
```python
WOODS_SAXON_PARAMS = {
    'V0': -51.0,  # MeV (derinlik)
    'r0': 1.27,   # fm
    'a': 0.67     # fm (yüzey diffüzlüğü)
}
```

**Hesaplanan Özellikler** (2 adet):
- `ws_surface_thick`: Yüzey kalınlığı
- `fermi_energy`: Fermi enerjisi (yaklaşık)

**⚠️ Not**: Hesaplama yoğun olduğu için varsayılan olarak kapalıdır.

### 7. Nilsson Model (Opsiyonel)

**Amaç**: Deformed çekirdeklerde tek-parçacık durumları

**Hesaplanan Özellikler** (2 adet):
- `nilsson_epsilon`: Nilsson deformasyon parametresi
- `nilsson_omega`: Osilatör frekansı

**Hesaplama**:
```python
# Sadece |β₂| > 0.15 olan çekirdekler için
nilsson_epsilon = 0.95 * Beta_2_estimated
nilsson_omega = 41.0 / A^(1/3)
```

**⚠️ Not**: Sadece yüksek deformasyonlu çekirdekler için hesaplanır.

### Teorik Hesaplama Özeti

| Hesaplama Grubu | Özellik Sayısı | Hesaplama Süresi | Varsayılan Durum |
|----------------|----------------|------------------|------------------|
| SEMF | 11 | Hızlı (~1s) | ✅ Aktif |
| Shell Model | 10 | Hızlı (~2s) | ✅ Aktif |
| Deformation | 4 | Hızlı (~1s) | ✅ Aktif |
| Schmidt Moments | 2 | Hızlı (~1s) | ✅ Aktif |
| Collective Model | 4 | Orta (~5s) | ✅ Aktif |
| Woods-Saxon | 2 | Yavaş (~30s) | ⚠️ Opsiyonel |
| Nilsson | 2 | Yavaş (~30s) | ⚠️ Opsiyonel |
| **TOPLAM** | **35** | **~40s** (hepsi) | - |

---

## 🎯 QM Filtreleme Sistemi

### Filtreleme Mantığı

QM (Quadrupole Moment) verisi eksik olabilen özel bir hedeftir. Filtreleme stratejisi hedef değişkene göre değişir:

```python
class QMFilterManager:
    """Target-specific intelligent QM filtering"""
    
    def apply_intelligent_filtering(self, df, target):
        if target == 'MM':
            # MM için QM gerekli değil
            return self._filter_for_mm(df)
        
        elif target == 'QM':
            # QM için QM kesinlikle gerekli
            return self._filter_for_qm(df)
        
        elif target == 'MM_QM':
            # Dual target - her ikisi de gerekli
            return self._filter_for_dual(df)
        
        elif target == 'Beta_2':
            # Beta_2 için QM opsiyonel
            return self._filter_for_beta2(df)
```

### Target-Specific Filtreler

#### 1. MM Target Filtresi
```python
def _filter_for_mm(self, df):
    """
    MM için:
    - MM değeri mutlaka olmalı
    - QM eksik olabilir (sorun değil)
    """
    valid = df['MM'].notna()
    filtered = df[valid].copy()
    
    logger.info(f"MM filtre: {len(filtered)}/{len(df)} çekirdek")
    return filtered
```

#### 2. QM Target Filtresi
```python
def _filter_for_qm(self, df):
    """
    QM için:
    - QM değeri mutlaka olmalı
    - Odd-A çekirdeklerde MM≠0 kontrolü
    """
    # QM olmalı
    valid_qm = df['Q'].notna()
    
    # Odd-A ise MM≠0 olmalı
    odd_a = (df['A'] % 2 == 1)
    valid_mm = (~odd_a) | ((odd_a) & (df['MM'] != 0))
    
    filtered = df[valid_qm & valid_mm].copy()
    
    logger.info(f"QM filtre: {len(filtered)}/{len(df)} çekirdek")
    return filtered
```

#### 3. MM_QM Dual Target Filtresi
```python
def _filter_for_dual(self, df):
    """
    MM_QM için:
    - Hem MM hem QM olmalı
    - En katı filtre
    """
    valid = df['MM'].notna() & df['Q'].notna()
    
    # Odd-A kontrolü
    odd_a = (df['A'] % 2 == 1)
    valid = valid & ((~odd_a) | (df['MM'] != 0))
    
    filtered = df[valid].copy()
    
    logger.info(f"MM_QM filtre: {len(filtered)}/{len(df)} çekirdek")
    return filtered
```

#### 4. Beta_2 Target Filtresi
```python
def _filter_for_beta2(self, df):
    """
    Beta_2 için:
    - Beta_2 değeri olmalı
    - QM opsiyonel (beta2 hesabında kullanılabilir ama şart değil)
    """
    valid = df['Beta_2'].notna()
    filtered = df[valid].copy()
    
    logger.info(f"Beta_2 filtre: {len(filtered)}/{len(df)} çekirdek")
    return filtered
```

### Filtreleme İstatistikleri

| Target | QM Gerekli mi? | MM Kontrolü | Tipik Veri Kaybı |
|--------|----------------|-------------|------------------|
| MM | ❌ Hayır | ✅ Evet | ~5% |
| QM | ✅ Evet | ✅ Odd-A için | ~25% |
| MM_QM | ✅ Evet | ✅ Evet | ~25% |
| Beta_2 | ❌ Hayır | ❌ Hayır | ~10% |

### Odd-A Kontrolü Neden Önemli?

**Fiziksel Sebep**: Odd-A (tek kütle numaralı) çekirdeklerde MM=0 fiziksel olarak olamaz!

```python
# MM=0 kontrolü
if A % 2 == 1:  # Odd-A
    if MM == 0:
        # BU FİZİKSEL OLARAK YANLIŞ!
        # Bu çekirdeği filtrelememiz gerekiyor
        logger.warning(f"{nucleus}: Odd-A ama MM=0 (FİZİKSEL HATA)")
```

**Even-Even vs Odd-A**:
- **Even-even** (A çift): MM=0 olabilir (ground state spin=0)
- **Odd-A** (A tek): MM≠0 olmalı (unpaired nucleon var)

---

## 📏 Dataset Boyutlandırma

### Boyut Seçenekleri

Projede 5 farklı dataset boyutu oluşturulur:

```python
NUCLEUS_COUNTS = [75, 100, 150, 200, 'ALL']
```

### Örnekleme Stratejisi

**Amaç**: Her boyut için temsili bir örneklem oluşturmak

```python
def sample_dataset(df, n_samples, target):
    """
    Stratified sampling - hedef değişkene göre katmanlı örnekleme
    """
    if n_samples == 'ALL':
        return df.copy()
    
    if n_samples >= len(df):
        logger.warning(f"İstenen {n_samples} > mevcut {len(df)}, hepsi kullanılacak")
        return df.copy()
    
    # Stratified sampling by target value ranges
    sampled = self._stratified_sample(df, n_samples, target)
    
    logger.info(f"✓ {n_samples} çekirdek örneklendi")
    return sampled
```

### Stratified Sampling Detayları

```python
def _stratified_sample(self, df, n, target_col):
    """
    Target değerinin dağılımını koruyarak örnekleme
    """
    # Target değerini kategorilere ayır
    df['target_bin'] = pd.qcut(df[target_col], q=5, labels=False, duplicates='drop')
    
    # Her kategoriden orantılı örnekle
    samples = []
    for bin_id in df['target_bin'].unique():
        bin_df = df[df['target_bin'] == bin_id]
        bin_n = int(n * len(bin_df) / len(df))
        if bin_n > 0:
            samples.append(bin_df.sample(n=bin_n, random_state=42))
    
    result = pd.concat(samples)
    result = result.drop('target_bin', axis=1)
    
    return result.sample(frac=1, random_state=42)  # Shuffle
```

### Gelişmiş Klasör Yapısı (Feature-Aware)

```
generated_datasets/
├── standard/                           # Standart datasetler (≤4 feature)
│   ├── MM/
│   │   ├── 3In1Out/                   # 3 giriş → 1 çıkış (MM)
│   │   │   ├── Basic/                 # 6 features
│   │   │   │   ├── MM_75_S70_3In1Out_Basic_Standard_Random.csv
│   │   │   │   ├── MM_75_S70_3In1Out_Basic_Standard_Stratified.csv
│   │   │   │   ├── MM_75_S80_3In1Out_Basic_Standard_Random.csv
│   │   │   │   └── ...
│   │   │   ├── Basic_SEMF/            # 12 features
│   │   │   ├── Basic_Shell/           # 12 features
│   │   │   └── Physics_Optimized/     # 12 features (best)
│   │   └── metadata/
│   │       ├── feature_mapping.json
│   │       └── dataset_catalog.xlsx
│   │
│   ├── QM/
│   │   ├── 3In1Out/                   # 3 giriş → 1 çıkış (QM)
│   │   │   ├── Basic/
│   │   │   ├── Basic_SEMF/
│   │   │   └── ...
│   │   └── metadata/
│   │
│   ├── MM_QM/                          # Dual target
│   │   ├── 3In2Out/                   # 3 giriş → 2 çıkış (MM+QM)
│   │   │   ├── Basic/
│   │   │   ├── Extended/
│   │   │   └── ...
│   │   └── metadata/
│   │
│   └── Beta_2/
│       ├── 2In1Out/                   # 2 giriş → 1 çıkış (minimal)
│       ├── 3In1Out/                   # 3 giriş → 1 çıkış
│       ├── 4In1Out/                   # 4 giriş → 1 çıkış
│       │   ├── Basic_Beta2/           # 7 features
│       │   ├── Extended_Beta2/        # 12 features
│       │   └── Full_Beta2/            # 19 features
│       └── metadata/
│
├── advanced/                           # Advanced datasetler (5+ features)
│   ├── MM/
│   │   ├── 5InAdv/                    # 5+ giriş
│   │   │   ├── Extended/              # 21 features
│   │   │   └── Full/                  # 44 features
│   │   ├── 10InAdv/                   # 10+ giriş
│   │   └── AutoML/                    # AutoML engineered features
│   │       └── PolyInteract/          # 200+ features
│   │
│   ├── QM/
│   │   └── ... (aynı yapı)
│   │
│   ├── MM_QM/
│   │   └── ... (aynı yapı)
│   │
│   └── Beta_2/
│       └── ... (aynı yapı)
│
├── anfis_optimized/                    # ANFIS için optimize edilmiş
│   ├── MM/
│   │   ├── ANFIS_Compact/             # 5 features (32 rules)
│   │   │   ├── MM_75_S70_ANFIS_Compact.csv
│   │   │   ├── MM_75_S70_ANFIS_Compact.mat
│   │   │   └── ...
│   │   ├── ANFIS_Standard/            # 8 features (256 rules)
│   │   └── ANFIS_Extended/            # 10 features (1024 rules)
│   │
│   └── ... (QM, MM_QM, Beta_2)
│
└── metadata/
    ├── master_catalog.xlsx            # Tüm datasetlerin master kataloğu
    ├── feature_combinations.json      # Feature kombinasyon detayları
    ├── excluded_nuclei_report.xlsx    # Filtrelenen çekirdekler
    └── generation_summary.json        # Genel özet
```

### Input-Output Konfigürasyonları

**Standart Konfigürasyonlar** (≤4 giriş):

| Config | Target | Input Count | Output Count | Features | Description |
|--------|--------|-------------|--------------|----------|-------------|
| **2In1Out** | Beta_2 | 2 | 1 | A, Z | Minimal |
| **3In1Out** | MM/QM | 3 | 1 | A, Z, N | Basic |
| **3In2Out** | MM_QM | 3 | 2 | A, Z, N | Dual basic |
| **4In1Out** | Beta_2 | 4 | 1 | A, Z, N, SPIN | Standard |

**Advanced Konfigürasyonlar** (5+ giriş):

| Config | Target | Input Count | Output Count | Features | Description |
|--------|--------|-------------|--------------|----------|-------------|
| **5InAdv** | All | 5-10 | 1-2 | Extended | Advanced physics |
| **10InAdv** | All | 10-20 | 1-2 | Full | Complete features |
| **20InAdv** | All | 20-44 | 1-2 | All | Maximum info |
| **AutoML** | All | 50-200 | 1-2 | Engineered | Polynomial+Interact |

### Excel Eğitim Sonuçları Yapısı

**Ana Rapor**: `training_results_master.xlsx`

#### Sheet 1: Model Performance Summary
```
Model_ID | Target | Feature_Set | Input_Config | Size | Scenario | Scaling | 
Model_Type | R² | RMSE | MAE | MAPE | Training_Time | Rules_Count |
```

**Örnek Satırlar**:
```
M_001 | MM | Basic | 3In1Out | 75 | S70 | Standard | RF | 0.91 | 0.42 | 0.31 | 8.2 | 45.3s | N/A
M_002 | MM | Basic_SEMF | 3In1Out | 75 | S70 | Standard | XGB | 0.93 | 0.38 | 0.28 | 7.1 | 52.1s | N/A
A_001 | MM | ANFIS_Compact | 3In1Out | 75 | S70 | Standard | ANFIS-GAU2MF | 0.88 | 0.48 | 0.35 | 9.5 | 78.3s | 32
A_002 | MM | ANFIS_Standard | 3In1Out | 75 | S70 | Standard | ANFIS-GEN2MF | 0.89 | 0.45 | 0.33 | 8.9 | 95.7s | 256
```

#### Sheet 2: Feature Set Mapping
```
Feature_Set_Name | Input_Count | Feature_List | Total_Features | Category | Recommended_For |
```

**Örnekler**:
```
Basic | 3 | A, Z, N, SPIN, PARITY, P_FACTOR | 6 | Standard | Quick tests
Basic_SEMF | 3 | A, Z, N, SPIN, PARITY, P_FACTOR, BE_volume, BE_surface, ... | 12 | Standard | SEMF analysis
Extended | 5 | [21 features] | 21 | Advanced | Deep analysis
ANFIS_Compact | 3 | A, Z, SPIN, magic_character, BE_per_A | 5 | ANFIS | ANFIS training
Full | 10 | [44 features] | 44 | Advanced | Maximum info
```

#### Sheet 3: Input-Output Configurations
```
Config_Name | Input_Features | Output_Features | Input_Count | Output_Count | 
Complexity | Rule_Count_2MF | Rule_Count_3MF | Recommended_Models |
```

**Örnekler**:
```
2In1Out | A, Z | Beta_2 | 2 | 1 | Simple | 4 | 9 | ANFIS, RF
3In1Out | A, Z, N | MM | 3 | 1 | Standard | 8 | 27 | All models
3In2Out | A, Z, N | MM, QM | 3 | 2 | Dual | 8 | 27 | DNN, ANFIS
4In1Out | A, Z, N, SPIN | Beta_2 | 4 | 1 | Advanced | 16 | 81 | XGB, DNN
5InAdv | [5 features] | MM | 5 | 1 | Advanced | 32 | 243 | DNN, BNN
```

#### Sheet 4: ANFIS Detailed Results
```
ANFIS_ID | Target | Feature_Set | Input_Config | Config_Name | MF_Type | 
Num_MFs | Num_Rules | Num_Params | Epochs | Converged | Final_Error |
R² | RMSE | MAE | Training_Time | Rule_Interpretability |
```

**Örnekler**:
```
A_001 | MM | ANFIS_Compact | 3In1Out | GAU2MF | gaussmf | 2 | 32 | 256 | 100 | Yes | 0.0123 | 0.88 | 0.48 | 0.35 | 78.3s | High
A_002 | MM | ANFIS_Standard | 3In1Out | GEN2MF | gbellmf | 2 | 256 | 2048 | 100 | Yes | 0.0098 | 0.89 | 0.45 | 0.33 | 95.7s | Medium
A_003 | QM | ANFIS_Extended | 3In1Out | SUBR05 | subclust | Auto | 45 | 720 | 100 | Yes | 0.0156 | 0.86 | 0.52 | 0.38 | 112.4s | High
```

#### Sheet 5: ANFIS Rules Analysis
```
ANFIS_ID | Rule_Number | Input_MFs | Output_Type | Rule_Weight | 
Rule_Activation | Physical_Interpretation |
```

**Örnekler**:
```
A_001 | 1 | IF A=Low AND Z=Low AND N=Low | Linear | 0.15 | 0.23 | Light nuclei, spherical
A_001 | 2 | IF A=Low AND Z=Low AND N=Medium | Linear | 0.08 | 0.12 | Light neutron-rich
A_001 | 7 | IF A=Medium AND Z=Medium AND N=Medium | Linear | 0.22 | 0.45 | Magic regions
A_001 | 15 | IF A=High AND Z=High AND N=High | Linear | 0.18 | 0.31 | Heavy deformed
```

#### Sheet 6: Training Configuration Details
```
Model_ID | Dataset_Path | Feature_Set_Path | Hyperparameters | 
Random_Seed | CV_Folds | Early_Stopping | Validation_Split |
```

#### Sheet 7: Feature Importance (per model)
```
Model_ID | Feature_Name | Importance_Score | SHAP_Value | Rank | 
Feature_Type | Physical_Meaning |
```

**Örnekler**:
```
M_002 | A | 0.192 | 2.45 | 1 | Basic | Mass number
M_002 | Z | 0.175 | 2.12 | 2 | Basic | Proton count
M_002 | SPIN | 0.128 | 1.56 | 3 | Basic | Nuclear spin
M_002 | magic_character | 0.097 | 1.18 | 4 | Shell | Shell structure
M_002 | BE_per_A | 0.083 | 1.01 | 5 | SEMF | Binding energy
```

#### Sheet 8: Dataset Statistics
```
Dataset_Name | Target | Feature_Set | Total_Samples | Train_Count | 
Check_Count | Test_Count | Excluded_Count | Exclusion_Reasons |
```

#### Sheet 9: Model Comparison Matrix
```
             | Basic | Basic_SEMF | Extended | Full | ANFIS_Compact |
-------------|-------|------------|----------|------|---------------|
RandomForest | 0.91  | 0.92       | 0.93     | 0.94 | N/A           |
XGBoost      | 0.93  | 0.94       | 0.95     | 0.96 | N/A           |
DNN          | 0.89  | 0.91       | 0.92     | 0.93 | N/A           |
ANFIS-GAU2MF | N/A   | N/A        | N/A      | N/A  | 0.88          |
ANFIS-GEN2MF | N/A   | N/A        | N/A      | N/A  | 0.89          |
```

#### Sheet 10: Advanced Datasets Performance
```
Model_ID | Target | Feature_Set | Input_Count | Advanced_Type | 
R² | RMSE | Improvement_vs_Standard | Training_Time_Ratio |
```

**Örnekler**:
```
M_101 | MM | Extended | 21 | 5InAdv | 0.95 | 0.32 | +2.1% | 2.3×
M_102 | MM | Full | 44 | 10InAdv | 0.96 | 0.28 | +3.2% | 4.8×
M_103 | MM | PolyInteract | 187 | AutoML | 0.97 | 0.25 | +4.5% | 12.7×
```

#### Sheet 11: Scenario Comparison (S70 vs S80)
```
Model_Type | Feature_Set | S70_R² | S80_R² | Difference | 
S70_Overfitting | S80_Overfitting | Better_Scenario |
```

#### Sheet 12: Pivot Tables

**Pivot 1: Average R² by Feature Set and Model**
```
          | RF   | XGB  | DNN  | ANFIS_Avg |
----------|------|------|------|-----------|
Basic     | 0.91 | 0.93 | 0.89 | 0.88      |
Extended  | 0.93 | 0.95 | 0.92 | 0.90      |
Full      | 0.94 | 0.96 | 0.93 | N/A       |
```

**Pivot 2: Best Model per Target and Feature Set**
```
Target | Basic      | Extended   | Full       |
-------|------------|------------|------------|
MM     | XGB (0.93) | XGB (0.95) | XGB (0.96) |
QM     | RF (0.91)  | DNN (0.92) | XGB (0.94) |
Beta_2 | XGB (0.89) | DNN (0.91) | BNN (0.93) |
```

**Pivot 3: Training Time by Input Configuration**
```
Input_Config | Avg_Time | Min_Time | Max_Time | Std_Dev |
-------------|----------|----------|----------|---------|
2In1Out      | 12.3s    | 8.1s     | 18.5s    | 2.4s    |
3In1Out      | 45.7s    | 32.1s    | 67.3s    | 8.9s    |
4In1Out      | 89.2s    | 61.4s    | 124.8s   | 15.3s   |
5InAdv       | 187.5s   | 143.2s   | 256.7s   | 28.4s   |
```

#### Sheet 13: ANFIS Rule Interpretability
```
ANFIS_Config | Num_Rules | Interpretability_Score | Rule_Complexity | 
Physical_Consistency | Redundancy_Score |
```

#### Sheet 14: Hyperparameter Logs
```
Model_ID | Param_Name | Param_Value | Default_Value | Tuned | 
Tuning_Method | Performance_Impact |
```

#### Sheet 15: Error Analysis
```
Model_ID | MAE_Train | MAE_Val | MAE_Test | Overfitting_Index | 
Underfitting_Index | Bias | Variance |
```

```matlab
% MATLAB .mat file structure
data = struct();

% Input-Output configuration
data.config = struct();
data.config.n_inputs = 3;              % Input sayısı
data.config.n_outputs = 1;             % Output sayısı
data.config.input_names = {'A', 'Z', 'N'};
data.config.output_names = {'MM'};

% Training data
data.train = struct();
data.train.X = X_train;                % [n_samples × n_inputs]
data.train.y = y_train;                % [n_samples × n_outputs]

% Validation data
data.check = struct();
data.check.X = X_check;
data.check.y = y_check;

% Test data
data.test = struct();
data.test.X = X_test;
data.test.y = y_test;

% Feature information
data.features = struct();
data.features.names = feature_names;
data.features.count = length(feature_names);
data.features.set_name = 'Basic';      % Basic/Extended/Full/etc.

% Scaling parameters
data.scaler = struct();
data.scaler.method = 'StandardScaler'; % or 'RobustScaler'
data.scaler.mean = scaler_mean;
data.scaler.std = scaler_std;

% Metadata
data.metadata = struct();
data.metadata.target = 'MM';
data.metadata.nucleus_count = 75;
data.metadata.scenario = 'S70';
data.metadata.split_ratio = [0.70, 0.15, 0.15];
data.metadata.generation_date = datestr(now);
data.metadata.excluded_count = 12;
```

### Yeni Dosya İsimlendirme Kuralları (Feature-Aware)

**Standart Format**:
```
{TARGET}_{SIZE}_{SCENARIO}_{FEATURE_SET}_{SCALING}_{SAMPLING}
```

**Örnekler**:
```
MM_75_S70_Basic_Standard_Random.csv
QM_100_S80_Extended_Robust_Stratified.csv
Beta_2_150_S70_Basic_Beta2_Standard_Random.csv
MM_QM_200_S80_Full_Standard_Stratified.csv
```

**Advanced Format** (4+ giriş kombinasyonları için):
```
{TARGET}_{SIZE}_{SCENARIO}_{INPUT_CONFIG}_{FEATURE_SET}_{SCALING}_{SAMPLING}
```

**Örnekler**:
```
# 3 giriş → 1 çıkış (MM)
MM_100_S70_3In1Out_Extended_Standard_Random.csv

# 3 giriş → 2 çıkış (MM, QM)
MMQM_100_S70_3In2Out_Extended_Standard_Random.csv

# 4 giriş → 1 çıkış (Beta_2)
Beta2_100_S70_4In1Out_Extended_Beta2_Standard_Random.csv

# Advanced: 5+ giriş
MM_150_S80_5InAdv_Full_Standard_Stratified.csv
```

**MAT Format** (ANFIS için):
```
{TARGET}_{SIZE}_{SCENARIO}_{SCALING}_{SAMPLING}.mat
```

İçerik:
```matlab
% MATLAB structure
data.X_train     % Training features
data.y_train     % Training targets
data.X_check     % Validation features
data.y_check     % Validation targets
data.X_test      % Test features
data.y_test      % Test targets
data.feature_names   % Feature names
data.scaler_params   % Scaler parameters
data.metadata        % Dataset metadata
```

**Excel Format** (Raporlar):
```
{TARGET}_{SIZE}_{SCENARIO}_report.xlsx
```

**Alt Klasör Yapısı**:
```
generated_datasets/
├── {TARGET}/              # Her target için
│   ├── csv/               # CSV dosyaları
│   ├── mat/               # MAT dosyaları (ANFIS için)
│   ├── excel/             # Excel raporları
│   └── excluded/          # Filtrelenen çekirdekler
└── metadata/
    └── naming_convention.json
```

### MAT Dosya Yapısı (ANFIS için)

```matlab
% MATLAB .mat file structure
data = struct();

% Input-Output configuration
data.config = struct();
data.config.n_inputs = 3;              % Input sayısı
data.config.n_outputs = 1;             % Output sayısı
data.config.input_names = {'A', 'Z', 'N'};
data.config.output_names = {'MM'};

% Training data
data.train = struct();
data.train.X = X_train;                % [n_samples × n_inputs]
data.train.y = y_train;                % [n_samples × n_outputs]

% Validation data
data.check = struct();
data.check.X = X_check;
data.check.y = y_check;

% Test data
data.test = struct();
data.test.X = X_test;
data.test.y = y_test;

% Feature information
data.features = struct();
data.features.names = feature_names;
data.features.count = length(feature_names);
data.features.set_name = 'Basic';      % Basic/Extended/Full/etc.

% Scaling parameters
data.scaler = struct();
data.scaler.method = 'StandardScaler'; % or 'RobustScaler'
data.scaler.mean = scaler_mean;
data.scaler.std = scaler_std;

% Metadata
data.metadata = struct();
data.metadata.target = 'MM';
data.metadata.nucleus_count = 75;
data.metadata.scenario = 'S70';
data.metadata.split_ratio = [0.70, 0.15, 0.15];
data.metadata.generation_date = datestr(now);
data.metadata.excluded_count = 12;
```

---

## 📊 ANFIS Detaylı Raporlama

### ANFIS Training Output Yapısı

Her ANFIS eğitimi için aşağıdaki dosyalar oluşturulur:

```
anfis_results/
├── MM/
│   ├── MM_75_S70_ANFIS_Compact_GAU2MF/
│   │   ├── trained_fis.fis                    # Trained FIS file
│   │   ├── training_report.json               # JSON metrics
│   │   ├── training_report.xlsx               # Excel detailed (10 sheets)
│   │   ├── rules_analysis.xlsx                # Rule interpretability
│   │   │
│   │   ├── visualizations/
│   │   │   ├── png/                           # TEZ için (300 DPI)
│   │   │   │   ├── learning_curve.png
│   │   │   │   ├── membership_functions.png
│   │   │   │   ├── rule_surface.png
│   │   │   │   ├── error_distribution.png
│   │   │   │   ├── predictions_vs_actual.png
│   │   │   │   ├── residual_plot.png
│   │   │   │   ├── rule_activation_heatmap.png
│   │   │   │   └── convergence_analysis.png
│   │   │   │
│   │   │   └── html/                          # İNTERAKTİF ANALİZ
│   │   │       ├── learning_curve.html        # ✅ Zoom, hover epoch
│   │   │       ├── membership_functions.html  # ✅ Slider ile MF explore
│   │   │       ├── rule_surface.html          # ✅ 3D rotate, zoom
│   │   │       ├── error_distribution.html    # ✅ Bin adjustment
│   │   │       ├── predictions_vs_actual.html # ✅ Nucleus hover
│   │   │       ├── residual_plot.html         # ✅ Outlier highlight
│   │   │       ├── rule_activation_heatmap.html # ✅ Rule details
│   │   │       └── convergence_analysis.html  # ✅ Epoch slider
│   │   │
│   │   └── outliers.csv                       # Detected outliers
│   │
│   └── MM_75_S70_ANFIS_Standard_GEN2MF/
│       └── ... (same dual structure)
│
└── metadata/
    ├── anfis_master_report.xlsx               # All ANFIS results
    ├── anfis_comparison.json                  # Config comparison
    │
    └── comparison_plots/
        ├── png/                               # Static comparison
        │   ├── config_comparison_bar.png
        │   ├── feature_set_comparison.png
        │   ├── convergence_comparison.png
        │   └── performance_radar.png
        │
        └── html/                              # Interactive comparison
            ├── config_comparison_bar.html     # ✅ Sort, filter
            ├── feature_set_comparison.html    # ✅ Toggle configs
            ├── convergence_comparison.html    # ✅ Multi-line
            └── performance_radar.html         # ✅ 6-axis interactive
```

**KRITIK**: Tüm .png dosyalarının .html versiyonu da oluşturulur!

### ANFIS Training Report (Excel)

**Sheet 1: Training Summary**
```
Metric                  | Value
------------------------|------------------
ANFIS Config            | GAU2MF
Input Features          | A, Z, N
Feature Set             | ANFIS_Compact
Number of Inputs        | 3
Number of Outputs       | 1
Number of Rules         | 32
Number of Parameters    | 256
Epochs Completed        | 100 / 100
Converged              | Yes
Final Training Error    | 0.0123
Training Time           | 78.3 seconds
Training Time per Epoch | 0.783 seconds
```

**Sheet 2: Performance Metrics**
```
Dataset  | RMSE  | MAE   | MAPE  | R²    | Max Error |
---------|-------|-------|-------|-------|-----------|
Train    | 0.38  | 0.28  | 7.1%  | 0.94  | 1.23      |
Check    | 0.42  | 0.31  | 8.2%  | 0.91  | 1.45      |
Test     | 0.48  | 0.35  | 9.5%  | 0.88  | 1.67      |
```

**Sheet 3: Rule Base Details**
```
Rule_ID | Input_1_MF | Input_2_MF | Input_3_MF | Output_Type | Parameters | Activation_Freq |
--------|------------|------------|------------|-------------|------------|-----------------|
Rule_1  | A_low      | Z_low      | N_low      | Linear      | [0.12, 0.45, -0.23, 1.56] | 0.234 |
Rule_2  | A_low      | Z_low      | N_high     | Linear      | [0.08, 0.38, -0.18, 1.42] | 0.121 |
...
Rule_32 | A_high     | Z_high     | N_high     | Linear      | [0.18, 0.67, -0.45, 2.34] | 0.312 |
```

**Sheet 4: Membership Functions**
```
Input    | MF_Name  | MF_Type  | Parameters           | Range      | Overlap  |
---------|----------|----------|----------------------|------------|----------|
A        | A_low    | gaussmf  | [σ=15.2, c=50.3]     | [20, 80]   | 0.35     |
A        | A_high   | gaussmf  | [σ=18.7, c=180.5]    | [140, 220] | 0.42     |
Z        | Z_low    | gaussmf  | [σ=8.3, c=25.1]      | [10, 40]   | 0.28     |
Z        | Z_high   | gaussmf  | [σ=12.4, c=75.8]     | [55, 95]   | 0.38     |
N        | N_low    | gaussmf  | [σ=10.1, c=30.2]     | [15, 45]   | 0.31     |
N        | N_high   | gaussmf  | [σ=15.6, c=110.4]    | [85, 135]  | 0.45     |
```

**Sheet 5: Physical Rule Interpretation**
```
Rule_ID | Physical_Region              | Nuclear_Type      | Interpretation          | Confidence |
--------|------------------------------|-------------------|-------------------------|------------|
Rule_1  | Light nuclei (A<80)          | Spherical         | Low MM, stable          | High       |
Rule_7  | Magic regions (Z≈50, N≈82)   | Closed shell      | Very low MM             | Very High  |
Rule_15 | Medium-heavy deformed (A>140)| Prolate           | Enhanced MM             | Medium     |
Rule_24 | Neutron-rich (N>>Z)          | Asymmetric        | Negative MM trend       | Medium     |
Rule_32 | Heavy nuclei (A>200)         | Strongly deformed | High MM variance        | Low        |
```

**Sheet 6: Parameter Evolution**
```
Epoch | Training_Error | Validation_Error | Param_1 | Param_2 | ... | Param_256 |
------|----------------|------------------|---------|---------|-----|-----------|
1     | 0.1234         | 0.1456           | 0.001   | 0.002   | ... | 0.003     |
10    | 0.0845         | 0.0987           | 0.045   | 0.056   | ... | 0.067     |
50    | 0.0234         | 0.0298           | 0.112   | 0.134   | ... | 0.145     |
100   | 0.0123         | 0.0176           | 0.145   | 0.167   | ... | 0.178     |
```

**Sheet 7: Error Analysis per Nucleus**
```
Nucleus  | A   | Z  | N   | Actual_MM | Predicted_MM | Absolute_Error | Relative_Error | Is_Outlier |
---------|-----|----|----|-----------|--------------|----------------|----------------|------------|
7wHf176  | 176 | 72 | 104| 0.340     | 0.328        | 0.012          | 3.5%           | No         |
8wTa181  | 181 | 73 | 108| 2.370     | 2.415        | 0.045          | 1.9%           | No         |
9wW182   | 182 | 74 | 108| 0.117     | 0.234        | 0.117          | 100.0%         | Yes        |
...
```

**Sheet 8: Convergence Analysis**
```
Metric                    | Value    | Status     |
--------------------------|----------|------------|
Epochs to 10% error       | 12       | Fast       |
Epochs to 5% error        | 28       | Normal     |
Epochs to 1% error        | 87       | Slow       |
Final convergence rate    | 0.0001   | Converged  |
Stuck in local minima?    | No       | Good       |
Overfitting detected?     | No       | Good       |
```

**Sheet 9: Sensitivity Analysis**
```
Input_Feature | Impact_Score | Sensitivity | Most_Sensitive_Range | Least_Sensitive_Range |
--------------|--------------|-------------|----------------------|-----------------------|
A             | 0.342        | High        | [100, 150]           | [20, 50]              |
Z             | 0.287        | Medium      | [40, 60]             | [10, 25]              |
N             | 0.156        | Low         | [80, 120]            | [15, 40]              |
```

**Sheet 10: Rule Activation Heatmap**
```
(Matrix showing which rules activate for different input regions)
         | A_low,Z_low | A_low,Z_high | A_high,Z_low | A_high,Z_high |
---------|-------------|--------------|--------------|---------------|
N_low    | 0.23 (R1)   | 0.15 (R5)    | 0.18 (R9)    | 0.12 (R13)    |
N_medium | 0.19 (R2)   | 0.21 (R6)    | 0.24 (R10)   | 0.17 (R14)    |
N_high   | 0.12 (R3)   | 0.08 (R7)    | 0.31 (R11)   | 0.28 (R15)    |
```

### ANFIS Comparison Report

**Master Report**: `anfis_comparison_master.xlsx`

**Sheet 1: Configuration Comparison**
```
Config_Name | MF_Type  | Num_Rules | Num_Params | Train_R² | Test_R² | Training_Time | Interpretability |
------------|----------|-----------|------------|----------|---------|---------------|------------------|
GAU2MF      | gaussmf  | 32        | 256        | 0.94     | 0.88    | 78.3s         | High             |
GEN2MF      | gbellmf  | 32        | 384        | 0.95     | 0.89    | 95.7s         | Medium           |
TRI2MF      | trimf    | 32        | 192        | 0.92     | 0.86    | 65.2s         | Very High        |
SUBR05      | subclust | 45        | 720        | 0.93     | 0.87    | 112.4s        | High             |
```

**Sheet 2: Feature Set Comparison**
```
Feature_Set      | Num_Features | Best_Config | Best_R² | Avg_R² | Training_Time |
-----------------|--------------|-------------|---------|--------|---------------|
ANFIS_Compact    | 5            | GAU2MF      | 0.88    | 0.85   | 78.3s         |
ANFIS_Standard   | 8            | GEN2MF      | 0.91    | 0.88   | 125.6s        |
ANFIS_Extended   | 10           | SUBR05      | 0.92    | 0.89   | 187.9s        |
```

**Sheet 3: Target-wise Performance**
```
Target  | Best_FeatureSet | Best_Config | R²    | RMSE  | Rules | Interpretability |
--------|-----------------|-------------|-------|-------|-------|------------------|
MM      | Standard        | GEN2MF      | 0.91  | 0.42  | 256   | Medium           |
QM      | Extended        | SUBR05      | 0.87  | 0.18  | 45    | High             |
Beta_2  | Compact         | GAU2MF      | 0.89  | 0.035 | 32    | High             |
```

**Sheet 4: Computational Efficiency**
```
Config     | Rules/Second | Params/Second | Memory_MB | CPU_Usage_Avg | GPU_Usage |
-----------|--------------|---------------|-----------|---------------|-----------|
GAU2MF     | 0.41         | 3.27          | 128       | 45%           | N/A       |
GEN2MF     | 0.33         | 4.01          | 156       | 52%           | N/A       |
SUBR05     | 0.40         | 6.41          | 187       | 58%           | N/A       |
```

---

## 🎯 İsimlendirme Komponenetleri

**TARGET** (4 seçenek):
- `MM`: Magnetic Moment
- `QM`: Quadrupole Moment  
- `MM_QM`: Dual target (hem MM hem QM)
- `Beta_2`: Deformation parameter

**SIZE** (5 seçenek):
- `75`: 75 çekirdek
- `100`: 100 çekirdek
- `150`: 150 çekirdek
- `200`: 200 çekirdek
- `ALL`: Tüm çekirdekler (~267)

**SCENARIO** (2 seçenek):
- `S70`: 70% train, 15% check, 15% test
- `S80`: 80% train, 10% check, 10% test

**SCALING** (3 seçenek):
- `Standard`: StandardScaler
- `Robust`: RobustScaler (outlier'lara daha dayanıklı)
- `None`: Scaling yok (ham veri)

**SAMPLING** (2 seçenek):
- `Random`: Random sampling
- `Stratified`: Stratified sampling (target değerine göre)

### Toplam Kombinasyon Sayısı

```python
Total = Targets × Sizes × Scenarios × Scaling × Sampling
Total = 4 × 5 × 2 × 3 × 2
Total = 240 dataset
```

**Gerçekte**: Her target için farklı özellik kombinasyonları var, bu yüzden daha fazla olabilir.

### Boyutlandırma Mantığı

**Neden 75, 100, 150, 200?**

1. **75**: Küçük veri setlerinde model davranışını test etmek
2. **100**: Minimum makul eğitim seti boyutu
3. **150**: Orta ölçekli deneyler
4. **200**: Büyük ölçekli eğitim
5. **ALL**: Maksimum veri kullanımı (~267)

**⚠️ Not**: İlk tasarımda 120 ve 250 de vardı ama kaldırıldı çünkü:
- 120: 100 ve 150 arasında yeterince fark yok
- 250: ALL'a çok yakın, gereksiz

### Senaryo Sistemleri (S70 / S80)

**Amaç**: Train/Validation/Test oranlarını değiştirerek model davranışını analiz etmek

```python
SCENARIOS = {
    'S70': (0.70, 0.15, 0.15),  # 70% train, 15% check, 15% test
    'S80': (0.80, 0.10, 0.10)   # 80% train, 10% check, 10% test
}
```

**S70 (70-15-15)**:
- **Avantajlar**: 
  - Daha fazla validation/test verisi
  - Daha güvenilir performans değerlendirmesi
  - Overfitting'i daha iyi tespit eder
- **Dezavantajlar**:
  - Daha az eğitim verisi
  - Küçük datasette sorun olabilir

**S80 (80-10-10)**:
- **Avantajlar**:
  - Daha fazla eğitim verisi
  - Daha iyi öğrenme (özellikle küçük datasette)
  - Daha stabil eğitim
- **Dezavantajlar**:
  - Daha az test verisi
  - Overfitting riski biraz daha yüksek

**Karşılaştırma**:

| Metrik | S70 | S80 | Hangisi İyi? |
|--------|-----|-----|--------------|
| Eğitim Verisi | Az | Çok | S80 |
| Test Güvenilirliği | Yüksek | Orta | S70 |
| Overfitting Tespiti | İyi | Orta | S70 |
| Küçük Dataset (<100) | Zor | İyi | S80 |
| Büyük Dataset (>150) | İyi | İyi | Her ikisi de |

**Proje Kullanımı**:
- Her dataset hem S70 hem S80 ile oluşturulur
- Toplam dataset sayısı 2× artar
- Model performansı her iki senaryoda da test edilir

**Cross-validation ile test**:
- Her boyut için 5-fold CV yapılacak
- Performans trendleri analiz edilecek
- Optimal boyut belirlenecek

---

## ✅ Kalite Kontrol

### 1. Missing Value Analizi

```python
def check_missing_values(df):
    """Her sütun için missing value sayısı"""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    report = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    
    return report[report['Missing Count'] > 0]
```

**Raporlama**:
```
          Missing Count  Percentage
Q                    67       25.09%
Beta_2               23        8.61%
MM                   12        4.49%
```

### 2. Outlier Detection

**Yöntem**: IQR (Interquartile Range) method

```python
def detect_outliers_iqr(df, column, threshold=3.0):
    """
    IQR yöntemiyle outlier tespiti
    
    Outlier tanımı: Q1 - threshold×IQR veya Q3 + threshold×IQR dışında
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    logger.info(f"{column}: {len(outliers)} outlier (threshold={threshold})")
    
    return outliers
```

**Threshold Seçimi**:
- `threshold=1.5`: Agresif (daha çok outlier)
- `threshold=3.0`: **Varsayılan** (dengelenmiş)
- `threshold=5.0`: Liberal (az outlier)

### 3. Fiziksel Tutarlılık Kontrolleri

```python
def validate_physical_constraints(df):
    """Fiziksel kısıtlamaları kontrol et"""
    
    issues = []
    
    # 1. A = Z + N kontrolü
    inconsistent = df[df['A'] != df['Z'] + df['N']]
    if len(inconsistent) > 0:
        issues.append(f"A≠Z+N: {len(inconsistent)} çekirdek")
    
    # 2. Spin değerleri (0, 0.5, 1, 1.5, ...)
    invalid_spin = df[~df['SPIN'].apply(lambda x: (2*x) % 1 == 0)]
    if len(invalid_spin) > 0:
        issues.append(f"Invalid spin: {len(invalid_spin)} çekirdek")
    
    # 3. Parite sadece ±1
    invalid_parity = df[~df['PARITY'].isin([1, -1])]
    if len(invalid_parity) > 0:
        issues.append(f"Invalid parity: {len(invalid_parity)} çekirdek")
    
    # 4. Odd-A ise MM≠0 kontrolü
    odd_a = df['A'] % 2 == 1
    zero_mm_odd = df[odd_a & (df['MM'] == 0)]
    if len(zero_mm_odd) > 0:
        issues.append(f"Odd-A but MM=0: {len(zero_mm_odd)} çekirdek (FİZİKSEL HATA)")
    
    # 5. Magic numbers yakınında deformation kontrolü
    # Magic çekirdeklerin |β₂| < 0.1 olmalı
    magic_mask = (df['Z_magic_dist'] < 2) & (df['N_magic_dist'] < 2)
    deformed_magic = df[magic_mask & (df['Beta_2'].abs() > 0.1)]
    if len(deformed_magic) > 0:
        issues.append(f"Magic but deformed: {len(deformed_magic)} çekirdek")
    
    return issues
```

### 4. Range Checks

```python
VALID_RANGES = {
    'A': (1, 300),
    'Z': (1, 120),
    'N': (0, 180),
    'SPIN': (0, 15),
    'MM': (-10, 10),       # µ_N
    'Q': (-5, 5),          # barn
    'Beta_2': (-0.5, 0.5)
}

def check_value_ranges(df):
    """Değerlerin fiziksel aralıklarda olup olmadığını kontrol et"""
    
    out_of_range = {}
    
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df.columns:
            invalid = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(invalid) > 0:
                out_of_range[col] = invalid
    
    return out_of_range
```

### 5. Correlation Analysis

```python
def analyze_feature_correlations(df, threshold=0.95):
    """
    Yüksek korelasyonlu özellik çiftlerini bul
    
    Amaç: Redundant features'ları tespit etmek
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Upper triangle (kendisiyle ve aşağı üçgeni atla)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Yüksek korelasyonlu çiftler
    high_corr = [
        (col, row, corr_matrix.loc[row, col])
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > threshold
    ]
    
    return high_corr
```

### 6. Data Quality Report

Her dataset için otomatik kalite raporu oluşturulur:

```python
def generate_quality_report(df, target, output_path):
    """Kapsamlı kalite raporu oluştur"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'target': target,
        'n_samples': len(df),
        'n_features': len(df.columns),
        
        'missing_values': check_missing_values(df).to_dict(),
        'outliers': detect_all_outliers(df).to_dict(),
        'physical_constraints': validate_physical_constraints(df),
        'range_violations': check_value_ranges(df),
        'high_correlations': analyze_feature_correlations(df),
        
        'statistics': {
            'mean': df.describe().loc['mean'].to_dict(),
            'std': df.describe().loc['std'].to_dict(),
            'min': df.describe().loc['min'].to_dict(),
            'max': df.describe().loc['max'].to_dict()
        }
    }
    
    # JSON olarak kaydet
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Kalite raporu kaydedildi: {output_path}")
    
    return report
```

### 7. Excluded Nuclei Tracking (Önemli!)

**Amaç**: Hangi çekirdeklerin neden filtrelendiğini kaydetmek

```python
class ExcludedNucleiTracker:
    """Filtrelenen çekirdekleri takip eder"""
    
    def __init__(self):
        self.excluded = []
    
    def add_exclusion(self, nucleus, reason, target, details=None):
        """
        Filtrelenen çekirdeği kaydet
        
        Args:
            nucleus: Çekirdek adı (örn: '7wHf176')
            reason: Filtreleme nedeni
            target: Hangi target için filtrelendi
            details: Ek detaylar
        """
        self.excluded.append({
            'NUCLEUS': nucleus,
            'Target': target,
            'Reason': reason,
            'Details': details or {},
            'Timestamp': datetime.now().isoformat()
        })
    
    def save_to_excel(self, output_path):
        """Excel dosyasına kaydet"""
        df = pd.DataFrame(self.excluded)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Tüm exclusions
            df.to_excel(writer, sheet_name='All_Excluded', index=False)
            
            # Reason'a göre gruplanmış
            reason_summary = df.groupby('Reason').size().reset_index(name='Count')
            reason_summary.to_excel(writer, sheet_name='By_Reason', index=False)
            
            # Target'a göre gruplanmış
            target_summary = df.groupby('Target').size().reset_index(name='Count')
            target_summary.to_excel(writer, sheet_name='By_Target', index=False)
            
            # Her reason için detaylı liste
            for reason in df['Reason'].unique():
                reason_df = df[df['Reason'] == reason]
                sheet_name = f"Reason_{reason[:25]}"  # Max 31 char
                reason_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"✓ Excluded nuclei report: {output_path}")
```

**Filtreleme Nedenleri**:

| Reason Code | Description | Example |
|-------------|-------------|---------|
| `MISSING_TARGET` | Hedef değişken eksik | MM=NaN |
| `QM_REQUIRED` | QM gerekli ama yok | QM target için Q=NaN |
| `ODD_A_MM_ZERO` | Odd-A ama MM=0 (fiziksel hata) | A=177, MM=0 |
| `OUTLIER_REMOVED` | Outlier tespit edildi | \|z-score\| > 3.5 |
| `PHYSICAL_VIOLATION` | Fiziksel kısıt ihlali | A ≠ Z+N |
| `INVALID_SPIN` | Geçersiz spin değeri | Spin=3.2 (yarım tam sayı değil) |
| `INVALID_PARITY` | Geçersiz parite | Parity=0 |
| `RANGE_VIOLATION` | Değer aralığı dışında | MM=15 (>10 limit) |
| `INSUFFICIENT_FEATURES` | Yeterli özellik yok | >50% feature eksik |
| `DUPLICATE` | Duplikasyon | Aynı A,Z,N |

**Excel Rapor Yapısı**:
```
excluded_nuclei_report.xlsx
├── Sheet 1: All_Excluded (Tüm filtrelenmiş çekirdekler)
│   ├── NUCLEUS
│   ├── A, Z, N
│   ├── Target
│   ├── Reason
│   ├── Details
│   └── Timestamp
├── Sheet 2: By_Reason (Reason'a göre özet)
│   ├── Reason
│   └── Count
├── Sheet 3: By_Target (Target'a göre özet)
│   ├── Target
│   └── Count
├── Sheet 4: Reason_MISSING_TARGET
├── Sheet 5: Reason_QM_REQUIRED
├── Sheet 6: Reason_ODD_A_MM_ZERO
└── ... (her reason için bir sheet)
```

**Kullanım Örneği**:

```python
# Tracker oluştur
tracker = ExcludedNucleiTracker()

# Filtreleme sırasında kaydet
for idx, row in df.iterrows():
    nucleus = row['NUCLEUS']
    
    # QM kontrolü
    if target == 'QM' and pd.isna(row['Q']):
        tracker.add_exclusion(
            nucleus=nucleus,
            reason='QM_REQUIRED',
            target='QM',
            details={'Q_value': 'NaN'}
        )
        continue
    
    # Odd-A MM=0 kontrolü
    if row['A'] % 2 == 1 and row['MM'] == 0:
        tracker.add_exclusion(
            nucleus=nucleus,
            reason='ODD_A_MM_ZERO',
            target=target,
            details={'A': row['A'], 'MM': 0}
        )
        continue
    
    # Outlier kontrolü
    if is_outlier(row['MM']):
        tracker.add_exclusion(
            nucleus=nucleus,
            reason='OUTLIER_REMOVED',
            target=target,
            details={'MM': row['MM'], 'z_score': calculate_z_score(row['MM'])}
        )
        continue

# Dataset generation sonunda kaydet
tracker.save_to_excel('generated_datasets/metadata/excluded_nuclei_report.xlsx')

# İstatistikleri yazdır
print(f"Total excluded: {len(tracker.excluded)}")
print(f"By target: {df.groupby('Target').size()}")
print(f"By reason: {df.groupby('Reason').size()}")
```

**Örnek Çıktı**:
```
Total excluded: 89 nuclei

By target:
  MM:       12 (4.5%)
  QM:       67 (25.1%)
  MM_QM:    67 (25.1%)
  Beta_2:   23 (8.6%)

By reason:
  MISSING_TARGET:      35 (39.3%)
  QM_REQUIRED:         32 (36.0%)
  ODD_A_MM_ZERO:       8 (9.0%)
  OUTLIER_REMOVED:     10 (11.2%)
  PHYSICAL_VIOLATION:  4 (4.5%)
```

---

## 🔧 Feature Kombinasyonları

### Standart Feature Setleri (MM, QM, MM_QM için)

```python
STANDARD_FEATURE_SETS = {
    'Basic': [
        'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR'
    ],  # 6 features
    
    'Basic_SEMF': [
        'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
        'BE_volume', 'BE_surface', 'BE_coulomb', 'BE_asymmetry', 'BE_pairing',
        'BE_per_A'
    ],  # 12 features
    
    'Basic_Shell': [
        'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
        'Z_magic_dist', 'N_magic_dist', 'Z_shell_gap', 'N_shell_gap',
        'magic_character', 'is_doubly_magic'
    ],  # 12 features
    
    'Extended': [
        'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
        # SEMF (6)
        'BE_volume', 'BE_surface', 'BE_coulomb', 'BE_asymmetry', 'BE_pairing', 'BE_per_A',
        # Shell Model (6)
        'Z_magic_dist', 'N_magic_dist', 'Z_shell_gap', 'N_shell_gap',
        'magic_character', 'is_doubly_magic',
        # Deformation (3)
        'Beta_2_estimated', 'deformation_magnitude', 'spherical_index'
    ],  # 21 features
    
    'Full': [
        # Basic (6)
        'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
        # SEMF (11)
        'BE_volume', 'BE_surface', 'BE_coulomb', 'BE_asymmetry', 'BE_pairing',
        'BE_total', 'BE_per_A', 'nuclear_radius', 'S_n_approx', 'S_p_approx',
        'mass_excess',
        # Shell Model (10)
        'Z_nearest_magic', 'N_nearest_magic', 'Z_magic_dist', 'N_magic_dist',
        'Z_shell_gap', 'N_shell_gap', 'Z_valence', 'N_valence',
        'magic_character', 'is_doubly_magic',
        # Deformation (4)
        'Beta_2_estimated', 'deformation_type', 'deformation_magnitude', 'spherical_index',
        # Schmidt (2)
        'schmidt_nearest', 'schmidt_deviation',
        # Collective (4)
        'collective_parameter', 'rotational_constant',
        'vibrational_frequency', 'nucleus_collective_type'
    ],  # 44 features
    
    'Physics_Optimized': [
        # En önemli fiziksel parametreler (SHAP analizi sonrası)
        'A', 'Z', 'N', 'SPIN',
        'BE_per_A', 'BE_pairing',
        'Z_magic_dist', 'N_magic_dist', 'magic_character',
        'Beta_2_estimated', 'spherical_index',
        'schmidt_nearest'
    ]  # 12 features (en iyi performans)
}
```

### Beta_2 İçin Özel Feature Setleri

```python
BETA2_FEATURE_SETS = {
    'Basic_Beta2': [
        'A', 'Z', 'N', 'SPIN', 'PARITY',
        'Z_magic_dist', 'N_magic_dist'
    ],  # 7 features
    
    'Extended_Beta2': [
        'A', 'Z', 'N', 'SPIN', 'PARITY',
        'BE_per_A', 'BE_asymmetry',
        'Z_magic_dist', 'N_magic_dist', 'magic_character',
        'Z_valence', 'N_valence'
    ],  # 12 features
    
    'Full_Beta2': [
        'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
        'BE_volume', 'BE_surface', 'BE_asymmetry', 'BE_per_A',
        'Z_magic_dist', 'N_magic_dist', 'Z_shell_gap', 'N_shell_gap',
        'magic_character', 'Z_valence', 'N_valence',
        'collective_parameter', 'nucleus_collective_type'
    ]  # 19 features (QM kullanmadan!)
}
```

### ANFIS İçin Özel Konfigürasyonlar

**ANFIS Feature Selection Kuralları**:
- **Maximum 10 features**: ANFIS curse of dimensionality'den etkilenir
- **Yüksek korelasyonlu features**: Birini seç (redundancy)
- **Fiziksel anlamlılık**: Yorumlanabilir kurallar için

```python
ANFIS_FEATURE_SETS = {
    'ANFIS_Compact': [
        # En önemli 5 feature (SHAP top-5)
        'A', 'Z', 'SPIN', 'magic_character', 'BE_per_A'
    ],  # 5 features → ~32 rules (2^5)
    
    'ANFIS_Standard': [
        # Dengeli 8 feature
        'A', 'Z', 'N', 'SPIN',
        'BE_per_A', 'magic_character',
        'Z_magic_dist', 'Beta_2_estimated'
    ],  # 8 features → ~256 rules (2^8)
    
    'ANFIS_Extended': [
        # Maksimum 10 feature
        'A', 'Z', 'N', 'SPIN', 'PARITY',
        'BE_per_A', 'BE_pairing',
        'magic_character', 'Z_magic_dist',
        'Beta_2_estimated'
    ]  # 10 features → ~1024 rules (2^10, limit!)
}
```

**Rule Explosion Problemi**:

| Feature Count | Grid 2-MF | Grid 3-MF | Pratik Limit |
|---------------|-----------|-----------|--------------|
| 5 | 32 rules | 243 rules | ✅ OK |
| 8 | 256 rules | 6561 rules | ✅ OK |
| 10 | 1024 rules | 59049 rules | ⚠️ Limit |
| 12 | 4096 rules | 531441 rules | ❌ Çok fazla |
| 15 | 32768 rules | 14M rules | ❌ İmkansız |

**Subtractive Clustering** ile bu sorun aşılabilir (otomatik rule generation).

### Feature Importance (SHAP Sonuçlarından)

**MM Prediction için Top-15**:

1. **A** (19.2%): Kütle numarası - en önemli
2. **Z** (17.5%): Proton sayısı
3. **SPIN** (12.8%): Nükleus spini
4. **magic_character** (9.7%): Kabuk yapısı etkisi
5. **BE_per_A** (8.3%): Bağlanma enerjisi/nükleon
6. **Beta_2_estimated** (7.1%): Deformasyon
7. **Z_magic_dist** (5.4%): Magic sayıya uzaklık
8. **N** (4.9%): Nötron sayısı
9. **BE_pairing** (4.2%): Pairing enerjisi
10. **schmidt_nearest** (3.8%): Schmidt moment
11. **N_magic_dist** (3.1%): Magic sayıya uzaklık (N)
12. **Z_shell_gap** (2.7%): Shell gap
13. **spherical_index** (2.4%): Küresellik
14. **PARITY** (2.1%): Parite
15. **P_FACTOR** (1.8%): P faktörü

**QM Prediction için Top-15**:

1. **Z** (21.5%): Elektrik yükü (QM ∝ Z)
2. **Beta_2_estimated** (18.3%): Deformasyon (QM ∝ β₂)
3. **A** (15.7%): Kütle
4. **magic_character** (10.2%): Kabuk etkisi
5. **SPIN** (8.9%): Spin
6. **BE_asymmetry** (6.4%): Asimetri enerjisi
7. **Z_valence** (5.1%): Valans protonlar
8. **N_valence** (4.8%): Valans nötronlar
9. **spherical_index** (4.3%): Küresellik
10. **collective_parameter** (3.7%): Kollektif parametre
11. **Z_magic_dist** (3.2%): Magic uzaklık
12. **N** (2.9%): Nötron sayısı
13. **PARITY** (2.3%): Parite
14. **BE_per_A** (2.1%): BE/A
15. **nucleus_collective_type** (1.8%): Kollektif tip

**Beta_2 Prediction için Top-12**:

1. **magic_character** (22.1%): Magic'ten uzaklık → deformasyon
2. **Z_magic_dist** (18.7%): Z magic uzaklık
3. **N_magic_dist** (17.3%): N magic uzaklık
4. **A** (12.9%): Kütle
5. **Z_valence** (8.4%): Valans protonlar
6. **N_valence** (7.8%): Valans nötronlar
7. **BE_asymmetry** (5.6%): Asimetri
8. **Z** (3.9%): Proton sayısı
9. **N** (3.4%): Nötron sayısı
10. **collective_parameter** (2.8%): Kollektif
11. **SPIN** (2.3%): Spin
12. **BE_per_A** (1.9%): BE/A

### Feature Kombinasyon Stratejisi

**İteratif Seçim**:

```python
def select_features_iteratively(X, y, max_features=15):
    """
    İteratif feature selection
    
    1. Tüm features ile başla
    2. Recursive Feature Elimination (RFE)
    3. SHAP importance'a göre sırala
    4. En iyi N feature'ı seç
    """
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor
    
    # RFE
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=max_features)
    rfe.fit(X, y)
    
    # Selected features
    selected_features = X.columns[rfe.support_]
    
    logger.info(f"✓ {len(selected_features)} features selected")
    return selected_features
```

**AutoML Feature Engineering** (PFAZ 13'te detaylı):
- Polynomial features (degree 2-3)
- Interaction terms (physics-inspired)
- Mathematical transforms (log, sqrt, exp)
- 200+ candidate features → 50-80 selected

---

## 📊 Veri Ölçeklendirme

### Neden StandardScaler?

**Kritik Kısıt**: Spin ve Parite değerleri 0 olamaz!

```python
# ❌ YANLIŞ: MinMaxScaler kullanmak
scaler = MinMaxScaler()  # [0, 1] aralığına ölçekler
# Sorun: 0 değerlerini 0 olarak tutar
# Spin=0 veya Parite=0 FİZİKSEL OLARAK YANLIŞ!

# ✅ DOĞRU: StandardScaler kullanmak
scaler = StandardScaler()  # (x - mean) / std
# 0 değerleri de dönüştürülür, sıfır kalmazlar
```

### StandardScaler Uygulaması

```python
from sklearn.preprocessing import StandardScaler

# Scaler oluştur
scaler = StandardScaler()

# SADECE TRAIN verisi üzerinde fit et
X_train_scaled = scaler.fit_transform(X_train)

# Aynı scaler'ı val ve test'e uygula
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

logger.info("✓ StandardScaler uygulandı")
```

### Kritik Kurallar

1. **Sadece train'de fit**:
   ```python
   # ❌ YANLIŞ
   scaler.fit(X_val)  # Val verisini görmemeli!
   
   # ✅ DOĞRU
   scaler.fit(X_train)  # Sadece train
   ```

2. **Aynı scaler'ı kullan**:
   ```python
   # ❌ YANLIŞ
   scaler_train = StandardScaler().fit(X_train)
   scaler_test = StandardScaler().fit(X_test)  # Farklı scaler!
   
   # ✅ DOĞRU
   scaler = StandardScaler().fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)  # Aynı scaler
   ```

3. **Data leakage'ı önle**:
   ```python
   # ❌ YANLIŞ: Tüm veriyi birlikte scale et
   scaler.fit(np.vstack([X_train, X_val, X_test]))
   
   # ✅ DOĞRU: Sadece train'i fit et
   scaler.fit(X_train)
   ```

### Target Scaling (Opsiyonel)

Bazı modeller için target değişkenini de scale edebiliriz:

```python
def scale_targets(y_train, y_val, y_test):
    """Target değişkenini scale et"""
    
    scaler_y = StandardScaler()
    
    # Reshape (sklearn 2D bekler)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return y_train_scaled, y_val_scaled, y_test_scaled, scaler_y
```

**İnverse transform** (tahminleri geri dönüştürme):
```python
# Scaled predictions → original scale
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
```

### Scaling Metadata

Scaler parametreleri kaydedilir (reproducibility için):

```python
scaling_metadata = {
    'method': 'standard',
    'features_scaled': list(feature_cols),
    'scaler_params': {
        'mean': scaler.mean_.tolist(),
        'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist()
    }
}

with open('scaling_metadata.json', 'w') as f:
    json.dump(scaling_metadata, f, indent=2)
```

---

## 📁 Çıktılar ve Raporlar

### 1. Dataset Files

**Format**: CSV (comma-separated)  
**Encoding**: UTF-8

```
generated_datasets/
└── MM/
    ├── dataset_75_nuclei.csv
    ├── dataset_100_nuclei.csv
    ├── dataset_150_nuclei.csv
    ├── dataset_200_nuclei.csv
    └── dataset_ALL_nuclei.csv
```

**CSV Yapısı**:
```csv
NUCLEUS,A,Z,N,SPIN,PARITY,MM,Beta_2,BE_volume,BE_surface,...
7wHf176,176,72,104,0.0,1,0.34,0.25,2772.0,-654.3,...
...
```

### 2. Metadata Files

#### dataset_catalog.json
```json
{
  "generation_timestamp": "2025-11-23T10:30:00",
  "total_nuclei": 267,
  "targets": ["MM", "QM", "MM_QM", "Beta_2"],
  "nucleus_counts": [75, 100, 150, 200, "ALL"],
  "features": {
    "total": 44,
    "basic": 12,
    "semf": 11,
    "shell_model": 10,
    "deformation": 4,
    "schmidt": 2,
    "collective": 4,
    "woods_saxon": 0,
    "nilsson": 0
  },
  "datasets": [
    {
      "target": "MM",
      "size": 75,
      "path": "MM/dataset_75_nuclei.csv",
      "n_samples": 75,
      "n_features": 44
    },
    ...
  ]
}
```

#### generation_report.json
```json
{
  "pipeline_version": "1.0.0",
  "execution_time_seconds": 125.3,
  "steps_completed": [
    "data_loading",
    "theoretical_calculations",
    "qm_filtering",
    "sampling",
    "quality_control",
    "dataset_saving"
  ],
  "data_quality": {
    "missing_values_pct": 8.5,
    "outliers_detected": 23,
    "physical_constraint_violations": 0
  },
  "filtering_statistics": {
    "MM": {
      "input": 267,
      "after_filter": 255,
      "loss_pct": 4.5
    },
    "QM": {
      "input": 267,
      "after_filter": 200,
      "loss_pct": 25.1
    },
    ...
  }
}
```

### 3. Quality Reports

#### data_quality_report.xlsx

Excel dosyası, 8 sheet:

1. **Summary**: Genel özet
2. **Missing Values**: Eksik değer analizi
3. **Outliers**: Aykırı değer listesi
4. **Physical Constraints**: Fiziksel kontrol sonuçları
5. **Range Violations**: Aralık ihlalleri
6. **Correlations**: Yüksek korelasyonlar
7. **Statistics**: Temel istatistikler
8. **Distributions**: Dağılım grafikleri

### 4. Visualization Outputs

**KRITIK**: Her görselleştirme hem PNG hem HTML formatında üretilir!

```
visualizations/
├── distributions/
│   ├── MM_distribution.png                    # Tez için (300 DPI)
│   ├── MM_distribution.html                   # İnteraktif analiz
│   ├── QM_distribution.png
│   ├── QM_distribution.html
│   ├── Beta_2_distribution.png
│   ├── Beta_2_distribution.html
│   └── feature_distributions.png
│       └── feature_distributions.html
├── correlations/
│   ├── correlation_matrix.png
│   ├── correlation_matrix.html                # Hover ile değerler
│   ├── high_correlations.png
│   └── high_correlations.html
├── outliers/
│   ├── MM_outliers_boxplot.png
│   ├── MM_outliers_boxplot.html               # Zoom, hover
│   ├── QM_outliers_boxplot.png
│   ├── QM_outliers_boxplot.html
│   └── all_outliers_summary.png
│       └── all_outliers_summary.html
└── quality/
    ├── missing_values_bar.png
    ├── missing_values_bar.html                # Detaylı tooltips
    ├── data_completeness.png
    ├── data_completeness.html
    └── feature_importance.png
        └── feature_importance.html            # Sıralama, filtreleme
```

**Görselleştirme Üretim Sistemi**:

```python
class DualVisualizationGenerator:
    """Her görsel için hem PNG hem HTML üretir"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.png_dir = self.output_dir / 'png'
        self.html_dir = self.output_dir / 'html'
        
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dual_plot(self, data, plot_type, title, filename):
        """
        Aynı grafiği hem PNG hem HTML olarak oluştur
        
        Args:
            data: Plot verisi
            plot_type: 'scatter', 'bar', 'line', 'heatmap', etc.
            title: Grafik başlığı
            filename: Dosya adı (uzantısız)
        """
        # 1. PNG için matplotlib
        self._create_png(data, plot_type, title, filename)
        
        # 2. HTML için plotly
        self._create_html(data, plot_type, title, filename)
        
        logger.info(f"✓ Dual visualization: {filename}.png + {filename}.html")
    
    def _create_png(self, data, plot_type, title, filename):
        """PNG (yüksek çözünürlük, tez için)"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        if plot_type == 'scatter':
            ax.scatter(data['x'], data['y'], alpha=0.6)
        elif plot_type == 'bar':
            ax.bar(data['labels'], data['values'])
        elif plot_type == 'line':
            ax.plot(data['x'], data['y'])
        elif plot_type == 'heatmap':
            sns.heatmap(data['matrix'], ax=ax, annot=True, fmt='.2f')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.png_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_html(self, data, plot_type, title, filename):
        """HTML (interaktif, analiz için)"""
        import plotly.graph_objects as go
        import plotly.express as px
        
        if plot_type == 'scatter':
            fig = go.Figure(data=go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                text=data.get('hover_text', None),
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ))
        
        elif plot_type == 'bar':
            fig = go.Figure(data=go.Bar(
                x=data['labels'],
                y=data['values'],
                text=data['values'],
                texttemplate='%{text:.2f}',
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
            ))
        
        elif plot_type == 'line':
            fig = go.Figure(data=go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='lines+markers',
                line=dict(width=2),
                hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ))
        
        elif plot_type == 'heatmap':
            fig = go.Figure(data=go.Heatmap(
                z=data['matrix'],
                x=data.get('x_labels', None),
                y=data.get('y_labels', None),
                colorscale='RdYlBu_r',
                text=data['matrix'],
                texttemplate='%{text:.2f}',
                hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            title_font_size=16,
            hovermode='closest',
            template='plotly_white'
        )
        
        fig.write_html(self.html_dir / f"{filename}.html")

# Kullanım
viz_gen = DualVisualizationGenerator('visualizations')

# Örnek: Distribution plot
viz_gen.create_dual_plot(
    data={'x': df['MM'], 'y': df['QM'], 'hover_text': df['NUCLEUS']},
    plot_type='scatter',
    title='MM vs QM Distribution',
    filename='mm_qm_scatter'
)
```

**PNG vs HTML Karşılaştırması**:

| Özellik | PNG | HTML | Kullanım |
|---------|-----|------|----------|
| **Çözünürlük** | 300 DPI | Vektör | PNG: Tez, PDF |
| **Dosya Boyutu** | ~500KB | ~200KB | HTML: Web, sunumlar |
| **İnteraktif** | ❌ | ✅ Zoom, hover, pan | HTML: Analiz |
| **Animasyon** | ❌ | ✅ | HTML: Time series |
| **Veri İndirme** | ❌ | ✅ CSV export | HTML: Veri paylaşımı |
| **Offline Görüntüleme** | ✅ | ✅ (standalone) | Her ikisi de |
| **LaTeX Uyumlu** | ✅ | ❌ | PNG: Tez yazımı |
| **Presentation** | ✅ | ✅ | Her ikisi de |

**HTML İnteraktif Özellikleri**:

1. **Hover Tooltips**: Her data point için detaylı bilgi
   ```html
   Nucleus: 7wHf176
   A: 176, Z: 72, N: 104
   MM: 0.340 μ_N
   QM: 6.82 barn
   Beta_2: 0.25
   ```

2. **Zoom & Pan**: Grafik üzerinde zoom yapabilme, kaydırma
3. **Legend Toggle**: Seriler açıp kapatılabilir
4. **Data Export**: Plotly toolbar'dan PNG, SVG, CSV indirme
5. **Selection**: Lasso/box ile data point seçimi
6. **Annotations**: Önemli noktalara otomatik etiketler

### 5. Log Files

```
logs/
├── dataset_generation_2025-11-23_10-30-00.log
└── error_log.txt
```

**Log Format**:
```
2025-11-23 10:30:00 - INFO - Starting dataset generation pipeline
2025-11-23 10:30:01 - INFO - ✓ Loaded 267 nuclei from aaa2.txt
2025-11-23 10:30:15 - INFO - ✓ SEMF: 11 features added
2025-11-23 10:30:20 - INFO - ✓ Shell Model: 10 features added
2025-11-23 10:30:45 - INFO - ✓ QM filter (MM): 255/267 nuclei
2025-11-23 10:31:00 - INFO - ✓ Dataset saved: MM/dataset_75_nuclei.csv
...
```

---

## 🚀 Kullanım Kılavuzu

### Basit Kullanım

```python
from dataset_generation_pipeline_v2 import DatasetGenerationPipeline

# Pipeline oluştur
pipeline = DatasetGenerationPipeline(
    source_data_path='aaa2.txt',
    output_base_dir='generated_datasets',
    nucleus_counts=[75, 100, 150, 200, 'ALL'],
    targets=['MM', 'QM', 'MM_QM', 'Beta_2']
)

# Pipeline'ı çalıştır
report = pipeline.run_complete_pipeline()

print(f"✓ {len(pipeline.generated_datasets)} dataset oluşturuldu")
print(f"Süre: {report['total_duration_seconds']:.2f} saniye")
```

### Adım Adım Kullanım

```python
# 1. Ham veriyi yükle
pipeline._load_raw_data()
print(f"✓ {len(pipeline.raw_data)} çekirdek yüklendi")

# 2. Teorik hesaplamalar ekle
pipeline._add_theoretical_calculations()
print(f"✓ {len(pipeline.enriched_data.columns)} özellik")

# 3. QM filtreleme uygula
pipeline._apply_qm_filtering()
print(f"✓ Filtrelenmiş data hazır")

# 4. Dataset'leri oluştur
pipeline._generate_all_datasets()
print(f"✓ {len(pipeline.generated_datasets)} dataset")

# 5. Kalite kontrol
pipeline._run_quality_checks()
print(f"✓ Kalite kontrol tamamlandı")

# 6. Raporları kaydet
pipeline._save_reports_and_visualizations()
print(f"✓ Raporlar kaydedildi")
```

### Özelleştirilmiş Konfigürasyon

```python
# Teorik hesaplamalarda Woods-Saxon ve Nilsson'u aktifleştir
pipeline.theoretical_calc_manager = TheoreticalCalculationsManager(enable_all=True)

# Outlier threshold'u değiştir
pipeline.outlier_handler = OutlierHandler(
    threshold=5.0,  # Daha liberal (varsayılan: 3.0)
    output_dir='quality_reports'
)

# Custom nucleus counts
pipeline.nucleus_counts = [50, 100, 200, 300, 'ALL']

# Sadece belirli targetler
pipeline.targets = ['MM', 'Beta_2']

# Pipeline'ı çalıştır
report = pipeline.run_complete_pipeline()
```

### Tek Bir Dataset Oluşturma

```python
# Sadece MM için 150 çekirdeklik dataset
df_mm = pipeline.generate_single_dataset(
    target='MM',
    n_samples=150
)

print(f"✓ {len(df_mm)} çekirdek, {len(df_mm.columns)} özellik")
df_mm.to_csv('mm_150.csv', index=False)
```

### Quality Check Sonuçlarını Görüntüleme

```python
# Quality report yükle
import json

with open('generated_datasets/metadata/generation_report.json') as f:
    report = json.load(f)

print("DATA KALİTESİ:")
print(f"  Missing values: {report['data_quality']['missing_values_pct']:.1f}%")
print(f"  Outliers: {report['data_quality']['outliers_detected']}")
print(f"  Violations: {report['data_quality']['physical_constraint_violations']}")

print("\nFİLTRELEME:")
for target, stats in report['filtering_statistics'].items():
    print(f"  {target}: {stats['after_filter']}/{stats['input']} "
          f"(kayıp: {stats['loss_pct']:.1f}%)")
```

### Hata Yönetimi

```python
try:
    report = pipeline.run_complete_pipeline()
except FileNotFoundError as e:
    logger.error(f"Veri dosyası bulunamadı: {e}")
except ValueError as e:
    logger.error(f"Veri doğrulama hatası: {e}")
except Exception as e:
    logger.error(f"Beklenmeyen hata: {e}")
    raise
finally:
    # Cleanup
    pipeline._cleanup_temp_files()
```

---

## 📝 Best Practices

### 1. Veri Kaydetme

```python
# ✅ DOĞRU: Index'siz kaydet
df.to_csv('dataset.csv', index=False, encoding='utf-8')

# ❌ YANLIŞ: Index'le kaydet
df.to_csv('dataset.csv', encoding='utf-8')  # İndex sütunu ekler
```

### 2. Missing Value Handling

```python
# ✅ DOĞRU: Target'a göre karar ver
if target == 'MM':
    df = df[df['MM'].notna()]  # Sadece MM olmalı
elif target == 'QM':
    df = df[df['Q'].notna()]   # Sadece QM olmalı

# ❌ YANLIŞ: Tüm missing'leri at
df = df.dropna()  # Çok fazla veri kaybı!
```

### 3. Reproducibility

```python
# ✅ DOĞRU: Random seed kullan
np.random.seed(42)
df_sample = df.sample(n=100, random_state=42)

# ❌ YANLIŞ: Seed yok
df_sample = df.sample(n=100)  # Her seferinde farklı!
```

### 4. Feature Naming

```python
# ✅ DOĞRU: Açıklayıcı isimler
'BE_volume', 'Z_magic_dist', 'schmidt_deviation'

# ❌ YANLIŞ: Kriptik isimler
'bev', 'zmd', 'sd'
```

### 5. Documentation

```python
def calculate_feature(df):
    """
    Özellik hesaplama fonksiyonu
    
    Args:
        df: DataFrame with A, Z, N columns
        
    Returns:
        DataFrame with added feature column
        
    Example:
        df = calculate_feature(df)
    """
    # Kod...
```

---

## ⚠️ Dikkat Edilmesi Gerekenler

### 1. Data Leakage

```python
# ❌ YANLIŞ
scaler.fit(X_all)  # Val/test verisi fit'te!

# ✅ DOĞRU
scaler.fit(X_train)  # Sadece train
```

### 2. Odd-A MM=0 Kontrolü

```python
# ✅ Mutlaka kontrol et
odd_a = df['A'] % 2 == 1
invalid = df[odd_a & (df['MM'] == 0)]
if len(invalid) > 0:
    logger.warning(f"FİZİKSEL HATA: {len(invalid)} odd-A çekirdek MM=0")
```

### 3. QM Filtreleme

```python
# ✅ Target'a özel filtre uygula
if target == 'QM':
    df = df[df['Q'].notna()]
elif target == 'MM':
    # QM gerekli değil
    pass
```

### 4. Memory Management

```python
# ✅ Büyük dataframe'leri temizle
del df_temp
import gc
gc.collect()
```

### 5. File Paths

```python
# ✅ DOĞRU: Path objesi kullan
from pathlib import Path
output_path = Path('output') / 'dataset.csv'

# ❌ YANLIŞ: String concatenation
output_path = 'output' + '/' + 'dataset.csv'
```

---

## 🔍 Troubleshooting

### Problem: "FileNotFoundError: aaa2.txt not found"

**Çözüm**:
```python
# 1. Dosyanın varlığını kontrol et
import os
if not os.path.exists('aaa2.txt'):
    print("❌ aaa2.txt bulunamadı!")

# 2. Doğru dizinde olduğunuzdan emin olun
print(f"Mevcut dizin: {os.getcwd()}")

# 3. Tam path kullanın
pipeline = DatasetGenerationPipeline(
    source_data_path='/tam/yol/aaa2.txt'
)
```

### Problem: "KeyError: 'Q'"

**Çözüm**:
```python
# Sütun isimlerini kontrol et
print(df.columns.tolist())

# Muhtemel sebep: Encoding sorunu
df.columns = df.columns.str.strip()  # Boşlukları temizle
```

### Problem: "MemoryError"

**Çözüm**:
```python
# 1. Chunk'lar halinde işle
for target in targets:
    for size in [75, 100]:  # Küçük gruplar halinde
        generate_dataset(target, size)

# 2. Teorik hesaplamaları kapat
theoretical_manager = TheoreticalCalculationsManager(enable_all=False)

# 3. Gereksiz sütunları sil
df = df[essential_columns]
```

### Problem: "Çok fazla veri kaybı (>50%)"

**Çözüm**:
```python
# 1. Filtreleme logunu kontrol et
logger.setLevel(logging.DEBUG)

# 2. QM filtrelemeyi gevşet
if target == 'Beta_2':
    # QM opsiyonel yap
    df = df[df['Beta_2'].notna()]  # QM şart değil

# 3. Threshold'u artır
outlier_handler = OutlierHandler(threshold=5.0)  # Daha liberal
```

---

## 📚 Referanslar

### Fizik Referansları

1. **SEMF**: Weizsäcker, C. F. von (1935). "Zur Theorie der Kernmassen"
2. **Shell Model**: Mayer, M. G., & Jensen, J. H. D. (1955). "Elementary Theory of Nuclear Shell Structure"
3. **Nilsson Model**: Nilsson, S. G. (1955). "Binding states of individual nucleons in strongly deformed nuclei"
4. **Woods-Saxon**: Woods, R. D., & Saxon, D. S. (1954). "Diffuse Surface Optical Model"

### Python Kütüphaneleri

- `pandas==1.5.3`: Veri manipülasyonu
- `numpy==1.24.3`: Sayısal hesaplamalar
- `scikit-learn==1.3.0`: Preprocessing ve scaling
- `matplotlib==3.7.1`: Görselleştirme
- `seaborn==0.12.2`: İstatistiksel grafikler
- `openpyxl==3.1.2`: Excel okuma/yazma

---

## 📞 İletişim ve Destek

**Proje**: Nükleer Fizik AI Sistemi  
**Faz**: PFAZ 1 - Dataset Generation  
**Versiyon**: 1.0.0  
**Tarih**: 2025-11-23

**Dokümantasyon Güncellemeleri**:
- Bu doküman proje ilerledikçe güncellenecektir
- Son güncelleme: 2025-11-23

---

## 📊 Görselleştirme Stratejisi Özeti

### ZORUNLU: Dual Output Format

**Her görsel hem PNG hem HTML olarak üretilir!**

#### PNG (Statik, Yüksek Kalite)
- **Çözünürlük**: 300 DPI
- **Format**: PNG, 24-bit color
- **Boyut**: ~300-800KB per görsel
- **Kullanım**: 
  - Tez dokümanları
  - PDF raporlar
  - Basılı yayınlar
  - LaTeX entegrasyonu
  - Sunum slide'ları

#### HTML (İnteraktif, Analiz)
- **Çözünürlük**: Vektör (sonsuz zoom)
- **Format**: Standalone HTML + JavaScript
- **Boyut**: ~150-500KB per görsel
- **Kullanım**:
  - Web sunumları
  - İnteraktif analiz
  - Veri keşfi (exploration)
  - Ekip paylaşımı
  - Online raporlar

#### Görsel Sayıları

**Dataset Generation (PFAZ 1)**:
- Distribution plots: 8 × 2 = 16 dosya
- Correlation matrices: 4 × 2 = 8 dosya
- Outlier analyses: 6 × 2 = 12 dosya
- Quality metrics: 6 × 2 = 12 dosya
- **Toplam**: ~48 dosya (24 PNG + 24 HTML)

**AI Training (PFAZ 2)**:
- Per model: 8 × 2 = 16 dosya
- 300 model × 16 = 4,800 dosya
- **Toplam**: ~4,800 dosya (2,400 PNG + 2,400 HTML)

**ANFIS Training (PFAZ 3)**:
- Per ANFIS: 8 × 2 = 16 dosya
- 288 ANFIS × 16 = 4,608 dosya
- **Toplam**: ~4,608 dosya (2,304 PNG + 2,304 HTML)

**Ensemble & Analysis (PFAZ 4-9)**:
- Comprehensive visualizations
- **Toplam**: ~1,000 dosya (500 PNG + 500 HTML)

**Grand Total**: ~10,500 görselleştirme dosyası
- **PNG**: ~5,250 dosya (~2.6GB)
- **HTML**: ~5,250 dosya (~1.3GB)
- **Toplam**: ~3.9GB görsel veri

### HTML İnteraktif Özellikleri

**1. Hover Tooltips** (Tüm grafikler)
```
Nucleus: 7wHf176
A: 176, Z: 72, N: 104
MM: 0.340 μ_N (Predicted: 0.328)
Error: 0.012 (3.5%)
Rule Activated: Rule_7 (Magic region)
```

**2. Zoom & Pan** (Scatter, Line, Surface)
- Mouse wheel: Zoom in/out
- Click-drag: Pan
- Double-click: Reset view

**3. Legend Toggle** (Multi-series)
- Click legend: Show/hide series
- Double-click: Isolate series

**4. Data Export** (Plotly toolbar)
- PNG download (custom resolution)
- SVG download (vector)
- CSV data export

**5. Selection Tools** (Scatter plots)
- Box select: Rectangle selection
- Lasso select: Free-form selection
- Selected data → CSV export

**6. Animations** (Time series)
- Play/pause controls
- Slider for epoch navigation
- Speed adjustment

**7. 3D Controls** (Surface plots)
- Rotate: Click-drag
- Zoom: Scroll
- Pan: Right-click-drag
- Reset camera: Home button

### Görselleştirme Kalite Kontrol

```python
class VisualizationQualityChecker:
    """Her PNG için HTML varlığını kontrol eder"""
    
    def check_dual_output(self, viz_dir):
        """
        PNG ve HTML dosyalarının eşleştiğini doğrula
        """
        png_dir = viz_dir / 'png'
        html_dir = viz_dir / 'html'
        
        png_files = set(f.stem for f in png_dir.glob('*.png'))
        html_files = set(f.stem for f in html_dir.glob('*.html'))
        
        missing_html = png_files - html_files
        missing_png = html_files - png_files
        
        if missing_html:
            logger.warning(f"❌ PNG without HTML: {missing_html}")
        if missing_png:
            logger.warning(f"❌ HTML without PNG: {missing_png}")
        
        if not missing_html and not missing_png:
            logger.info(f"✅ All visualizations have dual output ({len(png_files)} pairs)")
        
        return len(missing_html) == 0 and len(missing_png) == 0

# Usage
checker = VisualizationQualityChecker()
assert checker.check_dual_output('visualizations/'), "Missing dual outputs!"
```

---

## ✅ Checklist

Başarılı bir dataset generation için:

- [ ] aaa2.txt dosyası mevcut ve doğru formatta
- [ ] Tüm gerekli Python kütüphaneleri yüklü (matplotlib, plotly, seaborn)
- [ ] Output dizinleri oluşturuldu (standard/, advanced/, anfis_optimized/)
- [ ] Teorik hesaplama parametreleri doğrulandı
- [ ] QM filtreleme stratejisi belirlendi (target-specific)
- [ ] Dataset boyutları konfigüre edildi (75, 100, 150, 200, ALL)
- [ ] S70/S80 senaryoları aktif
- [ ] Feature kombinasyonları tanımlandı (Basic, Extended, Full, ANFIS)
- [ ] Input-output konfigürasyonları belirlendi (2In1Out, 3In1Out, vb.)
- [ ] Quality control threshold'ları ayarlandı
- [ ] StandardScaler kullanımı doğrulandı (MinMaxScaler değil!)
- [ ] Excluded nuclei tracker aktif
- [ ] **Dual visualization sistemi hazır (PNG + HTML)** ✅
- [ ] **PNG çözünürlük: 300 DPI** ✅
- [ ] **HTML interaktif özellikler: Hover, zoom, export** ✅
- [ ] Logging sistemi aktif
- [ ] Backup stratejisi hazır

---

**Son Not**: Bu dokümantasyon PFAZ 1'in tam ve güncel bir açıklamasıdır. Herhangi bir soru veya sorun için log dosyalarını kontrol edin ve gerekirse konfigürasyonu özelleştirin.

**Başarılar! 🚀**
