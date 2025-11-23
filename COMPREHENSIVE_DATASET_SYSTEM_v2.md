# Comprehensive Dataset System v2.0.0

**Nükleer Fizik AI Dataset Sistemi - Geliştirilmiş Sürüm**

Tarih: 2025-11-23
Versiyon: 2.0.0
Durum: ✅ Hazır

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Yeni Özellikler](#yeni-özellikler)
3. [Mimari](#mimari)
4. [Kullanım](#kullanım)
5. [Feature Set Oluşturma](#feature-set-oluşturma)
6. [Konfigürasyon](#konfigürasyon)
7. [Örnekler](#örnekler)
8. [Performans](#performans)

---

## 🎯 Genel Bakış

Dataset System v2.0.0, orijinal sistemin geliştirilmiş versiyonudur ve **binlerce farklı feature kombinasyonu** ile dataset oluşturabilme yeteneği sunar.

### Temel İyileştirmeler

**v1.0 (Eski Sistem):**
- ✅ 9 sabit STANDARD_FEATURE_SETS
- ✅ 9 sabit BETA2_FEATURE_SETS
- ✅ Toplam ~36-100 dataset

**v2.0 (Yeni Sistem):**
- ✅ **16,409 farklı feature kombinasyonu**
- ✅ Dinamik feature set oluşturma
- ✅ Target-bazlı optimizasyon
- ✅ Multi-group kombinasyonlar
- ✅ Teorik maksimum: **480,000 dataset**
- ✅ Pratik limit (konfigürasyonla): **10,000 dataset**

---

## 🚀 Yeni Özellikler

### 1. Dinamik Feature Set Builder

`core_modules/feature_set_builder.py` modülü:

```python
from core_modules.feature_set_builder import FeatureSetBuilder

builder = FeatureSetBuilder()

# Temel kombinasyonlar
base_sets = builder.generate_base_combinations()
# Sonuç: AZN, AZNS, AZNP, AZNSP

# Physics grupları ile
physics_sets = builder.generate_full_combinations()
# Sonuç: AZN_beta, AZN_p, AZNS_magic, vb. (116 adet)

# Multi-group kombinasyonlar (BINLERCE!)
multi_sets = builder.generate_multi_group_combinations(max_groups_per_combo=3)
# Sonuç: AZN_beta_p_magic, AZNSP_shell_BE_collective, vb. (16,240 adet)
```

### 2. Feature Grupları

#### Base Features
- `AZN`: A, Z, N (temel)
- `AZNS`: AZN + SPIN
- `AZNP`: AZN + PARITY
- `AZNSP`: AZN + SPIN + PARITY

#### Physics Feature Groups (29 grup)

1. **Deformation:** `beta`, `beta4`, `def_full`
2. **Pairing:** `p`, `pair_full`
3. **Magic Numbers:** `magic`, `magic_full`, `magic_adv`
4. **Binding Energy:** `BE`, `BE_full`, `BE_adv`
5. **SEMF:** `semf_basic`, `semf_full`, `semf_all`
6. **Shell Model:** `shell_basic`, `shell_full`, `shell_all`
7. **Schmidt Model:** `schmidt`, `schmidt_full` (MM için kritik)
8. **Collective Model:** `collective`, `collective_full`
9. **Asymmetry:** `asym`, `asym_full`
10. **Combined Groups:** `beta_p`, `magic_BE`, `shell_beta`, `physics_basic`, `physics_adv`, `physics_ultra`

### 3. Target-Optimized Feature Sets

Her target için optimize edilmiş 5 kompleksite seviyesi:

```python
builder.generate_target_optimized_sets('MM')
# MM_minimal: 3 features
# MM_basic: 6 features (kritik olanlar)
# MM_standard: 9 features (kritik + önemli)
# MM_advanced: 11 features (kritik + önemli + faydalı)
# MM_ultra: 14 features (her şey dahil)
```

### 4. Konfigürasyon Tabanlı Üretim

`dataset_generation_config.yaml` ile tam kontrol:

```yaml
feature_sets:
  mode: "comprehensive"  # standard, extended, comprehensive, targeted
  max_groups_per_combo: 3
  min_features: 3
  max_features: 20

limits:
  max_total_datasets: 10000  # Toplam dataset limiti

presets:
  quick_test: {...}
  comprehensive: {...}
```

---

## 🏗️ Mimari

### Dosya Yapısı

```
nucdatav1/
├── core_modules/
│   ├── feature_set_builder.py         # 🆕 Dinamik feature set oluşturucu
│   └── constants.py                   # 🔄 get_dynamic_feature_sets() eklendi
│
├── dataset_generation_config.yaml     # 🆕 Ana konfigürasyon dosyası
├── generate_comprehensive_datasets.py # 🆕 Utility script
│
└── pfaz_modules/
    └── pfaz01_dataset_generation/
        ├── dataset_generator.py       # Mevcut (feature sets kullanır)
        └── dataset_generation_pipeline_v2.py
```

### Veri Akışı

```
[Config YAML]
    ↓
[Feature Set Builder]
    ↓
[Feature Sets] → [Dataset Generator] → [ANFIS_Datasets/]
    ↓
[Targets, Scenarios, Scaling, etc.]
```

---

## 💻 Kullanım

### Basit Kullanım

#### 1. Feature Set Oluşturma (Python)

```python
from core_modules.constants import get_dynamic_feature_sets

# Legacy mode (eski sistem ile uyumlu)
sets = get_dynamic_feature_sets(mode='standard')
# 9 set (AZN, AZNS, AZNP, ...)

# Extended mode (~200 set)
sets = get_dynamic_feature_sets(mode='extended', max_sets=200)

# Comprehensive mode (binlerce!)
sets = get_dynamic_feature_sets(mode='comprehensive', max_sets=1000)

# Target-optimized
sets = get_dynamic_feature_sets(mode='targeted', target_name='MM')
```

#### 2. Utility Script ile

```bash
# Dry-run (kaç dataset oluşturulacağını göster)
python generate_comprehensive_datasets.py --dry-run

# Preset kullanarak
python generate_comprehensive_datasets.py --preset quick_test
python generate_comprehensive_datasets.py --preset comprehensive

# Custom config
python generate_comprehensive_datasets.py --config my_config.yaml

# Verbose output
python generate_comprehensive_datasets.py --dry-run --verbose
```

---

## 🔬 Feature Set Oluşturma

### Örnek 1: Temel Kombinasyonlar

```python
from core_modules.feature_set_builder import FeatureSetBuilder

builder = FeatureSetBuilder()

# AZN, AZNS, AZNP, AZNSP
base = builder.generate_base_combinations()
```

Sonuç:
```python
{
    'AZN': ['A', 'Z', 'N'],
    'AZNS': ['A', 'Z', 'N', 'SPIN'],
    'AZNP': ['A', 'Z', 'N', 'PARITY'],
    'AZNSP': ['A', 'Z', 'N', 'SPIN', 'PARITY']
}
```

### Örnek 2: Physics Grupları ile

```python
# Single physics group
sets = builder.generate_full_combinations(
    base_variants=['AZN', 'AZNS'],
    physics_groups=['beta', 'p', 'magic']
)
```

Sonuç:
```python
{
    'AZN_beta': ['A', 'Z', 'N', 'beta_2'],
    'AZN_p': ['A', 'Z', 'N', 'p_factor'],
    'AZN_magic': ['A', 'Z', 'N', 'Z_magic_dist', 'N_magic_dist'],
    'AZNS_beta': ['A', 'Z', 'N', 'SPIN', 'beta_2'],
    'AZNS_p': ['A', 'Z', 'N', 'SPIN', 'p_factor'],
    'AZNS_magic': ['A', 'Z', 'N', 'SPIN', 'Z_magic_dist', 'N_magic_dist']
}
```

### Örnek 3: Multi-Group Kombinasyonlar

```python
# 2-3 physics grubunu birleştir
sets = builder.generate_multi_group_combinations(
    max_groups_per_combo=2,
    max_combinations=10
)
```

Sonuç:
```python
{
    'AZN_beta_p': ['A', 'Z', 'N', 'beta_2', 'p_factor'],
    'AZN_beta_magic': ['A', 'Z', 'N', 'beta_2', 'Z_magic_dist', 'N_magic_dist'],
    'AZN_p_magic': ['A', 'Z', 'N', 'p_factor', 'Z_magic_dist', 'N_magic_dist'],
    'AZNS_beta_p': ['A', 'Z', 'N', 'SPIN', 'beta_2', 'p_factor'],
    ...
}
```

### Örnek 4: Target-Optimized

```python
# MM target için optimize edilmiş setler
mm_sets = builder.generate_target_optimized_sets('MM')
```

Sonuç:
```python
{
    'MM_minimal': ['A', 'Z', 'N'],  # 3 features
    'MM_basic': ['A', 'Z', 'N', 'SPIN', 'PARITY', 'schmidt_nearest'],  # 6 features
    'MM_standard': [...],  # 9 features
    'MM_advanced': [...],  # 11 features
    'MM_ultra': [...]  # 14 features
}
```

---

## ⚙️ Konfigürasyon

### dataset_generation_config.yaml

#### Feature Set Modes

```yaml
feature_sets:
  # Mode seçenekleri:
  # - "standard": Legacy (9-18 set)
  # - "extended": Base + Single physics (~200 set)
  # - "comprehensive": Multi-group kombinasyonlar (1000+ set)
  # - "targeted": Target-optimized setler
  mode: "comprehensive"

  # Base varyantlar
  base_variants:
    - "AZN"
    - "AZNS"
    - "AZNP"
    - "AZNSP"

  # Multi-group settings
  enable_multi_group_combinations: true
  max_groups_per_combo: 3  # 1, 2, veya 3 grubu birleştir

  # Filtreler
  min_features: 3
  max_features: 20
```

#### Limits

```yaml
limits:
  # Maksimum toplam dataset sayısı
  max_total_datasets: 10000

  # Redundant kombinasyonları atla
  skip_redundant: true

  # Parallel generation
  enable_parallel: true
  max_workers: 4
```

#### Presets

```yaml
presets:
  quick_test:
    feature_sets.mode: "standard"
    limits.max_total_datasets: 100
    dataset_params.nucleus_counts: [75, 100]

  comprehensive:
    feature_sets.mode: "comprehensive"
    limits.max_total_datasets: 10000
    feature_sets.enable_multi_group_combinations: true
```

Active preset:
```yaml
active_preset: "comprehensive"
```

---

## 📊 Örnekler

### Örnek 1: Hızlı Test (100 Dataset)

```bash
python generate_comprehensive_datasets.py --preset quick_test --dry-run
```

Sonuç:
```
Targets: 4
Nucleus counts: 2
Scenarios: 2
Feature sets: 9
Total: ~144 datasets (limited to 100)
```

### Örnek 2: Standard Research (1000 Dataset)

```bash
python generate_comprehensive_datasets.py --preset standard_research --dry-run
```

Sonuç:
```
Targets: 4
Nucleus counts: 4
Scenarios: 2
Feature sets: ~200
Total: ~12,800 datasets (limited to 1000)
```

### Örnek 3: Comprehensive (10,000 Dataset)

```bash
python generate_comprehensive_datasets.py --preset comprehensive --dry-run
```

Sonuç:
```
Targets: 4
Nucleus counts: 5
Scenarios: 2
Anomaly modes: 2
Feature sets: 1000
Scaling methods: 3
Sampling methods: 2

Theoretical Total: 480,000 datasets
Actual Total (with limits): 10,000 datasets
```

### Örnek 4: Python ile Custom

```python
from core_modules.feature_set_builder import FeatureSetBuilder

builder = FeatureSetBuilder()

# Sadece MM target için, beta ve schmidt içeren setler
mm_sets = builder.generate_multi_group_combinations(
    base_variants=['AZNSP'],  # Sadece AZNSP
    physics_groups=['beta', 'schmidt', 'magic'],
    max_groups_per_combo=2,
    max_combinations=20
)

print(f"Generated {len(mm_sets)} feature sets for MM target")
for name, features in list(mm_sets.items())[:5]:
    print(f"  {name}: {features}")
```

---

## ⚡ Performans

### Kombinasyon Sayıları

| Mode | Feature Sets | Örnek Kullanım |
|------|--------------|----------------|
| `standard` | 9-18 | Legacy sistem, hızlı test |
| `extended` | ~200 | Orta ölçekli araştırma |
| `comprehensive` | 1,000+ | Kapsamlı AI eğitimi |
| `targeted` | 20-100 | Target-specific optimize |

### Toplam Dataset Hesaplaması

```
Total = Targets × Nucleus_Counts × Scenarios × Anomaly × Features × Scaling × Sampling

Örnek (comprehensive mode):
= 4 × 5 × 2 × 2 × 1000 × 3 × 2
= 480,000 dataset (teorik)
```

Config limitiyle:
```
max_total_datasets = 10,000
→ İlk 10,000 en önemli kombinasyon seçilir
```

### Disk Kullanımı Tahmini

Dataset başına ortalama:
- CSV: ~50 KB
- Excel: ~100 KB
- MATLAB: ~150 KB (kapalı by default)
- Metadata: ~10 KB

**10,000 dataset için:**
- CSV only: ~500 MB
- CSV + Excel: ~1.5 GB
- Tümü: ~3 GB

---

## 🎓 Önerilen Kullanım

### Yeni Başlayanlar

1. **quick_test preset** ile başlayın (~100 dataset)
2. Feature kombinasyonlarını inceleyin
3. Target'ınıza göre mode seçin

```bash
python generate_comprehensive_datasets.py --preset quick_test --dry-run
```

### Araştırmacılar

1. **standard_research preset** (~1000 dataset)
2. Target-optimized setleri kullanın
3. Belirli feature gruplarını seçin

```python
sets = get_dynamic_feature_sets(
    mode='targeted',
    target_name='MM',
    include_comprehensive=True,
    max_sets=500
)
```

### İleri Seviye

1. **comprehensive mode** (binlerce dataset)
2. Custom physics group kombinasyonları
3. Multi-group combinations (2-3 grup)

```python
builder = FeatureSetBuilder()
sets = builder.generate_multi_group_combinations(
    max_groups_per_combo=3,
    max_combinations=5000
)
```

---

## 🔍 Feature Set İçerikleri

### Temel Gruplar

| Grup | İçerik | Feature Sayısı |
|------|--------|----------------|
| `AZN` | A, Z, N | 3 |
| `AZNS` | A, Z, N, SPIN | 4 |
| `AZNP` | A, Z, N, PARITY | 4 |
| `AZNSP` | A, Z, N, SPIN, PARITY | 5 |

### Physics Grupları (Örnekler)

| Grup | İçerik | Feature Sayısı | Kullanım |
|------|--------|----------------|----------|
| `beta` | beta_2 | 1 | Deformation (QM için kritik) |
| `p` | p_factor | 1 | Pairing (tüm targets) |
| `magic` | Z_magic_dist, N_magic_dist | 2 | Shell effects |
| `schmidt` | schmidt_nearest | 1 | MM prediction (kritik) |
| `beta_p` | beta_2, p_factor | 2 | Kombine deformation+pairing |
| `physics_basic` | beta_2, p_factor, BE_per_A, magic | 5 | Temel fizik |
| `physics_ultra` | 9 temel fizik feature | 9 | Gelişmiş fizik |

### Target-Optimized İçerikleri

#### MM Target
```python
MM_minimal: ['A', 'Z', 'N']
MM_basic: ['A', 'Z', 'N', 'SPIN', 'PARITY', 'schmidt_nearest']
MM_standard: [..., 'schmidt_deviation', 'magic_character', 'valence_nucleons']
MM_advanced: [..., 'beta_2', 'collective_parameter']
MM_ultra: [..., 'BE_per_A', 'Z_shell_gap', 'N_shell_gap']
```

#### QM Target
```python
QM_minimal: ['A', 'Z', 'N']
QM_basic: ['A', 'Z', 'N', 'beta_2']
QM_standard: [..., 'valence_nucleons', 'Z_magic_dist', 'N_magic_dist']
QM_advanced: [..., 'Beta_4', 'collective_parameter', 'nilsson_epsilon']
QM_ultra: [..., 'BE_per_A', 'asymmetry', 'rotational_constant']
```

---

## 📚 Ek Kaynaklar

- **AI_TRAINING_FEATURE_COMBINATIONS.md**: Detaylı feature açıklamaları
- **DATASET_GUIDE.md**: Orijinal dataset kullanım kılavuzu
- **feature_set_builder.py**: Kaynak kod ve örnekler
- **constants.py**: Feature set tanımları

---

## ✅ Durum ve Sürüm Bilgisi

**Versiyon:** 2.0.0
**Durum:** Hazır ve test edildi
**Tarih:** 2025-11-23

### Değişiklikler (v1.0 → v2.0)

✅ **Eklenen:**
- `feature_set_builder.py` modülü
- Dinamik feature set generation
- Multi-group kombinasyonlar (16,409 olası set)
- Target-optimized feature sets
- YAML konfigürasyon dosyası
- Comprehensive dataset generation utility

✅ **Güncellenen:**
- `constants.py`: `get_dynamic_feature_sets()` fonksiyonu
- Legacy setler korundu (geriye dönük uyumluluk)

✅ **Korunan:**
- Mevcut dataset_generator.py
- Mevcut feature set isimleri
- Tüm önceki fonksiyonalite

---

## 🎯 Sonuç

Dataset System v2.0.0 ile artık:

✅ **Binlerce** farklı feature kombinasyonu
✅ **Esnek** konfigürasyon
✅ **Target-optimized** setler
✅ **Ölçeklenebilir** mimari
✅ **Geriye dönük uyumlu**

**Toplam Kapasit:** 480,000 dataset (teorik), 10,000 dataset (pratik limit)

Bu sistem, nükleer fizik AI araştırmaları için kapsamlı ve esnek bir dataset altyapısı sağlar!

---

**Hazırlayan:** Nuclear Physics AI Project Team
**İletişim:** Proje dokümantasyonuna bakınız
