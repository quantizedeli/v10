# Dataset System v2.0.0 - Quick Start Guide

**Hızlı Başlangıç Kılavuzu**

---

## 🚀 5 Dakikada Başlayın

### 1. Feature Set Kombinasyonlarını Keşfet

```python
from core_modules.feature_set_builder import FeatureSetBuilder

builder = FeatureSetBuilder()

# Kaç kombinasyon mümkün?
stats = builder.count_possible_combinations(max_groups_per_combo=3)
print(f"Toplam olası kombinasyon: {stats['total_possible']:,}")
# Sonuç: 16,409 kombinasyon!
```

### 2. Basit Feature Set Oluştur

```python
from core_modules.constants import get_dynamic_feature_sets

# Legacy mode (eski sistem - 9 set)
sets = get_dynamic_feature_sets(mode='standard')

# Extended mode (~200 set)
sets = get_dynamic_feature_sets(mode='extended', max_sets=200)

# Comprehensive mode (1000+ set)
sets = get_dynamic_feature_sets(mode='comprehensive', max_sets=1000)

print(f"Oluşturulan feature set sayısı: {len(sets)}")
for name in list(sets.keys())[:5]:
    print(f"  - {name}")
```

### 3. Target-Specific Feature Sets

```python
# MM target için optimize edilmiş setler
mm_sets = get_dynamic_feature_sets(mode='targeted', target_name='MM')

for name, features in mm_sets.items():
    print(f"{name}: {len(features)} features")
    print(f"  {features}\n")
```

### 4. Utility Script ile Tahmin

```bash
# Kaç dataset oluşturulacağını gör (dry-run)
python generate_comprehensive_datasets.py --dry-run

# Sonuç:
# Theoretical Total: 480,000 datasets
# Actual Total (with limits): 10,000 datasets
```

### 5. Preset Kullan

```bash
# Quick test (100 dataset)
python generate_comprehensive_datasets.py --preset quick_test --dry-run

# Standard research (1000 dataset)
python generate_comprehensive_datasets.py --preset standard_research --dry-run

# Comprehensive (10,000 dataset)
python generate_comprehensive_datasets.py --preset comprehensive --dry-run
```

---

## 📝 Temel Örnekler

### Örnek 1: AZN, AZNS, AZNP, AZNSP

```python
from core_modules.feature_set_builder import FeatureSetBuilder

builder = FeatureSetBuilder()
base_sets = builder.generate_base_combinations()

for name, features in base_sets.items():
    print(f"{name}: {features}")
```

**Sonuç:**
```
AZN: ['A', 'Z', 'N']
AZNS: ['A', 'Z', 'N', 'SPIN']
AZNP: ['A', 'Z', 'N', 'PARITY']
AZNSP: ['A', 'Z', 'N', 'SPIN', 'PARITY']
```

### Örnek 2: Physics Grupları ile Kombinasyon

```python
# AZN + physics grupları
sets = builder.generate_full_combinations(
    base_variants=['AZN'],
    physics_groups=['beta', 'p', 'magic'],
    max_combinations=10
)

for name, features in sets.items():
    print(f"{name}: {features}")
```

**Sonuç:**
```
AZN_beta: ['A', 'Z', 'N', 'beta_2']
AZN_p: ['A', 'Z', 'N', 'p_factor']
AZN_magic: ['A', 'Z', 'N', 'Z_magic_dist', 'N_magic_dist']
```

### Örnek 3: Multi-Group Kombinasyonlar

```python
# 2-3 grubu birleştir (binlerce kombinasyon!)
multi_sets = builder.generate_multi_group_combinations(
    base_variants=['AZN', 'AZNSP'],
    physics_groups=['beta', 'p', 'magic', 'BE'],
    max_groups_per_combo=2,
    max_combinations=10
)

for name, features in multi_sets.items():
    print(f"{name}: {features}")
```

**Sonuç:**
```
AZN_beta: ['A', 'Z', 'N', 'beta_2']
AZN_p: ['A', 'Z', 'N', 'p_factor']
AZN_beta_p: ['A', 'Z', 'N', 'beta_2', 'p_factor']
AZN_beta_magic: ['A', 'Z', 'N', 'beta_2', 'Z_magic_dist', 'N_magic_dist']
...
```

---

## ⚙️ Konfigürasyon

`dataset_generation_config.yaml` dosyasını düzenleyin:

```yaml
# Mode seç
feature_sets:
  mode: "comprehensive"  # standard, extended, comprehensive, targeted

# Limit belirle
limits:
  max_total_datasets: 10000

# Preset kullan
active_preset: "comprehensive"
```

---

## 🎯 Kullanım Senaryoları

### Senaryo 1: Hızlı Test (Yeni kullanıcılar)

```bash
# 1. Config'de preset değiştir
# active_preset: "quick_test"

# 2. Dry-run ile kontrol et
python generate_comprehensive_datasets.py --dry-run

# Sonuç: ~100 dataset
```

### Senaryo 2: MM Target için Optimize (Araştırmacılar)

```python
# Python ile
sets = get_dynamic_feature_sets(
    mode='targeted',
    target_name='MM',
    include_comprehensive=True,
    max_sets=500
)

# 5 optimized + ~495 comprehensive = 500 MM-specific sets
```

### Senaryo 3: Binlerce Dataset (İleri seviye)

```python
# Feature set builder ile
builder = FeatureSetBuilder()

# Tüm base varyantlar × Tüm physics kombinasyonları
all_sets = builder.generate_multi_group_combinations(
    max_groups_per_combo=3,  # 1, 2, 3 grubu kombine et
    max_combinations=5000    # İlk 5000 kombinasyon
)

print(f"Oluşturulan: {len(all_sets)} feature set")
```

**Bu 5000 feature set ile:**
- 4 targets × 5000 features = 20,000 base datasets
- Scenarios, scaling, sampling ile × 12 = **240,000 dataset**
- Config limiti ile → **10,000 dataset** (en önemliler)

---

## 📊 Kombinasyon Sayıları

| Kombinasyon Tipi | Sayı |
|------------------|------|
| Base (AZN, AZNS, ...) | 4 |
| Physics grupları | 29 |
| Base × Physics (1 grup) | 116 |
| Base × Physics (2-3 grup) | 16,240 |
| **TOPLAM** | **16,409** |

---

## 💡 İpuçları

### 1. Önce Dry-Run Yapın

```bash
python generate_comprehensive_datasets.py --dry-run
```

Her zaman önce kaç dataset oluşturulacağını kontrol edin!

### 2. Preset ile Başlayın

İlk defa kullanıyorsanız:
1. `quick_test` ile başlayın (~100 dataset)
2. Sonuçları inceleyin
3. `standard_research` ile devam edin (~1000 dataset)
4. Sonra `comprehensive` (10,000 dataset)

### 3. Max Sets Limitini Ayarlayın

```python
# Çok fazla kombinasyon istemiyorsanız
sets = get_dynamic_feature_sets(
    mode='comprehensive',
    max_sets=500  # Sadece ilk 500
)
```

### 4. Target'a Göre Seçin

MM için:
- `schmidt` grubu KRİTİK
- `beta` grubu faydalı

QM için:
- `beta` grubu KRİTİK
- `magic` grubu önemli

### 5. Disk Alanını Hesaplayın

```
10,000 dataset × 150 KB (CSV+Excel ortalama) = ~1.5 GB
```

---

## ❓ Sık Sorulan Sorular

**S: Eski feature setleri hala çalışıyor mu?**

A: Evet! `mode='standard'` ile eski sistem aynen çalışır. Geriye dönük tam uyumlu.

**S: Binlerce dataset oluşturmak ne kadar sürer?**

A: Depends on:
- Dataset boyutu (75-267 nuclei)
- Feature sayısı (3-20)
- Sistemin işlemci gücü
- Teorik hesaplamalar (SEMF, Shell, vb.)

Tahmin: ~1-10 saniye per dataset → 10,000 dataset = 3-28 saat

**S: En iyi mode hangisi?**

A:
- Yeni başlayan: `standard` veya `extended`
- Araştırmacı: `targeted` veya `extended`
- İleri seviye: `comprehensive`

**S: Multi-group kombinasyonlar ne işe yarar?**

A: Farklı fizik etkilerini birleştirerek daha zengin feature sets oluşturur.

Örnek:
- `AZN_beta_p_magic`: Deformation + Pairing + Magic numbers
- Her üç fizik etkisini birlikte öğrenir

---

## 📚 Daha Fazla Bilgi

- **COMPREHENSIVE_DATASET_SYSTEM_v2.md**: Detaylı dokümantasyon
- **AI_TRAINING_FEATURE_COMBINATIONS.md**: Feature açıklamaları
- **feature_set_builder.py**: Kaynak kod örnekleri

---

## ✅ Checklist

İlk kez kullanıyorsanız:

- [ ] `feature_set_builder.py` modülünü test edin
- [ ] `get_dynamic_feature_sets()` ile feature set oluşturun
- [ ] `--dry-run` ile tahmin alın
- [ ] `quick_test` preset ile başlayın
- [ ] Sonuçları inceleyin
- [ ] İhtiyacınıza göre mode ve preset seçin
- [ ] Comprehensive generation yapın

---

**Hazır!** Artık binlerce farklı feature kombinasyonu ile dataset oluşturabilirsiniz! 🚀
