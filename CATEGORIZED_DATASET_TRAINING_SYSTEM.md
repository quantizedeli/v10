# Categorized Dataset & Training System v2.1.0

**Kısım-Bazlı Dataset Sistemi ve ANFIS/AI Eğitim Entegrasyonu**

Tarih: 2025-11-23
Versiyon: 2.1.0
Durum: ✅ Hazır - MATLAB Engine Aktif

---

## 📋 Genel Bakış

Bu sistem, dataset'leri **giriş sayısına göre kategorize ederek** farklı eğitim stratejileri uygular:

- **KISIM 1 (Minimal)**: 2-4 giriş → Hızlı eğitim, temel modeller
- **KISIM 2 (Advanced)**: 5+ giriş → Gelişmiş eğitim, karmaşık modeller

Her kategori **ayrı pipeline'lardan** geçer ve **ayrı output klasörlerinde** saklanır.

---

## 🎯 Kategoriler

### Kısım 1: Minimal Datasets

**Tanım**: Basit kombinasyonlar, hızlı eğitim için optimize edilmiş

**Giriş Sayısı**: 2-4 features

**Örnekler**:
- **2 giriş**: Beta2_AZ, Beta2_AN, Beta2_ZN → Beta_2
- **3 giriş**: AZN → MM, QM, Beta_2, MM_QM
- **4 giriş**: AZNS, AZNP → MM, QM, Beta_2
- **4-5 giriş** (MM_QM için): AZNS, AZNP, AZNSP → MM_QM

**Çıktı Klasörü**: `ANFIS_Datasets_Minimal/`

**Eğitim Özellikleri**:
- ✅ AI eğitimi: Aktif (RF, GBM, XGBoost, DNN, BNN, PINN)
- ✅ ANFIS eğitimi: Aktif (MATLAB Engine)
- ✅ Hızlı yineleme
- ✅ Yüksek öncelik (ilk işlenir)

### Kısım 2: Advanced Datasets

**Tanım**: Karmaşık kombinasyonlar, gelişmiş eğitim için

**Giriş Sayısı**: 5+ features

**Örnekler**:
- **AZNSP + Physics**: AZNSP_beta, AZNSP_p, AZNSP_magic
- **Target-Optimized**: MM_advanced (11 inputs), MM_ultra (14 inputs)
- **Multi-Group**: AZN_beta_p_magic, AZNSP_shell_BE_collective

**Çıktı Klasörü**: `ANFIS_Datasets_Advanced/`

**Eğitim Özellikleri**:
- ✅ AI eğitimi: Aktif (RF, GBM, XGBoost, DNN, BNN, PINN)
- ✅ ANFIS eğitimi: Aktif (MATLAB Engine)
- ✅ Derinlemesine öğrenme
- ✅ Normal öncelik (minimal'dan sonra)

---

## 🚀 Kullanım

### Python API

#### 1. Minimal Feature Sets (Kısım 1)

```python
from core_modules.constants import get_dynamic_feature_sets

# MM için minimal setler
minimal = get_dynamic_feature_sets(mode='minimal', target_name='MM')
# Sonuç: AZN, AZNS, AZNP (3 set)

# Beta_2 için minimal (2-input dahil)
beta2_min = get_dynamic_feature_sets(mode='minimal', target_name='Beta_2')
# Sonuç: Beta2_AZ, Beta2_AN, Beta2_ZN, AZN, AZNS, AZNP (6 set)

# MM_QM için minimal (AZNSP dahil)
mm_qm_min = get_dynamic_feature_sets(mode='minimal', target_name='MM_QM')
# Sonuç: AZN, AZNS, AZNP, AZNSP (4 set)
```

#### 2. Advanced Feature Sets (Kısım 2)

```python
# QM için advanced setler (max 100)
advanced = get_dynamic_feature_sets(mode='advanced', target_name='QM', max_sets=100)
# Sonuç: QM_advanced, QM_ultra, AZNS_beta, AZNSP_magic, ... (100 set)
```

#### 3. Categorized Mode (Her İkisi)

```python
# Hem minimal hem advanced
categorized = get_dynamic_feature_sets(mode='categorized', target_name='MM', max_sets=50)

# Erişim
minimal_sets = categorized['minimal']  # 3 set
advanced_sets = categorized['advanced']  # 50 set

print(f"Minimal: {len(minimal_sets)} sets")
print(f"Advanced: {len(advanced_sets)} sets")
```

### Command-Line

#### Preset Kullanımı

```bash
# 1. Sadece Minimal (Kısım 1)
python generate_comprehensive_datasets.py --preset minimal_only --dry-run

# 2. Sadece Advanced (Kısım 2)
python generate_comprehensive_datasets.py --preset advanced_only --dry-run

# 3. Categorized (Her ikisi, ayrı pipeline'lar)
python generate_comprehensive_datasets.py --preset categorized_training --dry-run
```

#### Config Düzenleme

`dataset_generation_config.yaml`:
```yaml
feature_sets:
  mode: "categorized"  # minimal, advanced, veya categorized

categorized_mode:
  minimal:
    output_dir: "ANFIS_Datasets_Minimal"
    enable_ai_training: true
    enable_anfis_training: true
    priority: "high"

  advanced:
    output_dir: "ANFIS_Datasets_Advanced"
    enable_ai_training: true
    enable_anfis_training: true
    priority: "normal"
    max_sets_per_target: 100

output:
  formats:
    matlab: true  # MATLAB Engine aktif!
```

---

## 📊 Feature Set Detayları

### Minimal Sets (Her Target için)

| Target | 2-Input | 3-Input | 4-Input | Total |
|--------|---------|---------|---------|-------|
| **MM** | - | AZN | AZNS, AZNP | **3** |
| **QM** | - | AZN | AZNS, AZNP | **3** |
| **Beta_2** | AZ, AN, ZN | AZN | AZNS, AZNP | **6** |
| **MM_QM** | - | AZN | AZNS, AZNP, AZNSP* | **4** |

*AZNSP (5 input) MM_QM için exception - çift output nedeniyle minimal'e dahil

### Advanced Sets (Örnek MM için)

| Set Name | Features | Input Count | Description |
|----------|----------|-------------|-------------|
| `MM_advanced` | A,Z,N,SPIN,PARITY,schmidt_nearest,schmidt_deviation,magic_character,valence_nucleons,beta_2,collective_parameter | 11 | Target-optimized advanced |
| `MM_ultra` | + BE_per_A, Z_shell_gap, N_shell_gap | 14 | Target-optimized ultra |
| `AZNSP_beta` | A,Z,N,SPIN,PARITY,beta_2 | 6 | Base + single physics |
| `AZNSP_beta_p` | A,Z,N,SPIN,PARITY,beta_2,p_factor | 7 | Base + multi physics |
| `AZN_beta_p_magic` | A,Z,N,beta_2,p_factor,Z_magic_dist,N_magic_dist | 7 | Multi-group combination |

---

## 🔧 MATLAB Engine & ANFIS Training

### Konfigürasyon

`config.json`:
```json
{
  "pfaz03_anfis_training": {
    "enabled": true,
    "matlab_engine": {
      "enabled": true,
      "fallback_python": true
    },
    "configurations": [
      "gridpartition_trimf",
      "gridpartition_gaussmf",
      "gridpartition_gbellmf",
      "gridpartition_trapmf",
      "subclust_trimf",
      "subclust_gaussmf",
      "fcm_trimf",
      "fcm_gaussmf"
    ]
  }
}
```

### ANFIS Eğitimi

#### Minimal Datasets için:
- **Grid Partition** öncelikli (2-4 giriş için ideal)
- **Hızlı eğitim** (epochs: 100)
- **Tüm MF tipleri** (trimf, gaussmf, gbellmf, trapmf)

#### Advanced Datasets için:
- **Subtractive Clustering** öncelikli (5+ giriş için ideal)
- **FCM clustering** alternatif
- **Daha fazla epoch** (epochs: 150-200)

### AI Model Eğitimi

Her iki kategori için:
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ XGBoost
- ✅ Deep Neural Network
- ✅ Bayesian Neural Network
- ✅ Physics-Informed Neural Network

---

## 📁 Output Yapısı

### Kategorize Edilmiş Output

```
Project Root/
├── ANFIS_Datasets_Minimal/          # Kısım 1
│   ├── MM/
│   │   ├── S70/
│   │   │   ├── MM_75_S70_anomalisiz_AZN_standard_stratified/
│   │   │   ├── MM_75_S70_anomalisiz_AZNS_standard_stratified/
│   │   │   └── MM_75_S70_anomalisiz_AZNP_standard_stratified/
│   │   └── S80/
│   ├── QM/
│   ├── Beta_2/
│   ├── MM_QM/
│   ├── Master_Nuclei_Catalog.xlsx
│   └── Dataset_Catalog_Minimal.xlsx
│
├── ANFIS_Datasets_Advanced/         # Kısım 2
│   ├── MM/
│   │   ├── S70/
│   │   │   ├── MM_100_S70_anomalisiz_MM_advanced_standard_stratified/
│   │   │   ├── MM_100_S70_anomalisiz_AZNSP_beta_standard_stratified/
│   │   │   └── ...
│   │   └── S80/
│   ├── QM/
│   ├── Beta_2/
│   ├── MM_QM/
│   ├── Master_Nuclei_Catalog.xlsx
│   └── Dataset_Catalog_Advanced.xlsx
│
└── trained_models/
    ├── minimal/                      # Minimal model'leri
    │   ├── anfis/
    │   │   ├── MM_AZN_GAU2MF.mat
    │   │   └── ...
    │   └── ai/
    │       ├── MM_AZN_RF.pkl
    │       └── ...
    └── advanced/                     # Advanced model'leri
        ├── anfis/
        └── ai/
```

---

## 🎓 Eğitim Pipeline'ı

### Minimal Pipeline (Kısım 1)

```
1. Dataset Generation
   ├─ Generate minimal feature sets (2-4 inputs)
   ├─ Apply QM filtering if needed
   ├─ Stratified sampling
   └─ Save to ANFIS_Datasets_Minimal/

2. AI Training (Parallel)
   ├─ Random Forest
   ├─ Gradient Boosting
   ├─ XGBoost
   ├─ DNN (smaller architecture)
   ├─ BNN
   └─ PINN

3. ANFIS Training (MATLAB Engine)
   ├─ Grid Partition methods
   │   ├─ GAU2MF (2 MF, Gaussian)
   │   ├─ GEN2MF (2 MF, Generalized Bell)
   │   ├─ TRI2MF (2 MF, Triangular)
   │   ├─ TRA2MF (2 MF, Trapezoidal)
   │   └─ GAU3MF (3 MF, Gaussian)
   └─ Subtractive Clustering
       ├─ SUBR03 (radii=0.3)
       ├─ SUBR05 (radii=0.5)
       └─ SUBR07 (radii=0.7)

4. Model Evaluation
   ├─ Performance metrics (R², RMSE, MAE)
   ├─ Model comparison
   └─ Best model selection

5. Decision: Continue or Stop?
   └─ If performance acceptable → Save models
   └─ If not → Try different feature set
```

### Advanced Pipeline (Kısım 2)

```
1. Dataset Generation
   ├─ Generate advanced feature sets (5+ inputs)
   ├─ Apply QM filtering if needed
   ├─ Stratified sampling
   └─ Save to ANFIS_Datasets_Advanced/

2. AI Training (Parallel)
   ├─ Random Forest (more estimators)
   ├─ Gradient Boosting
   ├─ XGBoost (deeper trees)
   ├─ DNN (larger architecture)
   ├─ BNN
   └─ PINN (more constraints)

3. ANFIS Training (MATLAB Engine)
   ├─ Subtractive Clustering (preferred for 5+ inputs)
   ├─ FCM Clustering
   └─ Grid Partition (if input count ≤ 6)

4. Model Evaluation
   ├─ Performance metrics
   ├─ Feature importance analysis
   ├─ Model complexity vs. accuracy trade-off
   └─ Best model selection

5. Decision: Continue or Stop?
   └─ Advanced models may need more epochs
   └─ Compare with minimal models
```

---

## 💡 Kullanım Önerileri

### Yeni Başlayanlar

1. **minimal_only preset** ile başlayın
   ```bash
   python generate_comprehensive_datasets.py --preset minimal_only --dry-run
   ```

2. Minimal dataset'leri oluşturun
3. AI ve ANFIS eğitimlerini çalıştırın
4. Sonuçları değerlendirin
5. İyi performans alırsanız → Minimal yeterli
6. Yetersizse → Advanced'e geçin

### Araştırmacılar

1. **categorized_training preset** kullanın
2. Her iki kategoriyi de paralel oluşturun
3. Minimal ile hızlı prototip yapın
4. Advanced ile derinlemesine analiz edin
5. En iyi modelleri karşılaştırın

### İleri Seviye

1. Custom feature set kombinasyonları oluşturun
2. Target-specific advanced setler kullanın
3. Multi-group combinations deneyin
4. Ensemble modeller oluşturun (minimal + advanced)

---

## 📊 Örnek Senaryolar

### Senaryo 1: MM Prediction (Hızlı)

```python
# 1. Minimal datasets oluştur
minimal = get_dynamic_feature_sets(mode='minimal', target_name='MM')
# AZN, AZNS, AZNP (3 set)

# 2. Eğit ve değerlendir
# - AZN: R² = 0.85 (yetersiz)
# - AZNS: R² = 0.92 (iyi)
# - AZNP: R² = 0.89 (iyi)

# 3. AZNS yeterli → Minimal'de kal, Advanced'e gerek yok
```

### Senaryo 2: QM Prediction (Gelişmiş)

```python
# 1. Minimal datasets oluştur
minimal = get_dynamic_feature_sets(mode='minimal', target_name='QM')
# AZN, AZNS, AZNP (3 set)

# 2. Eğit ve değerlendir
# - AZN: R² = 0.78 (yetersiz)
# - AZNS: R² = 0.81 (yetersiz)
# - AZNP: R² = 0.79 (yetersiz)

# 3. Advanced'e geç
advanced = get_dynamic_feature_sets(mode='advanced', target_name='QM', max_sets=50)

# 4. QM_advanced (beta_2, valence, magic) → R² = 0.94 ✓
```

### Senaryo 3: Kapsamlı Araştırma

```python
# Tüm target'lar için categorized
for target in ['MM', 'QM', 'Beta_2', 'MM_QM']:
    cat = get_dynamic_feature_sets(mode='categorized', target_name=target, max_sets=100)

    # Minimal eğit
    train_minimal(cat['minimal'])

    # Advanced eğit
    train_advanced(cat['advanced'])

    # Karşılaştır
    compare_and_select_best()
```

---

## ✅ Sistem Durumu

**Versiyon**: 2.1.0
**Durum**: Hazır ve Test Edildi
**MATLAB Engine**: ✅ Aktif
**AI Training**: ✅ Aktif
**ANFIS Training**: ✅ Aktif

### Özellikler

✅ **Kategorize Edilmiş Datasets**
- Minimal (2-4 giriş): ✓
- Advanced (5+ giriş): ✓
- Categorized mode: ✓

✅ **Target Support**
- MM: ✓
- QM: ✓
- Beta_2: ✓ (2-input support)
- MM_QM: ✓ (multi-output support)

✅ **Eğitim Sistemleri**
- AI Models (6 adet): ✓
- ANFIS (MATLAB Engine): ✓
- Parallel training: ✓ (4 workers)

✅ **Output Management**
- Ayrı klasörler: ✓
- Ayrı pipeline'lar: ✓
- MATLAB .mat files: ✓

---

## 🔗 Referanslar

- **Feature Set Builder**: `core_modules/feature_set_builder.py`
- **Constants**: `core_modules/constants.py` (`get_dynamic_feature_sets`)
- **Config**: `config.json` (MATLAB engine settings)
- **Dataset Config**: `dataset_generation_config.yaml`
- **Presets**: minimal_only, advanced_only, categorized_training

---

**Hazırlayan**: Nuclear Physics AI Project Team
**Tarih**: 2025-11-23
**Versiyon**: 2.1.0
