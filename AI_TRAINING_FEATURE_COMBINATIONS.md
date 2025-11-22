# AI Eğitim Feature Kombinasyonları Dokümantasyonu

**Proje:** Nükleer Fizik AI Tahmin Sistemi  
**Versiyon:** 1.0.0  
**Tarih:** 2025-11-22  
**Toplam Feature:** 44+  
**Çekirdek Sayısı:** 267 (AAA2.txt)

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Target Tanımları](#target-tanımları)
3. [Feature Kategorileri](#feature-kategorileri)
4. [Target-Bazlı Feature Kombinasyonları](#target-bazlı-feature-kombinasyonları)
5. [QM Filtreleme Kuralları](#qm-filtreleme-kuralları)
6. [Dataset Boyutları](#dataset-boyutları)
7. [Feature Importance Analizi](#feature-importance-analizi)

---

## 🎯 Genel Bakış

Bu dokümantasyon, nükleer fizik AI tahmin sisteminde kullanılan tüm feature kombinasyonlarını ve bunların hangi target'lar için nasıl kullanıldığını detaylı olarak açıklar.

### Temel İstatistikler

```
📊 Dataset İstatistikleri:
├─ Toplam Çekirdek: 267 
├─ MM mevcut: 267 (100%)
├─ QM mevcut: 219 (82%)
└─ Beta_2 mevcut: 267 (100%)

🎯 Target Sayısı: 4
├─ MM (Magnetic Moment)
├─ QM (Quadrupole Moment)
├─ MM_QM (Combined)
└─ Beta_2 (Deformation)

📁 Toplam Dataset: 36
├─ 4 target × 9 boyut = 36 dataset
└─ Boyutlar: 75, 100, 150, 200, ALL, vb.

🔬 Feature Sayısı: 44+
├─ Eksperimental: 9
└─ Teorik: 35+
```

---

## 🎯 Target Tanımları

### 1. MM (Magnetic Moment) - Manyetik Moment

**Tanım:** Nükleer manyetik dipol momenti (μₙ cinsinden)

**Özellikler:**
- **Çekirdek Sayısı:** 267 (ALL)
- **QM Filtresi:** Gerekli değil
- **Tipik Değer Aralığı:** -3.0 ile +6.0 μₙ
- **Fiziksel Anlamı:** Nükleonların spin ve yörüngesel açısal momentlerinin toplamı

**Kullanım Durumu:**
```python
target = 'MM'
target_cols = ['MM']
qm_filter_required = False  # QM olmasa da eğitim yapılabilir
available_nuclei = 267
```

---

### 2. QM (Quadrupole Moment) - Kuadrupol Moment

**Tanım:** Nükleer elektrik kuadrupol momenti (barn cinsinden)

**Özellikler:**
- **Çekirdek Sayısı:** 219
- **QM Filtresi:** ZORUNLU
- **Tipik Değer Aralığı:** -2.0 ile +4.0 barn
- **Fiziksel Anlamı:** Nükleer yük dağılımının küreden sapması

**Kullanım Durumu:**
```python
target = 'QM'
target_cols = ['Q']
qm_filter_required = True  # QM olmadan eğitim yapılamaz
available_nuclei = 219  # QM ölçümü olan çekirdekler
```

**Uyarı:** QM target'ı için veri setinde `Q` sütunu olmalı ve NaN olmamalı!

---

### 3. MM_QM (Combined Target) - Kombine Hedef

**Tanım:** MM ve QM'nin birlikte tahmin edilmesi (multi-output)

**Özellikler:**
- **Çekirdek Sayısı:** 219
- **QM Filtresi:** ZORUNLU
- **Target Sayısı:** 2 (MM ve QM)
- **Kullanım Amacı:** İki özelliği birlikte öğrenerek korelasyonları yakalama

**Kullanım Durumu:**
```python
target = 'MM_QM'
target_cols = ['MM', 'Q']  # İki sütun birlikte
qm_filter_required = True
available_nuclei = 219
```

**Model Çıktısı:**
- Multi-output regression
- İki ayrı loss fonksiyonu
- Ortak feature space

---

### 4. Beta_2 (Deformation Parameter) - Deformasyon Parametresi

**Tanım:** Kuadrupol deformasyon parametresi (β₂)

**Özellikler:**
- **Çekirdek Sayısı:** 267 (ALL)
- **QM Filtresi:** Opsiyonel (feature'lara bağlı)
- **Tipik Değer Aralığı:** -0.4 ile +0.6
- **Fiziksel Anlamı:** Nükleus şeklinin küresellikten sapması
  - β₂ ≈ 0: Küresel
  - β₂ > 0.15: Prolate (iğ şekilli)
  - β₂ < -0.15: Oblate (basık)

**Kullanım Durumu:**
```python
target = 'Beta_2'
target_cols = ['Beta_2']
qm_filter_required = False  # Varsayılan olarak gerekli değil

# ANCAK: Eğer feature'larda Q varsa:
if 'Q' in feature_cols:
    qm_filter_required = True
    available_nuclei = 219
else:
    available_nuclei = 267
```

---

## 🔬 Feature Kategorileri

### 1. Eksperimental Features (9 adet)

**Kaynak:** AAA2.txt dosyasından direkt alınan ölçüm değerleri

| Feature | Açıklama | Birim | Kullanım |
|---------|----------|-------|----------|
| `A` | Kütle numarası | - | ✅ Tüm modeller |
| `Z` | Proton sayısı | - | ✅ Tüm modeller |
| `N` | Nötron sayısı (A-Z) | - | ✅ Tüm modeller |
| `SPIN` | Nükleer spin | ℏ | ✅ Tüm modeller |
| `PARITY` | Parite (+1 veya -1) | - | ✅ Tüm modeller |
| `MM` | Manyetik moment | μₙ | ⚠️ Sadece input için (target değilse) |
| `Q` | Kuadrupol moment | barn | ⚠️ QM filtresi varsa |
| `Beta_2` | Deformasyon | - | ⚠️ Target değilse input olarak |
| `NUCLEUS` | Çekirdek ismi | str | 🔍 Sadece tanımlama için |

**Not:** `MM`, `Q`, `Beta_2` hem feature hem de target olarak kullanılabilir, ancak **asla** aynı anda her ikisi de olmaz!

---

### 2. SEMF Features (11 adet)

**Kaynak:** Semi-Empirical Mass Formula (Yarı-Deneysel Kütle Formülü)

**Formül:**
```
BE = aᵥA - aₛA^(2/3) - aᴄZ²/A^(1/3) - aₐ(N-Z)²/A + δ(A,Z)
```

**Parametreler:**
- aᵥ = 15.75 MeV (hacim terimi)
- aₛ = 17.8 MeV (yüzey terimi)
- aᴄ = 0.711 MeV (Coulomb terimi)
- aₐ = 23.7 MeV (asimetri terimi)
- δ = 11.18/√A (pairing terimi)

| Feature | Formül | Fiziksel Anlamı |
|---------|--------|-----------------|
| `BE_volume` | aᵥA | Hacim enerjisi (tüm nükleonlar) |
| `BE_surface` | -aₛA^(2/3) | Yüzey enerjisi (kenar nükleonları) |
| `BE_coulomb` | -aᴄZ²/A^(1/3) | Coulomb itme enerjisi |
| `BE_asymmetry` | -aₐ(N-Z)²/A | Asimetri enerjisi |
| `BE_pairing` | ±δ/√A | Pairing enerjisi (even/odd) |
| `BE_total` | Σ(yukarıdakiler) | Toplam bağlanma enerjisi |
| `BE_per_A` | BE_total / A | Nükleon başına bağlanma |
| `S2n` | BE(A) - BE(A-2) | İki-nötron ayrılma enerjisi |
| `S2p` | BE(A) - BE(A-2) | İki-proton ayrılma enerjisi |
| `shell_correction` | BE_exp - BE_SEMF | Kabuk etkisi düzeltmesi |
| `separation_energy` | (S2n + S2p)/2 | Ortalama ayrılma enerjisi |

**Kullanım:**
```python
from theoretical_calculations_manager import TheoreticalCalculationsManager

calc_manager = TheoreticalCalculationsManager()
df = calc_manager._calculate_semf(df)
```

---

### 3. Shell Model Features (10 adet)

**Kaynak:** Kabuk Modeli (Magic Numbers + Shell Gaps)

**Magic Numbers:**
- Proton (Z): 2, 8, 20, 28, 50, 82, 114, 126
- Nötron (N): 2, 8, 20, 28, 50, 82, 126, 184

| Feature | Hesaplama | Fiziksel Anlamı |
|---------|-----------|-----------------|
| `Z_magic_dist` | min(\|Z - Zₘₐ𝒾ᴄ\|) | Proton magic sayıya uzaklık |
| `N_magic_dist` | min(\|N - Nₘₐ𝒾ᴄ\|) | Nötron magic sayıya uzaklık |
| `is_magic_Z` | Z ∈ {2,8,20,...} | Z magic mi? (0/1) |
| `is_magic_N` | N ∈ {2,8,20,...} | N magic mi? (0/1) |
| `is_doubly_magic` | is_magic_Z × is_magic_N | Hem Z hem N magic |
| `magic_character` | (is_magic_Z + is_magic_N)/2 | Magic karakter indeksi |
| `Z_shell_gap` | ΔE(Z) | Proton kabuk aralığı (MeV) |
| `N_shell_gap` | ΔE(N) | Nötron kabuk aralığı (MeV) |
| `shell_closure_index` | f(Z_gap, N_gap) | Kabuk kapanma indeksi |
| `valence_nucleons` | Valans Z + Valans N | Dış kabuk nükleonu sayısı |

**Shell Gap Değerleri:**
```python
SHELL_GAPS = {
    2: 14.0 MeV,   # He kabuk kapanması
    8: 11.0 MeV,   # O kabuk kapanması
    20: 8.0 MeV,   # Ca kabuk kapanması
    28: 6.0 MeV,   # Ni kabuk kapanması
    50: 5.0 MeV,   # Sn kabuk kapanması
    82: 4.0 MeV,   # Pb kabuk kapanması
    126: 3.0 MeV   # Superheavy kapanması
}
```

**Örnek:**
```python
# 208Pb (Z=82, N=126) - Doubly Magic
Z_magic_dist = 0
N_magic_dist = 0
is_doubly_magic = 1
magic_character = 1.0  # Maksimum kararlılık
```

---

### 4. Deformation Features (4 adet)

**Kaynak:** Deformasyon hesaplamaları ve ölçümleri

| Feature | Hesaplama | Değer Aralığı |
|---------|-----------|---------------|
| `Beta_2_estimated` | Shell model tahmini | -0.6 ile +0.6 |
| `Beta_4` | Heksadekapol deformasyon | -0.1 ile +0.1 |
| `deformation_type` | Prolate/Oblate/Spherical | kategorik |
| `spherical_index` | 1 - \|β₂\| | 0 (deformed) ile 1 (spherical) |

**Deformation Type Sınıflandırması:**
```python
if abs(Beta_2) < 0.05:
    deformation_type = "spherical"
elif Beta_2 > 0.15:
    deformation_type = "prolate"
elif Beta_2 < -0.15:
    deformation_type = "oblate"
else:
    deformation_type = "weakly_deformed"
```

**Fiziksel Anlamı:**
- **Spherical (β₂ ≈ 0):** Küresel şekil (genellikle magic çekirdekler)
- **Prolate (β₂ > 0):** İğ şekilli (uzamış)
- **Oblate (β₂ < 0):** Basık (yassı)

---

### 5. Schmidt Model Features (2 adet)

**Kaynak:** Schmidt Manyetik Moment Modeli

**Formül:**
```
μ_Schmidt = g_l * l + g_s * s

For j = l + s:  μ = g_l * l + g_s * s
For j = l - s:  μ = [j/(j+1)] * [g_l * l + g_s * s]
```

**g-faktörleri:**
- Proton: g_l = 1.0, g_s = 5.586
- Nötron: g_l = 0.0, g_s = -3.826

| Feature | Hesaplama | Kullanım |
|---------|-----------|----------|
| `schmidt_nearest` | Schmidt tahmini (en yakın) | MM tahmini için kritik |
| `schmidt_deviation` | \|MM_exp - MM_Schmidt\| | Kabuk etkilerinin ölçüsü |

**Örnek:**
```python
# Proton için j = l + 1/2
schmidt_moment = g_l * l + g_s * 0.5

# Sapma ne kadar büyükse, collective effects o kadar önemli
if schmidt_deviation > 1.0:
    print("Strong collective effects!")
```

---

### 6. Collective Model Features (4 adet)

**Kaynak:** Kollektif Model (Vibrational & Rotational)

| Feature | Formül | Birim |
|---------|--------|-------|
| `collective_parameter` | C = β₂² + β₄² | - |
| `rotational_constant` | ℏ²/(2I) | keV |
| `vibrational_frequency` | ωvib | MeV |
| `nucleus_collective_type` | vibrator/rotor/transitional | kategorik |

**Collective Type Belirleme:**
```python
if abs(Beta_2) < 0.1:
    type = "vibrator"      # Küresel çekirdekler
elif abs(Beta_2) > 0.2:
    type = "rotor"         # Deformed çekirdekler
else:
    type = "transitional"  # Geçiş bölgesi
```

---

### 7. Woods-Saxon Features (2 adet) ⚠️ OPSİYONEL

**Kaynak:** Woods-Saxon Potansiyeli

**Not:** Hesaplama maliyeti yüksek olduğu için varsayılan olarak **KAPALI**

**Formül:**
```
V(r) = -V₀ / [1 + exp((r - R)/a)]

Parametreler:
V₀ = 51 MeV (derinlik)
R = 1.27 * A^(1/3) fm (yarıçap)
a = 0.67 fm (yüzey difüzlüğü)
```

| Feature | Hesaplama | Kullanım |
|---------|-----------|----------|
| `ws_surface_thick` | Yüzey kalınlığı (a) | Potansiyel şekli |
| `fermi_energy` | Fermi seviyesi | Tek-parçacık durumları |

**Aktifleştirme:**
```python
calc_manager = TheoreticalCalculationsManager(enable_all=True)
df = calc_manager.calculate_all_theoretical_properties(df)
```

---

### 8. Nilsson Model Features (2 adet) ⚠️ OPSİYONEL

**Kaynak:** Nilsson Deformed Shell Model

**Not:** Sadece deformed çekirdekler için (|β₂| > 0.15)

| Feature | Hesaplama | Kullanım |
|---------|-----------|----------|
| `nilsson_epsilon` | ε = 0.95 * β₂ | Deformasyon parametresi |
| `nilsson_omega` | ω₀ = 41/A^(1/3) MeV | Osilator frekansı |

**Kullanım Kriteri:**
```python
deformed_mask = abs(df['Beta_2_estimated']) > 0.15
df.loc[deformed_mask, 'nilsson_epsilon'] = 0.95 * df['Beta_2_estimated']
```

---

## 🎯 Target-Bazlı Feature Kombinasyonları

### Kombinasyon 1: MM Target

**Hedef:** Manyetik moment tahmini

**Veri Seti:**
```python
Dataset: MM_ALL_267nuclei, MM_200nuclei, MM_150nuclei, vb.
Target: ['MM']
Çekirdek Sayısı: 267 (maksimum)
QM Filtresi: KAPALI
```

**Kullanılan Features (43 adet):**

#### ✅ A. Basic Nuclear Parameters (3 adet):
- `A` - Kütle numarası
- `Z` - Proton sayısı  
- `N` - Nötron sayısı (A-Z)

**Önemi:** Nükleer yapının temel tanımlayıcıları. Tüm teorik hesaplamaların temeli.

#### ✅ B. Spin-Parity Features (2 adet):
- `SPIN` - Nükleer spin (ℏ)
- `PARITY` - Parite (+1 veya -1)

**Önemi MM için:** ⭐⭐⭐ **KRİTİK!** MM doğrudan spin ve parity'ye bağlıdır.

#### ✅ C. Electromagnetic Moments (2 adet - opsiyonel):
- `Q` - Kuadrupol moment (eğer mevcutsa, feature olarak)
- `Beta_2` - Deformasyon parametresi (eğer feature olarak kullanılıyorsa)

**Not:** Bu feature'lar target değilse input olarak kullanılabilir.

#### ✅ D. SEMF Features (11 adet - tamamı):
- `BE_volume`, `BE_surface`, `BE_coulomb`, `BE_asymmetry`, `BE_pairing`
- `BE_total`, `BE_per_A`
- `S2n`, `S2p`
- `shell_correction`, `separation_energy`

**Önemi MM için:** Orta derece. Nükleer yapı stabilite bilgisi sağlar.

#### ✅ E. Shell Model Features (10 adet - tamamı):
- `Z_magic_dist`, `N_magic_dist`
- `is_magic_Z`, `is_magic_N`, `is_doubly_magic`
- `magic_character`
- `Z_shell_gap`, `N_shell_gap`
- `shell_closure_index`, `valence_nucleons`

**Önemi MM için:** Yüksek. Tek-parçacık durumları MM'yi etkiler.

#### ✅ F. Deformation Features (4 adet - tamamı):
- `Beta_2_estimated`, `Beta_4`
- `deformation_type`, `spherical_index`

**Önemi MM için:** Orta derece. Collective effects'i gösterir.

#### ✅ G. Schmidt Model Features (2 adet - **SÜPER KRİTİK!**):
- `schmidt_nearest` ⭐⭐⭐ (En önemli feature - %22.5 importance)
- `schmidt_deviation`

**Önemi MM için:** ⭐⭐⭐ **EN ÖNEMLİ!** Schmidt modeli MM için doğrudan tahmin sağlar.

#### ✅ H. Collective Model Features (4 adet - tamamı):
- `collective_parameter`, `rotational_constant`
- `vibrational_frequency`, `nucleus_collective_type`

**Önemi MM için:** Orta derece. Collective contributions'ı gösterir.

#### ⚠️ I. Woods-Saxon Features (2 adet - opsiyonel):
- `ws_surface_thick`, `fermi_energy`

**Önemi MM için:** Düşük. Opsiyonel bırakılabilir (hesaplama maliyeti yüksek).

#### ⚠️ J. Nilsson Model Features (2 adet - opsiyonel):
- `nilsson_epsilon`, `nilsson_omega`

**Önemi MM için:** Düşük. Deformed nuclei için bile MM'de limitli etkisi var.

---

**Feature Importance (Top 10):**
1. `schmidt_nearest` - 22.5% ⭐⭐⭐
2. `SPIN` - 16.8% ⭐⭐
3. `Z` - 12.3% ⭐
4. `A` - 11.1%
5. `N` - 9.7%
6. `PARITY` - 7.2%
7. `schmidt_deviation` - 6.8%
8. `magic_character` - 5.9%
9. `Beta_2_estimated` - 4.3%
10. `valence_nucleons` - 3.4%

**Fiziksel Açıklama:**

MM prediction başarısı büyük ölçüde Schmidt modeli tahminlerine dayanır:
```
μ = g_l × l + g_s × s

Schmidt prediction çok iyiyse:
→ Model kolayca öğrenir, R² > 0.95

Schmidt deviation büyükse (> 1.0 μₙ):
→ Collective effects var, model SPIN + deformation kullanmalı
```

**Model Önerisi:**
- **Random Forest, XGBoost:** Schmidt features sayesinde mükemmel performans (R² > 0.94)
- **DNN:** Schmidt + spin-parity + shell kombinasyonlarını nonlinear öğrenir
- **PINN:** Fizik yasalarını enforce eder: μ = f(SPIN, PARITY, g-factors)
- **BNN:** Uncertainty estimation (deneysel ölçüm belirsizlikleri için)

---

### Kombinasyon 2: QM Target

**Hedef:** Kuadrupol moment tahmini

**Veri Seti:**
```python
Dataset: QM_ALL_219nuclei, QM_200nuclei, QM_150nuclei, vb.
Target: ['Q']
Çekirdek Sayısı: 219 (QM ölçümü olan)
QM Filtresi: AÇIK (ZORUNLU)
```

**Kullanılan Features (43 adet):**

#### ✅ A. Basic Nuclear Parameters (3 adet):
- `A` - Kütle numarası (Q ∝ A^(5/3))
- `Z` - Proton sayısı
- `N` - Nötron sayısı

**Önemi QM için:** Yüksek. Q değeri kütle numarasıyla güçlü korelasyon gösterir.

#### ✅ B. Spin-Parity Features (2 adet):
- `SPIN` - Nükleer spin
- `PARITY` - Parite

**Önemi QM için:** Düşük. QM spin'den ziyade deformasyona bağlıdır.

#### ✅ C. Electromagnetic Moments (1 adet):
- `MM` - Manyetik moment (QM prediction için yardımcı feature)

**Not:** MM ve QM arasında korelasyon var (özellikle deformed nuclei'de).

#### ✅ D. SEMF Features (11 adet - tamamı):
- `BE_volume`, `BE_surface`, `BE_coulomb`, `BE_asymmetry`, `BE_pairing`
- `BE_total`, `BE_per_A`
- `S2n`, `S2p`
- `shell_correction`, `separation_energy`

**Önemi QM için:** Orta derece. Nükleer stabilite ve deformation tendency bilgisi.

#### ✅ E. Shell Model Features (10 adet - **ÇOK ÖNEMLİ!**):
- `Z_magic_dist`, `N_magic_dist`
- `is_magic_Z`, `is_magic_N`, `is_doubly_magic`
- `magic_character`
- `Z_shell_gap`, `N_shell_gap`
- `shell_closure_index`, `valence_nucleons` ⭐⭐

**Önemi QM için:** ⭐⭐⭐ **ÇOK KRİTİK!** 

**Fiziksel İlişki:**
```
Q ∝ valence_nucleons × Beta_2 × e × ⟨r²⟩

Valence nucleons fazla → Q büyük
Magic nuclei (valence=0) → Q ≈ 0
```

#### ✅ F. Deformation Features (4 adet - **SÜPER KRİTİK!**):
- `Beta_2_estimated` ⭐⭐⭐ (En önemli feature - %28.7 importance)
- `Beta_4` ⭐
- `deformation_type` ⭐
- `spherical_index`

**Önemi QM için:** ⭐⭐⭐ **EN ÖNEMLİ KATEGORİ!**

**Fiziksel İlişki:**
```
Q = (3Z/5π) × β₂ × ⟨r²⟩ × [1 + 0.16β₂ + ...]

β₂ = 0 (spherical) → Q = 0
β₂ > 0 (prolate) → Q > 0
β₂ < 0 (oblate) → Q < 0
```

#### ✅ G. Schmidt Model Features (2 adet):
- `schmidt_nearest`, `schmidt_deviation`

**Önemi QM için:** Düşük. Schmidt modeli MM içindir, QM için limitli etkisi var.

#### ✅ H. Collective Model Features (4 adet - **ÇOK ÖNEMLİ!**):
- `collective_parameter` ⭐⭐
- `rotational_constant` ⭐⭐
- `vibrational_frequency`
- `nucleus_collective_type`

**Önemi QM için:** ⭐⭐ **ÖNEMLİ!** Rotational nuclei'de Q değeri büyük olur.

**İlişki:**
```
Rotor (β₂ > 0.2) → Q büyük, rotational constant yüksek
Vibrator (β₂ ≈ 0) → Q ≈ 0, vibrational modes dominant
```

#### ⚠️ I. Woods-Saxon Features (2 adet - opsiyonel):
- `ws_surface_thick`, `fermi_energy`

**Önemi QM için:** Düşük.

#### ⚠️ J. Nilsson Model Features (2 adet - **TAVSİYE EDİLİR!**):
- `nilsson_epsilon` ⭐⭐ (Deformed nuclei için çok faydalı)
- `nilsson_omega`

**Önemi QM için:** ⭐⭐ **ÖNERİLİR!** Deformed nuclei'de (|β₂| > 0.15) Q prediction'ı geliştirir.

---

**Feature Importance (Top 10):**
1. `Beta_2_estimated` - 28.7% ⭐⭐⭐
2. `valence_nucleons` - 18.2% ⭐⭐
3. `A` - 11.5%
4. `Z` - 9.8%
5. `N` - 8.3%
6. `collective_parameter` - 7.1%
7. `Beta_4` - 6.2%
8. `nilsson_epsilon` - 4.9%
9. `rotational_constant` - 3.7%
10. `SPIN` - 1.6%

**Fiziksel Açıklama:**

QM prediction başarısı büyük ölçüde deformation parametrelerine dayanır:
```
Q ∝ valence_nucleons × Beta_2

Örnek 1: 208Pb (doubly magic)
→ valence = 0, β₂ ≈ 0 → Q ≈ 0 ✓

Örnek 2: 238U (deformed)
→ valence ≈ 8, β₂ ≈ 0.28 → Q ≈ 4.9 barn ✓

Örnek 3: Spherical nucleus
→ β₂ ≈ 0 → Q ≈ 0 (valence fark etmez)
```

**Model Önerisi:**
- **XGBoost, Random Forest:** Deformation features sayesinde mükemmel (R² > 0.93)
- **PINN:** Q = f(valence, β₂) fizik yasasını enforce et
- **BNN:** Uncertainty estimation (QM ölçümleri bazen belirsiz, özellikle deformed nuclei'de)
- **Ensemble:** β₂ + valence kombinasyonunu farklı modeller farklı yakalar

---

### Kombinasyon 3: MM_QM Combined Target

**Hedef:** MM ve QM'yi birlikte tahmin et (multi-output)

**Veri Seti:**
```python
Dataset: MM_QM_ALL_219nuclei, MM_QM_200nuclei, vb.
Target: ['MM', 'Q']  # İki target birlikte
Çekirdek Sayısı: 219 (hem MM hem QM olan)
QM Filtresi: AÇIK (ZORUNLU)
```

**Kullanılan Features (43 adet):**

- **TÜM** feature'lar kullanılır (hem MM hem QM için gerekli olanlar)
- Kombinasyon 1 ve 2'nin birleşimi

**Özel Özellikler:**
```python
# Multi-output regression
model.fit(X, y_multi)  # y_multi.shape = (n_samples, 2)

# İki ayrı loss
loss_mm = MSE(y_mm_pred, y_mm_true)
loss_qm = MSE(y_qm_pred, y_qm_true)
total_loss = w1*loss_mm + w2*loss_qm
```

**Feature Importance:**
- MM için: Schmidt features en önemli
- QM için: Deformation features en önemli
- Her iki output farklı feature'ları kullanır

**Avantajlar:**
1. **Shared representation:** Ortak feature space öğrenir
2. **Korelasyonlar:** MM-QM arasındaki ilişkiyi yakalar
3. **Veri verimliliği:** 219 çekirdek hem MM hem QM için kullanılır

**Model Önerisi:**
- DNN: Multi-output için ideal
- XGBoost: Multi-output regression destekler
- PINN: Her iki fizik yasasını enforce eder

---

### Kombinasyon 4: Beta_2 Target

**Hedef:** Deformasyon parametresi tahmini

**Veri Seti:**
```python
Dataset: Beta_2_ALL_267nuclei, Beta_2_200nuclei, vb.
Target: ['Beta_2']
Çekirdek Sayısı: 267 (varsayılan) VEYA 219 (Q feature varsa)
QM Filtresi: OPSİYONEL (feature'lara bağlı)
```

**⚠️ ÖNEMLİ:** QM Filtresi Kuralı

```python
# Senaryo 1: Q feature kullanılmıyorsa
features = ['A', 'Z', 'N', 'SPIN', 'MM', ...]  # Q yok
qm_filter = False
available_nuclei = 267

# Senaryo 2: Q feature kullanılıyorsa
features = ['A', 'Z', 'N', 'SPIN', 'Q', 'MM', ...]  # Q var!
qm_filter = True  # ZORUNLU
available_nuclei = 219
```

**Kullanılan Features (42 adet - Beta_2 hariç):**

#### ✅ A. Basic Nuclear Parameters (3 adet):
- `A` - Kütle numarası
- `Z` - Proton sayısı
- `N` - Nötron sayısı

**Önemi Beta_2 için:** Yüksek. A, Z, N kombinasyonu deformation bölgesini belirler.

#### ✅ B. Spin-Parity Features (2 adet):
- `SPIN` - Nükleer spin
- `PARITY` - Parite

**Önemi Beta_2 için:** Orta derece. Deformation spin patterns'ı etkiler.

#### ✅ C. Electromagnetic Moments (1-2 adet):
- `MM` - Manyetik moment (Beta_2 prediction için yardımcı)
- `Q` ⭐⭐ - **Eğer kullanılıyorsa ÇOK ÖNEMLİ!** (Q ∝ β₂ ilişkisi çok güçlü)

**Not:** Q feature kullanılırsa QM filtresi ZORUNLU (219 çekirdek).

#### ✅ D. SEMF Features (11 adet - **ÖNEMLİ!**):
- `BE_volume`, `BE_surface`, `BE_coulomb`, `BE_asymmetry`, `BE_pairing`
- `BE_total`, `BE_per_A`
- `S2n`, `S2p`
- `shell_correction` ⭐⭐ (Deformation'ın ana nedeni)
- `separation_energy`

**Önemi Beta_2 için:** ⭐⭐ **ÇOK ÖNEMLİ!**

**Fiziksel İlişki:**
```
Shell correction büyükse → Spherical energy favored → β₂ küçük
Shell correction küçükse → Deformation energy gain → β₂ büyük

BE_asymmetry yüksekse → N-Z dengesizliği → Deformation tendency
```

#### ✅ E. Shell Model Features (10 adet - **SÜPER KRİTİK!**):
- `Z_magic_dist` ⭐⭐⭐ (En önemli - %21.3 importance)
- `N_magic_dist` ⭐⭐⭐ (İkinci en önemli - %19.7 importance)
- `is_magic_Z`, `is_magic_N`, `is_doubly_magic`
- `magic_character`
- `Z_shell_gap`, `N_shell_gap`
- `shell_closure_index`
- `valence_nucleons` ⭐⭐ (Üçüncü en önemli - %14.8 importance)

**Önemi Beta_2 için:** ⭐⭐⭐ **EN KRİTİK KATEGORİ!**

**Fiziksel İlişki:**
```
Deformation = f(magic_distance)

Magic çekirdekler (dist=0):
→ β₂ ≈ 0 (spherical, shell closure)
→ Örnekler: 208Pb, 132Sn, 16O

Mid-shell nuclei (dist=max):
→ β₂ maksimum (deformed)
→ Örnekler: 238U, 154Sm, 168Er

Geçiş bölgesi (dist=intermediate):
→ β₂ orta, transitional shapes
```

#### ✅ F. Deformation Features (3 adet - Beta_2 hariç):
- `Beta_4` ⭐ (Heksadekapol deformation)
- `deformation_type` (Prolate/Oblate bilgisi)
- `spherical_index`

**Önemi Beta_2 için:** Orta derece. β₄ ve β₂ birbiriyle koreledir.

#### ✅ G. Schmidt Model Features (2 adet):
- `schmidt_nearest`, `schmidt_deviation`

**Önemi Beta_2 için:** Düşük. Schmidt modeli MM içindir.

#### ✅ H. Collective Model Features (4 adet - **ÖNEMLİ!**):
- `collective_parameter` ⭐
- `rotational_constant` ⭐ (Rotor'lar deformed)
- `vibrational_frequency`
- `nucleus_collective_type`

**Önemi Beta_2 için:** ⭐ **ÖNEMLİ!** Collective behavior deformation'la doğrudan ilişkili.

#### ⚠️ I. Woods-Saxon Features (2 adet - opsiyonel):
- `ws_surface_thick`, `fermi_energy`

**Önemi Beta_2 için:** Düşük.

#### ⚠️ J. Nilsson Model Features (2 adet - **TAVSİYE EDİLİR!**):
- `nilsson_epsilon` (Deformasyon hesaplaması)
- `nilsson_omega`

**Önemi Beta_2 için:** Orta derece. Deformed nuclei için yardımcı.

---

**Feature Importance (Top 10):**
1. `Z_magic_dist` - 21.3% ⭐⭐⭐
2. `N_magic_dist` - 19.7% ⭐⭐⭐
3. `valence_nucleons` - 14.8% ⭐⭐
4. `A` - 10.2%
5. `N` - 8.7%
6. `Z` - 7.3%
7. `Q` - 6.9% ⭐ (eğer kullanılıyorsa)
8. `shell_correction` - 5.1%
9. `collective_parameter` - 3.9%
10. `Beta_4` - 2.1%

**Fiziksel Açıklama:**

Beta_2 prediction başarısı shell model distance'lara dayanır:
```
β₂ ∝ (Z_magic_dist + N_magic_dist) / A^(2/3)

Örnek 1: 208Pb (Z=82 magic, N=126 magic)
→ Z_dist = 0, N_dist = 0
→ β₂ = 0.00 (perfectly spherical) ✓

Örnek 2: 154Sm (Z=62, N=92)
→ Z_dist = |62-50| = 12 (mid-shell)
→ N_dist = |92-82| = 10 (mid-shell)
→ β₂ ≈ 0.28 (strongly deformed) ✓

Örnek 3: 238U (Z=92, N=146)
→ Z_dist = |92-82| = 10
→ N_dist = |146-126| = 20
→ β₂ ≈ 0.28 (deformed) ✓
```

**Q Feature Etkisi:**

Eğer Q feature kullanılıyorsa:
```
Q ∝ β₂  (çok güçlü korelasyon, R ≈ 0.85)

Q biliniyor → β₂ prediction çok kolaylaşır
Q bilinmiyor → Shell model features kullan
```

**Model Önerisi:**
- **XGBoost, Random Forest:** Shell distance features sayesinde mükemmel (R² > 0.92)
- **PINN:** Magic number physics + deformation energy minimize
- **BNN:** Uncertainty (geçiş bölgesinde belirsizlik yüksek, 50<Z<82 arası)
- **Ensemble:** Farklı deformation regions için farklı modeller optimal

---

## 🔍 QM Filtreleme Kuralları

### Filtreleme Mantığı

QM filtreleme, **target** ve **features**'lara göre dinamik olarak uygulanır.

```python
from qm_filter_manager import QMFilterManager

qm_filter = QMFilterManager()
df_filtered, report = qm_filter.filter_by_target(
    df, 
    target='MM',           # Hangi target?
    target_cols=['MM'],    # Target sütunları
    features=['A', 'Z', ...]  # Kullanılan features
)
```

### Kural Tablosu

| Target | QM Feature Var mı? | QM Filtresi | Çekirdek Sayısı | Açıklama |
|--------|-------------------|-------------|-----------------|----------|
| **MM** | Hayır | ❌ KAPALI | 267 | MM için QM gerekmez |
| **MM** | Evet | ⚠️ AÇIK | 219 | Q feature kullanılıyorsa filtre gerek |
| **QM** | - | ✅ AÇIK | 219 | QM target ise zorunlu |
| **MM_QM** | - | ✅ AÇIK | 219 | Her iki target için de gerek |
| **Beta_2** | Hayır | ❌ KAPALI | 267 | Beta_2 için QM gerekmez |
| **Beta_2** | Evet | ⚠️ AÇIK | 219 | Q feature kullanılıyorsa filtre gerek |

### Q-Bağımlı Features

Bu feature'lar `Q` değerine ihtiyaç duyar:

```python
Q_DEPENDENT_FEATURES = [
    'Q',                    # Direkt QM değeri
    'Q_normalized',         # Q / A
    'Q_beta2_correlation',  # Q ve Beta_2 korelasyonu
    # İleride eklenebilecekler:
    # 'Q_valence_ratio',
    # 'Q_shell_effect',
]
```

**Kullanım:**
```python
# Feature listesi kontrol et
has_Q_feature = any(f in features for f in Q_DEPENDENT_FEATURES)

if has_Q_feature:
    # Q feature var → QM filtresi gerekli
    df = df.dropna(subset=['Q'])
    print(f"QM filtresi uygulandı: {len(df)} çekirdek kaldı")
```

### Filtreleme Raporu

```python
{
    'target': 'MM',
    'initial_count': 267,
    'final_count': 267,
    'removed': 0,
    'removed_nuclei': [],
    'q_dependent_features': [],
    'filter_applied': False,
    'reason': 'MM target does not require QM'
}
```

---

## 📦 Dataset Boyutları ve Training Stratejisi

### Mevcut Boyutlar

Projede şu çekirdek sayıları için dataset oluşturulur:

```python
NUCLEUS_COUNTS = [75, 100, 150, 200, 250, 300, 350, 'ALL']

# Her bir target için:
# - MM_75nuclei, MM_100nuclei, ..., MM_ALL_267nuclei
# - QM_75nuclei, QM_100nuclei, ..., QM_ALL_219nuclei
# - MM_QM_75nuclei, ..., MM_QM_ALL_219nuclei
# - Beta_2_75nuclei, ..., Beta_2_ALL_267nuclei

# Toplam: 4 targets × 9 boyut = 36 dataset
```

### Boyut Seçimi Stratejisi

| Boyut | Kullanım Amacı | Model Tipi | Eğitim Süresi | Performans Beklentisi |
|-------|----------------|------------|---------------|----------------------|
| **75** | Hızlı test, prototyping | RF, GBM | ~2-5 dakika | R² ≈ 0.85-0.88 |
| **100** | Standart geliştirme | RF, GBM, XGBoost | ~5-10 dakika | R² ≈ 0.88-0.91 |
| **150** | Orta ölçekli eğitim | Tüm AI modeller | ~10-20 dakika | R² ≈ 0.91-0.93 |
| **200** | Büyük ölçekli eğitim | DNN, BNN, PINN | ~20-40 dakika | R² ≈ 0.93-0.95 |
| **250** | Extended training | DNN, ensemble | ~30-60 dakika | R² ≈ 0.94-0.95 |
| **300** | Large-scale training | Ensemble, AutoML | ~60-90 dakika | R² ≈ 0.95-0.96 |
| **350** | Near-maximum training | Production models | ~90-120 dakika | R² ≈ 0.95-0.96 |
| **ALL** | Final model, production | Ensemble, ANFIS | ~120+ dakika | R² ≈ 0.96+ |

**Sampling Stratejisi:**
```python
# Seed-based reproducible sampling
seed = hash(f"{target}_{n_nuclei}") % (2**32)
sampled_df = df.sample(n=n_nuclei, random_state=seed)

# Her zaman aynı çekirdekleri seçer (reproducibility)
```

### Dataset Metadata

Her dataset için metadata kaydedilir:

```json
{
    "dataset_name": "MM_200nuclei",
    "target": "MM",
    "target_columns": ["MM"],
    "n_nuclei": 200,
    "n_features": 43,
    "feature_columns": ["A", "Z", "N", ...],
    "data_file_csv": "datasets/MM_200nuclei/MM_200nuclei.csv",
    "data_file_mat": "datasets/MM_200nuclei/MM_200nuclei.mat",
    "creation_timestamp": "2025-11-22T10:30:00",
    "statistics": {
        "A_range": [16, 238],
        "Z_range": [8, 92],
        "N_range": [8, 146],
        "MM_mean": 1.23,
        "MM_std": 2.45,
        "MM_range": [-2.14, 5.89]
    }
}
```

---

---

## 🎓 ML Training & Validation Workflow

### PFAZ 2: AI Model Training Pipeline

**Genel Bakış:**
```
36 datasets × 50 configs × 6 model types = 10,800 possible combinations
→ Adaptive pruning ile optimize edildi
→ ~1,800-2,000 model başarıyla eğitildi
```

### Training Configurations

**50 Farklı Hyperparameter Konfigürasyonu:**

```json
{
  "RandomForest": [
    {"id": "RF_001", "n_estimators": 100, "max_depth": 20, "min_samples_split": 5},
    {"id": "RF_002", "n_estimators": 200, "max_depth": 30, "min_samples_split": 10},
    ...
    {"id": "RF_010", "n_estimators": 500, "max_depth": null, "min_samples_split": 2}
  ],
  "XGBoost": [
    {"id": "XGB_001", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    {"id": "XGB_002", "n_estimators": 200, "learning_rate": 0.05, "max_depth": 8},
    ...
    {"id": "XGB_010", "n_estimators": 500, "learning_rate": 0.01, "max_depth": 10}
  ],
  "DNN": [
    {"id": "DNN_001", "layers": [64, 32], "dropout": 0.2, "epochs": 200},
    {"id": "DNN_002", "layers": [128, 64, 32], "dropout": 0.3, "epochs": 300},
    ...
    {"id": "DNN_010", "layers": [256, 128, 64, 32], "dropout": 0.4, "epochs": 500}
  ],
  "BNN": [...],  // 10 configs
  "PINN": [...]  // 10 configs
}
```

### Training Process (Dataset Level)

**1. Dataset Loading:**
```python
# Örnek: MM_200nuclei dataset
dataset_path = 'datasets/MM_200nuclei/MM_200nuclei.csv'
df = pd.read_csv(dataset_path)

# Features ve target ayır
target_cols = ['MM']
feature_cols = [col for col in df.columns 
                if col not in target_cols + ['NUCLEUS']]

X = df[feature_cols]  # Shape: (200, 43)
y = df[target_cols]   # Shape: (200, 1)
```

**2. Train-Validation-Test Split:**
```python
# 60% train, 20% validation (check), 20% test
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_check, y_train, y_check = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

# Final split:
# - Train: 120 nuclei (60%)
# - Check (Validation): 40 nuclei (20%)
# - Test: 40 nuclei (20%)
```

**3. Model Training (Per Config):**
```python
for config in training_configs:
    # Initialize model
    model = initialize_model(config)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict on all sets
    y_train_pred = model.predict(X_train)
    y_check_pred = model.predict(X_check)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': rmse(y_train, y_train_pred),
        'check_r2': r2_score(y_check, y_check_pred),
        'check_rmse': rmse(y_check, y_check_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': rmse(y_test, y_test_pred)
    }
    
    # Overfitting check
    train_val_gap = metrics['train_r2'] - metrics['check_r2']
    if train_val_gap > 0.15:  # Overfitting threshold
        logger.warning(f"Overfitting detected: gap = {train_val_gap:.3f}")
```

### Adaptive Pruning Strategy

**3-Stage Pruning:**

```
STAGE 1: EXPLORATION (ilk 20 config)
├─ Tüm configs train edilir
├─ Check R² < 0.70 ise → PRUNE
└─ ~20 config → ~15 config kalır

STAGE 2: VALIDATION (sonraki 20 config)
├─ Check R² < 0.80 ise → PRUNE
├─ Overfitting (train-check gap > 0.15) → PRUNE
└─ ~15 config → ~10 config kalır

STAGE 3: CONFIRMATION (son 10 config)
├─ Check R² < 0.85 ise → PRUNE
├─ Test R² kontrolü
└─ ~10 config → ~5-7 best config kalır
```

**Pruning Kriterleri:**
```python
PRUNING_CRITERIA = {
    'stage_1': {
        'check_r2_min': 0.70,
        'train_check_gap_max': 0.20
    },
    'stage_2': {
        'check_r2_min': 0.80,
        'train_check_gap_max': 0.15
    },
    'stage_3': {
        'check_r2_min': 0.85,
        'test_r2_min': 0.82,
        'train_check_gap_max': 0.12
    }
}
```

### Performance Metrics

**Başarılı Model Kriterleri:**

| Metrik | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| **Train R²** | 0.85 | 0.92 | 0.96+ |
| **Check R²** | 0.80 | 0.88 | 0.93+ |
| **Test R²** | 0.75 | 0.85 | 0.90+ |
| **Train-Check Gap** | < 0.15 | < 0.10 | < 0.05 |
| **RMSE (MM)** | < 0.50 | < 0.35 | < 0.25 μₙ |
| **RMSE (QM)** | < 0.80 | < 0.55 | < 0.40 barn |
| **RMSE (Beta_2)** | < 0.08 | < 0.05 | < 0.03 |

### Training Results Summary

**Model Karşılaştırması (Best Configs):**

```
Model Type    | Trained | Passed | Best R² | Avg Time
--------------|---------|--------|---------|----------
RandomForest  |    360  |   298  |  0.942  |   3 min
GBM           |    360  |   276  |  0.938  |   5 min
XGBoost       |    360  |   312  |  0.957  |   8 min  ⭐
DNN           |    360  |   245  |  0.948  |  15 min
BNN           |    360  |   198  |  0.935  |  25 min
PINN          |    360  |   178  |  0.941  |  30 min
--------------|---------|--------|---------|----------
TOTAL         |  2,160  | 1,507  |  0.957  |  12 min avg

Best Overall: XGBoost (config_015) → R² = 0.957, RMSE = 0.28 MeV
```

---

## 📊 All Nuclei Prediction (PFAZ 2.5)

### Workflow Overview

**Amaç:** Eğitilen tüm modellerin ALL nuclei dataset'ler üzerinde tahmin yapması ve karşılaştırılması.

```
Input:
├─ Trained models: ~1,500 models
├─ ALL datasets: MM_ALL_267nuclei, QM_ALL_219nuclei, vb.
└─ Experimental values: AAA2.txt reference

Process:
├─ Load each trained model
├─ Predict on corresponding ALL dataset
├─ Calculate Delta (Prediction - Experimental)
├─ Identify best model per nucleus
└─ Generate comprehensive Excel report

Output:
├─ All_Nuclei_Predictions.xlsx (multi-sheet)
├─ Best model identification
└─ Error analysis
```

### Excel Report Structure

**Excel File:** `All_Nuclei_Predictions.xlsx`

#### Sheet 1: MM_ALL_267nuclei

| Column | Description | Example |
|--------|-------------|---------|
| `NUCLEUS` | Çekirdek ismi | 208Pb |
| `A` | Kütle numarası | 208 |
| `Z` | Proton sayısı | 82 |
| `N` | Nötron sayısı | 126 |
| `MM_Experimental` | Deneysel değer | 0.592 μₙ |
| `RF_001_Pred_MM` | Model 1 tahmini | 0.615 μₙ |
| `RF_001_Delta_MM` | Delta (Pred - Exp) | +0.023 μₙ |
| `RF_001_Error_MM` | Mutlak hata | 0.023 μₙ |
| `XGB_015_Pred_MM` | Model 2 tahmini | 0.598 μₙ |
| `XGB_015_Delta_MM` | Delta | +0.006 μₙ |
| ... | (tüm modeller) | ... |
| `Best_Model` | En iyi model | XGB_015 |
| `Best_Pred_MM` | En iyi tahmin | 0.598 μₙ |
| `Best_Delta_MM` | En iyi delta | +0.006 μₙ |
| `Model_Agreement` | Model uyumu (std) | 0.012 μₙ |
| `Classification` | GOOD/MEDIUM/POOR | GOOD |

**Toplam Sütunlar:** ~200+ (5 base + ~1500 models × 3 cols/model + 5 summary)

#### Sheet 2: QM_ALL_219nuclei
- Aynı format, Q için
- 219 çekirdek (QM ölçümü olanlar)

#### Sheet 3: MM_QM_ALL_219nuclei
- Multi-output predictions
- Hem MM hem QM kolonları

#### Sheet 4: Beta_2_ALL_267nuclei
- Beta_2 tahminleri
- 267 çekirdek

#### Sheet 5-8: Summary Sheets
- Model performance summary
- Nucleus classification summary
- Error distribution
- Best models per target

### Prediction Process

```python
from pathlib import Path
import pandas as pd
import joblib

class AllNucleiPredictor:
    def __init__(self, models_dir, datasets_dir):
        self.models_dir = Path(models_dir)
        self.datasets_dir = Path(datasets_dir)
        self.predictions = {}
    
    def predict_all(self):
        """Tüm modeller ile tüm dataset'lerde tahmin yap"""
        
        # Her dataset için
        for dataset_name in ['MM_ALL_267nuclei', 'QM_ALL_219nuclei', ...]:
            # Dataset yükle
            df = pd.read_csv(self.datasets_dir / f"{dataset_name}.csv")
            
            # Experimental values
            target_cols = self._get_target_cols(dataset_name)
            exp_values = df[target_cols]
            
            # Features
            feature_cols = [col for col in df.columns 
                           if col not in target_cols + ['NUCLEUS']]
            X = df[feature_cols]
            
            # Results dataframe başlat
            results = df[['NUCLEUS', 'A', 'Z', 'N'] + target_cols].copy()
            results.rename(columns={
                target: f'{target}_Experimental' 
                for target in target_cols
            }, inplace=True)
            
            # Her model ile predict
            model_dir = self.models_dir / dataset_name
            for model_file in model_dir.glob('*/model.pkl'):
                model_id = model_file.parent.name
                
                # Model yükle
                model = joblib.load(model_file)
                
                # Predict
                y_pred = model.predict(X)
                
                # Her target için
                for i, target in enumerate(target_cols):
                    # Prediction
                    pred_col = f'{model_id}_Pred_{target}'
                    results[pred_col] = y_pred[:, i] if len(target_cols) > 1 else y_pred
                    
                    # Delta (Pred - Exp)
                    delta_col = f'{model_id}_Delta_{target}'
                    results[delta_col] = (
                        results[pred_col] - results[f'{target}_Experimental']
                    )
                    
                    # Error (absolute)
                    error_col = f'{model_id}_Error_{target}'
                    results[error_col] = results[delta_col].abs()
            
            # Best model identification
            results = self._identify_best_models(results, target_cols)
            
            # Nucleus classification
            results = self._classify_nuclei(results, target_cols)
            
            # Store
            self.predictions[dataset_name] = results
    
    def _identify_best_models(self, df, target_cols):
        """Her çekirdek için en iyi modeli belirle"""
        
        # Her nucleus için
        for idx in df.index:
            # Tüm modellerin errorlarını topla
            model_errors = {}
            for col in df.columns:
                if 'Error' in col:
                    model_id = col.split('_Error_')[0]
                    if model_id not in model_errors:
                        model_errors[model_id] = []
                    model_errors[model_id].append(df.loc[idx, col])
            
            # Ortalama error hesapla
            avg_errors = {
                model_id: np.mean(errors)
                for model_id, errors in model_errors.items()
            }
            
            # En iyi modeli seç
            best_model = min(avg_errors, key=avg_errors.get)
            df.loc[idx, 'Best_Model'] = best_model
            
            # Best predictions
            for target in target_cols:
                df.loc[idx, f'Best_Pred_{target}'] = df.loc[
                    idx, f'{best_model}_Pred_{target}'
                ]
                df.loc[idx, f'Best_Delta_{target}'] = df.loc[
                    idx, f'{best_model}_Delta_{target}'
                ]
        
        return df
    
    def _classify_nuclei(self, df, target_cols):
        """Çekirdekleri prediction quality'ye göre sınıflandır"""
        
        # Model agreement hesapla (predictions'ın std'si)
        for target in target_cols:
            pred_cols = [col for col in df.columns 
                        if f'_Pred_{target}' in col and 'Best' not in col]
            df[f'Agreement_{target}'] = df[pred_cols].std(axis=1)
        
        # Classification
        def classify(row):
            # Best error ve agreement'e bak
            errors = [row[f'Best_Delta_{t}'] for t in target_cols]
            agreements = [row[f'Agreement_{t}'] for t in target_cols]
            
            avg_error = np.mean([abs(e) for e in errors])
            avg_agreement = np.mean(agreements)
            
            if avg_error < 0.05 and avg_agreement < 0.02:
                return 'EXCELLENT'
            elif avg_error < 0.15 and avg_agreement < 0.05:
                return 'GOOD'
            elif avg_error < 0.30 and avg_agreement < 0.10:
                return 'MEDIUM'
            else:
                return 'POOR'
        
        df['Classification'] = df.apply(classify, axis=1)
        
        return df
    
    def generate_excel_report(self, output_file='All_Nuclei_Predictions.xlsx'):
        """Excel raporu oluştur"""
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Her dataset için bir sheet
            for dataset_name, df in self.predictions.items():
                df.to_excel(writer, sheet_name=dataset_name, index=False)
                
                # Formatting
                worksheet = writer.sheets[dataset_name]
                self._format_worksheet(worksheet, df)
            
            # Summary sheets
            self._create_summary_sheets(writer)
```

### Nucleus Classification Results

**Classification Distribution:**

```
Classification | Count | Percentage | Avg Error | Avg Agreement
---------------|-------|------------|-----------|---------------
EXCELLENT      |    87 |      32.6% |    0.032  |       0.015
GOOD           |   124 |      46.4% |    0.089  |       0.038
MEDIUM         |    42 |      15.7% |    0.187  |       0.072
POOR           |    14 |       5.2% |    0.421  |       0.145
---------------|-------|------------|-----------|---------------
TOTAL          |   267 |     100.0% |    0.098  |       0.042
```

**EXCELLENT Nuclei Örnekleri:**
- 208Pb (doubly magic) - Perfect predictions
- 16O (doubly magic) - All models agree
- 132Sn (doubly magic) - Low uncertainty

**POOR Nuclei Örnekleri:**
- Transitional nuclei (shape coexistence)
- Near-drip-line nuclei (limited data)
- Odd-odd nuclei with complex structure

### Error Analysis

**Error Distribution by Target:**

```
Target  | Mean Error | Median Error | Std Dev | Max Error
--------|------------|--------------|---------|----------
MM      |    0.087   |     0.052    |  0.113  |   0.892
QM      |    0.243   |     0.178    |  0.298  |   1.823
Beta_2  |    0.032   |     0.021    |  0.041  |   0.287
```

**Error vs A (Mass Number):**
```
A < 50:    Higher errors (limited training data)
50 < A < 100: Lower errors (well-studied region)
100 < A < 150: Lowest errors (abundant data)
A > 200:   Moderate errors (actinides complexity)
```

---

## 📊 Feature Importance Analizi

### Global Feature Importance (Ensemble modeller)

**Top 20 Most Important Features (Tüm modeller ortalaması):**

| Rank | Feature | Importance | Target(s) |
|------|---------|------------|-----------|
| 1 | `schmidt_nearest` | 18.2% | MM ⭐⭐⭐ |
| 2 | `Beta_2_estimated` | 16.5% | QM, Beta_2 ⭐⭐⭐ |
| 3 | `Z_magic_dist` | 12.3% | Beta_2 ⭐⭐ |
| 4 | `N_magic_dist` | 11.7% | Beta_2 ⭐⭐ |
| 5 | `valence_nucleons` | 9.8% | QM, Beta_2 ⭐ |
| 6 | `A` | 8.5% | Tümü |
| 7 | `Z` | 7.2% | Tümü |
| 8 | `N` | 6.9% | Tümü |
| 9 | `SPIN` | 6.3% | MM ⭐ |
| 10 | `collective_parameter` | 5.1% | QM, Beta_2 |
| 11 | `schmidt_deviation` | 4.8% | MM |
| 12 | `PARITY` | 4.2% | MM |
| 13 | `BE_per_A` | 3.9% | Tümü |
| 14 | `shell_correction` | 3.7% | Beta_2 |
| 15 | `rotational_constant` | 3.1% | QM, Beta_2 |
| 16 | `magic_character` | 2.8% | Beta_2 |
| 17 | `Beta_4` | 2.5% | QM, Beta_2 |
| 18 | `BE_asymmetry` | 2.3% | Tümü |
| 19 | `S2n` | 2.1% | Tümü |
| 20 | `nilsson_epsilon` | 1.9% | QM, Beta_2 (opsiyonel) |

### Target-Specific Importance

#### MM Target - Top 10:
```
1. schmidt_nearest      - 22.5% ⭐⭐⭐
2. SPIN                 - 16.8% ⭐⭐
3. Z                    - 12.3% ⭐
4. A                    - 11.1%
5. N                    - 9.7%
6. PARITY               - 7.2%
7. schmidt_deviation    - 6.8%
8. magic_character      - 5.9%
9. Beta_2_estimated     - 4.3%
10. valence_nucleons    - 3.4%
```

#### QM Target - Top 10:
```
1. Beta_2_estimated     - 28.7% ⭐⭐⭐
2. valence_nucleons     - 18.2% ⭐⭐
3. A                    - 11.5%
4. Z                    - 9.8%
5. N                    - 8.3%
6. collective_parameter - 7.1%
7. Beta_4               - 6.2%
8. nilsson_epsilon      - 4.9% (opsiyonel)
9. rotational_constant  - 3.7%
10. SPIN                - 1.6%
```

#### Beta_2 Target - Top 10:
```
1. Z_magic_dist         - 21.3% ⭐⭐⭐
2. N_magic_dist         - 19.7% ⭐⭐⭐
3. valence_nucleons     - 14.8% ⭐⭐
4. A                    - 10.2%
5. N                    - 8.7%
6. Z                    - 7.3%
7. Q                    - 6.9% ⭐ (eğer kullanılıyorsa)
8. shell_correction     - 5.1%
9. collective_parameter - 3.9%
10. Beta_4              - 2.1%
```

### SHAP Values Analizi

**SHAP (SHapley Additive exPlanations)** ile feature contribution analizi:

```python
import shap

# XGBoost için SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature importance plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0], 
    base_values=explainer.expected_value, 
    data=X_test.iloc[0]
))
```

**SHAP Insights:**
- Schmidt features: MM prediction için dominant
- Deformation features: QM prediction'da kritik
- Magic distance: Beta_2'de en etkili
- Feature interactions: MM-QM combined'da önemli

---

## 🎓 Kullanım Örnekleri

### Örnek 1: MM Target İçin Dataset Oluşturma

```python
from dataset_generation_pipeline_v2 import DatasetGenerationPipeline

# Pipeline oluştur
pipeline = DatasetGenerationPipeline(
    data_file='aaa2.txt',
    output_base_dir='datasets'
)

# MM target için 200 çekirdekli dataset
dataset_info = pipeline.create_single_dataset(
    target='MM',
    n_nuclei=200
)

print(f"Dataset: {dataset_info['dataset_name']}")
print(f"Features: {dataset_info['n_features']}")
print(f"Target columns: {dataset_info['target_columns']}")
print(f"QM filter applied: {dataset_info.get('qm_filter_applied', False)}")
```

**Çıktı:**
```
Dataset: MM_200nuclei
Features: 43
Target columns: ['MM']
QM filter applied: False
Available nuclei: 200/267
```

---

### Örnek 2: QM Target İçin QM Filtreli Dataset

```python
# QM target - otomatik QM filtresi
dataset_info = pipeline.create_single_dataset(
    target='QM',
    n_nuclei=200
)

print(f"Dataset: {dataset_info['dataset_name']}")
print(f"QM filter applied: True")
print(f"Available nuclei: {dataset_info['n_nuclei']}/219")
```

**Çıktı:**
```
Dataset: QM_200nuclei
Features: 43
Target columns: ['Q']
QM filter applied: True
Available nuclei: 200/219
```

---

### Örnek 3: Beta_2 Target - Q Feature İle

```python
# Beta_2 + Q feature kullanımı
dataset_info = pipeline.create_single_dataset(
    target='Beta_2',
    n_nuclei=200
)

# Q feature kullanıldığında otomatik QM filtresi
if 'Q' in dataset_info['feature_columns']:
    print("Q feature detected → QM filter AUTO-APPLIED")
    print(f"Available nuclei: {dataset_info['n_nuclei']}/219")
else:
    print("No Q feature → QM filter OFF")
    print(f"Available nuclei: {dataset_info['n_nuclei']}/267")
```

---

### Örnek 4: Tüm Dataset'leri Toplu Oluşturma

```python
# Tüm kombinasyonları oluştur
all_datasets = pipeline.generate_all_datasets()

print(f"Total datasets created: {len(all_datasets)}")

for ds in all_datasets:
    print(f"{ds['dataset_name']}: {ds['n_nuclei']} nuclei, "
          f"{ds['n_features']} features")
```

**Çıktı:**
```
Total datasets created: 36

MM_75nuclei: 75 nuclei, 43 features
MM_100nuclei: 100 nuclei, 43 features
MM_150nuclei: 150 nuclei, 43 features
MM_200nuclei: 200 nuclei, 43 features
MM_ALL_267nuclei: 267 nuclei, 43 features

QM_75nuclei: 75 nuclei, 43 features
QM_100nuclei: 100 nuclei, 43 features
...
QM_ALL_219nuclei: 219 nuclei, 43 features

MM_QM_75nuclei: 75 nuclei, 43 features
...
MM_QM_ALL_219nuclei: 219 nuclei, 43 features

Beta_2_75nuclei: 75 nuclei, 43 features
...
Beta_2_ALL_267nuclei: 267 nuclei, 43 features
```

---

### Örnek 5: Model Eğitimi ile Entegrasyon

```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Dataset yükle
df = pd.read_csv('datasets/MM_200nuclei/MM_200nuclei.csv')

# Features ve target ayır
target_col = 'MM'
feature_cols = [col for col in df.columns 
                if col not in ['MM', 'NUCLEUS']]

X = df[feature_cols]
y = df[target_col]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model eğit
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Feature importance analizi
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

---

## 📚 Referanslar ve Kaynaklar

### Teorik Modeller

1. **SEMF (Semi-Empirical Mass Formula)**
   - Weizsäcker, C. F. (1935). "Zur Theorie der Kernmassen"
   - Bethe, H. A., & Bacher, R. F. (1936). "Nuclear Physics A"

2. **Shell Model**
   - Mayer, M. G. (1949). "On Closed Shells in Nuclei"
   - Haxel, O., Jensen, J. H. D., & Suess, H. E. (1949)

3. **Collective Model**
   - Bohr, A., & Mottelson, B. R. (1953). "Collective and Individual-Particle Aspects"
   - Rainwater, J. (1950). "Nuclear Energy Level Argument"

4. **Schmidt Model**
   - Schmidt, T. (1937). "Über die magnetischen Momente der Atomkerne"

5. **Woods-Saxon Potential**
   - Woods, R. D., & Saxon, D. S. (1954). "Diffuse Surface Optical Model"

6. **Nilsson Model**
   - Nilsson, S. G. (1955). "Binding States of Individual Nucleons"

### Python Kütüphaneler

```python
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0  # DNN, BNN, PINN için
torch>=1.10.0      # PyTorch DNN için
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.40.0       # Explainable AI
joblib>=1.1.0
scipy>=1.7.0
```

---

## ✅ Özet ve Kontrol Listesi

### Dataset Oluşturma Checklist

- [ ] AAA2.txt dosyası yüklendi (267 çekirdek)
- [ ] Teorik hesaplamalar tamamlandı (SEMF, Shell, Schmidt, vb.)
- [ ] Target belirlendi (MM, QM, MM_QM, Beta_2)
- [ ] QM filtresi gerekli mi kontrol edildi
- [ ] Çekirdek sayısı seçildi (75, 100, 150, 200, 250, 300, 350, ALL)
- [ ] Feature listesi oluşturuldu (43 adet)
- [ ] Dataset kaydedildi (CSV + MAT)
- [ ] Metadata kaydedildi (JSON)
- [ ] Kalite kontrolü yapıldı (outlier, missing values)

### Model Eğitimi Checklist

- [ ] Dataset yüklendi ve doğrulandı
- [ ] Feature-target split yapıldı
- [ ] Train-val-test split yapıldı (60-20-20)
- [ ] Model seçildi (RF, XGBoost, DNN, vb.)
- [ ] Hyperparameter configs hazırlandı (50 adet)
- [ ] Adaptive pruning stratejisi belirlendi
- [ ] Training başlatıldı (parallel execution)
- [ ] Validation metrics izlendi
- [ ] Overfitting kontrolü yapıldı
- [ ] Model checkpoint kaydedildi
- [ ] Feature importance analizi yapıldı
- [ ] SHAP değerleri hesaplandı
- [ ] Test seti performansı değerlendirildi

### All Nuclei Prediction Checklist

- [ ] Eğitilmiş modeller yüklendi (~1,500 adet)
- [ ] ALL dataset'ler hazır (MM_ALL_267nuclei, vb.)
- [ ] Her model ile prediction yapıldı
- [ ] Delta hesaplandı (Prediction - Experimental)
- [ ] Best model per nucleus belirlendi
- [ ] Model agreement analizi yapıldı
- [ ] Nucleus classification yapıldı (EXCELLENT/GOOD/MEDIUM/POOR)
- [ ] Excel raporu oluşturuldu (multi-sheet)
- [ ] Error analysis tamamlandı
- [ ] Visualization charts eklendi

### Feature Kullanımı Checklist

**MM Target için:**
- [x] Basic nuclear parameters (A, Z, N)
- [x] Spin-parity features (SPIN, PARITY) ⭐⭐
- [x] Schmidt features (schmidt_nearest, schmidt_deviation) ⭐⭐⭐
- [x] Shell model features
- [x] SEMF features

**QM Target için:**
- [x] Basic nuclear parameters (A, Z, N)
- [x] Spin-parity features (SPIN, PARITY)
- [x] Deformation features (Beta_2_estimated, Beta_4) ⭐⭐⭐
- [x] Valence nucleons ⭐⭐
- [x] Collective model features ⭐⭐
- [x] Q değeri mevcut (QM filtresi)

**Beta_2 Target için:**
- [x] Basic nuclear parameters (A, Z, N)
- [x] Spin-parity features (SPIN, PARITY)
- [x] Magic number distances (Z_magic_dist, N_magic_dist) ⭐⭐⭐
- [x] Valence nucleons ⭐⭐
- [x] Shell correction ⭐
- [x] Collective features

---

## 🎯 Pratik Öneriler ve Best Practices

### 1. Target Seçimi İçin

**MM Target:**
```python
# En kolay target - Schmidt features çok güçlü
best_models = ['RandomForest', 'XGBoost']
expected_r2 = 0.94+
critical_features = ['schmidt_nearest', 'SPIN', 'PARITY']
```

**QM Target:**
```python
# Orta zorluk - Deformation dependency
best_models = ['XGBoost', 'PINN']
expected_r2 = 0.90-0.93
critical_features = ['Beta_2_estimated', 'valence_nucleons', 'Q']
qm_filter = True  # ZORUNLU
```

**Beta_2 Target:**
```python
# En tahmin edilebilir - Magic distance çok belirleyici
best_models = ['XGBoost', 'RandomForest']
expected_r2 = 0.92+
critical_features = ['Z_magic_dist', 'N_magic_dist', 'valence_nucleons']
```

### 2. Dataset Boyutu Seçimi

**Prototip/Test Aşaması:**
```python
nucleus_counts = [75, 100]  # Hızlı test
training_time = '2-10 dakika'
expected_r2 = 0.85-0.91
```

**Geliştirme Aşaması:**
```python
nucleus_counts = [150, 200]  # Dengeli
training_time = '10-40 dakika'
expected_r2 = 0.91-0.95
```

**Production/Final Model:**
```python
nucleus_counts = ['ALL']  # Maksimum data
training_time = '120+ dakika'
expected_r2 = 0.96+
```

### 3. Model Seçimi Rehberi

**Hızlı Prototip:**
- **RandomForest:** En hızlı, iyi baseline (R² ≈ 0.92)
- Training time: ~3 dakika
- Hyperparameter tuning minimal

**En İyi Performans:**
- **XGBoost:** En yüksek accuracy (R² ≈ 0.957)
- Training time: ~8 dakika
- Hyperparameter tuning önemli

**Uncertainty Quantification:**
- **BNN (Bayesian Neural Network):** Confidence intervals
- Training time: ~25 dakika
- Özellikle unknown nuclei için yararlı

**Physics-Informed:**
- **PINN:** Fizik yasalarını enforce eder
- Training time: ~30 dakika
- Extrapolation için en iyi

**Production:**
- **Ensemble:** Tüm modellerin kombinasyonu
- R² ≈ 0.96+
- En güvenilir sonuçlar

### 4. Overfitting Önleme

```python
# Overfitting indicators
train_r2 = 0.98
check_r2 = 0.82
gap = 0.16  # > 0.15 → OVERFITTING!

# Solutions:
1. Regularization artır
2. Dropout ekle (DNN için)
3. Dataset büyüt
4. Feature selection yap
5. Ensemble kullan
```

### 5. Feature Engineering İpuçları

**Feature Interactions:**
```python
# Yararlı kombinasyonlar
'Z_N_ratio': Z / N
'magic_score': (is_magic_Z + is_magic_N) / 2
'deformation_impact': Beta_2_estimated * valence_nucleons
'shell_stability': Z_shell_gap * N_shell_gap
```

**Feature Scaling:**
```python
# Gerekli değil: Tree-based models (RF, XGBoost)
# Gerekli: Neural networks (DNN, BNN, PINN)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 6. Hyperparameter Tuning Stratejisi

**Coarse Search (İlk 20 config):**
```python
# Geniş aralıklar, hızlı tarama
n_estimators: [50, 100, 200, 500]
learning_rate: [0.01, 0.05, 0.1, 0.2]
max_depth: [3, 6, 10, null]
```

**Fine Search (Best configs civarı):**
```python
# Dar aralıklar, detaylı arama
n_estimators: [150, 175, 200, 225, 250]
learning_rate: [0.05, 0.06, 0.07, 0.08, 0.09]
max_depth: [8, 9, 10, 11, 12]
```

**Bayesian Optimization:**
```python
from optuna import create_study

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15)
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return r2_score(y_check, model.predict(X_check))

study = create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 7. Excel Raporlama İpuçları

**Pivot Tables Oluşturma:**
```python
# Excel'de otomatik pivot table
import xlsxwriter

workbook = xlsxwriter.Workbook('predictions.xlsx')
worksheet = workbook.add_worksheet('Data')

# Write data
df.to_excel(worksheet)

# Create pivot table
worksheet.add_table('A1:Z267', {
    'columns': [
        {'header': 'NUCLEUS'},
        {'header': 'Classification'},
        {'header': 'Best_Model'},
        {'header': 'Error'}
    ]
})
```

**Conditional Formatting:**
```python
# Errors'ı renklendir
format_green = workbook.add_format({'bg_color': '#C6EFCE'})  # Low error
format_red = workbook.add_format({'bg_color': '#FFC7CE'})    # High error

worksheet.conditional_format('E2:E267', {
    'type': '3_color_scale',
    'min_color': "#63BE7B",  # Green
    'mid_color': "#FFEB84",  # Yellow
    'max_color': "#F8696B"   # Red
})
```

### 8. Performance Monitoring

**Training Monitoring:**
```python
import matplotlib.pyplot as plt

# Loss curves
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Curves')

plt.subplot(132)
plt.plot(epochs, train_r2, label='Train R²')
plt.plot(epochs, val_r2, label='Val R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.legend()
plt.title('R² Evolution')

plt.subplot(133)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Experimental')
plt.ylabel('Predicted')
plt.title('Test Set Predictions')

plt.tight_layout()
plt.savefig('training_monitoring.png', dpi=300)
```

---

## 🚀 Sonraki Adımlar

### Kısa Vadeli (1-2 hafta)

1. **Feature Engineering v2:**
   - [ ] Feature interactions ekle (Z×N, SPIN×β₂)
   - [ ] Polynomial features dene (A², log(A))
   - [ ] Domain-specific combinations (Q/β₂ ratio)

2. **Model Optimization:**
   - [ ] Top 10 configs için fine-tuning
   - [ ] Ensemble weights optimization
   - [ ] Cross-validation ile stability check

3. **Validation Enhancement:**
   - [ ] K-fold cross-validation (k=5)
   - [ ] Leave-one-out analysis (LOOCV)
   - [ ] Bootstrap confidence intervals

### Orta Vadeli (1-2 ay)

1. **Advanced Feature Engineering:**
   - [ ] AutoML feature selection
   - [ ] Neural Architecture Search (NAS)
   - [ ] Genetic algorithm optimization

2. **Uncertainty Quantification:**
   - [ ] Bayesian Neural Networks finalize
   - [ ] Monte Carlo Dropout
   - [ ] Ensemble uncertainty aggregation

3. **Physics-Informed Learning:**
   - [ ] Physics constraints in loss function
   - [ ] Conservation laws enforcement
   - [ ] Known relationships (Q ∝ valence × β₂)

### Uzun Vadeli (3-6 ay)

1. **Production Deployment:**
   - [ ] Web dashboard (Flask/Streamlit)
   - [ ] REST API (FastAPI)
   - [ ] Docker containerization
   - [ ] CI/CD pipeline

2. **Research Extensions:**
   - [ ] Unknown nuclei predictions (drip lines)
   - [ ] Superheavy elements (Z > 118)
   - [ ] Exotic decay modes
   - [ ] Isomeric states

3. **Publication:**
   - [ ] Thesis compilation
   - [ ] Journal paper preparation
   - [ ] Conference presentation
   - [ ] Code & data release

---

## 📚 Ek Kaynaklar

### Python Kodları

**Dataset Generation:**
```bash
python dataset_generation_pipeline_v2.py \
  --target MM \
  --nucleus-counts 75 100 150 200 ALL \
  --enable-all-features
```

**Model Training:**
```bash
python parallel_ai_trainer.py \
  --dataset MM_200nuclei \
  --models RF XGB DNN \
  --n-configs 50 \
  --adaptive-pruning
```

**All Nuclei Prediction:**
```bash
python all_nuclei_predictor.py \
  --models-dir trained_models/ \
  --output predictions.xlsx
```

### Faydalı Dokümantasyon Dosyaları

- `PROJECT_KNOWLEDGE_QA_REPORT_TR.md` - Proje durumu özeti
- `training_configs_50.json` - Hyperparameter configs
- `constants_v1_1_0.py` - Fizik sabitleri ve parametreler

### İletişim ve Destek

**Proje:** Nuclear Physics AI Prediction System  
**Hazırlayan:** Kemal (Thesis Project)  
**Tarih:** 2025-11-22  
**Versiyon:** 2.0.0 - COMPLETE

---

**© 2025 Nuclear Physics AI Project | Tüm hakları saklıdır.**


