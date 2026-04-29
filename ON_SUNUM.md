# Nükleer Momentlerin Makine Öğrenimi ve ANFIS ile Tahmini
## Ön Sunum Raporu

**Kurum:** Sakarya Üniversitesi  
**Program:** Fizik / Yüksek Lisans  
**Konu:** 267 Çekirdek için Manyetik ve Kuadrupol Moment Tahmini  
**Sistem Adı:** nucdatav1 — PFAZ Pipeline v7.0  

---

## İÇİNDEKİLER

1. Projenin Amacı ve Bilimsel Önemi
2. Veri Seti
3. Sistem Mimarisi: 13 Fazlı Pipeline
4. PFAZ 1 — Veri Seti Üretimi
5. PFAZ 2 — AI Model Eğitimi
6. PFAZ 3 — ANFIS Eğitimi
7. PFAZ 4 — Bilinmeyen Çekirdek Tahminleri
8. PFAZ 5 — Çapraz Model Analizi
9. PFAZ 6 — Nihai Raporlama
10. PFAZ 7 — Ensemble Meta-Öğrenme
11. PFAZ 8 — Görselleştirme
12. PFAZ 9 — Monte Carlo ve Kontrol Grubu Analizi
13. PFAZ 10 — Tez Derleme
14. PFAZ 12 — İleri İstatistiksel Analiz
15. PFAZ 13 — AutoML
16. Metodoloji Özeti
17. Çıktı ve Raporlar
18. Teknik Altyapı

---

## 1. PROJENİN AMACI VE BİLİMSEL ÖNEMİ

### Temel Soru

Atomik çekirdeğin **manyetik momenti (MM)** ve **kuadrupol momenti (QM)** nükleer yapının en hassas göstergelerindendir. Ancak:

- Kararsız ve egzotik çekirdeklerde deneysel ölçüm mümkün değil ya da son derece güç.
- Klasik teorik modeller (Kabuk Modeli, SEMF) ağır ve deformasyonlu çekirdeklerde yetersiz kalıyor.
- Veri tabanında yalnızca 267 çekirdek için güvenilir ölçüm mevcut.

**Bu çalışma:** mevcut 267 çekirdeğin deneysel verilerini kullanarak makine öğrenimi modelleri ve ANFIS sistemi eğitir; ardından bu modelleri deneysel verisi olmayan çekirdeklere uygular.

### Hedef Değişkenler

| Sembol | Ad | Birim | Kapsam |
|--------|----|-------|--------|
| MM | Manyetik Moment | μN (nükleer magneton) | 267 çekirdek |
| QM | Kuadrupol Moment | barn | 219 çekirdek (QM ölçümü olan) |
| Beta_2 | Deformasyon Parametresi | boyutsuz | 267 çekirdek |
| MM_QM | Kombine (çok-çıktılı) | — | 219 çekirdek |

### Projenin Özgün Katkıları

| Katkı | Açıklama |
|-------|---------|
| 848 özgün veri seti | 5 boyutta sistematik kombinasyon fabrikası |
| Gerçek Takagi-Sugeno ANFIS | Hibrit LSE+L-BFGS-B öğrenme, 8 MF konfigürasyonu |
| Gerçek model ensemble | Kayıtlı PKL modellerini yükleyen voting+stacking |
| Monte Carlo belirsizlik analizi | 5 MC yöntemi, %95 GA, 1000+ simülasyon |
| Otomatik tez üretimi | 17 LaTeX bölümü + 4 ek, tüm figürler dahil |
| AutoML yeniden eğitim döngüsü | Düşük R² modelleri otomatik iyileştirme |

---

## 2. VERİ SETİ

### 2.1 Ham Veri: aaa2.txt

```
Dosya     : data/aaa2.txt
Çekirdek  : 267 (1H2'den 83Bi209'a kadar)
Ham sütun : 12
Türetilmiş: 35+
Toplam    : 44+ özellik
```

Ham sütunlar:

| Sütun | Açıklama | Örnek |
|-------|---------|-------|
| NUCLEUS | Çekirdek adı | "26 Fe 56", "82 Pb 208" |
| A | Kütle numarası | 2 — 209 |
| Z | Proton sayısı | 1 — 83 |
| N | Nötron sayısı | 1 — 126 |
| SPIN | Nükleer spin | 0, 0.5, 1, 1.5, 2... |
| PARITY | Parite | +1 / -1 |
| P-factor | P faktörü | 0.5, 0.75, 1.5 |
| Beta_2 | Deneysel deformasyon | -0.3 ... +1.0 |
| MAGNETIC MOMENT [µ] | Hedef: MM (μN) | -3.0 ... +6.0 |
| QUADRUPOLE MOMENT [Q] | Hedef: QM (barn) | -2.0 ... +4.0 |
| Nn | Valans nötron sayısı | 1, 3, 5... |
| Np | Valans proton sayısı | 1, 3, 5... |

### 2.2 Türetilen Teorik Özellikler

**SEMF (Semi-Empirical Mass Formula):**
- Bağlanma enerjisi / nükleon (BE_per_A)
- Toplam bağlanma enerjisi (BE_total)
- Asimetri enerjisi (BE_asymmetry)
- Pairing enerjisi (BE_pairing)

**Woods-Saxon Potansiyeli** (V₀=-51 MeV, r₀=1.25 fm, a=0.65 fm):
- Z/N kabuk boşluğu enerjisi (Z_shell_gap, N_shell_gap)
- Valans sayıları (Z_valence, N_valence)
- Nötron/proton ayrılma enerjisi (Sn, Sp)

**Nilsson Modeli:**
- Hesaplanan deformasyon (Beta_2_estimated)
- Sferiklik indeksi (spherical_index)
- İntrinsik kuadrupol (Q0_intrinsic)

**Kabuk Modeli** (büyü sayıları: 2, 8, 20, 28, 50, 82, 126):
- magic_character (0-1 arası kabuk yapısı skoru)
- Z_magic_dist, N_magic_dist (büyü sayısına uzaklık)

### 2.3 SHAP Tabanlı Özellik Önemi

SHAP analizi her hedef için hangi özelliklerin en bilgilendirici olduğunu gösterir:

| Hedef | İlk 5 Özellik |
|-------|--------------|
| MM | A (%19.2) > Z (%17.5) > SPIN (%12.8) > magic_char (%9.7) > BE/A (%8.3) |
| QM | Z (%21.5) > Beta_2_est (%18.3) > A (%15.7) > magic_char (%10.2) > SPIN (%8.9) |
| Beta_2 | magic_char (%22.1) > Z_mag_dist (%18.7) > N_mag_dist (%17.3) > A (%12.9) > Z_val (%8.4) |

---

## 3. SİSTEM MİMARİSİ: 13 FAZLI PIPELINE

Her faz önceki fazın çıktısını tüketir. Durum `pfaz_status.json`'da izlenir.

```
aaa2.txt (267 çekirdek)
    │
    ▼
PFAZ 1 ──→ 848 veri seti (CSV/XLSX/MAT + metadata.json)
    │
    ▼
PFAZ 2 ──→ ~85.000 AI model (RF/XGB/LGB/CB/SVR/DNN × 50 config)
    │
PFAZ 3 ──→ ~6.800 ANFIS model (8 MF config × veri seti)
    │
    ▼
PFAZ 4 ──→ Bilinmeyen çekirdek tahmini + degradasyon analizi
    │
PFAZ 5 ──→ AI vs ANFIS çapraz model analizi
    │
PFAZ 6 ──→ THESIS_COMPLETE_RESULTS.xlsx (20+ sayfa) + LaTeX
    │
PFAZ 7 ──→ Ensemble (voting + stacking) → iyileştirilmiş R²
    │
PFAZ 8 ──→ 80+ grafik (PNG 300 dpi + HTML/Plotly)
    │
PFAZ 9 ──→ Top-50 model × 267 çekirdek + Monte Carlo CI
    │
PFAZ 10 ──→ Otomatik LaTeX tezi (17 bölüm + 4 ek)
    │
PFAZ 11 ──→ [ATLANIR — üretim dağıtımı, tasarım gereği devre dışı]
    │
PFAZ 12 ──→ İstatistiksel testler + duyarlılık + nükleer örüntü
    │
PFAZ 13 ──→ AutoML yeniden eğitim → automl_improvement_report.xlsx
```

**Çalıştırma:**
```bash
python main.py --run-all           # tüm fazlar sırayla
python main.py --pfaz 3 --mode run # tek faz
python main.py --predict "Z=26 N=30 A=56"  # tek çekirdek
```

---

## 4. PFAZ 1 — VERİ SETİ ÜRETİMİ

**Amaç:** Ham aaa2.txt'i sistematik eğitim/doğrulama/test veri setlerine dönüştürmek.

**Ana Sınıf:** `DatasetGenerationPipelineV2`

### Adımlar

1. **Veri Yükleme** — CSV/XLSX/TXT otomatik format tespiti, sütun standardizasyonu
2. **Teorik Özellik Hesaplama** — TheoreticalCalculationsManager ile 35+ yeni sütun
3. **QM Filtreleme** — QM hedefi için QM=NaN satırları elenir (267→219 çekirdek)
4. **Kalite Kontrolü** — IQR aykırı değer tespiti (eşik=3.0), çekirdek bazlı açıklama
5. **Veri Seti Üretimi** — 5 boyut × 2 senaryo × 4 hedef × 60+ özellik seti
6. **Metadata ve Raporlar** — metadata.json, datasets_summary.xlsx

### İsimlendirme Sistemi

```
{HEDEF}_{BOYUT}_{SENARYO}_{ÖZELLIK_KODU}_{ÖLÇEKLEME}_{ÖRNEKLEME}[_NoAnomaly]

HEDEF   : MM | QM | Beta_2 | MM_QM
BOYUT   : 75 | 100 | 150 | 200 | ALL
SENARYO : S70 (70/15/15) | S80 (80/10/10)
```

Örnek: `MM_150_S70_AZSMC_NoScaling_Random`

### NoAnomaly Varyantları

Yalnızca 150/200/ALL boyutlarında üretilir. IQR eşiği=3.0 ile aykırı değerler çıkarılan veri seti versiyonları.

### Özellik Setleri (60+ kombinasyon, hedef bazlı)

SHAP analizinden seçilmiş, 3–5 girdi sayısı:

| Hedef | Örnek Setler (kısaltma → sütunlar) |
|-------|-----------------------------------|
| MM | AZS = A+Z+SPIN, AZSMC = A+Z+SPIN+magic_char, AZSMCBEPA = +BE/A |
| QM | AZB2EMC = A+Z+Beta_2_est+magic_char, ZB2EMCS = Z+B2E+MC+SPIN |
| Beta_2 | MCZMNM = MC+Z_mag_dist+N_mag_dist, MCZMNMZV = +Z_valence |
| MM_QM | AZSMC, AZB2EMC, AZSMCB2E |

**Neden 3-5 girdi?** 267 çekirdeklik veri seti küçük — daha fazla girdi overfitting riski yaratır. SHAP, ilk 3-5 özelliğin toplam varyansın %70-80'ini açıkladığını gösteriyor. ANFIS için kural sayısı: 3 girdi × 2MF = 8 kural (yönetilebilir).

### Ölçekleme

Ayrık değerler (A, Z, N, SPIN, PARITY, magic_character) **hiçbir zaman ölçeklenmez.** Ölçekleme yalnızca train split üzerinde fit edilir.

### Çıktı

```
outputs/generated_datasets/
└── MM_150_S70_AZSMC_NoScaling_Random/
    ├── train.csv       (başlıksız, sayısal)
    ├── val.csv
    ├── test.csv
    ├── train.xlsx      (NUCLEUS+özellikler+hedef)
    ├── val.xlsx / test.xlsx
    └── metadata.json   (feature_names, split_ratios, scaling...)
```

**Toplam: 848 veri seti**

| Hedef | Standart | NoAnomaly | Toplam |
|-------|----------|-----------|--------|
| MM | 140 | 84 | 224 |
| QM | 150 | 90 | 240 |
| Beta_2 | 130 | 78 | 208 |
| MM_QM | 110 | 66 | 176 |
| **TOPLAM** | **530** | **318** | **848** |

---

## 5. PFAZ 2 — AI MODEL EĞİTİMİ

**Amaç:** 848 veri seti üzerinde 6 model türünü, 50 konfigürasyonla paralel eğitmek.

**Ana Sınıf:** `ParallelAITrainer`

### Model Türleri

| Model | Kütüphane | Özellikler |
|-------|-----------|-----------|
| Random Forest (RF) | scikit-learn | Karar ağacı topluluk, küçük dataset'te n_estimators adaptif |
| XGBoost (XGB) | xgboost | Gradient boosting, early stopping |
| LightGBM (LGB) | lightgbm | Hızlı gradient boosting, MultiOutputRegressor |
| CatBoost (CB) | catboost | Kategorik özellik desteği |
| SVR | scikit-learn | Kernel SVM, dahili StandardScaler, rbf kernel |
| DNN | TensorFlow/Keras | BatchNorm+Dropout+L2, Huber loss, EarlyStopping |

**DNN Detayları:**
- Katmanlar: [128, 64, 32], Dropout: [0.2, 0.2, 0.1]
- Optimizer: Adam (clipnorm=1.0), Loss: Huber
- Hem özellikler hem hedef StandardScaler ile ölçeklenir
- `DNN_MIN_SAMPLES = 80` — daha az veri varsa atlanır

### 50 Konfigürasyon

RF_001–RF_020, XGB_021–XGB_035, DNN_036–DNN_050 — her biri farklı hiperparametre kombinasyonu.

### Paralel Çalışma

`ThreadPoolExecutor` ile n_workers=os.cpu_count() varsayılan. Her job:
- metadata.json'dan feature_names oku
- train.csv / val.csv / test.csv yükle
- Model eğit → R², RMSE, MAE hesapla → .pkl kaydet

### 5-Katlı Çapraz Doğrulama

Opsiyonel (`use_model_validation=True`). `CrossValidationAnalyzer` ile her model için 5 katlı CV, stabilitesi ölçülür.

### Çıktı

```
outputs/trained_models/
└── {dataset}/
    └── {model_turu}/
        └── {config_id}/
            ├── model_{model}_{config}.pkl
            ├── metrics_{config}.json      (R2, RMSE, MAE — train/val/test)
            └── cv_results_{config}.json
```

---

## 6. PFAZ 3 — ANFIS EĞİTİMİ

**Amaç:** Gerçek Takagi-Sugeno ANFIS'i 8 üyelik fonksiyonu konfigürasyonunda eğitmek.

**Ana Sınıf:** `ANFISParallelTrainerV2`  
**ANFIS Sınıfı:** `TakagiSugenoANFIS` (MATLAB gerekmez)

### ANFIS Nedir?

ANFIS (Adaptive Neuro-Fuzzy Inference System) — bulanık mantık kurallarını sinir ağı optimizasyonuyla birleştirir:

```
x₁, x₂, x₃ → Fuzzy-leştirme (MF)
             → Kural Ağırlıkları (ürün/min t-normu)
             → Normalizasyon
             → Lineer Sonuç (p_r·[1, x₁, x₂, x₃])
             → Ağırlıklı toplam → çıkış ŷ
```

### Hibrit Öğrenme

| Geçiş | Yöntem | Parametreler |
|-------|--------|-------------|
| İleri | LSE (Ridge regresyon) | Sonuç parametreleri p_r |
| Geri | L-BFGS-B | MF merkezleri ve genişlikleri |

EarlyStopping: val loss bazlı, patience=30. Dahili StandardScaler normalizasyonu.

### 8 Konfigürasyon

| Konfigürasyon | Yöntem | MF Tipi | n_mfs | Kural (3 girdi) |
|---|---|---|---|---|
| CFG_Grid_2MF_Gauss | Grid | Gaussian | 2 | 8 |
| CFG_Grid_2MF_Bell | Grid | Gen. Bell | 2 | 8 |
| CFG_Grid_2MF_Tri | Grid | Üçgen | 2 | 8 |
| CFG_Grid_2MF_Trap | Grid | Trapezoid | 2 | 8 |
| CFG_Grid_3MF_Gauss | Grid | Gaussian | 3 | 27 |
| CFG_Grid_3MF_Bell | Grid | Gen. Bell | 3 | 27 |
| CFG_SubClust_5 | SubClust (KMeans) | Gaussian | 5 küme | 5 |
| CFG_SubClust_8 | SubClust (KMeans) | Gaussian | 8 küme | 8 |

### Kural Patlaması Koruması

`_adaptive_n_mfs()`: kural sayısı `n_kural < n_eğitim / 3` sağlanana kadar n_mfs düşürülür.

| Girdi | n_eğitim | n_mfs | Kural |
|-------|---------|-------|-------|
| 3 | ~52 | 2 | 8 ✓ |
| 4 | ~52 | 2 | 16 ✓ |
| 5 | ~52 | 2 | 32 → düşürülür |

### Eğitim Sonrası Anomali Temizleme

`OutlierDetector`: IQR (multiplier=2.5) + Z-score (threshold=2.5) artık analizi.
Aykırı örnekler çıkarılır → model yeniden eğitilir → Val R² karşılaştırması → daha iyi model seçilir.

### Çıktı

```
outputs/anfis_models/
└── {dataset}/
    └── {config_id}/
        ├── model_{config}.pkl
        └── metrics_{config}.json   (R2, RMSE, MAE, n_rules, mf_type, n_outliers)
```

---

## 7. PFAZ 4 — BİLİNMEYEN ÇEKİRDEK TAHMİNLERİ

**Amaç:** Test split'teki (eğitimde görülmeyen) çekirdeklere tahmin yapmak ve genelleme kalitesini ölçmek.

**Ana Sınıf:** `UnknownNucleiPredictor`

### Çalışma Mantığı

Her veri setinin test.csv'si "bilinmeyen çekirdek" kümesidir (eğitimde hiç görülmedi). Tüm AI ve ANFIS modelleri bu kümeye uygulanır:

```
Degradasyon = Val_R² - Test_R²
```

Degradasyon düşükse model gerçekten genelleşiyor; yüksekse overfit.

### Excel Çıktısı (8 sayfa)

| Sayfa | İçerik |
|-------|--------|
| All_Results | Tüm model sonuçları (Dataset, Target, Model_Type, Train/Val/Test R²) |
| Best_Per_Dataset | Dataset × Target × Kategori başına en iyi |
| Degradation_Analysis | Val-Test farkı sıralaması |
| AI_vs_ANFIS | Yan yana + kazanan sütunu |
| Pivot_By_Target | Hedef bazlı ortalama Test R² |
| Per_Nucleus_Predictions | Bireysel çekirdek hataları |

---

## 8. PFAZ 5 — ÇAPRAZ MODEL ANALİZİ

**Amaç:** AI ve ANFIS modellerinin aynı çekirdekler üzerindeki tahminlerini karşılaştırmak.

**Ana Sınıf:** `CrossModelAnalysisPipeline`

### Analiz

Tüm modellerin test.csv tahminleri ortak çekirdek ID'leri üzerinde iç birleştirme (inner merge) ile hizalanır.

**Çekirdek Sınıflandırması:**

| Sınıf | Kriter |
|-------|--------|
| Good | R² > 0.90 |
| Medium | 0.70 ≤ R² ≤ 0.90 |
| Poor | R² < 0.70 |

### Çıktı

```
outputs/cross_model_analysis/
├── MM/MM_cross_model_report.xlsx
├── QM/QM_cross_model_report.xlsx
├── Beta_2/Beta_2_cross_model_report.xlsx
├── MASTER_CROSS_MODEL_REPORT.xlsx    (8 sayfa: Good/Med/Poor × target)
└── cross_model_analysis_summary.json
```

---

## 9. PFAZ 6 — NİHAİ RAPORLAMA

**Amaç:** PFAZ 2–5 JSON metriklerini toplayıp 20+ sayfalık yayın kalitesinde Excel ve LaTeX üretmek.

**Ana Sınıf:** `FinalReportingPipeline`

### Veri Toplama Kaynakları

- PFAZ 2: `metrics_{config}.json` — AI modeller
- PFAZ 3: `metrics_{config}.json` — ANFIS (+ mf_type, n_rules)
- PFAZ 4: `Unknown_Nuclei_Results.xlsx`
- PFAZ 5: `cross_model_analysis_summary.json`
- PFAZ 9 (varsa): MC robustluk verileri

### Excel: THESIS_COMPLETE_RESULTS.xlsx

20+ sayfa, büyük tablolar (>50.000 satır) chunked yazılır:

| Sayfa | İçerik |
|-------|--------|
| Overview | Dashboard özet |
| All_AI_Models | Tüm AI model sonuçları |
| RF_Models / XGB_Models / ... | Model türü bazlı |
| AI_vs_ANFIS_Comparison | Karşılaştırma, kazanan sütunu |
| Best_Models_Per_Target | Her hedef için en iyi |
| Anomaly_vs_NoAnomaly | Anomali temizlemenin etkisi |
| Robustness_CV | Çapraz doğrulama |
| Feature_Abbreviations | Özellik kısaltma tablosu |

Renk kodlaması: Excellent (R²≥0.95) / Good (≥0.90) / Medium (≥0.70) / Poor (<0.70)

### LaTeX Raporu

`LaTeXReportGenerator` — 8 bölüm: giriş, metodoloji, sonuçlar, tartışma, sonuç + appendix'ler. pdflatex varsa otomatik derleme.

---

## 10. PFAZ 7 — ENSEMBLE META-ÖĞRENME

**Amaç:** Kayıtlı AI ve ANFIS modellerini birleştirerek tek modelin ötesine geçmek.

**Ana Fonksiyon:** `pfaz7_complete_pipeline`

### Gerçek Model Entegrasyonu

`RealModelLoader` — trained_models/ ve anfis_models/ dizinlerini tarar, her modelin metadata.json'ından feature_names okur, val_R²'ye göre top-N seçer.

`RealPredictionCollector` — her model kendi özellik setiyle test/val.csv'yi yükler, NUCLEUS ID'leri üzerinde inner merge ile hizalar. Sonuç: (n_çekirdek × n_model) tahmin matrisi.

### Voting Yöntemleri

| Yöntem | Ağırlık |
|--------|---------|
| Simple Voting | Eşit |
| Weighted R² | val_R² oranında |
| Weighted Inv-RMSE | 1/RMSE oranında |
| Rank-based | Sıralama dönüşümü sonrası |
| Dynamic Weight | 10 iterasyonla optimize |

### Stacking (2 Seviye)

```
Seviye 0: RF + XGB + GBM + DNN + BNN + ANFIS  (base models)
    ↓ Out-of-fold (5-fold CV) tahminler
Seviye 1: Ridge / Lasso / ElasticNet / RF_meta / GBM_meta / MLP  (meta-model)
```

Meta-model val tahminleri üzerinde eğitilir, test üzerinde değerlendirilir.

### Çıktı

```
outputs/ensemble_results/
├── ensemble_report_{timestamp}.xlsx   (per-target sayfalar + tahminler)
└── ensemble_summary.json              (best_method, best_r2, improvement)
```

---

## 11. PFAZ 8 — GÖRSELLEŞTİRME

**Amaç:** 80+ bilimsel grafik ve interaktif görsel üretmek.

**Ana Sınıf:** `MasterVisualizationSystem`

### Grafik Türleri

| Kategori | Adet | Araç |
|----------|------|------|
| Tahmin vs Gerçek (scatter) | ~20 | matplotlib |
| Korelasyon/Hata ısı haritası | ~15 | seaborn |
| 3D yüzey (Z-N-MM) | ~10 | matplotlib 3D |
| Hata dağılım grafikleri | ~15 | matplotlib/KDE |
| Model karşılaştırma | ~12 | matplotlib |
| SHAP grafikleri (beeswarm, force) | ~5 | shap |
| İnteraktif HTML | ~8 | Plotly |

Tüm PNG: 300 dpi. Tez uyumlu font ve kenar boşlukları.

### İki Geçişli Sistem

**Geçiş 1 — Standart:** PFAZ 6 çıktılarından temel grafikler  
**Geçiş 2 — Tamamlayıcı:** PFAZ 9 MC grafikleri, PFAZ 12 istatistik, PFAZ 13 AutoML grafikleri eklenir

### Öne Çıkan Görselleştirmeler

- **Nükleer harita üzerinde hata ısı haritası** — Z × N grid, hangi çekirdek bölgelerinde hata yüksek?
- **İzotop zinciri grafikleri** — Z=50 (Sn), Z=82 (Pb) izotopları boyunca tahmin vs deneysel
- **SHAP beeswarm** — her özelliğin MM/QM üzerindeki etki yönü ve büyüklüğü
- **Monte Carlo CI bantları** — 267 çekirdek için %95 güven aralığı

---

## 12. PFAZ 9 — MONTE CARLO VE KONTROL GRUBU ANALİZİ

**Amaç:** Top-50 modeli tüm 267 çekirdeğe uygulamak; Monte Carlo ile tahmin belirsizliğini ölçmek.

**Ana Sınıf:** `AAA2ControlGroupAnalyzerComplete`

### 4 Aşamalı Pipeline

**Aşama 1: Zenginleştirme**  
aaa2.txt yüklenir + Woods-Saxon, Nilsson, Kabuk modeli teorik özellikleri hesaplanır (14 yeni özellik).

**Aşama 2: Top-50 Model Seçimi**  
trained_models/ ve anfis_models/ taranır, test_R²'ye göre en iyi 50 model seçilir.

**Aşama 3: Tüm Çekirdeklere Tahmin**  
Her model kendi metadata.json'ından feature_names okur, 267 çekirdeğe tahmin yapar.  
Sonuç: (50 model × 267 çekirdek) tahmin matrisi.

**Aşama 4: Belirsizlik Ölçümü**  
`MonteCarloUncertaintyQuantifier`:
- Her çekirdek için: ortalama, std, %95 CI, entropi, varyasyon katsayısı
- Yüksek belirsizlik: std > 0.3
- Düşük belirsizlik: std < 0.05

### 5 Monte Carlo Yöntemi

| Yöntem | Açıklama | n |
|--------|---------|---|
| MC Dropout | DNN için 100 ileri geçiş (training=True) | 100 |
| Bootstrap | Stratified resample | 100 |
| Noise Sensitivity | %1–20 gürültü seviyesi | 5 seviye × 100 |
| Feature Dropout | %10/%20/%30 özellik silme | 500 |
| Ensemble Uncertainty | Modeller arası std ve korelasyon | — |

### 8 Pivot Tablo (Excel'de)

Model Performansı / Çekirdek Kategorisi / Kütle Bölgesi / Magic Sayı Etkisi / Kabuk Kapanması / QM Boş Çekirdekler / Çekirdek Başına En İyi / En Kötü Performans

### Çıktı

```
outputs/aaa2_pfaz9_complete_results/
├── aaa2_enriched_with_theory.csv
├── AAA2_Complete_MM.xlsx         (15 sayfa)
├── AAA2_Complete_QM.xlsx
├── AAA2_Complete_Beta_2.xlsx
└── monte_carlo_analysis/
    ├── visualizations/{TARGET}/  (chart_01 – chart_13.png)
    └── excel_reports/MC_Analysis_{TARGET}_{timestamp}.xlsx
```

---

## 13. PFAZ 10 — TEZ DERLEMESİ

**Amaç:** Tüm PFAZ çıktılarından otomatik LaTeX tezi oluşturmak.

**Ana Sınıf:** `MasterThesisIntegration` (`execute_full_pipeline`)

### 8 Adımlı Pipeline

1. Tüm PFAZ çıktılarını tara (figürler, Excel, JSON)
2. 17 LaTeX bölümü üret (.tex dosyaları)
3. PFAZ 8 PNG'lerini figures/ dizinine kopyala
4. Excel tablolarını LaTeX booktabs formatına dönüştür
5. BibTeX kaynakça oluştur
6. main.tex ana belgesini birleştir (\include{chapters/*})
7. Kalite kontrolü (eksik figür/bölüm tespiti)
8. pdflatex ile derleme (compile_pdf=True ise)

### Tez Bölümleri

| Bölüm | İçerik |
|-------|--------|
| Kısaltmalar | 40+ kısaltma tablosu |
| Semboller | Nükleer fizik + ML + ANFIS sembolleri |
| Özet / Abstract | TR + EN |
| Nükleer Teori | Kabuk modeli, SEMF, β₂, büyü sayıları |
| Giriş | Motivasyon, 11 hedef, bölüm planı |
| Metodoloji | 13 PFAZ, metrikler, CV, IQR, MC |
| Veri Seti | PFAZ1 — aaa2.txt, 848 veri seti |
| AI Eğitimi | PFAZ2 — 6 model türü, 50 konfigürasyon |
| ANFIS | PFAZ3 — 8 MF konfigürasyonu, hibrit öğrenme |
| Sonuçlar | PFAZ2/3 toplu karşılaştırma |
| Bilinmeyen Tahminler | PFAZ4 — genelleme analizi |
| Çapraz Model | PFAZ5 — AI vs ANFIS |
| Ensemble | PFAZ7 — voting + stacking |
| İstatistiksel Analizler | PFAZ12 — Friedman, Wilcoxon, Sobol |
| AutoML | PFAZ13 — Optuna iyileştirme |
| Tartışma | Yorum ve karşılaştırma |
| Sonuç | Özet + gelecek çalışmalar |
| Ekler (4) | Hiperparametreler, veri seti detayları, özellikler, Excel referansları |

---

## 14. PFAZ 12 — İLERİ İSTATİSTİKSEL ANALİZ

**Amaç:** Model farklılıklarını istatistiksel olarak kanıtlamak; fiziksel örüntüleri tespit etmek.

### İstatistiksel Testler (`StatisticalTestingSuite`)

| Test | Kullanım | Etki Büyüklüğü |
|------|---------|----------------|
| Eşleştirilmiş t-testi | 2 model karşılaştırma | Cohen's d |
| Wilcoxon | Parametrik olmayan 2 model | Cliff's delta |
| Tek yönlü ANOVA | 3+ model | Eta-squared |
| Friedman | Parametrik olmayan 3+ model | — |
| Tukey HSD | ANOVA sonrası pairwise | — |
| Pairwise Wilcoxon | Bonferroni/Holm düzeltmeli | — |

### Duyarlılık Analizi (`AdvancedSensitivityAnalysis`)

- **Sobol Endeksleri** — birinci ve toplam dereceden varyans katkısı (S1, ST)
- **Morris Tarama** — hangi özellikler en etkili? (μ, μ*, σ)
- **Tornado Diyagramı** — ±%10 tek-seferde girdi değişimi

### Nükleer Örüntü Analizi (`NuclearPatternAnalyzer`)

İzotop/izotone/izobar zincirlerinde sıçrama tespiti, magic sayı etkisi, kabuk kapanması analizi. Türkçe sayfa başlıklı çok-sayfalı Excel çıktısı.

---

### Moment Değer Bandı ve Çapraz Örüntü Analizi (`NuclearMomentBandAnalyzer`) — YENİ

**Temel soru:** Farklı Z, N, A değerlerine sahip çekirdekler neden aynı MM/QM/Beta_2 değerinde buluşuyor? Komşu çekirdekte bu değer neden aniden değişiyor?

#### A) Değer Bandı Analizi

- MM/QM/Beta_2 değer ekseni **6 eşit olasılıklı bant**a bölünür (yüzdelik dilimlere göre)
- Her bant için: kaç çekirdek, ortalama A/Z/N, tek/çift çekirdek oranı
- **Ayırt edici özellikler**: Her bandın z-skoru en yüksek 5 özelliği → *Bu bantı diğerlerinden ayıran fiziksel özellik nedir?*
- **Örnek:** MM ∈ [3.5, 4.5] bandında → `SPIN`, `Z_valence`, `N_magic_dist` z-skoru yüksek

#### B) Komşu Çekirdek Sıçrama Analizi

Zincir türleri:
| Zincir | Sabit | Değişen | Fiziksel anlam |
|--------|-------|---------|----------------|
| İzotop | Z | N | Aynı element, farklı nötron → ani MM değişimi |
| İzoton | N | Z | Aynı nötron sayısı, farklı proton → ani QM değişimi |
| İzobar | A | Z | Aynı kütle sayısı → Beta_2 sıçraması |

- Eşik: zincir içi ΔTarget'ın `2σ` üzerindeki geçişler sıçrama sayılır
- Her sıçrama için: **hangi özellikler de değişti?** (`BE_pairing`, `SPIN`, `Z_valence`, vs.)
- **Not:** Veri setinde sihirli çekirdek (magic nucleus) yoktur; tüm sıçramalar sub-shell veya yapısal geçişten kaynaklanır

#### C) Çapraz Kütle Bölgesi Karşılaştırması

Aynı MM bandında hafif (A<50) ve ağır (A≥100) çekirdekler birlikte bulunabilir:

- **Ortak özellikler** (bant içi tüm bölgelerde düşük yayılma): benzer valans nökleon sayısı, benzer deformasyon indisi
- **Farklı özellikler** (bölgeden bölgeye yüksek yayılma): `BE_per_A`, `Z_N_ratio`, mutlak `Z_magic_dist`
- **Fiziksel yorum:** Farklı kütle bölgelerinden çekirdekler aynı moment bandında → muhtemel neden: *eşdeğer kabuk doluluk oranı* veya *benzer deformasyon*

#### D) Korelasyon Sıralaması

Spearman ve Pearson korelasyonuyla MM/QM/Beta_2 ile en güçlü ilişkili özellikler sıralanır:

| Sıra | Özellik | Spearman r | Yorum |
|------|---------|-----------|-------|
| 1 | SPIN | 0.72 | Güçlü korelasyon |
| 2 | Z_valence | 0.61 | Orta korelasyon |
| ... | ... | ... | ... |

#### Excel Çıktısı (`nuclear_band_analysis.xlsx`)

| Sayfa | İçerik |
|-------|--------|
| `Bant_Ozeti` | Her bant için n, ort, std, ayırt edici özellikler, çekirdek listesi |
| `Sicrama_Analizi` | Sıçrayan çekirdek çiftleri, Δdeğer, değişen özellikler |
| `Capraz_Kutle` | Hafif/orta/ağır bölgede ortak ve farklı özellikler |
| `Korelasyon` | MM/QM/Beta_2 ile tüm özelliklerin Spearman sıralaması |
| `Cekirdek_Detay` | Her çekirdek: bant ataması, z-skor, SPIN, valans, magic mesafesi |
| `Tahmin_Dogrulugu` | **YENİ** — Her çekirdek için PFAZ4 tahmin hataları, sıçrama/normal sınıfı, en iyi model |
| `Tahmin_Ozeti` | **YENİ** — Sıçrama vs normal ve bant bazında ortalama/max/min mutlak hata karşılaştırması |
| `Aciklama` | Otomatik oluşturulan fiziksel yorum özeti |

#### Tahmin Doğruluğu Analizi — Modeller Sıçramayı Görebiliyor Mu?

`_prediction_accuracy_analysis()` metodu PFAZ4 çıktısından (`AAA2_Original_vs_Predictions.xlsx`)
sıçrama çekirdeklerinin tahmin hatalarını ayrıştırır:

- **`Is_Jump_Nucleus = 1`**: İzotop/izoton/izobar zincirinde ani değişim yapan çekirdek
- **`Sinif`**: "Sicrama Nukleusu" | "Normal Nukleus"
- **`En_Iyi_Abs_Hata`**: En iyi model tipinin bu çekirdek için mutlak hatası
- **`Ort_Abs_Hata`**: Tüm model tiplerinin ortalaması

**Beklenen bulgu:** Sıçrama çekirdeklerinde ortalama mutlak hata genellikle yüksektir —
modeller doğrusal olmayan geçişleri tam olarak yakalayamaz.
Bu durum tezde "neden sıçrama bölgelerine özel model veya PINN gerekebilir" argümanını destekler.

**Grafik:** `band_analysis/jump_prediction_accuracy.png`  
Hedef başına sıçrama vs normal grubun ortalama hatasını karşılaştıran bar grafik (±std errorbars).

---

## 15. PFAZ 13 — AutoML

**Amaç:** Düşük R² modelleri otomatik tespit edip Optuna ile iyileştirmek — hem AI hem ANFIS için ayrı optimizasyon döngüleri içerir.

**Ana Sınıf:** `AutoMLRetrainingLoop`

### A) AI Model Optimizasyonu (Ana Döngü)

```
1. PFAZ 2 metrics_*.json tara → val_R² < 0.80 olanları bul
2. Her aday için AutoMLOptimizer çalıştır (30 Optuna denemesi)
3. before_r2 vs after_r2 karşılaştır
4. 3 sayfalık improvement Excel + JSON log üret
```

**Optimizer: Optuna TPE**

- Sampler: TPE (Tree-structured Parzen Estimator)
- Pruner: MedianPruner (verimsiz deneyleri erken durdur)
- 7 model türü: RF, XGB, GBM, LGB, CatBoost, SVR, DNN
- R² < -2.0 → ceza puanı (diverjans koruması)

**Arama Uzayı (Örnek: XGBoost):**  
n_estimators (50–500), max_depth (3–10), learning_rate (0.01–0.3), subsample (0.6–1.0), colsample_bytree, reg_alpha, reg_lambda

### B) ANFIS Optimizasyonu (`AutoMLANFISOptimizer`)

AI döngüsünün ardından en düşük başlangıç R²'ye sahip **1 dataset** için ANFIS hiperparametre araması yapılır.

```
Kapsam   : Yalnızca en kötü performanslı 1 dataset
Deneme   : max 20 Optuna denemesi (TPE)
Süre Sınırı: 600 saniye
Arama Uzayı:
  - mf_type    : gauss | bell | tri | trap
  - n_mfs      : 2 – 4
  - n_epochs   : 50 – 300
  - learning_rate_lbfgs : 0.001 – 0.5
  - lse_lambda : 0.0001 – 1.0
```

> **Not:** ANFIS optimizasyonu AI optimizasyonu kadar kapsamlı değildir. Tüm 6.800 ANFIS modelini yeniden optimize etmek; PFAZ 3'ü farklı konfigürasyonlarla (`--pfaz 3 --mode run`) yeniden çalıştırmak ile mümkündür.

### ANFIS Eğitim Stratejisi (PFAZ 3 Özeti)

| Parametre | Değer |
|-----------|-------|
| MF tipleri | Gauss, Gen.Bell, Üçgen, Trapezoid |
| Grid yöntemi (n_mfs=2) | 8 kural (2³) |
| Grid yöntemi (n_mfs=3) | 27 kural (3³) |
| SubClust (KMeans) | 5 veya 8 küme |
| Öğrenme | LSE (ileri) + L-BFGS-B (geri) — hibrit |
| EarlyStopping | patience=30, val loss bazlı |
| Kural patlaması koruması | n_kural < n_eğitim/3 sağlanana dek n_mfs düşürülür |
| Anomali temizleme | IQR+Z-score → yeniden eğitim → en iyi model seçilir |
| Toplam config | 8 (4 Grid + 2 SubClust) |
| Toplam model | ~6.800 (848 dataset × 8 config) |

### Çıktı

```
outputs/automl_results/
├── automl_improvement_report.xlsx       (Summary / Best_Params / Overview)
├── automl_retraining_log.json
├── automl_summary.json
└── anfis_optimization/
    ├── anfis_optimization_report.xlsx   (ANFIS AutoML sonuçları)
    └── best_anfis_config.json
```

---

## 16. METODOLOJİ ÖZETİ

### Genel Yaklaşım

```
Deneysel Veri → Özellik Mühendisliği → Veri Seti Fabrikası
  → Paralel Model Eğitimi (AI + ANFIS)
  → Ensemble Meta-Öğrenme
  → Belirsizlik Analizi (Monte Carlo)
  → İstatistiksel Doğrulama
  → Otomatik Raporlama
```

### Değerlendirme Metrikleri

| Metrik | Formül | İyi Eşik |
|--------|--------|---------|
| R² | 1 − SS_res/SS_tot | ≥ 0.90 (iyi), ≥ 0.95 (çok iyi) |
| RMSE | √(Σ(y−ŷ)²/n) | Hedef bağımlı |
| MAE | Σ|y−ŷ|/n | Hedef bağımlı |

### Overfitting Kontrolü

- 5-katlı çapraz doğrulama tüm AI modellerinde
- Train-Val-Test ayrı split (veri sızıntısı yok)
- DNN_MIN_SAMPLES = 80 (küçük veri koruması)
- Anomali tespiti ile veri kalitesi güvencesi

### Belirsizlik Nicelendirme

Her tahmin için %95 güven aralığı, 5 farklı Monte Carlo yöntemiyle.

---

## 17. ÇIKTI VE RAPORLAR

### Üretilen Çıktı Özeti

| Tür | Adet | Yer |
|-----|------|-----|
| Eğitim veri seti (CSV+XLSX+MAT) | 848 × 3 | outputs/generated_datasets/ |
| AI eğitilmiş model (.pkl) | ~85.000 | outputs/trained_models/ |
| ANFIS eğitilmiş model (.pkl) | ~6.800 | outputs/anfis_models/ |
| 20+ sayfa Excel raporları | 5+ dosya | outputs/reports/ |
| Görselleştirme (PNG 300dpi) | 80+ | outputs/visualizations/ |
| İnteraktif HTML grafikler | 8+ | outputs/visualizations/ |
| LaTeX tez dosyaları | 17 bölüm | outputs/thesis/ |
| Monte Carlo Excel/JSON | 3 hedef × 2 | outputs/aaa2_pfaz9_complete_results/ |

### Ana Rapor Dosyaları

| Dosya | Faz | İçerik |
|-------|-----|--------|
| THESIS_COMPLETE_RESULTS.xlsx | PFAZ 6 | 20+ sayfa, tüm model metrikleri |
| MASTER_CROSS_MODEL_REPORT.xlsx | PFAZ 5 | AI vs ANFIS çapraz analizi |
| Unknown_Nuclei_Results.xlsx | PFAZ 4 | Genelleme ve degradasyon |
| ensemble_report_{ts}.xlsx | PFAZ 7 | Voting+Stacking karşılaştırma |
| AAA2_Complete_MM/QM/Beta_2.xlsx | PFAZ 9 | 267 çekirdek tam tahmin |
| pfaz12_statistical_tests.xlsx | PFAZ 12 | İstatistiksel test sonuçları |
| automl_improvement_report.xlsx | PFAZ 13 | AutoML iyileştirme özeti |
| thesis.pdf | PFAZ 10 | Tamamlanmış tez (pdflatex ile) |

---

## 18. TEKNİK ALTYAPI

### Sistem Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|---------|
| Python | 3.8+ | 3.10+ |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB | 100 GB |
| GPU (opsiyonel) | CUDA 11.0+, 8 GB | CUDA 12.0+, 16 GB |
| MATLAB (opsiyonel) | R2020a+ | R2023a+ |

### Ana Python Kütüphaneleri

| Kütüphane | Kullanım |
|-----------|---------|
| scikit-learn | RF, GBM, CVA, ölçekleme |
| xgboost | XGBoost |
| lightgbm | LightGBM |
| catboost | CatBoost |
| TensorFlow/Keras | DNN, BNN, PINN |
| optuna | AutoML (TPE + MedianPruner) |
| shap | Özellik önemi yorumlanabilirliği |
| scipy | İstatistiksel testler, L-BFGS-B |
| pandas, numpy | Veri işleme |
| matplotlib, seaborn | Statik görselleştirme (300 dpi) |
| plotly | İnteraktif HTML grafikleri |
| openpyxl, xlsxwriter | Excel rapor üretimi |

### Pipeline Komutları

```bash
pip install -r requirements.txt

# Tüm fazları sırayla çalıştır
python main.py --run-all

# Tek faz
python main.py --pfaz 2 --mode run
python main.py --pfaz 2 --mode resume   # kesintiden devam
python main.py --pfaz 2 --mode pass     # önbellek kullan, atla

# Tek çekirdek tahmini (CLI)
python main.py --predict "Z=26 N=30 A=56"

# İnteraktif mod
python main.py --interactive
```

### Log ve İzleme

- `logs/main_*.log` — 200 MB × 5 döngüsel log dosyası
- `pfaz_status.json` — faz durumu (pending/running/completed/failed/skipped)
- `outputs/pipeline_warnings.json` — gerçek zamanlı uyarı biriktiricisi
- `outputs/pipeline_warnings_report.xlsx` — uyarı raporu

---

## EKLER

### Ek A: PFAZ Tamamlanma Durumu

| Faz | Modül Klasörü | Durum |
|-----|--------------|-------|
| PFAZ 0 | core_modules/ | Tamamlandı |
| PFAZ 1 | pfaz01_dataset_generation/ | Tamamlandı (848 veri seti) |
| PFAZ 2 | pfaz02_ai_training/ | Tamamlandı |
| PFAZ 3 | pfaz03_anfis_training/ | Tamamlandı |
| PFAZ 4 | pfaz04_unknown_predictions/ | Tamamlandı |
| PFAZ 5 | pfaz05_cross_model/ | Tamamlandı |
| PFAZ 6 | pfaz06_final_reporting/ | Tamamlandı |
| PFAZ 7 | pfaz07_ensemble/ | Tamamlandı (gerçek model yükleme) |
| PFAZ 8 | pfaz08_visualization/ | Tamamlandı |
| PFAZ 9 | pfaz09_aaa2_monte_carlo/ | Tamamlandı |
| PFAZ 10 | pfaz10_thesis_compilation/ | Tamamlandı |
| PFAZ 11 | pfaz11_production/ | ATLANIR (tasarım gereği) |
| PFAZ 12 | pfaz12_advanced_analytics/ | Tamamlandı |
| PFAZ 13 | pfaz13_automl/ | Tamamlandı |

### Ek B: Özellik Kısaltma Tablosu

| Kısaltma | Sütun | Kaynak |
|---|---|---|
| A | A | Ham veri |
| Z | Z | Ham veri |
| N | N | Ham veri |
| S | SPIN | Ham veri |
| PAR | PARITY | Ham veri |
| PF | P_FACTOR | Ham veri |
| MC | magic_character | Kabuk modeli |
| BEPA | BE_per_A | SEMF |
| B2E | Beta_2_estimated | Nilsson |
| ZMD | Z_magic_dist | Kabuk modeli |
| NMD | N_magic_dist | Kabuk modeli |
| BEA | BE_asymmetry | SEMF |
| ZV | Z_valence | Woods-Saxon |
| NV | N_valence | Woods-Saxon |
| ZSG | Z_shell_gap | Woods-Saxon |
| NSG | N_shell_gap | Woods-Saxon |
| BEP | BE_pairing | SEMF |
| SPHI | spherical_index | Nilsson |
| CP | Q0_intrinsic | Nilsson |
| BET | BE_total | SEMF |
| SN | S_n_approx | SEMF |
| SP | S_p_approx | SEMF |
| NN | Nn | Ham veri |
| NP | Np | Ham veri |

### Ek C: Veri Seti Üretim Sayıları

```
Toplam: 848 veri seti
  - 424 adet S70 (70/15/15 split)
  - 424 adet S80 (80/10/10 split)
  - 318 adet NoAnomaly varyantı (150/200/ALL boyutlarında)
  - Her klasörde: train/val/test .csv + .xlsx + .mat + metadata.json
```

---

*Son güncelleme: Nisan 2026 — nucdatav1 v7.0 kodundan türetilmiştir.*
