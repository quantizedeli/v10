# Reports Index

Bu dizin pipeline'in tüm raporlarini içerir.
Raporlar PFAZ (Pipeline Fazı) modüllerine göre organize edilmiştir.

---

## Dizin Yapısı

```
reports/
├── REPORTS_INDEX.md                      ← Bu dosya
├── theoretical_calculations/             ← PFAZ1: Teorik özellik hesaplama raporları
│   └── theoretical_calc_report_<ts>.json
├── generation_report.json                ← PFAZ1: Dataset üretim özeti (generated_datasets/ altında)
├── anomaly_explanation_report.json       ← PFAZ1: Anomali çekirdek açıklama raporu
├── quality_reports/                      ← PFAZ1: Veri kalite kontrol raporları
│   ├── outlier_report_<target>.json
│   └── validation_report_<target>.json
├── ai_training/                          ← PFAZ2: AI model eğitim metrikleri
│   └── training_summary.xlsx
├── anfis_training/                       ← PFAZ3: ANFIS eğitim metrikleri
│   └── anfis_summary.json
├── unknown_predictions/                  ← PFAZ4: Bilinmeyen çekirdek tahminleri
│   ├── unknown_predictions.xlsx
│   └── aaa2_comparison.xlsx
├── cross_model/                          ← PFAZ5: Çapraz model analiz sonuçları
│   └── cross_model_analysis_summary.json
└── final_report/                         ← PFAZ6: Tez için nihai birleştirme raporu
    ├── final_report_<ts>.xlsx
    └── final_report_<ts>.tex
```

---

## Dosya Açıklamaları

### PFAZ1 — Dataset Üretimi (`pfaz01_dataset_generation/`)

| Dosya | Açıklama |
|-------|----------|
| `reports/theoretical_calculations/theoretical_calc_report_<ts>.json` | Nükleer özellik teorik hesaplama raporları. Her çalıştırma için ayrı dosya (timestamp). Bağlanma enerjisi, kabuk boşluğu, deformasyon vb. hesaplar. |
| `generated_datasets/generation_report.json` | Üretilen tüm dataset kombinasyonlarının özeti. Hedef × boyut × senaryo × feature_set sayıları, QM filtreleme sonuçları. |
| `generated_datasets/anomaly_explanation_report.json` | Her hedef için anomali olarak tespit edilen çekirdeklerin açıklaması: hangi sütun IQR sınırını aştı, kaç kat aştı, yön (üstte/altta). |
| `generated_datasets/quality_reports/` | Veri kalite kontrol raporları (outlier yüzdesi, validasyon sorunları). |

### PFAZ2 — AI Model Eğitimi (`pfaz02_ai_training/`)

| Dosya | Açıklama |
|-------|----------|
| `outputs/trained_models/{dataset}/{model}/{config}/metrics_{config}.json` | Her model konfigürasyonu için train/val/test metrikleri (R², RMSE, MAE). |
| `outputs/trained_models/{dataset}/{model}/{config}/cv_results_{config}.json` | Cross-validation sonuçları (5-fold). |
| `outputs/trained_models/training_summary.xlsx` | Tüm model eğitimlerinin özet tablosu. PFAZ6 bu dosyayı temel girdi olarak kullanır. |

Model tipleri: `DNN`, `RF` (Random Forest), `GBT` (Gradient Boosting), `XGB` (XGBoost), `LGB` (LightGBM), `CB` (CatBoost), `SVR`

### PFAZ3 — ANFIS Eğitimi (`pfaz03_anfis_training/`)

| Dosya | Açıklama |
|-------|----------|
| `outputs/anfis_models/{dataset}/{config}/metrics_{config}.json` | ANFIS konfigürasyonu başına train/val/test metrikleri + `training_meta` (mf_type, n_rules, n_inputs, outlier bilgisi). |
| `outputs/anfis_models/{dataset}/{config}/workspace_{config}.mat` | MATLAB workspace dosyası (MATLAB ile açılabilir). |
| `outputs/anfis_models/{dataset}/{config}/model_{config}.pkl` | Eğitilmiş ANFIS modeli (joblib). |

ANFIS konfigürasyonları (başarı sırasıyla): `CFG_Grid_2MF_Trap` > `CFG_Grid_2MF_Bell` > `CFG_Grid_2MF_Gauss` > `CFG_Grid_2MF_Tri` > `CFG_Grid_3MF_Bell` > `CFG_Grid_3MF_Gauss` > `CFG_SubClust_5` > `CFG_SubClust_8`

### PFAZ4 — Bilinmeyen Çekirdek Tahminleri (`pfaz04_unknown_predictions/`)

| Dosya | Açıklama |
|-------|----------|
| `outputs/unknown_predictions/unknown_predictions.xlsx` | aaa2.txt'deki bilinmeyen çekirdeklerin her model tarafından tahmin edilen değerleri. |
| `outputs/unknown_predictions/aaa2_comparison.xlsx` | aaa2.txt orijinal değerleri vs model tahminleri karşılaştırması (hedef bazında sheet'ler). |

### PFAZ5 — Çapraz Model Analizi (`pfaz05_cross_model/`)

| Dosya | Açıklama |
|-------|----------|
| `outputs/cross_model/cross_model_analysis_summary.json` | Modeller arası karşılaştırma analizi: en iyi model per hedef, ensemble ağırlıkları, konsensüs tahminleri. |

### PFAZ6 — Nihai Raporlama (`pfaz06_final_reporting/`)

| Dosya | Açıklama |
|-------|----------|
| `outputs/reports/final_report_<ts>.xlsx` | Ana tez raporu Excel dosyası. Sheet'ler: Overview, AI_Results, ANFIS_Results, AI_vs_ANFIS_Comparison, Best_Models_Per_Target, Best_FeatureSet_Per_Target, Anomaly_vs_NoAnomaly, **Anomaly_Explained**, Target_Statistics, Overall_Statistics, Feature_Abbreviations. |
| `outputs/reports/final_report_<ts>.tex` | LaTeX tablo çıktısı. Tez dokümanına doğrudan eklenebilir. |

### PFAZ8 — Görselleştirme (`pfaz08_visualization/`)

Görselleştirme çıktıları için `VISUALIZATIONS_INDEX.md` dosyasına bakın.

### PFAZ9 — Monte Carlo (`pfaz09_aaa2_monte_carlo/`)

| Dosya | Açıklama |
|-------|----------|
| `outputs/monte_carlo/mc_results_<ts>.json` | Monte Carlo simülasyon sonuçları (belirsizlik analizi). |
| `outputs/monte_carlo/robustness_report_<ts>.xlsx` | Robustness test Excel raporu. |

---

## Naming Convention

Dataset isimleri: `{TARGET}_{SIZE}_{SCENARIO}_{FEATURE_SET}_{SCALING}_{SAMPLING}[_NoAnomaly]`

Örnekler:
- `MM_150_S70_AZSMC_NoScaling_Random`
- `Beta_2_200_S80_MCZMNM_Standard_Stratified_NoAnomaly`
- `MM_QM_ALL_S70_AZB2EMC_Robust_Random`

Değerler:
- **TARGET**: `MM`, `QM`, `MM_QM`, `Beta_2`
- **SIZE**: `75`, `100`, `150`, `200`, `ALL_N` (N=mevcut çekirdek sayısı)
- **SCENARIO**: `S70` (ağır çekirdekler dahil), `S80` (yalnızca A≥80)
- **FEATURE_SET**: PFAZ1 feature_combination_manager'dan üretilen kısaltma (ör. `AZSMC`, `MCZMNM`)
- **SCALING**: `NoScaling`, `Standard`, `Robust`, `MinMax`
- **SAMPLING**: `Random`, `Stratified`

---

_Bu dosya `pfaz_modules/pfaz06_final_reporting/` tarafından otomatik olarak okunmaz;
yalnızca geliştirici referansı için oluşturulmuştur._
