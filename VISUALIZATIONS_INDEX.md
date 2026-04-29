# Visualizations Index

PFAZ8 görselleştirme sisteminin ürettiği tüm grafik ve görseller.  
İki geçiş halinde çalışır:

- **Birinci Geçiş (PFAZ 8)**: PFAZ6 verileri üzerinden standart grafikler  
  Kaynak: `pfaz_modules/pfaz08_visualization/visualization_master_system.py`

- **İkinci Geçiş (PFAZ 8 Supplemental, PFAZ13 sonrası)**: PFAZ9/12/13 grafikleri  
  Kaynak: `pfaz_modules/pfaz08_visualization/supplemental_visualizer.py`

---

## Çıktı Dizini

```
outputs/visualizations/
├── VISUALIZATIONS_INDEX.md       ← Bu dosya (geliştirici referansı)
├── S70/                          ← Senaryo S70 grafikleri
│   ├── <grafik_id>_<açıklama>.png
│   └── ...
├── S80/                          ← Senaryo S80 grafikleri
│   └── ...
├── combined/                     ← S70+S80 birleşik karşılaştırma grafikleri
│   └── ...
├── thesis/                       ← Teze hazır, yüksek çözünürlük PNG/PDF
│   └── ...
└── supplemental/                 ← İkinci geçiş: PFAZ9/12/13 grafikleri
    ├── MC9-A_uncertainty_violin.png
    ├── MC9-B_high_uncertainty_nuclei.png
    ├── MC9-C_ci_width_scatter.png
    ├── ST12-A_pvalue_heatmap.png
    ├── ST12-B_r2_boxplot.png
    ├── AM13-A_before_after_r2.png
    ├── AM13-C_optuna_history.png
    └── AM13-D_improvement_counts.png
```

---

## Grafik Kataloğu

Her grafik bir ID ile tanımlanır (`S{nn}` = Sistem, `V{nn}` = Vizualizasyon).

### Performans Karşılaştırma

| ID | Dosya | Açıklama | Kaynak |
|----|-------|----------|--------|
| S70 | `s70_r2_comparison_bar.png` | Hedef başına model Val R² karşılaştırma bar grafiği (S70) | `_plot_r2_comparison()` |
| S80 | `s80_r2_comparison_bar.png` | Hedef başına model Val R² karşılaştırma bar grafiği (S80) | `_plot_r2_comparison()` |
| S71 | `s71_r2_heatmap.png` | Model × Hedef R² ısı haritası | `_plot_r2_heatmap()` |
| S72 | `s72_rmse_comparison.png` | RMSE karşılaştırma (AI vs ANFIS) | `_plot_rmse_comparison()` |
| S73 | `s73_feature_set_impact.png` | Feature set boyutuna göre R² değişimi | `_plot_feature_set_impact()` |
| S74 | `s74_scaling_impact.png` | Scaling yöntemi (NoScaling/Standard/Robust/MinMax) etkisi | `_plot_scaling_impact()` |
| S75 | `s75_sampling_impact.png` | Sampling yöntemi (Random/Stratified) etkisi | `_plot_sampling_impact()` |

### AI Model Analizi

| ID | Dosya | Açıklama |
|----|-------|----------|
| S76 | `s76_model_type_comparison.png` | Model tipi (DNN/RF/XGB/LGB/CB/SVR) × hedef R² kutu grafiği |
| S77 | `s77_train_val_test_r2.png` | Train/Val/Test R² karşılaştırma (aşırı uyum tespiti) |
| S78 | `s78_anomaly_impact.png` | Anomaly vs NoAnomaly dataset performans farkı |
| S79 | `s79_nucleus_count_impact.png` | Dataset boyutu (75/100/150/200/ALL) etkisi |
| S81 | `s81_best_config_per_target.png` | Hedef başına en iyi konfigürasyon özeti |

### ANFIS Analizi

| ID | Dosya | Açıklama |
|----|-------|----------|
| S82 | `s82_anfis_mf_comparison.png` | Üyelik fonksiyonu tipi × R² karşılaştırma |
| S83 | `s83_anfis_convergence.png` | ANFIS iterasyon × kayıp eğrileri |
| S84 | `s84_anfis_outlier_impact.png` | Outlier temizleme etkisi (R² değişimi) |

### Tahmin Karşılaştırma

| ID | Dosya | Açıklama |
|----|-------|----------|
| S85 | `s85_pred_vs_actual_MM.png` | MM: tahmin vs gerçek scatter (en iyi model) |
| S86 | `s86_pred_vs_actual_QM.png` | QM: tahmin vs gerçek scatter (en iyi model) |
| S87 | `s87_pred_vs_actual_Beta2.png` | Beta_2: tahmin vs gerçek scatter (en iyi model) |
| S88 | `s88_pred_vs_actual_MM_QM.png` | MM_QM: tahmin vs gerçek scatter (çok çıktılı model) |
| S89 | `s89_residual_distribution.png` | Residual dağılımı histogramı (hedef başına) |

### Bilinmeyen Çekirdek Tahminleri

| ID | Dosya | Açıklama |
|----|-------|----------|
| S90 | `s90_unknown_predictions_MM.png` | aaa2.txt bilinmeyen çekirdeklerin MM tahmin dağılımı |
| S91 | `s91_unknown_predictions_QM.png` | aaa2.txt bilinmeyen çekirdeklerin QM tahmin dağılımı |
| S92 | `s92_unknown_nuclear_chart.png` | Bilinen vs bilinmeyen çekirdekler nükleer haritada (N-Z) |

### Çapraz Model ve Ensemble

| ID | Dosya | Açıklama |
|----|-------|----------|
| S93 | `s93_cross_model_consensus.png` | Çapraz model konsensüs güven bandı |
| S94 | `s94_ensemble_weights.png` | Ensemble ağırlıkları (hedef başına model katkısı) |

---

## Supplemental Visualizations (2nd Pass — PFAZ13 sonrası)

Kaynak: `pfaz_modules/pfaz08_visualization/supplemental_visualizer.py`  
Çıktı dizini: `outputs/visualizations/supplemental/`

### Monte Carlo Belirsizlik (PFAZ9)

| ID | Dosya | Açıklama | Kaynak veri |
|----|-------|----------|-------------|
| MC9-A | `MC9-A_uncertainty_violin.png` | Hedef başına tahmin std dağılımı (violin plot) | `AAA2_Complete_{target}.xlsx` — `std` sütunu |
| MC9-B | `MC9-B_high_uncertainty_nuclei.png` | Yüksek belirsizlikli çekirdekler (CV > 0.3) bar grafiği | Aynı kaynak, `cv` sütunu |
| MC9-C | `MC9-C_ci_width_scatter.png` | CI genişliği vs tahmin değeri scatter (hedef başına) | `ci_width = upper95 - lower95` |

### İstatistiksel Testler (PFAZ12)

| ID | Dosya | Açıklama | Kaynak veri |
|----|-------|----------|-------------|
| ST12-A | `ST12-A_pvalue_heatmap.png` | Model çifti × p-değeri ısı haritası (pairwise Wilcoxon) | `pfaz12_statistical_tests.xlsx` — `Pairwise_Wilcoxon` sheet |
| ST12-B | `ST12-B_r2_boxplot.png` | Model tipi başına Val R² kutu grafiği (PFAZ2 metrikleri) | `pfaz12_statistical_tests.xlsx` — `AI_Model_R2` sheet |

### AutoML Gelişim (PFAZ13)

| ID | Dosya | Açıklama | Kaynak veri |
|----|-------|----------|-------------|
| AM13-A | `AM13-A_before_after_r2.png` | Önce/sonra Val R² scatter + gelişim bar (hedef bazlı) | `automl_retraining_log.json` |
| AM13-C | `AM13-C_optuna_history.png` | Optuna trial geçmişi (hedef başına best value eğrisi) | `{target}_{model}_automl.json` — `trials` listesi |
| AM13-D | `AM13-D_improvement_counts.png` | İyileşen vs iyileşmeyen kombinasyon sayısı (hedef bazlı) | `automl_retraining_log.json` |

---

## Hedef Renk Kodları

Tüm grafiklerde tutarlı renk paleti:

| Hedef | Renk | Hex |
|-------|------|-----|
| MM | Mavi | `#2196F3` |
| QM | Yeşil | `#4CAF50` |
| Beta_2 | Turuncu | `#FF9800` |
| MM_QM | Kırmızı-Turuncu | `#FF5722` |

---

## Grafik Üretim Notları

- **Çözünürlük**: `dpi=150` (hızlı önizleme), `dpi=300` (`thesis/` klasörü için)
- **Format**: PNG varsayılan; PDF `thesis/` için
- **R² alt sınır filtresi**: R² < -10 olan sonuçlar grafiklere dahil edilmez (eksen bozulmasını önler)
- **Y ekseni**: Dinamik min/max (S80 gibi negatif değerler olan grafiklerde sabit [0,1] yerine)
- **Tüm hedefler**: MM, QM, MM_QM, Beta_2 — tüm grafikler dört hedefi birlikte gösterir

---

_Bu dosya `pfaz_modules/pfaz08_visualization/` tarafından otomatik okunmaz;
yalnızca geliştirici referansı için oluşturulmuştur._
