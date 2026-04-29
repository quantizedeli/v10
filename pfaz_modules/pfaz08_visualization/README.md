# PFAZ 8 — Visualization System

80+ statik (PNG, 300 dpi) ve interaktif (HTML/Plotly) görselleştirme üretir.

## Ana Sınıf

**`MasterVisualizationSystem`** (`visualization_master_system.py`)

```python
from pfaz_modules.pfaz08_visualization import MasterVisualizationSystem

viz = MasterVisualizationSystem(
    results_dir="outputs",
    output_dir="outputs/visualizations",
)
viz.run_all()
```

## Görselleştirme Sınıfları

| Sınıf | Dosya | Ürettiği Grafikler |
|-------|-------|-------------------|
| `RobustnessVisualizer` | `robustness_visualizations_complete.py` | Pertürbasyon duyarlılığı, gürültü robustluk, CV stabilite |
| `SHAPVisualizer` | `shap_analysis.py` | Beeswarm, force plot, dependence plot |
| `AnomalyKernelVisualizer` | `anomaly_visualizations_complete.py` | Anomali vs normal dağılım, PCA/t-SNE kümeleme |
| `MasterReportVisualizer` | `master_report_visualizations_complete.py` | Plotly interaktif dashboard |
| `PredictionVisualizer` | `visualization_system.py` | Actual vs Predicted scatter, residual analizi |
| `ModelComparisonVisualizer` | `model_comparison_dashboard.py` | Model sıralama, karşılaştırma |
| `AIVisualizer` | `ai_visualizer.py` | AI modele özel grafikler |
| `InteractiveHTMLVisualizer` | `interactive_html_visualizer.py` | Plotly HTML çıktıları |
| `LogAnalyticsVisualizer` | `log_analytics_visualizations_complete.py` | Log analizi grafikleri |

## Grafik Konfigürasyonu

```python
PLOT_CONFIG = {
    'dpi': 300,
    'figsize_default': (14, 10),
    'figsize_small': (10, 6),
    'figsize_large': (18, 12),
    'style': 'seaborn-v0_8-darkgrid',
    'colormap': 'Set2',
}
```

## İki Geçişli Sistem

**Geçiş 1 — Standart:** PFAZ 6 çıktılarından temel grafikler  
**Geçiş 2 — Tamamlayıcı (`supplemental_visualizer.py`):** PFAZ 9 Monte Carlo, PFAZ 12 istatistik, PFAZ 13 AutoML grafikleri eklenir

## Üretilen Grafik Türleri

| Tür | Adet | Açıklama |
|-----|------|---------|
| Scatter (Tahmin vs Gerçek) | ~20 | Her hedef için, hata renk kodlu |
| Isı haritaları (Heatmap) | ~15 | Korelasyon matrisi, Z×N grid hatası |
| 3D yüzey grafikleri | ~10 | Z-N-MM tahmin yüzeyi |
| Hata dağılım grafikleri | ~15 | Histogram + KDE + normal fit |
| Model karşılaştırma | ~12 | R² bar chart, box plot |
| İnteraktif HTML | ~8 | Plotly scatter, dashboard |
| SHAP grafikleri | ~5 | Beeswarm, force, dependence |
| MC belirsizlik grafikleri | ~5 | CI bantlı tahmin |

## I/O Yapısı

**Giriş:** PFAZ 2–9 çıktı dizinleri (DataFrame, JSON, Excel, PKL)

**Çıkış:**
```
outputs/visualizations/
├── scatter_plots/
├── heatmaps/
├── 3d_plots/
├── distributions/
├── comparisons/
├── shap/
├── interactive/           <- .html dosyaları
└── supplemental/          <- PFAZ 9/12/13 grafikleri
```

## PFAZ 10 Tez Entegrasyonu

`pfaz8_thesis_charts.py` — tez figürü standartlarına uygun grafikler (LaTeX uyumlu font, kenar boşlukları)

## Modüller

| Dosya | Görev |
|-------|-------|
| `visualization_master_system.py` | MasterVisualizationSystem, RobustnessVisualizer, SHAPVisualizer, AnomalyKernelVisualizer, MasterReportVisualizer, PredictionVisualizer |
| `visualization_system.py` | Temel görselleştirme sistemi |
| `visualization_advanced_modules.py` | Gelişmiş istatistiksel grafikler |
| `shap_analysis.py` | SHAPAnalyzer |
| `ai_visualizer.py` | AI model görselleştirici |
| `anomaly_visualizations_complete.py` | Anomali analiz grafikleri |
| `model_comparison_dashboard.py` | Model karşılaştırma dashboard |
| `interactive_html_visualizer.py` | Plotly HTML |
| `supplemental_visualizer.py` | PFAZ 9/12/13 tamamlayıcı grafikleri |
| `pfaz8_thesis_charts.py` | Tez uyumlu grafik üretimi |
