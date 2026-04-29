# PFAZ 6 — Final Reporting

PFAZ 2–5 metrik JSON'larını toplayarak 20+ sayfalık kapsamlı Excel raporu ve LaTeX tez bölümü üretir.

## Ana Sınıf

**`FinalReportingPipeline`** (`pfaz6_final_reporting.py`)

```python
from pfaz_modules.pfaz06_final_reporting import FinalReportingPipeline

pipeline = FinalReportingPipeline(
    ai_models_dir="outputs/trained_models",
    anfis_models_dir="outputs/anfis_models",
    output_dir="outputs/reports",
    datasets_dir="outputs/generated_datasets",
    validation_dir="outputs/trained_models",
    unknown_dir="outputs/unknown_predictions",
    cross_model_dir="outputs/cross_model_analysis",
)
pipeline.collect_all_results()
pipeline.generate_thesis_tables()
```

## Veri Toplama

| Metod | Kaynak | Toplanan |
|-------|--------|---------|
| `_collect_ai_results()` | `{trained_models}/{dataset}/{model_type}/{config}/metrics_{config}.json` | R2, RMSE, MAE, Target, Feature_Set, Dataset meta |
| `_collect_anfis_results()` | `{anfis_models}/{dataset}/{config}/metrics_{config}.json` | + mf_type, n_rules, method |
| `_collect_robustness_results()` | `{validation_dir}/{model}/cv_summary.json` | CV sonuçları |
| `_collect_crossmodel_results()` | `cross_model_dir/cross_model_analysis_summary.json` | Anlaşma metrikleri |
| `_collect_unknown_predictions()` | `unknown_dir/Unknown_Nuclei_Results.xlsx` | Test R², degradasyon |

## Excel Çıktısı: `THESIS_COMPLETE_RESULTS.xlsx`

20+ sayfa, büyük veriler (>50k satır) chunked olarak yazılır:

| Sayfa | İçerik |
|-------|--------|
| Overview | Dashboard özet |
| All_AI_Models | Tüm AI model sonuçları (chunked) |
| RF_Models / XGB_Models / DNN_Models / ... | Model türü bazlı |
| AI_Dataset_Summary | Dataset başına Best/Mean R² |
| All_ANFIS_Models | Tüm ANFIS sonuçları |
| ANFIS_Dataset_Summary | Dataset başına ANFIS özeti |
| ANFIS_Config_Comparison | Konfigürasyon bazlı karşılaştırma |
| AI_vs_ANFIS_Comparison | Yan yana, kazanan sütunu |
| Best_Models_Per_Target | Her hedef için en iyi model |
| Best_FeatureSet_Per_Target | Her hedef için en iyi özellik seti |
| Anomaly_vs_NoAnomaly | Anomali temizlemenin etkisi |
| Anomaly_Explained | Aykırı değer açıklamaları |
| Target_Statistics | Hedef bazlı istatistikler |
| Robustness_CV | Çapraz doğrulama sonuçları |
| CrossModel_Summary | Model anlaşma özeti |
| Unknown_Predictions | Test seti tahminleri |
| Overall_Statistics | Genel istatistikler |
| Feature_Abbreviations | Özellik kısaltma tablosu |

### Renk Kodlaması (openpyxl)
- `CLR_EXCELLENT` — R² ≥ 0.95
- `CLR_GOOD` — R² ≥ 0.90
- `CLR_MEDIUM` — R² ≥ 0.70
- `CLR_POOR` — R² < 0.70
- `_R2_FLOOR = -10.0` — grafik ölçeği için minimum

## LaTeX Raporu (`latex_generator.py`)

**`LaTeXReportGenerator.generate_report(all_results)`**

Bölümler: Abstract → Introduction → Methodology → Results → Discussion → Conclusions → References → Appendices

- Appendix A: Özellik kısaltma tablosu
- Appendix B: 8 ANFIS konfigürasyonu
- `_compile_pdf()`: pdflatex varsa otomatik derleme

## Grafik Üretimi (`excel_charts.py`)

**`ExcelChartGenerator.generate_all_charts(all_results)`**

- `ai_models_chart.xlsx`: AI_Top2000, {ModelType}_Top200, Dataset_Summary sayfaları
- `anfis_models_chart.xlsx`: ANFIS_Models sayfası

## I/O Yapısı

**Giriş:** PFAZ 2/3/4/5 JSON ve Excel çıktıları

**Çıkış:**
```
outputs/reports/
├── THESIS_COMPLETE_RESULTS.xlsx          <- ana rapor (veya zaman damgalı)
├── ai_models_chart.xlsx
├── anfis_models_chart.xlsx
└── latex/
    ├── thesis_report.tex
    └── thesis_report.pdf                 <- pdflatex varsa
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `pfaz6_final_reporting.py` | FinalReportingPipeline | Ana orkestratör, 20+ sayfa Excel |
| `excel_charts.py` | ExcelChartGenerator | xlsxwriter grafik üretimi |
| `latex_generator.py` | LaTeXReportGenerator | LaTeX tez bölümleri |
| `excel_standardizer.py` | ExcelStandardizer | Renk, filtre, freeze pane yardımcıları |
| `comprehensive_excel_reporter.py` | ComprehensiveExcelReporter | Ek detaylı Excel raporlama |
