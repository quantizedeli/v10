# PFAZ 5 — Cross-Model Analysis

Tüm AI ve ANFIS modellerinin tahminlerini ortak çekirdekler üzerinde karşılaştırır; Good/Medium/Poor sınıflandırması yapar.

## Ana Sınıf

**`CrossModelAnalysisPipeline`** (`faz5_cross_model_analysis.py`)

```python
from pfaz_modules.pfaz05_cross_model import CrossModelAnalysisPipeline

pipeline = CrossModelAnalysisPipeline(
    trained_models_dir="outputs/trained_models",
    anfis_models_dir="outputs/anfis_models",
    datasets_dir="outputs/generated_datasets",
    output_dir="outputs/cross_model_analysis",
)
results = pipeline.run_complete_analysis()
```

## Çalışma Akışı

```
1. _collect_ai_predictions()
   -> trained_models/ taranır, her model test.csv üzerinde predict()
   -> self.all_predictions[target_key][model_label] = DataFrame(nucleus, experimental, predicted)

2. _collect_anfis_predictions()
   -> anfis_models/ taranır, aynı işlem

3. Her target için CrossModelEvaluator:
   -> evaluate_common_performance(target_name, top_n=50)
   -> Ortak çekirdek ID'leri üzerinde hizalama (inner merge)
   -> Model anlaşma matrisi hesaplama
   -> Çekirdek sınıflandırma: Good (R²>0.90) / Medium / Poor

4. _create_master_report(results) -> MASTER_CROSS_MODEL_REPORT.xlsx

5. _save_summary_json(results) -> cross_model_analysis_summary.json
```

## Tahmin Depolama

```python
self.all_predictions = {
    "MM": {
        "AI_RF_config001": DataFrame(nucleus, experimental, predicted),
        "AI_XGB_config002": ...,
        "ANFIS_CFG_Grid_2MF_Gauss": ...,
    },
    "QM": { ... },
    "Beta_2": { ... },
}
```

## Master Excel Sayfaları

| Sayfa | İçerik |
|-------|--------|
| Overall_Summary | Tüm target özet istatistikleri |
| MM_Good | R²>0.90 olan MM çekirdekleri |
| MM_Medium | 0.70≤R²≤0.90 MM çekirdekleri |
| MM_Poor | R²<0.70 MM çekirdekleri |
| QM_Good / QM_Medium / QM_Poor | QM için aynı |
| Beta2_Good / Beta2_Medium / Beta2_Poor | Beta_2 için aynı |
| Model_Statistics | Her model türünün istatistikleri |
| Agreement_Overview | Model anlaşma matrisi özeti |

## I/O Yapısı

**Giriş:**
- `outputs/trained_models/` — AI PKL modelleri
- `outputs/anfis_models/` — ANFIS PKL modelleri
- `outputs/generated_datasets/` — test.csv dosyaları

**Çıkış:**
```
outputs/cross_model_analysis/
├── MM/
│   └── MM_cross_model_report.xlsx
├── QM/
│   └── QM_cross_model_report.xlsx
├── Beta_2/
│   └── Beta_2_cross_model_report.xlsx
├── MASTER_CROSS_MODEL_REPORT.xlsx
└── cross_model_analysis_summary.json
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `faz5_cross_model_analysis.py` | CrossModelAnalysisPipeline | Ana orkestratör |
| `cross_model_evaluator.py` | CrossModelEvaluator | Model anlaşma metrikleri |
| `best_model_selector.py` | BestModelSelector | En iyi model seçimi |
| `faz5_complete_cross_model.py` | CompleteCrossModelAnalyzer | Genişletilmiş analiz |
| `isotope_chain_analyzer.py` | IsotopeChainAnalyzer | İzotop zinciri performansı |
| `optimizer_comparison_reporter.py` | OptimizerComparisonReporter | Optimizer bazlı karşılaştırma |
