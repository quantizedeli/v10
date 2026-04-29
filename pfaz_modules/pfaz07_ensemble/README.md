# PFAZ 7 — Ensemble & Meta-Learning

Kayıtlı AI ve ANFIS modellerini gerçekten yükleyerek voting ve stacking ensemble yöntemlerini çalıştırır.

## Ana Giriş Noktası

**`pfaz7_complete_pipeline`** (`pfaz7_complete_ensemble_pipeline.py`)

```python
from pfaz_modules.pfaz07_ensemble.pfaz7_complete_ensemble_pipeline import pfaz7_complete_pipeline

results = pfaz7_complete_pipeline(
    trained_models_dir="outputs/trained_models",
    anfis_models_dir="outputs/anfis_models",
    datasets_dir="outputs/generated_datasets",
    output_dir="outputs/ensemble_results",
    top_n=20,
)
```

## Çalışma Akışı

```
RealModelLoader.get_top_models(target, top_n=20)
  -> trained_models/ ve anfis_models/ taranır
  -> metadata_*.json'dan feature_names okunur
  -> val_r2'ye göre en iyi N model seçilir

RealPredictionCollector.collect(records, target, split="test")
  -> Her model kendi feature setiyle test/val.csv'yi yükler
  -> NUCLEUS ID'leri üzerinde inner merge ile hizalama
  -> predictions_matrix: (n_nuclei x n_models)

RealEnsembleRunner.run_target(target)
  -> _run_voting(): simple, weighted_r2, weighted_inv_rmse
  -> _run_stacking(): Ridge, Lasso, RF, GBM meta-modeller
  -> Her yöntem için R2, RMSE, MAE hesapla
```

## Voting Yöntemleri

| Yöntem | Ağırlık Mantığı |
|--------|----------------|
| Simple Voting | Eşit ağırlık (ortalama) |
| Weighted R² | val_r2 oranında ağırlık |
| Weighted Inv-RMSE | 1/rmse oranında ağırlık |
| Rank-based | Sıralama dönüşümü sonrası ensemble |
| Dynamic Weight Adj. | 10 iterasyonla ağırlık optimizasyonu |

## Stacking Meta-Öğreniciler

Seviye-0 modellerin val tahminleri üzerinde eğitilir, test tahminlerinde kullanılır:

| Meta-Model | Sınıf |
|------------|-------|
| Ridge | `AdvancedStackingEnsemble` içinde |
| Lasso | — |
| ElasticNet | — |
| RF Meta | — |
| GBM Meta | — |
| MLP | — |

5-fold CV ile out-of-fold (OOF) tahminler üretilir.

## Çeşitlilik Analizi

`EnsembleDiversityAnalyzer` — modeller arası:
- Korelasyon matrisi
- Disagreement ölçüsü
- Q-istatistiği
- Double fault

## MM_QM Desteği

Multi-output hedefi (n, 2) şeklinde işlenir — her sütun ayrı ensemble.

## I/O Yapısı

**Giriş:**
- `outputs/trained_models/` — AI PKL modelleri + metadata JSON
- `outputs/anfis_models/` — ANFIS PKL modelleri
- `outputs/generated_datasets/{dataset}/test.csv`, `val.csv`

**Çıkış:**
```
outputs/ensemble_results/
├── ensemble_report_{YYYYMMDD_HHMMSS}.xlsx
│   -> Sayfalar: MM_Results, QM_Results, Beta2_Results, Predictions_MM, ...
└── ensemble_summary.json
    -> best_method, best_r2, improvement_over_single, per_target
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `pfaz7_complete_ensemble_pipeline.py` | RealModelLoader, RealPredictionCollector, RealEnsembleRunner, AdvancedVotingEnsemble, AdvancedStackingEnsemble | Ana pipeline (gerçek model yükleme) |
| `ensemble_evaluator.py` | EnsembleDiversityAnalyzer, ComprehensiveEnsembleEvaluator | Çeşitlilik ve karşılaştırma |
| `stacking_meta_learner.py` | StackingMetaLearner | Stacking meta-öğrenici |
| `ensemble_model_builder.py` | EnsembleModelBuilder | Model oluşturma yardımcıları |
| `faz7_ensemble_pipeline.py` | Faz7EnsemblePipeline | Alternatif pipeline |
