# PFAZ 13 — AutoML

Düşük R² skorlu PFAZ 2 modellerini otomatik tespit edip Optuna ile yeniden optimize eder.

## Ana Sınıf: `AutoMLRetrainingLoop`

**`automl_retraining_loop.py`**

```python
from pfaz_modules.pfaz13_automl import AutoMLRetrainingLoop

loop = AutoMLRetrainingLoop(
    models_dir="outputs/trained_models",       # PFAZ2 çıktısı
    datasets_dir="outputs/generated_datasets",  # PFAZ1 çıktısı
    output_dir="outputs/automl_results",
    r2_threshold=0.80,    # bu altı = "düşük"
    n_trials=30,          # Optuna deneme sayısı / aday
    model_types=["rf", "xgb", "lgb"],
    max_retrain=0,        # 0 = sınırsız
)
loop.run()
```

## Çalışma Akışı

```
1. find_low_scoring_candidates()
   -> trained_models/ altındaki tüm metrics_*.json tara
   -> val_r2 < r2_threshold olan kayıtları topla
   -> Liste: [(target, dataset, model_type, original_r2)]

2. Her aday için _retrain_one():
   -> PFAZ1 CSV'lerini yükle (train.csv, val.csv)
   -> AutoMLOptimizer(X_train, y_train, X_val, y_val, model_type) çalıştır
   -> Iyileştirme kaydı: before_r2, after_r2, improvement, best_params

3. Sonuçlar:
   -> _save_json_log() -> automl_retraining_log.json
   -> _save_excel_report() -> automl_improvement_report.xlsx (3 sayfa)

4. Opsiyonel (try/except ile):
   -> AutoMLFeatureEngineer (en çok iyileşen dataset için özellik seçimi)
   -> AutoMLANFISOptimizer (en kötü dataset için ANFIS konfigürasyon arama)
   -> AutoMLVisualizer (Optuna grafikleri)
```

## Optimizer: `AutoMLOptimizer`

**`automl_optimizer.py`** — Optuna TPE sampler + MedianPruner

```python
optimizer = AutoMLOptimizer(
    X_train, y_train, X_val, y_val,
    model_type="xgb",  # rf | xgb | gbm | lgb | cb | svr | dnn
)
best_params = optimizer.optimize(n_trials=100, n_jobs=1, timeout=300)
optimizer.save_results("outputs/automl_results")
```

### Arama Uzayları

| Model | Arama Uzayı |
|-------|------------|
| RF | n_estimators, max_depth, min_samples_split/leaf, max_features |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, colsample, reg_alpha/lambda |
| GBM | n_estimators, max_depth, learning_rate, subsample |
| LightGBM | n_estimators, num_leaves, learning_rate, subsample, colsample, reg_alpha/lambda, min_child_samples |
| CatBoost | iterations, depth, learning_rate, l2_leaf_reg |
| SVR | C, epsilon, kernel (rbf/linear), gamma |
| DNN | n_layers(2–4), hidden_units(32–256), dropout, lr, batch_size, epochs=80 |

### DNN Özel Davranış
- Y-ölçekleme: StandardScaler (Huber loss)
- EarlyStopping: patience=10
- Predict sonrası inverse_transform uygulanır
- R² < -2.0 ceza puanı (diverjans koruması)

### Optuna Yapılandırması
- Sampler: TPE (Tree-structured Parzen Estimator)
- Pruner: MedianPruner
- Yön: maximize (R²)

## Excel Çıktısı: `automl_improvement_report.xlsx`

| Sayfa | İçerik |
|-------|--------|
| Summary | Target, Dataset, Orijinal Model, En İyi Model, Önceki/Sonraki R², İyileşme, Zaman |
| Best_Params | Uzun format — Target, Dataset, Model, Parametre, Değer |
| Overview | Toplam yeniden eğitilen, iyileştirilen, ortalama iyileşme, maksimum iyileşme, eşik |

## I/O Yapısı

**Giriş:**
- `outputs/trained_models/` — PFAZ2 metrics JSON'ları
- `outputs/generated_datasets/` — PFAZ1 train/val CSV'leri

**Çıkış:**
```
outputs/automl_results/
├── automl_improvement_report.xlsx   <- 3 sayfa
├── automl_retraining_log.json       <- kayıt listesi
├── automl_summary.json              <- özet metrikler
├── feature_engineering/             <- AutoMLFeatureEngineer (opsiyonel)
├── anfis_optimization/              <- AutoMLANFISOptimizer (opsiyonel)
└── automl_visualizations/           <- Optuna grafikleri (opsiyonel)
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `automl_retraining_loop.py` | AutoMLRetrainingLoop | Ana döngü, düşük R² tespiti ve yeniden eğitim |
| `automl_optimizer.py` | AutoMLOptimizer | Optuna TPE, 7 model türü arama uzayı |
| `automl_hyperparameter_optimizer.py` | AutoMLHyperparameterOptimizer | Geriye dönük uyumluluk takma adı |
| `automl_feature_engineer.py` | AutoMLFeatureEngineer | Özellik seçimi ve mühendisliği |
| `automl_anfis_optimizer.py` | AutoMLANFISOptimizer | ANFIS konfigürasyon arama |
| `automl_visualizer.py` | AutoMLVisualizer | Optuna grafikleri |
| `automl_logging_reporting_system.py` | AutoMLLoggingReportingSystem | Deneme bazlı loglama |
| `feature_engineering_extended.py` | — | Gelişmiş özellik mühendisliği |
