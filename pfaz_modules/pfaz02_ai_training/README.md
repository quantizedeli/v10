# PFAZ 2 — AI Model Training

PFAZ 1'den gelen 848 veri seti üzerinde 6 model türünü 50 konfigürasyonla paralel olarak eğitir.

## Ana Sınıf

**`ParallelAITrainer`** (`parallel_ai_trainer.py`)

```python
from pfaz_modules.pfaz02_ai_training import ParallelAITrainer

trainer = ParallelAITrainer(
    datasets_dir="outputs/generated_datasets",
    output_dir="outputs/trained_models",
    n_workers=None,          # None = os.cpu_count()
    gpu_enabled=False,
    use_hyperparameter_tuning=False,
    use_model_validation=True,
    use_advanced_models=True,
    use_parallel_training=True,
)
results = trainer.train_all_models_parallel(n_configs=50)
```

## Model Türleri

| Sınıf | Model | Kütüphane | Notlar |
|-------|-------|-----------|--------|
| `RandomForestTrainer` | RF | scikit-learn | Adaptif: küçük dataset'te n_estimators azaltılır |
| `XGBoostTrainer` | XGBoost | xgboost | Early stopping destekli |
| `LightGBMTrainer` | LightGBM | lightgbm | MultiOutputRegressor wrapper |
| `CatBoostTrainer` | CatBoost | catboost | Kategorik özellik desteği |
| `SVRTrainer` | SVR | scikit-learn | StandardScaler dahili, rbf kernel |
| `DNNTrainer` | DNN | TensorFlow/Keras | BatchNorm + Dropout + L2, Huber loss, EarlyStopping |

### DNN Mimarisi
- Katmanlar: `[128, 64, 32]` (yapılandırılabilir)
- Dropout: `[0.2, 0.2, 0.1]` per katman
- Optimizer: Adam (clipnorm=1.0, clipvalue=0.5)
- Loss: Huber
- Callbacks: EarlyStopping (patience=20), ReduceLROnPlateau
- Hem özellikler hem hedefler StandardScaler ile ölçeklenir
- `DNN_MIN_SAMPLES = 80` — daha az veri varsa ValueError atılır

## 50 Konfigürasyon

`_create_default_configs(n_configs=50)` ile üretilir:

| Grup | Aralık | Model |
|------|--------|-------|
| RF_001–RF_020 | 1–20 | Random Forest |
| XGB_021–XGB_035 | 21–35 | XGBoost |
| DNN_036–DNN_050 | 36–50 | DNN |

Her konfigürasyon farklı hiperparametre kombinasyonu içerir (n_estimators, max_depth, learning_rate vb.).

## Çalışma Akışı

```
1. _discover_datasets(datasets_dir)   -> List[Path]  (tüm alt klasörler)
2. _create_default_configs(50)        -> List[Dict]
3. create_training_jobs(...)          -> List[TrainingJob]
4. train_all_parallel(jobs)           -> ThreadPoolExecutor ile paralel
   └── train_single_job(job)
       ├── BaseAITrainer.load_dataset()   # train/val/test.csv oku
       ├── model.train(X_train, y_train, X_val, y_val)
       ├── model.predict(X_test)
       ├── calculate_metrics()            # R2, RMSE, MAE
       └── save_model(filepath)           # .pkl
5. save_summary_report()              -> training_summary.json + .xlsx
```

## Özellik Seçimi

- Her veri setinin `metadata.json` dosyasından `feature_names` okunur
- Hedef sütunlar: `MM` / `Q` veya `QM` / `Beta_2`
- Unicode eksi işareti, bilinmeyen string, NaN temizlenir
- Veri sızıntısı (data leakage) koruması: test verisinden fit yapılmaz

## I/O Yapısı

**Giriş:** `outputs/generated_datasets/{dataset_adi}/train.csv`, `val.csv`, `test.csv`

**Çıkış:**
```
outputs/trained_models/
└── {dataset_adi}/
    └── {model_turu}/
        └── {config_id}/
            ├── model_{model_turu}_{config_id}.pkl
            ├── metrics_{config_id}.json        <- R2, RMSE, MAE (train/val/test)
            ├── cv_results_{config_id}.json     <- (validation etkinse)
            └── overfitting_analysis_{config_id}.json
```

**Özet raporlar:**
```
outputs/trained_models/
├── training_summary.json
└── training_results_summary.xlsx
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `parallel_ai_trainer.py` | ParallelAITrainer | Ana orkestratör, 6 model türü |
| `model_trainer.py` | ParallelTrainingPipeline | Legacy trainer |
| `hyperparameter_tuner.py` | HyperparameterTuner | Opsiyonel hiperparametre arama |
| `model_validator.py` | CrossValidationAnalyzer | 5-fold CV ve robustness testleri |
| `overfitting_detector.py` | OverfittingDetector | Train-val fark analizi |
| `advanced_models.py` | BayesianNeuralNetwork, PINN | BNN ve Physics-Informed NN |
| `gpu_optimization.py` | GPUOptimizer | GPU bellek/batch yönetimi |
| `seed_tracker.py` | SeedTracker | Tekrarlanabilirlik için seed kayıt |
