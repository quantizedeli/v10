# Nuclear Physics AI Project - Kullanım Kılavuzu

## 📚 İçindekiler

1. [Proje Yapısı](#proje-yapısı)
2. [Hızlı Başlangıç](#hızlı-başlangıç)
3. [PFAZ Modülleri](#pfaz-modülleri)
4. [Optional Modüller](#optional-modüller)
5. [Kullanım Örnekleri](#kullanım-örnekleri)
6. [Konfigürasyon](#konfigürasyon)
7. [Testler](#testler)
8. [Sorun Giderme](#sorun-giderme)

---

## Proje Yapısı

```
nucdatav1/
├── main.py                          # Ana orchestrator
├── run_complete_pipeline.py         # Alternatif entry point
├── config.json                      # Konfigürasyon dosyası
│
├── pfaz_modules/                    # 13 PFAZ modülü
│   ├── pfaz01_dataset_generation/   # Dataset oluşturma
│   ├── pfaz02_ai_training/          # AI model eğitimi
│   ├── pfaz03_anfis_training/       # ANFIS eğitimi
│   ├── pfaz04_unknown_predictions/  # Bilinmeyen nuclei tahminleri
│   ├── pfaz05_cross_model/          # Model karşılaştırma
│   ├── pfaz06_final_reporting/      # Raporlama
│   ├── pfaz07_ensemble/             # Ensemble modeller
│   ├── pfaz08_visualization/        # Görselleştirme
│   ├── pfaz09_aaa2_monte_carlo/     # Monte Carlo analizi
│   ├── pfaz10_thesis_compilation/   # Tez derleme
│   ├── pfaz11_production/           # Production deployment
│   ├── pfaz12_advanced_analytics/   # İleri analitik
│   └── pfaz13_automl/               # AutoML entegrasyonu
│
├── utils/                           # Utility modüller
│   ├── smart_cache.py
│   ├── checkpoint_manager.py
│   ├── ai_model_checkpoint.py
│   └── file_io_utils.py
│
├── scripts/                         # Utility scriptler
│   ├── check_pfaz_completeness.py
│   ├── log_parser.py
│   ├── generate_sample_data.py
│   └── examples/                    # Örnek kullanımlar
│       ├── example_performance_pipeline.py
│       └── example_usage.py
│
├── tests/                           # Test dosyaları
│   ├── test_integration/
│   ├── test_units/
│   └── test_smoke/
│
└── outputs/                         # Çıktı dizini
    ├── generated_datasets/
    ├── trained_models/
    └── ...
```

---

## Hızlı Başlangıç

### 1. Gereksinimler

```bash
pip install -r requirements.txt
```

Temel kütüphaneler:
- numpy, pandas, scikit-learn
- tensorflow, torch, xgboost
- matplotlib, seaborn, plotly
- openpyxl

### 2. Tüm Pipeline'ı Çalıştırma

**İnteraktif Mod:**
```bash
python main.py --interactive
```

**Tüm PFAZ'ları Çalıştırma:**
```bash
python main.py --run-all
```

**Belirli Bir PFAZ'ı Çalıştırma:**
```bash
python main.py --pfaz 2  # PFAZ 2'yi çalıştır
```

### 3. Konfigürasyon Dosyası Oluşturma

```bash
# Varsayılan config.json oluşturulur
python main.py --check-deps
```

---

## PFAZ Modülleri

### PFAZ 01: Dataset Generation

**Amaç:** Nükleer fizik verileri üretimi ve zenginleştirme

**Ana Modül:**
```python
from pfaz_modules.pfaz01_dataset_generation import DatasetGenerationPipelineV2

pipeline = DatasetGenerationPipelineV2(
    aaa2_txt_path='aaa2.txt',
    output_dir='outputs/generated_datasets',
    targets=['MM', 'QM', 'Beta_2'],
    nucleus_counts=[75, 100, 150, 200, 'ALL']
)

results = pipeline.run_complete_pipeline()
```

**Optional Modüller:**
- `ControlGroupGenerator` - Kontrol grubu sentetik verisi
- `DataEnricher` - Veri zenginleştirme
- `DataQualityChecker` - Kalite kontrolü
- `NucleiDistributionAnalyzer` - Dağılım analizi

**Kullanım:**
```python
from pfaz_modules.pfaz01_dataset_generation import (
    DataEnricher,
    DATA_ENRICHER_AVAILABLE
)

if DATA_ENRICHER_AVAILABLE:
    enricher = DataEnricher()
    enriched_data = enricher.enrich(raw_data)
```

---

### PFAZ 02: AI Training

**Amaç:** Paralel AI model eğitimi (RF, XGBoost, DNN)

**Ana Modül:**
```python
from pfaz_modules.pfaz02_ai_training import ParallelAITrainer

trainer = ParallelAITrainer(
    datasets_dir='outputs/generated_datasets',
    models_dir='outputs/trained_models',
    training_config_path='pfaz_modules/pfaz02_ai_training/training_configs_50.json',
    use_hyperparameter_tuning=True,
    use_model_validation=True,
    use_advanced_models=False
)

results = trainer.train_all_models_parallel(n_configs=50)
```

**Optional Modüller:**

1. **Hyperparameter Tuner** - Optuna ile hiperparametre optimizasyonu
```python
from pfaz_modules.pfaz02_ai_training import (
    HyperparameterTuner,
    HYPERPARAMETER_TUNER_AVAILABLE
)

if HYPERPARAMETER_TUNER_AVAILABLE:
    tuner = HyperparameterTuner(model_type='RF', n_trials=50)
    best_params = tuner.tune(X_train, y_train, X_val, y_val)
```

2. **Model Validator** - Cross-validation ve robustness testing
```python
from pfaz_modules.pfaz02_ai_training import (
    CrossValidationAnalyzer,
    MODEL_VALIDATOR_AVAILABLE
)

if MODEL_VALIDATOR_AVAILABLE:
    validator = CrossValidationAnalyzer(model, 'MyModel')
    cv_results = validator.run_cv(X, y, cv=5)
```

3. **Overfitting Detector** - Train/Val gap analizi
```python
from pfaz_modules.pfaz02_ai_training import (
    OverfittingDetector,
    OVERFITTING_DETECTOR_AVAILABLE
)

if OVERFITTING_DETECTOR_AVAILABLE:
    detector = OverfittingDetector()
    analysis = detector.analyze_training_metrics(train_metrics, val_metrics)
```

4. **Advanced Models** - BNN, PINN, Transfer Learning
```python
from pfaz_modules.pfaz02_ai_training import (
    BayesianNeuralNetwork,
    PINN,
    ADVANCED_MODELS_AVAILABLE
)

if ADVANCED_MODELS_AVAILABLE:
    bnn = BayesianNeuralNetwork(input_dim=10)
    bnn.train(X_train, y_train)
```

---

### PFAZ 13: AutoML Integration

**Amaç:** Otomatik hiperparametre optimizasyonu ve feature engineering

**Ana Modül:**
```python
from pfaz_modules.pfaz13_automl import AutoMLHyperparameterOptimizer

automl = AutoMLHyperparameterOptimizer(output_dir='outputs/automl_results')
results = automl.optimize_all_models()
```

**Optional Modüller:**
- `AutoMLVisualizer` - AutoML sonuçlarının görselleştirilmesi
- `AutoMLFeatureEngineer` - Otomatik feature engineering
- `AutoMLOptimizer` - Alternatif AutoML motoru

---

### PFAZ 08: Visualization

**Amaç:** Kapsamlı görselleştirme ve dashboard oluşturma

**Ana Modül:**
```python
from pfaz_modules.pfaz08_visualization import VisualizationMasterSystem

viz = VisualizationMasterSystem(output_dir='outputs/visualizations')
results = viz.generate_all_visualizations()
```

**Optional Modüller:**
- `SHAPAnalyzer` - SHAP interpretability analizi
- `InteractiveHTMLVisualizer` - İnteraktif HTML dashboardlar
- `ModelComparisonDashboard` - Model karşılaştırma UI
- `AnomalyVisualizationSystem` - Anomali görselleştirmeleri

**Kullanım Örneği:**
```python
from pfaz_modules.pfaz08_visualization import (
    SHAPAnalyzer,
    SHAP_ANALYSIS_AVAILABLE
)

if SHAP_ANALYSIS_AVAILABLE:
    shap_analyzer = SHAPAnalyzer()
    shap_values = shap_analyzer.explain(model, X_test)
    shap_analyzer.plot_summary(shap_values)
```

---

## Optional Modüller

Tüm optional modüller `try-except` blokları ile import edilir ve availability flag'leri vardır.

### Kullanım Paterni:

```python
from pfaz_modules.pfaz02_ai_training import (
    HyperparameterTuner,
    HYPERPARAMETER_TUNER_AVAILABLE
)

# Önce mevcudiyeti kontrol et
if HYPERPARAMETER_TUNER_AVAILABLE:
    tuner = HyperparameterTuner(model_type='RF')
    best_params = tuner.tune(X_train, y_train, X_val, y_val)
    print(f"Best parameters: {best_params}")
else:
    print("HyperparameterTuner not available (Optuna not installed?)")
    # Fallback to default parameters
    best_params = {'n_estimators': 100, 'max_depth': 10}
```

---

## Konfigürasyon

### config.json Yapısı

```json
{
  "output_dir": "outputs",
  "data_file": "aaa2.txt",
  "pfaz_config": {
    "1": {
      "enabled": true,
      "dataset_sizes": [75, 100, 150, 200, "ALL"]
    },
    "2": {
      "enabled": true,
      "n_configs": 50,
      "parallel": true,
      "use_hyperparameter_tuning": true,
      "use_model_validation": true,
      "use_advanced_models": false
    },
    "13": {
      "enabled": true,
      "optimize": true
    }
  },
  "parallel": {
    "enabled": true,
    "n_jobs": -1
  },
  "gpu": {
    "enabled": true,
    "device": 0
  }
}
```

### PFAZ Modüllerini Aktif/Pasif Etme

```python
# config.json'da
{
  "pfaz_config": {
    "2": {
      "enabled": true,                    # PFAZ 2'yi çalıştır
      "use_hyperparameter_tuning": true,  # Hyperparameter tuner aktif
      "use_model_validation": true,       # Model validator aktif
      "use_advanced_models": false        # Advanced models pasif
    }
  }
}
```

---

## Testler

### Integration Testlerini Çalıştırma

```bash
# Tüm testler
pytest tests/test_integration/ -v

# Sadece module import testleri
pytest tests/test_integration/test_module_imports.py -v

# Sadece root cleanup testleri
pytest tests/test_integration/test_root_cleanup.py -v
```

### Test Kategorileri

1. **Module Import Tests** - Tüm modüllerin import edilebilirliği
2. **Root Cleanup Tests** - Root dizini temizlik doğrulaması
3. **Unit Tests** - Birim testleri
4. **Smoke Tests** - Temel çalışabilirlik testleri

---

## Sorun Giderme

### Kütüphane Yükleme Sorunları

```bash
# Otomatik kütüphane kontrolü ve yükleme
python main.py --check-deps

# Manuel yükleme
pip install numpy pandas scikit-learn tensorflow xgboost matplotlib seaborn plotly openpyxl
```

### Optional Modül Kullanılamıyor

```python
# Availability flag'lerini kontrol et
import pfaz_modules.pfaz02_ai_training as pfaz02

print(f"Hyperparameter Tuner: {pfaz02.HYPERPARAMETER_TUNER_AVAILABLE}")
print(f"Model Validator: {pfaz02.MODEL_VALIDATOR_AVAILABLE}")
print(f"Advanced Models: {pfaz02.ADVANCED_MODELS_AVAILABLE}")
```

Eğer `False` ise, ilgili kütüphaneler yüklü değildir:
- `HYPERPARAMETER_TUNER_AVAILABLE=False` → `pip install optuna`
- `ADVANCED_MODELS_AVAILABLE=False` → `pip install torch`

### GPU Kullanımı

```python
# config.json'da
{
  "gpu": {
    "enabled": true,
    "device": 0  # GPU ID
  }
}
```

GPU yoksa otomatik olarak CPU'ya geçiş yapar.

---

## Utility Modüller

### Smart Cache

```python
from utils import SmartCache, SMART_CACHE_AVAILABLE

if SMART_CACHE_AVAILABLE:
    cache = SmartCache()
    cache.save('model_results', results)
    loaded = cache.load('model_results')
```

### Checkpoint Manager

```python
from utils import CheckpointManager, CHECKPOINT_MANAGER_AVAILABLE

if CHECKPOINT_MANAGER_AVAILABLE:
    manager = CheckpointManager()
    manager.save_checkpoint(model, optimizer, epoch=10)
    model, optimizer, epoch = manager.load_checkpoint()
```

---

## Scripts Dizini

### Utility Scriptler

- `scripts/check_pfaz_completeness.py` - PFAZ tamamlanma kontrolü
- `scripts/log_parser.py` - Log dosyası analizi
- `scripts/generate_sample_data.py` - Örnek veri üretimi

### Example Scriptler

- `scripts/examples/example_usage.py` - Temel kullanım örneği
- `scripts/examples/example_performance_pipeline.py` - Performans pipeline örneği

**Kullanım:**
```bash
python scripts/check_pfaz_completeness.py
python scripts/examples/example_usage.py
```

---

## İleri Seviye Kullanım

### Custom PFAZ Workflow

```python
from pfaz_modules.pfaz01_dataset_generation import DatasetGenerationPipelineV2
from pfaz_modules.pfaz02_ai_training import ParallelAITrainer
from pfaz_modules.pfaz08_visualization import VisualizationMasterSystem

# 1. Dataset oluştur
dataset_pipeline = DatasetGenerationPipelineV2(...)
dataset_results = dataset_pipeline.run_complete_pipeline()

# 2. Model eğit
trainer = ParallelAITrainer(...)
training_results = trainer.train_all_models_parallel()

# 3. Görselleştir
viz = VisualizationMasterSystem(...)
viz_results = viz.generate_all_visualizations()
```

### Paralel PFAZ Çalıştırma

```python
orchestrator = NuclearPhysicsAIOrchestrator()

# PFAZ 1-5 arası paralel çalıştır
orchestrator.run_all_pfaz(start_from=1, end_at=5)
```

---

## Kaynaklar

- **Detaylı Rapor:** `MODULE_ACTIVATION_REPORT.md`
- **Root Cleanup Plan:** `ROOT_CLEANUP_PLAN.md`
- **API Dokümantasyonu:** Her modülün kendi docstring'leri
- **Test Örnekleri:** `tests/` dizini

---

**Son Güncelleme:** 2025-11-23
**Versiyon:** 6.0.0 - Production Ready
