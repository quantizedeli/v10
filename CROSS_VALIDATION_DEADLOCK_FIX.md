# Cross-Validation Deadlock Fix

## Problem / Sorun

**Türkçe:** DNN_DNN_048 modelinde 5-fold cross-validation sırasında program takılıyor (deadlock). Paralel eğitim kullanıldığında program saatlerce ilerlemeden bekliyor.

**English:** The program gets stuck (deadlock) during 5-fold cross-validation in DNN_DNN_048. When parallel training is used, the program waits for hours without progress.

---

## Root Cause / Ana Sebep

**NESTED PARALLELIZATION = DEADLOCK**

```
ThreadPoolExecutor (n_workers)
  └─> train_single_job()
       └─> run_model_validation()
            └─> CrossValidationAnalyzer.run_cv()
                 └─> cross_validate(n_jobs=-1)  ⚠️ DEADLOCK!
```

**Türkçe:** İç içe paralelleşme problemi. `ThreadPoolExecutor` içinde her worker ayrı bir model eğitiyor. Eğitim sırasında cross-validation çağrılıyor ve CV de `n_jobs=-1` ile tüm CPU çekirdeklerini kullanmaya çalışıyor. Bu nested parallelization deadlock'a neden oluyor.

**English:** Nested parallelization problem. Each worker in `ThreadPoolExecutor` trains a separate model. During training, cross-validation is called, and CV tries to use all CPU cores with `n_jobs=-1`. This nested parallelization causes a deadlock.

---

## Solution / Çözüm

### 1. User Prompt / Kullanıcı Sorgusu

Program başlangıcında kullanıcıya soruluyor:

```
PARALEL EĞİTİM SEÇENEĞİ
========================
Cross-validation ile paralel eğitim kullanılırsa deadlock riski vardır.
Seçenekler:
  1) PARALEL EĞİTİM (hızlı ama CV sıralı olacak)
  2) SIRALI EĞİTİM (yavaş ama CV paralel olabilir)

Seçiminiz (1 veya 2):
```

**Seçenek 1 (ÖNERİLEN):**
- ✅ Model eğitimleri paralel çalışır (hızlı)
- ⚠️ Cross-validation sıralı çalışır (`n_jobs=1`)
- 💡 **Deadlock riski yok**

**Seçenek 2:**
- ⚠️ Model eğitimleri sıralı çalışır (yavaş)
- ✅ Cross-validation paralel çalışabilir (`n_jobs=-1`)
- 💡 CV daha hızlı ama genel eğitim daha yavaş

---

### 2. Code Changes / Kod Değişiklikleri

#### A. `model_validator.py`

```python
def run_cv(self, X, y, cv=5, scoring=None, return_train_score=True, n_jobs=1):
    """
    Args:
        n_jobs: Number of parallel jobs (default=1 to avoid nested parallelization deadlock)
                Use n_jobs=-1 only for sequential training mode
    """
    self.cv_results = cross_validate(
        self.model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
        n_jobs=n_jobs,  # FIXED: Use n_jobs parameter instead of hardcoded -1
        verbose=1
    )
```

#### B. `parallel_ai_trainer.py`

**1. Added `use_parallel_training` parameter:**

```python
def __init__(self, ..., use_parallel_training: bool = None):
    """
    Args:
        use_parallel_training: Use parallel training (None = prompt user, True = parallel, False = sequential)
    """
    self.use_parallel_training = use_parallel_training
```

**2. User prompt in `train_all_models_parallel()`:**

```python
if use_parallel is None:
    # Prompt user for choice
    while True:
        choice = input("\nSeçiminiz (1 veya 2): ").strip()
        if choice == '1':
            use_parallel = True
            break
        elif choice == '2':
            use_parallel = False
            break
```

**3. CV n_jobs control:**

```python
# CRITICAL: If parallel training is used, use n_jobs=1 for CV to avoid deadlock
cv_n_jobs = 1 if self.use_parallel_training else -1

cv_results = self.run_model_validation(
    model=trainer.model,
    model_name=f"{job.model_type}_{job.config['id']}",
    X=X_combined,
    y=y_combined,
    cv_folds=5,
    cv_n_jobs=cv_n_jobs
)
```

---

## Other Improvements / Diğer İyileştirmeler

### 1. ✅ aaa2.txt Exclusion Tracking (Already Implemented)

**Türkçe:** `excluded_nuclei_tracker.py` zaten mevcut ve aaa2.txt'den çıkarılan veriler nedenleriyle Excel'e kaydediliyor.

**English:** `excluded_nuclei_tracker.py` already exists and tracks excluded nuclei from aaa2.txt with reasons, saving to Excel.

**Location:** `pfaz_modules/pfaz01_dataset_generation/excluded_nuclei_tracker.py`

---

### 2. ✅ ANFIS .mat Dataset Export (Already Implemented)

**Türkçe:** ANFIS için datasetler zaten `.mat` formatında kaydediliyor.

**English:** ANFIS datasets are already exported in `.mat` format.

**Locations:**
- `pfaz_modules/pfaz01_dataset_generation/dataset_generation_pipeline_v2.py:885,929`
- `pfaz_modules/pfaz01_dataset_generation/dataset_generator.py:409,415,421`
- `pfaz_modules/pfaz03_anfis_training/anfis_parallel_trainer_v2.py:701-724`

**Files created:**
- `train.mat`
- `val.mat`
- `test.mat`

---

### 3. 🆕 NEW: Seed Tracking System

**Türkçe:** Her dataset ve model için kullanılan seed'leri takip eden yeni bir sistem eklendi.

**English:** Added a new seed tracking system that tracks all random seeds used for each dataset and model.

**New File:** `pfaz_modules/pfaz02_ai_training/seed_tracker.py`

**Features:**
- Tracks all seeds used in:
  - Dataset generation (sampling)
  - Model training (RF, XGBoost, DNN)
  - Cross-validation splits
  - Data shuffling
- Exports to Excel with multiple sheets:
  - All_Seeds
  - By_Operation
  - By_Dataset
  - By_Model
  - Detailed sheets for each operation

**Usage:**
```python
seed_tracker = SeedTracker(output_dir='seed_reports')
seed_tracker.add_seed(
    operation='model_training',
    seed=42,
    dataset_name='MM_75nuclei',
    model_name='RandomForest',
    config_id='RF_001'
)
seed_tracker.save_to_excel('seed_tracking_report.xlsx')
```

---

### 4. 🆕 NEW: GPU Configuration for DNN

**Türkçe:** DNN eğitimi için GPU konfigürasyonu eklendi.

**English:** Added GPU configuration for DNN training.

**Changes:**
```python
class DNNTrainer(BaseAITrainer):
    def __init__(self, config: Dict, output_dir: Path, gpu_enabled: bool = False):
        self.gpu_enabled = gpu_enabled
        self._configure_gpu()

    def _configure_gpu(self):
        """Configure GPU settings for TensorFlow"""
        if not self.gpu_enabled:
            # Disable GPU, use CPU only
            tf.config.set_visible_devices([], 'GPU')
            logger.info("[DNN] GPU DISABLED - Using CPU only")
        else:
            # Enable GPU with memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"[DNN] GPU ENABLED - Found {len(gpus)} GPU(s)")
```

**Note:** RF ve XGBoost sadece CPU kullanır (`n_jobs=-1`).

---

## Usage / Kullanım

### Option 1: Interactive (Recommended)

```python
from pfaz_modules.pfaz02_ai_training.parallel_ai_trainer import ParallelAITrainer

trainer = ParallelAITrainer(
    datasets_dir='datasets',
    models_dir='trained_models',
    use_model_validation=True  # Enable cross-validation
)

# User will be prompted to choose parallel vs sequential
results = trainer.train_all_models_parallel(n_configs=50)
```

### Option 2: Specify in Constructor

```python
# PARALLEL TRAINING (fast, but CV is sequential)
trainer = ParallelAITrainer(
    datasets_dir='datasets',
    models_dir='trained_models',
    use_parallel_training=True,  # Parallel training
    use_model_validation=True,
    gpu_enabled=True  # Enable GPU for DNN
)

results = trainer.train_all_models_parallel(n_configs=50)
```

```python
# SEQUENTIAL TRAINING (slow, but CV can be parallel)
trainer = ParallelAITrainer(
    datasets_dir='datasets',
    models_dir='trained_models',
    use_parallel_training=False,  # Sequential training
    use_model_validation=True,
    gpu_enabled=True
)

results = trainer.train_all_models_parallel(n_configs=50)
```

### Option 3: Specify in Method Call

```python
trainer = ParallelAITrainer(
    datasets_dir='datasets',
    models_dir='trained_models'
)

# Parallel
results = trainer.train_all_models_parallel(n_configs=50, use_parallel=True)

# Sequential
results = trainer.train_all_models_parallel(n_configs=50, use_parallel=False)
```

---

## Outputs / Çıktılar

### 1. Training Results
- `trained_models/training_summary.json`
- `trained_models/{dataset_name}/{model_type}/{config_id}/`
  - `model_{model_type}_{config_id}.pkl`
  - `metrics_{config_id}.json`
  - `cv_results_{config_id}.json`

### 2. Seed Tracking Reports (NEW)
- `trained_models/seed_reports/seed_tracking_report.xlsx`
- `trained_models/seed_reports/seed_tracking_report.json`

### 3. Validation Reports
- `trained_models/model_validation/{model_name}/`
  - `cv_summary.json`
  - `cv_detailed.csv`
  - `cv_boxplots.png`

---

## Summary of Changes / Değişiklik Özeti

| File | Change | Status |
|------|--------|--------|
| `model_validator.py` | Added `n_jobs` parameter to `run_cv()` | ✅ Fixed |
| `parallel_ai_trainer.py` | Added `use_parallel_training` parameter | ✅ Fixed |
| `parallel_ai_trainer.py` | Added user prompt for training mode | ✅ Fixed |
| `parallel_ai_trainer.py` | CV n_jobs control based on training mode | ✅ Fixed |
| `parallel_ai_trainer.py` | GPU configuration for DNN | ✅ Added |
| `seed_tracker.py` | New seed tracking system | ✅ New |
| `parallel_ai_trainer.py` | Integrated seed tracking | ✅ Added |

---

## Testing / Test

```bash
# Test seed tracker
python pfaz_modules/pfaz02_ai_training/seed_tracker.py

# Test parallel trainer with prompt
python pfaz_modules/pfaz02_ai_training/parallel_ai_trainer.py
```

---

## Author
**Date:** 2025-12-04
**Version:** 1.0.0 - Cross-Validation Deadlock Fix + Seed Tracking + GPU Config
