# CODING_RULES.md
## Claude Code — Bu Projede Tekrarlanmayacak Hatalar

> Bu dosya, 2026-04-30 QA denetiminde tespit edilen 17 kritik bug'dan öğrenilen
> kuralları içerir. **Her yeni kod yazılırken bu kurallar kontrol edilmelidir.**

---

## 1. PARALLELISM — Nested Parallelism Yasak

**Kural:** `ThreadPoolExecutor` veya `ProcessPoolExecutor` içinde çalışan her iş,
`n_jobs=-1` **KULLANAMAZ**.

```python
# YANLIS — ThreadPool + n_jobs=-1 = CPU bombası
RandomForestRegressor(n_jobs=-1)  # 22 worker × 24 thread = 528 thread

# DOGRU
def _inner_n_jobs() -> int:
    return 1 if os.environ.get('_PFAZ_PARALLEL_ACTIVE') == '1' else -1

RandomForestRegressor(n_jobs=_inner_n_jobs())
```

**Uygulama alanı:** `RandomForestRegressor`, `XGBRegressor`, `cross_val_score`,
`MultiOutputRegressor`, `LassoCV`, `learning_curve`, `validation_curve` içindeki
tüm `n_jobs` parametreleri.

**Env flag ayarı:** Dış `ThreadPoolExecutor` başlamadan önce
`os.environ['_PFAZ_PARALLEL_ACTIVE'] = '1'` set et, bittikten sonra `pop` et.

---

## 2. MEMORY CLEANUP — Her Training Job Sonrası Zorunlu

**Kural:** TensorFlow / PyTorch kullanan her training döngüsünde `finally` bloğu
ekle:

```python
finally:
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    import gc
    gc.collect()
```

**Neden:** `clear_session()` olmadan GPU memory her model için birikir. 50 config
× 8 dataset = 400 model sonunda RAM/VRAM tükenir, VS Code crash olur.

---

## 3. ENCODING — Her Python Dosyası UTF-8

**Kural:** Tüm Python dosyaları UTF-8 kaydedilmeli. Windows'ta CP1254 ile
kaydedilmiş dosya Linux HPC'de `UnicodeDecodeError` verir.

**Kontrol:**
```python
python -c "
import glob
for fp in glob.glob('pfaz_modules/*/__init__.py'):
    with open(fp, 'rb') as f:
        raw = f.read()
    raw.decode('utf-8')  # hata verirse CP1254'tür
"
```

**Kural:** `open()` çağrılarında her zaman `encoding='utf-8'` belirt:
```python
with open(path, encoding='utf-8') as f: ...
```

---

## 4. MULTIPROCESSING — Linux'ta spawn Zorunlu

**Kural:** `main.py` ve pipeline giriş noktalarının `if __name__ == '__main__':` bloğunda:

```python
if __name__ == '__main__':
    import multiprocessing as _mp
    try:
        _mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
```

**Neden:** Linux varsayılan `fork` ile TensorFlow birlikte kullanıldığında CUDA
context çocuk processlerde bozulur, deadlock oluşur.

---

## 5. INTERACTIVE INPUT — HPC'de input() Çağrısı Yasak

**Kural:** Her `input()` çağrısı öncesinde `sys.stdin.isatty()` kontrolü:

```python
if not sys.stdin.isatty() or os.environ.get('HPC_MODE'):
    # Non-interactive: env var'dan oku veya default kullan
    value = os.environ.get('MY_PARAM', 'default')
else:
    value = input("Parametre girin: ").strip()
```

**HPC env var'ları:** `HPC_MODE=1`, `PARALLEL_TRAINING=1`, `THESIS_AUTHOR`,
`THESIS_SUPERVISOR`, `THESIS_UNIVERSITY`, `THESIS_COMPILE_PDF`.

---

## 6. XGBoost GPU API — Versiyon Kontrolü Zorunlu

**Kural:** XGBoost 2.0'da `gpu_hist`, `gpu_id`, `gpu_predictor` kaldırıldı.
Her GPU param ayarında version check:

```python
import xgboost as xgb
xgb_version = tuple(int(x) for x in xgb.__version__.split('.')[:2])

if xgb_version >= (2, 0):
    gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
else:
    gpu_params = {'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor'}
```

---

## 7. IMPORT — Sınıf Adları Büyük/Küçük Harf Hassas

**Kural:** Import yazılırken kaynak dosyada sınıf adını `grep` ile doğrula:

```bash
grep -n "^class " pfaz_modules/pfaz03_anfis_training/matlab_anfis_trainer.py
# Çıktı: class MATLABAnfisTrainer:  ← tam bu ismi kullan
```

**Yanlış örnek:**
```python
from .matlab_anfis_trainer import MatlabANFISTrainer  # ← yanlış, hata silently swallowed
```

**Doğru:**
```python
from .matlab_anfis_trainer import MATLABAnfisTrainer  # ← grep ile doğrulandı
```

**Kural:** `except ImportError: pass` KULLANMA — her zaman loglama ekle:
```python
except ImportError as e:
    logging.warning(f"Module not available: {e}")
```

---

## 8. MODÜL YOLLARI — Önce Grep, Sonra Import Yaz

**Kural:** Bir sınıfı import etmeden önce nerede olduğunu `grep` ile doğrula:

```bash
grep -rn "class NuclearDataLoader" pfaz_modules/
# → pfaz_modules/pfaz01_dataset_generation/data_loader.py:39:class NuclearDataLoader:
```

`data_processing/`, `utils/`, `models/` gibi **gerçekte var olmayan** modül yolları
kullanma. Her import gerçek dizin yapısına dayanmalı.

---

## 9. BARE EXCEPT — Asla Kullanma

**Kural:** `except:` (exception tipi belirtmeden) kesinlikle kullanılmamalı.
`KeyboardInterrupt` ve `SystemExit` gibi kontrol exception'larını da yakalar,
Ctrl+C çalışmaz hale gelir.

```python
# YANLIS
try:
    something()
except:
    pass

# DOGRU
try:
    something()
except Exception as e:
    logger.warning(f"something() failed: {e}")
```

---

## 10. SCALER — Eğitim ile Birlikte Kaydet

**Kural:** Bir model `StandardScaler` veya başka bir scaler kullanıyorsa,
`save_model()` metodunda scaler da kaydedilmeli:

```python
def save_model(self, filepath: Path):
    import joblib
    joblib.dump(self.model, filepath)
    if hasattr(self, 'scaler'):
        joblib.dump(self.scaler, filepath.parent / 'scaler.pkl')
```

**Neden:** Scaler kaydedilmezse inference zamanında yeni veriler yanlış scale
edilir, tahminler hatalı olur.

---

## 11. NAMESPACE — Aynı İsimde İki Sınıf Yasak

**Kural:** Aynı `__init__.py`'ye import edilen iki farklı sınıf aynı ismi taşıyamaz.
Framework farkını isme yansıt:

```python
# YANLIS — gpu_optimization.py ve advanced_models.py'de ikisi de GPUOptimizer
# DOGRU
class TensorFlowGPUOptimizer: ...   # gpu_optimization.py
class PyTorchGPUOptimizer: ...       # advanced_models.py
```

---

## 12. RANDOM SEED — Config-Bazlı Deterministik Seed

**Kural:** Sabit `random_seed = 42` KULLANMA. Her konfigürasyon kendi
deterministik seed'ini türetmeli:

```python
import hashlib
config_id = str(self.config.get('id', 'default'))
random_seed = self.config.get(
    'random_seed',
    int(hashlib.md5(config_id.encode()).hexdigest()[:8], 16) % (2**31)
)
```

**Neden:** 50 "farklı konfigürasyon" aynı seed ile başlatılırsa gerçekte
pseudo-replikasyon olur, varyans tahmini hatalı.

---

## 13. HPC / AutoInstaller — Çevre Kontrolü

**Kural:** `subprocess.check_call([sys.executable, '-m', 'pip', 'install', ...])` çağrısından
önce HPC kontrolü:

```python
if 'SLURM_JOB_ID' in os.environ or os.environ.get('HPC_MODE'):
    print("HPC ortaminda pip install yapilamaz.")
    sys.exit(1)
if not sys.stdin.isatty():
    print("Non-interactive: pip install atlanıyor.")
    sys.exit(1)
```

---

## 14. CHECKPOINT/RESUME — Long Jobs İçin Zorunlu

**Kural:** 12+ saatlik training job'larında her birim iş (model eğitimi) kendi
checkpoint'ini kaydetmeli. Job restart'ta tamamlananlar atlanabilmeli:

```python
checkpoint_file = job.output_dir / 'completed.json'
if checkpoint_file.exists():
    with open(checkpoint_file, encoding='utf-8') as f:
        saved = json.load(f)
    return TrainingResult(**saved)  # Skip, already done

# ... eğit ...

with open(checkpoint_file, 'w', encoding='utf-8') as f:
    json.dump({...result fields...}, f)
```

---

## 15. MATLAB ENGINE — Context Manager Kullan

**Kural:** `MATLABAnfisTrainer` her zaman context manager ile kullan:

```python
# DOGRU
with MATLABAnfisTrainer() as trainer:
    results = trainer.train_anfis(X_train, y_train, X_val, y_val)
# engine otomatik kapanır

# YANLIS — __del__ zamanlamasına güvenmek
trainer = MATLABAnfisTrainer()
trainer.initialize_engine()
results = trainer.train_anfis(...)
# Eğer exception fırlarsa engine açık kalır → lisans takılır
```

---

## 16. REQUIREMENTS.TXT — HPC'de Sorunlu Paketler

**Kural:** Şu paketler `requirements.txt`'e eklenmemeli veya comment'te olmalı:

| Paket | Sorun |
|-------|-------|
| `auto-sklearn` | Python 3.11+ build fail (swig/SMAC) |
| `pickle5` | Python 3.8+'da built-in, gereksiz |
| `tpot` (bazı sürümler) | auto-sklearn çeker |

HPC için `requirements-hpc.txt` kullan.

---

## 17. LOG MESAJLARI — ASCII Only

**Kural:** Log mesajlarında emoji/Unicode karakter **KULLANMA**:
- Windows konsolunda `UnicodeEncodeError`
- HPC log dosyasında bozuk karakter

```python
# YANLIS
logger.info("✅ Model eğitimi tamamlandı")
logger.info("🚀 Paralel eğitim başlıyor")

# DOGRU
logger.info("[OK] Model egitimi tamamlandi")
logger.info("[START] Paralel egitim basliyor")
```

---

*Son güncelleme: 2026-04-30*
