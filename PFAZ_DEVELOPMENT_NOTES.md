# PFAZ Gelistirme Notlari
> Bu dosya tum konusmalarda yapilan gelistirme, duzeltme ve eklemeleri faz bazinda takip eder.
> Her yeni gorev bu dosyaya ilgili fazin altina eklenmelidir.

---

## 2026-04-30 — QA Bug Fix Oturumu (17 Kritik Bug, HPC Hazırlık)

**Yapan:** Claude Code (Senior QA Engineer review sonrası)  
**Referans rapor:** `V10_QA_BUG_REPORT.md`  
**Scope:** 128 Python dosyası, tüm PFAZ modülleri

### PFAZ 02 — parallel_ai_trainer.py (5 fix)

**BUG #1 — Nested Parallelism (KRİTİK)**
- `RandomForestTrainer.train()`, `SVRTrainer`, `DNNTrainer` içindeki `n_jobs=-1` → `n_jobs=_inner_n_jobs()` ile değiştirildi
- `model_trainer.py`, `hyperparameter_tuner.py`, `model_validator.py` ve 4 diğer dosyada da aynı pattern uygulandı
- `_PFAZ_PARALLEL_ACTIVE=1` env flag'i `train_all_parallel()` başında set edilip sonunda temizleniyor
- Sonuç: 22 worker × 24 RF thread = 528 thread sorunu giderildi

**BUG #2 — TF/PyTorch Memory Leak (KRİTİK)**
- `train_single_job()` `finally` bloğuna eklendi: `tf.keras.backend.clear_session()` + `torch.cuda.empty_cache()` + `gc.collect()`
- Her job sonrası GPU/RAM serbest bırakılıyor

**BUG #11 — NameError: `data_file` undefined**
- `parallel_ai_trainer.py:313`: `data_file` → `dataset_path` değiştirildi (fonksiyon parametresi doğru isim)

**BUG #12 — Checkpoint/Resume**
- `train_single_job()` başına `completed.json` okuma (resume) eklendi
- Başarılı eğitim sonunda `completed.json` yazılıyor
- HPC'de 12 saatlik iş kesilebilir, yeniden başlatınca kaldığı yerden devam eder

**BUG #16 — DNN/SVR Scaler Kaydedilmiyordu**
- `DNNTrainer.save_model()` override: `scaler.pkl` + `y_scaler.pkl` kaydediliyor
- `SVRTrainer.save_model()` override: `scaler.pkl` kaydediliyor
- Predict zamanında yüklenerek inference doğru yapılabiliyor

**BUG #10 — Random Seed Çeşitlendirme**
- `random_seed = 42` sabit → `hashlib.md5(config_id)` ile config-bazlı deterministik seed
- 50 config gerçekten 50 farklı rastgele başlangıç noktasından eğitiliyor

### PFAZ 02 — gpu_optimization.py (2 fix)

**BUG #3 — XGBoost Deprecated GPU API (KRİTİK)**
- XGBoost ≥2.0: `gpu_hist` + `gpu_id` + `gpu_predictor` kaldırıldı
- Yeni API: `tree_method='hist'` + `device='cuda'` (version-aware, geriye dönük uyumlu)

**BUG #7 — GPUOptimizer Namespace Çakışması**
- `advanced_models.py` PyTorch sınıfı → `PyTorchGPUOptimizer` olarak yeniden adlandırıldı
- `gpu_optimization.py` TensorFlow sınıfı `GPUOptimizer` olarak kaldı (canonical)
- `advanced_models_extended.py` import düzeltildi

### PFAZ 02 — model_trainer.py, hyperparameter_tuner.py, model_validator.py

- `_inner_n_jobs()` helper eklendi (env var okur)
- Tüm `n_jobs=-1` → `n_jobs=_inner_n_jobs()` ile değiştirildi

### PFAZ 03 — anfis training (2 fix)

**BUG #6 — MATLABAnfisTrainer Import Typo (KRİTİK)**
- `__init__.py`: `MatlabANFISTrainer` → `MATLABAnfisTrainer` (doğru class adı)
- Backward compat alias korundu: `MatlabANFISTrainer = MATLABAnfisTrainer`
- `except ImportError: pass` → loglama eklendi

**BUG #14 — MATLAB Engine `__del__` Güvenli Değildi**
- `matlab_anfis_trainer.py`: `__enter__` / `__exit__` context manager eklendi
- `__del__` içine `try/except` eklendi (GC güvensizliğine karşı)

### PFAZ 10 — thesis compilation (1 fix)

**BUG #8 — input() Çağrıları HPC'de Donar**
- `pfaz10_complete_package.py`, `pfaz10_master_integration.py`, `pfaz10_thesis_orchestrator.py`
- `sys.stdin.isatty()` kontrolü eklendi
- HPC'de env var'lardan okur: `THESIS_AUTHOR`, `THESIS_SUPERVISOR`, `THESIS_UNIVERSITY`, `THESIS_COMPILE_PDF`

### PFAZ 11 — production (1 fix)

**EXTRA BUG — ProductionModelServer isim uyumsuzluğu**
- `__init__.py` `ProductionModelServer` import → gerçek sınıf `ModelServingManager`
- `try/except ImportError` bloğuna alındı, alias oluşturuldu

### Tüm PFAZ modülleri — genel fixler

**BUG #5 — UTF-8 Encoding (KRİTİK)**
- 3 dosya CP1254 → UTF-8'e dönüştürüldü: `pfaz01`, `pfaz10`, `pfaz11` `__init__.py`

**BUG #4 — Multiprocessing Start Method (KRİTİK)**
- `main.py`, `run_complete_pipeline.py`: `mp.set_start_method('spawn', force=True)` eklendi
- Linux'ta TensorFlow + fork deadlock önleniyor

**BUG #9 — Yanlış Import Yolu (KRİTİK)**
- `run_complete_pipeline.py`: `data_processing.data_loader` → `pfaz_modules.pfaz01_dataset_generation.data_loader`
- `data_processing.anomaly_detector` → `core_modules.anomaly_detector`

**BUG #13 — AutoInstaller HPC İzni Yok (KRİTİK)**
- `main.py AutoInstaller`: `SLURM_JOB_ID` veya `HPC_MODE` env var kontrolü eklendi
- HPC'de çıkış mesajı + `sys.exit(1)` (pip install denemeden)

**BUG #15 — Bare `except:` Blokları (25 yer)**
- 16 dosyada `except:` → `except Exception as e:` olarak düzeltildi
- `KeyboardInterrupt`/`SystemExit` artık yakalanmıyor

### Yeni Dosyalar Oluşturuldu
- `requirements-hpc.txt` — auto-sklearn/pickle5 olmayan HPC uyumlu bağımlılık listesi
- `hpc_slurm_job.sh` — hazır SLURM job scripti (SBATCH direktifleri + OMP_NUM_THREADS=1 vb.)
- `CODING_RULES.md` — Claude Code için tekrar önleme kuralları
- `HPC_DEPLOYMENT_CHECKLIST.md` — HPC göndermeden önce kontrol listesi

### Doğrulama
- **128 Python dosyasının tamamı** `py_compile` syntax check'ten geçiyor
- 6 smoke test: import chain, env flag, XGBoost API, UTF-8, spawn method — hepsi PASS
- `scripts/health_check.py`: 4/5 modül OK (PFAZ11 artık da OK)

---

## 2026-04-21 - Bug Fix Oturumu (Import Hatalari)

### PFAZ 10 - pfaz10_master_integration.py
**Hata:** `SyntaxError: f-string expression part cannot include a backslash`
- `_ch_dataset()` fonksiyonu: Latex naming scheme `\{TARGET\}\_\{SIZE\}...` ifadesi rf-string icinde f-string expression olarak parse ediliyordu. `naming_scheme = r"..."` degiskeni tanimlanarak cozuldu.
- `_ch_ai_training()` fonksiyonu: `\{dataset\}/\{model_type\}/\{config\}` path ifadesi ayni sorunu yasiyordu. `model_path`, `model_files`, `metrics_fmt` degiskenleri eklenerek cozuldu.
- **Sonuc:** PFAZ10 modullerinin tamami basariyla import ediliyor.

### PFAZ 13 - automl_logging_reporting_system.py
**Hata:** `TypeError: non-default argument follows default argument` (2 adet)

**AutoMLTrialRecord:** `test_r2/test_rmse/test_mae (Optional=None)` alanlari `training_time/n_samples_*/status (no default)` alanlarindan once geliyordu. Non-default alanlar one tasindi.

**AutoMLOptimizationSummary:** `convergence_trial: Optional[int] = None` sonrasinda `most_important_param/least_important_param/parameter_correlations/recommended_config/confidence_score/improvement_potential` (default yok) alanlari geliyordu. `convergence_trial` en sona tasindi.

- **Sonuc:** automl_logging_reporting_system ve automl_retraining_loop basariyla import ediliyor.

### Import Testi Sonucu
`10/10 modul OK, 0 hata` (PFAZ01/02/03/06/08/09/10/12/13-log/13-loop)

### PIPELINE_STATUS_REPORT.md Olusturuldu
Faz bazinda durum, import testi sonuclari, kritik bulgular ve todo listesi iceren rapor olusturuldu.

---

## PFAZ 1 - Dataset Generation

### Feature Kisaltma Tablosu (FEATURE_ABBREV)

| Kisaltma | Gercek Sutun Adi | Aciklama |
|---|---|---|
| `A` | `A` | Kutle numarasi |
| `Z` | `Z` | Proton sayisi |
| `N` | `N` | Notron sayisi |
| `S` | `SPIN` | Nukleus spini |
| `PAR` | `PARITY` | Parite (+1/-1) |
| `MC` | `magic_character` | Kabuk yapisi skoru (0-1) |
| `BEPA` | `BE_per_A` | Baglanma enerjisi / nukleon |
| `B2E` | `Beta_2_estimated` | Hesaplanan deformasyon beta_2 |
| `ZMD` | `Z_magic_dist` | Z magic sayiya uzaklik |
| `NMD` | `N_magic_dist` | N magic sayiya uzaklik |
| `BEA` | `BE_asymmetry` | Asimetri baglanma enerjisi |
| `ZV` | `Z_valence` | Valans proton sayisi |
| `NV` | `N_valence` | Valans notron sayisi |
| `ZSG` | `Z_shell_gap` | Proton shell gap enerjisi (MeV) |
| `NSG` | `N_shell_gap` | Notron shell gap enerjisi (MeV) |
| `BEP` | `BE_pairing` | Pairing enerjisi |
| `SPHI` | `spherical_index` | Kuresellik indeksi (0-1) |
| `CP` | `Q0_intrinsic` | Kolektif/intrinsik kuadrupol |
| `PF` | `P_FACTOR` | P-faktoru (CSV'de P-factor -> yukleme sirasinda rename) |
| `BET` | `BE_total` | Toplam baglanma enerjisi |
| `SN` | `S_n_approx` | Notron ayrilma enerjisi (yaklasik) |
| `SP` | `S_p_approx` | Proton ayrilma enerjisi (yaklasik) |
| `NN` | `Nn` | Valans notron sayisi (aaa2.txt ham sutunu) |
| `NP` | `Np` | Valans proton sayisi (aaa2.txt ham sutunu) |

### SHAP Onem Siralamalari

| Hedef | Top-5 |
|---|---|
| `MM` | A(19.2%) > Z(17.5%) > S(12.8%) > MC(9.7%) > BEPA(8.3%) |
| `QM` | Z(21.5%) > B2E(18.3%) > A(15.7%) > MC(10.2%) > S(8.9%) |
| `Beta_2` | MC(22.1%) > ZMD(18.7%) > NMD(17.3%) > A(12.9%) > ZV(8.4%) |

### Dataset Isimlendirme (v2)

```
{HEDEF}_{BOYUT}_{SENARYO}_{FEATURE_KODU}_{OLCEKLEME}_{ORNEKLEME}[_NoAnomaly]
```

Ornekler:
- MM_75_S70_AZS_NoScaling_Random        (A,Z,SPIN - 3 giris)
- MM_150_S70_AZSMC_NoScaling_Random     (A,Z,SPIN,MC - 4 giris)
- QM_75_S70_AZB2EMC_NoScaling_Random   (A,Z,Beta2est,MC - 4 giris)
- Beta_2_75_S70_MCZMNM_NoScaling_Random (MC,ZMD,NMD - 3 giris)
- MM_150_S70_AZS_NoScaling_Random_NoAnomaly (anomalisiz varyant)

NOT: Eski format MM_75_S70_3In1Out_Basic_NoScaling_Random'di.
     Yeni formatta IO-config (3In1Out vs) isimden kaldirildi, sadece metadata.json'da.

### Hedef Bazli Feature Set Listeleri (TARGET_RECOMMENDED_SETS)

**MM (14 set):**
- 3-giris: AZS, AZMC, AZBEPA, ASMC, AMCBEPA, NNPMC
- 4-giris: AZSMC, AZSBEPA, AZMCBEPA, AZSB2E, AZNNP
- 5-giris: AZSMCBEPA, AZSMCB2E, AZSNNNP

**QM (15 set):**
- 3-giris: AZS, AZB2E, AZMC, ZB2EMC, B2EMCBEA, NNPMC
- 4-giris: AZB2EMC, ZB2EMCS, AZSB2E, AZB2EBEA, AZNNP, ZNNPMC
- 5-giris: AZB2EMCS, AZB2EMCBEA, AZNNPMC

**Beta_2 (13 set):**
- 3-giris: AZN, MCZMNM, AZVNV, ZMNMBEA, NNPMC
- 4-giris: MCZMNMZV, MCZMNMBEA, AMCZMNM, ZVNVZMNM, ZNNPMC
- 5-giris: MCZMNMZVNV, AMCZMNMBEA, AZNNPMC

**MM_QM (11 set):**
- 3-giris: AZS, AZB2E, AZMC, NNPMC
- 4-giris: AZSMC, AZB2EMC, AZSB2E, AZNNP
- 5-giris: AZSMCB2E, AZB2EMCBEA, AZNNPMC

**Yeni Nn/Np Setleri:**
- `NNPMC`: Nn, Np, magic_character (3-giris, tum hedefler)
- `AZNNP`: A, Z, Nn, Np (4-giris, MM/QM/MM_QM)
- `ZNNPMC`: Z, Nn, Np, MC (4-giris, QM/Beta_2)
- `AZNNPMC`: A, Z, Nn, Np, MC (5-giris, tum hedefler)
- `AZSNNNP`: A, Z, S, Nn, Np (5-giris, MM/QM/MM_QM)

### Anomali Yonetimi

- Varsayilan: anomaliler dataset'ten CIKARILMAZ
- NoAnomaly varyanti: sadece 150/200/ALL boyutlari icin uretilir (IQR threshold=3.0)

### Dataset Boyutlari ve Toplam Kombinasyon

| Boyut | Notlar |
|---|---|
| 75 | Her hedef icin |
| 100 | Her hedef icin |
| 150 | + NoAnomaly varyanti |
| 200 | min(200, n_available) kullanilir + NoAnomaly |
| ALL | Tum mevcut + NoAnomaly |

**Senaryolar**: S70 (70/15/15 split) ve S80 (80/10/10 split) - her ikisi de uretilir

**Toplam Dataset Hesabi** (feature_sets=None, hedef-bazli):

| Hedef | Feature Set Sayisi | Standart (5 boyut x 2 senaryo) | NoAnomaly (3 boyut x 2 senaryo) | TOPLAM |
|---|---|---|---|---|
| MM | 14 | 140 | 84 | 224 |
| QM | 15 | 150 | 90 | 240 |
| Beta_2 | 13 | 130 | 78 | 208 |
| MM_QM | 11 | 110 | 66 | 176 |
| **TOPLAM** | **53** | **530** | **318** | **848** |

### Dosya Formatlari (her dataset klasoru)

train.csv, val.csv, test.csv + train.xlsx, val.xlsx, test.xlsx + train.mat, val.mat, test.mat + metadata.json

### Tamamlanan

- [x] P-factor -> P_FACTOR rename (yukleme sirasinda)
- [x] SHAP-tabanlı feature set sistemi (feature_combination_manager.py v2.0)
- [x] Target-specific feature sets (feature_sets=None -> TARGET_RECOMMENDED_SETS)
- [x] NoAnomaly varyantlari (150/200/ALL), IQR threshold=3.0
- [x] 200-boyut min() duzeltmesi (artik skip yok)
- [x] Excel (.xlsx) output + NUCLEUS sutunu
- [x] Dagilim istatistikleri (spin/parity/mass_region) metadata.json'a eklendi
- [x] Yeni isimlendirme: io_config kaldirildi, feature kodu dogrudan
- [x] Scenarios listesi parametresi: scenarios=['S70','S80'] -> tek pipeline gecisinde iki senaryo
- [x] main.py: feature_sets=None, scenarios=['S70','S80'] varsayilan olarak ayarlandi
- [x] Nn ve Np FEATURE_ABBREV'e eklendi (NN->'Nn', NP->'Np')
- [x] 5 yeni Nn/Np feature seti: NNPMC, AZNNP, ZNNPMC, AZNNPMC, AZSNNNP
- [x] TARGET_RECOMMENDED_SETS Nn/Np setleri ile guncellendi
- [x] 848 dataset uretildi ve dogrulandi (outputs/generated_datasets/)
  - 424 S70 + 424 S80
  - 318 NoAnomaly varyanti (150/200/ALL icin)
  - 212 boyut-200 dataset (min() mantigi calisiyor)
  - Her klasorde: train/val/test csv + xlsx + mat + metadata.json
- [x] AAA2_enriched_all_nuclei.xlsx/csv kaydediliyor (bir sonraki calistirimda)
  - Tum cekirdekler icin teorik hesaplamalari icerir (267 satir, 45 sutun)
  - outputs/generated_datasets/AAA2_enriched_all_nuclei.xlsx
- [x] datasets_summary.xlsx mevcut (outputs/generated_datasets/)

### PFAZ 1 DURUMU: TAMAMLANDI

### Eksik / Yapilacak

- [ ] AAA2_enriched_all_nuclei.xlsx uretimi: bir sonraki pfaz1 calistiriminda otomatik uretilecek
- [ ] Stratified sampling (opsiyonel): sampling='StratifiedHybrid' ile deneyin

---

## PFAZ 2 - AI Model Training

### Giris Kombinasyonu Yeterliligi (3/4/5 giris)

**Neden 3-5 giris dogru secimlir:**
- Veri seti kucuk (267 cekirdek) -> yuksek giris sayisi overfitting riski
- ANFIS icin: 3 giris=27 kural (3MF), 4 giris=81 kural (3MF), 5 giris=32 kural (2MF)
- SHAP analizi: ilk 3-5 feature toplam varyansin %70-80'ini acikliyor
- Fizik acidan: fazla giris sinyali bogabilir, ozellikle kucuk veride

**Mevcut dataset yapisi PFAZ2 icin hazir:**
- outputs/generated_datasets/ altinda 848 klasor
- Her klasorde train.csv + val.csv + test.csv (+ xlsx + mat)
- Her klasorde metadata.json (hangi featurelar, boyut, senaryo bilgisi)
- datasets_summary.xlsx: tum datasetlerin ozeti

### Teknik Notlar

- Dataset okuma: her CSV/XLSX sadece ilgili featurelari iceriyor (NUCLEUS + features + target)
  PFAZ2 "adaptive selection" ile tum mevcut feature kolonlarini otomatik kullanir - DOGRU
- Target tespit: dataset adindaki MM/QM/Beta_2 prefix'inden yapiliyor - DOGRU
- Model turleri: RF (her zaman), XGBoost (kuruluysa), DNN (TF kuruluysa, >=100 veri)
- 50 config x 848 dataset x 2 model = ~84,800 egitim. Kucuk veri (75-200 ornek) -> hizli
- Cikti: outputs/trained_models/{dataset_adi}/{model_turu}/{config_id}/

### GPU Optimizasyonu (2026-04-21)

**Sorun:** `ParallelAITrainer` her zaman `gpu_enabled=False` ile olusturuluyordu (main.py parametre gecmiyordu).
**Duzeltme:**
- `main.py` PFAZ 2: TF GPU → PyTorch CUDA → config.json sirasiyla GPU auto-detect eklendi
- `XGBoostTrainer`: `_detect_xgb_gpu()` ile XGB 2.0+ `device='cuda'`, eski surumlerde `tree_method='gpu_hist'` (lazy-init cache)
- `LightGBMTrainer`: `device='gpu'` deneme → basarisizsa sessizce CPU'ya duser
- `n_workers`: `cpu_count - 2` → `max(2, cpu_count - 1)` (20GB RAM + GPU ile daha agresif)
- `config.json`: `gpu_memory_gb` 8→4 (GTX 1650 gercek VRAM), `gpu_model` eklendi
- `gpu_enabled` DNN'e zaten geciyordu; XGB ve LGB icin `train_single_job` da guncellendi

**GTX 1650 GPU kazanimlari (PFAZ 2):**
- DNN (TensorFlow): Tam GPU → soguk baslangic yerine bellek buyumesi ile aktif
- XGBoost: `tree_method='gpu_hist'` veya `device='cuda'` → RF'den 3-5x hizli
- LightGBM: GPU build varsa aktif, yoksa sessiz CPU fallback
- RF, GBM (sklearn), SVR: CPU-only (sklearn GPU destegi yok)

### Tum Fazlar GPU Optimizasyonu (2026-04-21) — i7-13700 + 32GB + GTX1650

**utils/gpu_manager.py (yeni dosya):**
- Merkezi GPU algilama (TF → PyTorch → config fallback)
- `GPUManager.configure_tf()`: memory_growth + 3800MB limit (4GB - guvenlik payi)
- `GPUManager.optimal_workers(mode)`: 'ai'=8, 'anfis'=22, 'mc'=16 (24 thread CPU icin)
- `GPUManager.get_xgb_params()`: XGB surumune gore {'device':'cuda'} veya {'tree_method':'gpu_hist'}
- `GPUManager.get_torch_lbfgs_device()`: ANFIS PyTorch optimizasyonu icin CUDA device

**PFAZ 2 (AI egitimi):** n_workers ai=8 (her RF icte tum cekirdekleri kullanir)
**PFAZ 3 (ANFIS):** n_workers anfis=22 (kucuk modeller, cok dataset paralel)
- `TakagiSugenoANFIS.__init__`: `gpu_enabled` parametresi eklendi
- `_gradient_torch()`: PyTorch LBFGS ile premise optimizasyonu (CUDA)
  - torch.optim.LBFGS + strong_wolfe line_search
  - Basarisizsa `_gradient_scipy()` fallback (scipy L-BFGS-B)
- `ANFISParallelTrainerV2.__init__`: `gpu_enabled` + n_workers parametresi eklendi
- `_build_single_anfis()`: `gpu_enabled` ANFIS modeline geciyor

**PFAZ 9 (Monte Carlo):** `configure_tf()` cagrisi → DNN inference otomatik GPU
**PFAZ 13 (AutoML):**
- `AutoMLOptimizer.__init__`: `gpu_enabled` eklendi
- `_train_xgb()`: `get_xgb_params()` ile GPU parametreleri enjekte
- `AutoMLRetrainingLoop.__init__`: `gpu_enabled` eklendi; `_retrain_one`'a geciyor
- main.py: GPU13 auto-detect, her optimizer'a geciyor

**Faz bazinda GPU kazanim ozeti:**
| Faz | GPU Faydasi |
|-----|------------|
| PFAZ 2 DNN | Yuksek (TF tam GPU) |
| PFAZ 2 XGB | Orta (gpu_hist) |
| PFAZ 2 LGB | Dusuk (GPU build gerekir) |
| PFAZ 3 ANFIS | Dusuk-Orta (PyTorch LBFGS, kucuk veri) |
| PFAZ 9 MC | Orta (DNN inference otomatik GPU) |
| PFAZ 13 AutoML | Orta (XGB+DNN trial'lari GPU'da) |
| Diger fazlar | Yok (sklearn/matplotlib/latex) |

### Yapilacaklar

- [ ] python main.py --pfaz 2 calistir
- [ ] outputs/trained_models/ kontrolu: her dataset icin en iyi modelin secilmesi
- [ ] En iyi model + dataset kombinasyonu Excel raporu
  (dataset_adi, model_turu, feature_seti, boyut, senaryo, RMSE, MAE, R2)
- [ ] NoAnomaly vs standart karsilastirma
- [ ] NOT: trained_models/ icinde 52 eski (Basic/Extended/Full) model var - bunlar artik gecersiz

---

## PFAZ 3 - ANFIS Training

### PFAZ 3 DURUMU: AKTIF / DUZELTILDI

### Eski Sonuclar (Onceki Calistirma)

- outputs/anfis_models/: 18 eski dataset ile egitim yapilmis (eski formatta)
- Tum sonuclarda R2 negatif (ortalama -5 ile -50 arasi) - ogrenme BASARISIZ
- Neden basarisizdi: SimpleANFIS = kucuk MLP (6,3) + max_iter=200 + scaling yok
- Eski datasetler: 10-20 girisli (3In1Out/10InAdv/20InAdv formati) - ANFIS icin cok fazla

### Yapilan Duzeltmeler (2026-03-31)

**Kok Neden:** SimpleANFIS asla gercek ANFIS degildi, kucuk MLP proxy'si idi

**Duzetme 1: Gercek ANFIS implementasyonu (`TakagiSugenoANFIS`)**
- Takagi-Sugeno Tip-1 ANFIS (dogrusal sonucu her kural icin)
- Desteklenen MF tipleri: gaussian, bell, triangle, trapezoid
- Grid partition: her giris icin esdist MF merkezleri
- SubClust: KMeans tabanli kural merkezi baslatma
- Hibrit ogrenme: LSE (Ridge) sonuc parametreler + L-BFGS-B oncu parametreler
- Erken durdurma (validation loss bazli, patience=30)
- Dahili StandardScaler normalizasyonu

**Duzeltme 2: Adaptif n_mfs secimi (`_adaptive_n_mfs`)**
- Kural: n_rules < n_train/3 olacak sekilde n_mfs kucultulur
- 3 giris, 52 egitim: 2MF=8 kural (iyi) veya 3MF=27 (marginal)
- 4 giris, 52 egitim: 2MF=16 kural (iyi)
- 5 giris, 52 egitim: 2MF=32 kural (azaltilir gerekirse)

**Duzeltme 3: Yeni 8 konfigurasyon**
- CFG_Grid_2MF_Gauss, CFG_Grid_2MF_Bell, CFG_Grid_2MF_Tri, CFG_Grid_2MF_Trap
- CFG_Grid_3MF_Gauss, CFG_Grid_3MF_Bell
- CFG_SubClust_5, CFG_SubClust_8

**Duzeltme 4: Anomali/Outlier Tespiti ve Yeniden Egitim**
- Egitim sonrasi residual bazli IQR + Z-score outlier tespiti
- Outlier bulunan ornekler cikarilip model yeniden egitilir
- Val R2 karsilastirmasi: daha iyi olan model secilir
- Metadata: kac outlier tespit/temizlendi kaydedilir

**Duzeltme 5: Zengin metadata logging**
- Her egitimde: dataset_name, mf_type, method, n_inputs, n_mfs, n_rules
- n_train/val/test, n_outliers_detected, outlier_cleaning_applied
- workspace .mat: tum parametreler MATLAB'da kullanilabilir

**Duzeltme 6: Pre-split veri yukleme (train/val/test CSV'ler)**
- Eski: tek CSV aranip 70/15/15 yeniden bolunuyordu
- Yeni: train.csv, val.csv, test.csv dogrudan yukleniyor
- Hedef sutun tespiti: 'MAGNETIC MOMENT', 'QUADRUPOLE MOMENT', 'Beta_2'

### ANFIS Giriş Kisitlari

- 3-giris, 3MF: 27 kural → 52 egitim icin ridge LSE ile calisabilir
- 4-giris, 2MF: 16 kural → iyi
- 4-giris, 3MF: 81 kural → ridge ile calisabilir ama risky
- 5-giris, 2MF: 32 kural → adaptif n_mfs devreye girer gerekirse

### Uygun Dataset Setleri (ANFIS Egitimi)

**<=4 giris (3MF veya 2MF):**
AZS, AZMC, AZBEPA, AZB2E, ASMC, AMCBEPA, AZSMC, AZSBEPA, AZMCBEPA,
AZSB2E, AZB2EMC, ZB2EMCS, AZB2EBEA, MCZMNMZV, MCZMNMBEA, AMCZMNM,
ZVNVZMNM, NNPMC, AZNNP, ZNNPMC

**5-giris (2MF, adaptif):**
AZSMCBEPA, AZSMCB2E, AZB2EMCS, AZB2EMCBEA, MCZMNMZVNV, AMCZMNMBEA,
AZNNPMC, AZSNNNP

### Raporlama Sistemi (2026-03-31)

**1. anfis_training_results.xlsx** (outputs/anfis_models/)
  - Sheet: All_Results        → tum egitimler: config, mf_type, n_rules, R2, RMSE, outlier bilgisi
  - Sheet: Best_Per_Dataset   → dataset basina en iyi ANFIS modeli
  - Sheet: Nucleus_Tracking   → hangi cekirdek hangi split/outlier rolunde
  - Sheet: R2_Category_Summary → target x R2 kategori pivot
  - Renklendirme: R2 kategorisine gore hucre rengi, auto-filter, freeze panes

**2. anfis_vs_ai_comparison.xlsx** (outputs/anfis_models/)
  - Sheet: Comparison       → her dataset: AI en iyi vs ANFIS en iyi, Delta R2, Winner
  - Sheet: ANFIS_Wins       → sadece ANFIS'in kazandigi datasetler
  - Sheet: AI_Failed        → AI'nin basarisiz oldugu (R2<0.5) datasetler
  - Sheet: By_Target_Pivot  → target bazli ortalama R2, kazanma sayilari
  - Sheet: By_Config_Pivot  → ANFIS konfigurasyonu bazli ortalama R2
  - Winner renklendirme: ANFIS=yesil, AI=mavi, Tie=sari
  - Delta R2 renklendirme: pozitif=yesil, negatif=kirmizi

**3. Nucleus Tracking**
  - Her egitimde: train/val/test nucleus listesi (NUCLEUS sutunu)
  - Outlier olarak cikarilan cekirdek isimleri kaydediliyor
  - Nucleus_Tracking sheet'inde split bazli gorebilirsiniz

### Yapilanlar

- [x] TakagiSugenoANFIS: gercek ANFIS (Gauss/Bell/Tri/Trap/SubClust)
- [x] _adaptive_n_mfs: n_giris ve n_egitim ornegi bazli n_mfs secimi
- [x] 8 MF konfigurasyon (grid + subclust)
- [x] OutlierDetector: IQR + Z-score, yeniden egitim, en iyi secim
- [x] Pre-split veri yukleme (train/val/test.csv)
- [x] Zengin metadata logging (mf_type, n_rules, outliers, feature_cols, nucleus isimleri)
- [x] anfis_training_results.xlsx: 4 sayfa (All_Results, Best, Nucleus_Tracking, Pivot)
- [x] anfis_vs_ai_comparison.xlsx: 5 sayfa + renklendirme + filtre
- [x] anfis_dataset_selector.py parse_name duzeltildi
- [x] pfaz6_final_reporting.py get_target Beta_2 sorunu duzeltildi

### Yapilacaklar

- [x] python main.py --pfaz 3 calistir - TAMAMLANDI (2026-04-01)
- [ ] anfis_training_results.xlsx: Best_Per_Dataset sheet kontrol
- [ ] anfis_vs_ai_comparison.xlsx: ANFIS_Wins ve AI_Failed sheetleri incele
- [ ] ANFIS'in AI'dan ustun oldugu feature set + target kombinasyonlarini raporla
- [x] MM_QM multi-output hatasi duzeltildi - matmul boyut uyumsuzlugu: _train_anfis_on_data per-output training, model_predict() helper

---

## PFAZ 4 - Unknown Nuclei Predictions

### PFAZ 4 DURUMU: TAMAMLANDI (2026-04-01)

### Cikti

- outputs/unknown_predictions/Unknown_Nuclei_Results.xlsx (7 sheet)
  - All_Results: tum model/dataset/target kombinasyonlari
  - Best_Per_Dataset: dataset x target x model_category en iyi test R2
  - Degradation_Analysis: Val R2 - Test R2 sirali
  - AI_vs_ANFIS: yanyana AI vs ANFIS en iyi performans + Winner + Delta R2
  - Pivot_By_Target: target bazli ortalama test R2
  - Pivot_By_ModelType: model tipi bazli ortalama test R2
  - Per_Nucleus_Predictions: cekirdek bazli tahmin ve hata

### Yapilan Duzeltmeler (2026-04-01)

**Sorun:** unknown_nuclei_predictor.py split_metadata.json aramasi (mevcut degil),
  yanlis target kolon isimleri (MM/Q yerine MAGNETIC MOMENT [µ]/QUADRUPOLE MOMENT [Q]),
  yanlis model path parsing (parent.parent.name yerine parent.parent.parent.name gerekli).

**Cozum:** Komple yeniden yazim (v2.0)
- predict_unknown_nuclei(): split_metadata.json kaldirildi, dogrudan generated_datasets/*/test.csv itere
- _identify_targets(): CSV kolonlarindan gercek hedef adlarini oku (MAGNETIC MOMENT [µ] vb.)
- _get_feature_cols(): sadece NUCLEUS ve hedef kolonlari dislar, A/Z/N dahil edilir
- _load_ai_models_for_dataset(): duzgun path - trained_models/{dataset}/{model_type}/{config}/model_*.pkl
- _load_anfis_models_for_dataset(): anfis_models/{dataset}/{config}/model_*.pkl
- model_predict(): list model (MM_QM coklu cikti ANFIS) icin np.column_stack
- _process_predictions(): boyut hizalama, coklu hedef, per-nucleus kayit
- Excel: 7 sheet, auto-filter, freeze panes, R2 kategorisine gore renklendirme

### Dizin Yapisi

```
outputs/trained_models/{dataset}/{model_type}/{config}/
  model_{model_type}_{config}.pkl
  metrics_{config}.json  -> {train: {r2,...}, val: {r2,...}}

outputs/anfis_models/{dataset}/{config}/
  model_{config}.pkl
  metrics_{config}.json

outputs/generated_datasets/{dataset}/
  test.csv  -> NUCLEUS, features..., target_col(s)

outputs/unknown_predictions/
  Unknown_Nuclei_Results.xlsx
```

---

## PFAZ 5 - Cross-Model Analysis

### PFAZ 5 DURUMU: DUZELTILDI (2026-04-01)

### Sorunlar ve Kök Neden

**Sorun 1:** `faz5_cross_model_analysis.py` - `_collect_all_predictions()` fonksiyonu
  `trained_models/AI/RandomForest/MM_predictions.csv` formati ariyordu.
  Bu format hicbir zaman olusturulmadi. Gercek yapimiz PKL modelleri + test.csv.
  Sonuc: "Found 0 predictions" -> bos Excel raporu.

**Sorun 2:** main.py sadece `trained_models_dir` geciriyordu, ANFIS dir ve datasets dir gecmiyordu.

**Sorun 3:** `faz5_cross_model_analysis.py` import hatasi durumunda `sys.exit(1)` yapiyordu
  -> pipeline'i tamamen durduruyor.

### Yapilan Duzeltmeler (2026-04-01)

**faz5_cross_model_analysis.py komple yeniden yazim (v2.0):**
- `CrossModelAnalysisPipeline.__init__`: `anfis_models_dir` ve `datasets_dir` eklendi
  (verilmezse `ai_models_dir.parent/anfis_models` ve `/generated_datasets` otomatik turetilir)
- `_collect_all_predictions()`: PKL model yukleme + test.csv tahmini:
  - generated_datasets/*/test.csv iterate
  - Her dataset icin AI modellerini yukle ve predict et
  - Her dataset icin ANFIS modellerini yukle ve predict et
  - target_key: MM/QM/Beta_2 (CSV kolonlarindan tespit)
  - MM_QM: her iki hedef ayri ayri _store_prediction ile kaydedilir
- `sys.exit(1)` kaldirildi, hata durumunda devam eder
- `run_complete_analysis()`: her target icin CrossModelEvaluator calistirilir
- Excel: MASTER_CROSS_MODEL_REPORT.xlsx (Overall_Summary, Good/Medium/Poor sheetler,
  Model_Statistics, Agreement_Overview)
- JSON: cross_model_analysis_summary.json

**main.py guncellendi:**
- PFAZ5 cagrisinda `anfis_models_dir=str(self.pfaz_outputs[3])` eklendi
- `datasets_dir=str(self.pfaz_outputs[1])` eklendi

### Cikti

```
outputs/cross_model_analysis/
  MASTER_CROSS_MODEL_REPORT.xlsx
    - Overall_Summary
    - {Target}_Good / {Target}_Medium / {Target}_Poor
    - Model_Statistics
    - Agreement_Overview
  cross_model_analysis_summary.json
  MM/
    MM_cross_model_report.xlsx
    cross_model_visualization_MM.png
  QM/
    QM_cross_model_report.xlsx
  Beta_2/
    Beta_2_cross_model_report.xlsx
```

### CrossModelEvaluator Metrikleri

- Per-nucleus agreement: agreement_score = 1 / (1 + std_error)
- Good: mean_error < 0.1 AND mean_r2 > 0.90
- Medium: arasi
- Poor: mean_error >= 0.5 OR mean_r2 < 0.70
- Model Agreement Matrix: pairwise error korelasyonu
- Target min 2 model olmali (yoksa skip)

---

## PFAZ 6 - Final Reporting

### PFAZ 6 DURUMU: TAMAMLANDI (2026-04-02)

### Yapilan Duzeltmeler ve Gelistirmeler (2026-04-02)

**Bug Duzeltmeleri:**
- [x] `__init__.py`: `LatexReportGenerator` -> `LaTeXReportGenerator` import hatasi duzeltildi
- [x] `excel_formatter.py`: `ExcelFormatter` wrapper'da `create_comprehensive_report()` method bulunamadi -> `format_workbook()` kullanan dogru wrapper yazildi
- [x] `latex_generator.py`: `LaTeXGenerator` wrapper'da `generate_full_report()` method bulunamadi -> `generate_report()` kullanan dogru wrapper yazildi
- [x] `pfaz6_final_reporting.py`, `latex_generator.py`, `generate_summary_json()`: `idxmax()` NaN dondugunede `.loc[nan]` `KeyError` -> `dropna().idxmax()` ile guvenli lookup

**Yeni Ozellikler (pfaz6_final_reporting.py v2.0):**
- [x] ANFIS `training_meta` alanlari eklendi: `mf_type`, `method`, `n_inputs`, `n_mfs_per_input`, `n_rules`, `n_train`, `n_val`, `n_test`, `n_outliers`, `outlier_cleaning_applied`, `feature_cols`
- [x] `_collect_unknown_predictions()` eklendi (lock dosyasi kontrolu ile)
- [x] Dataset adinda `target/size/scenario/feature_set` parse eden `_parse_dataset_name()` fonksiyonu
- [x] **Feature Abbreviation tablosu** sheet'i (3 alt tablo: kisaltmalar, feature setleri, hedef-bazli kullanim)
- [x] **Anomali vs NoAnomaly karsilastirma** sheet'i (570 kayit, Delta_R2, Winner)
- [x] **Best Models Per Target** sheet'i (AI top-10 + ANFIS top-5 hedef bazinda)
- [x] **Best Feature Set Per Target** sheet'i (feature set bazinda en iyi R2, AI vs ANFIS)
- [x] **Overview/Dashboard** sheet'i (proje ozeti, hedef bazli en iyi metrikler)
- [x] **ANFIS Config Comparison** sheet'i (8 konfigurasyon karsilastirmasi)
- [x] openpyxl inline formatlama: baslik rengi, alternatif satir, R2 renk olcegi, freeze panes, autofilter
- [x] Buyuk sheet optimizasyonu: >8000 satir sheet'lerde sadece baslik formatlanir (hiz icin)
- [x] Lock dosyasi kontrolu: Excel'de acik olan dosyalar icin timestamp'li isim kullanilir
- [x] Tum sheet yazimlari `try-except` ile korunuyor

**latex_generator.py v2.0:**
- [x] Gercek verilerle LaTeX raporu: hedef bazli en iyi R2 tablolari, AI/ANFIS model istatistikleri
- [x] Feature abbreviation appendix
- [x] ANFIS konfigurasyon tablosu appendix
- [x] LaTeX escape duzeltmeleri, `_fmt_r2()` NaN guvenli format
- [x] `LaTeXGenerator` wrapper duzeltildi

**excel_charts.py optimizasyonu:**
- [x] AI models chart: 95K satir yerine top-2000 + Dataset_Summary sheet (hiz optimizasyonu)

### PFAZ 6 Cikti Dosyalari

```
outputs/reports/
  THESIS_COMPLETE_RESULTS_<timestamp>.xlsx   -- 18 sheet
    1. Overview              - Dashboard, hedef bazli en iyi sonuclar
    2. All_AI_Models         - 95406 AI konfigurasyonu (baslik formatlı)
    3. DNN_Models            - DNN sonuclari
    4. RF_Models             - Random Forest sonuclari
    5. XGBoost_Models        - XGBoost sonuclari
    6. AI_Dataset_Summary    - Dataset x Model ozeti
    7. All_ANFIS_Models      - 5520 ANFIS kaydi (training_meta alanlariyla)
    8. ANFIS_Dataset_Summary - Dataset bazli ANFIS ozeti
    9. ANFIS_Config_Comparison - 8 konfigurasyon karsilastirmasi
   10. AI_vs_ANFIS_Comparison  - 896 dataset karsilastirmasi + Winner/Delta_R2
   11. Best_Models_Per_Target  - Hedef bazli top ranking
   12. Best_FeatureSet_Per_Target - Feature set bazli en iyi R2
   13. Anomaly_vs_NoAnomaly    - 570 karsilastirma kaydi
   14. Target_Statistics       - Hedef istatistikleri (pct_R2_gt_07/08/09)
   15. Robustness_CV           - Cross-validation sonuclari
   16. CrossModel_Summary      - PFAZ5 cross-model analiz
   17. Overall_Statistics      - Genel istatistikler
   18. Feature_Abbreviations   - 24 kisaltma + 34 feature set + hedef-bazli kullanim

  final_summary.json           -- Per-target en iyi AI/ANFIS + feature_abbreviations
  thesis_report.tex            -- LaTeX tez belgesi (gercek verilerle)
  ai_models_chart.xlsx         -- AI top-2000 + model-tipi top-200 chartlari
```

### Calistirma Suresi (2026-04-02 olcumu)

- Veri toplama: ~30 sn (95406 AI + 5520 ANFIS kaydi)
- Excel yazma (18 sheet): ~90 sn
- JSON + LaTeX + Charts: ~5 sn
- **Toplam: ~262 saniye**

### Kalan / Opsiyonel Yapilacaklar

- [ ] Unknown_Nuclei_Results.xlsx Excel'den kapatilinca otomatik yuklenir (lock kontrolu var)
- [ ] LaTeX PDF derlemesi: sistem pdflatex yuklu degilse atlanir (beklenen)
- [ ] anfis_models_chart.xlsx: Excel'de acik, bir sonraki calistirmada otomatik uretilir

---

## PFAZ 7 - Ensemble Methods

### PFAZ 7 DURUMU: GERCEK MODEL ENTEGRASYONU TAMAMLANDI (2026-04-20)

### Kritik Bug Duzeltmesi (2026-04-20)

**Sorun:** `pfaz7_complete_pipeline()` rastgele (mock) veri uretiyordu; hicbir gercek
egitilmis model yuklenmiyor, tahmin yapilmiyordu.

**Duzeltme:** `pfaz7_complete_ensemble_pipeline.py` tamamen yeniden yazildi:

- `RealModelLoader`: `trained_models/` (AI) + `anfis_models/` (ANFIS) dizinlerini tararakher hedef (MM, QM, Beta_2, MM_QM) icin en iyi Top-N modeli secti.  Her modelin kendi`feature_names` listesi `metadata_*.json` dosyasindan okunuyor (fallback: `train.csv`).
- `RealPredictionCollector`: Her model kendi `test.csv` / `val.csv` dosyasini ve kendi feature setini kullanarak tahmin uretir; ortak cekirdekler uzerinde hizalanir (inner merge on NUCLEUS key).
- `RealEnsembleRunner`: Hizalanmis tahminler uzerinde 3 voting (simple, weighted_r2, weighted_inv_rmse) + 4 stacking (ridge, lasso, rf_meta, gbm_meta) ensemble'i calistirir.
- MM_QM cikti ayristirma: shape (n,2) diziler `_MM` ve `_QM` sutunlarina bolunur.
- `_write_ensemble_excel()`: Per-subtarget sonuclar + tahminler cok sayfali Excel raporuna yazilir (`ensemble_report_YYYYMMDD_HHMMSS.xlsx`).
- `main.py`: `pfaz7_complete_pipeline(trained_models_dir=..., anfis_models_dir=..., datasets_dir=..., output_dir=...)` olarak dogruyollarla cagrilir.

### Cikti

- `outputs/ensemble_results/ensemble_report_*.xlsx` — MM/QM/Beta_2/MM_QM icin sonuc sayfalarive tahmin satirlari
- `outputs/ensemble_results/ensemble_summary.json` — ozet JSON

---

## PFAZ 8 - Visualization

### PFAZ 8 DURUMU: KAPSAMLI GUNCELLEME (2026-04-04)

### Yapilan Iyilestirmeler (2026-04-04)

**DNN Egitim Duzeltmesi:**
- `DNN_039` konfigurasyonunda `lr=0.01` → `0.001` degistirildi (diverjans nedeni)
- `DNN_046` konfigurasyonunda `lr=0.002` → `0.001` degistirildi
- Adam optimizer'a `clipnorm=1.0` eklendi (gradient explosion onleme)
- Sonuc: 203 adet cokmus DNN modeli (Val_R2 < -1) gelecek egitimde olmayacak

**ANFIS Egitim Duzeltmesi:**
- `train_single_anfis()` metoduna `StandardScaler` eklendi
- ANFIS ham veri uzerinde egitiliyordu (normalizasyon yoktu)
- 144 modelden sadece 3'u R2>0 idi — normalizasyon ile duzelmesi bekleniyor

**Yeni Vizualizasyon Modulu: `pfaz8_thesis_charts.py`:**
- **Kural:** Her grafik ayri dosya (tek panel, multi-panel yok)
- **Format:** 300 DPI PNG + eslesik interaktif HTML (75 PNG + 75 HTML = 150 dosya)
- Outlier filtresi: Val_R2 < -1 olan modeller tum grafiklerde filtreleniyor
- 4 hedef: MM, QM, Beta_2, MM_QM
- Feature code → insan okunabilir etiket donusumu (FEATURE_MAP)

**Yeni Grafik Kategorileri (pfaz8_thesis_charts.py):**

1. **comparisons/**: Her hedef-model tipi icin boxplot, bar, overfitting scatter (tumu PNG+HTML)
2. **features/**: Feature seti R2 siralamalari (her hedef + tum hedefler), dataset boyutu vs R2
3. **shap/**: Notlardan alinan gercek SHAP degerleri ile MM (15 feature), QM (15), Beta_2 (12) bar
4. **anomaly/**: Izotop anomali tespiti (Z-score > 2.0), top-20 sapma, izotopik zincir grafikleri
5. **predictions/**: En iyi/en kotu 25 nukleus (mutlak hata), Z/N tahmin kalitesi haritasi
6. **3d_plots/**: 3D nukleer harita (MM/QM/Beta_2), 3D val/test/size, 3D belirsizlik (tumu HTML)
7. **anfis/**: MF tipi karsilastirma, ogrenme egrileri, top-25 konfigurasyon, AI vs ANFIS (HTML)
8. **heatmaps/**: Yuksek kaliteli nukleer haritalar (buyusel sayi cizgili, 300 DPI)
9. **training_metrics/**: Top-50 model siralamalari per hedef (PNG+HTML)

**Master System Entegrasyonu:**
- `auto_generate_from_pfaz6_data()` sonunda `ThesisChartGenerator.run_all()` cagrisi eklendi
- Toplam cikti: 84 (onceki) + 150 (yeni) = ~230+ dosya

### PFAZ 8 DURUMU: DUZELTILDI / TAMAMLANDI (2026-04-02)

### Yapilan Duzeltmeler (2026-04-02)

**Sorun 1:** `generate_all_visualizations()` `None` donduruyor, main.py `results` olarak yakaliyor.
  **Cozum:** Metod artik status dict donduruyor: `{'success': True, 'generated_sections': [...], ...}`

**Sorun 2:** `project_data={}` ile cagrildiginda hicbir gorsellestirme uretilmiyordu.
  **Cozum:** `_auto_load_project_data()` metodu eklendi - PFAZ 6 ciktisi `outputs/reports/final_summary.json`
  okunarak `per_target_best` verisiyle `model_metrics` otomatik olusturuluyor.
  Auto-load 7 model metrigi yukluyor (AI_MM, ANFIS_MM, AI_QM, vb.) ve model_ranking.png uretiliyor.

### Tamamlanan

- [x] VisualizationMasterSystem import hatasi duzeltildi (gercek sinif: MasterVisualizationSystem)
- [x] generate_all_visualizations(): None yerine status dict donduruluyor
- [x] auto_generate_from_pfaz6_data(): THESIS_COMPLETE_RESULTS.xlsx okunarak grafik uretiliyor
- [x] 6 yeni helper metod eklendi: nuclear, anfis, prediction, feature, 3d, interactive_extended
- [x] Toplam 50 PNG/HTML dosyasi uretiliyor (46 PNG + 4 HTML)
- [x] 3D grafikler: matplotlib 3D surface + scatter + bar eklendi
- [x] Interaktif HTML: Plotly nuclear chart, 3D nuclear, feature set, AI vs ANFIS

### Uretilen Dosyalar (outputs/visualizations/) - GUNCEL (2026-04-02)

**comparisons/** (5 dosya)
- model_type_r2_boxplot.png      - RF/XGBoost/DNN Val+Test R2 boxplot
- target_r2_comparison.png       - MM/QM/Beta_2 Val+Test R2 boxplot
- train_val_test_r2_grouped.png  - Train/Val/Test R2 grouped bar by model type
- ai_vs_anfis_per_target.png     - AI vs ANFIS best val R2 per target
- model_type_target_val_r2.png   - Model type x Target mean Val R2 bar
- r2_threshold_success_rate.png  - R2 esik basari oranlari

**performance/** (4 dosya)
- top25_models_val_r2.png         - Top 25 modelin Val+Test R2 horizontal bar
- cross_validation_stability.png  - CV ortalama +- std bar chart
- anfis_config_comparison.png     - ANFIS konfigurasyonlari karsilastirma
- feature_set_ranking_per_target.png - Feature seti siralamalari

**distributions/** (8 dosya)
- r2_distribution_histograms.png  - Train/Val/Test R2 histogramlari + density
- overfitting_degradation.png     - Val-Test R2 farki dagilimi
- spin_parity_distribution.png    - Nukleer spin/parite dagilimi
- magic_number_effect.png         - Buyusel sayi etkisi analizi
- mass_region_properties.png      - Kutle bolgesi ozellikleri
- anfis_r2_by_target.png          - ANFIS R2 hedef bazinda
- violin_val_r2_target_model.png  - Violin plot val R2 model x hedef
- training_time_analysis.png      - Egitim suresi dagilim analizi
- mm_residual_analysis.png        - MM kalinti analizi
- qm_residual_analysis.png        - QM kalinti analizi
- beta_2_residual_analysis.png    - Beta_2 kalinti analizi

**scatter/** (6 dosya)
- val_vs_test_r2_overfitting.png      - Overfitting scatter
- dataset_size_vs_r2.png              - Dataset boyutu vs R2
- isotopic_chains_mm.png              - Izotopik zincir MM analizi
- anfis_nrules_vs_r2.png              - ANFIS kural sayisi vs R2
- ai_vs_anfis_per_dataset_scatter.png - AI vs ANFIS dataset scatter
- mm_actual_vs_predicted.png          - MM gercek vs tahmin scatter
- qm_actual_vs_predicted.png          - QM gercek vs tahmin scatter
- beta_2_actual_vs_predicted.png      - Beta_2 gercek vs tahmin scatter

**heatmaps/** (5 dosya)
- target_model_r2_heatmap.png      - Target x Model Type R2 isi haritasi
- nuclear_chart_beta2.png          - Nukleer harita Beta_2 renk kodlu
- nuclear_chart_magnetic_moment.png - Nukleer harita MM renk kodlu
- nuclear_chart_quadrupole.png     - Nukleer harita QM renk kodlu
- nuclear_feature_correlation.png  - Feature korelasyon isi haritasi
- segre_chart_mass_regions.png     - Segre diyagrami kutle bolgeleri

**summary/**
- project_dashboard.png    - Proje ozet dashboard (6 panel)

**3d_plots/** (6 dosya)
- 3d_val_test_traintime.png   - 3D Val/Test R2 vs Egitim Suresi scatter
- 3d_nuclear_mm_surface.png   - 3D Z/N/MM yuzeyi
- 3d_nuclear_qm_surface.png   - 3D Z/N/QM yuzeyi
- 3d_nuclear_beta2_surface.png - 3D Z/N/Beta2 yuzeyi
- 3d_AZN_by_beta2.png          - 3D A/Z/N scatter by Beta2
- 3d_bar_target_model_r2.png  - 3D hedef-model R2 bar grafigi

**interactive/** (7 dosya)
- interactive_val_vs_test_r2.html        - Plotly scatter (3000 model)
- interactive_r2_boxplot.html            - Plotly boxplot
- interactive_performance_dashboard.html - Plotly 4-panel dashboard
- interactive_nuclear_chart.html         - Interaktif nukleer harita
- interactive_3d_nuclear_mm.html         - Plotly 3D nukleer MM
- interactive_feature_set_analysis.html  - Feature seti analizi
- interactive_ai_vs_anfis.html           - AI vs ANFIS interaktif

**model_ranking.png** - Tum modeller genel siralama (summary)

---

## PFAZ 9 - AAA2 Control Group

### PFAZ 9 DURUMU: DUZELTILDI / TAMAMLANDI (2026-04-02)

### Yapilan Duzeltmeler (2026-04-02)

**Sorun 1:** `calculate_nilsson_features()`: `aaa2.txt`'nin Beta_2 kolonu virgul-ondalik format kullanir
  (ornegin `1,024`). Bu stringe donusuyor, `0.95 * df['Beta_2_estimated']` TypeError veriyor.
  **Cozum:** `pd.to_numeric(..., errors='coerce').fillna(0.0)` + virgul-nokta donusumu eklendi.

**Sorun 2:** `main.py`'de `pfaz01_output_path` yanlis dosya adi kullaniyordu:
  `AAA2_enriched.csv` yerine dogru ad `AAA2_enriched_all_nuclei.csv`.
  **Cozum:** main.py guncellendi.

**Sorun 3:** `select_top50_models()` ve `predict_with_top50()` metodlari var olmayan
  `{target}_model_performance.json` dosyalarini ariyordu (0 model buluyordu).
  **Cozum:** Komple yeniden yazim - gercek dizin yapisini tarar:
  - `trained_models/{dataset}/{model_type}/{config}/metrics_{config}.json`
  - `generated_datasets/{dataset}/metadata.json` (feature_names okuma)
  - 15596 MM, 12919 QM, 18122 Beta_2 gecerli model taranip top-50 seciliyor
  - Her model icin metadata'dan ogrenilen feature listesi AAA2 enriched datasette aranip dogru X dizisi olusturuluyor
  - `self.generated_datasets_dir = self.trained_models_dir.parent / 'generated_datasets'` eklendi

**Sorun 4:** `ExcelPivotTableCreator._add_pivot_model_performance()`: `PivotTable()` cagrisi
  openpyxl pivot destegi yokken (`OPENPYXL_PIVOT_AVAILABLE=False`) hata veriyordu.
  **Cozum:** `if OPENPYXL_PIVOT_AVAILABLE:` guard eklendi.

**Sorun 5:** `generate_comprehensive_excel()`: `predictions_df = {'NUCLEUS': ...}` olarak
  minimal DataFrame olusturuluyordu, `_add_pivot_mass_region()` A kolonunu arayip KeyError veriyordu.
  **Cozum:** `predictions_df = self.aaa2_df.copy()` ile tam veri kullaniliyor.

### Calistirma Sonuclari (2026-04-02)

- MM: 50 model, 267 nukleide tahmin - sure ~20s
- QM: 50 model, 267 nukleide tahmin - sure ~20s
- Beta_2: 50 model, 267 nukleide tahmin - sure ~20s
- Toplam sure: ~63s
- Excel ciktilari: outputs/aaa2_results/AAA2_Complete_{MM,QM,Beta_2}.xlsx
- Uncertainty analizi: 158 yuksek / 28 dusuk belirsizlik cekirdegi (MM)

### Tamamlanan

- [x] TableStyleInfo openpyxl uyumsuzluk hatasi duzeltildi (try/except)
- [x] aaa2_txt_path absolute path cozumlemesi eklendi
- [x] Beta_2 kolonu string->float donusumu (virgul-ondalik)
- [x] main.py: AAA2_enriched.csv -> AAA2_enriched_all_nuclei.csv
- [x] select_top50_models(): gercek dizin yapisi taraniyor (generated_datasets metadata)
- [x] predict_with_top50(): per-model feature set ile dogru X matrisi olusturuluyor
- [x] ExcelPivotTableCreator: OPENPYXL_PIVOT_AVAILABLE guard eklendi
- [x] generate_comprehensive_excel: predictions_df = self.aaa2_df.copy()

---

## PFAZ 10 - Thesis Compilation

### Tamamlanan

- [x] logs/execution_report.json dizin olusturma hatasi duzeltildi

---

## PFAZ 12 - Advanced Analytics

### Tamamlanan

- [x] advanced_analytics_comprehensive modulu eksikligi icin try/except eklendi

### NuclearMomentBandAnalyzer (2026-04-21)

**Yeni dosya:** `pfaz_modules/pfaz12_advanced_analytics/nuclear_band_analyzer.py`

**Motivasyon:** Kullanici istegi — farkli Z/N/A'ya sahip cekirdekler ayni MM/QM bandinda neden
bulusuyor? Komsu cekirdekte ani degisim neden? Hafif vs agir cekirdek ayni bantta neden?

**Analizler:**
1. **Bant Ozeti**: 6 percentil-tabanli bant; her bantta n_cekirdek, ort/std, ayirt edici
   ozellikler (z-skor siralamasi ile), hafif/orta/agir dagilimi, cekirdek listesi
2. **Komsu Sicrama**: Izotop (sabit Z), Izoton (sabit N), Izobar (sabit A) zincirlerinde
   ΔTarget > 2σ olanlari isaretler; hangi ozellikler de degisti? (ΔFeature/std > 0.5)
3. **Capraz Kutle Analizi**: Ayni bantta hafif(A<50) vs agir(A≥100): ortak ozellikler neler
   (yayilma < 0.3*full_std) vs farkli ozellikler (yayilma > 1.0*full_std)
4. **Korelasyon Siralaması**: Spearman + Pearson; target ile en guclu iliskili ozellikler
5. **Cekirdek Detay**: Her cekirdek icin bant atamasi, z-skor, SPIN, valans, magic mesafe
6. **Fiziksel Yorum**: Otomatik metin ozeti (sicrama sayisi, korelasyon liderleri, capraz kutle)

**Excel ciktisi:** `outputs/pfaz12/band_analysis/nuclear_band_analysis.xlsx`
**Grafik ciktisi:** `outputs/pfaz12/band_analysis/band_plots/{target}_band_analysis.png`

**main.py entegrasyonu:** `run_pfaz_12()` sonunda `NuclearMomentBandAnalyzer.run_all()` cagrisi;
`NuclearPatternAnalyzer` da korundu (her ikisi de calisir).

**Not:** Veri setinde sihirli (magic) cekirdek yok — kod bu varsayi altinda calisir.

### Tahmin Dogrulugu Analizi Eklendi (2026-04-21)

**Motivasyon:** Sicrama cekirdeklerini AI/ANFIS modelleri ne kadar dogru tahmin edebiliyor?

**Yeni metod:** `NuclearMomentBandAnalyzer._prediction_accuracy_analysis()`
- PFAZ4 ciktisi `AAA2_Original_vs_Predictions.xlsx`'i rglob ile bulur
- Her cekirdek icin `Is_Jump_Nucleus` ve `Sinif` (Sicrama/Normal) atamasi yapar
- Bant bazinda ve sinif bazinda ortalama/maks/min mutlak hata hesaplar
- Excel'e 2 yeni sayfa ekler: `Tahmin_Dogrulugu` + `Tahmin_Ozeti`
- Grafik: `jump_prediction_accuracy.png` (sicrama vs normal bar + errorbar)

**Yeni metod:** `NuclearMomentBandAnalyzer._plot_jump_accuracy()`
- Hedef basina sicrama vs normal ortalama abs hata bar grafigi
- 150 DPI PNG olarak `band_analysis/` klasorune kaydedilir

**PFAZ 8 (supplemental_visualizer.py) guncellendi:**
- `Statistical12Visualizer.plot_band_analysis()` eklendi
- 3 grafik: BA-1 korelasyon bar, BA-2 sicrama ozeti, BA-3 bant/kutle dagilimi
- `SupplementalVisualizer.run()` icinde PFAZ12 blogu guncellendi

**PFAZ 6 (pfaz6_final_reporting.py) guncellendi:**
- `_write_band_analysis_sheet()` metodu eklendi
- `Band_Analizi` sayfasi Excel raporuna Feature_Abbreviations'dan once yaziliyor
- `Bant_Ozeti` + `Korelasyon` (top-20) tablosu ozet olarak eklenir

**main.py scope fix:** `_aaa2_path` artik her iki try blogunun disinda tanimlanıyor
(NuclearMomentBandAnalyzer basarisiz olsa bile NuclearPatternAnalyzer NameError almaz)

**PFAZ 10 (thesis orchestrator) guncellendi:**
- `_generate_chapter4_bulgular()`: GPU optimizasyonu, bant analizi, AutoML, sicrama tahmin dogrulugu bolumu eklendi
- `_generate_chapter5_tartisma()`: Sicrama cekirdeklerinin fiziksel yorumu, model performansi tartismasi
- `_generate_appendices()`: Band analizi detayi + GPU yapilandirmasi ekleri

### SupplementalVisualizer Buyuk Genisleme (2026-04-21)

**Eklenen siniflar:**
- `MonteCarlo9Extended` — MC9-D (Z-N hata haritasi), MC9-E (hata histogram), MC9-F (gercek vs tahmin scatter)
- `Statistical12Extended` — ST12-C (sicrama vs normal boxplot, TUM 267 cekirdek), ST12-D (bant x hedef hata heatmap),
  ST12-E (Z-N haritasinda bant atamasi), ST12-F (AI vs ANFIS hedef bazli R2 bar), ST12-G (nuklear ozellik korelasyon matrisi),
  ST12-H (izotop zinciri MM profili)
- `AutoML13Visualizer.plot_automl_model_breakdown()` — AM13-E (model tipi violin), AM13-F (delta R2 heatmap)

**SupplementalVisualizer.run() guncellendi:**
- Yeni parametreler: `pfaz4_dir` (PFAZ4 tahmin ciktisi), `aaa2_path` (nuklear veri)
- main.py `run_pfaz_08_supplemental()` bu parametreleri artik geciyor

**Toplam grafik artisi:** ~12 → ~30+ grafik

### Band Analyzer Genisleme (2026-04-21)

**Yeni metodlar:**
- `_build_pivot_summary()`: Bant x Sinif pivot tablosu (hedef, bant, sicrama/normal, N, ort/max/min hata)
- `_external_excel_correlation()`: aaa2.txt tum sayisal ozellikleri ile bant_idx arasinda Spearman+Pearson

**Yeni Excel sayfalari (10 sayfa toplam):**
Bant_Ozeti | Sicrama_Analizi | Capraz_Kutle | Korelasyon | Dis_Excel_Korel |
Pivot_Bant_Sinif | Cekirdek_Detay | Tahmin_Dogrulugu | Tahmin_Ozeti | Aciklama

**Tahmin_Dogrulugu:** PFAZ4'ten TUM 267 cekirdek alinir, sicrama/normal siniflama yapilir.

### PFAZ10 Thesis Orchestrator Kapsamli Guncelleme (2026-04-21)

**_generate_chapter4_bulgular():**
- Dis_Excel_Korel, Pivot_Bant_Sinif, 30+ grafik katalogu, AI vs ANFIS bootstrap CI
- Tahmin_Dogrulugu: TUM 267 cekirdek (sadece sicrama degil)
- ST12-D bant x hedef hata heatmap aciklamasi eklendi

**_generate_chapter5_tartisma():**
- Nükleer ozellik korelasyon yorumu (Dis_Excel_Korel)
- Pivot tablo ve bant x hedef hata yorum bolumu
- ANFIS bootstrap CI tartismasi
- AM13-F: model tipi x hedef delta R2 analizi

**_generate_appendices():**
- 10 Excel sayfasi tam aciklama (Dis_Excel_Korel, Pivot_Bant_Sinif eklendi)
- 30+ grafik katalogu (MC9-D/E/F, ST12-C/D/E/F/G/H, AM13-E/F)
- Bootstrap CI ciktilari eki (AI + ANFIS her ikisi icin)

### ANFIS Bootstrap Saglamlik Testi Eklendi (2026-04-21)

**Dosya:** `pfaz6_final_reporting.py`

Daha once sadece AI icin yapilan `BootstrapConfidenceIntervals` (5000 bootstrap, %95 CI)
artik ANFIS Val R2 icin de yapiliyor.
Cikti: `outputs/reports/bootstrap_ci/anfis_val_r2_bootstrap_ci.json`

---

## PFAZ 13 - AutoML

### Tamamlanan

- [x] AutoMLHyperparameterOptimizer -> AutoMLOptimizer alias duzeltildi
- [x] automl_logging_reporting_system dataclass TypeError -> except Exception ile yakalandi

---

## Genel Notlar

### Windows Encoding
- Tum logger mesajlarinda emoji/Unicode karakter KULLANILMAMALI
- ASCII alternatifler kullanilmali

### Path Yonetimi
- main.py: self.project_root = Path(__file__).parent eklendi
- Tum relative path'ler project_root'a gore cozumlenmeli

---

## proje_D01.md Gelistirmeleri (Nisan 2026)

Kullanicinin sonraki calistirmadan sonra bildirdigi kritik sorunlar ve yapilan duzeltmeler.

---

### KRITIK: CSV/MAT Formatinda Cekirdek Adi Sorunu

**Sorun:** CSV ve MAT dosyalari NUCLEUS, A, Z, N sutunlarini iceriyordu.
Modeller nucleus adi gibi string degerleri NaN olarak isliyordu; hesaplamalar yanlis.

**Duzeltme — `pfaz01_dataset_generation/dataset_generation_pipeline_v2.py`:**
- CSV kaydetme: sadece `feature_cols + target_cols` (numeric) — NUCLEUS/A/Z/N YOK
- Excel kaydetme: `id_cols + feature_cols + target_cols` (insan icin okunabilir, NUCLEUS dahil)
- `_save_as_mat()` tamamen yeniden yazildi:
  - Sadece `data = [features | target]` (2D float64 matrisi) kaydedilir
  - Ek meta: `n_features`, `n_targets`, `n_samples`, `dataset_name`, `target`, `split`
  - ANFIS beklentisi: `data` degiskeni, son sutun = hedef, oncekiler = girisler

```python
# MAT uyumlu format (ANFIS icin)
mat_data = df[num_cols].apply(pd.to_numeric, errors='coerce').dropna().values.astype(np.float64)
mat_dict = {
    'data': mat_data,           # (n_samples, n_features + n_targets)
    'n_features': len(feature_cols),
    'n_targets':  len(target_cols),
    ...
}
scipy.io.savemat(mat_file, mat_dict)
```

---

### 75/100 Cekirdek — Kucuk Dataset Kisitlamasi

**Sorun:** 75 veya 100 cekirdeklik datasetler icin tum senaryo x feature kombinasyonlari
uretiliyordu; disk doldu, egitim gunlerce surdu.

**Duzeltme — `dataset_generation_pipeline_v2.py`:**
```python
SMALL_NUCLEUS_THRESHOLD = 100   # bu es-deger veya altindaki n_nuclei
SMALL_NUCLEUS_SCENARIO  = 'S70' # sadece bu senaryo uretilir
SMALL_NUCLEUS_FEATURE_SETS = ['Basic', 'Standard']  # sadece bu feature setler
```
- Senaryo dongusu: `effective_count <= 100` ise S70 disindakiler SKIP
- Feature set dongusu: Basic ve Standard ile sinirlandirilir

---

### Log Rotasyonu

**Sorun:** Log dosyasi 2 GB'a ulasmis, acilamiyor.

**Duzeltme — `main.py` `setup_logging()`:**
```python
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    log_file, maxBytes=50*1024*1024, backupCount=5,
    encoding='utf-8'
)
```
Maksimum ~300 MB (50 MB x 6 dosya).

---

### Excel Satir Bolme

**Sorun:** Buyuk Excel dosyalari (All_AI_Models vs.) gerceksiz boyutlara ulasiyor.

**Duzeltme — `pfaz06_final_reporting/pfaz6_final_reporting.py`:**
- `_write_df_chunked(df, writer, base_sheet_name, max_rows=50_000)` yardimci fonksiyonu eklendi
- 50.000 satirdan uzun DataFrame'ler `SheetName`, `SheetName_2`, ... seklinde bolunuyor
- `All_AI_Models`, `XGBoost`, `LightGBM`, `RF`, `DNN` sheet'leri bu yontemle yaziliyor

---

### Terminal Ilerleme Gostergesi

**Sorun:** Calisma suresi ve tahmini bitis zamani terminal ciktisinda yoktu.

**Duzeltme — `main.py` `run_all_pfaz()`:**
- Her faz oncesi/sonrasi sure olculuyor (`time.time()`)
- Bitis tahmini: `elapsed / done * remaining`
- Gorsel cubuk: `█` (tamamlanan) + `░` (kalan) / 20 karakter

---

### Basarisiz Dataset Otomatik Silme

**Yeni ozellik — `main.py` `cleanup_failed_datasets()`:**
- PFAZ2 metrikleri taranir, her dataset icin en iyi Val R2 bulunur
- `val_r2 < threshold` olan dataset dizinleri silinir
- `dry_run=True` varsayilandir (once simule et, sonra onayla)
- `run_pfaz_02()` sonunda `cleanup_r2_threshold` config anahtari varsa otomatik cagrilir

---

### DNN Minimum Dataset Boyutu Filtresi

**Sorun:** Cok kucuk datasetlerde DNN uyariyla devam ediyordu, sonuc anlamsizdi.

**Duzeltme — `pfaz02_ai_training/parallel_ai_trainer.py`:**
```python
DNN_MIN_SAMPLES = 80
if self.model_type == 'DNN' and len(X_train) < DNN_MIN_SAMPLES:
    raise ValueError(
        f"[DNN-SKIP] Egitim seti cok kucuk ({len(X_train)} < {DNN_MIN_SAMPLES}). "
        f"DNN bu dataset icin atlaniyor."
    )
```
`ValueError` ile is atlanir (pass), uyari duzeyinde devam edilmez.

---

### Tek Cekirdek Tahmin Sistemi

**Yeni dosya:** `pfaz_modules/pfaz04_unknown_predictions/single_nucleus_predictor.py`

**Sinif:** `SingleNucleusPredictor`

Kullanim amaci: Egitilmis tum AI (PFAZ2) ve ANFIS (PFAZ3) modellerini yukleyerek,
kullanicinin girdigi tek bir cekirdek icin hedef degerleri tahmin eder.

Kabul edilen giris formatlari:
- `dict`: `{'Z': 26, 'N': 30, 'A': 56, 'SPIN': 0.0, 'PARITY': 1}`
- Dosya yolu: aaa2.txt formatinda bir satir veya CSV
- DataFrame: pandas DataFrame (bir satir)

Ana metodlar:
- `predict_from_dict(nucleus_dict, target=None)` — sozlukten tahmin
- `predict_from_file(file_path, target=None)` — dosyadan tahmin
- `predict_from_dataframe(df, target=None)` — DataFrame'den tahmin

Cikti:
- `single_nucleus_predictions.xlsx` — model bazli ve konsensus tahminleri
- `single_nucleus_predictions.json` — makine okunabilir cikti

**main.py entegrasyonu:**
- `run_single_prediction(nucleus_input=None, target=None)` metodu eklendi
- Interaktif menude `predict` secenegi eklendi
- CLI: `python main.py --predict "Z=26 N=30 A=56"` veya `--predict aaa2.txt`

---

### PFAZ6 — AutoML Iyilestirme Sheet'i

**Yeni ozellik — `pfaz6_final_reporting.py`:**
- `_write_automl_improvements()` metodu eklendi
- `automl_retraining_log.json` okunarak Excel'e yaziliyor
- 3 sheet: `AutoML_Improvements` (renk kodlu), `AutoML_BestParams`, `AutoML_Overview`
- `generate_thesis_tables()` icinde cagriliyor

---

### PFAZ10 — Master Integration Tamamen Yeniden Yazildi (v5.0)

**Sorun:** Eski `pfaz10_master_integration.py` hardcoded `/mnt/project` Linux yollarini
kullaniyordu ve yalnizca 1 bolum uretiyordu.

**Duzeltme:** Dosya tamamen yeniden yazildi:
- Windows/Linux uyumlu `pathlib.Path` yol yonetimi
- `pfaz_outputs` dict'i `main.py`'den enjekte ediliyor
- 11 bolum + 2 ek (abstract EN+TR dahil)
- PFAZ2/3/6/7/8/9/12/13 ciktilarindan veri toplaniyor
- `compile_pdf=False` varsayilan (pdflatex olmadan kilitlenmiyor)
- `compile.sh` ve `compile.bat` uretiliyor

---

### PFAZ13 — AutoML Yeniden Egitim Dongusu

**Yeni dosya:** `pfaz_modules/pfaz13_automl/automl_retraining_loop.py`

**Sinif:** `AutoMLRetrainingLoop`

Isleyisi:
1. PFAZ2 `metrics_*.json` dosyalari taranir
2. Val R2 < `threshold` olan (target, dataset, model_type) kombinasyonlari bulunur
3. Her kombinasyon icin `AutoMLOptimizer` calistirilir (Optuna)
4. Once/sonra metrikler kaydedilir
5. `automl_retraining_log.json` ve `automl_improvement_report.xlsx` uretilir

`main.py run_pfaz_13()` entegrasyonu:
- Standart AutoML optimizasyonundan SONRA `AutoMLRetrainingLoop` calisir
- `retrain_threshold` ve `max_retrain` config.json'dan okunur

---

### PFAZ8 — Supplemental Visualizer

**Yeni metod — `main.py`:** `run_pfaz_08_supplemental()`
- PFAZ9/12/13 tamamlandiktan sonra cagrilir (`run_all_pfaz` sonunda)
- `pfaz_modules/pfaz08_visualization/supplemental_visualizer.py` modulu kullanilir
- Monte Carlo, istatistiksel test ve AutoML iyilestirme grafikleri uretilir
- Cikti: `outputs/pfaz08_visualization/supplemental/`

**Ilgili katalog:** `VISUALIZATIONS_INDEX.md` guncellendi:
- MC9-A/B/C, ST12-A/B, AM13-A/C/D grafikleri eklendi
- dataset_generation_pipeline_v2.py: _load_raw_data'ya fallback path cozumleme eklendi

---

## Nisan 2026 Gelistirmeleri (Oturum 2)

---

### Log Rotasyon Limiti Guncellendi

**Degisiklik — `main.py` `setup_logging()`:**
- `maxBytes` 50 MB -> 200 MB olarak guncellendi
- 200 MB x 5 yedek = toplam maks ~1.2 GB
- Uzun sureli egitimler icin yeterli bant genisligi

---

### SingleNucleusPredictor — Tam Yeniden Yazim

**Dosya:** `pfaz_modules/pfaz04_unknown_predictions/single_nucleus_predictor.py`

#### Giris Formati (pred_input.txt / aaa2.txt uyumlu)

Onerilen format (`pred_input.txt`):
```
# Sutunlar: Z  N  SPIN  PARITY
# Ornek: Fe-56
26  30  0.0  1
```

Kabul edilen formatlar:
- **Dict**: `{'Z': 26, 'N': 30, 'SPIN': 0.0, 'PARITY': 1}`
- **pred_input.txt**: Bosluk/tab ayrimli, `#` ile yorum satirlari
- **aaa2.txt**: Ilk 4 sutun `Z N SPIN PARITY` olarak okunur
- **CSV**: `Z,N,SPIN,PARITY` baslikli (veya daha fazla sutun)
- **Excel**: Ayni sutun yapisina sahip

Minimum gereksinim: Sadece Z ve N yeterli.
A = Z + N otomatik hesaplanir.

#### Feature Zenginlestirme (TheoreticalCalculationsManager)

Kullanici sadece Z, N, SPIN, PARITY verdiginde:
1. `TheoreticalCalculationsManager.calculate_all_theoretical_properties(df)` cagrilir
2. `magic_character`, `BE_per_A`, `Z_magic_dist`, `N_magic_dist`, `Beta_2_estimated` vb. hesaplanir
3. Modellerin egitildigi feature setleri otomatik tamamlanir
4. Zenginlestirme basarisiz olursa eksik feature'lar 0 ile doldurulur (uyari ile)

#### Top 25 Model

`top_n_models = 25` (varsayilan, degistirildi 5 -> 25)
- Her hedef icin PFAZ2 metrics'ten en yuksek Val R2'ye gore siralar
- En iyi 25 AI model + max 12-13 ANFIS model kullanilir
- model_id artik R2 degerini icerir: `AI_datasetname_modelname_R2x.xxx`

#### Gorsellestime

Yeni `_generate_visualizations()` metodu:
- Per-target bar chart: her modelin tahmini + consensus cizgisi
- Consensus ozet chart: tum hedefler yan yana
- Cikti: `outputs/pfaz04.../single_predict_TIMESTAMP/plots/`

---

### main.py — Fazlar Sonrasi Tahmin Sorusu

**Yeni metod:** `_ask_prediction_after_pipeline()`

`run_all_pfaz()` tamamlandiginda (PFAZ4 dahil ise) otomatik cagrilir:

```
[PREDICT] Tahmin Sistemi
  1. Tek cekirdek / kucuk liste tahmini
     (Mevcut egitilmis modeller kullanilir, dataset uretilmez)
  2. Yeni buyuk liste gir
     (PFAZ1-13 yeni bir cikti klasorunde yeniden calistirilir)
  3. Hayir, atlama
```

---

### main.py — Yeni Liste icin Tam Pipeline

**Yeni metod:** `run_all_pfaz_with_custom_data(data_file)`

- Giris dosyasinin satir sayisini kontrol eder
- 1 satir -> tek cekirdek tahmini (dataset yok)
- Cok satir -> `outputs/custom_run_YYYYMMDD_HHMMSS/` olusturulur
- Gecici `config.json` yazilir (data_file + yeni output dizinleri)
- Yeni `NuclearPhysicsAIOrchestrator` ornegi ile PFAZ1-13 calistirilir
- Her ozel calistirma kendi klasorunde izole tutulur

---

### main.py — run_single_prediction Guncellendi

- `target` parametresi kaldirildi (artik tum hedefler tahmin edilir)
- `top_n_models=25` ile SingleNucleusPredictor baslatilir
- Cikti klasoru: `pfaz_outputs[4]/single_predict_TIMESTAMP/`
- Terminal ciktisi: consensus deger + model sayisi per hedef
- Grafik, Excel, JSON ciktilari raporlanir

---

### PFAZ1 — Target-Feature Iliskisi Dogrulama

Mevcut kod **dogru** calistiyor. Ozet:

```
TARGET_RECOMMENDED_SETS = {
    'MM':     ['AZS', 'AZMC', 'AZBEPA', 'ASMC', ...]  # spin + magic odakli
    'QM':     ['AZB2E', 'ZB2EMC', 'AZSB2E', ...]      # deformasyon odakli
    'Beta_2': ['MCZMNM', 'AZVNV', 'ZVNVZMNM', ...]   # magic distance odakli
    'MM_QM':  ['AZS', 'AZB2EMC', 'AZSMCB2E', ...]    # karma set
}
```

- `_generate_all_datasets()` her hedef icin `get_target_feature_sets(target)` cagirir
- Her hedef farkli feature kombinasyonlari ile egitilir
- SHAP analizine dayali secimlerin mantigi:
  - MM: A(19.2%), Z(17.5%), SPIN(12.8%) — kirsallik + spin etkileri
  - QM: Z(21.5%), Beta_2(18.3%) — cekirdek sekli + proton yuzu
  - Beta_2: magic_char(22.1%), Z_magic_dist(18.7%) — kabuk yapisi

---

## Nisan 2026 Gelistirmeleri (Oturum 3) — QA Dead Code Audit & Wiring

**Tarih:** 2026-04-13  
**Kapsam:** 23 dead/unwired modülün tespiti, 8 modülün pipeline'a bağlanması, 5 duplikat dosyanın silinmesi

---

### QA Denetimi — Dead Code Bulguları

Pipeline genelinde `__init__.py`'de export edilmiş fakat hiç instantiate/çağrılmamış 23 modül tespit edildi. Detaylı liste ve sınıflandırma: `QA_WIRING_REPORT.md`

---

### Bağlanan Modüller (8 adet)

#### PFAZ3 — `anfis_parallel_trainer_v2.py`
- **`ANFISPerformanceAnalyzer`**: `train_all_anfis_parallel()` sonuna eklendi. `metrics_*.json` dosyalarından config bazlı R2/RMSE/MAE ortalama istatistikleri çıkarır; `performance_analysis/ANFIS_Performance_Analysis.xlsx` üretir.
- **`ANFISVisualizer`**: `plot_target_comparison()` ile hedef (MM/QM/Beta_2/MM_QM) bazlı kutu grafikleri üretir → `anfis_visualizations/`

#### PFAZ4 — `unknown_nuclei_predictor.py`
- **`GeneralizationAnalyzer`**: `predict_unknown_nuclei()` sonuna eklendi. Val R² (known) / Test R² (unknown) oranından Genelleme Skoru (GS%) hesaplar. Modelleri Excellent/Good/Moderate/Poor sınıflandırır → `generalization_analysis/`

#### PFAZ6 — `pfaz6_final_reporting.py`
- **`ComprehensiveExcelReporter`**: `run_complete_pipeline()` sonuna eklendi. ANFIS sonuç DataFrame'i ile 18 sayfalık ANFIS detay Excel raporu üretir → `ANFIS_Comprehensive_Report_*.xlsx`
- **`BootstrapConfidenceIntervals`** (PFAZ12'den): AI Val R² skorları üzerinde n=5000 bootstrap ile %95 güven aralığı hesaplar. Percentile metot. Sonuç JSON olarak kaydedilir → `bootstrap_ci/ai_val_r2_bootstrap_ci.json`

#### PFAZ8 — `visualization_master_system.py`
- **`SHAPAnalyzer`**: `generate_all_visualizations()` sonuna eklendi. Her hedef için trained_models dizinindeki en iyi tree-based modeli (RF/XGB/GBM/LGB) otomatik bulur, SHAP summary/beeswarm/dependence/waterfall/force grafikleri üretir → `shap_analysis/`. SHAP kurulu değilse sessizce atlanır.

#### PFAZ13 — `automl_retraining_loop.py`
- **`AutoMLFeatureEngineer`**: `run()` sonuna eklendi. En fazla iyileşen dataset üzerinde polinomial (d=2) + interaction + transform ile aday feature üretir; RFE/LASSO/mutual_info ile seçim → `feature_engineering/`
- **`AutoMLANFISOptimizer`**: `run()` sonuna eklendi. En düşük başlangıç R²'li dataset üzerinde ANFIS hyperparametre optimizasyonu (FIS method, MF tipi, n_mfs, epochs). MATLAB yoksa sklearn fallback → `anfis_optimization/`

---

### Silinen Duplikat Dosyalar (5 adet)

| Dosya | Neden Silindiği |
|-------|-----------------|
| `pfaz03/anfis_all_nuclei_predictor.py` | `pfaz04/all_nuclei_predictor.py` ile duplikat |
| `pfaz04/all_nuclei_predictor.py` | `SingleNucleusPredictor` + `UnknownNucleiPredictor` karşılıyor |
| `pfaz04/unknown_nuclei_splitter.py` | PFAZ1 zaten train/val/test split yapıyor |
| `pfaz06/excel_formatter.py` | Kullanılmayan basit wrapper |
| `pfaz09/aaa2_control_group_comprehensive.py` | `aaa2_control_group_complete_v4.py` karşılıyor |

---

### Güncellenen `__init__.py` Dosyaları

- `pfaz03/__init__.py`: `ANFISAllNucleiPredictor` kaldırıldı
- `pfaz04/__init__.py`: `AllNucleiPredictor` + `UnknownNucleiSplitter` kaldırıldı
- `pfaz06/__init__.py`: `ExcelFormatter` kaldırıldı
- `pfaz09/__init__.py`: `AAA2ControlGroupComprehensive` kaldırıldı

---

### Teknik Notlar

- Tüm wiring işlemleri `try/except Exception` ile korunmuştur → pipeline hiç durmaz
- `BootstrapCI`: `bootstrap_distribution` numpy array JSON'a serialize edilmez (boyut çok büyük), sadece scalar değerler kaydedilir
- `SHAPAnalyzer`: Sadece tree-based modeller (RF/XGB/GBM/LGB) desteklenir; DNN için TreeExplainer çalışmaz
- `AutoMLANFISOptimizer`: `use_matlab=False` ile başlatılır → TakagiSugenoANFIS (pfaz03) yerine sklearn approximation kullanır. MATLAB entegrasyonu gelecek oturum.
- Beklemede kalan 14 modül: `QA_WIRING_REPORT.md` § 4'te listelenmiştir

---

## Nisan 2026 Gelistirmeleri (Oturum 4) — WarningTracker + NuclearPatternAnalyzer + Kalan Wirings

**Tarih:** 2026-04-13
**Kapsam:** Pipeline uyarı takibi, nükleer desen analizi modülü, QA Oturum 3'te beklemede kalan 14 modülün tamamlanması

---

### WarningTracker — Pipeline Uyarı İzleme Sistemi

**Yeni dosya:** `utils/warning_tracker.py`

**Sorun:** Her pipeline bloğu `try/except Exception` ile korunmuş; başarısız olunca sadece `[WARNING]` log yazılıyor. Bu uyarılar takip edilemiyordu.

**Çözüm:** `WarningTracker` sınıfı — Python logging altyapısına özel bir `Handler` ekleyerek WARNING/ERROR düzeyindeki tüm mesajları yapılandırılmış olarak yakalar.

**Mimari:**
```python
class _WarningHandler(logging.Handler):
    def emit(self, record):
        entry = {timestamp, level, logger, module, pfaz, message, traceback}
        tracker._add(entry)   # anlık JSON yazımı (resume desteği)

class WarningTracker:
    def attach(logger=None)      # root logger'a ekle
    def warn(pfaz, comp, msg, exc)  # try/except bloğundan manuel kayıt
    def save_report()            # Excel: 3 sayfa (Tüm_Uyarılar, PFAZ_Özeti, Seviye_Özeti)
    def print_summary()          # pipeline sonu terminal özeti
```

**Çıktı dosyaları:**
- `outputs/pipeline_warnings.json` — anlık yazım (her yeni uyarıda güncellenir)
- `outputs/pipeline_warnings_report.xlsx` — `save_report()` çağrısı sonrası

**main.py entegrasyonu:**
- `setup_logging()` içinde `WarningTracker` instantiate + `tracker.attach(root)` eklendi
- `run_all_pfaz()` sonunda `get_tracker().save_report()` + `print_summary()` eklendi

**Erişim:**
```python
from utils.warning_tracker import get_tracker
get_tracker().warn('PFAZ3', 'ANFISRobustnessTester', str(e), e)
```

---

### ExcelStandardizer — Yeni Excel Standartlaştırma Aracı

**Yeni dosya:** `pfaz_modules/pfaz06_final_reporting/excel_standardizer.py`

**Neden:** `AdvancedExcelFormatter` / `ExcelFormatter` dosyaları silinmişti; yerine tüm Excel çıktılarını standartlaştıracak genel amaçlı araç gerekiyordu.

**Özellikler:**
- Context manager API: `with ExcelStandardizer(path) as es: es.write_sheet(...)`
- `write_sheet()`: bold mavi başlık, autosize sütunlar, R² renk skalası (kırmızı→sarı→yeşil), data bar
- `write_pivot()`: pivot tablo oluşturma
- Standalone fonksiyonlar: `autosize_and_header()`, `add_r2_color_scale()`, `color_cell()`
- Primary backend: xlsxwriter; openpyxl fallback

**`pfaz06/__init__.py` güncellemesi:**
- `ExcelStandardizer`, `autosize_and_header`, `add_r2_color_scale`, `color_cell` export'a eklendi
- `AdvancedAnalysisReportingManager` import/export kaldırıldı (FinalReportingPipeline ile örtüşüyordu)

---

### Kalan 14 Wiring Tamamlandı

Oturum 3'te "Beklemede" bırakılan 14 modülün tamamı pipeline'a bağlandı:

#### PFAZ3 — `anfis_parallel_trainer_v2.py`
- **`ANFISRobustnessTester`**: En iyi 3 sonuç üzerinde iteratif outlier test (max_iterations=3). Dataset CSV'lerini `generated_datasets/` dizininden okur → `robustness_analysis/ANFIS_Robustness_Report.xlsx`
- **`ANFISAdaptiveStrategy` / `PatternTracker`**: Tüm eğitim sonuçlarından model/feature/scaling kategorileri bazlı R² örüntü analizi → `adaptive_pattern_analysis/pattern_analysis.json` + `.xlsx`
- **`ANFISDatasetSelector`**: PFAZ2 `training_summary.xlsx` okunarak Method1 + Method2 dataset seçimi → `anfis_selected_datasets/`

#### PFAZ6 — `pfaz6_final_reporting.py`
- **`AdvancedSensitivityAnalysis`**: En iyi RF/XGB modeli bulunur, feature means = baseline, ±1 std = ranges ile tornado analizi → `sensitivity_analysis/`

#### PFAZ8 — `visualization_master_system.py`
- **`AnomalyVisualizationsComplete`**: AAA2 enriched CSV → IQR anomali tespiti → radar/outlier grafikleri → `anomaly_visualizations/`
- **`InteractiveHTMLVisualizer`**: AI `training_summary.xlsx` → Plotly interaktif HTML visuals → `interactive_visuals/`
- **`LogAnalyticsVisualizationsComplete`**: `.log` dosyası regex parse → 5 grafik → `log_analytics/`
- **`MasterReportVisualizationsComplete`**: PFAZ6 Excel çıktıları → executive dashboard → `master_visuals/`
- **`ModelComparisonDashboard`**: AI + ANFIS summary dosyaları → karşılaştırma dashboard → `model_comparison/`

#### PFAZ9 — `aaa2_control_group_complete_v4.py`
- **`AAA2DataQualityChecker`**: `aaa2.txt` → detaylı kalite Excel → `data_quality/`
- **`MonteCarloSimulationSystem`**: Her hedef için `run_complete_mc_analysis()` → `monte_carlo_analysis/`

#### PFAZ13 — `automl_retraining_loop.py`
- **`AutoMLVisualizer`**: `automl_improvement_report.xlsx` veya ANFIS/feature çıktısı → `generate_all_plots()` → `automl_visualizations/`. Optuna visualization kurulu değilse `pip install optuna[visualization]` mesajı ile atlanır.

---

### NuclearPatternAnalyzer — Nükleer Desen Analizi Modülü

**Yeni dosya:** `pfaz_modules/pfaz12_advanced_analytics/nuclear_pattern_analyzer.py`

**Sınıf:** `NuclearPatternAnalyzer`

**Amaç:** AI modelleri eğitildikten sonra nükleer verideki fiziksel desenleri tespit etmek. Hedef değerlerindeki ani sıçramaları, sihirli sayı etkilerini ve sıçrayan çekirdeklerle normal çekirdekler arasındaki özellik farklarını analiz eder.

**Analiz Metodları:**

| Metod | Açıklama |
|-------|----------|
| `_chain_analysis(groupby, varyby)` | İzotop/izotone/izobar zincirlerde `np.diff` ile hedef sıçramalarını tespit |
| `_mean_cluster_analysis()` | Hedef değerine göre 5 küme (Tipik ±1σ, Yüksek/ÇokYüksek, Düşük/ÇokDüşük) |
| `_magic_number_analysis()` | Her sihirli sayı (2,8,20,28,50,82,126) için near(±3) vs far KS + Mann-Whitney testi |
| `_jump_feature_analysis()` | İzotop+izotone sıçramalarını birleştirip jump vs normal t-test feature karşılaştırması |

**Çıktılar:**
- `nuclear_pattern_analysis_TIMESTAMP.xlsx` (6 sayfa: Genel_Özet, Küme, Sıçrama, ZincirStat, Magic, Özellik)
- 12 PNG grafik (zincir çizgileri top-8, magic violin, feature diff bar, cluster bar)

**Desteklenen veri formatları:** `aaa2.txt`, enriched CSV, enriched XLSX

**Test sonuçları (267 çekirdek, 3 hedef):**
```
MM:     mean=1.597, izotop_sicrama=20, jump_nuclei=46
QM:     mean=0.536, izotop_sicrama=15, jump_nuclei=29
Beta_2: mean=0.211, izotop_sicrama=24, jump_nuclei=55
```

**Wiring:**
- **PFAZ2** (`parallel_ai_trainer.py`): `train_all_models_parallel()` Step 8 olarak eklendi
- **PFAZ3** (`anfis_parallel_trainer_v2.py`): `train_all_anfis_parallel()` return'dan önce eklendi
- Her iki wiring `try/except` korumalı; `aaa2.txt` bulunamazsa uyarıyla atlanır

---

### Teknik Düzeltmeler (Oturum 4)

- **`pd.to_numeric(errors="coerce")`**: `errors="ignore"` pandas 2.x'de element-bazlı değil kolon-bazlı çalışıyor → tüm `nuclear_pattern_analyzer.py` numeric dönüşümleri `"coerce"` ile güncellendi
- **`plots_dir` NameError**: Plot metodlarında parametre adı `out_dir` iken gövdede `plots_dir` kullanılıyordu → tüm referanslar `out_dir` olarak düzeltildi
- **`"Anlamlı (p<0.05)"` KeyError**: Windows'ta Türkçe karakter içeren sütun adı → `"Sig_p05"` olarak yeniden adlandırıldı

---


---

## Oturum 2026-04-20: Bug Tarama + PFAZ7 Gercek Model Entegrasyonu

### Kapsam
Bu oturumda PFAZ4, PFAZ7, PFAZ9 ve PFAZ10 modulleri buyuk olcude guncellendi.

### PFAZ4 — SingleNucleusPredictor: Per-Model Feature Set

**Sorun:**  tek bir feature seti bularak TUM modellere uyguluyordu.
Oysa her model kendi ozgu feature setiyle egitilmistir.

**Duzeltme:**
- : Her modelin  dosyasindan  okunur.
  AI ve ANFIS modelleri taranir, val_r2'ye gore siralamir, top_n secilir.
- : Her model icin ayri X matrisi olusturulur; eksik feature 0 ile doldurulur.
- : Her kayit icin kendi  ile tahmin yapilir.
- MM_QM cikisi: shape (n,2) → , 

### PFAZ9 — AAA2 Monte Carlo: ANFIS Dahil Edildi + Experimental Values

**Sorun 1:**  sadece  tariyordu; ANFIS modelleri hic dahil edilmiyordu.

**Duzeltme 1:** Hem  hem  taranarak  alani ('AI'/'ANFIS') eklendi.
Log satiri: 

**Sorun 2:**  sadece NUCLEUS/A/Z/N yaziyordu; deneysel deger ve artik yok.

**Duzeltme 2:** Predictions sayfasina , , , ,  sutunlari eklendi.

### PFAZ7 — Ensemble: Mock Veri → Gercek Model Entegrasyonu

**Sorun (KRITIK):**  tamamen rastgele  veri uretiyordu.
Hicbir gercek model yuklenmiyordu.

**Duzeltme:**  yeniden yazildi:
-  —  +  taramasi;  feature list
-  — her model kendi / + feature seti; cekirdek ID ile ic eslesme
-  — 3 voting + 4 stacking ensemble; MM_QM (n,2) cikis destegi
-  — per-hedef cok sayfali Excel raporu
-  guncellendi:  seklinde cagri

### PFAZ10 — Tez Modulu Kapsamli Guncelleme

**Eklenen bolumler:**
- : SEMF formulasyonu, kabuk modeli, Schmidt limitleri, Q0, S2n, Woods-Saxon
- : 35 kisaltma (longtable)
- : Nukleer fizik + ML + ANFIS sembol tablolari
- : PFAZ4 pipeline, SingleNucleusPredictor akisi, 95% CI ensemble formulu
- : PFAZ5, kalite siniflandirma tablosu, uzlasim metrik formulleri
- : Oylama + yigilma formulleri, meta-ogrenme mimarisi
- : 6 buyuk Excel raporunun sutun basliklari referansi
- : 21 feature + formul tablosu
- : R^2/RMSE/MAE, IQR sinis formulu (k=3.0), MC perturbasyonu, Friedman
- : SEMF tureyen featurelar, dataset adlandirma, anomali kaldirilma metrigi
- : RF, XGB, SVR, DNN formulleri; 5-kat CV
- : 5 katman denklemleri, Gauss/Bell/Trap/Tri formulleri, hibrit ogrenme

### Dokumantasyon Guncelleme

- : Bu notlar eklendi
- : PFAZ7 bolumu gercek entegrasyon ile guncellendi
- : PFAZ7 ozeti + kullanim kodu guncellendi

---

## Oturum 2026-04-24 — PFAZ2/3/13 Kapsamli Guncelleme

### PFAZ03 — ANFIS: 3-Faz Strateji, Outlier Guclendirilmesi, Yeni Excel Sayfasi

**1. OutlierDetector Degerleri Guncellendi**
- `iqr_multiplier`: 2.5 → 3.0
- `zscore_threshold`: 2.5 → 3.0
- Guvenlik zemini: 80% → 90% (kucuk benzersiz-cekirdek datasetleri icin)
- Neden: Projede sihirli cekirdek yok, her cekirdek benzersiz. 2.5 deger fazla
  agresif outlier kaldirimina yol aciyordu.

**2. 3-Faz Soft Strateji (train_all_anfis_parallel)**
- Wave 1 PILOT: 4x 2MF config (Trap/Bell/Gauss/Tri) tum datasetler icin
- Wave 2 ADVANCED: 4x 3MF+SubClust config; best pilot val_R2 < 0.3 ise max_iter=100
- Hic eliminasyon yok — tum 8 config calisiyor (soft versiyon)
- Yeni `Training_Mode` kolonu: 'pilot' / 'advanced' / 'advanced_reduced'
- `ANFISTrainingJob` ve `ANFISTrainingResult` dataclass'larina `training_mode` alani eklendi
- `_run_parallel_wave()` yardimci metod eklendi

**3. ANFISDatasetSelector Deaktive Edildi**
- `train_all_anfis_parallel()` icindeki ANFISDatasetSelector cagri blogu kaldırildi
- Neden: 3-faz strateji zaten tum datasetleri kapsıyor; PFAZ2'ye bagimlılık gereksiz

**4. Excel Cikti Guncellemeleri (save_summary_report)**
- `Status_Note` kolonu eklendi: 'OK' / 'POOR_R2_FILTER' / 'DIVERGED' / 'ERROR: ...'
- `Training_Mode` kolonu eklendi: pilot/advanced/advanced_reduced
- Yeni sayfa `Outlier_Frequency`: cekirdek bazli outlier frekans ozeti
  - Sutunlar: Nucleus, Times_Outlier_Detected, Times_Actually_Removed,
    Removal_Rate_%, Datasets_Affected_Count, Datasets_Affected
- R2_Category_Summary Sheet 4→6 (Outlier_Frequency ile Nucleus_Tracking araya girdi)
- `_apply_excel_formatting()` Outlier_Frequency sayfasini da biçimlendiriyor

### PFAZ02 — AI Trainer: Status_Note Kolonu

- `training_results_summary.xlsx` Excel raporuna `Status_Note` kolonu eklendi
- Degerler: 'OK' / 'POOR_R2_FILTER' / 'DIVERGED' / 'ERROR: ...' / 'FAILED'
- PKL_Saved ile birlikte her modelin durumu net gorulur

### PFAZ13 — AutoML: Excellent Kategori + aaa2_txt_path Otomasyonu

**1. Excellent Kategori Eklendi**
- `EXCELLENT_MIN = 0.95` sabiti eklendi
- `find_categorized_candidates()`: R2 >= 0.95 artık 'excellent' kategorisinde
- Her kategoriden en dusuk R2'li 25 aday seciliyor (n_per_category=25)
- `run()` log'u guncellendi: excellent sayisi da gosteriliyor

**2. aaa2_txt_path Otomasyonu**
- `_auto_detect_aaa2_path()` metodu eklendi
- Once config.json'dan `paths.data_dir` + `data.input_file` okur
- Fallback: `data/aaa2.txt` standart konumlara bakar
- `__init__` parametresinde None gelirse otomatik doldurulur

### Import Test Sonuclari (2026-04-24)
- PFAZ02 parallel_ai_trainer: OK
- PFAZ03 anfis_parallel_trainer_v2: OK
- PFAZ13 automl_retraining_loop: OK


## Oturum 2026-04-27 — Hedef Kısıtlaması: Sadece MM ve QM

### Karar
Tüm pipeline sadece **Manyetik Moment (MM)** ve **Kuadrupol Moment (QM)** için çalışacak.
**Beta_2** ve **MM_QM** hedefleri tüm fazlardan çıkarıldı.

### Etki: Dataset Sayısı
- Önceki: 4 hedef × kombinasyonlar ≈ ~5,039 dataset → ~755,882 PFAZ2 job
- Sonrası: 2 hedef × kombinasyonlar ≈ ~2,520 dataset → ~377,000 PFAZ2 job (yaklaşık yarı)

### Değiştirilen Dosyalar (13 dosya)
| Dosya | Değişen Satır |
|-------|---------------|
| config.json | data.targets + pfaz01.targets |
| main.py | pfaz_config[1].targets, pfaz_config[5].targets, PFAZ1 default, PFAZ9 call, PFAZ12 targets |
| pfaz01/dataset_generation_pipeline_v2.py | self.targets default + TARGETS sabit |
| pfaz04/single_nucleus_predictor.py | ALL_TARGETS sabit |
| pfaz04/unknown_nuclei_predictor.py | target_label döngüsü |
| pfaz05/faz5_complete_cross_model.py | self.targets default |
| pfaz05/faz5_cross_model_analysis.py | target_key döngüsü |
| pfaz06/pfaz6_final_reporting.py | 3 yer: döngü + summary.targets |
| pfaz06/latex_generator.py | 2 yer: target_best döngüleri |
| pfaz07/pfaz7_complete_ensemble_pipeline.py | targets list |
| pfaz07/pfaz7_ensemble.py | run_complete_pipeline default param |
| pfaz08/pfaz8_thesis_charts.py | TARGETS sabiti |
| pfaz08/supplemental_visualizer.py | target döngüsü |

### Korunan (değiştirilmeyen) yerler
- target_col_map sözlükleri (Beta_2/MM_QM lookup, zararsız)
- Kolon dışlama setleri ({NUCLEUS, A, Z, N, MM, QM, Beta_2, ...})
- Veri tespiti if/else blokları (veri yoksa zaten atlanır)
- visualization_master_system.py (if t in df.unique() filtresi — data-driven)

### Import Testi (2026-04-27): 7/7 OK

