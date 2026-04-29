# HPC Deployment Checklist
## Nuclear Moments AI Pipeline — HPC'ye Gondermeden Once Yapilacaklar

**Son guncelleme:** 2026-04-30  
**Hazirlik durumu:** 17 kritik bug duzeltildi, HPC'ye hazir

---

## BOLUM 1 — Lokal Dogrulama (Gondermeden Once)

### 1.1 Syntax ve Import Kontrolleri

- [ ] **Python syntax check** — tum kaynak dosyalar PASS olmali:
  ```bash
  python -m py_compile main.py run_complete_pipeline.py
  find pfaz_modules core_modules analysis_modules physics_modules -name "*.py" | \
      xargs -I{} python -m py_compile {}
  ```

- [ ] **Health check** calistir:
  ```bash
  python scripts/health_check.py
  # Beklenen: 5/5 modul OK, veri dosyasi bulundu
  ```

- [ ] **Smoke testler** calistir:
  ```bash
  pytest tests/test_smoke -v
  # Beklenen: 8/8 PASS
  ```

- [ ] **Import testleri** calistir:
  ```bash
  pytest tests/test_integration -v
  # Beklenen: tum import testleri PASS
  ```

### 1.2 Kritik Bug Kontrolu

- [ ] **Nested parallelism** — `parallel_ai_trainer.py` icinde `_inner_n_jobs()` helper mevcut:
  ```bash
  grep -n "_inner_n_jobs\|_PFAZ_PARALLEL_ACTIVE" pfaz_modules/pfaz02_ai_training/parallel_ai_trainer.py | head -10
  ```

- [ ] **TF clear_session** — training loop `finally` blogu mevcut:
  ```bash
  grep -n "clear_session\|empty_cache\|gc.collect" pfaz_modules/pfaz02_ai_training/parallel_ai_trainer.py
  ```

- [ ] **XGBoost GPU API** — version check mevcut:
  ```bash
  grep -n "xgb_version\|device.*cuda\|gpu_hist" pfaz_modules/pfaz02_ai_training/gpu_optimization.py
  ```

- [ ] **Multiprocessing spawn** — `main.py` ve `run_complete_pipeline.py` icinde:
  ```bash
  grep -n "set_start_method.*spawn" main.py run_complete_pipeline.py
  ```

- [ ] **input() yasak** — HPC'de donduracak `input()` cagrilari yok olmali:
  ```bash
  grep -rn "= input(" pfaz_modules/ main.py | grep -v "isatty\|#"
  # Sonuc bos olmali
  ```

- [ ] **UTF-8 encoding** — tum dosyalar UTF-8:
  ```bash
  python -c "
  import glob
  errors = []
  for fp in glob.glob('pfaz_modules/**/*.py', recursive=True):
      try:
          open(fp, encoding='utf-8').read()
      except UnicodeDecodeError:
          errors.append(fp)
  print('UTF-8 hatali dosya:', errors if errors else 'YOK')
  "
  ```

- [ ] **bare except** — `except:` (tipsiz) kalmamis olmali:
  ```bash
  grep -rn "^\s*except:\s*$" pfaz_modules/ core_modules/ analysis_modules/
  # Sonuc bos olmali
  ```

### 1.3 Veri ve Konfigurasyonu

- [ ] `data/aaa2.txt` mevcut ve okunabilir (267 satirdan fazla olmali)
- [ ] `config.json` gecerli JSON:
  ```bash
  python -c "import json; json.load(open('config.json'))"
  ```
- [ ] `pfaz_status.json` mevcut (yoksa `{}` ile olustur)
- [ ] `outputs/` alt dizinleri mevcut:
  ```bash
  ls outputs/generated_datasets outputs/trained_models outputs/anfis_models \
     outputs/reports outputs/visualizations outputs/thesis outputs/ensemble_results \
     outputs/unknown_predictions outputs/cross_model_analysis outputs/aaa2_results
  ```

---

## BOLUM 2 — Dosya Transferi

### 2.1 Transfer Edilecek Dosyalar

- [ ] **Proje kodu** — tam dizin transferi (rsync veya scp):
  ```bash
  rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='outputs/' --exclude='.venv/' \
        nucdatav1/ user@hpc.cluster.edu:/home/user/nucdatav1/
  ```

- [ ] `data/aaa2.txt` — veri dosyasi transfer edildi
- [ ] `config.json` — konfigurasyonu transfer edildi
- [ ] `requirements-hpc.txt` — HPC bagimliliklari transfer edildi
- [ ] `hpc_slurm_job.sh` — SLURM script transfer edildi

### 2.2 Transfer Sonrasi Dogrulama

- [ ] Hedef makinede dosya sayisi esit:
  ```bash
  find /home/user/nucdatav1/pfaz_modules -name "*.py" | wc -l
  # Lokal ile esit olmali
  ```

- [ ] `data/aaa2.txt` MD5 eslesme:
  ```bash
  # Lokal:
  md5sum data/aaa2.txt
  # HPC:
  md5sum /home/user/nucdatav1/data/aaa2.txt
  # Ayni olmali
  ```

---

## BOLUM 3 — HPC Ortam Kurulumu

### 3.1 Python Ortami

- [ ] Python versiyonu uygun (3.8 minimum, 3.11 onerilir):
  ```bash
  python --version
  ```

- [ ] Virtual environment veya conda ortami olustur:
  ```bash
  python -m venv venv_nucdatav1
  source venv_nucdatav1/bin/activate
  # veya
  conda create -n nucdatav1 python=3.11
  conda activate nucdatav1
  ```

- [ ] HPC bagimliliklari yukle (`auto-sklearn` **OLMADAN**):
  ```bash
  pip install -r requirements-hpc.txt
  # DIKKAT: requirements.txt degil, requirements-hpc.txt kullan!
  ```

- [ ] Kritik paketler yuklu mu kontrol:
  ```bash
  python -c "import numpy, pandas, sklearn, xgboost, tensorflow; print('OK')"
  python -c "import optuna, joblib, scipy, matplotlib; print('OK')"
  ```

### 3.2 SLURM Script Konfigurasyonu

`hpc_slurm_job.sh` dosyasini HPC'ye ozgu guncelle:

- [ ] `#SBATCH --partition=` — dogru partition adinla guncelle (orn: `gpu`, `compute`)
- [ ] `#SBATCH --nodelist=` — node kisitlama varsa guncelle
- [ ] `#SBATCH --mail-user=` — kendi email adresiyle guncelle
- [ ] `#SBATCH --time=` — beklenen sure (varsayilan: 48 saat = `48:00:00`)
- [ ] `#SBATCH --mem=` — yeterli RAM (min 32GB, onerilir 64GB: `--mem=64G`)
- [ ] `#SBATCH --gres=gpu:1` — GPU varsa, yoksa sil
- [ ] `VENV_PATH` degiskenini kendi venv yoluyla guncelle
- [ ] `PROJECT_DIR` degiskenini kendi proje yoluyla guncelle

### 3.3 Ortam Degiskenleri

SLURM scripti otomatik set eder. Manuel test icin:

```bash
export HPC_MODE=1
export PARALLEL_TRAINING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export THESIS_AUTHOR="Ahmet Kemal Acar"
export THESIS_SUPERVISOR="Prof. Dr. ..."
export THESIS_UNIVERSITY="..."
export THESIS_COMPILE_PDF=0
```

---

## BOLUM 4 — HPC'de On-Test

### 4.1 Mini Test Kosturma

Tam job gondermeden once kucuk test:

- [ ] Health check HPC'de calistir:
  ```bash
  cd /home/user/nucdatav1
  python scripts/health_check.py
  ```

- [ ] Smoke testler HPC'de calistir:
  ```bash
  pytest tests/test_smoke -v
  ```

- [ ] Import testleri HPC'de:
  ```bash
  pytest tests/test_integration -v
  ```

- [ ] Kisa PFAZ01 testi (ilk 5 dataset):
  ```bash
  HPC_MODE=1 OMP_NUM_THREADS=1 python main.py --pfaz 1 --mode run
  # Hata yoksa iptal et (Ctrl+C), SLURM ile gonder
  ```

### 4.2 GPU Kontrolu (Varsa)

- [ ] GPU gorulabilir mi:
  ```bash
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
  ```

- [ ] XGBoost GPU testi:
  ```bash
  python -c "
  import xgboost as xgb
  v = tuple(int(x) for x in xgb.__version__.split('.')[:2])
  print('XGBoost', xgb.__version__, '— device param:', 'cuda' if v >= (2,0) else 'gpu_hist')
  "
  ```

---

## BOLUM 5 — SLURM Job Gonderimi

### 5.1 Job Gondermeden Once

- [ ] Output log dizini mevcut:
  ```bash
  mkdir -p logs
  ```

- [ ] `pfaz_status.json` sifirla (temiz baslangic istiyorsan):
  ```bash
  echo '{}' > pfaz_status.json
  ```
  veya devam ettirmek istiyorsan dokunma.

- [ ] `outputs/` dizinleri HPC'de mevcut:
  ```bash
  mkdir -p outputs/generated_datasets outputs/trained_models outputs/anfis_models \
            outputs/reports outputs/visualizations outputs/thesis outputs/ensemble_results \
            outputs/unknown_predictions outputs/cross_model_analysis outputs/aaa2_results
  ```

### 5.2 Job Gonderimi

- [ ] SLURM scripti yuklenebilir:
  ```bash
  sbatch --test-only hpc_slurm_job.sh
  # "would submit batch job" mesaji gorulmeli
  ```

- [ ] Job gonder:
  ```bash
  sbatch hpc_slurm_job.sh
  ```

- [ ] Job ID'yi kaydet:
  ```bash
  # Cikti: "Submitted batch job 123456"
  JOB_ID=123456
  ```

### 5.3 Job Izleme

- [ ] Job durumunu izle:
  ```bash
  squeue -u $USER
  squeue -j $JOB_ID
  ```

- [ ] Log dosyasini izle:
  ```bash
  tail -f logs/nucdatav1_${JOB_ID}.out
  ```

- [ ] Hata logunu kontrol et:
  ```bash
  tail -f logs/nucdatav1_${JOB_ID}.err
  ```

---

## BOLUM 6 — Job Tamamlandi mi Kontrol

### 6.1 Cikti Dosyalari Kontrolu

- [ ] PFAZ01 ciktilari:
  ```bash
  ls outputs/generated_datasets/ | wc -l
  # 848 olmali (veya kaldigi yerden devam ettiyse daha az)
  ```

- [ ] PFAZ02 ciktilari:
  ```bash
  ls outputs/trained_models/ | wc -l
  # Dataset sayisi kadar klasor olmali
  ```

- [ ] Ana Excel raporu:
  ```bash
  ls -lh outputs/reports/THESIS_COMPLETE_RESULTS_*.xlsx
  ```

- [ ] Gorsellestime dosyalari:
  ```bash
  ls outputs/visualizations/*.png | wc -l
  ```

### 6.2 Pipeline Log Analizi

- [ ] Log parser calistir:
  ```bash
  python scripts/log_parser.py
  ```

- [ ] PFAZ tamamlanma durumu:
  ```bash
  python -c "import json; s=json.load(open('pfaz_status.json')); [print(k,v['status']) for k,v in s.items()]"
  ```

- [ ] Uyari ozeti:
  ```bash
  python -c "
  import json
  with open('outputs/pipeline_warnings.json') as f:
      w = json.load(f)
  print('Uyari sayisi:', len(w))
  for item in w[-10:]:
      print(' -', item)
  "
  ```

---

## BOLUM 7 — Sorun Giderme

### Job baslamadan oldu / kaynak yetersiz

```bash
# Daha az kaynak ile test:
sbatch --ntasks=1 --cpus-per-task=8 --mem=32G --time=06:00:00 hpc_slurm_job.sh
```

### Checkpoint'ten devam etme

```bash
# pfaz_status.json'da failed olan fazlari pending yap:
python -c "
import json
s = json.load(open('pfaz_status.json'))
for k in s:
    if s[k].get('status') == 'failed':
        s[k]['status'] = 'pending'
        print('Reset:', k)
json.dump(s, open('pfaz_status.json','w'), indent=2)
"
sbatch hpc_slurm_job.sh
```

### PFAZ01 yeniden baslat (resume):

```bash
HPC_MODE=1 python main.py --pfaz 1 --mode resume
```

### Import hatasi gorursen

```bash
python scripts/health_check.py
# Hangi modul eksikse:
pip install <paket>
```

---

## Son Kontrol Ozeti

| Kontrol | Durum |
|---------|-------|
| Syntax check (128 dosya) | |
| Health check 5/5 OK | |
| Smoke testler 8/8 PASS | |
| UTF-8 encoding OK | |
| nested parallelism fix mevcut | |
| TF clear_session fix mevcut | |
| XGBoost 2.0 fix mevcut | |
| spawn set edildi | |
| input() guvenli | |
| SLURM script guncellendi | |
| Veri dosyasi transfer edildi | |
| requirements-hpc.txt yuklendi | |

**Tum kutular isaretlendikten sonra job gonderilebilir.**

---

*Hazirlandigi tarih: 2026-04-30*  
*Referans: V10_QA_BUG_REPORT.md, CODING_RULES.md, hpc_slurm_job.sh*
