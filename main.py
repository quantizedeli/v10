"""
Nuclear Physics AI Project - Main Orchestrator
===============================================
Nükleer Fizik AI Projesi - Ana Yönetim Sistemi

PFAZ (Project Phase) Management System
- PFAZ 1: Dataset Generation
- PFAZ 2: AI Model Training
- PFAZ 3: ANFIS Training
- PFAZ 4: Unknown Nuclei Predictions
- PFAZ 5: Cross-Model Analysis
- PFAZ 6: Final Reporting
- PFAZ 7: Ensemble & Meta-Models
- PFAZ 8: Visualization & Dashboard
- PFAZ 9: AAA2 Control Group Analysis
- PFAZ 10: Thesis Compilation
- PFAZ 11: Production Deployment
- PFAZ 12: Advanced Analytics
- PFAZ 13: AutoML Integration

Features:
- Resume: Kaldığı yerden devam et
- Pass: Fazı atla
- Update: Mevcut sonuçları güncelle
- Otomatik kütüphane yükleme
- Paralel işlem desteği
- Progress tracking

Author: Nuclear Physics AI Research Team
Version: 6.0.0 - Production Ready
Date: November 2025
"""

import os
import sys
import json
import logging
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*joblib.*')

# Windows konsol encoding fix
if sys.platform == 'win32':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass  # Ignore if encoding fix fails

# ============================================================================
# OTOMATIK KÜTÜPHANE YÜKLEME
# ============================================================================

class AutoInstaller:
    """Otomatik kütüphane yükleyici"""
    
    REQUIRED_PACKAGES = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'xgboost': 'xgboost',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'openpyxl': 'openpyxl',
        'plotly': 'plotly',
        'joblib': 'joblib',
        'tqdm': 'tqdm',
    }
    
    @staticmethod
    def check_and_install():
        """Eksik kütüphaneleri kontrol et ve yükle"""
        print("\n" + "="*80)
        print("[CHECK] Kütüphane kontrolü yapılıyor...")
        print("="*80)

        missing = []
        installed = []

        for package, import_name in AutoInstaller.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(import_name)
                installed.append(package)
                print(f"[OK] {package}")
            except ImportError:
                missing.append(package)
                print(f"[MISSING] {package} - YÜKLENMESİ GEREKİYOR")

        if missing:
            print(f"\n[WARNING] {len(missing)} eksik kutüphane bulundu!")
            print(f"Yuklenecekler: {', '.join(missing)}")

            # HPC mode: auto-install not allowed
            if os.environ.get('HPC_MODE') or 'SLURM_JOB_ID' in os.environ:
                print(f"[HPC] Eksik kutüphaneler: {', '.join(missing)}")
                print("HPC ortaminda otomatik kurulum yapilmaz. Lütfen onceden:")
                print("  module load python/3.11")
                print("  python -m venv ~/v10_env")
                print("  source ~/v10_env/bin/activate")
                print("  pip install -r requirements.txt")
                sys.exit(1)

            # Non-interactive mode: cannot prompt
            if not sys.stdin.isatty():
                print(f"[ERROR] Eksik: {missing}. Non-interactive modda otomatik install yapilmaz.")
                print(f"  pip install {' '.join(missing)}")
                sys.exit(1)

            response = input("\nOtomatik yuklensin mi? (E/h): ").lower()
            if response == 'e' or response == '':
                for package in missing:
                    print(f"\n[INSTALL] {package} yukleniyor...")
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package, "-q"
                        ])
                        print(f"[OK] {package} basariyla yuklendi")
                    except Exception as e:
                        print(f"[ERROR] {package} yuklenemedi: {e}")
                print("\n[SUCCESS] Kutüphane yukleme tamamlandi!")
                print("[RESTART] Lütfen programi yeniden baslatin...")
                sys.exit(0)
            else:
                print("\n[WARNING] Kutüphaneler yuklenmedi. Lütfen manuel yukleyin:")
                print(f"pip install {' '.join(missing)}")
                sys.exit(1)
        else:
            print(f"\n[SUCCESS] Tüm kütüphaneler mevcut! ({len(installed)}/{len(AutoInstaller.REQUIRED_PACKAGES)})")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(
    max_bytes: int = 200 * 1024 * 1024,  # 200 MB per log file
    backup_count: int = 5,               # En fazla 5 yedek (toplam ~1 GB)
):
    """
    Logging konfigürasyonu — dönen (rotating) dosya ile boyut kontrolü.

    Her log dosyası en fazla max_bytes büyür.  Limit dolunca dosya
    main_TIMESTAMP.log.1, .2, ... şeklinde yedeklenir; 5 yedekten fazlası silinir.
    Konsola da yazılır (StreamHandler).
    Ayrıca WarningTracker tüm WARNING/ERROR mesajlarını yapılandırılmış JSON'a yazar.
    """
    from logging.handlers import RotatingFileHandler

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = log_dir / f'main_{timestamp}.log'

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Rotating file handler — 200 MB × 5 yedek = max ~1 GB
    fh = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8',
    )
    fh.setFormatter(fmt)

    # Konsol handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Önceki handler'ları temizle (çift çıktıyı önler)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)

    # --- WarningTracker: WARNING/ERROR'ları JSON + Excel'e yaz ---
    try:
        from utils.warning_tracker import WarningTracker
        _wt = WarningTracker(
            json_path='outputs/pipeline_warnings.json',
            excel_path='outputs/pipeline_warnings_report.xlsx',
        )
        _wt.attach(root)
        logging.info("[LOG] WarningTracker aktif: outputs/pipeline_warnings.json")
    except Exception as _wt_e:
        logging.warning(f"[LOG] WarningTracker başlatılamadı: {_wt_e}")

    logging.info(f"[LOG] Rotating log: {log_file} (max {max_bytes//1024//1024} MB x {backup_count} yedek = toplam ~{max_bytes//1024//1024*(backup_count+1)//1024} GB)")
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# PFAZ STATUS MANAGER
# ============================================================================

class PFAZStatusManager:
    """PFAZ faz durum yöneticisi"""
    
    def __init__(self, status_file='pfaz_status.json'):
        self.status_file = Path(status_file)
        self.status = self._load_status()
    
    def _load_status(self) -> Dict:
        """Durum dosyasını yükle"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._create_default_status()
    
    def _create_default_status(self) -> Dict:
        """Varsayılan durum oluştur"""
        return {
            'pfaz_01': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_02': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_03': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_04': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_05': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_06': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_07': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_08': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_09': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_10': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_11': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_12': {'status': 'pending', 'progress': 0, 'last_update': None},
            'pfaz_13': {'status': 'pending', 'progress': 0, 'last_update': None},
        }
    
    def save_status(self):
        """Durumu kaydet"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2)
    
    def update_pfaz(self, pfaz_id: str, status: str, progress: int):
        """PFAZ durumunu güncelle"""
        pfaz_key = f'pfaz_{pfaz_id:02d}' if isinstance(pfaz_id, int) else pfaz_id
        self.status[pfaz_key] = {
            'status': status,  # pending, running, completed, failed, skipped
            'progress': progress,
            'last_update': datetime.now().isoformat()
        }
        self.save_status()
    
    def get_pfaz_status(self, pfaz_id: str) -> Dict:
        """PFAZ durumunu al"""
        pfaz_key = f'pfaz_{pfaz_id:02d}' if isinstance(pfaz_id, int) else pfaz_id
        return self.status.get(pfaz_key, {'status': 'pending', 'progress': 0})
    
    def can_resume(self, pfaz_id: str) -> bool:
        """PFAZ resume edilebilir mi?"""
        status = self.get_pfaz_status(pfaz_id)
        return status['status'] == 'running' and status['progress'] > 0
    
    def print_status(self):
        """Durum özeti yazdır"""
        print("\n" + "="*80)
        print("[STATUS] PFAZ DURUM OZETI")
        print("="*80)

        for pfaz_id, info in self.status.items():
            status_icon = {
                'pending': '[WAIT]',
                'running': '[RUN]',
                'completed': '[DONE]',
                'failed': '[FAIL]',
                'skipped': '[SKIP]'
            }.get(info['status'], '[???]')

            print(f"{status_icon} {pfaz_id.upper()}: {info['status']:<10} ({info['progress']:>3}%) "
                  f"- Son güncelleme: {info['last_update'] or 'Hiç'}")
        print("="*80)

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class NuclearPhysicsAIOrchestrator:
    """Ana orkestratör - Tüm PFAZ fazlarını yönetir"""
    
    def __init__(self, config_path='config.json'):
        """Orkestratörü başlat"""
        self.project_root = Path(__file__).parent
        self.config_path = config_path
        self.config = self._load_config()
        self.status_manager = PFAZStatusManager()

        # Çıktı dizinleri
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # PFAZ çıktı dizinleri
        self.pfaz_outputs = {
            1: self.output_dir / 'generated_datasets',
            2: self.output_dir / 'trained_models',
            3: self.output_dir / 'anfis_models',
            4: self.output_dir / 'unknown_predictions',
            5: self.output_dir / 'cross_model_analysis',
            6: self.output_dir / 'reports',
            7: self.output_dir / 'ensemble_results',
            8: self.output_dir / 'visualizations',
            9: self.output_dir / 'aaa2_results',
            10: self.output_dir / 'thesis',
            11: self.output_dir / 'production',
            12: self.output_dir / 'advanced_analytics',
            13: self.output_dir / 'automl_results'
        }
        
        for output_path in self.pfaz_outputs.values():
            output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("[START] NUCLEAR PHYSICS AI ORCHESTRATOR v6.0")
        logger.info("="*80)
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Output: {self.output_dir}")

    def _load_config(self) -> Dict:
        """Konfigürasyon yükle"""
        default_config = self._default_config()

        if Path(self.config_path).exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # Merge with defaults to ensure all required keys exist
            for key in default_config:
                if key not in loaded_config:
                    loaded_config[key] = default_config[key]

            # Ensure pfaz_config has all PFAZ entries
            if 'pfaz_config' in loaded_config:
                for pfaz_id in range(1, 14):
                    if pfaz_id not in loaded_config['pfaz_config']:
                        loaded_config['pfaz_config'][pfaz_id] = default_config['pfaz_config'][pfaz_id]
            else:
                loaded_config['pfaz_config'] = default_config['pfaz_config']

            return loaded_config

        return default_config
    
    def _default_config(self) -> Dict:
        """Varsayılan konfigürasyon"""
        return {
            'output_dir': 'outputs',
            'data_file': 'aaa2.txt',
            'pfaz_config': {
                1: {
                    'enabled': True,
                    'dataset_sizes': [75, 100, 150, 200, 'ALL'],
                    'scenarios': ['S70', 'S80'],
                    'targets': ['MM', 'QM'],
                    # Çoklu scaling ve sampling varyantları (her kombinasyon için ayrı dataset üretilir)
                    'scaling_methods': ['NoScaling', 'Standard', 'Robust', 'MinMax'],
                    'sampling_methods': ['Random', 'Stratified'],
                    'feature_sets': None,  # None = hedef-bazli SHAP setleri
                },
                2: {'enabled': True, 'n_configs': 50, 'parallel': True,
                    'use_hyperparameter_tuning': False, 'use_model_validation': True, 'use_advanced_models': False},
                3: {'enabled': True, 'n_configs': 8, 'use_matlab': False,
                    'use_config_manager': True, 'use_adaptive_strategy': False,
                    'use_performance_analyzer': True, 'save_datasets': True},
                4: {'enabled': True, 'test_split': 0.2},
                5: {'enabled': True, 'targets': ['MM', 'QM'], 'use_best_model_selector': True},
                6: {'enabled': True, 'formats': ['excel', 'latex'], 'use_excel_charts': True, 'use_latex_generator': True},
                7: {'enabled': True, 'methods': ['voting', 'stacking']},
                8: {'enabled': True, 'n_plots': 80},
                9: {'enabled': True, 'monte_carlo': True},
                10: {'enabled': True, 'compile_pdf': True},
                11: {'enabled': False, 'deploy': False},
                12: {'enabled': True, 'shap_analysis': True},
                13: {
                    'enabled': True,
                    'n_trials': 30,           # Optuna trial sayısı (her model tipi × hedef için)
                    'model_types': ['rf', 'xgb', 'lgb'],  # Hangi model tipleri optimize edilsin
                }
            },
            'parallel': {'enabled': True, 'n_jobs': -1},
            'gpu': {'enabled': True, 'device': 0},
            'logging': {'level': 'INFO', 'file': 'logs/main.log'}
        }
    
    # ========================================================================
    # PFAZ 1: DATASET GENERATION
    # ========================================================================
    
    def run_pfaz_01(self, mode='run', **kwargs):
        """
        PFAZ 1: Dataset Generation

        Args:
            mode: 'run', 'resume', 'update', 'pass'
        """
        pfaz_id = 1
        logger.info("\n" + "="*80)
        logger.info(f"[PFAZ {pfaz_id}] DATASET GENERATION")
        logger.info("="*80)

        if mode == 'pass':
            logger.info("[SKIP] PFAZ 1 atlanıyor...")
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            status = self.status_manager.get_pfaz_status(f'pfaz_{pfaz_id:02d}')
            if status['status'] != 'running':
                logger.warning("[WARNING] Resume edilecek bir işlem yok, normal çalıştırılıyor...")
                mode = 'run'

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 10)

            # Import
            from pfaz_modules.pfaz01_dataset_generation.dataset_generation_pipeline_v2 import (
                DatasetGenerationPipelineV2
            )

            self.status_manager.update_pfaz(pfaz_id, 'running', 30)

            # Initialize
            config = self.config['pfaz_config'][pfaz_id]
            data_file = self.config.get('data_file', 'aaa2.txt')
            data_file_path = Path(data_file)
            if not data_file_path.is_absolute():
                data_file_path = self.project_root / data_file_path

            # Scaling ve Sampling kombinasyonları
            # Config'den al, yoksa default kombinasyonları kullan
            scaling_methods = config.get(
                'scaling_methods',
                config.get('scaling_methods_list', ['NoScaling', 'Standard', 'Robust'])
            )
            sampling_methods = config.get(
                'sampling_methods',
                config.get('sampling_methods_list', ['Random', 'Stratified'])
            )

            # Geriye dönük uyumluluk: tek değer verilmişse listeye çevir
            if isinstance(scaling_methods, str):
                scaling_methods = [scaling_methods]
            if isinstance(sampling_methods, str):
                sampling_methods = [sampling_methods]

            logger.info(f"[PFAZ 1] Scaling varyantları: {scaling_methods}")
            logger.info(f"[PFAZ 1] Sampling varyantları: {sampling_methods}")
            logger.info(f"[PFAZ 1] Toplam kombinasyon: {len(scaling_methods) * len(sampling_methods)}")

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            # Her kombinasyon için pipeline çalıştır
            all_results = {}
            combo_count = 0
            total_combos = len(scaling_methods) * len(sampling_methods)

            for scaling in scaling_methods:
                for sampling in sampling_methods:
                    combo_count += 1
                    combo_name = f"{scaling}_{sampling}"
                    logger.info(f"\n[PFAZ 1] Kombinasyon {combo_count}/{total_combos}: {combo_name}")

                    pipeline = DatasetGenerationPipelineV2(
                        aaa2_txt_path=str(data_file_path),
                        output_dir=str(self.pfaz_outputs[pfaz_id]),
                        targets=config.get('targets', ['MM', 'QM']),
                        nucleus_counts=config.get('dataset_sizes', [75, 100, 150, 200, 'ALL']),
                        scenarios=config.get('scenarios', ['S70', 'S80']),
                        scaling=scaling,
                        sampling=sampling,
                        feature_sets=config.get('feature_sets', None),
                    )

                    try:
                        combo_results = pipeline.run_complete_pipeline()
                        all_results[combo_name] = {
                            'status': 'success',
                            'datasets_generated': combo_results.get('dataset_generation', {}).get('total_generated', 0)
                        }
                        logger.info(f"[OK] {combo_name}: {all_results[combo_name]['datasets_generated']} dataset")
                    except Exception as e_combo:
                        logger.error(f"[ERROR] {combo_name} başarısız: {e_combo}")
                        all_results[combo_name] = {'status': 'failed', 'error': str(e_combo)}

            results = {
                'status': 'completed',
                'combinations': all_results,
                'total_combinations': total_combos
            }

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 1 tamamlandı!")

            return results

        except Exception as e:
            logger.error(f"[ERROR] PFAZ 1 başarısız: {e}", exc_info=True)
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise
    
    # ========================================================================
    # PFAZ 2: AI MODEL TRAINING
    # ========================================================================
    
    def run_pfaz_02(self, mode='run', **kwargs):
        """PFAZ 2: AI Model Training"""
        pfaz_id = 2
        logger.info("\n" + "="*80)
        logger.info(f"[PFAZ {pfaz_id}] AI MODEL TRAINING")
        logger.info("="*80)

        if mode == 'pass':
            logger.info("[SKIP] PFAZ 2 atlanıyor...")
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _existing = list(self.pfaz_outputs[2].rglob('metrics_*.json'))
            if _existing:
                logger.info(f"[RESUME] PFAZ 2 ciktisi mevcut ({len(_existing)} metrics), atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 10)

            from pfaz_modules.pfaz02_ai_training.parallel_ai_trainer import ParallelAITrainer

            self.status_manager.update_pfaz(pfaz_id, 'running', 30)

            config = self.config['pfaz_config'][pfaz_id]

            # GPU manager: merkezi algilama + TF bellek yapilandirmasi
            from utils.gpu_manager import get_gpu_manager
            _gm = get_gpu_manager()
            _gm.configure_tf()
            _gpu_available = _gm.available
            _n_workers = _gm.optimal_workers(mode='ai')
            logger.info(f"[GPU] PFAZ2 gpu={_gpu_available}, workers={_n_workers}")

            trainer = ParallelAITrainer(
                datasets_dir=str(self.pfaz_outputs[1]),
                models_dir=str(self.pfaz_outputs[2]),
                training_config_path=str(self.project_root / 'pfaz_modules' / 'pfaz02_ai_training' / 'training_configs_50.json'),
                gpu_enabled=_gpu_available,
                n_workers=_n_workers,
                use_hyperparameter_tuning=config.get('use_hyperparameter_tuning', False),
                use_model_validation=config.get('use_model_validation', True),
                use_advanced_models=config.get('use_advanced_models', False)
            )

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            results = trainer.train_all_models_parallel(
                n_configs=config.get('n_configs', 50),
                use_parallel=config.get('parallel', True)
            )

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 2 tamamlandı!")

            # Otomatik temizlik: çok düşük skorlu dataset dizinleri
            cleanup_threshold = self.config.get('pfaz_config', {}).get(pfaz_id, {}).get(
                'cleanup_r2_threshold', None
            )
            if cleanup_threshold is not None:
                logger.info(f"[CLEANUP] Otomatik temizlik başlıyor (threshold={cleanup_threshold})...")
                cleanup = self.cleanup_failed_datasets(
                    val_r2_threshold=float(cleanup_threshold),
                    dry_run=False,
                )
                results['cleanup'] = cleanup

            return results

        except Exception as e:
            logger.error(f"[ERROR] PFAZ 2 başarısız: {e}", exc_info=True)
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise
    
    # ========================================================================
    # PFAZ 3: ANFIS TRAINING
    # ========================================================================
    
    def run_pfaz_03(self, mode='run', **kwargs):
        """PFAZ 3: ANFIS Training"""
        pfaz_id = 3
        logger.info("\n" + "="*80)
        logger.info(f"[PFAZ {pfaz_id}] ANFIS TRAINING")
        logger.info("="*80)

        if mode == 'pass':
            logger.info("[SKIP] PFAZ 3 atlanıyor...")
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 10)

            from pfaz_modules.pfaz03_anfis_training.anfis_parallel_trainer_v2 import ANFISParallelTrainerV2

            self.status_manager.update_pfaz(pfaz_id, 'running', 30)

            config = self.config['pfaz_config'][pfaz_id]

            from utils.gpu_manager import get_gpu_manager
            _gm = get_gpu_manager()
            _anfis_workers = _gm.optimal_workers(mode='anfis')
            logger.info(f"[GPU] PFAZ3 gpu={_gm.available}, workers={_anfis_workers}")

            trainer = ANFISParallelTrainerV2(
                datasets_dir=str(self.pfaz_outputs[1]),
                output_dir=str(self.pfaz_outputs[3]),
                n_workers=_anfis_workers,
                gpu_enabled=_gm.available,
                use_config_manager=config.get('use_config_manager', True),
                use_adaptive_strategy=config.get('use_adaptive_strategy', False),
                use_performance_analyzer=config.get('use_performance_analyzer', True),
                save_datasets=config.get('save_datasets', True)
            )

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            results = trainer.train_all_anfis_parallel(
                n_configs=config.get('n_configs', 8)
            )

            # Generate kernel usage report
            if config.get('save_datasets', True):
                trainer.generate_kernel_usage_report()

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 3 tamamlandı!")

            return results

        except Exception as e:
            logger.error(f"[ERROR] PFAZ 3 başarısız: {e}", exc_info=True)
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise
    
    # ========================================================================
    # PFAZ 4-13: DİĞER FAZLAR (Benzer yapı)
    # ========================================================================
    
    def run_pfaz_04(self, mode='run', **kwargs):
        """PFAZ 4: Unknown Nuclei Predictions"""
        pfaz_id = 4
        logger.info(f"\n[PFAZ {pfaz_id}] UNKNOWN NUCLEI PREDICTIONS")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _key = self.pfaz_outputs[4] / 'Unknown_Nuclei_Results.xlsx'
            if _key.exists():
                logger.info(f"[RESUME] PFAZ 4 ciktisi mevcut, atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz04_unknown_predictions.unknown_nuclei_predictor import (
                UnknownNucleiPredictor
            )

            predictor = UnknownNucleiPredictor(
                ai_models_dir=str(self.pfaz_outputs[2]),
                anfis_models_dir=str(self.pfaz_outputs[3]),
                splits_dir=str(self.pfaz_outputs[1]),
                output_dir=str(self.pfaz_outputs[4])
            )
            results = predictor.predict_unknown_nuclei()

            # Generate standard Excel report
            predictor.generate_excel_report()

            # Generate AAA2 comparison Excel (orijinal değer vs tüm model tahminleri)
            data_file = self.config.get('data_file', 'aaa2.txt')
            data_file_path = Path(data_file)
            if not data_file_path.is_absolute():
                data_file_path = self.project_root / data_file_path
            predictor.generate_aaa2_comparison_excel(
                aaa2_txt_path=str(data_file_path),
                filename='AAA2_Original_vs_Predictions.xlsx'
            )
            logger.info("[OK] AAA2 karşılaştırma Excel raporu oluşturuldu")

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 4 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 4 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_05(self, mode='run', **kwargs):
        """PFAZ 5: Cross-Model Analysis"""
        pfaz_id = 5
        logger.info(f"\n[PFAZ {pfaz_id}] CROSS-MODEL ANALYSIS")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _key = self.pfaz_outputs[5] / 'MASTER_CROSS_MODEL_REPORT.xlsx'
            if _key.exists():
                logger.info(f"[RESUME] PFAZ 5 ciktisi mevcut, atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 30)

            # Use the complete cross-model analysis pipeline
            from pfaz_modules.pfaz05_cross_model.faz5_cross_model_analysis import CrossModelAnalysisPipeline

            config = self.config['pfaz_config'][pfaz_id]

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            # Initialize pipeline
            pipeline = CrossModelAnalysisPipeline(
                trained_models_dir=str(self.pfaz_outputs[2]),    # AI models from PFAZ2
                output_dir=str(self.pfaz_outputs[5]),
                anfis_models_dir=str(self.pfaz_outputs[3]),      # ANFIS models from PFAZ3
                datasets_dir=str(self.pfaz_outputs[1]),          # generated_datasets from PFAZ1
            )

            self.status_manager.update_pfaz(pfaz_id, 'running', 70)

            # Run complete analysis
            results = pipeline.run_complete_analysis()

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 5 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 5 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_06(self, mode='run', **kwargs):
        """PFAZ 6: Final Reporting"""
        pfaz_id = 6
        logger.info(f"\n[PFAZ {pfaz_id}] FINAL REPORTING")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _existing = list(self.pfaz_outputs[6].glob('THESIS_COMPLETE_RESULTS*.xlsx'))
            if _existing:
                logger.info(f"[RESUME] PFAZ 6 ciktisi mevcut ({_existing[0].name}), atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz06_final_reporting.pfaz6_final_reporting import FinalReportingPipeline

            config = self.config['pfaz_config'][pfaz_id]

            reporter = FinalReportingPipeline(
                output_dir=str(self.pfaz_outputs[6]),
                ai_models_dir=str(self.pfaz_outputs[2]),
                anfis_models_dir=str(self.pfaz_outputs[3]),
                use_excel_charts=config.get('use_excel_charts', True),
                use_latex_generator=config.get('use_latex_generator', True)
            )
            # Pass aaa2.txt path for isotope chain analysis
            _aaa2_f = self.config.get('data_file', 'aaa2.txt')
            _aaa2_p = Path(_aaa2_f) if Path(_aaa2_f).is_absolute() else self.project_root / _aaa2_f
            if not _aaa2_p.exists():
                _aaa2_p = self.project_root / 'data' / 'aaa2.txt'
            reporter.aaa2_txt_path = str(_aaa2_p)
            # Pass PFAZ9 output dir for Monte Carlo summary
            reporter.pfaz9_output_dir = str(self.pfaz_outputs[9])
            # Pass PFAZ13 output dir for AutoML improvements sheet
            reporter.pfaz13_output_dir = str(self.pfaz_outputs[13])
            results = reporter.run_complete_pipeline()

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 6 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 6 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_07(self, mode='run', **kwargs):
        """PFAZ 7: Ensemble & Meta-Models"""
        pfaz_id = 7
        logger.info(f"\n[PFAZ {pfaz_id}] ENSEMBLE & META-MODELS")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _existing = list(self.pfaz_outputs[7].glob('ensemble_report*.xlsx'))
            if not _existing:
                _existing = list(self.pfaz_outputs[7].glob('ensemble_comparison*.xlsx'))
            if _existing:
                logger.info(f"[RESUME] PFAZ 7 ciktisi mevcut ({_existing[0].name}), atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz07_ensemble.pfaz7_complete_ensemble_pipeline import (
                pfaz7_complete_pipeline
            )

            results = pfaz7_complete_pipeline(
                trained_models_dir=str(self.pfaz_outputs[2]),
                anfis_models_dir=str(self.pfaz_outputs[3]),
                datasets_dir=str(self.pfaz_outputs[1]),
                output_dir=str(self.pfaz_outputs[7]),
            )

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 7 tamamlandı!")
            return results if results else {'status': 'completed'}
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 7 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise
    
    def run_pfaz_08(self, mode='run', **kwargs):
        """PFAZ 8: Visualization & Dashboard"""
        pfaz_id = 8
        logger.info(f"\n[PFAZ {pfaz_id}] VISUALIZATION & DASHBOARD")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _png_files = list(self.pfaz_outputs[8].glob('*.png'))
            _json_files = list(self.pfaz_outputs[8].glob('comparison_report.json'))
            if _png_files or _json_files:
                logger.info(f"[RESUME] PFAZ 8 ciktisi mevcut ({len(_png_files)} PNG), atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 30)
            from pfaz_modules.pfaz08_visualization.visualization_master_system import (
                MasterVisualizationSystem
            )

            viz_system = MasterVisualizationSystem(output_dir=str(self.pfaz_outputs[8]))
            results = viz_system.generate_all_visualizations(project_data={})

            # PFAZ 8 - THESIS CHARTS: 300 DPI PNG + HTML (single panel per file)
            self.status_manager.update_pfaz(pfaz_id, 'running', 60)
            try:
                from pfaz_modules.pfaz08_visualization.pfaz8_thesis_charts import ThesisChartGenerator
                thesis_gen = ThesisChartGenerator(str(self.pfaz_outputs[8]), project_root=str(self.project_root))
                thesis_result = thesis_gen.run_all()
                n_thesis = thesis_result.get('total', 0)
                logger.info(f"[OK] Thesis charts: {thesis_result.get('png',0)} PNG + {thesis_result.get('html',0)} HTML = {n_thesis} dosya")
                if results and isinstance(results, dict):
                    results['thesis_files'] = n_thesis
            except Exception as te:
                logger.warning(f"[WARNING] Thesis charts failed (devam): {te}")

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 8 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 8 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_09(self, mode='run', **kwargs):
        """PFAZ 9: AAA2 Control Group Analysis"""
        pfaz_id = 9
        logger.info(f"\n[PFAZ {pfaz_id}] AAA2 CONTROL GROUP ANALYSIS")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _existing = list(self.pfaz_outputs[9].glob('AAA2_Complete_*.xlsx'))
            if _existing:
                logger.info(f"[RESUME] PFAZ 9 ciktisi mevcut ({len(_existing)} dosya), atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            # TF GPU: DNN inference otomatik GPU kullanir
            from utils.gpu_manager import get_gpu_manager
            _gm9 = get_gpu_manager()
            _gm9.configure_tf()
            logger.info(f"[GPU] PFAZ9 gpu={_gm9.available} (DNN inference)")

            from pfaz_modules.pfaz09_aaa2_monte_carlo.aaa2_control_group_complete_v4 import (
                AAA2ControlGroupAnalyzerComplete
            )

            _aaa2_root = self.project_root / 'aaa2.txt'
            _aaa2_data = self.project_root / 'data' / 'aaa2.txt'
            _aaa2_path = str(_aaa2_root if _aaa2_root.exists() else _aaa2_data)
            analyzer = AAA2ControlGroupAnalyzerComplete(
                pfaz01_output_path=str(self.pfaz_outputs[1] / 'AAA2_enriched_all_nuclei.csv'),
                aaa2_txt_path=_aaa2_path,
                trained_models_dir=str(self.pfaz_outputs[2]),
                output_dir=str(self.pfaz_outputs[9])
            )
            results = analyzer.run_complete_pfaz9_pipeline(
                targets=['MM', 'QM']
            )

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 9 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 9 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_10(self, mode='run', **kwargs):
        """PFAZ 10: Thesis Compilation"""
        pfaz_id = 10
        logger.info(f"\n[PFAZ {pfaz_id}] THESIS COMPILATION")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _key = self.pfaz_outputs[10] / 'execution_report.json'
            if not _key.exists():
                _key = self.pfaz_outputs[10] / 'main.tex'
            if _key.exists():
                logger.info(f"[RESUME] PFAZ 10 ciktisi mevcut, atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 20)
            from pfaz_modules.pfaz10_thesis_compilation.pfaz10_master_integration import (
                MasterThesisIntegration
            )

            config = self.config['pfaz_config'].get(pfaz_id, {})
            compile_pdf = config.get('compile_pdf', False)

            # Build metadata from config
            meta = {
                'title':       config.get('title',      'Machine Learning and ANFIS-Based Prediction of Nuclear Properties'),
                'subtitle':    config.get('subtitle',   'Magnetic Moments, Quadrupole Moments and Deformation Parameters'),
                'author':      config.get('author',     'Research Student'),
                'supervisor':  config.get('supervisor', 'Prof. Supervisor Name'),
                'university':  config.get('university', 'University Name'),
                'department':  config.get('department', 'Department of Physics'),
                'thesis_type': config.get('thesis_type','Master of Science'),
            }

            thesis = MasterThesisIntegration(
                project_dir=str(self.output_dir),
                output_dir=str(self.pfaz_outputs[pfaz_id]),
                pfaz_outputs={k: str(v) for k, v in self.pfaz_outputs.items()},
                metadata=meta,
            )

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            results = thesis.execute_full_pipeline(compile_pdf=compile_pdf)

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 10 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 10 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_11(self, mode='run', **kwargs):
        """PFAZ 11: Production Deployment (DEFERRED - Ertelendi)"""
        pfaz_id = 11

        # PFAZ11 kullanıcı talebi doğrultusunda ertelendi
        logger.info(f"\n[PFAZ {pfaz_id}] PRODUCTION DEPLOYMENT - DEFERRED")
        logger.info("[INFO] PFAZ11 proje tamamlandıktan sonra için ertelenmiştir.")
        logger.info("[INFO] Şu anda PFAZ11 otomatik olarak atlanacaktır.")

        # Otomatik olarak skip et
        self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
        return {
            'status': 'skipped',
            'reason': 'PFAZ11 deferred per user request - will be implemented after project completion',
            'message': 'PFAZ11 ertelendi - proje tamamen bittikten sonra yapılacak'
        }

    def run_pfaz_12(self, mode='run', **kwargs):
        """PFAZ 12: Advanced Analytics — istatistiksel model karşılaştırması"""
        pfaz_id = 12
        logger.info(f"\n[PFAZ {pfaz_id}] ADVANCED ANALYTICS")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _key = self.pfaz_outputs[12] / 'statistical_tests' / 'pfaz12_statistical_tests.xlsx'
            if not _key.exists():
                _key_alt = list(self.pfaz_outputs[12].glob('*.xlsx'))
                _key = _key_alt[0] if _key_alt else _key
            if _key.exists():
                logger.info(f"[RESUME] PFAZ 12 ciktisi mevcut, atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 20)
            from pfaz_modules.pfaz12_advanced_analytics.statistical_testing_suite import (
                StatisticalTestingSuite
            )
            from pfaz_modules.pfaz12_advanced_analytics.bayesian_model_comparison import (
                BayesianModelComparison
            )
            import json, numpy as np
            from pathlib import Path

            output_dir = Path(str(self.pfaz_outputs[12]))
            output_dir.mkdir(parents=True, exist_ok=True)

            ai_models_dir = self.pfaz_outputs[2]

            # ---- Collect Val R² scores per model type from PFAZ2 metrics ----
            model_scores: dict = {}  # {model_type: [r2_val, ...]}
            targets = ['MM', 'QM']

            if ai_models_dir.exists():
                for metrics_file in ai_models_dir.rglob('metrics_*.json'):
                    try:
                        with open(metrics_file, encoding='utf-8') as f:
                            m = json.load(f)
                        val_r2 = m.get('val', {}).get('r2', None)
                        if val_r2 is None or np.isnan(val_r2) or val_r2 < -10:
                            continue
                        # Derive model type from path: .../RF/cfg/metrics_cfg.json
                        parts = metrics_file.parts
                        model_type = parts[-3] if len(parts) >= 3 else 'unknown'
                        model_scores.setdefault(model_type, []).append(float(val_r2))
                    except Exception:
                        continue

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            results = {'status': 'completed', 'n_models': len(model_scores)}

            if len(model_scores) >= 2:
                suite = StatisticalTestingSuite(
                    output_dir=str(output_dir / 'statistical_tests')
                )

                # Build scores_dict for pairwise/ANOVA tests
                scores_dict = {k: np.array(v) for k, v in model_scores.items() if len(v) >= 3}

                if len(scores_dict) >= 2:
                    # ANOVA across all model types
                    try:
                        anova_res = suite.one_way_anova(list(scores_dict.values()),
                                                        list(scores_dict.keys()))
                        logger.info(f"[PFAZ12] ANOVA p={anova_res.get('p_value', '?'):.4f}")
                    except Exception as e:
                        logger.warning(f"[PFAZ12] ANOVA skipped: {e}")

                    # Friedman test (non-parametric ANOVA)
                    try:
                        suite.friedman_test(list(scores_dict.values()),
                                            list(scores_dict.keys()))
                    except Exception as e:
                        logger.warning(f"[PFAZ12] Friedman skipped: {e}")

                    # Pairwise Wilcoxon
                    try:
                        suite.pairwise_wilcoxon(scores_dict)
                    except Exception as e:
                        logger.warning(f"[PFAZ12] Pairwise Wilcoxon skipped: {e}")

                    # Comprehensive comparison
                    try:
                        comp_res = suite.compare_models_comprehensive(scores_dict)
                        logger.info(f"[PFAZ12] Comprehensive comparison done: {len(comp_res)} results")
                    except Exception as e:
                        logger.warning(f"[PFAZ12] Comprehensive comparison skipped: {e}")

                    # Export to Excel
                    try:
                        excel_path = suite.export_to_excel(
                            str(output_dir / 'statistical_tests' / 'pfaz12_statistical_tests.xlsx')
                        )
                        results['excel_report'] = str(excel_path)
                        logger.info(f"[PFAZ12] Excel: {excel_path}")
                    except Exception as e:
                        logger.warning(f"[PFAZ12] Excel export skipped: {e}")

                # Bayesian comparison (best two models by mean R²)
                try:
                    sorted_models = sorted(scores_dict.items(),
                                          key=lambda x: np.mean(x[1]), reverse=True)
                    if len(sorted_models) >= 2:
                        m1_name, m1_scores = sorted_models[0]
                        m2_name, m2_scores = sorted_models[1]
                        min_len = min(len(m1_scores), len(m2_scores))
                        bayes = BayesianModelComparison()
                        bf_res = bayes.bayes_factor(1 - m1_scores[:min_len],
                                                    1 - m2_scores[:min_len])
                        logger.info(f"[PFAZ12] Bayes Factor {m1_name} vs {m2_name}: "
                                    f"BF={bf_res.get('bayes_factor', '?'):.3f}")
                        results['bayes_factor'] = bf_res
                except Exception as e:
                    logger.warning(f"[PFAZ12] Bayesian comparison skipped: {e}")

                results['model_types'] = list(scores_dict.keys())
                results['n_models_tested'] = len(scores_dict)
            else:
                logger.warning(f"[PFAZ12] Yeterli model verisi bulunamadı "
                               f"({len(model_scores)} model tipi). PFAZ2'nin tamamlanması gerekiyor.")

            # ------------------------------------------------------------------
            # NuclearMomentBandAnalyzer — bant + oruntu + sicrama + capraz kutle
            # ------------------------------------------------------------------
            _aaa2_root = self.project_root / 'aaa2.txt'
            _aaa2_data = self.project_root / 'data' / 'aaa2.txt'
            _aaa2_path = str(_aaa2_root if _aaa2_root.exists() else _aaa2_data)

            try:
                logger.info("\n[PFAZ12] NuclearMomentBandAnalyzer basliyor...")
                from pfaz_modules.pfaz12_advanced_analytics.nuclear_band_analyzer import (
                    NuclearMomentBandAnalyzer
                )

                band_output = str(self.pfaz_outputs[12] / 'band_analysis')
                band_analyzer = NuclearMomentBandAnalyzer(
                    data_path=_aaa2_path,
                    output_dir=band_output,
                    jump_sigma=2.0,
                    n_bands=6,
                )
                band_results = band_analyzer.run_all()
                results['band_analysis'] = {
                    'excel': band_results.get('excel_path'),
                    'targets': list(band_results.get('results', {}).keys()),
                }
                logger.info(f"[PFAZ12] Bant analizi tamamlandi: {band_results.get('excel_path')}")
            except Exception as _band_err:
                logger.warning(f"[PFAZ12] Bant analizi atlandı: {_band_err}")

            # ------------------------------------------------------------------
            # NuclearPatternAnalyzer — mevcut izotop/magic analizi
            # ------------------------------------------------------------------
            try:
                from pfaz_modules.pfaz12_advanced_analytics.nuclear_pattern_analyzer import (
                    NuclearPatternAnalyzer
                )
                pattern_output = str(self.pfaz_outputs[12] / 'nuclear_patterns')
                npa = NuclearPatternAnalyzer(
                    data_path=_aaa2_path,
                    output_dir=pattern_output,
                )
                npa_results = npa.run_all()
                results['nuclear_patterns'] = {
                    'excel': npa_results.get('excel_path'),
                }
                logger.info(f"[PFAZ12] Oruntu analizi tamamlandi: {npa_results.get('excel_path')}")
            except Exception as _npa_err:
                logger.warning(f"[PFAZ12] Pattern analizi atlandı: {_npa_err}")

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 12 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 12 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_13(self, mode='run', **kwargs):
        """
        PFAZ 13: AutoML — Good/Medium/Poor kategorilerinden 25'er dataset secilip
        AI ve ANFIS modelleri optimize edilir; deneysel veriyle karsilastirma yapilir.
        """
        pfaz_id = 13
        logger.info(f"\n[PFAZ {pfaz_id}] AUTOML INTEGRATION")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        if mode == 'resume':
            _key = self.pfaz_outputs[13] / 'automl_improvement_report.xlsx'
            if _key.exists():
                logger.info(f"[RESUME] PFAZ 13 ciktisi mevcut, atlaniyor.")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'resumed_from_existing'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 10)

            from utils.gpu_manager import get_gpu_manager
            _gm13 = get_gpu_manager()
            _gm13.configure_tf()
            _gpu13 = _gm13.available
            logger.info(f"[GPU] PFAZ13 gpu={_gpu13}")

            from pfaz_modules.pfaz13_automl.automl_optimizer import (
                AutoMLOptimizer, OPTUNA_AVAILABLE
            )
            import json, numpy as np, pandas as pd
            from pathlib import Path

            output_dir = Path(str(self.pfaz_outputs[13]))
            output_dir.mkdir(parents=True, exist_ok=True)

            if not OPTUNA_AVAILABLE:
                logger.warning("[PFAZ 13] optuna kurulu değil -> 'pip install optuna'")
                self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
                return {'status': 'skipped', 'reason': 'optuna not installed'}

            # ---- Find best dataset from PFAZ2 results -----------------------
            ai_models_dir = self.pfaz_outputs[2]
            best_by_target = {}   # {target: {'dataset_path': ..., 'r2': float, 'model': str}}

            if ai_models_dir.exists():
                for metrics_file in ai_models_dir.rglob('metrics_*.json'):
                    try:
                        with open(metrics_file, encoding='utf-8') as f:
                            m = json.load(f)
                        val_r2 = m.get('val', {}).get('r2', -999)
                        if val_r2 < -2 or np.isnan(val_r2):
                            continue
                        # dataset_path is the grandparent of the model dir
                        # structure: ai_models/{dataset}/{model_type}/{config}/metrics_*.json
                        ds_dir = metrics_file.parent.parent.parent  # dataset dir
                        target = m.get('target', '')
                        if not target:
                            continue
                        if target not in best_by_target or val_r2 > best_by_target[target]['r2']:
                            best_by_target[target] = {
                                'dataset_path': ds_dir,
                                'r2': float(val_r2),
                                'model': metrics_file.parent.parent.name,
                            }
                    except Exception:
                        continue

            self.status_manager.update_pfaz(pfaz_id, 'running', 30)

            # ---- Run AutoML per target --------------------------------------
            config = self.config['pfaz_config'].get(pfaz_id, {})
            n_trials     = int(config.get('n_trials', 30))
            model_types  = config.get('model_types', ['rf', 'xgb', 'lgb'])
            automl_results = {}

            for target, info in best_by_target.items():
                ds_path = Path(info['dataset_path'])
                train_csv = ds_path / 'train.csv'
                val_csv   = ds_path / 'val.csv'

                if not train_csv.exists() or not val_csv.exists():
                    logger.warning(f"[PFAZ13] CSV bulunamadi: {ds_path}")
                    continue

                try:
                    train_df = pd.read_csv(train_csv)
                    val_df   = pd.read_csv(val_csv)

                    # Identify feature and target columns
                    non_feat = {'NUCLEUS', 'MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]',
                                'Beta_2', 'MM', 'QM', 'MM_QM'}
                    feat_cols = [c for c in train_df.columns
                                 if c not in non_feat and train_df[c].dtype != object]

                    # Target column mapping
                    target_col_map = {
                        'MM':     'MAGNETIC MOMENT [µ]',
                        'QM':     'QUADRUPOLE MOMENT [Q]',
                        'Beta_2': 'Beta_2',
                        'MM_QM':  ['MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]'],
                    }
                    tcol = target_col_map.get(target, target)
                    if isinstance(tcol, list):
                        avail = [c for c in tcol if c in train_df.columns]
                        if not avail:
                            continue
                        y_train = train_df[avail].values
                        y_val   = val_df[avail].values
                    else:
                        if tcol not in train_df.columns:
                            continue
                        y_train = train_df[tcol].values
                        y_val   = val_df[tcol].values

                    X_train = train_df[feat_cols].values
                    X_val   = val_df[feat_cols].values

                    logger.info(f"\n[PFAZ13] {target}: {ds_path.name} | "
                                f"{len(X_train)} train, {X_train.shape[1]} features")

                    target_results = {}
                    for mtype in model_types:
                        try:
                            optimizer = AutoMLOptimizer(X_train, y_train, X_val, y_val,
                                                        model_type=mtype,
                                                        gpu_enabled=_gpu13)
                            study = optimizer.optimize(n_trials=n_trials)
                            optimizer.save_results(
                                study,
                                str(output_dir / f'{target}_{mtype}_automl.json')
                            )
                            target_results[mtype] = {
                                'best_r2': study.best_value,
                                'best_params': study.best_params,
                                'n_trials': len(study.trials),
                            }
                            logger.info(f"  [OK] {target}/{mtype}: best_R2={study.best_value:.4f}")
                        except Exception as e:
                            logger.warning(f"  [SKIP] {target}/{mtype}: {e}")

                    automl_results[target] = target_results

                except Exception as e:
                    logger.error(f"[PFAZ13] {target} dataset load error: {e}")

            self.status_manager.update_pfaz(pfaz_id, 'running', 88)

            # ---- AutoML Retraining Loop -------------------------------------
            # Düşük skorlu PFAZ2 modelleri tespit et ve Optuna ile yeniden eğit
            retraining_result = {}
            r2_threshold = float(config.get('retrain_threshold', 0.80))
            max_retrain  = int(config.get('max_retrain', 20))
            try:
                from pfaz_modules.pfaz13_automl.automl_retraining_loop import AutoMLRetrainingLoop
                _aaa2_root = self.project_root / 'aaa2.txt'
                _aaa2_data = self.project_root / 'data' / 'aaa2.txt'
                _aaa2_path = str(_aaa2_root if _aaa2_root.exists() else _aaa2_data) \
                             if (_aaa2_root.exists() or _aaa2_data.exists()) else None
                n_per_cat = int(config.get('n_per_category', 25))
                loop = AutoMLRetrainingLoop(
                    models_dir=str(self.pfaz_outputs[2]),
                    datasets_dir=str(self.pfaz_outputs[1]),
                    output_dir=str(output_dir),
                    r2_threshold=r2_threshold,
                    n_trials=n_trials,
                    model_types=model_types,
                    max_retrain=max_retrain,
                    n_per_category=n_per_cat,
                    aaa2_txt_path=_aaa2_path,
                    anfis_models_dir=str(self.pfaz_outputs[3]),
                    gpu_enabled=_gpu13,
                )
                retraining_result = loop.run()
                logger.info(f"[PFAZ13] Retraining loop tamamlandi: "
                            f"AI={retraining_result.get('ai_retrained',0)} "
                            f"(iyilesen={retraining_result.get('ai_improved',0)}), "
                            f"ANFIS={retraining_result.get('anfis_retrained',0)} "
                            f"(iyilesen={retraining_result.get('anfis_improved',0)})")
            except Exception as re_err:
                logger.warning(f"[PFAZ13] Retraining loop atlandı: {re_err}")

            self.status_manager.update_pfaz(pfaz_id, 'running', 95)

            # ---- Save summary -----------------------------------------------
            summary_path = output_dir / 'automl_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(automl_results, f, indent=2)
            logger.info(f"[PFAZ13] Summary: {summary_path}")

            if not best_by_target:
                logger.warning("[PFAZ13] PFAZ2 sonucu bulunamadı. Önce PFAZ2'yi çalıştırın.")

            results = {
                'status': 'completed',
                'targets_optimized': list(automl_results.keys()),
                'summary': str(summary_path),
                'n_trials': n_trials,
                'retraining': retraining_result,
            }

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 13 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 13 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise
    
    # ========================================================================
    # DATASET TEMİZLEME — Başarısız / düşük skorlu dataset dizinlerini sil
    # ========================================================================

    def cleanup_failed_datasets(
        self,
        val_r2_threshold: float = 0.0,
        dry_run: bool = True,
    ) -> Dict:
        """
        PFAZ2 eğitim sonuçlarına bakarak TÜMÜ başarısız olan dataset dizinlerini sil.

        Mantık:
          - Her dataset dizinindeki TÜM model metrik dosyalarına bak.
          - Bir dataset'teki tüm modellerin en iyi Val R² değeri threshold altında ise
            o dataset dizinini silme adayı say.
          - dry_run=True  → sadece listele, silme
          - dry_run=False → dizinleri kalıcı olarak sil (geri alınamaz!)

        Args:
            val_r2_threshold: Bu değerin altındaki best Val R² → başarısız (default: 0.0)
            dry_run: True = simülasyon, False = gerçek silme

        Returns:
            {'deleted': [...], 'kept': [...], 'total_freed_mb': float}
        """
        import shutil

        models_dir = self.pfaz_outputs[2]
        datasets_dir = self.pfaz_outputs[1]

        if not models_dir.exists():
            logger.warning(f"[CLEANUP] Modeller dizini bulunamadı: {models_dir}")
            return {'deleted': [], 'kept': [], 'total_freed_mb': 0.0}

        logger.info(f"\n[CLEANUP] Başarısız dataset taraması başlıyor...")
        logger.info(f"  Val R^2 eşiği: {val_r2_threshold}")
        logger.info(f"  Mod: {'Simülasyon (dry_run)' if dry_run else 'GERÇEK SİLME'}")

        # Her dataset için en iyi Val R² bul
        dataset_best: Dict[str, float] = {}
        for metrics_file in models_dir.rglob('metrics_*.json'):
            try:
                with open(metrics_file, encoding='utf-8') as f:
                    m = json.load(f)
                val_r2 = m.get('val', {}).get('r2')
                if val_r2 is None:
                    val_r2 = m.get('val_r2') or m.get('Val_R2', -999.0)
                val_r2 = float(val_r2)

                # Dataset dizini (3 seviye yukarı: model/config/split/metrics → dataset)
                ds_name = metrics_file.parent.parent.parent.name
                if ds_name not in dataset_best or val_r2 > dataset_best[ds_name]:
                    dataset_best[ds_name] = val_r2
            except Exception:
                continue

        deleted = []
        kept    = []
        freed_bytes = 0.0

        for ds_name, best_r2 in sorted(dataset_best.items()):
            ds_path = datasets_dir / ds_name
            if not ds_path.exists():
                continue

            if best_r2 < val_r2_threshold:
                # Boyut hesapla
                ds_size = sum(f.stat().st_size for f in ds_path.rglob('*') if f.is_file())
                freed_bytes += ds_size

                logger.info(f"  [DEL] {ds_name}  best_R2={best_r2:.4f}  "
                            f"size={ds_size/1024/1024:.1f} MB")

                if not dry_run:
                    try:
                        shutil.rmtree(ds_path)
                        logger.info(f"  [OK] Silindi: {ds_path}")
                    except Exception as e:
                        logger.error(f"  [FAIL] Silinemedi {ds_path}: {e}")

                deleted.append({'dataset': ds_name, 'best_r2': best_r2,
                                 'size_mb': round(ds_size / 1024 / 1024, 2)})
            else:
                kept.append({'dataset': ds_name, 'best_r2': best_r2})

        freed_mb = round(freed_bytes / 1024 / 1024, 2)
        logger.info(f"\n[CLEANUP] Özet: {len(deleted)} silindi / {len(kept)} tutuldu")
        logger.info(f"[CLEANUP] Kazanılan alan: {freed_mb} MB"
                    + (" (simülasyon)" if dry_run else " (gerçek)"))

        return {'deleted': deleted, 'kept': kept, 'total_freed_mb': freed_mb}

    # ========================================================================
    # SINGLE NUCLEUS PREDICTION
    # ========================================================================

    def run_single_prediction(self, nucleus_input=None):
        """
        Tek cekirdek veya kucuk liste tahmini.
        Egitilmis tum AI/ANFIS modeller (top 25 per target) kullanilir.
        Dataset uretilmez; sadece mevcut egitilmis modeller kullanilir.

        Args:
            nucleus_input : dict {'Z':int,'N':int,'SPIN':float,'PARITY':int}
                           veya dosya yolu str/Path (pred_input.txt/CSV/aaa2.txt)
                           veya None → interaktif giris alinir
        """
        logger.info("\n[SINGLE-PREDICT] TEK CEKIRDEK TAHMIN SISTEMI")
        logger.info("[SINGLE-PREDICT] Top-25 model per target, feature zenginlestirme aktif")

        try:
            from pfaz_modules.pfaz04_unknown_predictions.single_nucleus_predictor import (
                SingleNucleusPredictor
            )

            # Cikti dizini: pfaz_outputs[4] altinda timestamps'li alt klasor
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = self.pfaz_outputs[4] / f'single_predict_{ts}'

            predictor = SingleNucleusPredictor(
                ai_models_dir=str(self.pfaz_outputs[2]),
                anfis_models_dir=str(self.pfaz_outputs[3]),
                splits_dir=str(self.pfaz_outputs[1]),
                output_dir=str(out_dir),
                top_n_models=25,
            )

            # ---- interaktif giris ----
            if nucleus_input is None:
                if not sys.stdin.isatty() or os.environ.get('HPC_MODE'):
                    logger.warning("[SKIP] Non-interactive mode: nucleus_input required via --predict")
                    return
                print()
                print("[INPUT] Cekirdek bilgilerini girin:")
                print("  Ornek (tek cekirdek) : Z=26 N=30 SPIN=0.0 PARITY=1")
                print("  Ornek (dosya)        : pred_input.txt  veya  mylist.csv")
                print("  Cikis icin bos birak.")
                print()
                raw = input("Girdiniz: ").strip()
                if not raw:
                    logger.info("[SKIP] Giris bos, islem iptal.")
                    return
                nucleus_input = self._parse_nucleus_input(raw)
                if nucleus_input is None:
                    return

            # ---- tahmin ----
            if isinstance(nucleus_input, dict):
                results = predictor.predict_from_dict(nucleus_input)
            else:
                results = predictor.predict_from_file(str(nucleus_input))

            # ---- terminale ozet yaz ----
            if results and 'predictions' in results:
                print("\n" + "="*70)
                print("[RESULTS] TAHMIN SONUCLARI")
                print("="*70)
                for pred in results['predictions']:
                    nuc = pred.get('nucleus', '?')
                    print(f"\n  Cekirdek: {nuc}")
                    for tgt in predictor.targets:
                        tdata = pred.get(tgt, {})
                        consensus = tdata.get('consensus')
                        n_models  = tdata.get('n_models', 0)
                        if consensus is not None and not (
                            isinstance(consensus, float) and consensus != consensus
                        ):
                            print(f"    {tgt:<10s}: {consensus:+.4f}  ({n_models} model)")
                        else:
                            print(f"    {tgt:<10s}: N/A")
                print()
                print(f"  Detayli Excel : {results.get('excel_path', '-')}")
                print(f"  JSON          : {results.get('json_path',  '-')}")
                print(f"  Grafikler     : {len(results.get('plot_paths', []))} dosya -> {out_dir}/plots/")
                print("="*70)

            logger.info("[OK] Tek cekirdek tahmini tamamlandi.")
            return results

        except ImportError as e:
            logger.error(f"[ERROR] SingleNucleusPredictor yuklenemedi: {e}")
        except Exception as e:
            logger.error(f"[ERROR] Tek cekirdek tahmini basarisiz: {e}", exc_info=True)

    # ========================================================================
    # PFAZ 8 SUPPLEMENTAL — runs AFTER PFAZ 9/12/13
    # ========================================================================

    def run_pfaz_08_supplemental(self):
        """
        PFAZ 8 Ek Geçiş — PFAZ9/12/13 tamamlandıktan sonra çalışır.
        Monte Carlo, istatistiksel test ve AutoML gelişim grafiklerini üretir.
        """
        logger.info("\n[PFAZ 8-SUPPLEMENTAL] EK VİZÜALİZASYON GEÇEĞI (PFAZ9/12/13 grafikleri)")
        try:
            from pfaz_modules.pfaz08_visualization.supplemental_visualizer import SupplementalVisualizer

            pfaz9_dir  = str(self.pfaz_outputs[9])
            pfaz12_dir = str(self.pfaz_outputs[12])
            pfaz13_dir = str(self.pfaz_outputs[13])
            pfaz4_dir  = str(self.pfaz_outputs[4])
            out_dir    = str(self.pfaz_outputs[8] / 'supplemental')

            _aaa2_root = self.project_root / 'aaa2.txt'
            _aaa2_data = self.project_root / 'data' / 'aaa2.txt'
            _aaa2_path = str(_aaa2_root if _aaa2_root.exists() else _aaa2_data)

            sv = SupplementalVisualizer(output_dir=out_dir)
            result = sv.run(
                pfaz9_dir=pfaz9_dir,
                pfaz12_dir=pfaz12_dir,
                pfaz13_dir=pfaz13_dir,
                pfaz4_dir=pfaz4_dir,
                aaa2_path=_aaa2_path,
            )
            logger.info(f"[OK] PFAZ8-Supplemental tamamlandı: {result}")
            return result
        except Exception as e:
            logger.warning(f"[WARNING] PFAZ8-Supplemental başarısız (devam): {e}")
            return {'status': 'failed', 'error': str(e)}

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    def run_all_pfaz(self, start_from=1, end_at=13, modes=None):
        """
        Tüm PFAZ fazlarını çalıştır

        Args:
            start_from: Başlangıç fazı
            end_at: Bitiş fazı
            modes: Dict[int, str] - Her faz için mod ('run', 'resume', 'update', 'pass')

        Note:
            PFAZ 11 (Production Deployment) kullanıcı talebi doğrultusunda ertelenmiştir.
            Otomatik olarak atlanacaktır.
        """
        if modes is None:
            modes = {i: 'run' for i in range(start_from, end_at + 1)}

        # PFAZ11'i otomatik olarak skip moduna al
        if 11 in modes and modes[11] != 'pass':
            logger.info("[INFO] PFAZ11 (Production Deployment) kullanıcı talebi ile ertelenmiştir.")
            modes[11] = 'pass'

        import time as _time

        pfaz_list = [i for i in range(start_from, end_at + 1)]
        n_total   = len(pfaz_list)
        pipeline_start = _time.time()

        def _eta_str(elapsed: float, done: int, total: int) -> str:
            if done == 0:
                return '?'
            remaining = elapsed / done * (total - done)
            h, rem = divmod(int(remaining), 3600)
            m, s   = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        def _elapsed_str(seconds: float) -> str:
            h, rem = divmod(int(seconds), 3600)
            m, s   = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        logger.info("\n" + "="*80)
        logger.info("[START] TUM PFAZ FAZLARI BASLATILIYOR")
        logger.info("="*80)
        logger.info(f"Aralık  : PFAZ {start_from} -> PFAZ {end_at}  ({n_total} faz)")
        logger.info(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("[NOTE] PFAZ 11 otomatik olarak atlanacaktır (deferred)")

        results    = {}
        done_count = 0

        for pfaz_id in pfaz_list:
            mode = modes.get(pfaz_id, 'run')

            # Resume modunda: completed fazları otomatik atla
            if mode == 'resume':
                current_status = self.status_manager.get_pfaz_status(f'pfaz_{pfaz_id:02d}')
                if current_status['status'] == 'completed':
                    logger.info(f"[SKIP] PFAZ {pfaz_id} zaten tamamlanmis (completed), atlaniyor...")
                    done_count += 1
                    continue

            # ---- İlerleme başlığı ----
            elapsed  = _time.time() - pipeline_start
            eta      = _eta_str(elapsed, done_count, n_total)
            pct      = int(done_count / n_total * 100)
            bar_done = int(pct / 5)
            bar      = '█' * bar_done + '░' * (20 - bar_done)
            print(f"\n[{'='*78}]")
            print(f"  PFAZ {pfaz_id:>2}/{end_at}  [{bar}] {pct:>3}%  "
                  f"Geçen: {_elapsed_str(elapsed)}  Tahmini kalan: {eta}  "
                  f"Şu an: {datetime.now().strftime('%H:%M:%S')}")
            print(f"  Mod: {mode.upper()}")
            print(f"[{'='*78}]")

            pfaz_start = _time.time()
            try:
                method_name = f'run_pfaz_{pfaz_id:02d}'
                if hasattr(self, method_name):
                    method  = getattr(self, method_name)
                    results[pfaz_id] = method(mode=mode)
                else:
                    logger.warning(f"[WARNING] PFAZ {pfaz_id} metodu bulunamadı!")
                    self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)

                pfaz_elapsed = _time.time() - pfaz_start
                logger.info(f"[PFAZ {pfaz_id}] Tamamlandı -- süre: {_elapsed_str(pfaz_elapsed)}")

            except Exception as e:
                pfaz_elapsed = _time.time() - pfaz_start
                logger.error(f"[ERROR] PFAZ {pfaz_id} başarısız ({_elapsed_str(pfaz_elapsed)}): {e}")

                # Hata durumunda devam etmek istiyor musunuz?
                if not sys.stdin.isatty() or os.environ.get('HPC_MODE'):
                    logger.warning("[AUTO] Non-interactive mode: hatada devam ediliyor")
                else:
                    response = input("\n[WARNING] Devam etmek istiyor musunuz? (E/h): ").lower()
                    if response == 'h':
                        logger.info("[STOP] Kullanici tarafindan durduruldu")
                        break

            done_count += 1

        # PFAZ 8 Supplemental — PFAZ9/12/13 bittikten sonra ek grafik geçişi
        if end_at >= 13:
            logger.info("\n[PFAZ 8-SUPPLEMENTAL] PFAZ13 sonrası ek grafik geçişi başlatılıyor...")
            try:
                results['8_supplemental'] = self.run_pfaz_08_supplemental()
            except Exception as e:
                logger.warning(f"[WARNING] Supplemental visualization atlandı: {e}")

        # Toplam süre
        total_elapsed = _time.time() - pipeline_start
        logger.info(f"\n[DONE] Toplam sure: {_elapsed_str(total_elapsed)}")
        logger.info(f"[DONE] Bitis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # --- WarningTracker: rapor kaydet + özet yazdır ---
        try:
            from utils.warning_tracker import get_tracker
            _wt = get_tracker()
            _wt.save_report()
            _wt.print_summary()
        except Exception as _wt_e:
            logger.warning(f"[WARNING] WarningTracker raporu kaydedilemedi: {_wt_e}")

        # Özet
        self.status_manager.print_status()

        # ---- Tahmin sistemi sorusu ----
        if end_at >= 4:   # PFAZ4 tamamlandiysa modeller hazir
            if sys.stdin.isatty() and not os.environ.get('HPC_MODE'):
                self._ask_prediction_after_pipeline()
            else:
                logger.info("[AUTO] Non-interactive mode: skipping prediction prompt")

        return results

    def _ask_prediction_after_pipeline(self):
        """
        Tüm fazlar tamamlandiktan sonra kullaniciya tahmin sistemi sorusu sor.

        Secenekler:
          1 — Tek cekirdek veya kucuk liste: mevcut modeller ile aninda tahmin
          2 — Yeni liste: tum fazlari yeni cikti klasorunde yeniden calistir
          3 — Atlama
        """
        print("\n" + "="*70)
        print("[PREDICT] Tahmin Sistemi")
        print("="*70)
        print("  Tum fazlar tamamlandi. Tahmin yapmak ister misiniz?")
        print()
        print("  1. Tek cekirdek / kucuk liste tahmini")
        print("     (Mevcut egitilmis modeller kullanilir, dataset uretilmez)")
        print()
        print("  2. Yeni buyuk liste gir")
        print("     (PFAZ1-13 yeni bir cikti klasorunde yeniden calistirilir)")
        print()
        print("  3. Hayir, atlama")
        print("-"*70)

        choice = input("Seciminiz (1/2/3): ").strip()

        if choice == '1':
            print()
            print("[INPUT] Cekirdek giris formati:")
            print("  a) Tek cekirdek : Z=26 N=30 SPIN=0.0 PARITY=1")
            print("  b) Dosya yolu   : pred_input.txt (bkz. format aciklamasi)")
            print("  c) CSV dosyasi  : Z,N,SPIN,PARITY sutunlarini icermelidir")
            print()
            raw = input("Girisiniz (bos birak = iptal): ").strip()
            if not raw:
                logger.info("[SKIP] Tahmin iptal edildi.")
                return

            nucleus_input = self._parse_nucleus_input(raw)
            if nucleus_input is not None:
                self.run_single_prediction(nucleus_input=nucleus_input)

        elif choice == '2':
            print()
            raw = input("[INPUT] Yeni liste dosyasi yolu (pred_input.txt veya CSV): ").strip()
            if not raw:
                logger.info("[SKIP] Yeni liste tahmini iptal edildi.")
                return
            fp = Path(raw)
            if not fp.is_absolute():
                fp = self.project_root / fp
            if not fp.exists():
                logger.error(f"[ERROR] Dosya bulunamadi: {fp}")
                return
            self.run_all_pfaz_with_custom_data(str(fp))

        else:
            logger.info("[SKIP] Tahmin atlanili.")

    @staticmethod
    def _parse_nucleus_input(raw: str):
        """
        Kullanici girdisini parse et.
        Dondurulen deger: dict {'Z':..., 'N':...} veya str (dosya yolu) veya None.
        """
        candidate = Path(raw)
        if candidate.exists():
            return str(candidate)

        # key=value ciftleri
        try:
            pairs: Dict = {}
            for token in raw.split():
                k, v = token.split('=')
                pairs[k.upper()] = float(v)
            # int'e cevir gereken alanlar
            for key in ('Z', 'N', 'A', 'PARITY'):
                if key in pairs:
                    pairs[key] = int(pairs[key])
            return pairs
        except Exception:
            logger.error(f"[ERROR] Giris formati anlasılamadi: {raw}")
            return None

    def run_all_pfaz_with_custom_data(self, data_file: str):
        """
        Kullanicinin verdigi yeni liste dosyasi ile tum PFAZ fazlarini
        AYRI bir cikti klasorunde yeniden calistir.

        Bu yontem:
          1. Yeni bir cikti dizini olusturur:
             outputs/custom_run_YYYYMMDD_HHMMSS/
          2. Config'i gecici olarak gunceller (data_file + output_base_dir)
          3. Yeni bir NuclearPhysicsAIOrchestrator ornegi baslatir
          4. run_all_pfaz(1, 13) cagririr

        Args:
            data_file: Giris verisinin yolu (pred_input.txt, CSV, aaa2.txt)
        """
        import shutil

        data_path = Path(data_file)
        if not data_path.exists():
            logger.error(f"[ERROR] Veri dosyasi bulunamadi: {data_path}")
            return

        # Satir sayisini kontrol et
        try:
            if data_path.suffix.lower() == '.csv':
                row_count = sum(1 for _ in open(str(data_path), encoding='utf-8')) - 1
            elif data_path.suffix.lower() in ('.xlsx', '.xls'):
                import openpyxl
                wb = openpyxl.load_workbook(str(data_path), read_only=True)
                row_count = wb.active.max_row - 1
            else:
                row_count = sum(
                    1 for line in open(str(data_path), encoding='utf-8', errors='replace')
                    if line.strip() and not line.startswith('#')
                )
        except Exception:
            row_count = -1

        logger.info(f"[CUSTOM-RUN] Dosya: {data_path.name}, tahmin edilen satir: {row_count}")

        if row_count == 1:
            logger.info("[CUSTOM-RUN] Tek satir -- tam pipeline yerine tek cekirdek tahmini yapiliyor")
            self.run_single_prediction(nucleus_input=str(data_path))
            return

        # Yeni cikti klasoru
        ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        custom_out = self.project_root / 'outputs' / f'custom_run_{ts}'
        custom_out.mkdir(parents=True, exist_ok=True)

        logger.info(f"[CUSTOM-RUN] Yeni cikti klasoru: {custom_out}")

        # Gecici config olustur
        tmp_config_path = custom_out / 'config.json'
        base_cfg = dict(self.config)
        base_cfg['data_file']  = str(data_path)
        base_cfg['output_dir'] = str(custom_out)  # orchestrator uses this key to build all pfaz_outputs

        with open(tmp_config_path, 'w', encoding='utf-8') as f:
            json.dump(base_cfg, f, indent=2, ensure_ascii=False)

        logger.info(f"[CUSTOM-RUN] Gecici config kaydedildi: {tmp_config_path}")
        logger.info("[CUSTOM-RUN] Yeni orchestrator baslatiliyor...")

        try:
            custom_orch = NuclearPhysicsAIOrchestrator(config_path=str(tmp_config_path))
            custom_orch.run_all_pfaz(start_from=1, end_at=13)
            logger.info(f"[CUSTOM-RUN] Tamamlandi. Ciktilar: {custom_out}")
        except Exception as e:
            logger.error(f"[CUSTOM-RUN] Hata: {e}", exc_info=True)
    
    def interactive_mode(self):
        """İnteraktif mod - Kullanıcı ile etkileşimli"""
        print("\n" + "="*80)
        print("[INTERACTIVE] INTERAKTIF MOD")
        print("="*80)

        self.status_manager.print_status()

        while True:
            print("\n" + "-"*80)
            print("Seçenekler:")
            print("  1-13: PFAZ fazını çalıştır")
            print("  all: Tüm fazları çalıştır")
            print("  predict: Tek çekirdek tahmini (SingleNucleusPredictor)")
            print("  status: Durum özeti göster")
            print("  reset: Durumu sıfırla")
            print("  exit: Çıkış")
            print("-"*80)

            choice = input("\nSeçiminiz: ").strip().lower()

            if choice == 'exit':
                print("[EXIT] Çıkış yapılıyor...")
                break

            elif choice == 'predict':
                self.run_single_prediction()

            elif choice == 'status':
                self.status_manager.print_status()

            elif choice == 'reset':
                confirm = input("[WARNING] Tüm durum sıfırlanacak. Emin misiniz? (E/h): ")
                if confirm.lower() == 'e':
                    self.status_manager.status = self.status_manager._create_default_status()
                    self.status_manager.save_status()
                    print("[SUCCESS] Durum sıfırlandı!")

            elif choice == 'all':
                print("\nMod seçin:")
                print("  1. Run (Yeni başlat)")
                print("  2. Resume (Devam et)")
                print("  3. Update (Güncelle)")
                mode_choice = input("Seçim (1-3): ").strip()

                mode = {'1': 'run', '2': 'resume', '3': 'update'}.get(mode_choice, 'run')
                self.run_all_pfaz(modes={i: mode for i in range(1, 14)})

            elif choice.isdigit() and 1 <= int(choice) <= 13:
                pfaz_id = int(choice)

                print(f"\nPFAZ {pfaz_id} modu seçin:")
                print("  1. Run (Yeni başlat)")
                print("  2. Resume (Devam et)")
                print("  3. Update (Güncelle)")
                print("  4. Pass (Atla)")
                mode_choice = input("Seçim (1-4): ").strip()

                mode = {'1': 'run', '2': 'resume', '3': 'update', '4': 'pass'}.get(
                    mode_choice, 'run'
                )

                method_name = f'run_pfaz_{pfaz_id:02d}'
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    method(mode=mode)
                else:
                    print(f"[ERROR] PFAZ {pfaz_id} bulunamadı!")

            else:
                print("[ERROR] Geçersiz seçim!")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Komut satırı argümanlarını parse et"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Nuclear Physics AI Project Orchestrator'
    )
    
    parser.add_argument(
        '--pfaz', type=int, choices=range(1, 14),
        help='Tek bir PFAZ fazını çalıştır (1-13)'
    )
    
    parser.add_argument(
        '--run-all', action='store_true',
        help='Tüm PFAZ fazlarını çalıştır'
    )
    
    parser.add_argument(
        '--start-from', type=int, default=1,
        help='Başlangıç fazı (varsayılan: 1)'
    )
    
    parser.add_argument(
        '--end-at', type=int, default=13,
        help='Bitiş fazı (varsayılan: 13)'
    )
    
    parser.add_argument(
        '--mode', choices=['run', 'resume', 'update', 'pass'],
        default='run',
        help='Çalıştırma modu'
    )
    
    parser.add_argument(
        '--interactive', '-i', action='store_true',
        help='İnteraktif mod'
    )
    
    parser.add_argument(
        '--predict', type=str, default=None,
        metavar='FILE_OR_DICT',
        help=(
            'Tek çekirdek tahmini: dosya yolu (aaa2.txt) veya '
            'key=value çiftleri (örn. "Z=26 N=30 A=56")'
        )
    )

    parser.add_argument(
        '--check-deps', action='store_true',
        help='Sadece kütüphane kontrolü yap'
    )
    
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='Konfigürasyon dosyası yolu'
    )
    
    return parser.parse_args()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ana fonksiyon"""

    # Banner
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║   NUCLEAR PHYSICS AI PROJECT - MAIN ORCHESTRATOR                        ║
    ║                                                                          ║
    ║   13 PFAZ Pipeline Management System                                    ║
    ║   Version 6.0.0 - Production Ready                                      ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    # Parse arguments
    args = parse_arguments()

    # Kütüphane kontrolü
    if args.check_deps or not args.interactive:
        AutoInstaller.check_and_install()

    if args.check_deps:
        return

    # Orkestratör başlat
    orchestrator = NuclearPhysicsAIOrchestrator(config_path=args.config)

    try:
        # İnteraktif mod
        if args.interactive:
            orchestrator.interactive_mode()

        # Tek çekirdek tahmini
        elif args.predict:
            nucleus_input = NuclearPhysicsAIOrchestrator._parse_nucleus_input(
                args.predict.strip()
            )
            if nucleus_input is None:
                sys.exit(1)
            orchestrator.run_single_prediction(nucleus_input=nucleus_input)

        # Tek faz çalıştır
        elif args.pfaz:
            method_name = f'run_pfaz_{args.pfaz:02d}'
            if hasattr(orchestrator, method_name):
                method = getattr(orchestrator, method_name)
                method(mode=args.mode)
            else:
                logger.error(f"[ERROR] PFAZ {args.pfaz} bulunamadı!")

        # Tüm fazları çalıştır
        elif args.run_all:
            orchestrator.run_all_pfaz(
                start_from=args.start_from,
                end_at=args.end_at,
                modes={i: args.mode for i in range(args.start_from, args.end_at + 1)}
            )

        # Varsayılan: İnteraktif mod
        else:
            print("\n[TIP] Kullanım için --help kullanın")
            print("[TIP] Interaktif mod için: python main.py --interactive")
            orchestrator.interactive_mode()

    except KeyboardInterrupt:
        logger.info("\n[STOP] Kullanıcı tarafından durduruldu")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n[CRITICAL ERROR] Kritik hata: {e}", exc_info=True)
        sys.exit(1)

    finally:
        logger.info("\n" + "="*80)
        logger.info("[EXIT] Program sonlandı")
        logger.info("="*80)

if __name__ == "__main__":
    import multiprocessing as _mp
    try:
        _mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    main()
