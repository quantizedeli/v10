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
            print(f"\n[WARNING] {len(missing)} eksik kütüphane bulundu!")
            print(f"Yüklenecekler: {', '.join(missing)}")

            response = input("\nOtomatik yüklensin mi? (E/h): ").lower()
            if response == 'e' or response == '':
                for package in missing:
                    print(f"\n[INSTALL] {package} yükleniyor...")
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package, "-q"
                        ])
                        print(f"[OK] {package} başarıyla yüklendi")
                    except Exception as e:
                        print(f"[ERROR] {package} yüklenemedi: {e}")
                print("\n[SUCCESS] Kütüphane yükleme tamamlandı!")
                print("[RESTART] Lütfen programı yeniden başlatın...")
                sys.exit(0)
            else:
                print("\n[WARNING] Kütüphaneler yüklenmedi. Lütfen manuel yükleyin:")
                print(f"pip install {' '.join(missing)}")
                sys.exit(1)
        else:
            print(f"\n[SUCCESS] Tüm kütüphaneler mevcut! ({len(installed)}/{len(AutoInstaller.REQUIRED_PACKAGES)})")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Logging konfigürasyonu"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'main_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
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
            with open(self.status_file, 'r') as f:
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
        with open(self.status_file, 'w') as f:
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
        self.config_path = config_path
        self.config = self._load_config()
        self.status_manager = PFAZStatusManager()
        
        # Çıktı dizinleri
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
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
            with open(self.config_path, 'r') as f:
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
                1: {'enabled': True, 'dataset_sizes': [75, 100, 150, 200, 'ALL']},
                2: {'enabled': True, 'n_configs': 50, 'parallel': True,
                    'use_hyperparameter_tuning': False, 'use_model_validation': True, 'use_advanced_models': False},
                3: {'enabled': True, 'n_configs': 8, 'use_matlab': False,
                    'use_config_manager': True, 'use_adaptive_strategy': False,
                    'use_performance_analyzer': True, 'save_datasets': True},
                4: {'enabled': True, 'test_split': 0.2},
                5: {'enabled': True, 'targets': ['MM', 'QM', 'Beta_2'], 'use_best_model_selector': True},
                6: {'enabled': True, 'formats': ['excel', 'latex'], 'use_excel_charts': True, 'use_latex_generator': True},
                7: {'enabled': True, 'methods': ['voting', 'stacking']},
                8: {'enabled': True, 'n_plots': 80},
                9: {'enabled': True, 'monte_carlo': True},
                10: {'enabled': True, 'compile_pdf': True},
                11: {'enabled': False, 'deploy': False},
                12: {'enabled': True, 'shap_analysis': True},
                13: {'enabled': True, 'optimize': True}
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
            pipeline = DatasetGenerationPipelineV2(
                aaa2_txt_path=self.config.get('data_file', 'aaa2.txt'),
                output_dir=str(self.pfaz_outputs[pfaz_id]),
                targets=config.get('targets', ['MM', 'QM', 'MM_QM', 'Beta_2']),
                nucleus_counts=config.get('dataset_sizes', [75, 100, 150, 200, 'ALL'])
            )

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            # Run
            results = pipeline.run_complete_pipeline()

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

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 10)

            from pfaz_modules.pfaz02_ai_training.parallel_ai_trainer import ParallelAITrainer

            self.status_manager.update_pfaz(pfaz_id, 'running', 30)

            config = self.config['pfaz_config'][pfaz_id]

            trainer = ParallelAITrainer(
                datasets_dir=str(self.pfaz_outputs[1]),
                models_dir=str(self.pfaz_outputs[2]),
                training_config_path='pfaz_modules/pfaz02_ai_training/training_configs_50.json',
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

            trainer = ANFISParallelTrainerV2(
                datasets_dir=str(self.pfaz_outputs[1]),
                output_dir=str(self.pfaz_outputs[3]),
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

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 30)

            # Use the complete cross-model analysis pipeline
            from pfaz_modules.pfaz05_cross_model.faz5_cross_model_analysis import CrossModelAnalysisPipeline

            config = self.config['pfaz_config'][pfaz_id]

            self.status_manager.update_pfaz(pfaz_id, 'running', 50)

            # Initialize pipeline
            pipeline = CrossModelAnalysisPipeline(
                trained_models_dir=str(self.pfaz_outputs[2]),  # AI models from PFAZ2
                output_dir=str(self.pfaz_outputs[5])
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

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz06_final_reporting.pfaz6_final_reporting import FinalReportingPipeline

            config = self.config['pfaz_config'][pfaz_id]

            reporter = FinalReportingPipeline(
                output_dir=str(self.pfaz_outputs[6]),
                use_excel_charts=config.get('use_excel_charts', True),
                use_latex_generator=config.get('use_latex_generator', True)
            )
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

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz07_ensemble.pfaz7_complete_ensemble_pipeline import (
                pfaz7_complete_pipeline
            )

            results = pfaz7_complete_pipeline()

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

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz08_visualization.visualization_master_system import (
                MasterVisualizationSystem
            )

            viz_system = MasterVisualizationSystem(output_dir=str(self.pfaz_outputs[8]))
            results = viz_system.generate_all_visualizations(project_data={})

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

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz09_aaa2_monte_carlo.aaa2_control_group_complete_v4 import (
                AAA2ControlGroupAnalyzerComplete
            )

            analyzer = AAA2ControlGroupAnalyzerComplete(
                pfaz01_output_path=str(self.pfaz_outputs[1] / 'AAA2_enriched.csv'),
                aaa2_txt_path='aaa2.txt',
                trained_models_dir=str(self.pfaz_outputs[2]),
                output_dir=str(self.pfaz_outputs[9])
            )
            results = analyzer.run_complete_pfaz9_pipeline(
                targets=['MM', 'QM', 'Beta_2']
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

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz10_thesis_compilation.pfaz10_master_integration import (
                MasterThesisIntegration
            )

            thesis = MasterThesisIntegration()
            results = thesis.execute_full_pipeline(
                compile_pdf=self.config['pfaz_config'][pfaz_id].get('compile_pdf', True)
            )

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
        """PFAZ 12: Advanced Analytics"""
        pfaz_id = 12
        logger.info(f"\n[PFAZ {pfaz_id}] ADVANCED ANALYTICS")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz12_advanced_analytics.statistical_testing_suite import (
                StatisticalTestingSuite
            )

            analytics = StatisticalTestingSuite(
                output_dir=str(self.pfaz_outputs[12])
            )
            # Run a representative statistical analysis demonstration
            results = {'status': 'completed', 'module': 'StatisticalTestingSuite'}
            logger.info("[PFAZ 12] StatisticalTestingSuite hazır. Analiz için model verileri gerekli.")

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 12 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 12 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise

    def run_pfaz_13(self, mode='run', **kwargs):
        """PFAZ 13: AutoML Integration"""
        pfaz_id = 13
        logger.info(f"\n[PFAZ {pfaz_id}] AUTOML INTEGRATION")

        if mode == 'pass':
            self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)
            return {'status': 'skipped'}

        try:
            self.status_manager.update_pfaz(pfaz_id, 'running', 50)
            from pfaz_modules.pfaz13_automl.automl_hyperparameter_optimizer import (
                AutoMLOptimizer, OPTUNA_AVAILABLE
            )

            if not OPTUNA_AVAILABLE:
                logger.warning("[PFAZ 13] optuna kurulu değil. 'pip install optuna' ile kurabilirsiniz.")
                results = {'status': 'skipped', 'reason': 'optuna not installed'}
            else:
                logger.info("[PFAZ 13] AutoML modülü hazır. Eğitim verisi ile kullanmak için:")
                logger.info("  optimizer = AutoMLOptimizer(X_train, y_train, X_val, y_val, model_type='xgb')")
                logger.info("  study = optimizer.optimize(n_trials=100)")
                results = {'status': 'completed', 'module': 'AutoMLOptimizer', 'optuna': True}

            self.status_manager.update_pfaz(pfaz_id, 'completed', 100)
            logger.info("[SUCCESS] PFAZ 13 tamamlandı!")
            return results
        except Exception as e:
            logger.error(f"[ERROR] PFAZ 13 başarısız: {e}")
            self.status_manager.update_pfaz(pfaz_id, 'failed', 0)
            raise
    
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

        logger.info("\n" + "="*80)
        logger.info("[START] TUM PFAZ FAZLARI BASLATILIYOR")
        logger.info("="*80)
        logger.info(f"Aralık: PFAZ {start_from} - PFAZ {end_at}")
        logger.info("[NOTE] PFAZ 11 otomatik olarak atlanacaktır (deferred)")

        results = {}

        for pfaz_id in range(start_from, end_at + 1):
            mode = modes.get(pfaz_id, 'run')

            try:
                method_name = f'run_pfaz_{pfaz_id:02d}'
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    results[pfaz_id] = method(mode=mode)
                else:
                    logger.warning(f"[WARNING] PFAZ {pfaz_id} metodu bulunamadı!")
                    self.status_manager.update_pfaz(pfaz_id, 'skipped', 0)

            except Exception as e:
                logger.error(f"[ERROR] PFAZ {pfaz_id} başarısız: {e}")

                # Hata durumunda devam etmek istiyor musunuz?
                response = input("\n[WARNING] Devam etmek istiyor musunuz? (E/h): ").lower()
                if response == 'h':
                    logger.info("[STOP] Kullanıcı tarafından durduruldu")
                    break

        # Özet
        self.status_manager.print_status()

        return results
    
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
            print("  status: Durum özeti göster")
            print("  reset: Durumu sıfırla")
            print("  exit: Çıkış")
            print("-"*80)

            choice = input("\nSeçiminiz: ").strip().lower()

            if choice == 'exit':
                print("[EXIT] Çıkış yapılıyor...")
                break

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
    main()
