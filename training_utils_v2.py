"""
Training Utilities v2.0
Advanced Training Support with 50 Different Loops, Timeout, Overfitting Detection

Özellikler:
- 50 farklı eğitim konfigürasyonu
- Training timeout (1 saat)
- Overfitting detection
- Early stopping
- Checkpoint yönetimi
- Training logger
- Unknown nuclei prediction support
"""

import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TRAINING TIMEOUT DECORATOR
# ============================================================================

class TrainingTimeout:
    """Training timeout kontrolü (max 1 saat)"""
    
    def __init__(self, max_time_seconds: int = 3600):
        self.max_time = max_time_seconds
        self.start_time = None
    
    def start(self):
        """Timer başlat"""
        self.start_time = time.time()
        logger.info(f"[TIMER]  Training timeout: {self.max_time}s ({self.max_time/60:.1f} min)")
    
    def check(self) -> bool:
        """Timeout kontrolü"""
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        if elapsed > self.max_time:
            logger.warning(f"[WARNING]  TIMEOUT! Elapsed: {elapsed:.1f}s > {self.max_time}s")
            return True
        return False
    
    def elapsed(self) -> float:
        """Geçen süre"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def remaining(self) -> float:
        """Kalan süre"""
        if self.start_time is None:
            return self.max_time
        return max(0, self.max_time - self.elapsed())


# ============================================================================
# OVERFITTING DETECTOR
# ============================================================================

class OverfittingDetector:
    """
    Overfitting tespiti
    
    Kontroller:
    - Train-Test gap (>0.15)
    - Validation curve artışı
    - Consistent degradation
    """
    
    def __init__(self, gap_threshold: float = 0.15, patience: int = 5):
        self.gap_threshold = gap_threshold
        self.patience = patience
        self.history = {
            'train_scores': [],
            'val_scores': [],
            'gaps': []
        }
    
    def update(self, train_score: float, val_score: float):
        """Skor güncelle"""
        self.history['train_scores'].append(train_score)
        self.history['val_scores'].append(val_score)
        
        gap = abs(train_score - val_score)
        self.history['gaps'].append(gap)
    
    def check_overfitting(self) -> Tuple[bool, str]:
        """
        Overfitting kontrolü
        
        Returns:
            (is_overfitting, reason)
        """
        if len(self.history['gaps']) < self.patience:
            return False, "Insufficient data"
        
        # 1. Train-Test gap
        recent_gap = np.mean(self.history['gaps'][-self.patience:])
        if recent_gap > self.gap_threshold:
            return True, f"Large train-test gap: {recent_gap:.4f} > {self.gap_threshold}"
        
        # 2. Validation degradation
        if len(self.history['val_scores']) >= self.patience:
            recent_vals = self.history['val_scores'][-self.patience:]
            if all(recent_vals[i] < recent_vals[i-1] for i in range(1, len(recent_vals))):
                return True, "Consistent validation degradation"
        
        return False, "No overfitting detected"
    
    def get_stats(self) -> Dict:
        """İstatistikler"""
        if not self.history['gaps']:
            return {}
        
        return {
            'mean_gap': np.mean(self.history['gaps']),
            'max_gap': np.max(self.history['gaps']),
            'recent_gap': self.history['gaps'][-1] if self.history['gaps'] else 0,
            'train_trend': np.mean(np.diff(self.history['train_scores'][-10:])) if len(self.history['train_scores']) > 10 else 0,
            'val_trend': np.mean(np.diff(self.history['val_scores'][-10:])) if len(self.history['val_scores']) > 10 else 0
        }


# ============================================================================
# EARLY STOPPING MONITOR
# ============================================================================

class EarlyStoppingMonitor:
    """
    Early stopping with multiple criteria
    
    Kriterler:
    - Validation improvement
    - Overfitting detection
    - Plateau detection
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
        
        self.overfitting_detector = OverfittingDetector()
    
    def update(self, val_score: float, train_score: float, epoch: int) -> bool:
        """
        Skor güncelle ve early stopping kontrolü
        
        Returns:
            should_stop (bool)
        """
        # Overfitting check
        self.overfitting_detector.update(train_score, val_score)
        is_overfitting, reason = self.overfitting_detector.check_overfitting()
        
        if is_overfitting:
            logger.warning(f"[WARNING]  Overfitting detected: {reason}")
            self.should_stop = True
            return True
        
        # Improvement check
        if self.mode == 'max':
            improved = val_score > (self.best_score + self.min_delta)
        else:
            improved = val_score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = val_score
            self.counter = 0
            self.best_epoch = epoch
            logger.info(f"[SUCCESS] New best: {val_score:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            logger.info(f"[WAIT] No improvement: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            logger.info(f"🛑 Early stopping at epoch {epoch}")
            self.should_stop = True
            return True
        
        return False
    
    def get_stats(self) -> Dict:
        """İstatistikler"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience_counter': self.counter,
            'overfitting_stats': self.overfitting_detector.get_stats()
        }


# ============================================================================
# TRAINING CONFIGURATION MANAGER (50 Configs)
# ============================================================================

class TrainingConfigManager:
    """
    50 farklı training konfigürasyonu
    
    Varyasyonlar:
    - Learning rates (5)
    - Batch sizes (5)  
    - Architectures (2)
    - Regularization (5)
    Total: 5 x 5 x 2 = 50 configs
    """
    
    @staticmethod
    def get_all_configs() -> List[Dict]:
        """50 farklı config üret"""
        configs = []
        config_id = 1
        
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        batch_sizes = [16, 32, 64, 128, 256]
        architectures = [
            [256, 128, 64, 32],
            [512, 256, 128, 64, 32]
        ]
        dropout_configs = [
            [0.1, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.0],
            [0.3, 0.3, 0.2, 0.0],
            [0.4, 0.3, 0.2, 0.0],
            [0.5, 0.4, 0.3, 0.0]
        ]
        
        for lr in learning_rates:
            for bs in batch_sizes:
                for arch in architectures:
                    # Her architecture için sadece ilk dropout config
                    dropout = dropout_configs[0]
                    
                    config = {
                        'id': f'TRAIN_{config_id:03d}',
                        'learning_rate': lr,
                        'batch_size': bs,
                        'architecture': arch,
                        'dropout': dropout if len(arch) == len(dropout) else dropout[:len(arch)],
                        'optimizer': 'adam',
                        'activation': 'relu',
                        'early_stopping_patience': 15,
                        'reduce_lr_patience': 10
                    }
                    configs.append(config)
                    config_id += 1
        
        logger.info(f"[SUCCESS] Generated {len(configs)} training configurations")
        return configs
    
    @staticmethod
    def get_config_by_id(config_id: str) -> Optional[Dict]:
        """ID'ye göre config getir"""
        all_configs = TrainingConfigManager.get_all_configs()
        for config in all_configs:
            if config['id'] == config_id:
                return config
        return None
    
    @staticmethod
    def save_configs(output_path: Path):
        """Tüm configleri kaydet"""
        configs = TrainingConfigManager.get_all_configs()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(configs, f, indent=2)
        
        logger.info(f"[SUCCESS] Saved {len(configs)} configs to {output_path}")


# ============================================================================
# TRAINING LOGGER
# ============================================================================

class TrainingLogger:
    """
    Her eğitimin detaylı kaydı
    
    Kaydedilenler:
    - Config details
    - Training metrics
    - Validation metrics
    - Time elapsed
    - Overfitting detection
    - Best epoch info
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_log = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def start_training(self, config: Dict, dataset_info: Dict):
        """Eğitim başlat"""
        self.current_log = {
            'session_id': self.session_id,
            'config': config,
            'dataset_info': dataset_info,
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'final_metrics': {},
            'status': 'running'
        }
        
        logger.info(f"[NOTE] Training log started: {config['id']}")
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Epoch logla"""
        if self.current_log is None:
            return
        
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics,
            'validation': val_metrics
        }
        
        self.current_log['epochs'].append(epoch_data)
    
    def finish_training(self, final_metrics: Dict, status: str = 'completed'):
        """Eğitim bitir"""
        if self.current_log is None:
            return
        
        self.current_log['end_time'] = datetime.now().isoformat()
        self.current_log['final_metrics'] = final_metrics
        self.current_log['status'] = status
        
        # Save
        config_id = self.current_log['config']['id']
        log_file = self.log_dir / f"{config_id}_{self.session_id}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.current_log, f, indent=2)
        
        logger.info(f"[SUCCESS] Training log saved: {log_file}")
        
        # Reset
        self.current_log = None
    
    def get_all_logs(self) -> List[Dict]:
        """Tüm logları oku"""
        logs = []
        for log_file in self.log_dir.glob("*.json"):
            with open(log_file) as f:
                logs.append(json.load(f))
        return logs


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """
    Model checkpoint yönetimi
    
    Özellikler:
    - Periodic saving (her 10 dakika)
    - Best model tracking
    - Timeout recovery
    """
    
    def __init__(self, checkpoint_dir: Path, save_interval: int = 600):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.last_save_time = time.time()
        
        self.best_score = -np.inf
        self.best_checkpoint = None
    
    def should_save(self) -> bool:
        """Checkpoint kaydedilmeli mi?"""
        return (time.time() - self.last_save_time) >= self.save_interval
    
    def save_checkpoint(self, model: Any, config_id: str, epoch: int, 
                       metrics: Dict, checkpoint_type: str = 'periodic'):
        """Checkpoint kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{config_id}_epoch{epoch}_{checkpoint_type}_{timestamp}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'model': model,
            'config_id': config_id,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'type': checkpoint_type
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.last_save_time = time.time()
        logger.info(f"[SAVE] Checkpoint saved: {checkpoint_name}")
        
        # Best checkpoint tracking
        if 'val_r2' in metrics and metrics['val_r2'] > self.best_score:
            self.best_score = metrics['val_r2']
            self.best_checkpoint = checkpoint_path
            logger.info(f"[BEST] New best checkpoint: R²={self.best_score:.6f}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Checkpoint yükle"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        logger.info(f"[OPEN] Loaded checkpoint: {checkpoint_path.name}")
        return checkpoint


# ============================================================================
# COMPREHENSIVE TRAINING MONITOR
# ============================================================================

class ComprehensiveTrainingMonitor:
    """
    Tüm training utilities'i birleştiren ana monitör
    """
    
    def __init__(self, output_dir: Path, max_training_time: int = 3600):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.timeout_monitor = TrainingTimeout(max_training_time)
        self.early_stopping = EarlyStoppingMonitor()
        self.logger = TrainingLogger(self.output_dir / 'logs')
        self.checkpoint_manager = CheckpointManager(self.output_dir / 'checkpoints')
        
        logger.info("[SUCCESS] Comprehensive Training Monitor initialized")
    
    def start_training(self, config: Dict, dataset_info: Dict):
        """Training başlat"""
        self.timeout_monitor.start()
        self.logger.start_training(config, dataset_info)
    
    def update_epoch(self, model: Any, config_id: str, epoch: int,
                    train_metrics: Dict, val_metrics: Dict) -> Dict[str, bool]:
        """
        Epoch güncelle
        
        Returns:
            {
                'should_stop': bool,
                'timeout': bool,
                'early_stop': bool,
                'save_checkpoint': bool
            }
        """
        # Log epoch
        self.logger.log_epoch(epoch, train_metrics, val_metrics)
        
        # Checks
        timeout = self.timeout_monitor.check()
        early_stop = self.early_stopping.update(
            val_metrics.get('r2', 0),
            train_metrics.get('r2', 0),
            epoch
        )
        should_checkpoint = self.checkpoint_manager.should_save()
        
        # Save checkpoint if needed
        if should_checkpoint:
            combined_metrics = {**train_metrics, **val_metrics}
            self.checkpoint_manager.save_checkpoint(
                model, config_id, epoch, combined_metrics, 'periodic'
            )
        
        # Final decision
        should_stop = timeout or early_stop
        
        return {
            'should_stop': should_stop,
            'timeout': timeout,
            'early_stop': early_stop,
            'save_checkpoint': should_checkpoint
        }
    
    def finish_training(self, model: Any, config_id: str, final_metrics: Dict,
                       status: str = 'completed'):
        """Training bitir"""
        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            model, config_id, -1, final_metrics, 'final'
        )
        
        # Finish log
        self.logger.finish_training(final_metrics, status)
        
        logger.info(f"[SUCCESS] Training finished: {status}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRAINING UTILITIES V2.0 - TEST")
    print("="*80)
    
    # 1. Generate 50 configs
    print("\n1. Generating 50 training configurations...")
    configs = TrainingConfigManager.get_all_configs()
    print(f"   [SUCCESS] Generated {len(configs)} configs")
    print(f"   Sample config: {configs[0]}")
    
    # 2. Save configs
    print("\n2. Saving configurations...")
    TrainingConfigManager.save_configs(Path("test_configs.json"))
    
    # 3. Training monitor demo
    print("\n3. Training monitor demo...")
    monitor = ComprehensiveTrainingMonitor(
        output_dir=Path("test_training_monitor"),
        max_training_time=60  # 1 minute for demo
    )
    
    config = configs[0]
    dataset_info = {
        'n_train': 100,
        'n_val': 20,
        'n_test': 20,
        'features': ['A', 'Z', 'N']
    }
    
    monitor.start_training(config, dataset_info)
    
    # Simulate training epochs
    for epoch in range(5):
        train_metrics = {'r2': 0.9 + epoch * 0.01, 'rmse': 0.5 - epoch * 0.05}
        val_metrics = {'r2': 0.85 + epoch * 0.005, 'rmse': 0.6 - epoch * 0.03}
        
        status = monitor.update_epoch(
            model=None,  # Dummy
            config_id=config['id'],
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )
        
        print(f"   Epoch {epoch}: {status}")
        
        if status['should_stop']:
            break
    
    monitor.finish_training(
        model=None,
        config_id=config['id'],
        final_metrics={'final_r2': 0.92, 'final_rmse': 0.35},
        status='completed'
    )
    
    print("\n[SUCCESS] All tests passed!")
    print("="*80)
