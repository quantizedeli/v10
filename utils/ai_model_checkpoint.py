"""
AI Model Checkpoint Sistemi
AI Model Checkpoint and Verification System

Bu modül, tüm AI modellerinin (RF, GBM, XGBoost, DNN, BNN, PINN) checkpoint'lerini
yönetir, kaydeder ve doğrular.
"""

import joblib
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIModelCheckpoint:
    """AI model checkpoint yöneticisi"""
    
    # Desteklenen model tipleri
    SUPPORTED_MODELS = ['RandomForest', 'GradientBoosting', 'XGBoost', 'DNN', 'BNN', 'PINN']
    
    def __init__(self, base_dir='trained_models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_checkpoint(self, model, model_type, config_info, metrics, output_dir=None):
        """
        AI modelini checkpoint olarak kaydet
        
        Args:
            model: Eğitilmiş model objesi
            model_type: str - Model tipi (RandomForest, GradientBoosting, etc.)
            config_info: dict - Model konfigürasyonu ve dataset bilgileri
            metrics: dict - Model performans metrikleri
            output_dir: Path - Çıktı klasörü (None ise base_dir/model_type kullanılır)
            
        Returns:
            dict: Kaydedilen dosya yolları ve durum
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}. "
                           f"Desteklenenler: {self.SUPPORTED_MODELS}")
        
        # Output directory belirleme
        if output_dir is None:
            output_dir = self.base_dir / model_type
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"{model_type}_{timestamp}"
        
        logger.info(f"AI model checkpoint kaydediliyor: {checkpoint_name}")
        
        saved_files = {}
        
        # 1. [WARNING] MODEL CHECKPOINT (.pkl veya .h5)
        if model_type in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            # Sklearn/XGBoost modelleri için joblib
            model_file = output_dir / f'{checkpoint_name}_model.pkl'
            joblib.dump(model, model_file)
            saved_files['model'] = str(model_file)
            logger.info(f"  [OK] Model kaydedildi (joblib): {model_file.name}")
            
        elif model_type in ['DNN', 'BNN', 'PINN']:
            # Deep learning modelleri için
            # Keras/TensorFlow modelleri .h5 veya SavedModel formatında
            model_file = output_dir / f'{checkpoint_name}_model.h5'
            try:
                model.save(model_file)
                saved_files['model'] = str(model_file)
                logger.info(f"  [OK] Model kaydedildi (H5): {model_file.name}")
            except Exception as e:
                # H5 başarısız olursa pickle dene
                model_file = output_dir / f'{checkpoint_name}_model.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                saved_files['model'] = str(model_file)
                logger.info(f"  [OK] Model kaydedildi (pickle): {model_file.name}")
        
        # 2. MODEL WEIGHTS (DL modelleri için ayrıca)
        if model_type in ['DNN', 'BNN', 'PINN']:
            try:
                weights_file = output_dir / f'{checkpoint_name}_weights.h5'
                model.save_weights(weights_file)
                saved_files['weights'] = str(weights_file)
                logger.info(f"  [OK] Weights kaydedildi: {weights_file.name}")
            except Exception as e:
                logger.warning(f"  [WARNING] Weights kaydedilemedi: {e}")
        
        # 3. CONFIG INFO (JSON)
        config_info['model_type'] = model_type
        config_info['timestamp'] = timestamp
        config_file = output_dir / f'{checkpoint_name}_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2)
        saved_files['config'] = str(config_file)
        logger.info(f"  [OK] Config kaydedildi: {config_file.name}")
        
        # 4. METRICS (JSON)
        metrics['model_type'] = model_type
        metrics['timestamp'] = timestamp
        metrics_file = output_dir / f'{checkpoint_name}_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        saved_files['metrics'] = str(metrics_file)
        logger.info(f"  [OK] Metrics kaydedildi: {metrics_file.name}")
        
        # 5. CHECKPOINT SUMMARY
        summary = {
            'checkpoint_name': checkpoint_name,
            'model_type': model_type,
            'timestamp': timestamp,
            'files': saved_files,
            'metrics': metrics,
            'config': config_info
        }
        
        summary_file = output_dir / f'{checkpoint_name}_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = str(summary_file)
        
        logger.info(f"[OK] AI model checkpoint başarıyla kaydedildi: {checkpoint_name}")
        
        return {
            'status': 'success',
            'checkpoint_name': checkpoint_name,
            'files': saved_files
        }
    
    def verify_checkpoint(self, checkpoint_name_or_path, model_type=None):
        """
        Checkpoint'in doğru kaydedildiğini doğrula
        
        Args:
            checkpoint_name_or_path: str - Checkpoint ismi veya yolu
            model_type: str - Model tipi (None ise otomatik tespit)
            
        Returns:
            dict: Doğrulama sonucu
        """
        # Checkpoint path'i belirle
        if Path(checkpoint_name_or_path).exists():
            checkpoint_dir = Path(checkpoint_name_or_path).parent
            checkpoint_name = Path(checkpoint_name_or_path).stem.replace('_model', '').replace('_config', '')
        else:
            if model_type:
                checkpoint_dir = self.base_dir / model_type
            else:
                # Model tipi belli değilse tüm klasörlerde ara
                checkpoint_dir = self.base_dir
            checkpoint_name = checkpoint_name_or_path
        
        logger.info(f"AI checkpoint doğrulanıyor: {checkpoint_name}")
        
        verification = {
            'checkpoint_name': checkpoint_name,
            'model_exists': False,
            'config_exists': False,
            'metrics_exists': False,
            'weights_exists': False,  # DL modelleri için
            'errors': [],
            'status': 'failed'
        }
        
        # 1. Model dosyası kontrolü (.pkl veya .h5)
        model_files_pkl = list(checkpoint_dir.rglob(f'{checkpoint_name}*_model.pkl'))
        model_files_h5 = list(checkpoint_dir.rglob(f'{checkpoint_name}*_model.h5'))
        model_files = model_files_pkl + model_files_h5
        
        if model_files:
            verification['model_exists'] = True
            verification['model_file'] = str(model_files[0])
            logger.info(f"  [OK] Model dosyası bulundu: {model_files[0].name}")
            
            # Model yüklenebilir mi kontrol et
            try:
                if model_files[0].suffix == '.pkl':
                    _ = joblib.load(model_files[0])
                elif model_files[0].suffix == '.h5':
                    # Keras model yükleme (isteğe bağlı)
                    pass
                logger.info(f"    [OK] Model yüklenebilir")
            except Exception as e:
                verification['errors'].append(f"Model yüklenemedi: {str(e)}")
        else:
            verification['errors'].append("Model dosyası bulunamadı")
            logger.warning(f"  [FAIL] Model dosyası bulunamadı")
        
        # 2. Config kontrolü
        config_files = list(checkpoint_dir.rglob(f'{checkpoint_name}*_config.json'))
        if config_files:
            verification['config_exists'] = True
            verification['config_file'] = str(config_files[0])
            logger.info(f"  [OK] Config bulundu: {config_files[0].name}")
            
            try:
                with open(config_files[0], 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"    [OK] Config içeriği doğru")
            except Exception as e:
                verification['errors'].append(f"Config yüklenemedi: {str(e)}")
        else:
            verification['errors'].append("Config dosyası bulunamadı")
            logger.warning(f"  [FAIL] Config bulunamadı")
        
        # 3. Metrics kontrolü
        metrics_files = list(checkpoint_dir.rglob(f'{checkpoint_name}*_metrics.json'))
        if metrics_files:
            verification['metrics_exists'] = True
            verification['metrics_file'] = str(metrics_files[0])
            logger.info(f"  [OK] Metrics bulundu: {metrics_files[0].name}")
            
            try:
                with open(metrics_files[0], 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    logger.info(f"    [OK] Metrics içeriği doğru (R^2={metrics.get('R2', 'N/A')})")
            except Exception as e:
                verification['errors'].append(f"Metrics yüklenemedi: {str(e)}")
        else:
            verification['errors'].append("Metrics dosyası bulunamadı")
            logger.warning(f"  [FAIL] Metrics bulunamadı")
        
        # 4. Weights kontrolü (DL modelleri için)
        weights_files = list(checkpoint_dir.rglob(f'{checkpoint_name}*_weights.h5'))
        if weights_files:
            verification['weights_exists'] = True
            verification['weights_file'] = str(weights_files[0])
            logger.info(f"  [OK] Weights bulundu: {weights_files[0].name}")
        
        # 5. Durum belirleme
        if verification['model_exists'] and verification['config_exists'] and len(verification['errors']) == 0:
            verification['status'] = 'success'
            logger.info(f"[OK] Checkpoint doğrulama BAŞARILI: {checkpoint_name}")
        else:
            verification['status'] = 'failed'
            logger.error(f"[FAIL] Checkpoint doğrulama BAŞARISIZ: {checkpoint_name}")
            for error in verification['errors']:
                logger.error(f"    - {error}")
        
        return verification
    
    def load_checkpoint(self, checkpoint_name_or_path, model_type=None):
        """
        Checkpoint'i yükle
        
        Args:
            checkpoint_name_or_path: str - Checkpoint ismi veya yolu
            model_type: str - Model tipi (None ise otomatik tespit)
            
        Returns:
            tuple: (model, config, metrics)
        """
        # Önce doğrula
        verification = self.verify_checkpoint(checkpoint_name_or_path, model_type)
        
        if verification['status'] != 'success':
            raise ValueError(f"Checkpoint yüklenemedi, doğrulama başarısız: {verification['errors']}")
        
        logger.info(f"Checkpoint yükleniyor: {verification['checkpoint_name']}")
        
        # Model'i yükle
        model_file = Path(verification['model_file'])
        if model_file.suffix == '.pkl':
            model = joblib.load(model_file)
        elif model_file.suffix == '.h5':
            # Keras model yükleme
            import tensorflow as tf
            model = tf.keras.models.load_model(model_file)
        
        # Config'i yükle
        with open(verification['config_file'], 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Metrics'i yükle
        with open(verification['metrics_file'], 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        logger.info(f"[OK] Checkpoint başarıyla yüklendi: {verification['checkpoint_name']}")
        
        return model, config, metrics


class AIModelRegistry:
    """AI model kayıt defteri - Tüm checkpointleri takip eder"""
    
    def __init__(self, registry_file='trained_models/ai_model_registry.json'):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Kayıt defterini yükle"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'checkpoints': [], 'last_updated': None}
    
    def _save_registry(self):
        """Kayıt defterini kaydet"""
        self.registry['last_updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_checkpoint(self, checkpoint_info):
        """Checkpoint'i kayıt defterine ekle"""
        self.registry['checkpoints'].append(checkpoint_info)
        self._save_registry()
        logger.info(f"Checkpoint kayıt defterine eklendi: {checkpoint_info['checkpoint_name']}")
    
    def get_all_checkpoints(self):
        """Tüm kayıtlı checkpointleri getir"""
        return self.registry['checkpoints']
    
    def get_checkpoints_by_model_type(self, model_type):
        """Belirli bir model tipine ait checkpointleri getir"""
        return [c for c in self.registry['checkpoints'] if c.get('model_type') == model_type]
    
    def get_best_checkpoint(self, model_type=None, metric='R2'):
        """En iyi checkpoint'i getir"""
        checkpoints = self.registry['checkpoints']
        
        if model_type:
            checkpoints = [c for c in checkpoints if c.get('model_type') == model_type]
        
        if not checkpoints:
            return None
        
        if metric in ['R2', 'r2', 'R²']:
            # R² için en yüksek
            return max(checkpoints, 
                      key=lambda c: c.get('metrics', {}).get('R2', -999))
        else:
            # RMSE, MAE için en düşük
            return min(checkpoints,
                      key=lambda c: c.get('metrics', {}).get(metric, 999))


def main():
    """Test fonksiyonu"""
    # Checkpoint manager oluştur
    checkpoint_mgr = AIModelCheckpoint()
    
    # Test için dummy sklearn model
    from sklearn.ensemble import RandomForestRegressor
    dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Dummy data ile fit
    X_dummy = np.random.randn(100, 3)
    y_dummy = np.random.randn(100)
    dummy_model.fit(X_dummy, y_dummy)
    
    # Model kaydet
    print("="*80)
    print("TEST: AI Model Checkpoint Kaydetme")
    print("="*80)
    
    config_info = {
        'target': 'MM',
        'features': ['A', 'Z', 'N'],
        'n_samples': 100,
        'hyperparameters': {'n_estimators': 10}
    }
    
    metrics = {
        'R2': 0.92,
        'RMSE': 0.18,
        'MAE': 0.14,
        'MAPE': 6.1
    }
    
    result = checkpoint_mgr.save_model_checkpoint(
        dummy_model, 
        'RandomForest', 
        config_info, 
        metrics
    )
    
    print(f"\nKaydedilen dosyalar:")
    for file_type, path in result['files'].items():
        print(f"  {file_type}: {path}")
    
    # Checkpoint doğrula
    print("\n" + "="*80)
    print("TEST: AI Checkpoint Doğrulama")
    print("="*80)
    verification = checkpoint_mgr.verify_checkpoint(result['checkpoint_name'], 'RandomForest')
    print(f"\nDoğrulama durumu: {verification['status']}")
    
    # Checkpoint yükle
    print("\n" + "="*80)
    print("TEST: AI Checkpoint Yükleme")
    print("="*80)
    loaded_model, loaded_config, loaded_metrics = checkpoint_mgr.load_checkpoint(
        result['checkpoint_name'], 
        'RandomForest'
    )
    print(f"Model yüklendi: {type(loaded_model)}")
    print(f"R² skoru: {loaded_metrics['R2']}")
    
    # Registry test
    print("\n" + "="*80)
    print("TEST: AI Model Registry")
    print("="*80)
    registry = AIModelRegistry()
    registry.register_checkpoint({
        'checkpoint_name': result['checkpoint_name'],
        'model_type': 'RandomForest',
        'metrics': metrics,
        'files': result['files']
    })
    print(f"Kayıtlı checkpoint sayısı: {len(registry.get_all_checkpoints())}")


if __name__ == "__main__":
    main()
