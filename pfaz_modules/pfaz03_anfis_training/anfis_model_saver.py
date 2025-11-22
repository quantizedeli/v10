"""
ANFIS Model Kayıt ve Kontrol Sistemi
ANFIS Model Save and Verification System

Bu modül, ANFIS modellerinin doğru şekilde kaydedilmesini ve kontrol edilmesini sağlar.
Workspace ve FIS dosyaları ayrı ayrı kaydedilir ve doğrulanır.
"""

import scipy.io as sio
from pathlib import Path
import logging
import json
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ANFISModelSaver:
    """ANFIS model kaydetme ve kontrol sistemi"""
    
    def __init__(self, base_dir='trained_models/ANFIS'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_anfis_model(self, model_data, config_name, dataset_info, output_dir=None):
        """
        ANFIS modelini kaydet (workspace + FIS ayrı)
        
        Args:
            model_data: dict - Model verileri
                - 'fis': FIS object/data
                - 'training_error': list
                - 'validation_error': list
                - 'metrics': dict (R2, RMSE, MAE, etc.)
                - 'outliers': list (optional)
            config_name: str - ANFIS konfigürasyon ismi (GAU2MF, GEN2MF, etc.)
            dataset_info: dict - Dataset bilgileri
            output_dir: Path - Çıktı klasörü (None ise base_dir kullanılır)
            
        Returns:
            dict: Kaydedilen dosya yolları ve durum
        """
        if output_dir is None:
            output_dir = self.base_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{config_name}_{timestamp}"
        
        logger.info(f"ANFIS modeli kaydediliyor: {model_name}")
        
        saved_files = {}
        
        # 1. [WARNING] WORKSPACE KAYDI (.mat) - Tüm veriler
        workspace_data = {
            'fis_data': model_data.get('fis'),
            'training_error': np.array(model_data.get('training_error', [])),
            'validation_error': np.array(model_data.get('validation_error', [])),
            'metrics': model_data.get('metrics', {}),
            'config_name': config_name,
            'dataset_info': dataset_info,
            'timestamp': timestamp
        }
        
        # Outliers varsa ekle
        if 'outliers' in model_data and model_data['outliers'] is not None:
            workspace_data['outliers'] = np.array(model_data['outliers'])
        
        workspace_file = output_dir / f'{model_name}_workspace.mat'
        sio.savemat(workspace_file, workspace_data)
        saved_files['workspace'] = str(workspace_file)
        logger.info(f"  ✓ Workspace kaydedildi: {workspace_file.name}")
        
        # 2. [WARNING] FIS KAYDI (.fis veya .mat) - Sadece FIS yapısı
        fis_file = output_dir / f'{model_name}_fis.mat'
        sio.savemat(fis_file, {'fis': model_data.get('fis')})
        saved_files['fis'] = str(fis_file)
        logger.info(f"  ✓ FIS kaydedildi: {fis_file.name}")
        
        # 3. Metrikleri JSON olarak kaydet
        metrics = model_data.get('metrics', {})
        metrics['config_name'] = config_name
        metrics['dataset_info'] = dataset_info
        metrics['timestamp'] = timestamp
        
        metrics_file = output_dir / f'{model_name}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        saved_files['metrics'] = str(metrics_file)
        logger.info(f"  ✓ Metrikler kaydedildi: {metrics_file.name}")
        
        # 4. Hata grafikleri verileri
        if 'training_error' in model_data and 'validation_error' in model_data:
            errors_file = output_dir / f'{model_name}_errors.npz'
            np.savez(errors_file,
                    training_error=model_data['training_error'],
                    validation_error=model_data['validation_error'])
            saved_files['errors'] = str(errors_file)
            logger.info(f"  ✓ Hata verileri kaydedildi: {errors_file.name}")
        
        # 5. Outliers (varsa)
        if 'outliers' in model_data and model_data['outliers'] is not None:
            outliers_file = output_dir / f'{model_name}_outliers.csv'
            np.savetxt(outliers_file, model_data['outliers'], delimiter=',', fmt='%d')
            saved_files['outliers'] = str(outliers_file)
            logger.info(f"  ✓ Outliers kaydedildi: {outliers_file.name}")
        
        # 6. Model bilgilerini özet dosyasına kaydet
        summary = {
            'model_name': model_name,
            'config_name': config_name,
            'timestamp': timestamp,
            'files': saved_files,
            'metrics': metrics,
            'dataset_info': dataset_info
        }
        
        summary_file = output_dir / f'{model_name}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = str(summary_file)
        
        logger.info(f"✓ ANFIS modeli başarıyla kaydedildi: {model_name}")
        
        return {
            'status': 'success',
            'model_name': model_name,
            'files': saved_files
        }
    
    def verify_anfis_model(self, model_name_or_path):
        """
        ANFIS modelinin kaydedildiğini doğrula
        
        Args:
            model_name_or_path: str - Model ismi veya yolu
            
        Returns:
            dict: Doğrulama sonucu
        """
        # Model path'i belirle
        if Path(model_name_or_path).exists():
            model_dir = Path(model_name_or_path).parent
            model_name = Path(model_name_or_path).stem.replace('_workspace', '').replace('_fis', '')
        else:
            model_dir = self.base_dir
            model_name = model_name_or_path
        
        logger.info(f"ANFIS modeli doğrulanıyor: {model_name}")
        
        verification = {
            'model_name': model_name,
            'workspace_exists': False,
            'fis_exists': False,
            'metrics_exists': False,
            'errors': [],
            'status': 'failed'
        }
        
        # 1. Workspace kontrolü
        workspace_files = list(model_dir.glob(f'{model_name}*_workspace.mat'))
        if workspace_files:
            verification['workspace_exists'] = True
            verification['workspace_file'] = str(workspace_files[0])
            logger.info(f"  ✓ Workspace bulundu: {workspace_files[0].name}")
            
            # Workspace içeriğini kontrol et
            try:
                workspace_data = sio.loadmat(workspace_files[0])
                required_keys = ['fis_data', 'metrics', 'config_name']
                missing_keys = [k for k in required_keys if k not in workspace_data]
                
                if missing_keys:
                    verification['errors'].append(f"Workspace'te eksik key'ler: {missing_keys}")
                else:
                    logger.info(f"    ✓ Workspace içeriği doğru")
            except Exception as e:
                verification['errors'].append(f"Workspace yükleme hatası: {str(e)}")
        else:
            verification['errors'].append("Workspace dosyası bulunamadı")
            logger.warning(f"  ✗ Workspace bulunamadı")
        
        # 2. FIS kontrolü
        fis_files = list(model_dir.glob(f'{model_name}*_fis.mat'))
        if fis_files:
            verification['fis_exists'] = True
            verification['fis_file'] = str(fis_files[0])
            logger.info(f"  ✓ FIS bulundu: {fis_files[0].name}")
            
            # FIS içeriğini kontrol et
            try:
                fis_data = sio.loadmat(fis_files[0])
                if 'fis' not in fis_data:
                    verification['errors'].append("FIS dosyasında 'fis' key'i yok")
                else:
                    logger.info(f"    ✓ FIS içeriği doğru")
            except Exception as e:
                verification['errors'].append(f"FIS yükleme hatası: {str(e)}")
        else:
            verification['errors'].append("FIS dosyası bulunamadı")
            logger.warning(f"  ✗ FIS bulunamadı")
        
        # 3. Metrics kontrolü
        metrics_files = list(model_dir.glob(f'{model_name}*_metrics.json'))
        if metrics_files:
            verification['metrics_exists'] = True
            verification['metrics_file'] = str(metrics_files[0])
            logger.info(f"  ✓ Metrics bulundu: {metrics_files[0].name}")
            
            try:
                with open(metrics_files[0], 'r') as f:
                    metrics = json.load(f)
                    logger.info(f"    ✓ Metrics içeriği doğru (R²={metrics.get('R2', 'N/A')})")
            except Exception as e:
                verification['errors'].append(f"Metrics yükleme hatası: {str(e)}")
        else:
            verification['errors'].append("Metrics dosyası bulunamadı")
            logger.warning(f"  ✗ Metrics bulunamadı")
        
        # 4. Durum belirleme
        if verification['workspace_exists'] and verification['fis_exists'] and len(verification['errors']) == 0:
            verification['status'] = 'success'
            logger.info(f"✓ Model doğrulama BAŞARILI: {model_name}")
        else:
            verification['status'] = 'failed'
            logger.error(f"✗ Model doğrulama BAŞARISIZ: {model_name}")
            for error in verification['errors']:
                logger.error(f"    - {error}")
        
        return verification
    
    def load_anfis_model(self, model_name_or_path):
        """
        ANFIS modelini yükle
        
        Args:
            model_name_or_path: str - Model ismi veya yolu
            
        Returns:
            dict: Yüklenen model verisi
        """
        # Önce doğrula
        verification = self.verify_anfis_model(model_name_or_path)
        
        if verification['status'] != 'success':
            raise ValueError(f"Model yüklenemedi, doğrulama başarısız: {verification['errors']}")
        
        logger.info(f"ANFIS modeli yükleniyor: {verification['model_name']}")
        
        # Workspace'i yükle
        workspace_data = sio.loadmat(verification['workspace_file'])
        
        # FIS'i yükle
        fis_data = sio.loadmat(verification['fis_file'])
        
        # Metrics'i yükle
        with open(verification['metrics_file'], 'r') as f:
            metrics = json.load(f)
        
        model_data = {
            'fis': fis_data['fis'],
            'workspace': workspace_data,
            'metrics': metrics,
            'verification': verification
        }
        
        logger.info(f"✓ Model başarıyla yüklendi: {verification['model_name']}")
        
        return model_data


class ANFISModelRegistry:
    """ANFIS model kayıt defteri - Tüm modelleri takip eder"""
    
    def __init__(self, registry_file='trained_models/ANFIS/model_registry.json'):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Kayıt defterini yükle"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': [], 'last_updated': None}
    
    def _save_registry(self):
        """Kayıt defterini kaydet"""
        self.registry['last_updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_info):
        """Modeli kayıt defterine ekle"""
        self.registry['models'].append(model_info)
        self._save_registry()
        logger.info(f"Model kayıt defterine eklendi: {model_info['model_name']}")
    
    def get_all_models(self):
        """Tüm kayıtlı modelleri getir"""
        return self.registry['models']
    
    def get_models_by_config(self, config_name):
        """Belirli bir config'e ait modelleri getir"""
        return [m for m in self.registry['models'] if m.get('config_name') == config_name]
    
    def get_best_model(self, metric='R2'):
        """En iyi modeli getir (R2, RMSE, vs.)"""
        if not self.registry['models']:
            return None
        
        if metric in ['R2', 'r2', 'R²']:
            # R² için en yüksek
            return max(self.registry['models'], 
                      key=lambda m: m.get('metrics', {}).get('R2', -999))
        else:
            # RMSE, MAE için en düşük
            return min(self.registry['models'],
                      key=lambda m: m.get('metrics', {}).get(metric, 999))


def main():
    """Test fonksiyonu"""
    # Model saver oluştur
    saver = ANFISModelSaver()
    
    # Test model verisi
    test_model = {
        'fis': {'dummy': 'fis_data'},
        'training_error': [0.5, 0.3, 0.2, 0.15, 0.1],
        'validation_error': [0.6, 0.4, 0.25, 0.2, 0.15],
        'metrics': {
            'R2': 0.95,
            'RMSE': 0.15,
            'MAE': 0.12,
            'MAPE': 5.2
        },
        'outliers': [10, 25, 47]
    }
    
    dataset_info = {
        'target': 'MM',
        'features': ['A', 'Z', 'N'],
        'n_samples': 500
    }
    
    # Modeli kaydet
    print("="*80)
    print("TEST: ANFIS Model Kaydetme")
    print("="*80)
    result = saver.save_anfis_model(test_model, 'GAU2MF', dataset_info)
    print(f"\nKaydedilen dosyalar:")
    for file_type, path in result['files'].items():
        print(f"  {file_type}: {path}")
    
    # Modeli doğrula
    print("\n" + "="*80)
    print("TEST: ANFIS Model Doğrulama")
    print("="*80)
    verification = saver.verify_anfis_model(result['model_name'])
    print(f"\nDoğrulama durumu: {verification['status']}")
    
    # Model registry test
    print("\n" + "="*80)
    print("TEST: ANFIS Model Registry")
    print("="*80)
    registry = ANFISModelRegistry()
    registry.register_model({
        'model_name': result['model_name'],
        'config_name': 'GAU2MF',
        'metrics': test_model['metrics'],
        'files': result['files']
    })
    print(f"Kayıtlı model sayısı: {len(registry.get_all_models())}")


if __name__ == "__main__":
    main()
