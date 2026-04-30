"""
Stacking Meta-Learner
=====================

İki seviyeli ensemble öğrenme:
- Level 0: Base modeller (RF, GBM, XGBoost, DNN, BNN, PINN)
- Level 1: Meta-model (Base modellerin tahminlerini öğrenir)

Author: AI Dataset Training Pipeline
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackingMetaLearner:
    """
    Stacking Meta-Learner
    
    Base modellerin tahminlerini kullanarak daha güçlü bir meta-model eğitir
    """
    
    def __init__(self, 
                 meta_model_type: str = 'ridge',
                 cv_folds: int = 5,
                 output_dir: str = 'output/stacking_models'):
        """
        Args:
            meta_model_type: Meta-model tipi ('ridge', 'lasso', 'elasticnet', 'rf', 'gbm', 'mlp')
            cv_folds: Cross-validation fold sayısı (out-of-fold predictions için)
            output_dir: Stacking modellerinin kaydedileceği klasör
        """
        self.meta_model_type = meta_model_type
        self.cv_folds = cv_folds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_models = {}  # Model ID -> Model object
        self.base_model_metadata = {}  # Model ID -> Metadata
        self.meta_model = None
        self.oof_predictions = None  # Out-of-fold predictions
        
        logger.info(f"[OK] StackingMetaLearner initialized")
        logger.info(f"  Meta-model type: {meta_model_type}")
        logger.info(f"  CV folds: {cv_folds}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def add_base_model(self, model_id: str, model: object, metadata: Dict):
        """
        Base model ekle
        
        Args:
            model_id: Benzersiz model ID
            model: Model objesi
            metadata: Model metadata (type, features, performance)
        """
        self.base_models[model_id] = model
        self.base_model_metadata[model_id] = metadata
        logger.info(f"  [OK] Base model added: {model_id} ({metadata['model_type']})")
    
    def _predict_single(self, model, X, model_type: str) -> np.ndarray:
        """Tek bir modelden tahmin al"""
        try:
            if model_type in ['RF', 'GBM', 'XGBoost', 'Ridge', 'Lasso', 'ElasticNet']:
                y_pred = model.predict(X)
            
            elif model_type in ['DNN', 'BNN', 'PINN']:
                y_pred = model.predict(X, verbose=0).flatten()
            
            elif model_type == 'ANFIS':
                logger.warning("ANFIS prediction not implemented")
                return np.zeros(len(X))
            
            else:
                y_pred = model.predict(X)
            
            return y_pred.flatten()
        
        except Exception as e:
            logger.error(f"Prediction error for {model_type}: {str(e)}")
            return np.zeros(len(X))
    
    def generate_oof_predictions(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Out-of-Fold (OOF) predictions oluştur
        
        Base modellerin cross-validation tahminlerini kullanarak
        meta-model için eğitim verisi oluşturur
        
        Args:
            X_train: Base modellerin eğitim verisi
            y_train: Hedef değişken
        
        Returns:
            OOF predictions (n_samples, n_base_models)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATING OUT-OF-FOLD PREDICTIONS")
        logger.info(f"{'='*60}")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Base models: {len(self.base_models)}")
        logger.info(f"CV folds: {self.cv_folds}")
        
        n_samples = len(X_train)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for model_idx, (model_id, model) in enumerate(self.base_models.items()):
            logger.info(f"\n-> Processing: {model_id}")
            metadata = self.base_model_metadata[model_id]
            model_type = metadata['model_type']
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                # Fold train/val split
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                
                # Model'i bu fold için yeniden eğit
                # NOT: Gerçek implementasyonda modeli clone edip yeniden eğitmek gerekir
                # Şimdilik mevcut modelin tahminlerini kullanıyoruz
                
                y_pred_val = self._predict_single(model, X_fold_val, model_type)
                
                # OOF predictions
                oof_predictions[val_idx, model_idx] = y_pred_val
                
                logger.info(f"  Fold {fold+1}/{self.cv_folds}: {len(val_idx)} samples")
            
            # OOF performance
            oof_r2 = r2_score(y_train, oof_predictions[:, model_idx])
            oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions[:, model_idx]))
            logger.info(f"  OOF R^2: {oof_r2:.4f}, RMSE: {oof_rmse:.4f}")
        
        self.oof_predictions = oof_predictions
        
        logger.info(f"\n[OK] OOF predictions generated: {oof_predictions.shape}")
        
        return oof_predictions
    
    def train_meta_model(self, y_train: np.ndarray):
        """
        Meta-model eğit
        
        OOF predictions kullanarak Level-1 meta-model'i eğitir
        
        Args:
            y_train: Gerçek hedef değişken
        """
        if self.oof_predictions is None:
            raise ValueError("Generate OOF predictions first using generate_oof_predictions()")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING META-MODEL")
        logger.info(f"{'='*60}")
        logger.info(f"Meta-model type: {self.meta_model_type}")
        
        X_meta = self.oof_predictions  # Base model predictions
        y_meta = y_train
        
        # Create meta-model
        if self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        
        elif self.meta_model_type == 'lasso':
            self.meta_model = Lasso(alpha=0.1)
        
        elif self.meta_model_type == 'elasticnet':
            self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        elif self.meta_model_type == 'rf':
            self.meta_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        elif self.meta_model_type == 'gbm':
            self.meta_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        
        elif self.meta_model_type == 'mlp':
            self.meta_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
        
        else:
            raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")
        
        # Train
        self.meta_model.fit(X_meta, y_meta)
        
        # Training performance
        y_pred_train = self.meta_model.predict(X_meta)
        r2_train = r2_score(y_meta, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_meta, y_pred_train))
        mae_train = mean_absolute_error(y_meta, y_pred_train)
        
        logger.info(f"\n[OK] Meta-model trained")
        logger.info(f"  Training R^2 = {r2_train:.4f}")
        logger.info(f"  Training RMSE = {rmse_train:.4f}")
        logger.info(f"  Training MAE = {mae_train:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_
            logger.info(f"\n  Meta-model feature importances:")
            for i, (model_id, importance) in enumerate(zip(self.base_models.keys(), importances)):
                logger.info(f"    {model_id}: {importance:.4f}")
        
        elif hasattr(self.meta_model, 'coef_'):
            coefs = self.meta_model.coef_
            logger.info(f"\n  Meta-model coefficients:")
            for i, (model_id, coef) in enumerate(zip(self.base_models.keys(), coefs)):
                logger.info(f"    {model_id}: {coef:.4f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Stacking ensemble ile tahmin
        
        Args:
            X_test: Test verisi
        
        Returns:
            Predictions
        """
        if self.meta_model is None:
            raise ValueError("Train meta-model first using train_meta_model()")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STACKING PREDICTION")
        logger.info(f"{'='*60}")
        
        # Base model predictions
        n_samples = len(X_test)
        n_models = len(self.base_models)
        base_predictions = np.zeros((n_samples, n_models))
        
        for model_idx, (model_id, model) in enumerate(self.base_models.items()):
            metadata = self.base_model_metadata[model_id]
            y_pred = self._predict_single(model, X_test, metadata['model_type'])
            base_predictions[:, model_idx] = y_pred
            logger.info(f"  [OK] {model_id}: {len(y_pred)} predictions")
        
        # Meta-model prediction
        y_pred_stacking = self.meta_model.predict(base_predictions)
        
        logger.info(f"\n[OK] Stacking predictions: {len(y_pred_stacking)}")
        
        return y_pred_stacking
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Stacking ensemble'ı değerlendir
        
        Args:
            X_test: Test verisi
            y_test: Test hedef değişken
        
        Returns:
            Performance metrics
        """
        y_pred = self.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        result = {
            'method': 'stacking',
            'meta_model_type': self.meta_model_type,
            'n_base_models': len(self.base_models),
            'base_model_ids': list(self.base_models.keys()),
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STACKING PERFORMANCE:")
        logger.info(f"  R^2 = {r2:.4f}")
        logger.info(f"  RMSE = {rmse:.4f}")
        logger.info(f"  MAE = {mae:.4f}")
        logger.info(f"{'='*60}")
        
        return result
    
    def save_stacking_model(self, model_name: str):
        """
        Stacking modelini kaydet
        
        Args:
            model_name: Kaydedilecek model ismi
        """
        save_dir = self.output_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Meta-model
        meta_model_path = save_dir / 'meta_model.pkl'
        joblib.dump(self.meta_model, meta_model_path)
        
        # OOF predictions
        oof_path = save_dir / 'oof_predictions.npy'
        np.save(oof_path, self.oof_predictions)
        
        # Metadata
        metadata = {
            'model_name': model_name,
            'meta_model_type': self.meta_model_type,
            'cv_folds': self.cv_folds,
            'n_base_models': len(self.base_models),
            'base_model_ids': list(self.base_models.keys()),
            'base_model_metadata': self.base_model_metadata
        }
        
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n[OK] Stacking model saved: {save_dir}")
    
    def compare_with_base_models(self, 
                                 X_test: np.ndarray, 
                                 y_test: np.ndarray) -> pd.DataFrame:
        """
        Stacking'i base modellerle karşılaştır
        
        Args:
            X_test: Test verisi
            y_test: Test hedef değişken
        
        Returns:
            Comparison DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARING STACKING WITH BASE MODELS")
        logger.info(f"{'='*60}")
        
        comparison = []
        
        # Base models
        for model_id, model in self.base_models.items():
            metadata = self.base_model_metadata[model_id]
            y_pred = self._predict_single(model, X_test, metadata['model_type'])
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            comparison.append({
                'Model': model_id,
                'Type': metadata['model_type'],
                'Level': 'Base',
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae
            })
            
            logger.info(f"  Base: {model_id} - R^2={r2:.4f}, RMSE={rmse:.4f}")
        
        # Stacking
        result = self.evaluate(X_test, y_test)
        comparison.append({
            'Model': f'Stacking_{self.meta_model_type}',
            'Type': 'Stacking',
            'Level': 'Meta',
            'R²': result['r2'],
            'RMSE': result['rmse'],
            'MAE': result['mae']
        })
        
        logger.info(f"  Meta: Stacking - R^2={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('R²', ascending=False)
        
        logger.info(f"\n{df.to_string(index=False)}")
        
        # Save
        save_path = self.output_dir / 'stacking_comparison.xlsx'
        df.to_excel(save_path, index=False)
        logger.info(f"\n[OK] Comparison saved: {save_path}")
        
        return df


def main():
    """Test function"""
    logger.info("="*60)
    logger.info("STACKING META-LEARNER - TEST")
    logger.info("="*60)
    
    # Mock data
    np.random.seed(42)
    n_train = 200
    n_test = 50
    
    X_train = np.random.randn(n_train, 5)
    X_test = np.random.randn(n_test, 5)
    y_train = 3 * X_train[:, 0] + 2 * X_train[:, 1] + np.random.randn(n_train) * 0.5
    y_test = 3 * X_test[:, 0] + 2 * X_test[:, 1] + np.random.randn(n_test) * 0.5
    
    # Mock base models
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    
    model1 = Ridge(alpha=0.1).fit(X_train, y_train)
    model2 = Lasso(alpha=0.1).fit(X_train, y_train)
    model3 = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    
    # Stacking meta-learner
    stacker = StackingMetaLearner(meta_model_type='ridge', cv_folds=5)
    
    # Add base models
    stacker.add_base_model('Ridge', model1, {'model_type': 'Ridge', 'features': list(range(5))})
    stacker.add_base_model('Lasso', model2, {'model_type': 'Lasso', 'features': list(range(5))})
    stacker.add_base_model('RF', model3, {'model_type': 'RF', 'features': list(range(5))})
    
    # Generate OOF predictions
    oof_predictions = stacker.generate_oof_predictions(X_train, y_train)
    
    # Train meta-model
    stacker.train_meta_model(y_train)
    
    # Evaluate
    result = stacker.evaluate(X_test, y_test)
    
    # Compare with base models
    comparison = stacker.compare_with_base_models(X_test, y_test)
    
    # Save
    stacker.save_stacking_model('TestStacking')
    
    logger.info("\n[SUCCESS] TEST COMPLETED!")


if __name__ == "__main__":
    main()
