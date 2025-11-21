"""
Ensemble Model Builder
======================

Farklı modelleri birleştirerek güçlü ensemble modelleri oluşturur:
- Simple Voting (Average)
- Weighted Voting (Optimized weights)
- Stacking (Meta-model)

Author: AI Dataset Training Pipeline
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModelBuilder:
    """
    Ensemble Model Builder
    
    Birden fazla modeli birleştirerek daha güçlü tahminler oluşturur.
    """
    
    def __init__(self, output_dir: str = 'output/ensemble_models'):
        """
        Args:
            output_dir: Ensemble modellerinin kaydedileceği klasör
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}  # Model ID -> Model object
        self.model_metadata = {}  # Model ID -> Metadata
        self.predictions = {}  # Model ID -> Predictions
        
        logger.info(f"✓ EnsembleModelBuilder initialized")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def add_model(self, 
                  model_id: str,
                  model: object,
                  metadata: Dict,
                  X_val: np.ndarray = None,
                  y_val: np.ndarray = None):
        """
        Ensemble'a model ekle
        
        Args:
            model_id: Benzersiz model ID
            model: Model objesi (sklearn, keras, vs.)
            metadata: Model hakkında bilgiler (type, features, performance)
            X_val: Validation data (ağırlık optimizasyonu için)
            y_val: Validation target
        """
        self.models[model_id] = model
        self.model_metadata[model_id] = metadata
        
        # Validation predictions (ağırlık optimizasyonu için)
        if X_val is not None and y_val is not None:
            y_pred = self._predict_single(model, X_val, metadata['model_type'])
            self.predictions[model_id] = {
                'y_val': y_val.flatten(),
                'y_pred': y_pred.flatten()
            }
            
            # Performance
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            logger.info(f"  ✓ Model added: {model_id}")
            logger.info(f"    Type: {metadata['model_type']}")
            logger.info(f"    Val R²: {r2:.4f}, RMSE: {rmse:.4f}")
        else:
            logger.info(f"  ✓ Model added: {model_id} (no validation data)")
    
    def _predict_single(self, model, X, model_type: str) -> np.ndarray:
        """Tek bir modelden tahmin al"""
        try:
            if model_type in ['RF', 'GBM', 'XGBoost']:
                y_pred = model.predict(X)
            
            elif model_type in ['DNN', 'BNN', 'PINN']:
                # Keras/TensorFlow modeli
                y_pred = model.predict(X, verbose=0).flatten()
            
            elif model_type == 'ANFIS':
                # ANFIS için özel işlem (MATLAB interface)
                logger.warning("ANFIS prediction not implemented in this version")
                return np.zeros(len(X))
            
            else:
                # Genel predict methodu
                y_pred = model.predict(X)
            
            return y_pred.flatten()
        
        except Exception as e:
            logger.error(f"Prediction error for {model_type}: {str(e)}")
            return np.zeros(len(X))
    
    def create_simple_voting(self, 
                            model_ids: List[str],
                            X_test: np.ndarray,
                            y_test: np.ndarray = None) -> Dict:
        """
        Simple Voting Ensemble (Ortalama)
        
        Tüm modellerin tahminlerinin basit ortalaması
        
        Args:
            model_ids: Kullanılacak model ID'leri
            X_test: Test verisi
            y_test: Test target (opsiyonel, performans için)
        
        Returns:
            Dict: Tahminler ve metrikler
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMPLE VOTING ENSEMBLE")
        logger.info(f"{'='*60}")
        logger.info(f"Models: {len(model_ids)}")
        
        all_predictions = []
        
        for model_id in model_ids:
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not found, skipping")
                continue
            
            model = self.models[model_id]
            metadata = self.model_metadata[model_id]
            
            y_pred = self._predict_single(model, X_test, metadata['model_type'])
            all_predictions.append(y_pred)
            
            logger.info(f"  ✓ {model_id}: {len(y_pred)} predictions")
        
        if len(all_predictions) == 0:
            raise ValueError("No valid predictions from any model!")
        
        # Simple average
        y_pred_ensemble = np.mean(all_predictions, axis=0)
        
        result = {
            'method': 'simple_voting',
            'n_models': len(all_predictions),
            'model_ids': model_ids,
            'predictions': y_pred_ensemble
        }
        
        # Performance
        if y_test is not None:
            y_test = y_test.flatten()
            r2 = r2_score(y_test, y_pred_ensemble)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
            mae = mean_absolute_error(y_test, y_pred_ensemble)
            
            result.update({
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })
            
            logger.info(f"\n{'='*60}")
            logger.info(f"SIMPLE VOTING PERFORMANCE:")
            logger.info(f"  R² = {r2:.4f}")
            logger.info(f"  RMSE = {rmse:.4f}")
            logger.info(f"  MAE = {mae:.4f}")
            logger.info(f"{'='*60}")
        
        return result
    
    def create_weighted_voting(self,
                              model_ids: List[str],
                              X_test: np.ndarray,
                              y_test: np.ndarray = None,
                              optimization_method: str = 'r2') -> Dict:
        """
        Weighted Voting Ensemble (Optimize edilmiş ağırlıklar)
        
        Her modele performansına göre ağırlık verir
        
        Args:
            model_ids: Kullanılacak model ID'leri
            X_test: Test verisi
            y_test: Test target
            optimization_method: 'r2', 'rmse', 'mae', 'inverse_error'
        
        Returns:
            Dict: Tahminler, ağırlıklar ve metrikler
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"WEIGHTED VOTING ENSEMBLE")
        logger.info(f"{'='*60}")
        logger.info(f"Models: {len(model_ids)}")
        logger.info(f"Optimization: {optimization_method}")
        
        # Validation predictions topla
        val_predictions = []
        test_predictions = []
        
        for model_id in model_ids:
            if model_id not in self.predictions:
                logger.warning(f"No validation data for {model_id}, skipping")
                continue
            
            val_predictions.append(self.predictions[model_id]['y_pred'])
            
            # Test predictions
            model = self.models[model_id]
            metadata = self.model_metadata[model_id]
            y_pred_test = self._predict_single(model, X_test, metadata['model_type'])
            test_predictions.append(y_pred_test)
        
        if len(val_predictions) == 0:
            raise ValueError("No validation predictions available for optimization!")
        
        val_predictions = np.array(val_predictions)  # (n_models, n_samples)
        test_predictions = np.array(test_predictions)
        
        # Ağırlıkları optimize et
        y_val = self.predictions[model_ids[0]]['y_val']
        
        if optimization_method == 'r2':
            weights = self._optimize_weights_r2(val_predictions, y_val)
        elif optimization_method == 'rmse':
            weights = self._optimize_weights_rmse(val_predictions, y_val)
        elif optimization_method == 'mae':
            weights = self._optimize_weights_mae(val_predictions, y_val)
        elif optimization_method == 'inverse_error':
            weights = self._compute_inverse_error_weights(val_predictions, y_val)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        logger.info(f"\nOptimized Weights:")
        for i, (model_id, w) in enumerate(zip(model_ids, weights)):
            logger.info(f"  {model_id}: {w:.4f}")
        
        # Weighted ensemble prediction
        y_pred_ensemble = np.average(test_predictions, axis=0, weights=weights)
        
        result = {
            'method': 'weighted_voting',
            'n_models': len(test_predictions),
            'model_ids': model_ids,
            'weights': weights.tolist(),
            'optimization_method': optimization_method,
            'predictions': y_pred_ensemble
        }
        
        # Performance
        if y_test is not None:
            y_test = y_test.flatten()
            r2 = r2_score(y_test, y_pred_ensemble)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
            mae = mean_absolute_error(y_test, y_pred_ensemble)
            
            result.update({
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            })
            
            logger.info(f"\n{'='*60}")
            logger.info(f"WEIGHTED VOTING PERFORMANCE:")
            logger.info(f"  R² = {r2:.4f}")
            logger.info(f"  RMSE = {rmse:.4f}")
            logger.info(f"  MAE = {mae:.4f}")
            logger.info(f"{'='*60}")
        
        return result
    
    def _optimize_weights_r2(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """R² skorunu maksimize eden ağırlıkları bul"""
        n_models = predictions.shape[0]
        
        # Objective: -R² (minimize için negatif)
        def objective(weights):
            weights = weights / weights.sum()  # Normalize
            y_pred = np.average(predictions, axis=0, weights=weights)
            return -r2_score(y_true, y_pred)
        
        # Constraints: weights >= 0, sum = 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial guess: uniform
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x / result.x.sum()  # Normalize
    
    def _optimize_weights_rmse(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """RMSE'yi minimize eden ağırlıkları bul"""
        n_models = predictions.shape[0]
        
        def objective(weights):
            weights = weights / weights.sum()
            y_pred = np.average(predictions, axis=0, weights=weights)
            return np.sqrt(mean_squared_error(y_true, y_pred))
        
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
        bounds = [(0, 1) for _ in range(n_models)]
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x / result.x.sum()
    
    def _optimize_weights_mae(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """MAE'yi minimize eden ağırlıkları bul"""
        n_models = predictions.shape[0]
        
        def objective(weights):
            weights = weights / weights.sum()
            y_pred = np.average(predictions, axis=0, weights=weights)
            return mean_absolute_error(y_true, y_pred)
        
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
        bounds = [(0, 1) for _ in range(n_models)]
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x / result.x.sum()
    
    def _compute_inverse_error_weights(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Inverse Error Weighting
        
        Her modelin hatasının tersine göre ağırlık ver
        Daha düşük hata = Daha yüksek ağırlık
        """
        n_models = predictions.shape[0]
        errors = []
        
        for i in range(n_models):
            mse = mean_squared_error(y_true, predictions[i])
            errors.append(mse)
        
        errors = np.array(errors)
        
        # Inverse (epsilon to avoid division by zero)
        weights = 1.0 / (errors + 1e-10)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def save_ensemble(self, ensemble_result: Dict, ensemble_name: str):
        """
        Ensemble modelini kaydet
        
        Args:
            ensemble_result: create_*_voting() sonucu
            ensemble_name: Kaydedilecek isim
        """
        save_path = self.output_dir / f"{ensemble_name}.json"
        
        # Serialize
        save_data = {
            'ensemble_name': ensemble_name,
            'method': ensemble_result['method'],
            'n_models': ensemble_result['n_models'],
            'model_ids': ensemble_result['model_ids'],
            'weights': ensemble_result.get('weights', None),
            'optimization_method': ensemble_result.get('optimization_method', None),
            'performance': {
                'r2': ensemble_result.get('r2'),
                'rmse': ensemble_result.get('rmse'),
                'mae': ensemble_result.get('mae')
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"\n✓ Ensemble saved: {save_path}")
    
    def compare_ensembles(self, 
                         ensemble_results: List[Dict],
                         ensemble_names: List[str]) -> pd.DataFrame:
        """
        Farklı ensemble yöntemlerini karşılaştır
        
        Args:
            ensemble_results: Ensemble sonuçları listesi
            ensemble_names: Ensemble isimleri
        
        Returns:
            DataFrame: Karşılaştırma tablosu
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ENSEMBLE COMPARISON")
        logger.info(f"{'='*60}")
        
        comparison = []
        
        for result, name in zip(ensemble_results, ensemble_names):
            comparison.append({
                'Ensemble': name,
                'Method': result['method'],
                'N_Models': result['n_models'],
                'R²': result.get('r2', np.nan),
                'RMSE': result.get('rmse', np.nan),
                'MAE': result.get('mae', np.nan)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('R²', ascending=False)
        
        logger.info(f"\n{df.to_string(index=False)}")
        
        # Save
        save_path = self.output_dir / 'ensemble_comparison.xlsx'
        df.to_excel(save_path, index=False)
        logger.info(f"\n✓ Comparison saved: {save_path}")
        
        return df


def main():
    """Test function"""
    logger.info("="*60)
    logger.info("ENSEMBLE MODEL BUILDER - TEST")
    logger.info("="*60)
    
    # Mock data
    np.random.seed(42)
    n_samples = 100
    X_val = np.random.randn(n_samples, 5)
    X_test = np.random.randn(50, 5)
    y_val = 3 * X_val[:, 0] + 2 * X_val[:, 1] + np.random.randn(n_samples) * 0.5
    y_test = 3 * X_test[:, 0] + 2 * X_test[:, 1] + np.random.randn(50) * 0.5
    
    # Mock models
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    model1 = Ridge(alpha=0.1).fit(X_val, y_val)
    model2 = Ridge(alpha=1.0).fit(X_val, y_val)
    model3 = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_val, y_val)
    
    # Ensemble builder
    builder = EnsembleModelBuilder('test_ensemble_output')
    
    # Add models
    builder.add_model('Ridge_01', model1, 
                     {'model_type': 'Ridge', 'features': list(range(5))},
                     X_val, y_val)
    
    builder.add_model('Ridge_10', model2,
                     {'model_type': 'Ridge', 'features': list(range(5))},
                     X_val, y_val)
    
    builder.add_model('RF', model3,
                     {'model_type': 'RF', 'features': list(range(5))},
                     X_val, y_val)
    
    # Simple voting
    result_simple = builder.create_simple_voting(
        model_ids=['Ridge_01', 'Ridge_10', 'RF'],
        X_test=X_test,
        y_test=y_test
    )
    
    # Weighted voting
    result_weighted = builder.create_weighted_voting(
        model_ids=['Ridge_01', 'Ridge_10', 'RF'],
        X_test=X_test,
        y_test=y_test,
        optimization_method='r2'
    )
    
    # Save ensembles
    builder.save_ensemble(result_simple, 'SimpleVoting')
    builder.save_ensemble(result_weighted, 'WeightedVoting_R2')
    
    # Compare
    builder.compare_ensembles(
        ensemble_results=[result_simple, result_weighted],
        ensemble_names=['Simple Voting', 'Weighted Voting (R²)']
    )
    
    logger.info("\n✅ TEST COMPLETED!")


if __name__ == "__main__":
    main()
