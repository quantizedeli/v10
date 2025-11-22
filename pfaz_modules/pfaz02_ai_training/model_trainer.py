"""
Model Trainer - COMPLETE VERSION
Tüm ML Model Trainerları

İçerik:
- Random Forest
- Gradient Boosting
- XGBoost
- Deep Neural Network (DNN)
- Parallel training support
- Adaptive learning integration
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available")

# Local imports
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core_modules.constants import *
from adaptive_learning.adaptive_strategy import AdaptiveLearningStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# BASE TRAINER
# ============================================================================

class BaseModelTrainer:
    """Base class for all model trainers"""
    
    def __init__(self, model_name, output_dir='trained_models'):
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.target_names = None
        
        logger.info(f"{model_name} Trainer initialized")
    
    def load_dataset(self, dataset_path):
        """Load dataset from path"""
        dataset_path = Path(dataset_path)
        
        datasets = {}
        for split in ['train', 'check', 'test']:
            X = np.load(dataset_path / f'X_{split}.npy')
            y = np.load(dataset_path / f'y_{split}.npy')
            
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            datasets[split] = {'X': X, 'y': y}
        
        # Load metadata
        metadata_file = dataset_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names')
                self.target_names = metadata.get('target_names')
        
        return datasets
    
    def evaluate(self, X, y, y_pred, split_name):
        """Evaluate predictions"""
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        metrics = {
            'R2': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae)
        }
        
        logger.info(f"  {split_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        return metrics
    
    def save_model(self, save_path):
        """Save model"""
        raise NotImplementedError


# ============================================================================
# RANDOM FOREST TRAINER
# ============================================================================

class RandomForestTrainer(BaseModelTrainer):
    """Random Forest Regressor"""
    
    def __init__(self, output_dir='trained_models', n_estimators=400, max_depth=None):
        super().__init__('RandomForest', output_dir)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def train(self, datasets):
        """Train Random Forest"""
        logger.info(f"Training {self.model_name}...")
        
        X_train = datasets['train']['X']
        y_train = datasets['train']['y']
        
        # Adaptive n_estimators based on sample size
        if len(X_train) < 100:
            n_est = 200
        elif len(X_train) < 200:
            n_est = 300
        else:
            n_est = self.n_estimators
        
        logger.info(f"  Using {n_est} trees")
        
        # Train
        start_time = time.time()
        
        self.model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=self.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train.ravel())
        
        training_time = time.time() - start_time
        
        # Predictions
        predictions = {
            'train': self.model.predict(datasets['train']['X']),
            'check': self.model.predict(datasets['check']['X']),
            'test': self.model.predict(datasets['test']['X'])
        }
        
        # Metrics
        metrics = {
            'train': self.evaluate(datasets['train']['X'], datasets['train']['y'], 
                                  predictions['train'].reshape(-1, 1), 'train'),
            'check': self.evaluate(datasets['check']['X'], datasets['check']['y'], 
                                  predictions['check'].reshape(-1, 1), 'check'),
            'test': self.evaluate(datasets['test']['X'], datasets['test']['y'], 
                                 predictions['test'].reshape(-1, 1), 'test')
        }
        
        metrics['training_time'] = training_time
        
        logger.info(f"✓ Training completed in {training_time:.2f}s")
        
        return predictions, metrics
    
    def save_model(self, save_path):
        """Save Random Forest model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_file = save_path / 'model.pkl'
        joblib.dump(self.model, model_file)
        
        # Feature importance
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(save_path / 'feature_importance.csv', index=False)
        
        logger.info(f"✓ Model saved: {save_path}")


# ============================================================================
# GRADIENT BOOSTING TRAINER
# ============================================================================

class GradientBoostingTrainer(BaseModelTrainer):
    """Gradient Boosting Regressor"""
    
    def __init__(self, output_dir='trained_models', n_estimators=300, learning_rate=0.1):
        super().__init__('GradientBoosting', output_dir)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
    
    def train(self, datasets):
        """Train Gradient Boosting"""
        logger.info(f"Training {self.model_name}...")
        
        X_train = datasets['train']['X']
        y_train = datasets['train']['y']
        
        # Adaptive parameters
        if len(X_train) < 100:
            n_est = 150
            lr = 0.15
        elif len(X_train) < 200:
            n_est = 200
            lr = 0.12
        else:
            n_est = self.n_estimators
            lr = self.learning_rate
        
        logger.info(f"  Using {n_est} estimators, lr={lr}")
        
        # Train
        start_time = time.time()
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        self.model.fit(X_train, y_train.ravel())
        
        training_time = time.time() - start_time
        
        # Predictions
        predictions = {
            'train': self.model.predict(datasets['train']['X']),
            'check': self.model.predict(datasets['check']['X']),
            'test': self.model.predict(datasets['test']['X'])
        }
        
        # Metrics
        metrics = {
            'train': self.evaluate(datasets['train']['X'], datasets['train']['y'], 
                                  predictions['train'].reshape(-1, 1), 'train'),
            'check': self.evaluate(datasets['check']['X'], datasets['check']['y'], 
                                  predictions['check'].reshape(-1, 1), 'check'),
            'test': self.evaluate(datasets['test']['X'], datasets['test']['y'], 
                                 predictions['test'].reshape(-1, 1), 'test')
        }
        
        metrics['training_time'] = training_time
        
        logger.info(f"✓ Training completed in {training_time:.2f}s")
        
        return predictions, metrics
    
    def save_model(self, save_path):
        """Save GBM model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_path / 'model.pkl')
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(save_path / 'feature_importance.csv', index=False)
        
        logger.info(f"✓ Model saved: {save_path}")


# ============================================================================
# XGBOOST TRAINER
# ============================================================================

class XGBoostTrainer(BaseModelTrainer):
    """XGBoost Regressor"""
    
    def __init__(self, output_dir='trained_models', n_estimators=300):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        super().__init__('XGBoost', output_dir)
        self.n_estimators = n_estimators
    
    def train(self, datasets):
        """Train XGBoost"""
        logger.info(f"Training {self.model_name}...")
        
        X_train = datasets['train']['X']
        y_train = datasets['train']['y']
        X_check = datasets['check']['X']
        y_check = datasets['check']['y']
        
        # Adaptive parameters
        n_est = min(self.n_estimators, max(100, len(X_train)))
        
        logger.info(f"  Using {n_est} estimators")
        
        # Train
        start_time = time.time()
        
        self.model = XGBRegressor(
            n_estimators=n_est,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        self.model.fit(
            X_train, y_train.ravel(),
            eval_set=[(X_check, y_check.ravel())],
            early_stopping_rounds=20,
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        predictions = {
            'train': self.model.predict(datasets['train']['X']),
            'check': self.model.predict(datasets['check']['X']),
            'test': self.model.predict(datasets['test']['X'])
        }
        
        # Metrics
        metrics = {
            'train': self.evaluate(datasets['train']['X'], datasets['train']['y'], 
                                  predictions['train'].reshape(-1, 1), 'train'),
            'check': self.evaluate(datasets['check']['X'], datasets['check']['y'], 
                                  predictions['check'].reshape(-1, 1), 'check'),
            'test': self.evaluate(datasets['test']['X'], datasets['test']['y'], 
                                 predictions['test'].reshape(-1, 1), 'test')
        }
        
        metrics['training_time'] = training_time
        
        logger.info(f"✓ Training completed in {training_time:.2f}s")
        
        return predictions, metrics
    
    def save_model(self, save_path):
        """Save XGBoost model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(save_path / 'model.json'))
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(save_path / 'feature_importance.csv', index=False)
        
        logger.info(f"✓ Model saved: {save_path}")


# ============================================================================
# DNN TRAINER
# ============================================================================

class DNNTrainer(BaseModelTrainer):
    """Deep Neural Network Trainer"""
    
    def __init__(self, output_dir='trained_models', epochs=100):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed")
        
        super().__init__('DNN', output_dir)
        self.epochs = epochs
    
    def build_model(self, input_dim, output_dim):
        """Build DNN architecture"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, datasets):
        """Train DNN"""
        logger.info(f"Training {self.model_name}...")
        
        X_train = datasets['train']['X']
        y_train = datasets['train']['y']
        X_check = datasets['check']['X']
        y_check = datasets['check']['y']
        
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        
        # Adaptive epochs
        if len(X_train) < 100:
            epochs = 50
        else:
            epochs = self.epochs
        
        logger.info(f"  Training for {epochs} epochs")
        
        # Build model
        self.model = self.build_model(input_dim, output_dim)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_check, y_check),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        predictions = {
            'train': self.model.predict(datasets['train']['X'], verbose=0),
            'check': self.model.predict(datasets['check']['X'], verbose=0),
            'test': self.model.predict(datasets['test']['X'], verbose=0)
        }
        
        # Metrics
        metrics = {
            'train': self.evaluate(datasets['train']['X'], datasets['train']['y'], 
                                  predictions['train'], 'train'),
            'check': self.evaluate(datasets['check']['X'], datasets['check']['y'], 
                                  predictions['check'], 'check'),
            'test': self.evaluate(datasets['test']['X'], datasets['test']['y'], 
                                 predictions['test'], 'test')
        }
        
        metrics['training_time'] = training_time
        metrics['history'] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        
        logger.info(f"✓ Training completed in {training_time:.2f}s")
        
        return predictions, metrics
    
    def save_model(self, save_path):
        """Save DNN model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(save_path / 'model.h5'))
        
        logger.info(f"✓ Model saved: {save_path}")


# ============================================================================
# PARALLEL TRAINING PIPELINE
# ============================================================================

class ParallelTrainingPipeline:
    """Parallel model training with multiprocessing"""
    
    def __init__(self, output_dir='trained_models', max_workers=4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        logger.info(f"Parallel Training Pipeline: {max_workers} workers")
    
    def train_single_model(self, model_name, dataset_path):
        """Train single model on dataset"""
        try:
            # Create trainer
            if model_name == 'RF':
                trainer = RandomForestTrainer(self.output_dir)
            elif model_name == 'GBM':
                trainer = GradientBoostingTrainer(self.output_dir)
            elif model_name == 'XGBoost':
                trainer = XGBoostTrainer(self.output_dir)
            elif model_name == 'DNN':
                trainer = DNNTrainer(self.output_dir)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Load dataset
            datasets = trainer.load_dataset(dataset_path)
            
            # Train
            predictions, metrics = trainer.train(datasets)
            
            # Save
            save_path = self.output_dir / model_name / Path(dataset_path).name
            trainer.save_model(save_path)
            
            return {
                'model': model_name,
                'dataset': str(dataset_path),
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Training failed: {model_name} on {dataset_path}: {e}")
            return {
                'model': model_name,
                'dataset': str(dataset_path),
                'error': str(e),
                'success': False
            }
    
    def train_all_parallel(self, model_names, dataset_paths):
        """Train all models in parallel"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PARALLEL TRAINING")
        logger.info(f"Models: {model_names}")
        logger.info(f"Datasets: {len(dataset_paths)}")
        logger.info(f"Workers: {self.max_workers}")
        logger.info(f"{'='*80}\n")
        
        tasks = [
            (model, dataset)
            for model in model_names
            for dataset in dataset_paths
        ]
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.train_single_model, model, dataset): (model, dataset)
                for model, dataset in tasks
            }
            
            for future in as_completed(futures):
                model, dataset = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        logger.info(f"✓ {model} - {Path(dataset).name}")
                    else:
                        logger.error(f"✗ {model} - {Path(dataset).name}")
                        
                except Exception as e:
                    logger.error(f"✗ Task failed: {model} - {dataset}: {e}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ PARALLEL TRAINING COMPLETED")
        logger.info(f"Success: {sum(1 for r in results if r['success'])}/{len(results)}")
        logger.info(f"{'='*80}\n")
        
        return results


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n✅ MODEL_TRAINER.PY - COMPLETE VERSION")
    print("Includes: RF, GBM, XGBoost, DNN + Parallel Training")

# ==================== EKLEME BAŞI ====================
class ModelTrainer:
    """
    Generic model trainer wrapper
    """
    def __init__(self, model_type='auto'):
        self.model_type = model_type
        self.model = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train model"""
        from sklearn.ensemble import RandomForestRegressor
        
        if self.model is None:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        return {
            'model': self.model,
            'training_complete': True
        }
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
# ==================== EKLEME SON ====================