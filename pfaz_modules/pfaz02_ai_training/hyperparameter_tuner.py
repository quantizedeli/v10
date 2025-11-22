"""
Hyperparameter Tuning with Optuna
Optuna ile Hiperparametre Optimizasyonu

Desteklenen modeller:
- Random Forest
- Gradient Boosting
- XGBoost
- Deep Neural Networks (DNN)

Özellikler:
- Bayesian optimization
- Pruning (early stopping for trials)
- Multi-objective optimization
- Parallel trials
- Visualization of optimization history

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available! pip install optuna")

# Sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Local imports
from pathlib import Path
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# HYPERPARAMETER TUNER BASE CLASS
# ============================================================================

class HyperparameterTuner:
    """
    Hyperparameter tuning base class
    
    Özellikler:
    - Optuna ile Bayesian optimization
    - Pruning (kötü trial'ları erken durdur)
    - Multi-objective (R2 + training time)
    - History tracking
    """
    
    def __init__(self, model_type, output_dir='tuning_results', n_trials=50, timeout=3600):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna gerekli! pip install optuna")
        
        self.model_type = model_type
        self.output_dir = Path(output_dir) / model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = None
        
        logger.info(f"Hyperparameter Tuner başlatıldı: {model_type}")
    
    def optimize(self, X_train, y_train, X_val=None, y_val=None, cv_folds=5):
        """
        Hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, uses CV if None)
            y_val: Validation targets (optional)
            cv_folds: Cross-validation folds
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"HYPERPARAMETER TUNING: {self.model_type}")
        logger.info(f"Trials: {self.n_trials}, Timeout: {self.timeout}s")
        logger.info(f"{'='*80}\n")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.model_type}_tuning"
        )
        
        # Optimize
        objective_func = lambda trial: self._objective(
            trial, X_train, y_train, X_val, y_val, cv_folds
        )
        
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            n_jobs=1  # Parallel için artırılabilir
        )
        
        # Best parameters
        self.best_params = self.study.best_params
        
        logger.info(f"\n[OK] Tuning tamamlandı!")
        logger.info(f"Best R² Score: {self.study.best_value:.4f}")
        logger.info(f"Best Parameters: {self.best_params}")
        
        # Save results
        self._save_results()
        
        return self.best_params
    
    def _objective(self, trial, X_train, y_train, X_val, y_val, cv_folds):
        """Objective function - implemented in subclasses"""
        raise NotImplementedError("Subclass must implement _objective")
    
    def _save_results(self):
        """Save tuning results"""
        
        # Best parameters
        with open(self.output_dir / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Trials dataframe
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv(self.output_dir / 'trials_history.csv', index=False)
        
        # Summary
        summary = {
            'model_type': self.model_type,
            'n_trials': len(self.study.trials),
            'best_value': float(self.study.best_value),
            'best_params': self.best_params,
            'best_trial': self.study.best_trial.number,
            'datetime': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"[OK] Sonuçlar kaydedildi: {self.output_dir}")
    
    def plot_optimization_history(self):
        """Plot optimization history"""
        if not self.study:
            logger.warning("Study yok, önce optimize() çalıştırın")
            return
        
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_slice
            )
            
            # Optimization history
            fig1 = plot_optimization_history(self.study)
            fig1.write_html(str(self.output_dir / 'optimization_history.html'))
            
            # Parameter importances
            fig2 = plot_param_importances(self.study)
            fig2.write_html(str(self.output_dir / 'param_importances.html'))
            
            # Parallel coordinate
            fig3 = plot_parallel_coordinate(self.study)
            fig3.write_html(str(self.output_dir / 'parallel_coordinate.html'))
            
            # Slice plot
            fig4 = plot_slice(self.study)
            fig4.write_html(str(self.output_dir / 'slice_plot.html'))
            
            logger.info(f"[OK] Grafikler kaydedildi: {self.output_dir}")
            
        except Exception as e:
            logger.warning(f"Grafik oluşturma hatası: {e}")


# ============================================================================
# RANDOM FOREST TUNER
# ============================================================================

class RandomForestTuner(HyperparameterTuner):
    """Random Forest hyperparameter tuning"""
    
    def __init__(self, output_dir='tuning_results', n_trials=50, timeout=3600):
        super().__init__('RandomForest', output_dir, n_trials, timeout)
    
    def _objective(self, trial, X_train, y_train, X_val, y_val, cv_folds):
        """RF objective function"""
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 100, step=10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Create model
        model = RandomForestRegressor(**params)
        
        # Evaluate
        if X_val is not None and y_val is not None:
            # Use validation set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
        else:
            # Use cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            score = scores.mean()
        
        return score


# ============================================================================
# GRADIENT BOOSTING TUNER
# ============================================================================

class GradientBoostingTuner(HyperparameterTuner):
    """Gradient Boosting hyperparameter tuning"""
    
    def __init__(self, output_dir='tuning_results', n_trials=50, timeout=3600):
        super().__init__('GradientBoosting', output_dir, n_trials, timeout)
    
    def _objective(self, trial, X_train, y_train, X_val, y_val, cv_folds):
        """GBM objective function"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
        else:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            score = scores.mean()
        
        return score


# ============================================================================
# XGBOOST TUNER
# ============================================================================

class XGBoostTuner(HyperparameterTuner):
    """XGBoost hyperparameter tuning"""
    
    def __init__(self, output_dir='tuning_results', n_trials=50, timeout=3600):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost yüklü değil!")
        super().__init__('XGBoost', output_dir, n_trials, timeout)
    
    def _objective(self, trial, X_train, y_train, X_val, y_val, cv_folds):
        """XGBoost objective function"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBRegressor(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
        else:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            score = scores.mean()
        
        return score


# ============================================================================
# DNN TUNER
# ============================================================================

class DNNTuner(HyperparameterTuner):
    """Deep Neural Network hyperparameter tuning"""
    
    def __init__(self, output_dir='tuning_results', n_trials=30, timeout=3600):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow yüklü değil!")
        super().__init__('DNN', output_dir, n_trials, timeout)
    
    def _objective(self, trial, X_train, y_train, X_val, y_val, cv_folds):
        """DNN objective function"""
        
        # Architecture
        n_layers = trial.suggest_int('n_layers', 2, 5)
        layers_config = []
        for i in range(n_layers):
            n_units = trial.suggest_int(f'n_units_l{i}', 32, 256, step=32)
            layers_config.append(n_units)
        
        dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        
        for i, units in enumerate(layers_config):
            model.add(layers.Dense(units, activation='relu'))
            if i < len(layers_config) - 1:
                model.add(layers.Dropout(dropout_rate))
        
        model.add(layers.Dense(y_train.shape[1] if len(y_train.shape) > 1 else 1))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            y_pred = model.predict(X_val, verbose=0)
            score = r2_score(y_val, y_pred)
        else:
            # CV için basitleştirilmiş
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            y_pred = model.predict(X_train, verbose=0)
            score = r2_score(y_train, y_pred)
        
        # Pruning
        for epoch in range(len(history.history['val_loss'])):
            trial.report(history.history['val_loss'][epoch], epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return score


# ============================================================================
# UNIFIED TUNER (ALL MODELS)
# ============================================================================

class UnifiedTuner:
    """
    All models için unified tuning interface
    """
    
    def __init__(self, output_dir='tuning_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tuners = {
            'RF': RandomForestTuner(output_dir=output_dir),
            'GBM': GradientBoostingTuner(output_dir=output_dir),
        }
        
        if XGBOOST_AVAILABLE:
            self.tuners['XGBoost'] = XGBoostTuner(output_dir=output_dir)
        
        if TF_AVAILABLE:
            self.tuners['DNN'] = DNNTuner(output_dir=output_dir, n_trials=20)
        
        logger.info(f"Unified Tuner başlatıldı: {list(self.tuners.keys())}")
    
    def tune_all_models(self, X_train, y_train, X_val=None, y_val=None, 
                       models=None, n_trials=50, timeout=3600):
        """
        Tüm modeller için tuning
        
        Args:
            models: List of model names to tune (None = all)
        """
        
        if models is None:
            models = list(self.tuners.keys())
        
        results = {}
        
        for model_name in models:
            if model_name not in self.tuners:
                logger.warning(f"Model bulunamadı: {model_name}")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"TUNING: {model_name}")
            logger.info(f"{'='*80}")
            
            tuner = self.tuners[model_name]
            tuner.n_trials = n_trials
            tuner.timeout = timeout
            
            try:
                best_params = tuner.optimize(X_train, y_train, X_val, y_val)
                tuner.plot_optimization_history()
                
                results[model_name] = {
                    'best_params': best_params,
                    'best_score': float(tuner.study.best_value),
                    'n_trials': len(tuner.study.trials)
                }
                
                logger.info(f"[OK] {model_name} tuning tamamlandı")
                
            except Exception as e:
                logger.error(f"[FAIL] {model_name} tuning hatası: {e}")
                results[model_name] = {'error': str(e)}
        
        # Save combined results
        with open(self.output_dir / 'all_models_best_params.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n[OK] Tüm tuning işlemleri tamamlandı!")
        logger.info(f"Sonuçlar: {self.output_dir}")
        
        return results


# ============================================================================
# MAIN TEST
# ============================================================================

def test_hyperparameter_tuning():
    """Test hyperparameter tuning"""
    
    if not OPTUNA_AVAILABLE:
        print("Optuna yüklü değil, test atlanıyor")
        return
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING TEST")
    print("="*80)
    
    # Dummy data
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * (-1) + np.random.randn(n_samples) * 0.1
    
    # Split
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    # Test unified tuner
    tuner = UnifiedTuner(output_dir='test_tuning')
    
    results = tuner.tune_all_models(
        X_train, y_train,
        X_val, y_val,
        models=['RF', 'GBM'],  # Hızlı test için sadece 2 model
        n_trials=10,  # Az trial
        timeout=300
    )
    
    print("\n" + "="*80)
    print("TUNING SONUÇLARI:")
    print("="*80)
    for model, result in results.items():
        if 'error' not in result:
            print(f"\n{model}:")
            print(f"  Best Score: {result['best_score']:.4f}")
            print(f"  Best Params: {result['best_params']}")
    
    print("\n[OK] Test tamamlandı!")


if __name__ == "__main__":
    test_hyperparameter_tuning()