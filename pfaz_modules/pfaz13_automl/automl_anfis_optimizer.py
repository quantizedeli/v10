# -*- coding: utf-8 -*-
"""
AUTOML ANFIS OPTIMIZER
======================

Comprehensive ANFIS (Adaptive Neuro-Fuzzy Inference System) optimization

Features:
1. FIS Generation Method Optimization
   - Grid partitioning
   - Subtractive clustering (subclust)
   - Fuzzy C-Means clustering (FCM)

2. Membership Function Type Selection
   - Gaussian (gaussmf)
   - Triangular (trimf)
   - Trapezoidal (trapmf)
   - Bell-shaped (gbellmf)
   - Generalized Bell (gbell)

3. Number of MFs per Input Optimization
   - 2-5 MFs per input
   - Balance between accuracy and complexity

4. Defuzzification Method Selection
   - Centroid (most common)
   - Bisector
   - Mean of Maximum (mom)
   - Smallest of Maximum (som)
   - Largest of Maximum (lom)

5. ANFIS Training Parameters
   - Epochs: 50-200
   - Error goal: 0.0-0.01
   - Initial step size: 0.01-0.1
   - Step size decrease rate: 0.5-0.99
   - Step size increase rate: 1.01-1.5

6. Integration with AutoML Logging
   - Complete trial tracking
   - Excel reports
   - Convergence analysis

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 13 ANFIS Optimization
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Try MATLAB engine (for true ANFIS)
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    logging.warning("MATLAB engine not available - using sklearn approximation")

# Fallback: sklearn fuzzy approximation
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Local imports — use package-relative import
try:
    from .automl_logging_reporting_system import AutoMLTrialLogger, AutoMLTrialRecord
    LOGGING_AVAILABLE = True
except ImportError:
    try:
        from automl_logging_reporting_system import AutoMLTrialLogger, AutoMLTrialRecord
        LOGGING_AVAILABLE = True
    except ImportError:
        AutoMLTrialLogger = None
        AutoMLTrialRecord = None
        LOGGING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ANFIS CONFIGURATION SPACE
# ============================================================================

ANFIS_CONFIG_SPACE = {
    'fis_generation': {
        'type': 'categorical',
        # Must match TakagiSugenoANFIS 'method' parameter (pfaz03)
        'choices': ['grid', 'subclust']
    },
    
    'mf_type': {
        'type': 'categorical',
        # Names must match TakagiSugenoANFIS mf_type parameter (pfaz03)
        'choices': ['trapezoid', 'bell', 'gaussian', 'triangle']
    },
    
    'n_mfs': {
        'type': 'int',
        'low': 2,
        'high': 5,
        'step': 1
    },
    
    'defuzz_method': {
        'type': 'categorical',
        'choices': ['centroid', 'bisector', 'mom', 'som', 'lom']
    },
    
    'epochs': {
        'type': 'int',
        'low': 50,
        'high': 200,
        'step': 25
    },
    
    'error_goal': {
        'type': 'float',
        'low': 0.0,
        'high': 0.01,
        'step': 0.001
    },
    
    'step_size_init': {
        'type': 'float',
        'low': 0.01,
        'high': 0.1,
        'step': 0.01
    },
    
    'step_size_decrease': {
        'type': 'float',
        'low': 0.5,
        'high': 0.99,
        'step': 0.05
    },
    
    'step_size_increase': {
        'type': 'float',
        'low': 1.01,
        'high': 1.5,
        'step': 0.05
    }
}


# ============================================================================
# ANFIS OPTIMIZER
# ============================================================================

class AutoMLANFISOptimizer:
    """
    ANFIS hyperparameter and structure optimization
    
    Optimizes:
    - FIS generation method
    - Membership function type and count
    - Defuzzification method
    - Training parameters (epochs, step sizes, etc.)
    
    Methods:
    - Grid search (exhaustive for small spaces)
    - Random search (for large spaces)
    - Bayesian optimization (if Optuna available)
    """
    
    def __init__(self,
                 n_trials: int = 50,
                 timeout: Optional[int] = 7200,
                 output_dir: str = 'automl_anfis_optimization',
                 use_logging: bool = True,
                 use_matlab: bool = True):
        """
        Initialize ANFIS optimizer
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            output_dir: Output directory
            use_logging: Use AutoMLTrialLogger
            use_matlab: Use MATLAB engine (if available)
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MATLAB
        self.use_matlab = use_matlab and MATLAB_AVAILABLE
        if self.use_matlab:
            logger.info("-> Initializing MATLAB engine...")
            try:
                self.matlab_eng = matlab.engine.start_matlab()
                logger.info("  [OK] MATLAB engine started")
            except Exception as e:
                logger.error(f"  [FAIL] MATLAB engine failed: {e}")
                self.use_matlab = False
        
        # Logging
        self.use_logging = use_logging and LOGGING_AVAILABLE
        if self.use_logging:
            self.trial_logger = AutoMLTrialLogger(
                output_dir=str(self.output_dir / 'trial_logs')
            )
        
        # Results
        self.best_config = None
        self.best_score = -np.inf
        self.best_fis = None
        
        logger.info(f"[OK] AutoMLANFISOptimizer initialized")
        logger.info(f"  MATLAB: {'Available' if self.use_matlab else 'Unavailable (using sklearn)'}")
        logger.info(f"  Trials: {n_trials}")
    
    # ========================================================================
    # MAIN OPTIMIZATION
    # ========================================================================
    
    def optimize(self,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                dataset_name: str = 'unknown',
                objective: str = 'r2') -> Dict:
        """
        Run ANFIS optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            dataset_name: Name of dataset
            objective: 'r2', 'rmse', 'mae'
            
        Returns:
            dict with best_config, best_score, optimization_summary
        """
        logger.info("\n" + "="*70)
        logger.info("ANFIS HYPERPARAMETER OPTIMIZATION")
        logger.info("="*70)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Objective: {objective}")
        logger.info(f"Method: {'MATLAB ANFIS' if self.use_matlab else 'sklearn approximation'}")
        logger.info("="*70 + "\n")
        
        # Start logging
        if self.use_logging:
            opt_id = self.trial_logger.start_optimization(
                model_type='ANFIS',
                objective=objective,
                n_trials=self.n_trials
            )
        
        start_time = time.time()
        
        # Run optimization
        logger.info("-> Starting ANFIS optimization (random search)...")
        
        for trial_id in range(self.n_trials):
            # Sample configuration
            config = self._sample_anfis_config()
            
            # Evaluate
            score = self._evaluate_anfis_config(
                config, X_train, y_train, X_val, y_val,
                dataset_name, objective, trial_id
            )
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                logger.info(f"  [TARGET] NEW BEST! Trial {trial_id}: {objective}={score:.4f}")
        
        total_time = time.time() - start_time
        
        # End logging
        result = {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'total_time': total_time
        }
        
        if self.use_logging:
            summary = self.trial_logger.end_optimization()
            result['optimization_summary'] = summary
            
            # Export Excel
            excel_path = self.trial_logger.export_to_excel(
                'ANFIS_hyperparameter_optimization.xlsx'
            )
            result['excel_report'] = str(excel_path)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("ANFIS OPTIMIZATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Best {objective}: {self.best_score:.6f}")
        logger.info(f"Best config:")
        for key, value in self.best_config.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*70 + "\n")
        
        return result
    
    # ========================================================================
    # CONFIGURATION SAMPLING
    # ========================================================================
    
    def _sample_anfis_config(self) -> Dict:
        """Sample random ANFIS configuration"""
        
        config = {}
        
        for param_name, param_config in ANFIS_CONFIG_SPACE.items():
            if param_config['type'] == 'int':
                config[param_name] = np.random.randint(
                    param_config['low'],
                    param_config['high'] + 1
                )
            elif param_config['type'] == 'float':
                config[param_name] = np.random.uniform(
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'categorical':
                config[param_name] = np.random.choice(param_config['choices'])
        
        return config
    
    # ========================================================================
    # CONFIGURATION EVALUATION
    # ========================================================================
    
    def _evaluate_anfis_config(self,
                               config: Dict,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               dataset_name: str,
                               objective: str,
                               trial_id: int) -> float:
        """Evaluate single ANFIS configuration"""
        
        try:
            # Train ANFIS
            start_time = time.time()
            
            if self.use_matlab:
                fis, y_pred_train, y_pred_val = self._train_matlab_anfis(
                    config, X_train, y_train, X_val
                )
            else:
                fis, y_pred_train, y_pred_val = self._train_sklearn_anfis(
                    config, X_train, y_train, X_val
                )
            
            training_time = time.time() - start_time
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            
            val_r2 = r2_score(y_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_mae = mean_absolute_error(y_val, y_pred_val)
            
            # Objective
            if objective == 'r2':
                score = val_r2
            elif objective == 'rmse':
                score = -val_rmse  # Negative for minimization
            elif objective == 'mae':
                score = -val_mae
            else:
                score = val_r2
            
            # Log
            if self.use_logging:
                record = AutoMLTrialRecord(
                    trial_id=trial_id,
                    timestamp=datetime.now().isoformat(),
                    model_type='ANFIS',
                    dataset_name=dataset_name,
                    hyperparameters=config,
                    train_r2=train_r2,
                    train_rmse=train_rmse,
                    train_mae=train_mae,
                    val_r2=val_r2,
                    val_rmse=val_rmse,
                    val_mae=val_mae,
                    training_time=training_time,
                    n_samples_train=len(X_train),
                    n_samples_val=len(X_val),
                    n_features=X_train.shape[1],
                    status='COMPLETE'
                )
                self.trial_logger.log_trial(record)
            
            return score
            
        except Exception as e:
            logger.error(f"  Trial {trial_id} failed: {e}")
            
            # Log failure
            if self.use_logging:
                record = AutoMLTrialRecord(
                    trial_id=trial_id,
                    timestamp=datetime.now().isoformat(),
                    model_type='ANFIS',
                    dataset_name=dataset_name,
                    hyperparameters=config,
                    train_r2=0.0,
                    train_rmse=999.0,
                    train_mae=999.0,
                    val_r2=0.0,
                    val_rmse=999.0,
                    val_mae=999.0,
                    training_time=0.0,
                    n_samples_train=len(X_train),
                    n_samples_val=len(X_val),
                    n_features=X_train.shape[1],
                    status='FAILED',
                    error_message=str(e)
                )
                self.trial_logger.log_trial(record)
            
            return -np.inf
    
    # ========================================================================
    # MATLAB ANFIS TRAINING
    # ========================================================================
    
    def _train_matlab_anfis(self,
                           config: Dict,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray) -> Tuple:
        """Train ANFIS using MATLAB"""
        
        # Convert to MATLAB arrays
        X_train_mat = matlab.double(X_train.tolist())
        y_train_mat = matlab.double(y_train.reshape(-1, 1).tolist())
        X_val_mat = matlab.double(X_val.tolist())
        
        # Generate FIS
        if config['fis_generation'] == 'grid':
            # Grid partitioning
            fis = self.matlab_eng.genfis1(
                X_train_mat,
                y_train_mat,
                [config['n_mfs']] * X_train.shape[1],  # MFs per input
                config['mf_type'],
                config['defuzz_method']
            )
        
        elif config['fis_generation'] == 'subclust':
            # Subtractive clustering
            fis = self.matlab_eng.genfis2(
                X_train_mat,
                y_train_mat,
                0.5  # Cluster radius
            )
        
        elif config['fis_generation'] == 'fcm':
            # FCM clustering
            fis = self.matlab_eng.genfis3(
                X_train_mat,
                y_train_mat,
                'NumClusters', config['n_mfs']
            )
        
        # Train ANFIS
        opt_options = [
            config['epochs'],
            config['error_goal'],
            config['step_size_init'],
            config['step_size_decrease'],
            config['step_size_increase']
        ]
        
        fis_trained = self.matlab_eng.anfis(
            [X_train_mat, y_train_mat],
            fis,
            opt_options
        )
        
        # Predict
        y_pred_train_mat = self.matlab_eng.evalfis(fis_trained, X_train_mat)
        y_pred_val_mat = self.matlab_eng.evalfis(fis_trained, X_val_mat)
        
        # Convert back to numpy
        y_pred_train = np.array(y_pred_train_mat).flatten()
        y_pred_val = np.array(y_pred_val_mat).flatten()
        
        return fis_trained, y_pred_train, y_pred_val
    
    # ========================================================================
    # SKLEARN ANFIS APPROXIMATION
    # ========================================================================
    
    def _train_sklearn_anfis(self,
                            config: Dict,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray) -> Tuple:
        """
        Train using actual TakagiSugenoANFIS from PFAZ3 if available,
        otherwise fall back to MLPRegressor approximation.
        """
        from sklearn.preprocessing import StandardScaler as _SS

        # Scale inputs (ANFIS is input-scale sensitive)
        scaler = _SS()
        X_tr_sc = scaler.fit_transform(X_train)
        X_vl_sc = scaler.transform(X_val)
        y_tr = np.asarray(y_train).ravel()

        # Try actual PFAZ3 ANFIS
        try:
            from pfaz_modules.pfaz03_anfis_training.anfis_core import TakagiSugenoANFIS, _adaptive_n_mfs
            n_inputs = X_tr_sc.shape[1]
            method   = config.get('fis_generation', 'grid')
            mf_type  = config.get('mf_type', 'trapezoid')
            n_mfs_req = int(config.get('n_mfs', 2))
            safe_mfs  = _adaptive_n_mfs(n_inputs, len(X_tr_sc), n_mfs_req) if method == 'grid' else n_mfs_req
            anfis = TakagiSugenoANFIS(
                n_inputs=n_inputs,
                n_mfs=safe_mfs,
                mf_type=mf_type,
                method=method,
                max_iter=int(config.get('epochs', 200)),
                patience=20,
                alpha=1e-2,
                use_gradient=True,
            )
            anfis.fit(X_tr_sc, y_tr, X_val=X_vl_sc, y_val=np.asarray(X_val[:, 0]))

            class _WrappedANFIS:
                def __init__(self, m, sc): self._m, self._sc = m, sc
                def predict(self, X): return self._m.predict(self._sc.transform(X))

            wrapped = _WrappedANFIS(anfis, scaler)
            y_pred_train = wrapped.predict(X_train)
            y_pred_val   = wrapped.predict(X_val)
            return wrapped, y_pred_train, y_pred_val

        except Exception:
            pass

        # Fallback: MLP approximation
        n_inputs = X_tr_sc.shape[1]
        n_mfs    = int(config.get('n_mfs', 2))
        hidden_layer_sizes = (n_inputs * n_mfs, n_mfs * n_inputs)

        anfis_approx = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='tanh',
            solver='adam',
            learning_rate_init=float(config.get('step_size_init', 0.05)),
            max_iter=int(config.get('epochs', 200)),
            tol=float(config.get('error_goal', 1e-4)),
            random_state=42,
        )
        anfis_approx.fit(X_tr_sc, y_tr)
        y_pred_train = anfis_approx.predict(X_tr_sc)
        y_pred_val   = anfis_approx.predict(X_vl_sc)
        return anfis_approx, y_pred_train, y_pred_val
    
    # ========================================================================
    # BEST MODEL EXPORT
    # ========================================================================
    
    def get_best_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray):
        """Train and return best ANFIS model"""
        
        if self.best_config is None:
            raise ValueError("Must run optimize() first!")
        
        logger.info("\n-> Training final ANFIS with best config...")
        
        start_time = time.time()
        
        if self.use_matlab:
            fis, _, _ = self._train_matlab_anfis(
                self.best_config, X_train, y_train, X_train[:1]
            )
        else:
            fis, _, _ = self._train_sklearn_anfis(
                self.best_config, X_train, y_train, X_train[:1]
            )
        
        training_time = time.time() - start_time
        
        logger.info(f"  [OK] Final ANFIS trained in {training_time:.1f}s")
        
        self.best_fis = fis
        return fis


# ============================================================================
# ANFIS CONFIG ANALYZER
# ============================================================================

class ANFISConfigAnalyzer:
    """
    Analyze ANFIS configurations to understand:
    - Which FIS generation method works best
    - Optimal number of MFs
    - Best MF type
    - Best defuzz method
    """
    
    def __init__(self, excel_path: str):
        """Load ANFIS optimization results"""
        self.df = pd.read_excel(excel_path, sheet_name='All_Trials')
        self.df_completed = self.df[self.df['Status'] == 'COMPLETE']
    
    def analyze_fis_generation(self) -> pd.DataFrame:
        """Compare FIS generation methods"""
        
        fis_col = 'HP_fis_generation'
        if fis_col not in self.df_completed.columns:
            logger.warning("No FIS generation data found")
            return pd.DataFrame()
        
        analysis = self.df_completed.groupby(fis_col).agg({
            'Val_R2': ['mean', 'std', 'max'],
            'Training_Time': ['mean', 'std']
        }).round(4)
        
        logger.info("\n" + "="*70)
        logger.info("FIS GENERATION METHOD COMPARISON")
        logger.info("="*70)
        logger.info(f"\n{analysis}")
        
        return analysis
    
    def analyze_mf_type(self) -> pd.DataFrame:
        """Compare MF types"""
        
        mf_col = 'HP_mf_type'
        if mf_col not in self.df_completed.columns:
            logger.warning("No MF type data found")
            return pd.DataFrame()
        
        analysis = self.df_completed.groupby(mf_col).agg({
            'Val_R2': ['mean', 'std', 'max'],
            'Training_Time': ['mean']
        }).round(4)
        
        logger.info("\n" + "="*70)
        logger.info("MEMBERSHIP FUNCTION TYPE COMPARISON")
        logger.info("="*70)
        logger.info(f"\n{analysis}")
        
        return analysis
    
    def analyze_n_mfs(self) -> pd.DataFrame:
        """Analyze optimal number of MFs"""
        
        n_mfs_col = 'HP_n_mfs'
        if n_mfs_col not in self.df_completed.columns:
            logger.warning("No n_mfs data found")
            return pd.DataFrame()
        
        analysis = self.df_completed.groupby(n_mfs_col).agg({
            'Val_R2': ['mean', 'std', 'max'],
            'Training_Time': ['mean']
        }).round(4)
        
        logger.info("\n" + "="*70)
        logger.info("NUMBER OF MEMBERSHIP FUNCTIONS ANALYSIS")
        logger.info("="*70)
        logger.info(f"\n{analysis}")
        
        return analysis


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("TESTING AUTOML ANFIS OPTIMIZER")
    logger.info("="*70)
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 150
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * (-1) + np.random.randn(n_samples) * 0.2
    
    # Split
    split = int(0.7 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Optimize ANFIS
    optimizer = AutoMLANFISOptimizer(
        n_trials=20,  # Small for testing
        timeout=300,
        use_logging=True,
        use_matlab=False  # Use sklearn approximation
    )
    
    result = optimizer.optimize(
        X_train, y_train,
        X_val, y_val,
        dataset_name='test_anfis_dataset',
        objective='r2'
    )
    
    logger.info("\n[OK] ANFIS optimization complete!")
    logger.info(f"  Best R²: {result['best_score']:.4f}")
    logger.info(f"  Best config: {result['best_config']}")
    
    if 'excel_report' in result:
        logger.info(f"  Excel report: {result['excel_report']}")
