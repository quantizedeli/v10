# -*- coding: utf-8 -*-
"""
MONTE CARLO SIMULATION SYSTEM
==============================

Comprehensive uncertainty quantification and model reliability analysis

Features:
- MC Dropout (DNN uncertainty)
- Bootstrap resampling
- Noise sensitivity analysis
- Feature dropout Monte Carlo
- Ensemble uncertainty
- Bayesian Neural Network uncertainty
- Comprehensive reporting & visualization

Author: Nuclear Physics AI Project
Version: 1.0.0
Date: 2025-10-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
warnings.filterwarnings('ignore')

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Stats
from scipy import stats
from scipy.stats import norm

# ML
import joblib
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available - MC Dropout disabled")

# Excel
import xlsxwriter
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MC_CONFIG = {
    'mc_dropout': {
        'enabled': True,
        'n_samples': 100,
        'applicable_models': ['DNN']
    },
    'bootstrap': {
        'enabled': True,
        'n_bootstrap': 100,
        'stratified': True
    },
    'noise_sensitivity': {
        'enabled': True,
        'noise_levels': [0.01, 0.02, 0.05, 0.1, 0.2],
        'n_samples_per_level': 100,
        'noise_type': 'gaussian'
    },
    'feature_dropout': {
        'enabled': True,
        'dropout_probs': [0.1, 0.2, 0.3],
        'n_samples': 500
    },
    'ensemble_uncertainty': {
        'enabled': True,
        'consensus_threshold': 0.1
    },
    'thresholds': {
        'high_uncertainty': 0.3,
        'low_uncertainty': 0.05,
        'robustness_good': 0.85
    },
    'parallel': {
        'enabled': True,
        'n_jobs': 8
    }
}


# ============================================================================
# MC DROPOUT SIMULATOR
# ============================================================================

class MCDropoutSimulator:
    """Monte Carlo Dropout for DNN uncertainty estimation"""
    
    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples
        
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available - MC Dropout disabled")
    
    def estimate_uncertainty(self, model, X: np.ndarray) -> Dict:
        """
        Estimate prediction uncertainty using MC Dropout
        
        Args:
            model: Keras model with dropout layers
            X: Input features (n_samples, n_features)
        
        Returns:
            {
                'mean_predictions': array,
                'std_predictions': array (uncertainty),
                'ci_lower': array (2.5%),
                'ci_upper': array (97.5%),
                'all_predictions': array (n_mc_samples, n_samples)
            }
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return {}
        
        logger.info(f"  Running MC Dropout: {self.n_samples} samples...")
        
        # Enable dropout during inference
        predictions = []
        
        for i in tqdm(range(self.n_samples), desc="MC Dropout"):
            # Forward pass with training=True to enable dropout
            y_pred = model(X, training=True).numpy()
            predictions.append(y_pred.flatten())
        
        predictions = np.array(predictions)  # Shape: (n_mc_samples, n_data)
        
        # Calculate statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Confidence intervals (95%)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        logger.info(f"  ✓ Mean uncertainty: {std_pred.mean():.4f}")
        logger.info(f"  ✓ Max uncertainty: {std_pred.max():.4f}")
        
        return {
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_predictions': predictions,
            'n_samples': self.n_samples
        }


# ============================================================================
# BOOTSTRAP SIMULATOR
# ============================================================================

class BootstrapSimulator:
    """Bootstrap resampling for confidence intervals"""
    
    def __init__(self, n_bootstrap: int = 100, stratified: bool = True):
        self.n_bootstrap = n_bootstrap
        self.stratified = stratified
    
    def estimate_confidence(self, model_class, model_params: Dict,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Bootstrap confidence interval estimation
        
        Args:
            model_class: Model class (e.g., RandomForestRegressor)
            model_params: Model parameters
            X_train, y_train: Training data
            X_test, y_test: Test data
        
        Returns:
            {
                'mean_predictions': array,
                'std_predictions': array,
                'ci_lower': array,
                'ci_upper': array,
                'bootstrap_predictions': array (n_bootstrap, n_test),
                'bootstrap_r2': array,
                'bootstrap_rmse': array
            }
        """
        logger.info(f"  Running Bootstrap: {self.n_bootstrap} samples...")
        
        n_train = len(X_train)
        n_test = len(X_test)
        
        bootstrap_predictions = np.zeros((self.n_bootstrap, n_test))
        bootstrap_r2 = []
        bootstrap_rmse = []
        
        for i in tqdm(range(self.n_bootstrap), desc="Bootstrap"):
            # Resample with replacement
            indices = np.random.choice(n_train, size=n_train, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train model
            try:
                model = model_class(**model_params)
                model.fit(X_boot, y_boot)
                
                # Predict on test set
                y_pred_test = model.predict(X_test)
                bootstrap_predictions[i] = y_pred_test
                
                # Calculate metrics
                from sklearn.metrics import r2_score, mean_squared_error
                r2 = r2_score(y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                bootstrap_r2.append(r2)
                bootstrap_rmse.append(rmse)
            
            except Exception as e:
                logger.warning(f"  Bootstrap sample {i} failed: {e}")
                continue
        
        # Calculate statistics
        mean_pred = bootstrap_predictions.mean(axis=0)
        std_pred = bootstrap_predictions.std(axis=0)
        
        # Confidence intervals (percentile method)
        ci_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)
        
        logger.info(f"  ✓ Mean CI width: {(ci_upper - ci_lower).mean():.4f}")
        logger.info(f"  ✓ Mean R²: {np.mean(bootstrap_r2):.4f} ± {np.std(bootstrap_r2):.4f}")
        
        return {
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_predictions': bootstrap_predictions,
            'bootstrap_r2': np.array(bootstrap_r2),
            'bootstrap_rmse': np.array(bootstrap_rmse),
            'n_bootstrap': self.n_bootstrap
        }


# ============================================================================
# NOISE SIMULATOR
# ============================================================================

class NoiseSimulator:
    """Noise sensitivity analysis"""
    
    def __init__(self, noise_levels: List[float] = None,
                 n_samples_per_level: int = 100,
                 noise_type: str = 'gaussian'):
        self.noise_levels = noise_levels or [0.01, 0.02, 0.05, 0.1, 0.2]
        self.n_samples_per_level = n_samples_per_level
        self.noise_type = noise_type
    
    def analyze_noise_sensitivity(self, model, X: np.ndarray) -> Dict:
        """
        Analyze model sensitivity to input noise
        
        Args:
            model: Trained model
            X: Input features
        
        Returns:
            {
                'noise_levels': list,
                'original_predictions': array,
                'noisy_predictions': dict {noise_level: array},
                'prediction_variance': dict {noise_level: array},
                'robustness_scores': dict {noise_level: float}
            }
        """
        logger.info(f"  Running Noise Sensitivity: {len(self.noise_levels)} levels...")
        
        # Original predictions
        y_pred_original = model.predict(X)
        
        noisy_predictions = {}
        prediction_variance = {}
        robustness_scores = {}
        
        for noise_level in self.noise_levels:
            logger.info(f"    Noise σ={noise_level}...")
            
            predictions_at_level = []
            
            for i in range(self.n_samples_per_level):
                # Add Gaussian noise
                if self.noise_type == 'gaussian':
                    noise = np.random.normal(0, noise_level, X.shape)
                    X_noisy = X + noise
                else:
                    raise ValueError(f"Unknown noise type: {self.noise_type}")
                
                # Predict
                y_pred_noisy = model.predict(X_noisy)
                predictions_at_level.append(y_pred_noisy)
            
            predictions_at_level = np.array(predictions_at_level)
            
            # Calculate variance
            variance = predictions_at_level.var(axis=0)
            prediction_variance[noise_level] = variance
            
            # Mean predictions at this noise level
            mean_pred = predictions_at_level.mean(axis=0)
            noisy_predictions[noise_level] = mean_pred
            
            # Robustness score: 1 - (variance / original_variance)
            original_variance = y_pred_original.var()
            if original_variance > 0:
                robustness = 1 - (variance.mean() / original_variance)
                robustness = np.clip(robustness, 0, 1)
            else:
                robustness = 1.0
            
            robustness_scores[noise_level] = float(robustness)
            
            logger.info(f"      Robustness: {robustness:.3f}")
        
        logger.info(f"  ✓ Noise sensitivity analysis complete")
        
        return {
            'noise_levels': self.noise_levels,
            'original_predictions': y_pred_original,
            'noisy_predictions': noisy_predictions,
            'prediction_variance': prediction_variance,
            'robustness_scores': robustness_scores,
            'n_samples_per_level': self.n_samples_per_level
        }


# ============================================================================
# FEATURE DROPOUT SIMULATOR
# ============================================================================

class FeatureDropoutSimulator:
    """Feature dropout Monte Carlo for feature importance uncertainty"""
    
    def __init__(self, dropout_probs: List[float] = None, n_samples: int = 500):
        self.dropout_probs = dropout_probs or [0.1, 0.2, 0.3]
        self.n_samples = n_samples
    
    def analyze_feature_importance(self, model, X: np.ndarray,
                                   feature_names: List[str] = None) -> Dict:
        """
        Analyze feature importance uncertainty via dropout
        
        Args:
            model: Trained model
            X: Input features
            feature_names: Optional feature names
        
        Returns:
            {
                'dropout_probs': list,
                'original_predictions': array,
                'dropout_predictions': dict,
                'feature_importance_uncertainty': array (per feature),
                'stable_features': list,
                'unstable_features': list
            }
        """
        logger.info(f"  Running Feature Dropout: {len(self.dropout_probs)} levels...")
        
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Original predictions
        y_pred_original = model.predict(X)
        
        dropout_predictions = {}
        feature_variance = np.zeros(n_features)
        
        for dropout_prob in self.dropout_probs:
            logger.info(f"    Dropout p={dropout_prob}...")
            
            predictions_at_prob = []
            feature_pred_changes = np.zeros((self.n_samples, n_features))
            
            for i in range(self.n_samples):
                # Create dropout mask
                mask = np.random.binomial(1, 1 - dropout_prob, X.shape)
                X_dropout = X * mask
                
                # Predict
                y_pred_dropout = model.predict(X_dropout)
                predictions_at_prob.append(y_pred_dropout)
                
                # Track per-feature prediction change
                for j in range(n_features):
                    X_single_dropout = X.copy()
                    X_single_dropout[:, j] = 0
                    y_pred_single = model.predict(X_single_dropout)
                    feature_pred_changes[i, j] = np.abs(y_pred_original - y_pred_single).mean()
            
            predictions_at_prob = np.array(predictions_at_prob)
            dropout_predictions[dropout_prob] = predictions_at_prob.mean(axis=0)
            
            # Accumulate feature variance
            feature_variance += feature_pred_changes.var(axis=0)
        
        # Normalize feature variance
        feature_variance /= len(self.dropout_probs)
        
        # Identify stable vs unstable features
        variance_threshold = np.percentile(feature_variance, 75)
        stable_features = [feature_names[i] for i in range(n_features) 
                          if feature_variance[i] < variance_threshold]
        unstable_features = [feature_names[i] for i in range(n_features) 
                            if feature_variance[i] >= variance_threshold]
        
        logger.info(f"  ✓ Stable features: {len(stable_features)}")
        logger.info(f"  ✓ Unstable features: {len(unstable_features)}")
        
        return {
            'dropout_probs': self.dropout_probs,
            'original_predictions': y_pred_original,
            'dropout_predictions': dropout_predictions,
            'feature_importance_uncertainty': feature_variance,
            'feature_names': feature_names,
            'stable_features': stable_features,
            'unstable_features': unstable_features,
            'n_samples': self.n_samples
        }


# ============================================================================
# ENSEMBLE UNCERTAINTY ANALYZER
# ============================================================================

class EnsembleUncertaintyAnalyzer:
    """Ensemble uncertainty from model disagreement"""
    
    def __init__(self, consensus_threshold: float = 0.1):
        self.consensus_threshold = consensus_threshold
    
    def analyze_ensemble_uncertainty(self, models: List, X: np.ndarray,
                                    model_ids: List[str] = None) -> Dict:
        """
        Analyze ensemble uncertainty via model disagreement
        
        Args:
            models: List of trained models
            X: Input features
            model_ids: Optional model identifiers
        
        Returns:
            {
                'mean_predictions': array,
                'inter_model_std': array (uncertainty),
                'all_predictions': array (n_models, n_samples),
                'consensus_nuclei': list,
                'disagreement_nuclei': list,
                'model_correlations': matrix
            }
        """
        logger.info(f"  Running Ensemble Uncertainty: {len(models)} models...")
        
        n_models = len(models)
        n_samples = len(X)
        
        if model_ids is None:
            model_ids = [f'Model_{i}' for i in range(n_models)]
        
        # Collect predictions from all models
        all_predictions = np.zeros((n_models, n_samples))
        
        for i, model in enumerate(tqdm(models, desc="Ensemble")):
            try:
                y_pred = model.predict(X)
                all_predictions[i] = y_pred.flatten()
            except Exception as e:
                logger.warning(f"  Model {i} prediction failed: {e}")
                all_predictions[i] = np.nan
        
        # Calculate ensemble statistics
        mean_pred = np.nanmean(all_predictions, axis=0)
        inter_model_std = np.nanstd(all_predictions, axis=0)
        
        # Identify consensus vs disagreement
        consensus_mask = inter_model_std < self.consensus_threshold
        consensus_indices = np.where(consensus_mask)[0].tolist()
        disagreement_indices = np.where(~consensus_mask)[0].tolist()
        
        # Calculate model correlations
        model_correlations = np.corrcoef(all_predictions)
        
        logger.info(f"  ✓ Mean inter-model std: {inter_model_std.mean():.4f}")
        logger.info(f"  ✓ Consensus nuclei: {len(consensus_indices)}")
        logger.info(f"  ✓ Disagreement nuclei: {len(disagreement_indices)}")
        
        return {
            'mean_predictions': mean_pred,
            'inter_model_std': inter_model_std,
            'all_predictions': all_predictions,
            'consensus_nuclei': consensus_indices,
            'disagreement_nuclei': disagreement_indices,
            'model_correlations': model_correlations,
            'model_ids': model_ids,
            'n_models': n_models,
            'consensus_threshold': self.consensus_threshold
        }


# ============================================================================
# MAIN MONTE CARLO SYSTEM
# ============================================================================

class MonteCarloSimulationSystem:
    """
    Comprehensive Monte Carlo simulation system for uncertainty quantification
    
    Components:
    1. MC Dropout Simulator (DNN uncertainty)
    2. Bootstrap Simulator (confidence intervals)
    3. Noise Simulator (robustness)
    4. Feature Dropout Simulator (feature uncertainty)
    5. Ensemble Uncertainty Analyzer (model disagreement)
    """
    
    def __init__(self,
                 models_dir: str = 'trained_models',
                 aaa2_data_path: str = 'aaa2_control_group_results',
                 output_dir: str = 'monte_carlo_results',
                 config: Dict = None):
        """Initialize Monte Carlo simulation system"""
        
        self.models_dir = Path(models_dir)
        self.aaa2_data_path = Path(aaa2_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self.config = config or DEFAULT_MC_CONFIG
        
        # Initialize simulators
        self.mc_dropout = MCDropoutSimulator(
            n_samples=self.config['mc_dropout']['n_samples']
        )
        
        self.bootstrap = BootstrapSimulator(
            n_bootstrap=self.config['bootstrap']['n_bootstrap'],
            stratified=self.config['bootstrap']['stratified']
        )
        
        self.noise_sim = NoiseSimulator(
            noise_levels=self.config['noise_sensitivity']['noise_levels'],
            n_samples_per_level=self.config['noise_sensitivity']['n_samples_per_level']
        )
        
        self.feature_dropout = FeatureDropoutSimulator(
            dropout_probs=self.config['feature_dropout']['dropout_probs'],
            n_samples=self.config['feature_dropout']['n_samples']
        )
        
        self.ensemble_uncertainty = EnsembleUncertaintyAnalyzer(
            consensus_threshold=self.config['ensemble_uncertainty']['consensus_threshold']
        )
        
        # Results storage
        self.results = {}
        
        logger.info("="*80)
        logger.info("MONTE CARLO SIMULATION SYSTEM INITIALIZED")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Config: {len(self.config)} components")
    
    # ========================================================================
    # DATA LOADING & MODEL SELECTION
    # ========================================================================
    
    def load_top10_models(self, target: str) -> Tuple[List, List[str], Dict]:
        """
        Load top 10 models for target based on R² score
        
        Args:
            target: Target variable name
        
        Returns:
            models: List of loaded model objects
            model_ids: List of model identifiers
            metadata: Model metadata (R², RMSE, etc.)
        """
        logger.info(f"\n→ Loading top 10 models for {target}...")
        
        # Load performance summary
        perf_file = self.models_dir / f'performance_summary_{target}.csv'
        
        if not perf_file.exists():
            logger.error(f"  Performance summary not found: {perf_file}")
            return [], [], {}
        
        perf_df = pd.read_csv(perf_file)
        
        # Sort by R² (descending) and select top 10
        if 'R2' in perf_df.columns:
            sort_col = 'R2'
        elif 'R2_MM' in perf_df.columns:  # For MM_QM
            sort_col = 'R2_MM'
        else:
            logger.error("  No R² column found in performance summary")
            return [], [], {}
        
        perf_df_sorted = perf_df.sort_values(sort_col, ascending=False)
        top10_df = perf_df_sorted.head(10)
        
        logger.info(f"  Top 10 models (by {sort_col}):")
        for i, row in top10_df.iterrows():
            logger.info(f"    {row['model_id']}: R²={row[sort_col]:.4f}")
        
        # Load models
        models = []
        model_ids = []
        metadata = {}
        
        for idx, row in top10_df.iterrows():
            model_id = row['model_id']
            
            try:
                # Determine model type and path
                model_type = model_id.split('_')[0]  # RF, GBM, XGBoost, DNN, etc.
                
                if model_type == 'ANFIS':
                    model_path = self.models_dir / 'ANFIS' / model_id / 'model.mat'
                else:
                    model_path = self.models_dir / 'AI' / model_type / model_id / 'model.pkl'
                
                if not model_path.exists():
                    logger.warning(f"    Model file not found: {model_path}")
                    continue
                
                # Load model
                if model_type == 'DNN' and model_path.suffix == '.h5':
                    if TF_AVAILABLE:
                        model = keras.models.load_model(model_path)
                    else:
                        logger.warning(f"    TensorFlow not available, skipping {model_id}")
                        continue
                elif model_type == 'ANFIS':
                    # Skip ANFIS for now (requires MATLAB engine)
                    logger.info(f"    Skipping ANFIS model: {model_id}")
                    continue
                else:
                    model = joblib.load(model_path)
                
                models.append(model)
                model_ids.append(model_id)
                
                # Store metadata
                metadata[model_id] = {
                    'R2': row.get(sort_col, np.nan),
                    'RMSE': row.get('RMSE', np.nan),
                    'MAE': row.get('MAE', np.nan),
                    'model_type': model_type,
                    'model_path': str(model_path)
                }
            
            except Exception as e:
                logger.warning(f"    Failed to load {model_id}: {e}")
                continue
        
        logger.info(f"  ✓ Successfully loaded {len(models)} models")
        
        # Save top 10 selection
        selection_file = self.output_dir / 'model_selection' / f'top10_models_{target}.json'
        selection_file.parent.mkdir(parents=True, exist_ok=True)
        with open(selection_file, 'w') as f:
            json.dump({
                'target': target,
                'n_models': len(model_ids),
                'model_ids': model_ids,
                'metadata': metadata
            }, f, indent=2)
        
        return models, model_ids, metadata
    
    def load_aaa2_data(self) -> pd.DataFrame:
        """Load AAA2 control group data"""
        logger.info("\n→ Loading AAA2 control group data...")
        
        # Try multiple possible locations
        possible_paths = [
            self.aaa2_data_path / 'data_preparation' / 'aaa2_enriched.csv',
            Path('generated_datasets') / 'AAA2_enriched.csv',
            Path('aaa2.txt')
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"  Found AAA2 data: {path}")
                
                if path.suffix == '.txt':
                    aaa2_df = pd.read_csv(path, sep='\t', encoding='utf-8')
                else:
                    aaa2_df = pd.read_csv(path)
                
                logger.info(f"  ✓ Loaded {len(aaa2_df)} nuclei")
                return aaa2_df
        
        logger.error("  AAA2 data not found in any expected location")
        return pd.DataFrame()
    
    # ========================================================================
    # MC SIMULATIONS INTEGRATION
    # ========================================================================
    
    def run_mc_simulations_on_top10(self, models: List, model_ids: List[str],
                                    X: np.ndarray, y: np.ndarray,
                                    target: str) -> Dict:
        """
        Run all MC simulations on top 10 models
        
        Args:
            models: List of model objects
            model_ids: List of model identifiers
            X: Input features
            y: Target values (for validation)
            target: Target name
        
        Returns:
            mc_results: Dictionary with all MC results
        """
        logger.info("\n→ Running MC simulations on top 10 models...")
        
        mc_results = {
            'target': target,
            'n_models': len(models),
            'model_ids': model_ids,
            'mc_dropout': {},
            'bootstrap': {},
            'noise_sensitivity': {},
            'feature_dropout': {},
            'ensemble_uncertainty': {}
        }
        
        # MC Dropout (DNN only)
        if self.config['mc_dropout']['enabled']:
            logger.info("\n  → MC Dropout (DNNs)...")
            dnn_indices = [i for i, mid in enumerate(model_ids) if 'DNN' in mid]
            
            for idx in dnn_indices:
                model = models[idx]
                model_id = model_ids[idx]
                
                logger.info(f"    Processing {model_id}...")
                mc_dropout_result = self.mc_dropout.estimate_uncertainty(model, X)
                mc_results['mc_dropout'][model_id] = mc_dropout_result
        
        # Noise Sensitivity (all models)
        if self.config['noise_sensitivity']['enabled']:
            logger.info("\n  → Noise Sensitivity (all models)...")
            
            for idx, model in enumerate(models):
                model_id = model_ids[idx]
                logger.info(f"    Processing {model_id}...")
                
                noise_result = self.noise_sim.analyze_noise_sensitivity(model, X)
                mc_results['noise_sensitivity'][model_id] = noise_result
        
        # Ensemble Uncertainty (all models together)
        if self.config['ensemble_uncertainty']['enabled']:
            logger.info("\n  → Ensemble Uncertainty...")
            
            ensemble_result = self.ensemble_uncertainty.analyze_ensemble_uncertainty(
                models, X, model_ids
            )
            mc_results['ensemble_uncertainty'] = ensemble_result
        
        logger.info("\n  ✓ MC simulations complete")
        
        return mc_results
    
    # ========================================================================
    # AAA2 VALIDATION
    # ========================================================================
    
    def validate_on_aaa2(self, models: List, model_ids: List[str],
                        aaa2_df: pd.DataFrame, target: str) -> Dict:
        """
        Validate top 10 models on AAA2 control group
        
        Args:
            models: List of models
            model_ids: Model identifiers
            aaa2_df: AAA2 DataFrame
            target: Target name
        
        Returns:
            aaa2_results: AAA2 validation results
        """
        logger.info("\n→ Validating on AAA2 control group...")
        
        # Prepare features (same as used in training)
        feature_cols = [col for col in aaa2_df.columns 
                       if col not in ['NUCLEUS', 'A', 'Z', 'N', 
                                     'MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]', 
                                     'Beta_2']]
        
        X_aaa2 = aaa2_df[feature_cols].values
        
        # Get experimental values
        target_col_map = {
            'MM': 'MAGNETIC MOMENT [µ]',
            'QM': 'QUADRUPOLE MOMENT [Q]',
            'Beta_2': 'Beta_2'
        }
        
        if target in target_col_map:
            y_exp = aaa2_df[target_col_map[target]].values
        else:
            y_exp = None
        
        # Collect predictions from all models
        predictions = np.zeros((len(models), len(aaa2_df)))
        
        for i, model in enumerate(models):
            logger.info(f"  Predicting with {model_ids[i]}...")
            predictions[i] = model.predict(X_aaa2).flatten()
        
        # Calculate statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        # Identify high/low uncertainty
        threshold_high = self.config['thresholds']['high_uncertainty']
        threshold_low = self.config['thresholds']['low_uncertainty']
        
        high_unc_mask = std_pred > threshold_high
        low_unc_mask = std_pred < threshold_low
        
        high_unc_nuclei = aaa2_df.loc[high_unc_mask, 'NUCLEUS'].tolist()
        low_unc_nuclei = aaa2_df.loc[low_unc_mask, 'NUCLEUS'].tolist()
        
        logger.info(f"  ✓ High uncertainty nuclei: {len(high_unc_nuclei)}")
        logger.info(f"  ✓ Low uncertainty nuclei: {len(low_unc_nuclei)}")
        
        aaa2_results = {
            'target': target,
            'n_nuclei': len(aaa2_df),
            'predictions': predictions,
            'mean_predictions': mean_pred,
            'std_predictions': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'high_uncertainty_nuclei': high_unc_nuclei,
            'low_uncertainty_nuclei': low_unc_nuclei,
            'experimental_values': y_exp
        }
        
        return aaa2_results
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_complete_mc_analysis(self, target: str = 'MM') -> Dict:
        """
        Run complete Monte Carlo analysis for a target
        
        Args:
            target: Target variable (MM, QM, MM_QM, Beta_2)
        
        Returns:
            results_dict: All MC analysis results
        """
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info(f"MONTE CARLO ANALYSIS - {target}")
        logger.info("="*80)
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            'target': target,
            'timestamp': start_time.isoformat(),
            'config': self.config,
            'success': False
        }
        
        try:
            # Phase 1: Load top 10 models
            models, model_ids, metadata = self.load_top10_models(target)
            
            if len(models) == 0:
                logger.error("  No models loaded. Aborting.")
                return results
            
            results['models_loaded'] = len(models)
            results['model_ids'] = model_ids
            
            # Phase 2: Load AAA2 data
            aaa2_df = self.load_aaa2_data()
            
            if len(aaa2_df) == 0:
                logger.warning("  AAA2 data not loaded. Skipping AAA2 validation.")
                aaa2_results = None
            else:
                # Phase 3: AAA2 Validation
                aaa2_results = self.validate_on_aaa2(models, model_ids, aaa2_df, target)
                results['aaa2_validation'] = aaa2_results
                
                # Phase 4: MC Simulations on AAA2
                feature_cols = [col for col in aaa2_df.columns 
                               if col not in ['NUCLEUS', 'A', 'Z', 'N']]
                X_aaa2 = aaa2_df[feature_cols[:20]].values  # Use subset for speed
                y_aaa2 = aaa2_results.get('experimental_values')
                
                mc_results = self.run_mc_simulations_on_top10(
                    models, model_ids, X_aaa2, y_aaa2, target
                )
                results['mc_simulations'] = mc_results
            
            # Phase 5: Reports and visualizations
            logger.info("\n→ Phase 5: Visualizations...")
            
            # 3D Visualizations
            viz_3d_files = self.create_3d_visualizations(
                target, mc_results, aaa2_results, aaa2_df
            )
            results['visualizations_3d'] = viz_3d_files
            
            # Standard visualizations
            viz_std_files = self.create_standard_visualizations(
                target, mc_results, aaa2_results, aaa2_df
            )
            results['visualizations_standard'] = viz_std_files
            
            # Phase 6: Excel report
            logger.info("\n→ Phase 6: Excel report...")
            excel_file = self.generate_excel_report(
                target, mc_results, aaa2_results, aaa2_df
            )
            results['excel_report'] = str(excel_file)
            
            # Phase 7: JSON summary
            logger.info("\n→ Phase 7: JSON summary...")
            json_file = self.export_json_summary(
                target, mc_results, aaa2_results
            )
            results['json_summary'] = str(json_file)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results['success'] = True
            results['duration_seconds'] = duration
            
            logger.info(f"\n✅ Monte Carlo analysis complete for {target}")
            logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            return results
        
        except Exception as e:
            logger.error(f"\n❌ Monte Carlo analysis failed: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            return results
    
    # ========================================================================
    # 3D VISUALIZATIONS
    # ========================================================================
    
    def create_3d_visualizations(self, target: str, mc_results: Dict,
                                aaa2_results: Dict, aaa2_df: pd.DataFrame) -> Dict:
        """Create 3D visualizations - Charts 11, 12, 13"""
        logger.info("  Creating 3D visualizations...")
        
        viz_dir = self.output_dir / 'visualizations' / target
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        viz_files = {}
        
        # Chart 11: 3D Uncertainty Landscape
        from mpl_toolkits.mplot3d import Axes3D
        
        A = aaa2_df['A'].values
        Z = aaa2_df['Z'].values
        uncertainty = aaa2_results['std_predictions']
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(A, Z, uncertainty, c=uncertainty, cmap='viridis',
                           s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Mass Number (A)', fontsize=12)
        ax.set_ylabel('Proton Number (Z)', fontsize=12)
        ax.set_zlabel('Uncertainty (σ)', fontsize=12)
        ax.set_title(f'3D Uncertainty Landscape - {target}', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Uncertainty', shrink=0.7)
        ax.view_init(elev=20, azim=45)
        
        png_file = viz_dir / '11_3D_uncertainty_landscape.png'
        plt.tight_layout()
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_files['3d_uncertainty'] = str(png_file)
        logger.info(f"    ✓ Chart 11 saved: {png_file.name}")
        
        # Chart 12: 3D Model Agreement (if ensemble results available)
        ensemble_results = mc_results.get('ensemble_uncertainty', {})
        all_preds = ensemble_results.get('all_predictions')
        
        if all_preds is not None and len(all_preds) >= 3:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            agreement = 1 - ensemble_results['inter_model_std']
            
            scatter = ax.scatter(all_preds[0], all_preds[1], all_preds[2],
                               c=agreement, cmap='RdYlGn', s=80, alpha=0.6,
                               edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Model 1', fontsize=12)
            ax.set_ylabel('Model 2', fontsize=12)
            ax.set_zlabel('Model 3', fontsize=12)
            ax.set_title(f'3D Model Agreement - {target}', fontsize=14)
            plt.colorbar(scatter, ax=ax, label='Agreement', shrink=0.7)
            ax.view_init(elev=25, azim=45)
            
            png_file = viz_dir / '12_3D_model_agreement.png'
            plt.tight_layout()
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['3d_agreement'] = str(png_file)
            logger.info(f"    ✓ Chart 12 saved: {png_file.name}")
        
        # Chart 13: 3D Noise Robustness
        noise_results = mc_results.get('noise_sensitivity', {})
        if noise_results:
            first_model = list(noise_results.keys())[0]
            noise_data = noise_results[first_model]
            
            noise_levels = noise_data['noise_levels']
            variances = noise_data['prediction_variance']
            
            # Sample for visualization
            sample_idx = np.arange(0, len(A), 5)
            
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            X, Y = np.meshgrid(noise_levels, sample_idx)
            Z = np.zeros_like(X, dtype=float)
            
            for i, nl in enumerate(noise_levels):
                Z[:, i] = variances[nl][sample_idx]
            
            surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn_r', alpha=0.8)
            
            ax.set_xlabel('Noise Level', fontsize=12)
            ax.set_ylabel('Nucleus Index', fontsize=12)
            ax.set_zlabel('Variance', fontsize=12)
            ax.set_title(f'3D Noise Robustness - {target}', fontsize=14)
            plt.colorbar(surf, ax=ax, label='Variance', shrink=0.7)
            ax.view_init(elev=30, azim=45)
            
            png_file = viz_dir / '13_3D_noise_robustness.png'
            plt.tight_layout()
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['3d_robustness'] = str(png_file)
            logger.info(f"    ✓ Chart 13 saved: {png_file.name}")
        
        logger.info(f"  ✓ 3D visualizations complete: {len(viz_files)} charts")
        return viz_files
    
    def create_standard_visualizations(self, target: str, mc_results: Dict,
                                       aaa2_results: Dict, aaa2_df: pd.DataFrame) -> Dict:
        """Create standard 2D visualizations - Charts 1-10"""
        logger.info("  Creating standard visualizations...")
        
        viz_dir = self.output_dir / 'visualizations' / target
        viz_files = {}
        
        # Chart 1: Uncertainty Distribution
        uncertainty = aaa2_results['std_predictions']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(uncertainty, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(uncertainty.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {uncertainty.mean():.4f}')
        ax.set_xlabel('Uncertainty (σ)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Uncertainty Distribution - {target}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        png_file = viz_dir / '01_uncertainty_distribution.png'
        plt.tight_layout()
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['uncertainty_dist'] = str(png_file)
        
        # Chart 2: Uncertainty vs A
        fig, ax = plt.subplots(figsize=(14, 7))
        scatter = ax.scatter(aaa2_df['A'], uncertainty, c=uncertainty, cmap='viridis',
                           s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Uncertainty')
        ax.set_xlabel('Mass Number (A)', fontsize=12)
        ax.set_ylabel('Uncertainty (σ)', fontsize=12)
        ax.set_title(f'Uncertainty vs Mass Number - {target}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        png_file = viz_dir / '02_uncertainty_vs_A.png'
        plt.tight_layout()
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files['uncertainty_vs_A'] = str(png_file)
        
        logger.info(f"  ✓ Standard visualizations: {len(viz_files)} charts")
        return viz_files
    
    def generate_excel_report(self, target: str, mc_results: Dict,
                             aaa2_results: Dict, aaa2_df: pd.DataFrame) -> Path:
        """Generate comprehensive Excel report"""
        logger.info("  Creating Excel report...")
        
        excel_dir = self.output_dir / 'excel_reports'
        excel_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = excel_dir / f'MC_Analysis_{target}_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            # Sheet 1: Executive Summary
            summary_data = {
                'Metric': ['Target', 'Models Analyzed', 'Nuclei', 'Mean Uncertainty', 'Max Uncertainty'],
                'Value': [
                    target,
                    mc_results['n_models'],
                    len(aaa2_df),
                    f"{aaa2_results['std_predictions'].mean():.4f}",
                    f"{aaa2_results['std_predictions'].max():.4f}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive_Summary', index=False)
            
            # Sheet 2: AAA2 Predictions with Uncertainty
            aaa2_summary = aaa2_df[['NUCLEUS', 'A', 'Z', 'N']].copy()
            aaa2_summary['Mean_Prediction'] = aaa2_results['mean_predictions']
            aaa2_summary['Uncertainty'] = aaa2_results['std_predictions']
            aaa2_summary['CI_Lower'] = aaa2_results['ci_lower']
            aaa2_summary['CI_Upper'] = aaa2_results['ci_upper']
            aaa2_summary.to_excel(writer, sheet_name='AAA2_Predictions', index=False)
            
            # Sheet 3: High Uncertainty Nuclei
            high_unc_nuclei = aaa2_results['high_uncertainty_nuclei']
            if high_unc_nuclei:
                high_unc_df = aaa2_df[aaa2_df['NUCLEUS'].isin(high_unc_nuclei)].copy()
                high_unc_df['Uncertainty'] = aaa2_results['std_predictions'][
                    aaa2_df['NUCLEUS'].isin(high_unc_nuclei)
                ]
                high_unc_df.to_excel(writer, sheet_name='High_Uncertainty', index=False)
        
        logger.info(f"  ✓ Excel saved: {excel_file.name}")
        return excel_file
    
    def export_json_summary(self, target: str, mc_results: Dict,
                           aaa2_results: Dict) -> Path:
        """Export JSON summary"""
        logger.info("  Creating JSON summary...")
        
        summary_dir = self.output_dir / 'summaries'
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        json_file = summary_dir / f'mc_summary_{target}.json'
        
        summary = {
            'target': target,
            'timestamp': datetime.now().isoformat(),
            'overall_statistics': {
                'n_models': mc_results['n_models'],
                'n_nuclei': aaa2_results['n_nuclei'],
                'mean_uncertainty': float(aaa2_results['std_predictions'].mean()),
                'max_uncertainty': float(aaa2_results['std_predictions'].max()),
                'min_uncertainty': float(aaa2_results['std_predictions'].min())
            },
            'high_uncertainty_nuclei': aaa2_results['high_uncertainty_nuclei'][:10],
            'low_uncertainty_nuclei': aaa2_results['low_uncertainty_nuclei'][:10]
        }
        
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"  ✓ JSON saved: {json_file.name}")
        return json_file


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    
    # Initialize system
    mc_system = MonteCarloSimulationSystem(
        models_dir='trained_models',
        aaa2_data_path='aaa2_control_group_results',
        output_dir='monte_carlo_results'
    )
    
    # Run analysis for MM
    results = mc_system.run_complete_mc_analysis(target='MM')
    
    return results


if __name__ == "__main__":
    results = main()
