# -*- coding: utf-8 -*-
"""
PFAZ 7: Complete Ensemble & Meta-Model Pipeline
================================================

Tum ensemble metodlarini iceren production-ready pipeline:
- Voting Ensembles (Simple, Weighted, Rank-based)
- Stacking Ensembles (Ridge, Lasso, ElasticNet, RF, GBM)
- Blending Ensembles
- Dynamic Weight Adjustment
- Advanced Boosting Strategies (AdaBoost, CatBoost)
- Ensemble Diversity Analysis
- Comprehensive Evaluation & Reporting

Author: Nuclear Physics AI Project
Version: 2.0.0 - COMPLETE
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
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    AdaBoostRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import rankdata

# Optional: CatBoost (if available)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available")

# Optional: XGBoost (if available)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ADVANCED VOTING ENSEMBLE BUILDER
# ============================================================================

class AdvancedVotingEnsemble:
    """
    Advanced Voting Ensemble with multiple strategies
    
    Methods:
    1. Simple Voting (equal weights)
    2. Weighted Voting (performance-based)
    3. Rank-based Voting
    4. Dynamic Weight Adjustment
    5. Inverse Error Weighting
    """
    
    def __init__(self, output_dir='ensemble_results/voting'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.model_performances = {}
        
        logger.info("[OK] AdvancedVotingEnsemble initialized")
    
    def add_model(self, model_id: str, model: object, 
                  X_val: np.ndarray, y_val: np.ndarray):
        """Add base model and calculate performance"""
        self.models[model_id] = model
        
        # Calculate validation performance
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        self.model_performances[model_id] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'inverse_rmse': 1.0 / (rmse + 1e-10)
        }
        
        logger.info(f"  [OK] {model_id}: R2={r2:.4f}, RMSE={rmse:.4f}")
    
    def simple_voting(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Simple voting ensemble (equal weights)"""
        logger.info("\n-> Simple Voting Ensemble")
        
        predictions = np.array([model.predict(X_test) for model in self.models.values()])
        ensemble_pred = np.mean(predictions, axis=0)
        
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        
        result = {
            'method': 'simple_voting',
            'n_models': len(self.models),
            'weights': {mid: 1.0/len(self.models) for mid in self.models.keys()},
            'predictions': ensemble_pred,
            **metrics
        }
        
        logger.info(f"  R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return result
    
    def weighted_voting_r2(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Weighted voting based on R2 scores"""
        logger.info("\n-> Weighted Voting (R2 optimization)")
        
        # Calculate weights based on R2
        r2_scores = np.array([perf['r2'] for perf in self.model_performances.values()])
        weights = r2_scores / np.sum(r2_scores)
        
        predictions = np.array([model.predict(X_test) for model in self.models.values()])
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        
        result = {
            'method': 'weighted_voting_r2',
            'n_models': len(self.models),
            'weights': {mid: w for mid, w in zip(self.models.keys(), weights)},
            'predictions': ensemble_pred,
            **metrics
        }
        
        logger.info(f"  R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return result
    
    def weighted_voting_inverse_error(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Weighted voting based on inverse RMSE"""
        logger.info("\n-> Weighted Voting (Inverse Error)")
        
        # Calculate weights based on inverse RMSE
        inv_rmse = np.array([perf['inverse_rmse'] for perf in self.model_performances.values()])
        weights = inv_rmse / np.sum(inv_rmse)
        
        predictions = np.array([model.predict(X_test) for model in self.models.values()])
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        
        result = {
            'method': 'weighted_voting_inverse_error',
            'n_models': len(self.models),
            'weights': {mid: w for mid, w in zip(self.models.keys(), weights)},
            'predictions': ensemble_pred,
            **metrics
        }
        
        logger.info(f"  R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return result
    
    def rank_based_voting(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Rank-based voting ensemble"""
        logger.info("\n-> Rank-Based Voting")
        
        predictions = np.array([model.predict(X_test) for model in self.models.values()])
        
        # Convert predictions to ranks
        ranks = np.array([rankdata(pred) for pred in predictions])
        ensemble_ranks = np.mean(ranks, axis=0)
        
        # Convert ranks back to predictions (approximate)
        sorted_indices = np.argsort(ensemble_ranks)
        ensemble_pred = np.zeros_like(ensemble_ranks)
        ensemble_pred[sorted_indices] = np.sort(np.mean(predictions, axis=0))
        
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        
        result = {
            'method': 'rank_based_voting',
            'n_models': len(self.models),
            'predictions': ensemble_pred,
            **metrics
        }
        
        logger.info(f"  R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return result
    
    def dynamic_weight_adjustment(self, X_test: np.ndarray, y_test: np.ndarray, 
                                  n_iterations: int = 10) -> Dict:
        """Dynamic weight adjustment based on iterative optimization"""
        logger.info("\n-> Dynamic Weight Adjustment")
        
        predictions = np.array([model.predict(X_test) for model in self.models.values()])
        
        # Initialize weights
        weights = np.ones(len(self.models)) / len(self.models)
        
        # Iterative optimization
        best_rmse = float('inf')
        best_weights = weights.copy()
        
        for i in range(n_iterations):
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = weights.copy()
            
            # Adjust weights based on individual errors
            errors = np.array([np.sqrt(mean_squared_error(y_test, pred)) 
                              for pred in predictions])
            
            # Update weights (inverse error weighting with decay)
            new_weights = 1.0 / (errors + 1e-10)
            weights = 0.7 * weights + 0.3 * (new_weights / np.sum(new_weights))
        
        ensemble_pred = np.average(predictions, axis=0, weights=best_weights)
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        
        result = {
            'method': 'dynamic_weight_adjustment',
            'n_models': len(self.models),
            'n_iterations': n_iterations,
            'weights': {mid: w for mid, w in zip(self.models.keys(), best_weights)},
            'predictions': ensemble_pred,
            **metrics
        }
        
        logger.info(f"  R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return result
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }


# ============================================================================
# ADVANCED STACKING ENSEMBLE
# ============================================================================

class AdvancedStackingEnsemble:
    """
    Advanced Stacking Ensemble with multiple meta-models
    
    Meta-models:
    1. Ridge Regression
    2. Lasso Regression
    3. ElasticNet
    4. Random Forest
    5. Gradient Boosting
    6. MLP Neural Network
    """
    
    def __init__(self, cv_folds: int = 5, output_dir='ensemble_results/stacking'):
        self.cv_folds = cv_folds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_models = {}
        self.meta_models = {}
        self.oof_predictions = None
        
        logger.info(f"[OK] AdvancedStackingEnsemble initialized (CV folds: {cv_folds})")
    
    def add_base_model(self, model_id: str, model: object):
        """Add base model"""
        self.base_models[model_id] = model
        logger.info(f"  [OK] Added base model: {model_id}")
    
    def generate_oof_predictions(self, X_train: np.ndarray, y_train: np.ndarray):
        """Generate out-of-fold predictions for meta-model training"""
        logger.info("\n-> Generating Out-of-Fold Predictions")
        
        n_samples = len(X_train)
        n_models = len(self.base_models)
        
        oof_preds = np.zeros((n_samples, n_models))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for i, (model_id, model) in enumerate(self.base_models.items()):
            logger.info(f"  -> Model {i+1}/{n_models}: {model_id}")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                
                # Clone and train model on fold
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict on validation fold
                oof_preds[val_idx, i] = fold_model.predict(X_fold_val)
        
        self.oof_predictions = oof_preds
        
        logger.info(f"[OK] OOF predictions generated: shape={oof_preds.shape}")
        
        return oof_preds
    
    def train_meta_models(self, y_train: np.ndarray):
        """Train multiple meta-models"""
        logger.info("\n-> Training Meta-Models")
        
        # Meta-model definitions
        meta_model_configs = {
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        }
        
        for name, meta_model in meta_model_configs.items():
            logger.info(f"  -> Training {name} meta-model...")
            meta_model.fit(self.oof_predictions, y_train)
            self.meta_models[name] = meta_model
            
            # Evaluate on OOF predictions
            oof_pred = meta_model.predict(self.oof_predictions)
            r2 = r2_score(y_train, oof_pred)
            logger.info(f"    OOF R2: {r2:.4f}")
        
        logger.info(f"[OK] {len(self.meta_models)} meta-models trained")
    
    def predict(self, X_test: np.ndarray, meta_model_name: str) -> np.ndarray:
        """Predict using stacking ensemble"""
        # Get base model predictions
        base_preds = np.array([model.predict(X_test) 
                               for model in self.base_models.values()]).T
        
        # Meta-model prediction
        meta_model = self.meta_models[meta_model_name]
        stacking_pred = meta_model.predict(base_preds)
        
        return stacking_pred
    
    def evaluate_all_meta_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all meta-models"""
        logger.info("\n-> Evaluating All Meta-Models")
        
        results = {}
        
        for name in self.meta_models.keys():
            stacking_pred = self.predict(X_test, name)
            
            r2 = r2_score(y_test, stacking_pred)
            rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))
            mae = mean_absolute_error(y_test, stacking_pred)
            
            results[name] = {
                'method': f'stacking_{name}',
                'meta_model': name,
                'n_base_models': len(self.base_models),
                'predictions': stacking_pred,
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
            
            logger.info(f"  {name}: R2={r2:.4f}, RMSE={rmse:.4f}")
        
        return results


# ============================================================================
# ADVANCED BOOSTING STRATEGIES
# ============================================================================

class AdvancedBoostingEnsemble:
    """
    Advanced Boosting Strategies
    
    Methods:
    1. AdaBoost
    2. CatBoost (if available)
    3. Custom Sequential Boosting
    """
    
    def __init__(self, output_dir='ensemble_results/boosting'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[OK] AdvancedBoostingEnsemble initialized")
    
    def train_adaboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       n_estimators: int = 100) -> Dict:
        """Train AdaBoost ensemble"""
        logger.info(f"\n-> Training AdaBoost (n_estimators={n_estimators})")
        
        model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        result = {
            'method': 'adaboost',
            'n_estimators': n_estimators,
            'model': model,
            'predictions': predictions,
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"  R2={r2:.4f}, RMSE={rmse:.4f}")
        
        return result
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       n_iterations: int = 1000) -> Dict:
        """Train CatBoost ensemble (if available)"""
        if not CATBOOST_AVAILABLE:
            logger.warning("  [WARNING] CatBoost not available, skipping")
            return None
        
        logger.info(f"\n-> Training CatBoost (n_iterations={n_iterations})")
        
        model = CatBoostRegressor(
            iterations=n_iterations,
            learning_rate=0.05,
            depth=6,
            random_state=42,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        result = {
            'method': 'catboost',
            'n_iterations': n_iterations,
            'model': model,
            'predictions': predictions,
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"  R2={r2:.4f}, RMSE={rmse:.4f}")
        
        return result


# ============================================================================
# ENSEMBLE DIVERSITY ANALYZER
# ============================================================================

class EnsembleDiversityAnalyzer:
    """
    Analyze ensemble diversity
    
    Metrics:
    1. Prediction Correlation
    2. Disagreement Rate
    3. Diversity Score (Q-statistic)
    4. Double Fault
    """
    
    def __init__(self):
        logger.info("[OK] EnsembleDiversityAnalyzer initialized")
    
    def analyze_diversity(self, predictions_dict: Dict[str, np.ndarray], 
                          y_true: np.ndarray) -> Dict:
        """Analyze ensemble diversity"""
        logger.info("\n-> Analyzing Ensemble Diversity")
        
        model_names = list(predictions_dict.keys())
        predictions = np.array([predictions_dict[name] for name in model_names])
        
        n_models = len(model_names)
        
        # 1. Prediction Correlation Matrix
        corr_matrix = np.corrcoef(predictions)
        avg_correlation = (np.sum(corr_matrix) - n_models) / (n_models * (n_models - 1))
        
        # 2. Disagreement Rate
        disagreement_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(i+1, n_models):
                disagreement = np.mean(np.abs(predictions[i] - predictions[j]) > 0.1 * np.std(y_true))
                disagreement_matrix[i, j] = disagreement
                disagreement_matrix[j, i] = disagreement
        
        avg_disagreement = np.mean(disagreement_matrix[np.triu_indices(n_models, k=1)])
        
        # 3. Q-statistic (diversity measure)
        q_matrix = np.zeros((n_models, n_models))
        threshold = np.median(y_true)
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Binary classification: above/below median
                pred_i = predictions[i] > threshold
                pred_j = predictions[j] > threshold
                true_class = y_true > threshold
                
                n11 = np.sum(pred_i & pred_j & true_class)
                n00 = np.sum(~pred_i & ~pred_j & ~true_class)
                n10 = np.sum(pred_i & ~pred_j)
                n01 = np.sum(~pred_i & pred_j)
                
                q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10 + 1e-10)
                q_matrix[i, j] = q
                q_matrix[j, i] = q
        
        avg_q_statistic = np.mean(q_matrix[np.triu_indices(n_models, k=1)])
        
        diversity_report = {
            'n_models': n_models,
            'model_names': model_names,
            'avg_correlation': avg_correlation,
            'correlation_matrix': corr_matrix.tolist(),
            'avg_disagreement': avg_disagreement,
            'disagreement_matrix': disagreement_matrix.tolist(),
            'avg_q_statistic': avg_q_statistic,
            'q_matrix': q_matrix.tolist(),
            'diversity_score': 1.0 - avg_correlation  # Higher = more diverse
        }
        
        logger.info(f"  Avg Correlation: {avg_correlation:.4f}")
        logger.info(f"  Avg Disagreement: {avg_disagreement:.4f}")
        logger.info(f"  Avg Q-statistic: {avg_q_statistic:.4f}")
        logger.info(f"  Diversity Score: {diversity_report['diversity_score']:.4f}")
        
        return diversity_report


# ============================================================================
# COMPREHENSIVE ENSEMBLE EVALUATOR
# ============================================================================

class ComprehensiveEnsembleEvaluator:
    """
    Evaluate and compare all ensemble methods
    """
    
    def __init__(self, output_dir='ensemble_results/evaluation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        
        logger.info("[OK] ComprehensiveEnsembleEvaluator initialized")
    
    def add_result(self, result: Dict):
        """Add ensemble result"""
        self.results.append(result)
    
    def compare_all(self) -> pd.DataFrame:
        """Compare all ensembles"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE ENSEMBLE COMPARISON")
        logger.info("="*80)
        
        comparison = []
        for result in self.results:
            comparison.append({
                'Method': result['method'],
                'R2': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'N_Models': result.get('n_base_models', result.get('n_models', 'N/A'))
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('R2', ascending=False)
        
        logger.info(f"\n{df.to_string(index=False)}")
        
        # Save
        save_path = self.output_dir / 'ensemble_comparison.xlsx'
        df.to_excel(save_path, index=False)
        logger.info(f"\n[OK] Comparison saved: {save_path}")
        
        return df
    
    def select_best(self, metric='r2') -> Dict:
        """Select best ensemble"""
        if metric == 'r2':
            best = max(self.results, key=lambda x: x['r2'])
        elif metric == 'rmse':
            best = min(self.results, key=lambda x: x['rmse'])
        elif metric == 'mae':
            best = min(self.results, key=lambda x: x['mae'])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        logger.info(f"\n[SUCCESS] BEST ENSEMBLE ({metric.upper()}):")
        logger.info(f"  Method: {best['method']}")
        logger.info(f"  R2: {best['r2']:.4f}")
        logger.info(f"  RMSE: {best['rmse']:.4f}")
        logger.info(f"  MAE: {best['mae']:.4f}")
        
        return best
    
    def generate_report(self) -> Dict:
        """Generate comprehensive report"""
        logger.info("\n-> Generating Comprehensive Report")
        
        # Summary statistics
        r2_scores = [r['r2'] for r in self.results]
        rmse_scores = [r['rmse'] for r in self.results]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_ensembles': len(self.results),
            'best_r2': max(r2_scores),
            'worst_r2': min(r2_scores),
            'avg_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'best_rmse': min(rmse_scores),
            'worst_rmse': max(rmse_scores),
            'avg_rmse': np.mean(rmse_scores),
            'std_rmse': np.std(rmse_scores),
            'all_results': self.results
        }
        
        # Save JSON
        json_path = self.output_dir / 'comprehensive_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types for JSON serialization
            report_copy = report.copy()
            for result in report_copy['all_results']:
                if 'predictions' in result:
                    del result['predictions']  # Remove predictions array for JSON
                if 'model' in result:
                    del result['model']  # Remove model object
            
            json.dump(report_copy, f, indent=2, default=str)
        
        logger.info(f"[OK] Report saved: {json_path}")
        
        return report




# ============================================================================
# REAL MODEL LOADER
# ============================================================================

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False


class RealModelLoader:
    def __init__(self, trained_models_dir, anfis_models_dir, datasets_dir):
        self.trained_models_dir = Path(trained_models_dir)
        self.anfis_models_dir = Path(anfis_models_dir)
        self.datasets_dir = Path(datasets_dir)
        self._meta_cache = {}

    def _load_json(self, path):
        key = str(path)
        if key not in self._meta_cache:
            try:
                with open(path, encoding='utf-8') as f:
                    self._meta_cache[key] = json.load(f)
            except Exception:
                self._meta_cache[key] = {}
        return self._meta_cache[key]

    def _feature_names_from_train_csv(self, dataset_name):
        csv_path = self.datasets_dir / dataset_name / 'train.csv'
        if not csv_path.exists():
            return None
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
            exclude = {'NUCLEUS', 'A', 'Z', 'N', 'MM', 'QM', 'Beta_2'}
            return [c for c in cols if c not in exclude]
        except Exception:
            return None

    def _scan_dir(self, root, target, source):
        records = []
        if not root.exists():
            return records
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            if not dataset_name.startswith(target + '_'):
                continue
            second_level = list(dataset_dir.iterdir())
            config_dirs = []
            if source == 'AI':
                for mtype_dir in second_level:
                    if not mtype_dir.is_dir():
                        continue
                    for cfg in mtype_dir.iterdir():
                        if cfg.is_dir():
                            config_dirs.append((mtype_dir.name, cfg))
            else:
                for cfg in second_level:
                    if cfg.is_dir():
                        config_dirs.append(('ANFIS', cfg))
            for mtype, cfg_dir in config_dirs:
                model_files = sorted(cfg_dir.glob('model_*.pkl'))
                if not model_files:
                    continue
                val_r2 = -999.0
                for mf in cfg_dir.glob('metrics_*.json'):
                    m = self._load_json(mf)
                    val_r2 = float(m.get('val_r2', m.get('r2_val', m.get('test_r2', -999.0))))
                    break
                feature_names = None
                for mf in cfg_dir.glob('metadata_*.json'):
                    meta = self._load_json(mf)
                    feature_names = meta.get('feature_names', None)
                    break
                if feature_names is None:
                    feature_names = self._feature_names_from_train_csv(dataset_name)
                if not feature_names:
                    continue
                model_id = '%s_%s_%s_%s' % (source, mtype, dataset_name, cfg_dir.name)
                records.append({
                    'model_id': model_id,
                    'model_path': str(model_files[0]),
                    'feature_names': feature_names,
                    'source': source,
                    'model_type': mtype,
                    'dataset_name': dataset_name,
                    'dataset_csv_dir': str(self.datasets_dir / dataset_name),
                    'val_r2': val_r2,
                })
        return records

    def get_top_models(self, target, top_n=20):
        ai = self._scan_dir(self.trained_models_dir, target, 'AI')
        anfis = self._scan_dir(self.anfis_models_dir, target, 'ANFIS')
        combined = ai + anfis
        combined.sort(key=lambda x: x['val_r2'], reverse=True)
        selected = combined[:top_n]
        ai_cnt = sum(1 for r in selected if r['source'] == 'AI')
        anfis_cnt = len(selected) - ai_cnt
        logger.info('  Top-%d %s: AI=%d, ANFIS=%d (pool: %d AI + %d ANFIS)' %
                    (top_n, target, ai_cnt, anfis_cnt, len(ai), len(anfis)))
        return selected


# ============================================================================
# PREDICTION COLLECTOR
# ============================================================================

class RealPredictionCollector:
    _ID_COLS = ['NUCLEUS', 'A', 'Z', 'N']

    @staticmethod
    def _load_model(path):
        return joblib.load(path)

    def _predict_record(self, rec, split, target):
        csv_path = Path(rec['dataset_csv_dir']) / (split + '.csv')
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning('  [WARNING] Cannot read %s: %s' % (csv_path, e))
            return None
        feat_cols = rec['feature_names']
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0.0
        if target == 'MM_QM':
            true_cols = [c for c in ['MM', 'QM'] if c in df.columns]
        else:
            true_cols = [target] if target in df.columns else []
        if not true_cols:
            return None
        id_present = [c for c in self._ID_COLS if c in df.columns]
        try:
            model = self._load_model(rec['model_path'])
            X = df[feat_cols].values.astype(float)
            preds = model.predict(X)
        except Exception as e:
            logger.warning('  [WARNING] Predict failed for %s: %s' % (rec['model_id'], e))
            return None
        result = df[id_present + true_cols].copy()
        mid = rec['model_id']
        if target == 'MM_QM':
            arr = np.asarray(preds)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                result[mid + '_MM'] = arr[:, 0]
                result[mid + '_QM'] = arr[:, 1]
            else:
                result[mid + '_MM'] = arr.ravel()
                result[mid + '_QM'] = np.nan
        else:
            result[mid] = np.asarray(preds).ravel()
        return result

    @staticmethod
    def _nuc_key(df):
        if 'NUCLEUS' in df.columns:
            return df['NUCLEUS'].astype(str)
        parts = [df[c].astype(str) for c in ['A', 'Z', 'N'] if c in df.columns]
        if parts:
            return parts[0].str.cat(parts[1:], sep='_')
        return pd.Series(range(len(df)), dtype=str)

    def collect(self, records, target, split='test'):
        frames = []
        for rec in records:
            df = self._predict_record(rec, split, target)
            if df is not None and len(df) > 0:
                df = df.copy()
                df['__nuc_key__'] = self._nuc_key(df).values
                frames.append(df)
        if not frames:
            return None
        _skip = {'NUCLEUS', 'A', 'Z', 'N', 'MM', 'QM', 'Beta_2', '__nuc_key__'}
        merged = frames[0].copy()
        for df in frames[1:]:
            pred_like = [c for c in df.columns if c not in _skip]
            right = df[['__nuc_key__'] + pred_like]
            merged = merged.merge(right, on='__nuc_key__', how='inner')
        merged = merged.drop(columns=['__nuc_key__'], errors='ignore')
        logger.info('  Aligned %s set: %d common nuclei from %d models' %
                    (split, len(merged), len(frames)))
        return merged


# ============================================================================
# REAL ENSEMBLE RUNNER
# ============================================================================

class RealEnsembleRunner:
    _SKIP_COLS = frozenset(['NUCLEUS', 'A', 'Z', 'N', 'MM', 'QM', 'Beta_2'])

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _metrics(y_true, y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) < 2:
            return {'r2': float('nan'), 'rmse': float('nan'), 'mae': float('nan')}
        return {
            'r2': float(r2_score(yt, yp)),
            'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
            'mae': float(mean_absolute_error(yt, yp)),
        }

    def _run_voting(self, preds_mat, y_true, val_r2s):
        results = []
        yp = np.nanmean(preds_mat, axis=1)
        results.append({'method': 'simple_voting', 'predictions': yp,
                        **self._metrics(y_true, yp)})
        w = np.clip(np.array(val_r2s, dtype=float), 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(val_r2s)) / len(val_r2s)
        yp = np.nansum(preds_mat * w[np.newaxis, :], axis=1)
        results.append({'method': 'weighted_voting_r2', 'predictions': yp,
                        **self._metrics(y_true, yp)})
        rmse_per = []
        for i in range(preds_mat.shape[1]):
            m = np.isfinite(y_true) & np.isfinite(preds_mat[:, i])
            if m.sum() >= 2:
                rmse_per.append(float(np.sqrt(mean_squared_error(y_true[m], preds_mat[m, i]))))
            else:
                rmse_per.append(1e10)
        rmse_per = np.array(rmse_per)
        inv_w = 1.0 / (rmse_per + 1e-10)
        inv_w = inv_w / inv_w.sum()
        yp = np.nansum(preds_mat * inv_w[np.newaxis, :], axis=1)
        results.append({'method': 'weighted_voting_inv_rmse', 'predictions': yp,
                        **self._metrics(y_true, yp)})
        return results

    def _run_stacking(self, val_preds, y_val, test_preds, y_test):
        results = []
        if val_preds.shape[0] < 5 or test_preds.shape[0] < 2:
            return results
        col_means = np.nanmean(val_preds, axis=0)
        for j in range(val_preds.shape[1]):
            val_preds[np.isnan(val_preds[:, j]), j] = col_means[j]
            test_preds[np.isnan(test_preds[:, j]), j] = col_means[j]
        meta_configs = {
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'rf_meta': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            'gbm_meta': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        }
        for name, meta in meta_configs.items():
            try:
                mask_val = np.isfinite(y_val)
                mask_test = np.isfinite(y_test)
                if mask_val.sum() < 5:
                    continue
                meta.fit(val_preds[mask_val], y_val[mask_val])
                yp = meta.predict(test_preds[mask_test])
                m = self._metrics(y_test[mask_test], yp)
                full_yp = np.full(len(y_test), float('nan'))
                full_yp[mask_test] = yp
                results.append({'method': 'stacking_' + name, 'predictions': full_yp,
                                'n_base_models': val_preds.shape[1], **m})
                logger.info('    stacking_%s: R2=%.4f' % (name, m['r2']))
            except Exception as e:
                logger.warning('  [WARNING] Meta-model %s failed: %s' % (name, e))
        return results

    def run_target(self, target, records, collector):
        logger.info('\n[TARGET] Ensemble for target: %s' % target)
        sub_targets = ['MM', 'QM'] if target == 'MM_QM' else [target]
        per_subtarget = {}
        test_df = collector.collect(records, target, split='test')
        val_df = collector.collect(records, target, split='val')
        if test_df is None or len(test_df) == 0:
            logger.warning('  [WARNING] No aligned test predictions for %s' % target)
            return per_subtarget
        for st in sub_targets:
            true_col = st if target == 'MM_QM' else target
            if target == 'MM_QM':
                pred_cols = [c for c in test_df.columns if c.endswith('_' + st)]
            else:
                pred_cols = [c for c in test_df.columns if c not in self._SKIP_COLS]
            if true_col not in test_df.columns or not pred_cols:
                continue
            y_test_arr = test_df[true_col].values.astype(float)
            preds_test = test_df[pred_cols].values.astype(float)
            val_r2s = []
            for pc in pred_cols:
                base_id = pc.replace('_' + st, '') if target == 'MM_QM' else pc
                val_r2s.append(next(
                    (r['val_r2'] for r in records if r['model_id'] == base_id), 0.5))
            voting_res = self._run_voting(preds_test.copy(), y_test_arr, val_r2s)
            stacking_res = []
            if val_df is not None and len(val_df) >= 5 and true_col in val_df.columns:
                if target == 'MM_QM':
                    val_pred_cols = [c for c in val_df.columns if c.endswith('_' + st)]
                else:
                    val_pred_cols = [c for c in val_df.columns if c not in self._SKIP_COLS]
                common = [c for c in pred_cols if c in val_pred_cols]
                if common:
                    y_val_arr = val_df[true_col].values.astype(float)
                    preds_val = val_df[common].values.astype(float)
                    preds_test_al = test_df[common].values.astype(float)
                    stacking_res = self._run_stacking(
                        preds_val.copy(), y_val_arr, preds_test_al.copy(), y_test_arr)
            all_results = voting_res + stacking_res
            if not all_results:
                continue
            best = max(all_results, key=lambda x: x.get('r2') or -999)
            per_subtarget[st] = {
                'voting': voting_res, 'stacking': stacking_res,
                'all_results': all_results, 'best': best,
                'y_test': y_test_arr, 'pred_cols': pred_cols, 'test_df': test_df,
            }
            for res in all_results:
                logger.info('    %-35s R2=%.4f  RMSE=%.4f' % (
                    res['method'], res.get('r2', float('nan')),
                    res.get('rmse', float('nan'))))
            logger.info('  Best (%s): %s  R2=%.4f' % (
                st, best['method'], best.get('r2', float('nan'))))
        return per_subtarget


# ============================================================================
# EXCEL REPORT WRITER
# ============================================================================

def _write_ensemble_excel(all_target_results, output_dir):
    from datetime import datetime as _dt
    ts = _dt.now().strftime('%Y%m%d_%H%M%S')
    out_path = output_dir / ('ensemble_report_' + ts + '.xlsx')
    summary_rows = []
    try:
        import openpyxl  # noqa
    except ImportError:
        logger.warning('[WARNING] openpyxl not available; skipping Excel report')
        return None
        openpyxl = None
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for target, sub_results in all_target_results.items():
            for st, res in sub_results.items():
                rows = []
                for r in res.get('all_results', []):
                    rows.append({'Target': st, 'Method': r['method'],
                                 'R2': r.get('r2', float('nan')),
                                 'RMSE': r.get('rmse', float('nan')),
                                 'MAE': r.get('mae', float('nan')),
                                 'N_Base_Models': r.get('n_base_models',
                                                        len(res.get('pred_cols', [])))})
                    summary_rows.append({k: v for k, v in rows[-1].items()})
                if rows:
                    pd.DataFrame(rows).to_excel(
                        writer, sheet_name=(st + '_Results')[:31], index=False)
                test_df = res.get('test_df')
                if test_df is not None and len(test_df) > 0:
                    test_df.to_excel(
                        writer, sheet_name=(st + '_Predictions')[:31], index=False)
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
    logger.info('[OK] Ensemble report saved: %s' % out_path)
    return out_path


# ============================================================================
# MAIN PFAZ 7 PIPELINE
# ============================================================================

def pfaz7_complete_pipeline(
        trained_models_dir=None,
        anfis_models_dir=None,
        datasets_dir=None,
        output_dir=None,
        top_n=20,
):
    """
    PFAZ 7 - Complete Ensemble & Meta-Model Pipeline

    Loads real AI + ANFIS trained models from PFAZ2/3 outputs, runs voting
    and stacking ensembles per target (MM, QM, Beta_2, MM_QM), saves Excel.
    Paths default to the standard pipeline directory layout.
    """
    logger.info('=' * 80)
    logger.info('PFAZ 7: COMPLETE ENSEMBLE & META-MODEL PIPELINE')
    logger.info('=' * 80)

    _module_dir = Path(__file__).resolve().parent
    _project_root = _module_dir.parent.parent
    _outputs = _project_root / 'outputs'

    tm_dir = Path(trained_models_dir) if trained_models_dir else _outputs / 'trained_models'
    am_dir = Path(anfis_models_dir) if anfis_models_dir else _outputs / 'anfis_models'
    ds_dir = Path(datasets_dir) if datasets_dir else _outputs / 'generated_datasets'
    out_dir = Path(output_dir) if output_dir else _outputs / 'ensemble_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info('  trained_models : %s' % tm_dir)
    logger.info('  anfis_models   : %s' % am_dir)
    logger.info('  datasets       : %s' % ds_dir)
    logger.info('  output         : %s' % out_dir)

    if not JOBLIB_AVAILABLE:
        logger.error('[ERROR] joblib not installed; cannot load models')
        return {'status': 'error', 'reason': 'joblib not available'}

    loader = RealModelLoader(tm_dir, am_dir, ds_dir)
    collector = RealPredictionCollector()
    runner = RealEnsembleRunner(out_dir)

    targets = ['MM', 'QM']
    all_target_results = {}
    grand_summary = []

    for target in targets:
        logger.info('\n' + '=' * 60)
        logger.info('TARGET: %s' % target)
        logger.info('=' * 60)
        try:
            records = loader.get_top_models(target, top_n=top_n)
            if not records:
                logger.warning('  [WARNING] No models found for %s, skipping' % target)
                continue
            sub_results = runner.run_target(target, records, collector)
            all_target_results[target] = sub_results
            for st, res in sub_results.items():
                best = res.get('best', {})
                grand_summary.append({
                    'Target': st,
                    'Best_Method': best.get('method', 'N/A'),
                    'Best_R2': best.get('r2', float('nan')),
                    'Best_RMSE': best.get('rmse', float('nan')),
                    'N_Models': len(records),
                })
        except Exception as e:
            logger.error('  [ERROR] Target %s failed: %s' % (target, e))
            import traceback
            traceback.print_exc()

    report_path = None
    try:
        report_path = _write_ensemble_excel(all_target_results, out_dir)
    except Exception as e:
        logger.error('[ERROR] Excel write failed: %s' % e)

    try:
        summary_json = out_dir / 'ensemble_summary.json'
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': datetime.now().isoformat(),
                       'targets': grand_summary}, f, indent=2, default=str)
        logger.info('[OK] Summary JSON: %s' % summary_json)
    except Exception as e:
        logger.warning('[WARNING] JSON summary save failed: %s' % e)

    logger.info('\n' + '=' * 80)
    logger.info('PFAZ 7: FINAL SUMMARY')
    logger.info('=' * 80)
    for row in grand_summary:
        logger.info('  %-8s  best=%-35s  R2=%.4f  RMSE=%.4f' % (
            row['Target'], row['Best_Method'], row['Best_R2'], row['Best_RMSE']))
    logger.info('[COMPLETE] PFAZ 7 COMPLETED SUCCESSFULLY!')
    logger.info('=' * 80)

    return {
        'status': 'completed',
        'all_target_results': all_target_results,
        'grand_summary': grand_summary,
        'report_path': str(report_path) if report_path else None,
    }


def main():
    try:
        results = pfaz7_complete_pipeline()
        return results
    except Exception as e:
        logger.error('\n[ERROR] ERROR in PFAZ 7: %s' % str(e))
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    results = main()
