"""
Robustness Testing & Iterative Outlier Cleaning Module
Inspired by ooo.py optimization approach
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustnessTester:
    """Model sağlamlık testi ve iterative cleaning"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.cleaning_history = []
        
    def _default_config(self) -> Dict:
        return {
            'max_iterations': 5,
            'outlier_threshold': 2.0,  # Z-score
            'min_improvement': 0.01,   # Minimum R² improvement
            'retrain_thresholds': {
                'critical': 0.3,    # R² < 0.3 -> Kesinlikle retrain
                'poor': 0.5,        # 0.3 < R² < 0.5 -> Epoch artır
                'acceptable': 0.7,  # 0.5 < R² < 0.7 -> Normal
                'good': 0.85        # R² > 0.85 -> Mükemmel
            },
            'replacement_strategy': 'nearest_neighbor',  # or 'random_sample'
            'max_outliers_per_iteration': 5
        }
    
    def iterative_training_with_cleaning(self,
                                         model,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_val: np.ndarray,
                                         y_val: np.ndarray,
                                         source_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Iterative training with outlier detection and replacement
        
        Args:
            model: Model object with train() and predict() methods
            X_train, y_train: Training data
            X_val, y_val: Validation data
            source_data: Source dataframe for nucleus replacement
            
        Returns:
            Best training results with cleaning history
        """
        
        logger.info("\n" + "="*60)
        logger.info("ITERATIVE ROBUSTNESS TRAINING")
        logger.info("="*60)
        
        best_r2 = -999
        best_model = None
        best_iteration = 0
        
        X_current = X_train.copy()
        y_current = y_train.copy()
        
        used_indices = set()  # For replacement tracking
        
        for iteration in range(self.config['max_iterations']):
            logger.info(f"\n[RUN] Iteration {iteration + 1}/{self.config['max_iterations']}")
            
            # Train model
            model.train(X_current, y_current, X_val, y_val)
            
            # Evaluate
            y_train_pred = model.predict(X_current)
            y_val_pred = model.predict(X_val)
            
            train_r2 = r2_score(y_current, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            logger.info(f"   Train R^2: {train_r2:.4f} | Val R^2: {val_r2:.4f}")
            
            # Check if best
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_model = model
                best_iteration = iteration
                logger.info(f"   [SUCCESS] New best model! R^2 = {best_r2:.4f}")
            
            # Detect outliers
            outliers, outlier_info = self._detect_outliers(
                y_current, y_train_pred, X_current
            )
            
            logger.info(f"   Outliers detected: {len(outliers)}")
            
            if len(outliers) == 0:
                logger.info(f"   [TARGET] No outliers - stopping at iteration {iteration + 1}")
                break
            
            # Check if should continue
            quality_level = self._assess_quality(val_r2)
            logger.info(f"   Quality: {quality_level}")
            
            if quality_level in ['good', 'acceptable'] and iteration > 0:
                logger.info(f"   [SUCCESS] Acceptable quality - stopping")
                break
            
            # Replace outliers
            if source_data is not None:
                X_current, y_current, replacements = self._replace_outliers(
                    X_current, y_current, outliers,
                    source_data, used_indices
                )
                
                self.cleaning_history.append({
                    'iteration': iteration + 1,
                    'n_outliers': len(outliers),
                    'n_replaced': len(replacements),
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'outlier_details': outlier_info
                })
                
                logger.info(f"   [RUN] Replaced {len(replacements)} outliers")
            else:
                # No replacement available - just remove
                mask = np.ones(len(X_current), dtype=bool)
                mask[outliers[:self.config['max_outliers_per_iteration']]] = False
                X_current = X_current[mask]
                y_current = y_current[mask]
                
                logger.info(f"   [ERROR] Removed {np.sum(~mask)} outliers (no source data)")
                
                self.cleaning_history.append({
                    'iteration': iteration + 1,
                    'n_outliers': len(outliers),
                    'n_removed': np.sum(~mask),
                    'train_r2': train_r2,
                    'val_r2': val_r2
                })
            
            # Check for minimal improvement
            if iteration > 0:
                prev_r2 = self.cleaning_history[iteration - 1]['val_r2']
                improvement = val_r2 - prev_r2
                
                if improvement < self.config['min_improvement']:
                    logger.info(f"   [WARNING] Minimal improvement ({improvement:.4f}) - stopping")
                    break
        
        # Final results
        results = {
            'best_r2': best_r2,
            'best_iteration': best_iteration + 1,
            'total_iterations': len(self.cleaning_history),
            'cleaning_history': self.cleaning_history,
            'final_train_size': len(X_current),
            'original_train_size': len(X_train),
            'samples_modified': len(X_train) - len(X_current)
        }
        
        logger.info("\n" + "="*60)
        logger.info(f"[SUCCESS] Best R^2: {best_r2:.4f} at iteration {best_iteration + 1}")
        logger.info(f"[REPORT] Total modifications: {results['samples_modified']}")
        logger.info("="*60)
        
        return results
    
    def _detect_outliers(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect outliers using Z-score on residuals
        
        Returns:
            outlier_indices, outlier_info
        """
        residuals = y_true - y_pred
        
        # Z-score
        z_scores = np.abs(stats.zscore(residuals))
        
        # Outliers
        outlier_mask = z_scores > self.config['outlier_threshold']
        outlier_indices = np.where(outlier_mask)[0]
        
        # Info
        outlier_info = {
            'z_scores': z_scores[outlier_mask].tolist(),
            'residuals': residuals[outlier_mask].tolist(),
            'predictions': y_pred[outlier_mask].tolist(),
            'actuals': y_true[outlier_mask].tolist()
        }
        
        return outlier_indices, outlier_info
    
    def _assess_quality(self, r2: float) -> str:
        """Assess model quality based on R²"""
        thresholds = self.config['retrain_thresholds']
        
        if r2 >= thresholds['good']:
            return 'excellent'
        elif r2 >= thresholds['acceptable']:
            return 'good'
        elif r2 >= thresholds['poor']:
            return 'acceptable'
        elif r2 >= thresholds['critical']:
            return 'poor'
        else:
            return 'critical'
    
    def _replace_outliers(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         outlier_indices: np.ndarray,
                         source_data: pd.DataFrame,
                         used_indices: set) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Replace outliers with similar samples from source
        
        Returns:
            X_new, y_new, replacement_list
        """
        X_new = X.copy()
        y_new = y.copy()
        replacements = []
        
        # Limit replacements per iteration
        n_replace = min(len(outlier_indices), self.config['max_outliers_per_iteration'])
        outliers_to_replace = outlier_indices[:n_replace]
        
        # Available samples
        available_mask = ~source_data.index.isin(used_indices)
        available_data = source_data[available_mask]
        
        if len(available_data) < 10:
            logger.warning("   [WARNING] Insufficient replacement samples")
            return X_new, y_new, []
        
        # Feature columns (assuming source_data has same features)
        feature_cols = list(range(X.shape[1]))
        
        for idx in outliers_to_replace:
            outlier_features = X[idx]
            
            if self.config['replacement_strategy'] == 'nearest_neighbor':
                # Find nearest neighbor
                if hasattr(available_data, 'values'):
                    available_X = available_data.iloc[:, :X.shape[1]].values
                    distances = np.sqrt(np.sum((available_X - outlier_features)**2, axis=1))
                    
                    # Get top 10 closest
                    closest_indices = np.argsort(distances)[:10]
                    
                    # Random select from top 10
                    selected_idx = np.random.choice(closest_indices)
                    replacement_row = available_data.iloc[selected_idx]
                    
                    # Replace
                    X_new[idx] = replacement_row.iloc[:X.shape[1]].values
                    y_new[idx] = replacement_row.iloc[X.shape[1]]  # Assuming target is next column
                    
                    # Mark as used
                    used_indices.add(replacement_row.name)
                    
                    replacements.append({
                        'original_idx': int(idx),
                        'replacement_idx': int(replacement_row.name),
                        'distance': float(distances[selected_idx])
                    })
        
        return X_new, y_new, replacements
    
    def get_cleaning_report(self) -> pd.DataFrame:
        """Generate cleaning history report"""
        if not self.cleaning_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.cleaning_history)
    
    def decide_retrain_strategy(self, current_r2: float, previous_attempts: List[Dict]) -> Dict:
        """
        Decide if retrain needed and with what parameters
        Based on ooo.py strategy
        """
        strategy = {
            'should_retrain': False,
            'epochs_multiplier': 1.0,
            'reason': '',
            'quality_level': self._assess_quality(current_r2)
        }
        
        if current_r2 < self.config['retrain_thresholds']['critical']:
            strategy['should_retrain'] = True
            strategy['epochs_multiplier'] = 2.0
            strategy['reason'] = 'Critical low R² - aggressive retrain'
            
        elif current_r2 < self.config['retrain_thresholds']['poor']:
            strategy['should_retrain'] = True
            strategy['epochs_multiplier'] = 1.5
            strategy['reason'] = 'Poor performance - increase epochs'
            
        elif current_r2 < self.config['retrain_thresholds']['acceptable']:
            # Check improvement trend
            if len(previous_attempts) > 0:
                last_r2 = previous_attempts[-1].get('r2', 0)
                improvement = current_r2 - last_r2
                
                if improvement < self.config['min_improvement']:
                    strategy['should_retrain'] = True
                    strategy['epochs_multiplier'] = 1.2
                    strategy['reason'] = 'Stagnant improvement - mild retrain'
        
        return strategy


class NucleusReplacer:
    """
    Smart nucleus replacement system
    From ooo.py - finds similar nuclei for outlier replacement
    """
    
    def __init__(self, source_file: str):
        """
        Args:
            source_file: Path to aaa.txt or similar source data
        """
        self.df = pd.read_csv(source_file, sep='\t')
        self.used_indices = set()
        logger.info(f"Nucleus replacer initialized with {len(self.df)} samples")
    
    def find_replacement(self,
                        outlier_row: np.ndarray,
                        feature_names: List[str],
                        n_candidates: int = 10) -> Tuple[Optional[pd.Series], Optional[Dict]]:
        """
        Find replacement nucleus for outlier
        
        Args:
            outlier_row: Feature vector of outlier
            feature_names: Names of features
            n_candidates: Number of candidates to consider
            
        Returns:
            replacement_row, metadata
        """
        # Available samples
        available = self.df[~self.df.index.isin(self.used_indices)]
        
        if len(available) < n_candidates:
            logger.warning(f"Only {len(available)} replacement candidates available")
            return None, None
        
        # Calculate distances
        try:
            available_features = available[feature_names].values
            distances = np.sqrt(np.sum((available_features - outlier_row)**2, axis=1))
            
            # Get top candidates
            closest_indices = np.argsort(distances)[:n_candidates]
            
            # Random selection from top candidates
            selected_idx = np.random.choice(closest_indices)
            selected_distance = distances[selected_idx]
            
            # Get replacement
            replacement = available.iloc[selected_idx]
            
            # Mark as used
            self.used_indices.add(replacement.name)
            
            metadata = {
                'distance': float(selected_distance),
                'nucleus_id': replacement.get('NUCLEUS', 'N/A'),
                'candidate_rank': int(np.where(closest_indices == selected_idx)[0][0] + 1)
            }
            
            logger.debug(f"Found replacement: distance={selected_distance:.4f}, rank={metadata['candidate_rank']}")
            
            return replacement, metadata
            
        except Exception as e:
            logger.error(f"Replacement failed: {e}")
            return None, None


if __name__ == "__main__":
    # Test robustness tester
    np.random.seed(42)
    
    # Generate test data with outliers
    X_train = np.random.randn(500, 5)
    y_train = np.sum(X_train, axis=1) + np.random.randn(500) * 0.1
    
    # Add outliers
    outlier_indices = np.random.choice(500, 20, replace=False)
    y_train[outlier_indices] += np.random.randn(20) * 5.0  # Strong outliers
    
    X_val = np.random.randn(100, 5)
    y_val = np.sum(X_val, axis=1) + np.random.randn(100) * 0.1
    
    # Mock model
    class MockModel:
        def train(self, X, y, X_val, y_val):
            from sklearn.linear_model import Ridge
            self.model = Ridge()
            self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)
    
    # Test
    tester = RobustnessTester()
    model = MockModel()
    
    results = tester.iterative_training_with_cleaning(
        model, X_train, y_train, X_val, y_val
    )
    
    print("\n=== Robustness Test Results ===")
    for key, value in results.items():
        if key != 'cleaning_history':
            print(f"{key}: {value}")
    
    # Cleaning report
    report = tester.get_cleaning_report()
    print("\n=== Cleaning History ===")
    print(report)
