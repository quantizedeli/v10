"""
Data Enrichment Module for PYV5_5
Synthetic data generation, augmentation, and quality enhancement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataEnricher:
    """Advanced data enrichment with multiple techniques"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.scaler = StandardScaler()
        
    def _default_config(self) -> Dict:
        return {
            'interpolation_method': 'cubic',
            'smoothing_sigma': 1.5,
            'noise_level': 0.02,
            'augmentation_factor': 3,
            'synthetic_samples': 1000,
            'outlier_threshold': 3.0
        }
    
    def enrich_dataset(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main enrichment pipeline
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            methods: List of enrichment methods to apply
            
        Returns:
            X_enriched, y_enriched: Augmented dataset
        """
        if methods is None:
            methods = ['interpolate', 'smooth', 'augment', 'synthetic']
        
        X_list, y_list = [X], [y]
        
        logger.info(f"Starting data enrichment with methods: {methods}")
        logger.info(f"Original dataset size: {X.shape[0]} samples")
        
        for method in methods:
            if method == 'interpolate':
                X_interp, y_interp = self._interpolate_data(X, y)
                X_list.append(X_interp)
                y_list.append(y_interp)
                
            elif method == 'smooth':
                X_smooth, y_smooth = self._smooth_data(X, y)
                X_list.append(X_smooth)
                y_list.append(y_smooth)
                
            elif method == 'augment':
                X_aug, y_aug = self._augment_data(X, y)
                X_list.append(X_aug)
                y_list.append(y_aug)
                
            elif method == 'synthetic':
                X_syn, y_syn = self._generate_synthetic(X, y)
                X_list.append(X_syn)
                y_list.append(y_syn)
                
            elif method == 'noise':
                X_noise, y_noise = self._add_controlled_noise(X, y)
                X_list.append(X_noise)
                y_list.append(y_noise)
        
        # Combine all enriched data
        X_enriched = np.vstack(X_list)
        y_enriched = np.concatenate(y_list)
        
        # Remove outliers
        X_enriched, y_enriched = self._remove_outliers(X_enriched, y_enriched)
        
        logger.info(f"Enriched dataset size: {X_enriched.shape[0]} samples")
        logger.info(f"Enrichment factor: {X_enriched.shape[0] / X.shape[0]:.2f}x")
        
        return X_enriched, y_enriched
    
    def _interpolate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate data points for smoother distribution"""
        n_samples = X.shape[0]
        n_new_samples = int(n_samples * 1.5)
        
        X_interp = np.zeros((n_new_samples, X.shape[1]))
        
        # Sort by first feature for interpolation
        sort_idx = np.argsort(X[:, 0])
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Interpolate each feature
        for i in range(X.shape[1]):
            if self.config['interpolation_method'] == 'cubic':
                f = CubicSpline(np.arange(n_samples), X_sorted[:, i])
            else:
                f = interp1d(np.arange(n_samples), X_sorted[:, i], 
                           kind=self.config['interpolation_method'])
            
            new_indices = np.linspace(0, n_samples - 1, n_new_samples)
            X_interp[:, i] = f(new_indices)
        
        # Interpolate targets
        f_y = CubicSpline(np.arange(n_samples), y_sorted)
        y_interp = f_y(np.linspace(0, n_samples - 1, n_new_samples))
        
        return X_interp, y_interp
    
    def _smooth_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian smoothing to reduce noise"""
        X_smooth = np.copy(X)
        
        for i in range(X.shape[1]):
            X_smooth[:, i] = gaussian_filter1d(X[:, i], 
                                              sigma=self.config['smoothing_sigma'])
        
        y_smooth = gaussian_filter1d(y, sigma=self.config['smoothing_sigma'])
        
        return X_smooth, y_smooth
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment data with variations"""
        n_augments = self.config['augmentation_factor']
        X_aug_list, y_aug_list = [], []
        
        for _ in range(n_augments):
            # Random scaling
            scale = np.random.uniform(0.95, 1.05, size=(1, X.shape[1]))
            X_scaled = X * scale
            
            # Random shift
            shift = np.random.normal(0, 0.01, size=(1, X.shape[1]))
            X_shifted = X_scaled + shift
            
            X_aug_list.append(X_shifted)
            y_aug_list.append(y)
        
        X_aug = np.vstack(X_aug_list)
        y_aug = np.concatenate(y_aug_list)
        
        return X_aug, y_aug
    
    def _generate_synthetic(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples using statistical properties"""
        n_synthetic = self.config['synthetic_samples']
        
        # Calculate statistics
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_cov = np.cov(X.T)
        
        # Generate synthetic features
        X_synthetic = np.random.multivariate_normal(X_mean, X_cov, n_synthetic)
        
        # Fit relationship between X and y
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)
        
        # Predict targets for synthetic features
        y_synthetic = rf.predict(X_synthetic)
        
        return X_synthetic, y_synthetic
    
    def _add_controlled_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add controlled noise for robustness"""
        noise_X = np.random.normal(0, self.config['noise_level'], X.shape)
        noise_y = np.random.normal(0, self.config['noise_level'] * np.std(y), y.shape)
        
        X_noise = X + noise_X
        y_noise = y + noise_y
        
        return X_noise, y_noise
    
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove statistical outliers"""
        # Z-score based outlier removal
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        mask = z_scores < self.config['outlier_threshold']
        
        logger.info(f"Removed {np.sum(~mask)} outliers")
        
        return X[mask], y[mask]
    
    def create_balanced_dataset(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create balanced dataset across target value ranges"""
        # Bin targets
        bins = np.linspace(y.min(), y.max(), n_bins + 1)
        bin_indices = np.digitize(y, bins)
        
        # Find minimum samples per bin
        min_samples = min([np.sum(bin_indices == i) for i in range(1, n_bins + 1)])
        
        X_balanced, y_balanced = [], []
        
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            X_bin = X[mask]
            y_bin = y[mask]
            
            if len(X_bin) > min_samples:
                # Randomly sample to match minimum
                idx = np.random.choice(len(X_bin), min_samples, replace=False)
                X_balanced.append(X_bin[idx])
                y_balanced.append(y_bin[idx])
            else:
                X_balanced.append(X_bin)
                y_balanced.append(y_bin)
        
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)
        
        logger.info(f"Created balanced dataset: {X_balanced.shape[0]} samples")
        
        return X_balanced, y_balanced
    
    def enrich_with_physics_constraints(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       physics_func: callable) -> Tuple[np.ndarray, np.ndarray]:
        """Enrich data while maintaining physics constraints"""
        # Generate candidates
        X_candidates = []
        y_candidates = []
        
        for _ in range(1000):
            # Perturb existing samples
            idx = np.random.randint(0, X.shape[0])
            X_new = X[idx] + np.random.normal(0, 0.05, X.shape[1])
            
            # Check physics constraint
            if physics_func(X_new):
                X_candidates.append(X_new)
                # Calculate corresponding y value
                y_new = y[idx] + np.random.normal(0, 0.05 * np.std(y))
                y_candidates.append(y_new)
        
        if X_candidates:
            X_enriched = np.vstack([X] + X_candidates)
            y_enriched = np.concatenate([y] + y_candidates)
            logger.info(f"Added {len(X_candidates)} physics-constrained samples")
            return X_enriched, y_enriched
        
        return X, y
    
    def get_enrichment_report(self, X_original: np.ndarray, 
                             X_enriched: np.ndarray) -> Dict:
        """Generate enrichment statistics report"""
        return {
            'original_samples': X_original.shape[0],
            'enriched_samples': X_enriched.shape[0],
            'enrichment_factor': X_enriched.shape[0] / X_original.shape[0],
            'original_feature_stats': {
                'mean': np.mean(X_original, axis=0).tolist(),
                'std': np.std(X_original, axis=0).tolist()
            },
            'enriched_feature_stats': {
                'mean': np.mean(X_enriched, axis=0).tolist(),
                'std': np.std(X_enriched, axis=0).tolist()
            }
        }


if __name__ == "__main__":
    # Test the enricher
    np.random.seed(42)
    X_test = np.random.randn(100, 5)
    y_test = np.sum(X_test, axis=1) + np.random.randn(100) * 0.1
    
    enricher = DataEnricher()
    X_enriched, y_enriched = enricher.enrich_dataset(X_test, y_test)
    
    report = enricher.get_enrichment_report(X_test, X_enriched)
    print("\n=== Data Enrichment Report ===")
    for key, value in report.items():
        print(f"{key}: {value}")
