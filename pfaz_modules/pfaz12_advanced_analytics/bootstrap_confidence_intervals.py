# -*- coding: utf-8 -*-
"""
BOOTSTRAP CONFIDENCE INTERVALS
===============================

Comprehensive bootstrap methods for uncertainty quantification

Methods:
1. Percentile Bootstrap
2. BCa (Bias-Corrected and Accelerated)
3. Bootstrap for any statistic (mean, median, std, R², etc.)
4. Bootstrap hypothesis testing
5. Bootstrap model comparison
6. Visualization of bootstrap distributions

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 12 Complete
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BootstrapConfidenceIntervals:
    """
    Bootstrap methods for confidence interval estimation
    
    Bootstrap resampling provides:
    - Confidence intervals without distributional assumptions
    - Uncertainty quantification for any statistic
    - Robust inference for complex metrics
    """
    
    def __init__(self, 
                 n_bootstrap: int = 10000,
                 confidence_level: float = 0.95,
                 random_state: int = 42,
                 output_dir: str = 'bootstrap_results'):
        """
        Initialize bootstrap system
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            random_state: Random seed for reproducibility
            output_dir: Directory for outputs
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(random_state)
        
        self.results = {}
        
        logger.info(f"✓ BootstrapCI initialized (n={n_bootstrap}, CI={confidence_level*100}%)")
    
    # ========================================================================
    # BASIC BOOTSTRAP
    # ========================================================================
    
    def bootstrap_statistic(self,
                           data: np.ndarray,
                           statistic: Callable = np.mean,
                           method: str = 'percentile') -> Dict:
        """
        Bootstrap confidence interval for any statistic
        
        Args:
            data: Original data
            statistic: Function to compute (e.g., np.mean, np.median, np.std)
            method: 'percentile' or 'bca' (bias-corrected accelerated)
            
        Returns:
            dict with point_estimate, ci_lower, ci_upper, bootstrap_distribution
        """
        logger.info(f"\n-> Bootstrap CI for statistic ({method})")
        
        # Original statistic
        point_estimate = statistic(data)
        
        # Bootstrap samples
        bootstrap_stats = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats[i] = statistic(sample)
        
        # Confidence interval
        if method == 'percentile':
            ci_lower, ci_upper = self._percentile_ci(bootstrap_stats)
        elif method == 'bca':
            ci_lower, ci_upper = self._bca_ci(data, statistic, bootstrap_stats, point_estimate)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result = {
            'method': method,
            'n_bootstrap': self.n_bootstrap,
            'point_estimate': float(point_estimate),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_width': float(ci_upper - ci_lower),
            'bootstrap_mean': float(np.mean(bootstrap_stats)),
            'bootstrap_std': float(np.std(bootstrap_stats)),
            'bootstrap_distribution': bootstrap_stats
        }
        
        logger.info(f"  Point estimate: {point_estimate:.4f}")
        logger.info(f"  {self.confidence_level*100}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return result
    
    def _percentile_ci(self, bootstrap_stats: np.ndarray) -> Tuple[float, float]:
        """Percentile method confidence interval"""
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _bca_ci(self, 
                data: np.ndarray,
                statistic: Callable,
                bootstrap_stats: np.ndarray,
                point_estimate: float) -> Tuple[float, float]:
        """
        BCa (Bias-Corrected and Accelerated) confidence interval
        More accurate than percentile method
        """
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < point_estimate))
        
        # Acceleration (jackknife)
        n = len(data)
        jackknife_stats = np.zeros(n)
        
        for i in range(n):
            # Leave-one-out sample
            sample = np.delete(data, i)
            jackknife_stats[i] = statistic(sample)
        
        jackknife_mean = np.mean(jackknife_stats)
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
        acceleration = numerator / denominator if denominator != 0 else 0
        
        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(self.alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - self.alpha / 2)
        
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - acceleration * (z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - acceleration * (z0 + z_alpha_upper)))
        
        ci_lower = np.percentile(bootstrap_stats, p_lower * 100)
        ci_upper = np.percentile(bootstrap_stats, p_upper * 100)
        
        return ci_lower, ci_upper
    
    # ========================================================================
    # MODEL PERFORMANCE BOOTSTRAP
    # ========================================================================
    
    def bootstrap_model_performance(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   metrics: List[str] = ['r2', 'rmse', 'mae']) -> Dict:
        """
        Bootstrap confidence intervals for model performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to compute
            
        Returns:
            dict with CIs for each metric
        """
        logger.info(f"\n-> Bootstrap model performance ({len(metrics)} metrics)")
        
        results = {}
        
        for metric_name in metrics:
            logger.info(f"  -> Metric: {metric_name}")
            
            # Define metric function
            if metric_name == 'r2':
                def metric_func(yt, yp):
                    return 1 - np.sum((yt - yp)**2) / np.sum((yt - np.mean(yt))**2)
            elif metric_name == 'rmse':
                def metric_func(yt, yp):
                    return np.sqrt(np.mean((yt - yp)**2))
            elif metric_name == 'mae':
                def metric_func(yt, yp):
                    return np.mean(np.abs(yt - yp))
            elif metric_name == 'mape':
                def metric_func(yt, yp):
                    return np.mean(np.abs((yt - yp) / yt)) * 100
            else:
                logger.warning(f"    Unknown metric: {metric_name}, skipping")
                continue
            
            # Original metric
            point_estimate = metric_func(y_true, y_pred)
            
            # Bootstrap
            bootstrap_values = np.zeros(self.n_bootstrap)
            n = len(y_true)
            
            for i in range(self.n_bootstrap):
                # Resample indices
                indices = np.random.choice(n, size=n, replace=True)
                yt_sample = y_true[indices]
                yp_sample = y_pred[indices]
                
                bootstrap_values[i] = metric_func(yt_sample, yp_sample)
            
            # CI
            ci_lower, ci_upper = self._percentile_ci(bootstrap_values)
            
            results[metric_name] = {
                'point_estimate': float(point_estimate),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'ci_width': float(ci_upper - ci_lower),
                'bootstrap_std': float(np.std(bootstrap_values)),
                'bootstrap_distribution': bootstrap_values
            }
            
            logger.info(f"    {metric_name.upper()}: {point_estimate:.4f} "
                       f"[{ci_lower:.4f}, {ci_upper:.4f}]")
        
        self.results['model_performance'] = results
        return results
    
    # ========================================================================
    # MODEL COMPARISON BOOTSTRAP
    # ========================================================================
    
    def bootstrap_model_comparison(self,
                                   y_true: np.ndarray,
                                   y_pred_a: np.ndarray,
                                   y_pred_b: np.ndarray,
                                   metric: str = 'r2',
                                   model_a_name: str = 'Model A',
                                   model_b_name: str = 'Model B') -> Dict:
        """
        Bootstrap test for difference between two models
        
        Args:
            y_true: True values
            y_pred_a: Predictions from model A
            y_pred_b: Predictions from model B
            metric: Metric to compare
            
        Returns:
            dict with difference CI and p-value
        """
        logger.info(f"\n-> Bootstrap model comparison: {model_a_name} vs {model_b_name}")
        logger.info(f"  Metric: {metric}")
        
        # Define metric
        if metric == 'r2':
            def metric_func(yt, yp):
                return 1 - np.sum((yt - yp)**2) / np.sum((yt - np.mean(yt))**2)
        elif metric == 'rmse':
            def metric_func(yt, yp):
                return np.sqrt(np.mean((yt - yp)**2))
        elif metric == 'mae':
            def metric_func(yt, yp):
                return np.mean(np.abs(yt - yp))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Original difference
        metric_a = metric_func(y_true, y_pred_a)
        metric_b = metric_func(y_true, y_pred_b)
        observed_diff = metric_a - metric_b
        
        # Bootstrap differences
        bootstrap_diffs = np.zeros(self.n_bootstrap)
        n = len(y_true)
        
        for i in range(self.n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            yt_sample = y_true[indices]
            ya_sample = y_pred_a[indices]
            yb_sample = y_pred_b[indices]
            
            metric_a_boot = metric_func(yt_sample, ya_sample)
            metric_b_boot = metric_func(yt_sample, yb_sample)
            
            bootstrap_diffs[i] = metric_a_boot - metric_b_boot
        
        # CI for difference
        ci_lower, ci_upper = self._percentile_ci(bootstrap_diffs)
        
        # Bootstrap p-value (two-sided)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= 0),
            np.mean(bootstrap_diffs <= 0)
        )
        
        # Significant if CI doesn't include 0
        significant = not (ci_lower <= 0 <= ci_upper)
        
        result = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'metric': metric,
            'metric_a': float(metric_a),
            'metric_b': float(metric_b),
            'difference': float(observed_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'p_value': float(p_value),
            'significant': significant,
            'bootstrap_distribution': bootstrap_diffs
        }
        
        logger.info(f"  {model_a_name} {metric}: {metric_a:.4f}")
        logger.info(f"  {model_b_name} {metric}: {metric_b:.4f}")
        logger.info(f"  Difference: {observed_diff:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  p-value: {p_value:.4f}, Significant: {significant}")
        
        self.results['model_comparison'] = result
        return result
    
    # ========================================================================
    # PREDICTION INTERVALS
    # ========================================================================
    
    def bootstrap_prediction_intervals(self,
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_test: np.ndarray,
                                      model_func: Callable) -> Dict:
        """
        Bootstrap prediction intervals
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            model_func: Function that trains model and returns predictions
                       Signature: model_func(X_train, y_train, X_test) -> y_pred
            
        Returns:
            dict with prediction intervals for each test point
        """
        logger.info(f"\n-> Bootstrap prediction intervals ({len(X_test)} test points)")
        
        n_train = len(X_train)
        n_test = len(X_test)
        
        # Bootstrap predictions
        bootstrap_preds = np.zeros((self.n_bootstrap, n_test))
        
        for i in range(self.n_bootstrap):
            # Resample training data
            indices = np.random.choice(n_train, size=n_train, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train and predict
            y_pred = model_func(X_boot, y_boot, X_test)
            bootstrap_preds[i] = y_pred
        
        # Point predictions (median) and intervals
        point_preds = np.median(bootstrap_preds, axis=0)
        ci_lower = np.percentile(bootstrap_preds, (self.alpha/2) * 100, axis=0)
        ci_upper = np.percentile(bootstrap_preds, (1 - self.alpha/2) * 100, axis=0)
        
        result = {
            'n_test': n_test,
            'point_predictions': point_preds,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'bootstrap_predictions': bootstrap_preds
        }
        
        logger.info(f"  Average CI width: {np.mean(ci_upper - ci_lower):.4f}")
        
        self.results['prediction_intervals'] = result
        return result
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_bootstrap_distribution(self,
                                    bootstrap_dist: np.ndarray,
                                    point_estimate: float,
                                    ci_lower: float,
                                    ci_upper: float,
                                    title: str = 'Bootstrap Distribution',
                                    xlabel: str = 'Statistic') -> Path:
        """Plot bootstrap distribution with CI"""
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return None
        
        logger.info(f"\n-> Creating bootstrap distribution plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Histogram
        ax.hist(bootstrap_dist, bins=50, alpha=0.7, color='steelblue',
               edgecolor='black', density=True, label='Bootstrap Distribution')
        
        # Point estimate
        ax.axvline(point_estimate, color='red', linestyle='-', linewidth=2.5,
                  label=f'Point Estimate: {point_estimate:.4f}')
        
        # Confidence interval
        ax.axvline(ci_lower, color='green', linestyle='--', linewidth=2,
                  label=f'{self.confidence_level*100}% CI Lower: {ci_lower:.4f}')
        ax.axvline(ci_upper, color='green', linestyle='--', linewidth=2,
                  label=f'{self.confidence_level*100}% CI Upper: {ci_upper:.4f}')
        
        # Fill CI region
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='green')
        
        # Overlay normal distribution
        x_range = np.linspace(bootstrap_dist.min(), bootstrap_dist.max(), 100)
        normal_dist = stats.norm.pdf(x_range, np.mean(bootstrap_dist), np.std(bootstrap_dist))
        ax.plot(x_range, normal_dist, 'k--', linewidth=2, alpha=0.5,
               label='Normal Approximation')
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'bootstrap_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def export_to_excel(self, filename: str = 'bootstrap_results.xlsx') -> Path:
        """Export bootstrap results to Excel"""
        logger.info(f"\n-> Exporting to {filename}...")
        
        try:
            import xlsxwriter
        except ImportError:
            logger.error("  xlsxwriter not available")
            return None
        
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = []
            
            for key, result in self.results.items():
                if isinstance(result, dict):
                    if 'point_estimate' in result:
                        summary_data.append({
                            'Analysis': key,
                            'Point_Estimate': result['point_estimate'],
                            'CI_Lower': result['ci_lower'],
                            'CI_Upper': result['ci_upper'],
                            'CI_Width': result['ci_width']
                        })
            
            if summary_data:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Model performance sheet
            if 'model_performance' in self.results:
                perf_data = []
                for metric, values in self.results['model_performance'].items():
                    perf_data.append({
                        'Metric': metric,
                        'Point_Estimate': values['point_estimate'],
                        'CI_Lower': values['ci_lower'],
                        'CI_Upper': values['ci_upper'],
                        'Bootstrap_Std': values['bootstrap_std']
                    })
                
                pd.DataFrame(perf_data).to_excel(writer, sheet_name='Model_Performance', index=False)
            
            # Model comparison sheet
            if 'model_comparison' in self.results:
                comp = self.results['model_comparison']
                comp_data = pd.DataFrame([{
                    'Model_A': comp['model_a'],
                    'Model_B': comp['model_b'],
                    'Metric': comp['metric'],
                    'Metric_A': comp['metric_a'],
                    'Metric_B': comp['metric_b'],
                    'Difference': comp['difference'],
                    'CI_Lower': comp['ci_lower'],
                    'CI_Upper': comp['ci_upper'],
                    'P_Value': comp['p_value'],
                    'Significant': comp['significant']
                }])
                comp_data.to_excel(writer, sheet_name='Model_Comparison', index=False)
        
        logger.info(f"  ✓ Exported to: {filepath}")
        return filepath


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("TESTING BOOTSTRAP CONFIDENCE INTERVALS")
    logger.info("="*70)
    
    np.random.seed(42)
    
    # Initialize
    bootstrap = BootstrapConfidenceIntervals(n_bootstrap=10000, output_dir='test_bootstrap_results')
    
    # Test 1: CI for mean
    data = np.random.normal(5.0, 2.0, 100)
    result = bootstrap.bootstrap_statistic(data, statistic=np.mean, method='bca')
    
    # Plot
    if PLOTTING_AVAILABLE:
        bootstrap.plot_bootstrap_distribution(
            result['bootstrap_distribution'],
            result['point_estimate'],
            result['ci_lower'],
            result['ci_upper'],
            title='Bootstrap CI for Mean',
            xlabel='Mean Value'
        )
    
    # Test 2: Model performance
    y_true = np.random.randn(100) * 2 + 5
    y_pred = y_true + np.random.randn(100) * 0.3
    
    bootstrap.bootstrap_model_performance(y_true, y_pred, metrics=['r2', 'rmse', 'mae'])
    
    # Test 3: Model comparison
    y_pred_a = y_true + np.random.randn(100) * 0.3
    y_pred_b = y_true + np.random.randn(100) * 0.25  # Slightly better
    
    bootstrap.bootstrap_model_comparison(y_true, y_pred_a, y_pred_b, 
                                        model_a_name='RF', model_b_name='XGBoost')
    
    # Export
    bootstrap.export_to_excel()
    
    logger.info("\n✓ Testing complete! Check test_bootstrap_results/")
