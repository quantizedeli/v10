"""
Model Validation Module
Cross-validation & Robustness Testing

Özellikler:
1. K-Fold Cross-Validation
2. Stratified CV (for regression)
3. Time Series CV
4. Learning Curves
5. Validation Curves
6. Robustness Tests:
   - Noise injection
   - Outlier sensitivity
   - Feature perturbation
   - Data subset testing

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold,
    learning_curve, validation_curve
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CROSS-VALIDATION ANALYZER
# ============================================================================

class CrossValidationAnalyzer:
    """
    K-Fold Cross-Validation Analysis
    """
    
    def __init__(self, model, model_name, output_dir='validation'):
        self.model = model
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cv_results = None
        
        logger.info(f"CV Analyzer başlatıldı: {model_name}")
    
    def run_cv(self, X, y, cv=5, scoring=None, return_train_score=True, n_jobs=1):
        """
        Cross-validation çalıştır

        Args:
            cv: Fold sayısı veya CV object
            scoring: Metrics list
            return_train_score: Training scores'ları da hesapla
            n_jobs: Number of parallel jobs (default=1 to avoid nested parallelization deadlock)
                    Use n_jobs=-1 only for sequential training mode
        """

        if scoring is None:
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

        logger.info(f"\n{self.model_name} - {cv}-Fold Cross-Validation")
        logger.info(f"Scoring: {scoring}")
        logger.info(f"n_jobs: {n_jobs} (1=sequential CV to avoid deadlock in parallel training)")

        # Run CV
        self.cv_results = cross_validate(
            self.model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            n_jobs=n_jobs,  # FIXED: Use n_jobs parameter instead of hardcoded -1
            verbose=1
        )
        
        # Calculate statistics
        results_summary = {}
        
        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            # Test scores
            test_scores = self.cv_results[test_key]
            results_summary[f'{metric}_test_mean'] = test_scores.mean()
            results_summary[f'{metric}_test_std'] = test_scores.std()
            
            # Train scores
            if return_train_score:
                train_scores = self.cv_results[train_key]
                results_summary[f'{metric}_train_mean'] = train_scores.mean()
                results_summary[f'{metric}_train_std'] = train_scores.std()
            
            logger.info(f"  {metric}: {test_scores.mean():.4f} (±{test_scores.std():.4f})")
        
        # Save results
        self._save_results(results_summary)
        
        # Plot
        self._plot_cv_results(scoring)
        
        return results_summary
    
    def _save_results(self, results_summary):
        """Save CV results"""
        
        # Summary JSON
        with open(self.output_dir / 'cv_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Detailed CSV
        cv_df = pd.DataFrame(self.cv_results)
        cv_df.to_csv(self.output_dir / 'cv_detailed.csv', index=False)
        
        logger.info(f"[OK] CV results saved: {self.output_dir}")
    
    def _plot_cv_results(self, scoring):
        """Plot CV results"""
        
        n_metrics = len(scoring)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(scoring):
            ax = axes[idx]
            
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            test_scores = self.cv_results[test_key]
            
            # Box plot
            data_to_plot = [test_scores]
            labels = ['Test']
            
            if train_key in self.cv_results:
                train_scores = self.cv_results[train_key]
                data_to_plot.append(train_scores)
                labels.append('Train')
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Colors
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric}')
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'{self.model_name} - Cross-Validation Results', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / 'cv_boxplots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] CV plot saved: {save_path}")


# ============================================================================
# LEARNING CURVE ANALYZER
# ============================================================================

class LearningCurveAnalyzer:
    """
    Learning Curve Analysis
    
    Training size vs performance
    """
    
    def __init__(self, model, model_name, output_dir='validation'):
        self.model = model
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_sizes = None
        self.train_scores = None
        self.test_scores = None
        
        logger.info(f"Learning Curve Analyzer: {model_name}")
    
    def compute_learning_curve(self, X, y, cv=5, train_sizes=None, scoring='r2'):
        """
        Learning curve hesapla
        
        Args:
            train_sizes: Training sizes (fractions or absolute)
        """
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        logger.info(f"\nLearning curve hesaplanıyor: {self.model_name}")
        
        self.train_sizes, self.train_scores, self.test_scores = learning_curve(
            self.model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Plot
        self._plot_learning_curve()
        
        # Save
        self._save_results()
        
        logger.info("[OK] Learning curve tamamlandı")
    
    def _plot_learning_curve(self):
        """Plot learning curve"""
        
        train_mean = self.train_scores.mean(axis=1)
        train_std = self.train_scores.std(axis=1)
        test_mean = self.test_scores.mean(axis=1)
        test_std = self.test_scores.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Train curve
        ax.plot(self.train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax.fill_between(self.train_sizes, 
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1, color='r')
        
        # Test curve
        ax.plot(self.train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        ax.fill_between(self.train_sizes,
                        test_mean - test_std,
                        test_mean + test_std,
                        alpha=0.1, color='g')
        
        ax.set_xlabel('Training Size', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.model_name} - Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'learning_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Learning curve plot: {save_path}")
    
    def _save_results(self):
        """Save learning curve data"""
        
        df = pd.DataFrame({
            'train_size': self.train_sizes,
            'train_score_mean': self.train_scores.mean(axis=1),
            'train_score_std': self.train_scores.std(axis=1),
            'test_score_mean': self.test_scores.mean(axis=1),
            'test_score_std': self.test_scores.std(axis=1)
        })
        
        save_path = self.output_dir / 'learning_curve_data.csv'
        df.to_csv(save_path, index=False)
        
        logger.info(f"[OK] Learning curve data: {save_path}")


# ============================================================================
# VALIDATION CURVE ANALYZER
# ============================================================================

class ValidationCurveAnalyzer:
    """
    Validation Curve Analysis
    
    Hyperparameter vs performance
    """
    
    def __init__(self, model, model_name, output_dir='validation'):
        self.model = model
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Validation Curve Analyzer: {model_name}")
    
    def compute_validation_curve(self, X, y, param_name, param_range, cv=5, scoring='r2'):
        """
        Validation curve hesapla
        
        Args:
            param_name: Parameter name
            param_range: Parameter values to test
        """
        
        logger.info(f"\nValidation curve: {param_name}")
        
        train_scores, test_scores = validation_curve(
            self.model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Plot
        self._plot_validation_curve(param_name, param_range, train_scores, test_scores)
        
        # Save
        self._save_results(param_name, param_range, train_scores, test_scores)
        
        logger.info("[OK] Validation curve tamamlandı")
    
    def _plot_validation_curve(self, param_name, param_range, train_scores, test_scores):
        """Plot validation curve"""
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(param_range, train_mean, 'o-', color='r', label='Training score')
        ax.fill_between(param_range,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1, color='r')
        
        ax.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')
        ax.fill_between(param_range,
                        test_mean - test_std,
                        test_mean + test_std,
                        alpha=0.1, color='g')
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{self.model_name} - Validation Curve: {param_name}', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'validation_curve_{param_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Validation curve plot: {save_path}")
    
    def _save_results(self, param_name, param_range, train_scores, test_scores):
        """Save validation curve data"""
        
        df = pd.DataFrame({
            param_name: param_range,
            'train_score_mean': train_scores.mean(axis=1),
            'train_score_std': train_scores.std(axis=1),
            'test_score_mean': test_scores.mean(axis=1),
            'test_score_std': test_scores.std(axis=1)
        })
        
        save_path = self.output_dir / f'validation_curve_{param_name}.csv'
        df.to_csv(save_path, index=False)
        
        logger.info(f"[OK] Validation curve data: {save_path}")


# ============================================================================
# ROBUSTNESS TESTER
# ============================================================================

class RobustnessTester:
    """
    Model Robustness Testing
    
    Tests:
    1. Noise injection
    2. Outlier sensitivity
    3. Feature perturbation
    4. Data subset testing
    """
    
    def __init__(self, model, model_name, output_dir='robustness'):
        self.model = model
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        
        logger.info(f"Robustness Tester: {model_name}")
    
    def test_noise_sensitivity(self, X_test, y_test, noise_levels=[0.01, 0.05, 0.1, 0.2, 0.5]):
        """
        Gaussian noise injection test
        
        Args:
            noise_levels: Noise levels (as fraction of std)
        """
        
        logger.info("\n-> Noise Sensitivity Test")
        
        results = {
            'noise_level': [],
            'r2': [],
            'rmse': [],
            'mae': []
        }
        
        # Baseline (no noise)
        y_pred_baseline = self.model.predict(X_test)
        baseline_r2 = r2_score(y_test, y_pred_baseline)
        
        results['noise_level'].append(0.0)
        results['r2'].append(baseline_r2)
        results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
        results['mae'].append(mean_absolute_error(y_test, y_pred_baseline))
        
        # With noise
        X_std = X_test.std(axis=0)
        
        for noise_level in noise_levels:
            # Add noise
            noise = np.random.randn(*X_test.shape) * X_std * noise_level
            X_noisy = X_test + noise
            
            # Predict
            y_pred = self.model.predict(X_noisy)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results['noise_level'].append(noise_level)
            results['r2'].append(r2)
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            
            logger.info(f"  Noise {noise_level:.2f}: R²={r2:.4f} (Δ={r2-baseline_r2:+.4f})")
        
        self.test_results['noise_sensitivity'] = results
        
        # Plot
        self._plot_noise_sensitivity(results)
        
        # Save
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'noise_sensitivity.csv', index=False)
        
        logger.info("[OK] Noise sensitivity test tamamlandı")
    
    def test_outlier_sensitivity(self, X_train, y_train, X_test, y_test, 
                                 outlier_fractions=[0.01, 0.05, 0.1]):
        """
        Outlier injection test
        
        Args:
            outlier_fractions: Fraction of samples to corrupt
        """
        
        logger.info("\n-> Outlier Sensitivity Test")
        
        results = {
            'outlier_fraction': [],
            'r2': [],
            'rmse': [],
            'mae': []
        }
        
        # Baseline (no outliers)
        self.model.fit(X_train, y_train)
        y_pred_baseline = self.model.predict(X_test)
        baseline_r2 = r2_score(y_test, y_pred_baseline)
        
        results['outlier_fraction'].append(0.0)
        results['r2'].append(baseline_r2)
        results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred_baseline)))
        results['mae'].append(mean_absolute_error(y_test, y_pred_baseline))
        
        # With outliers
        for outlier_frac in outlier_fractions:
            # Inject outliers in training data
            X_train_corrupt = X_train.copy()
            y_train_corrupt = y_train.copy()
            
            n_outliers = int(len(X_train) * outlier_frac)
            outlier_indices = np.random.choice(len(X_train), n_outliers, replace=False)
            
            # Corrupt features (extreme values)
            X_train_corrupt[outlier_indices] = np.random.randn(n_outliers, X_train.shape[1]) * 10
            
            # Corrupt targets
            y_train_corrupt[outlier_indices] = np.random.randn(n_outliers) * 10
            
            # Retrain
            from copy import deepcopy
            model_copy = deepcopy(self.model)
            model_copy.fit(X_train_corrupt, y_train_corrupt)
            
            # Predict
            y_pred = model_copy.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results['outlier_fraction'].append(outlier_frac)
            results['r2'].append(r2)
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            
            logger.info(f"  Outliers {outlier_frac:.2%}: R²={r2:.4f} (Δ={r2-baseline_r2:+.4f})")
        
        self.test_results['outlier_sensitivity'] = results
        
        # Plot
        self._plot_outlier_sensitivity(results)
        
        # Save
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'outlier_sensitivity.csv', index=False)
        
        logger.info("[OK] Outlier sensitivity test tamamlandı")
    
    def test_feature_perturbation(self, X_test, y_test, feature_names=None, 
                                  perturbation_levels=[0.05, 0.1, 0.2]):
        """
        Feature perturbation test
        
        Test each feature's importance by perturbing it
        """
        
        logger.info("\n-> Feature Perturbation Test")
        
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(X_test.shape[1])]
        
        # Baseline
        y_pred_baseline = self.model.predict(X_test)
        baseline_r2 = r2_score(y_test, y_pred_baseline)
        
        results = {
            'feature': [],
            'perturbation_level': [],
            'r2': [],
            'r2_drop': []
        }
        
        for feature_idx, feature_name in enumerate(feature_names):
            feature_std = X_test[:, feature_idx].std()
            
            for perturb_level in perturbation_levels:
                # Perturb feature
                X_perturbed = X_test.copy()
                X_perturbed[:, feature_idx] += np.random.randn(len(X_test)) * feature_std * perturb_level
                
                # Predict
                y_pred = self.model.predict(X_perturbed)
                r2 = r2_score(y_test, y_pred)
                r2_drop = baseline_r2 - r2
                
                results['feature'].append(feature_name)
                results['perturbation_level'].append(perturb_level)
                results['r2'].append(r2)
                results['r2_drop'].append(r2_drop)
        
        self.test_results['feature_perturbation'] = results
        
        # Plot
        self._plot_feature_perturbation(results, perturbation_levels)
        
        # Save
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'feature_perturbation.csv', index=False)
        
        logger.info("[OK] Feature perturbation test tamamlandı")
    
    def _plot_noise_sensitivity(self, results):
        """Plot noise sensitivity"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² vs Noise
        axes[0].plot(results['noise_level'], results['r2'], 'o-', linewidth=2)
        axes[0].set_xlabel('Noise Level', fontsize=12)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² vs Noise Level')
        axes[0].grid(alpha=0.3)
        
        # RMSE vs Noise
        axes[1].plot(results['noise_level'], results['rmse'], 'o-', linewidth=2, color='orange')
        axes[1].set_xlabel('Noise Level', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE vs Noise Level')
        axes[1].grid(alpha=0.3)
        
        # MAE vs Noise
        axes[2].plot(results['noise_level'], results['mae'], 'o-', linewidth=2, color='green')
        axes[2].set_xlabel('Noise Level', fontsize=12)
        axes[2].set_ylabel('MAE', fontsize=12)
        axes[2].set_title('MAE vs Noise Level')
        axes[2].grid(alpha=0.3)
        
        fig.suptitle(f'{self.model_name} - Noise Sensitivity', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'noise_sensitivity_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Noise plot: {save_path}")
    
    def _plot_outlier_sensitivity(self, results):
        """Plot outlier sensitivity"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² vs Outliers
        axes[0].plot(results['outlier_fraction'], results['r2'], 'o-', linewidth=2)
        axes[0].set_xlabel('Outlier Fraction', fontsize=12)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² vs Outlier Fraction')
        axes[0].grid(alpha=0.3)
        
        # RMSE
        axes[1].plot(results['outlier_fraction'], results['rmse'], 'o-', linewidth=2, color='orange')
        axes[1].set_xlabel('Outlier Fraction', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE vs Outlier Fraction')
        axes[1].grid(alpha=0.3)
        
        # MAE
        axes[2].plot(results['outlier_fraction'], results['mae'], 'o-', linewidth=2, color='green')
        axes[2].set_xlabel('Outlier Fraction', fontsize=12)
        axes[2].set_ylabel('MAE', fontsize=12)
        axes[2].set_title('MAE vs Outlier Fraction')
        axes[2].grid(alpha=0.3)
        
        fig.suptitle(f'{self.model_name} - Outlier Sensitivity', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'outlier_sensitivity_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Outlier plot: {save_path}")
    
    def _plot_feature_perturbation(self, results, perturbation_levels):
        """Plot feature perturbation"""
        
        df = pd.DataFrame(results)
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            values='r2_drop',
            index='feature',
            columns='perturbation_level',
            aggfunc='mean'
        )
        
        # Sort by max drop
        pivot['max_drop'] = pivot.max(axis=1)
        pivot = pivot.sort_values('max_drop', ascending=False)
        pivot = pivot.drop('max_drop', axis=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot)*0.3)))
        
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', 
                    cbar_kws={'label': 'R² Drop'}, ax=ax)
        
        ax.set_title(f'{self.model_name} - Feature Perturbation Sensitivity', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Perturbation Level', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / 'feature_perturbation_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Feature perturbation plot: {save_path}")
    
    def generate_report(self):
        """Generate comprehensive robustness report"""
        
        report = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'tests_completed': list(self.test_results.keys())
        }
        
        # Add summaries
        if 'noise_sensitivity' in self.test_results:
            ns = self.test_results['noise_sensitivity']
            report['noise_max_drop'] = float(max(ns['r2']) - min(ns['r2']))
        
        if 'outlier_sensitivity' in self.test_results:
            os = self.test_results['outlier_sensitivity']
            report['outlier_max_drop'] = float(max(os['r2']) - min(os['r2']))
        
        # Save
        with open(self.output_dir / 'robustness_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[OK] Robustness report: {self.output_dir / 'robustness_report.json'}")


# ============================================================================
# UNIFIED MODEL VALIDATOR
# ============================================================================

class ModelValidator:
    """
    Unified model validation
    
    Combines all validation methods
    """
    
    def __init__(self, model, model_name, output_dir='validation_results'):
        self.model = model
        self.model_name = model_name
        self.output_dir = Path(output_dir) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model Validator: {model_name}")
    
    def validate_all(self, X_train, y_train, X_test, y_test, 
                    feature_names=None, run_robustness=True):
        """
        Tüm validation testleri
        """
        
        logger.info("\n" + "="*80)
        logger.info(f"MODEL VALIDATION: {self.model_name}")
        logger.info("="*80)
        
        # 1. Cross-Validation
        logger.info("\n-> CROSS-VALIDATION")
        cv_analyzer = CrossValidationAnalyzer(self.model, self.model_name, self.output_dir)
        cv_results = cv_analyzer.run_cv(X_train, y_train, cv=5)
        
        # 2. Learning Curve
        logger.info("\n-> LEARNING CURVE")
        lc_analyzer = LearningCurveAnalyzer(self.model, self.model_name, self.output_dir)
        lc_analyzer.compute_learning_curve(X_train, y_train, cv=5)
        
        # 3. Robustness Tests
        if run_robustness:
            logger.info("\n-> ROBUSTNESS TESTS")
            robustness_tester = RobustnessTester(self.model, self.model_name, self.output_dir / 'robustness')
            
            robustness_tester.test_noise_sensitivity(X_test, y_test)
            robustness_tester.test_outlier_sensitivity(X_train, y_train, X_test, y_test)
            robustness_tester.test_feature_perturbation(X_test, y_test, feature_names)
            robustness_tester.generate_report()
        
        logger.info("\n" + "="*80)
        logger.info("[OK] VALIDATION TAMAMLANDI")
        logger.info("="*80)


# ============================================================================
# MAIN TEST
# ============================================================================

def test_model_validation():
    """Test model validation"""
    
    print("\n" + "="*80)
    print("MODEL VALIDATION TEST")
    print("="*80)
    
    # Dummy data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * (-1) + np.random.randn(n_samples) * 0.1
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Split
    train_size = int(0.7 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Train model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Validate
    validator = ModelValidator(model, 'RandomForest', output_dir='test_validation')
    validator.validate_all(X_train, y_train, X_test, y_test, feature_names, run_robustness=True)
    
    print("\n[OK] Test tamamlandı!")


if __name__ == "__main__":
    test_model_validation()