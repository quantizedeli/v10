"""
MASTER VISUALIZATION SYSTEM
============================

Tez için Kritik Görselleştirmeler - Kapsamlı Modüler Sistem

1. ROBUSTNESS TESTING VISUALIZATIONS
2. SHAP (Shapley Additive Explanations) 
3. ANOMALY KERNEL FEATURES ANALYSIS
4. MASTER REPORT VISUALIZATIONS
5. DATASET RESULTS VISUALIZATIONS (AI + ANFIS)
6. PREDICTION VISUALIZATIONS
7. MODEL COMPARISON DASHBOARDS
8. FEATURE IMPORTANCE ANALYSIS
9. TRAINING METRICS VISUALIZATIONS
10. OPTIMIZATION METRICS VISUALIZATIONS
11. DATA CATALOG VISUALIZATIONS
12. REPORTS VISUALIZATIONS
13. TEST PREDICTIONS & AAA2 RESULTS
14. LOG ANALYTICS VISUALIZATIONS
15. ADVANCED ANALYSIS (3D, Clustering, etc.)

Author: Nuclear Physics AI Project
Date: October 2025
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Scipy for advanced analysis
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Matplotlib 3D
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PLOT_CONFIG = {
    'dpi': 300,
    'figsize_default': (14, 10),
    'figsize_small': (10, 6),
    'figsize_large': (18, 12),
    'style': 'seaborn-v0_8-darkgrid',
    'colormap': 'Set2'
}


# =============================================================================
# 1. ROBUSTNESS TESTING VISUALIZATIONS
# =============================================================================

class RobustnessVisualizer:
    """Sağlamlık testlerinin görselleştirmesi"""
    
    def __init__(self, output_dir='visualizations/robustness'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
        
    def plot_perturbation_sensitivity(self, 
                                      predictions: pd.DataFrame,
                                      perturbation_results: Dict,
                                      save_name: str = 'perturbation_sensitivity'):
        """Perturbation analizi"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Perturbation Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Mean absolute change
        ax = axes[0, 0]
        features = list(perturbation_results.keys())
        mean_changes = [perturbation_results[f]['mean_abs_change'] for f in features]
        ax.barh(features, mean_changes, color='steelblue')
        ax.set_xlabel('Mean Absolute Change')
        ax.set_title('Average Impact per Feature')
        ax.grid(True, alpha=0.3)
        
        # 2. Max deviation
        ax = axes[0, 1]
        max_devs = [perturbation_results[f]['max_deviation'] for f in features]
        ax.barh(features, max_devs, color='coral')
        ax.set_xlabel('Maximum Deviation')
        ax.set_title('Maximum Impact per Feature')
        ax.grid(True, alpha=0.3)
        
        # 3. Robustness score
        ax = axes[1, 0]
        robustness_scores = [perturbation_results[f]['robustness_score'] for f in features]
        colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in robustness_scores]
        ax.barh(features, robustness_scores, color=colors)
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='High Threshold')
        ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium Threshold')
        ax.set_xlabel('Robustness Score')
        ax.set_title('Robustness Scores by Feature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Sensitivity distribution
        ax = axes[1, 1]
        sensitivities = []
        for f in features:
            sensitivities.extend(perturbation_results[f]['sensitivity_distribution'])
        ax.hist(sensitivities, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Sensitivity Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Sensitivity Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_noise_robustness(self,
                              original_pred: np.ndarray,
                              noisy_predictions: Dict[str, np.ndarray],
                              save_name: str = 'noise_robustness'):
        """Gürültü karşı dayanıklılık"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Noise Robustness Analysis', fontsize=16, fontweight='bold')
        
        # 1. Error by noise level
        ax = axes[0, 0]
        noise_levels = sorted([k for k in noisy_predictions.keys() if isinstance(k, (int, float))])
        errors = []
        for nl in noise_levels:
            error = np.mean(np.abs(original_pred - noisy_predictions[nl]))
            errors.append(error)
        ax.plot(noise_levels, errors, marker='o', linewidth=2, markersize=8, color='darkblue')
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Prediction Error vs Noise Level')
        ax.grid(True, alpha=0.3)
        
        # 2. R² degradation
        ax = axes[0, 1]
        r2_scores = []
        for nl in noise_levels:
            r2 = 1 - (np.sum((original_pred - noisy_predictions[nl])**2) / 
                      np.sum((original_pred - original_pred.mean())**2))
            r2_scores.append(r2)
        ax.plot(noise_levels, r2_scores, marker='s', linewidth=2, markersize=8, color='darkred')
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Good Threshold')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Degradation with Noise')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([max(0, min(r2_scores)-0.1), 1.05])
        
        # 3. Box plot of errors at each noise level
        ax = axes[1, 0]
        error_data = []
        labels = []
        for nl in noise_levels:
            errors_at_level = np.abs(original_pred - noisy_predictions[nl])
            error_data.append(errors_at_level)
            labels.append(f'{nl:.2f}')
        bp = ax.boxplot(error_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error Distribution at Each Noise Level')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Percentage predictions within tolerance
        ax = axes[1, 1]
        tolerances = [0.01, 0.02, 0.05, 0.1, 0.2]
        for tolerance in tolerances:
            percentages = []
            for nl in noise_levels:
                within = np.sum(np.abs(original_pred - noisy_predictions[nl]) <= tolerance) / len(original_pred) * 100
                percentages.append(within)
            ax.plot(noise_levels, percentages, marker='o', label=f'±{tolerance}', linewidth=2)
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Percentage within tolerance (%)')
        ax.set_title('Robustness: Predictions within Tolerance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_cross_validation_stability(self,
                                        cv_results: Dict[str, List[float]],
                                        save_name: str = 'cv_stability'):
        """Cross-validation stability analizi"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Cross-Validation Stability Analysis', fontsize=16, fontweight='bold')
        
        metrics = list(cv_results.keys())
        
        # 1. Box plot of metrics
        ax = axes[0, 0]
        data_to_plot = [cv_results[m] for m in metrics]
        bp = ax.boxplot(data_to_plot, labels=metrics, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('Score')
        ax.set_title('Metric Distribution across CV Folds')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Mean ± std
        ax = axes[0, 1]
        means = [np.mean(cv_results[m]) for m in metrics]
        stds = [np.std(cv_results[m]) for m in metrics]
        x_pos = np.arange(len(metrics))
        ax.bar(x_pos, means, yerr=stds, capsize=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylabel('Mean Score')
        ax.set_title('Mean Metrics ± Standard Deviation')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Coefficient of variation
        ax = axes[1, 0]
        cvs = []
        for m in metrics:
            mean_val = np.mean(cv_results[m])
            std_val = np.std(cv_results[m])
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0
            cvs.append(cv)
        colors = ['green' if cv < 5 else 'orange' if cv < 10 else 'red' for cv in cvs]
        ax.barh(metrics, cvs, color=colors)
        ax.axvline(x=5, color='green', linestyle='--', alpha=0.5, label='Low Variation')
        ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Medium Variation')
        ax.set_xlabel('Coefficient of Variation (%)')
        ax.set_title('Stability: Coefficient of Variation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Fold-by-fold performance
        ax = axes[1, 1]
        n_folds = len(next(iter(cv_results.values())))
        folds = np.arange(1, n_folds + 1)
        for metric in metrics:
            ax.plot(folds, cv_results[metric], marker='o', label=metric, linewidth=2)
        ax.set_xlabel('Fold Number')
        ax.set_ylabel('Score')
        ax.set_title('Performance Across CV Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 2. SHAP (SHAPLEY ADDITIVE EXPLANATIONS) VISUALIZATIONS
# =============================================================================

class SHAPVisualizer:
    """SHAP açıklayıcılık görselleştirmeleri"""
    
    def __init__(self, output_dir='visualizations/shap'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_shap_summary(self,
                          feature_names: List[str],
                          shap_values: np.ndarray,
                          X: np.ndarray,
                          save_name: str = 'shap_summary'):
        """SHAP summary plot - Beeswarm plot"""
        fig, ax = plt.subplots(figsize=self.config['figsize_large'])
        
        # Sort features by importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_shap = shap_values[:, sorted_idx]
        sorted_X = X[:, sorted_idx]
        
        # Create beeswarm-like plot
        y_pos = np.arange(len(sorted_features))
        
        for i, feature_idx in enumerate(sorted_idx):
            # Get SHAP values and feature values
            shap_vals = sorted_shap[:, i]
            feature_vals = sorted_X[:, i]
            
            # Normalize feature values for color
            if feature_vals.std() > 0:
                colors = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
            else:
                colors = np.ones_like(shap_vals)
            
            # Scatter plot with jitter
            jitter = np.random.normal(0, 0.02, size=len(shap_vals))
            ax.scatter(shap_vals, y_pos[i] + jitter, c=colors, cmap='RdBu_r', 
                      s=30, alpha=0.6, edgecolors='none')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('SHAP Value (impact on model output)')
        ax.set_title('SHAP Summary Plot: Feature Importance & Direction', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_shap_force(self,
                        shap_values: np.ndarray,
                        X: np.ndarray,
                        feature_names: List[str],
                        base_value: float,
                        sample_idx: int = 0,
                        save_name: str = 'shap_force'):
        """SHAP force plot - İnidividual prediction açıklaması"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sample_shap = shap_values[sample_idx]
        sample_X = X[sample_idx]
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:10]  # Top 10
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_shap = sample_shap[sorted_idx]
        sorted_X = sample_X[sorted_idx]
        
        # Create visualization
        y_pos = np.arange(len(sorted_features))
        colors = ['red' if v < 0 else 'blue' for v in sorted_shap]
        
        ax.barh(y_pos, sorted_shap, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{f} = {x:.3f}' for f, x in zip(sorted_features, sorted_X)])
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'SHAP Force Plot: Sample {sample_idx} (Base Value: {base_value:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}_{sample_idx}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}_{sample_idx}.png")
    
    def plot_shap_dependence(self,
                            feature_idx: int,
                            feature_name: str,
                            shap_values: np.ndarray,
                            X: np.ndarray,
                            save_name: str = 'shap_dependence'):
        """SHAP dependence plot"""
        fig, ax = plt.subplots(figsize=self.config['figsize_large'])
        
        feature_vals = X[:, feature_idx]
        shap_vals = shap_values[:, feature_idx]
        
        # Create scatter plot
        scatter = ax.scatter(feature_vals, shap_vals, c=shap_vals, cmap='RdBu_r', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(feature_vals, shap_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(feature_vals.min(), feature_vals.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_xlabel(f'{feature_name} (Feature Value)')
        ax.set_ylabel('SHAP Value')
        ax.set_title(f'SHAP Dependence Plot: {feature_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('SHAP Value')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}_{feature_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}_{feature_name}.png")


# =============================================================================
# 3. ANOMALY KERNEL FEATURES ANALYSIS
# =============================================================================

class AnomalyKernelVisualizer:
    """Anomali çekirdeklerin ortak özellikleri analizi"""
    
    def __init__(self, output_dir='visualizations/anomaly_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_anomaly_characteristics(self,
                                     normal_data: pd.DataFrame,
                                     anomaly_data: pd.DataFrame,
                                     feature_cols: List[str],
                                     save_name: str = 'anomaly_characteristics'):
        """Anomalilerin karakteristik özellikleri"""
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten()
        fig.suptitle('Anomaly Kernel Characteristics Analysis', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(feature_cols):
            ax = axes[idx]
            
            # Histogram comparison
            ax.hist(normal_data[feature].dropna(), bins=30, alpha=0.6, label='Normal', color='blue', edgecolor='black')
            ax.hist(anomaly_data[feature].dropna(), bins=30, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
            
            # Statistics
            normal_mean = normal_data[feature].mean()
            anomaly_mean = anomaly_data[feature].mean()
            normal_std = normal_data[feature].std()
            anomaly_std = anomaly_data[feature].std()
            
            # Plot means
            ax.axvline(normal_mean, color='blue', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvline(anomaly_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            ax.set_title(f'{feature}\nNormal: μ={normal_mean:.3f}±{normal_std:.3f}, Anomaly: μ={anomaly_mean:.3f}±{anomaly_std:.3f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove unused subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_anomaly_clustering(self,
                               data: pd.DataFrame,
                               anomaly_mask: np.ndarray,
                               feature_cols: List[str],
                               n_components: int = 2,
                               save_name: str = 'anomaly_clustering'):
        """Anomalilerin clustering analizi"""
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[feature_cols].fillna(data[feature_cols].mean()))
        
        # PCA for visualization
        if n_components == 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, axes = plt.subplots(1, 2, figsize=self.config['figsize_large'])
            fig.suptitle('Anomaly Clustering Analysis', fontsize=16, fontweight='bold')
            
            # PCA visualization
            ax = axes[0]
            scatter_normal = ax.scatter(X_pca[~anomaly_mask, 0], X_pca[~anomaly_mask, 1],
                                       c='blue', label='Normal', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            scatter_anomaly = ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
                                        c='red', label='Anomaly', alpha=0.8, s=100, marker='^', edgecolors='darkred', linewidth=1)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title(f'PCA Projection (Variance: {pca.explained_variance_ratio_.sum():.1%})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # t-SNE visualization
            ax = axes[1]
            try:
                tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
                X_tsne = tsne.fit_transform(X_scaled)
                
                ax.scatter(X_tsne[~anomaly_mask, 0], X_tsne[~anomaly_mask, 1],
                          c='blue', label='Normal', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                ax.scatter(X_tsne[anomaly_mask, 0], X_tsne[anomaly_mask, 1],
                          c='red', label='Anomaly', alpha=0.8, s=100, marker='^', edgecolors='darkred', linewidth=1)
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_title('t-SNE Projection')
                ax.legend()
                ax.grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"t-SNE failed: {e}")
        
        elif n_components == 3:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle('3D Anomaly Clustering Analysis', fontsize=16, fontweight='bold')
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(X_pca[~anomaly_mask, 0], X_pca[~anomaly_mask, 1], X_pca[~anomaly_mask, 2],
                      c='blue', label='Normal', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], X_pca[anomaly_mask, 2],
                      c='red', label='Anomaly', alpha=0.8, s=100, marker='^', edgecolors='darkred', linewidth=1)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.set_title(f'3D PCA Projection (Variance: {pca.explained_variance_ratio_.sum():.1%})')
            ax.legend()
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_anomaly_common_patterns(self,
                                    anomaly_data: pd.DataFrame,
                                    feature_cols: List[str],
                                    save_name: str = 'anomaly_patterns'):
        """Anomalilerin ortak desenleri"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Common Patterns in Anomaly Kernels', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap (anomalies)
        ax = axes[0, 0]
        corr_matrix = anomaly_data[feature_cols].corr()
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_yticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_yticklabels(feature_cols)
        ax.set_title('Correlation Matrix (Anomalies)')
        plt.colorbar(im, ax=ax)
        
        # 2. Feature distributions comparison
        ax = axes[0, 1]
        feature_means = anomaly_data[feature_cols].mean()
        feature_stds = anomaly_data[feature_cols].std()
        x_pos = np.arange(len(feature_cols))
        ax.bar(x_pos, feature_means, yerr=feature_stds, capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Means ± Std (Anomalies)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Skewness and kurtosis
        ax = axes[1, 0]
        skewness = [stats.skew(anomaly_data[f].dropna()) for f in feature_cols]
        kurtosis = [stats.kurtosis(anomaly_data[f].dropna()) for f in feature_cols]
        x_pos = np.arange(len(feature_cols))
        width = 0.35
        ax.bar(x_pos - width/2, skewness, width, label='Skewness', alpha=0.7, edgecolor='black')
        ax.bar(x_pos + width/2, kurtosis, width, label='Kurtosis', alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Skewness and Kurtosis')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Range analysis
        ax = axes[1, 1]
        feature_ranges = [(anomaly_data[f].max() - anomaly_data[f].min()) for f in feature_cols]
        feature_mins = [anomaly_data[f].min() for f in feature_cols]
        x_pos = np.arange(len(feature_cols))
        ax.bar(x_pos, feature_ranges, bottom=feature_mins, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_ylabel('Range (min to max)')
        ax.set_title('Feature Range Analysis (Anomalies)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 4. MASTER REPORT VISUALIZATIONS
# =============================================================================

class MasterReportVisualizer:
    """Master rapor görselleştirmeleri"""
    
    def __init__(self, output_dir='visualizations/master_report'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def create_comprehensive_dashboard(self,
                                      results_summary: Dict,
                                      model_results: Dict,
                                      save_name: str = 'master_dashboard'):
        """Kapsamlı master dashboard"""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, using matplotlib instead")
            return self._create_matplotlib_dashboard(results_summary, model_results, save_name)
        
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Model Performance', 'Target Comparison', 
                          'Error Distribution', 'R² Scores',
                          'Dataset Sizes', 'Training Time'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                   [{'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Model Performance
        models = list(model_results.keys())
        r2_scores = [model_results[m].get('r2', 0) for m in models]
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name='R² Score', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Save
        fig.write_html(self.output_dir / f'{save_name}.html')
        logger.info(f"✓ Saved: {save_name}.html")
    
    def plot_master_summary_stats(self,
                                  results_df: pd.DataFrame,
                                  save_name: str = 'master_summary'):
        """Master özet istatistikleri"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Master Report: Summary Statistics', fontsize=16, fontweight='bold')
        
        # 1. Model count by type
        ax = axes[0, 0]
        if 'model_type' in results_df.columns:
            model_counts = results_df['model_type'].value_counts()
            ax.barh(model_counts.index, model_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Count')
            ax.set_title('Number of Models by Type')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Performance distribution
        ax = axes[0, 1]
        if 'r2_score' in results_df.columns:
            ax.hist(results_df['r2_score'].dropna(), bins=30, color='green', alpha=0.7, edgecolor='black')
            ax.axvline(results_df['r2_score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(results_df['r2_score'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
            ax.set_xlabel('R² Score')
            ax.set_ylabel('Frequency')
            ax.set_title('R² Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Target distribution
        ax = axes[1, 0]
        if 'target' in results_df.columns:
            target_counts = results_df['target'].value_counts()
            wedges, texts, autotexts = ax.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                                              colors=plt.cm.Set3(np.linspace(0, 1, len(target_counts))))
            ax.set_title('Target Distribution')
        
        # 4. Training statistics
        ax = axes[1, 1]
        if 'training_time' in results_df.columns:
            metrics = ['Mean Time', 'Max Time', 'Min Time']
            times = [
                results_df['training_time'].mean(),
                results_df['training_time'].max(),
                results_df['training_time'].min()
            ]
            colors_bar = ['green', 'red', 'blue']
            ax.bar(metrics, times, color=colors_bar, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time Statistics')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 5. AI & ANFIS PREDICTION VISUALIZATIONS
# =============================================================================

class PredictionVisualizer:
    """AI ve ANFIS tahmin görselleştirmeleri"""
    
    def __init__(self, output_dir='visualizations/predictions'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_prediction_comparison(self,
                                  experimental: np.ndarray,
                                  predictions: Dict[str, np.ndarray],
                                  target_name: str,
                                  save_name: str = 'prediction_comparison'):
        """Model tahminlerinin karşılaştırması"""
        models = list(predictions.keys())
        n_models = len(models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Prediction Comparison: {target_name}', fontsize=16, fontweight='bold')
        
        overall_r2 = []
        overall_mae = []
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            pred = predictions[model]
            
            # Scatter plot
            ax.scatter(experimental, pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(experimental.min(), pred.min())
            max_val = max(experimental.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            # Metrics
            r2 = 1 - (np.sum((experimental - pred)**2) / np.sum((experimental - experimental.mean())**2))
            mae = np.mean(np.abs(experimental - pred))
            rmse = np.sqrt(np.mean((experimental - pred)**2))
            
            overall_r2.append(r2)
            overall_mae.append(mae)
            
            ax.set_xlabel('Experimental')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model}\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([min_val, max_val])
            ax.set_ylim([min_val, max_val])
        
        # Remove unused subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved: {save_name}.png")
        logger.info(f"  Mean R²: {np.mean(overall_r2):.4f}, Mean MAE: {np.mean(overall_mae):.4f}")
    
    def plot_residual_analysis(self,
                              experimental: np.ndarray,
                              predictions: Dict[str, np.ndarray],
                              target_name: str,
                              save_name: str = 'residual_analysis'):
        """Residual analizi"""
        models = list(predictions.keys())
        n_models = len(models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Residual Analysis: {target_name}', fontsize=16, fontweight='bold')
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            pred = predictions[model]
            residuals = experimental - pred
            
            # Histogram + KDE
            ax.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(residuals)
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            # Normal distribution
            mu, sigma = residuals.mean(), residuals.std()
            ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'g--', linewidth=2, label='Normal')
            
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Density')
            ax.set_title(f'{model}\nMean={mu:.4f}, Std={sigma:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove unused subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_error_distribution_3d(self,
                                   experimental: np.ndarray,
                                   predictions: Dict[str, np.ndarray],
                                   X: np.ndarray,
                                   feature_idx_1: int = 0,
                                   feature_idx_2: int = 1,
                                   save_name: str = 'error_3d'):
        """3D hata dağılımı"""
        fig = plt.figure(figsize=(16, 12))
        
        models = list(predictions.keys())
        n_models = len(models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        for idx, model in enumerate(models):
            ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
            
            pred = predictions[model]
            errors = np.abs(experimental - pred)
            
            scatter = ax.scatter(X[:, feature_idx_1], X[:, feature_idx_2], errors,
                               c=errors, cmap='RdYlGn_r', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(f'Feature {feature_idx_1}')
            ax.set_ylabel(f'Feature {feature_idx_2}')
            ax.set_zlabel('Absolute Error')
            ax.set_title(f'{model}\nMean Error: {errors.mean():.4f}')
            
            plt.colorbar(scatter, ax=ax, label='Error')
        
        fig.suptitle('3D Error Distribution', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 6. MODEL COMPARISON VISUALIZATIONS
# =============================================================================

class ModelComparisonVisualizer:
    """Model karşılaştırma görselleştirmeleri"""
    
    def __init__(self, output_dir='visualizations/model_comparison'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_model_ranking(self,
                          models_metrics: Dict[str, Dict[str, float]],
                          metrics: List[str] = None,
                          save_name: str = 'model_ranking'):
        """Model ranking görselleştirmesi"""
        if metrics is None:
            metrics = ['r2_score', 'mae', 'rmse']
        
        models = list(models_metrics.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Model Ranking Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = []
            for model in models:
                val = models_metrics[model].get(metric, 0)
                values.append(val)
            
            # Sort
            sorted_idx = np.argsort(values)[::-1]
            sorted_models = [models[i] for i in sorted_idx]
            sorted_values = [values[i] for i in sorted_idx]
            
            # Color gradient
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_models)))
            
            ax.barh(sorted_models, sorted_values, color=colors, edgecolor='black', linewidth=1)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Ranking')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add values on bars
            for i, v in enumerate(sorted_values):
                ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_parallel_coordinates(self,
                                 models_metrics: Dict[str, Dict[str, float]],
                                 save_name: str = 'parallel_coordinates'):
        """Parallel coordinates plot"""
        metrics_list = list(next(iter(models_metrics.values())).keys())
        models = list(models_metrics.keys())
        
        # Normalize metrics to 0-1
        normalized_data = {}
        for metric in metrics_list:
            values = [models_metrics[m].get(metric, 0) for m in models]
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized_data[metric] = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_data[metric] = [0.5] * len(values)
        
        fig, ax = plt.subplots(figsize=self.config['figsize_large'])
        
        # Plot lines
        for i, model in enumerate(models):
            values = [normalized_data[m][i] for m in metrics_list]
            x = np.arange(len(metrics_list))
            ax.plot(x, values, marker='o', label=model, linewidth=2, markersize=8, alpha=0.7)
        
        ax.set_xticks(np.arange(len(metrics_list)))
        ax.set_xticklabels(metrics_list, rotation=45, ha='right')
        ax.set_ylabel('Normalized Score')
        ax.set_ylim([0, 1])
        ax.set_title('Parallel Coordinates: Model Metrics Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 7. TRAINING METRICS VISUALIZATIONS
# =============================================================================

class TrainingMetricsVisualizer:
    """Eğitim metriklerinin görselleştirmesi"""
    
    def __init__(self, output_dir='visualizations/training_metrics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_training_curves(self,
                            history: Dict[str, List[float]],
                            save_name: str = 'training_curves'):
        """Eğitim eğrileri"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Training Curves Analysis', fontsize=16, fontweight='bold')
        
        epochs = np.arange(1, len(next(iter(history.values()))) + 1)
        
        # 1. Loss
        ax = axes[0, 0]
        if 'train_loss' in history and 'val_loss' in history:
            ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # 2. R² Score
        ax = axes[0, 1]
        if 'train_r2' in history and 'val_r2' in history:
            ax.plot(epochs, history['train_r2'], label='Train R²', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, history['val_r2'], label='Val R²', linewidth=2, marker='s', markersize=4)
            ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Target (0.95)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('R² Score')
            ax.set_title('R² Score vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # 3. MAE
        ax = axes[1, 0]
        if 'train_mae' in history and 'val_mae' in history:
            ax.plot(epochs, history['train_mae'], label='Train MAE', linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, history['val_mae'], label='Val MAE', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE')
            ax.set_title('MAE vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Learning rate (if available)
        ax = axes[1, 1]
        if 'learning_rate' in history:
            ax.plot(epochs, history['learning_rate'], label='Learning Rate', linewidth=2, marker='o', markersize=4, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_training_convergence(self,
                                 history_dict: Dict[str, Dict[str, List[float]]],
                                 save_name: str = 'convergence_analysis'):
        """Eğitim yakınsaması analizi"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Training time per epoch
        ax = axes[0, 0]
        models = list(history_dict.keys())
        if all('epoch_time' in history_dict[m] for m in models):
            for model in models:
                epochs_to_plot = min(100, len(history_dict[model]['epoch_time']))
                ax.plot(range(1, epochs_to_plot+1), history_dict[model]['epoch_time'][:epochs_to_plot],
                       label=model, linewidth=2, marker='o', markersize=3, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (s)')
            ax.set_title('Training Time per Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Convergence speed
        ax = axes[0, 1]
        for model in models:
            if 'val_loss' in history_dict[model]:
                losses = history_dict[model]['val_loss']
                # Find convergence point (when loss plateaus)
                threshold = losses[-1] * 1.05
                converged_at = next((i for i, l in enumerate(losses) if l < threshold), len(losses))
                ax.scatter(converged_at, losses[converged_at], s=200, marker='*', label=model, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Convergence Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Improvement rate
        ax = axes[1, 0]
        for model in models:
            if 'val_r2' in history_dict[model]:
                r2_scores = history_dict[model]['val_r2']
                improvements = [r2_scores[0]]
                for i in range(1, len(r2_scores)):
                    improvements.append(r2_scores[i] - r2_scores[i-1])
                ax.plot(range(1, len(improvements)+1), improvements, label=model, linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Improvement')
        ax.set_title('Per-Epoch Improvement Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Final metrics comparison
        ax = axes[1, 1]
        final_r2 = [history_dict[m].get('val_r2', [0])[-1] for m in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        ax.barh(models, final_r2, color=colors, edgecolor='black', linewidth=1)
        ax.set_xlabel('Final R² Score')
        ax.set_title('Final Performance Comparison')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 1])
        for i, v in enumerate(final_r2):
            ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 8. OPTIMIZATION METRICS VISUALIZATIONS
# =============================================================================

class OptimizationMetricsVisualizer:
    """Optimizasyon metriklerinin görselleştirmesi"""
    
    def __init__(self, output_dir='visualizations/optimization'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_hyperparameter_importance(self,
                                      hyperparams: Dict[str, List[float]],
                                      scores: np.ndarray,
                                      save_name: str = 'hyperparam_importance'):
        """Hiperparametre önem analizi"""
        n_params = len(hyperparams)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]
        fig.suptitle('Hyperparameter Importance Analysis', fontsize=16, fontweight='bold')
        
        for idx, (param_name, values) in enumerate(hyperparams.items()):
            ax = axes[idx]
            
            scatter = ax.scatter(values, scores, c=scores, cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Trend line
            z = np.polyfit(values, scores, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(values), max(values), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Score')
            ax.set_title(f'{param_name} vs Score\nCorr: {np.corrcoef(values, scores)[0,1]:.3f}')
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Score')
        
        # Remove unused
        for idx in range(n_params, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_optimization_history(self,
                                 best_scores: List[float],
                                 all_scores: List[float],
                                 save_name: str = 'optimization_history'):
        """Optimizasyon geçmişi"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Optimization History', fontsize=16, fontweight='bold')
        
        iterations = np.arange(1, len(best_scores) + 1)
        
        # 1. Best score over iterations
        ax = axes[0, 0]
        ax.plot(iterations, best_scores, marker='o', linewidth=2, markersize=6, color='green', label='Best Score')
        ax.fill_between(iterations, best_scores, alpha=0.3, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Score')
        ax.set_title('Best Score Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. All scores distribution
        ax = axes[0, 1]
        ax.hist(all_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_scores):.4f}')
        ax.axvline(np.max(all_scores), color='green', linestyle='--', linewidth=2, label=f'Max: {np.max(all_scores):.4f}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Improvement per iteration
        ax = axes[1, 0]
        improvements = [0] + [best_scores[i] - best_scores[i-1] for i in range(1, len(best_scores))]
        colors_imp = ['green' if imp >= 0 else 'red' for imp in improvements]
        ax.bar(iterations, improvements, color=colors_imp, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Improvement')
        ax.set_title('Score Improvement per Iteration')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Convergence analysis
        ax = axes[1, 1]
        target_score = np.max(best_scores) * 0.99
        converged_at = next((i for i, s in enumerate(best_scores) if s >= target_score), len(best_scores))
        
        ax.plot(iterations, best_scores, marker='o', linewidth=2, markersize=6, label='Best Score')
        ax.axhline(y=target_score, color='g', linestyle='--', linewidth=2, alpha=0.5, label=f'Target (99%)')
        ax.axvline(x=converged_at, color='r', linestyle='--', linewidth=2, alpha=0.5, label=f'Convergence at iter {converged_at}')
        
        ax.scatter([converged_at], [best_scores[converged_at-1]], color='red', s=200, marker='*', zorder=5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Score')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 9. FEATURE IMPORTANCE & DATA ANALYSIS
# =============================================================================

class FeatureImportanceVisualizer:
    """Feature importance görselleştirmeleri"""
    
    def __init__(self, output_dir='visualizations/feature_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_feature_importance_comparison(self,
                                          importances_dict: Dict[str, Dict[str, float]],
                                          feature_names: List[str],
                                          save_name: str = 'feature_importance'):
        """Feature importance karşılaştırması"""
        models = list(importances_dict.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            
            importances = [importances_dict[model].get(f, 0) for f in feature_names]
            sorted_idx = np.argsort(importances)[::-1][:15]  # Top 15
            
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importances = [importances[i] for i in sorted_idx]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
            ax.barh(sorted_features, sorted_importances, color=colors, edgecolor='black', linewidth=1)
            ax.set_xlabel('Importance')
            ax.set_title(f'{model}\nTop 15 Features')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add values
            for i, v in enumerate(sorted_importances):
                ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_best_features_analysis(self,
                                   data: pd.DataFrame,
                                   best_features: List[str],
                                   target_col: str,
                                   save_name: str = 'best_features'):
        """En iyi features'ın analizi"""
        n_features = len(best_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        fig.suptitle('Best Features Analysis', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(best_features):
            ax = axes[idx]
            
            # Scatter plot with target
            scatter = ax.scatter(data[feature], data[target_col], c=data[target_col], 
                               cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Trend line
            z = np.polyfit(data[feature], data[target_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[feature].min(), data[feature].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Correlation
            corr = np.corrcoef(data[feature], data[target_col])[0, 1]
            
            ax.set_xlabel(feature)
            ax.set_ylabel(target_col)
            ax.set_title(f'{feature}\nCorr: {corr:.4f}')
            ax.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(target_col)
        
        # Remove unused
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class MasterVisualizationSystem:
    """Tüm görselleştirme modüllerini entegre eden master sistem"""
    
    def __init__(self, output_dir='visualizations/master'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all visualizers
        self.robustness_viz = RobustnessVisualizer(self.output_dir / 'robustness')
        self.shap_viz = SHAPVisualizer(self.output_dir / 'shap')
        self.anomaly_viz = AnomalyKernelVisualizer(self.output_dir / 'anomaly')
        self.master_report_viz = MasterReportVisualizer(self.output_dir / 'master_report')
        self.prediction_viz = PredictionVisualizer(self.output_dir / 'predictions')
        self.model_comparison_viz = ModelComparisonVisualizer(self.output_dir / 'model_comparison')
        self.training_metrics_viz = TrainingMetricsVisualizer(self.output_dir / 'training_metrics')
        self.optimization_viz = OptimizationMetricsVisualizer(self.output_dir / 'optimization')
        self.feature_viz = FeatureImportanceVisualizer(self.output_dir / 'features')
        
        logger.info("Master Visualization System initialized successfully!")
    
    def generate_all_visualizations(self, project_data: Dict):
        """Generate all visualizations from project data"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING ALL VISUALIZATIONS")
        logger.info("="*80)
        
        # Robustness
        if 'robustness' in project_data:
            logger.info("\n-> Robustness Visualizations")
            self.robustness_viz.plot_perturbation_sensitivity(
                project_data['robustness']['predictions'],
                project_data['robustness']['results']
            )
        
        # SHAP
        if 'shap' in project_data:
            logger.info("\n-> SHAP Visualizations")
            self.shap_viz.plot_shap_summary(
                project_data['shap']['features'],
                project_data['shap']['values'],
                project_data['shap']['X']
            )
        
        # Anomaly
        if 'anomaly' in project_data:
            logger.info("\n-> Anomaly Analysis")
            self.anomaly_viz.plot_anomaly_characteristics(
                project_data['anomaly']['normal'],
                project_data['anomaly']['anomalous'],
                project_data['anomaly']['features']
            )
        
        # Master Report
        if 'master_results' in project_data:
            logger.info("\n-> Master Report Visualizations")
            self.master_report_viz.plot_master_summary_stats(
                project_data['master_results']['summary_df']
            )
        
        # Predictions
        if 'predictions' in project_data:
            logger.info("\n-> Prediction Visualizations")
            self.prediction_viz.plot_prediction_comparison(
                project_data['predictions']['experimental'],
                project_data['predictions']['models']
            )
        
        # Model Comparison
        if 'model_metrics' in project_data:
            logger.info("\n-> Model Comparison")
            self.model_comparison_viz.plot_model_ranking(
                project_data['model_metrics']
            )
        
        # Training Metrics
        if 'training_history' in project_data:
            logger.info("\n-> Training Metrics")
            self.training_metrics_viz.plot_training_curves(
                project_data['training_history']
            )
        
        logger.info("\n" + "="*80)
        logger.info("✓ ALL VISUALIZATIONS COMPLETED!")
        logger.info("="*80)


def main():
    """Test ve demo"""
    logger.info("\n" + "="*80)
    logger.info("MASTER VISUALIZATION SYSTEM - TEST")
    logger.info("="*80)
    
    # Initialize system
    viz_system = MasterVisualizationSystem('output/visualizations_master')
    
    logger.info(f"\n✓ System initialized")
    logger.info(f"✓ Output directory: output/visualizations_master")
    logger.info(f"\nAvailable visualizers:")
    logger.info("  - RobustnessVisualizer")
    logger.info("  - SHAPVisualizer")
    logger.info("  - AnomalyKernelVisualizer")
    logger.info("  - MasterReportVisualizer")
    logger.info("  - PredictionVisualizer")
    logger.info("  - ModelComparisonVisualizer")
    logger.info("  - TrainingMetricsVisualizer")
    logger.info("  - OptimizationMetricsVisualizer")
    logger.info("  - FeatureImportanceVisualizer")


if __name__ == "__main__":
    main()
