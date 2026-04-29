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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}_{sample_idx}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}_{feature_name}.png")


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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
        logger.info(f"[OK] Saved: {save_name}.html")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
        
        logger.info(f"[OK] Saved: {save_name}.png")
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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
        logger.info(f"[OK] Saved: {save_name}.png")
    
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
        logger.info(f"[OK] Saved: {save_name}.png")


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
    
    def _find_reports_dir(self) -> Optional[Path]:
        """Find PFAZ 6 reports directory"""
        candidates = [
            self.output_dir.parent / 'reports',
            self.output_dir.parent.parent / 'outputs' / 'reports',
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _find_thesis_excel(self, reports_dir: Path) -> Optional[Path]:
        """Find the main THESIS Excel file"""
        # Prefer the base name (no timestamp) if exists
        base = reports_dir / 'THESIS_COMPLETE_RESULTS.xlsx'
        if base.exists():
            return base
        # Otherwise pick the latest timestamped one
        candidates = sorted(reports_dir.glob('THESIS_COMPLETE_RESULTS_*.xlsx'), reverse=True)
        return candidates[0] if candidates else None

    def auto_generate_from_pfaz6_data(self) -> Dict:
        """
        Generate comprehensive visualizations from PFAZ 6 Excel output.
        Produces ~15 PNG charts and interactive HTML files.
        """
        reports_dir = self._find_reports_dir()
        if reports_dir is None:
            logger.warning("  Reports directory not found - skipping auto generation")
            return {}

        excel_path = self._find_thesis_excel(reports_dir)
        if excel_path is None:
            logger.warning("  THESIS_COMPLETE_RESULTS.xlsx not found")
            return {}

        logger.info(f"  Loading data from: {excel_path.name}")

        try:
            xl = pd.ExcelFile(excel_path)
            available_sheets = xl.sheet_names
        except Exception as e:
            logger.warning(f"  Cannot open Excel: {e}")
            return {}

        generated_files = []

        # ----------------------------------------------------------------
        # Sub-output directories
        # ----------------------------------------------------------------
        dirs = {
            'comparisons':   self.output_dir / 'comparisons',
            'distributions': self.output_dir / 'distributions',
            'heatmaps':      self.output_dir / 'heatmaps',
            'scatter':       self.output_dir / 'scatter',
            'performance':   self.output_dir / 'performance',
            'interactive':   self.output_dir / 'interactive',
            'summary':       self.output_dir / 'summary',
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Load primary data sheets
        # ----------------------------------------------------------------
        ai_df = None
        if 'All_AI_Models' in available_sheets:
            try:
                ai_df = pd.read_excel(xl, sheet_name='All_AI_Models')
                # Extract Target from Dataset name  (MM_ / QM_ / Beta_2_)
                def _get_target(ds):
                    ds = str(ds)
                    if ds.startswith('MM_QM'):
                        return 'MM_QM'
                    elif ds.startswith('MM_'):
                        return 'MM'
                    elif ds.startswith('QM_'):
                        return 'QM'
                    elif ds.startswith('Beta_2_'):
                        return 'Beta_2'
                    return ds.split('_')[0]

                ai_df['Target'] = ai_df['Dataset'].apply(_get_target)
                # Extract dataset size
                def _get_size(ds):
                    parts = str(ds).split('_')
                    for p in parts[1:]:
                        try:
                            return int(p)
                        except ValueError:
                            pass
                    return None
                ai_df['Size'] = ai_df['Dataset'].apply(_get_size)
                # Extract scenario (S70/S80)
                ai_df['Scenario'] = ai_df['Dataset'].apply(
                    lambda ds: 'S80' if '_S80_' in str(ds) else 'S70'
                )
                logger.info(f"  Loaded All_AI_Models: {len(ai_df)} rows")

                # OUTLIER FILTER: Remove extreme R2 rows that break chart scales
                # DNN divergence produces R2 values like -140000 which hide all other results
                R2_LOWER_BOUND = -10.0
                r2_cols = [c for c in ['Val_R2', 'Test_R2', 'Train_R2'] if c in ai_df.columns]
                before = len(ai_df)
                for col in r2_cols:
                    mask = pd.to_numeric(ai_df[col], errors='coerce') >= R2_LOWER_BOUND
                    ai_df = ai_df[mask | ai_df[col].isna()]
                removed = before - len(ai_df)
                if removed > 0:
                    logger.warning(
                        f"  [FILTER] Removed {removed} rows with R2 < {R2_LOWER_BOUND} "
                        f"(diverged models) from charts. Remaining: {len(ai_df)}"
                    )
            except Exception as e:
                logger.warning(f"  Could not load All_AI_Models: {e}")

        comparison_df = None
        if 'AI_vs_ANFIS_Comparison' in available_sheets:
            try:
                comparison_df = pd.read_excel(xl, sheet_name='AI_vs_ANFIS_Comparison')
                logger.info(f"  Loaded AI_vs_ANFIS_Comparison: {len(comparison_df)} rows")
            except Exception as e:
                logger.warning(f"  Could not load AI_vs_ANFIS_Comparison: {e}")

        cv_df = None
        if 'Robustness_CV_Results' in available_sheets:
            try:
                cv_df = pd.read_excel(xl, sheet_name='Robustness_CV_Results')
                logger.info(f"  Loaded Robustness_CV_Results: {len(cv_df)} rows")
            except Exception as e:
                logger.warning(f"  Could not load Robustness_CV_Results: {e}")

        # ----------------------------------------------------------------
        # CHART 1: Model Type R2 Boxplot (RF vs XGBoost vs DNN)
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                fig.suptitle('AI Model Performance by Model Type', fontsize=16, fontweight='bold')

                ax = axes[0]
                ax.set_title('Validation R2 by Model Type', fontsize=13)
                colors = {'RF': '#2196F3', 'XGBoost': '#4CAF50', 'DNN': '#FF9800'}
                data_dict = {mt: group['Val_R2'].dropna().values
                             for mt, group in ai_df.groupby('Model_Type')}
                bplot = ax.boxplot(data_dict.values(), labels=data_dict.keys(),
                                   patch_artist=True, notch=False)
                for patch, label in zip(bplot['boxes'], data_dict.keys()):
                    patch.set_facecolor(colors.get(label, '#9C27B0'))
                    patch.set_alpha(0.75)
                ax.set_ylabel('R² Score')
                ax.axhline(0.8, color='red', linestyle='--', alpha=0.6, label='R²=0.80')
                ax.axhline(0.9, color='green', linestyle='--', alpha=0.6, label='R²=0.90')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

                ax2 = axes[1]
                ax2.set_title('Test R2 by Model Type', fontsize=13)
                data_test = {mt: group['Test_R2'].dropna().values
                             for mt, group in ai_df.groupby('Model_Type')}
                bplot2 = ax2.boxplot(data_test.values(), labels=data_test.keys(),
                                     patch_artist=True, notch=False)
                for patch, label in zip(bplot2['boxes'], data_test.keys()):
                    patch.set_facecolor(colors.get(label, '#9C27B0'))
                    patch.set_alpha(0.75)
                ax2.set_ylabel('R² Score')
                ax2.axhline(0.8, color='red', linestyle='--', alpha=0.6)
                ax2.axhline(0.9, color='green', linestyle='--', alpha=0.6)
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                fpath = dirs['comparisons'] / 'model_type_r2_boxplot.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 1 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 2: Target R2 Comparison (MM vs QM vs Beta_2)
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                targets = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                fig.suptitle('AI Model Performance by Target Variable', fontsize=16, fontweight='bold')

                tcolors = {'MM': '#E91E63', 'QM': '#3F51B5', 'MM_QM': '#FF5722', 'Beta_2': '#009688'}

                for col_idx, metric in enumerate(['Val_R2', 'Test_R2']):
                    ax = axes[col_idx]
                    ax.set_title(f'{metric.replace("_"," ")} by Target', fontsize=13)
                    data_dict = {t: ai_df[ai_df['Target'] == t][metric].dropna().values
                                 for t in targets}
                    bplot = ax.boxplot(data_dict.values(), labels=data_dict.keys(),
                                       patch_artist=True)
                    for patch, lbl in zip(bplot['boxes'], data_dict.keys()):
                        patch.set_facecolor(tcolors.get(lbl, '#795548'))
                        patch.set_alpha(0.75)
                    ax.set_ylabel('R² Score')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='0.80')
                    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='0.90')
                    ax.legend(fontsize=9)

                plt.tight_layout()
                fpath = dirs['comparisons'] / 'target_r2_comparison.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 2 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 3: Train / Val / Test R2 mean comparison (grouped bar)
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                grp = ai_df.groupby('Model_Type')[['Train_R2', 'Val_R2', 'Test_R2']].mean()
                x = np.arange(len(grp))
                width = 0.25
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.bar(x - width, grp['Train_R2'], width, label='Train R²', color='#4CAF50', alpha=0.8)
                ax.bar(x,         grp['Val_R2'],   width, label='Val R²',   color='#2196F3', alpha=0.8)
                ax.bar(x + width, grp['Test_R2'],  width, label='Test R²',  color='#FF5722', alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(grp.index, fontsize=12)
                ax.set_ylabel('Mean R² Score', fontsize=12)
                ax.set_title('Train / Validation / Test R² by Model Type', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3, axis='y')
                for bars, vals in zip([ax.containers[0], ax.containers[1], ax.containers[2]],
                                       [grp['Train_R2'], grp['Val_R2'], grp['Test_R2']]):
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
                plt.tight_layout()
                fpath = dirs['comparisons'] / 'train_val_test_r2_grouped.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 3 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 4: Top 25 Models by Val R2 (horizontal bar)
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                top25 = ai_df.nlargest(25, 'Val_R2')[['Dataset', 'Model_Type', 'Val_R2', 'Test_R2']].copy()
                top25['Label'] = top25['Model_Type'] + ' | ' + top25['Dataset'].str[:35]
                cmap = {'RF': '#2196F3', 'XGBoost': '#4CAF50', 'DNN': '#FF9800'}
                colors_bar = [cmap.get(m, '#9C27B0') for m in top25['Model_Type']]

                fig, ax = plt.subplots(figsize=(14, 10))
                y = range(len(top25))
                bars = ax.barh(y, top25['Val_R2'], color=colors_bar, alpha=0.8, label='Val R²')
                ax.scatter(top25['Test_R2'].values, list(y), color='red', zorder=5,
                           s=60, label='Test R²', marker='D')
                ax.set_yticks(list(y))
                ax.set_yticklabels(top25['Label'].values, fontsize=8)
                ax.set_xlabel('R² Score', fontsize=12)
                ax.set_title('Top 25 AI Models by Validation R²', fontsize=14, fontweight='bold')
                ax.axvline(0.9, color='green', linestyle='--', alpha=0.5, label='R²=0.90')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='x')
                ax.set_xlim(0, 1.05)
                plt.tight_layout()
                fpath = dirs['performance'] / 'top25_models_val_r2.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 4 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 5: R2 Distribution Histograms
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('R² Score Distribution - All AI Models', fontsize=16, fontweight='bold')

                for ax, (metric, title) in zip(axes.flat,
                    [('Train_R2', 'Train R² Distribution'),
                     ('Val_R2',   'Validation R² Distribution'),
                     ('Test_R2',  'Test R² Distribution'),
                     ('Val_R2',   'Val R² by Model Type (Density)')]):
                    if title.endswith('Density)'):
                        for mt, grp in ai_df.groupby('Model_Type'):
                            grp['Val_R2'].dropna().plot.hist(ax=ax, bins=30, alpha=0.5,
                                                              label=mt, density=True)
                        ax.set_title(title, fontsize=12)
                        ax.legend()
                    else:
                        ai_df[metric].dropna().hist(ax=ax, bins=40, color='steelblue',
                                                     edgecolor='white', alpha=0.8)
                        mean_v = ai_df[metric].mean()
                        ax.axvline(mean_v, color='red', linestyle='--',
                                   label=f'Mean={mean_v:.3f}')
                        ax.set_title(title, fontsize=12)
                        ax.legend()
                    ax.set_xlabel('R² Score')
                    ax.set_ylabel('Count / Density')
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                fpath = dirs['distributions'] / 'r2_distribution_histograms.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 5 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 6: Val_R2 vs Test_R2 Scatter (Overfitting Analysis)
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                cmap = {'RF': '#2196F3', 'XGBoost': '#4CAF50', 'DNN': '#FF9800'}
                fig, ax = plt.subplots(figsize=(10, 8))
                for mt, grp in ai_df.groupby('Model_Type'):
                    ax.scatter(grp['Val_R2'], grp['Test_R2'],
                               alpha=0.3, s=20, color=cmap.get(mt, '#555'),
                               label=mt)
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                        max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'k--', alpha=0.5, label='Val = Test')
                ax.set_xlabel('Validation R²', fontsize=12)
                ax.set_ylabel('Test R²', fontsize=12)
                ax.set_title('Overfitting Analysis: Val R² vs Test R²', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fpath = dirs['scatter'] / 'val_vs_test_r2_overfitting.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 6 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 7: Dataset Size vs R2 (75/100/150/200)
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                size_grp = ai_df.dropna(subset=['Size']).groupby('Size').agg(
                    Val_R2_mean=('Val_R2', 'mean'),
                    Val_R2_std=('Val_R2', 'std'),
                    Test_R2_mean=('Test_R2', 'mean'),
                    Test_R2_std=('Test_R2', 'std'),
                    Count=('Val_R2', 'count')
                ).reset_index().sort_values('Size')

                fig, ax = plt.subplots(figsize=(10, 7))
                ax.errorbar(size_grp['Size'], size_grp['Val_R2_mean'],
                            yerr=size_grp['Val_R2_std'], marker='o', markersize=8,
                            linewidth=2, capsize=5, label='Val R²', color='#2196F3')
                ax.errorbar(size_grp['Size'], size_grp['Test_R2_mean'],
                            yerr=size_grp['Test_R2_std'], marker='s', markersize=8,
                            linewidth=2, capsize=5, label='Test R²', color='#FF5722',
                            linestyle='--')
                for _, row in size_grp.iterrows():
                    ax.annotate(f"n={int(row['Count'])}", (row['Size'], row['Val_R2_mean']),
                                textcoords='offset points', xytext=(5, 5), fontsize=9)
                ax.set_xlabel('Dataset Size (# Nuclei)', fontsize=12)
                ax.set_ylabel('Mean R² Score', fontsize=12)
                ax.set_title('Effect of Dataset Size on Model Performance', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.05)
                plt.tight_layout()
                fpath = dirs['scatter'] / 'dataset_size_vs_r2.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 7 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 8: Target x Model Type R2 Heatmap
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                targets_order = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]
                pivot_val = ai_df[ai_df['Target'].isin(targets_order)].pivot_table(
                    values='Val_R2', index='Target', columns='Model_Type', aggfunc='mean'
                )
                pivot_test = ai_df[ai_df['Target'].isin(targets_order)].pivot_table(
                    values='Test_R2', index='Target', columns='Model_Type', aggfunc='mean'
                )

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle('R² Heatmap: Target x Model Type', fontsize=15, fontweight='bold')

                for ax, data, title in [
                    (axes[0], pivot_val, 'Validation R²'),
                    (axes[1], pivot_test, 'Test R²')
                ]:
                    im = ax.imshow(data.values, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
                    ax.set_xticks(range(len(data.columns)))
                    ax.set_yticks(range(len(data.index)))
                    ax.set_xticklabels(data.columns, fontsize=11)
                    ax.set_yticklabels(data.index, fontsize=11)
                    ax.set_title(title, fontsize=13)
                    for i in range(len(data.index)):
                        for j in range(len(data.columns)):
                            v = data.values[i, j]
                            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                                    fontsize=12, fontweight='bold',
                                    color='white' if v < 0.6 else 'black')
                    plt.colorbar(im, ax=ax, shrink=0.8)

                plt.tight_layout()
                fpath = dirs['heatmaps'] / 'target_model_r2_heatmap.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 8 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 9: S70 vs S80 Scenario Comparison
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                scen_grp = ai_df.groupby('Scenario')[['Val_R2', 'Test_R2']].agg(['mean', 'std'])
                scen_grp.columns = ['Val_mean', 'Val_std', 'Test_mean', 'Test_std']

                fig, ax = plt.subplots(figsize=(9, 6))
                x = np.arange(len(scen_grp))
                w = 0.35
                ax.bar(x - w/2, scen_grp['Val_mean'], w, yerr=scen_grp['Val_std'],
                       label='Val R²', color='#2196F3', alpha=0.8, capsize=6)
                ax.bar(x + w/2, scen_grp['Test_mean'], w, yerr=scen_grp['Test_std'],
                       label='Test R²', color='#FF5722', alpha=0.8, capsize=6)
                ax.set_xticks(x)
                ax.set_xticklabels(scen_grp.index, fontsize=13)
                ax.set_ylabel('Mean R² Score', fontsize=12)
                ax.set_title('Train/Test Split Scenario: S70 vs S80', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                _all_vals = list(scen_grp['Val_mean'].dropna()) + list(scen_grp['Test_mean'].dropna())
                _ymin = max(-1.0, min(_all_vals) - 0.1) if _all_vals else -0.1
                ax.set_ylim(_ymin, 1.1)
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
                ax.grid(True, alpha=0.3, axis='y')
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', fontsize=9)
                plt.tight_layout()
                fpath = dirs['comparisons'] / 'scenario_s70_vs_s80.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 9 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 10: AI vs ANFIS per Target (Best Val R2)
        # ----------------------------------------------------------------
        if comparison_df is not None:
            try:
                targets_present = [t for t in ['MM', 'QM', 'Beta_2', 'MM_QM']
                                   if t in comparison_df.get('Target', pd.Series(dtype=str)).unique()]
                if 'Target' not in comparison_df.columns and len(comparison_df.columns) > 2:
                    # Attempt to extract target
                    comparison_df['Target'] = comparison_df['Dataset'].apply(
                        lambda ds: 'MM_QM' if str(ds).startswith('MM_QM')
                        else 'MM' if str(ds).startswith('MM_')
                        else 'QM' if str(ds).startswith('QM_')
                        else 'Beta_2' if str(ds).startswith('Beta_2_')
                        else 'Unknown'
                    )

                grp = comparison_df.groupby('Target').agg(
                    AI_Val=('AI_Best_Val_R2', 'max'),
                    ANFIS_Val=('ANFIS_Best_Val_R2', 'max')
                ).reset_index()

                if len(grp) > 0:
                    x = np.arange(len(grp))
                    w = 0.35
                    fig, ax = plt.subplots(figsize=(11, 7))
                    ax.bar(x - w/2, grp['AI_Val'],    w, label='AI Best Val R²',    color='#1565C0', alpha=0.85)
                    ax.bar(x + w/2, grp['ANFIS_Val'], w, label='ANFIS Best Val R²', color='#2E7D32', alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels(grp['Target'], fontsize=12)
                    ax.set_ylabel('Best Validation R²', fontsize=12)
                    ax.set_title('AI vs ANFIS: Best Validation R² per Target', fontsize=14, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='0.90')
                    ax.set_ylim(0, 1.1)
                    ax.grid(True, alpha=0.3, axis='y')
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.3f', fontsize=10)
                    plt.tight_layout()
                    fpath = dirs['comparisons'] / 'ai_vs_anfis_per_target.png'
                    fig.savefig(fpath, dpi=200, bbox_inches='tight')
                    plt.close()
                    generated_files.append(str(fpath))
                    logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 10 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 11: Cross-Validation Stability (from Robustness sheet)
        # ----------------------------------------------------------------
        if cv_df is not None and len(cv_df) > 0:
            try:
                cv_plot = cv_df[['Model', 'r2_test_mean', 'r2_test_std']].dropna().head(20)
                if len(cv_plot) > 0:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    y = range(len(cv_plot))
                    ax.barh(y, cv_plot['r2_test_mean'], xerr=cv_plot['r2_test_std'],
                            color='#3F51B5', alpha=0.8, capsize=4,
                            error_kw={'elinewidth': 1.5, 'capthick': 1.5})
                    ax.set_yticks(list(y))
                    ax.set_yticklabels(cv_plot['Model'].str[:40], fontsize=9)
                    ax.set_xlabel('CV Test R² (mean ± std)', fontsize=12)
                    ax.set_title('Cross-Validation Stability by Model', fontsize=14, fontweight='bold')
                    ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='0.80')
                    ax.axvline(0.9, color='green', linestyle='--', alpha=0.5, label='0.90')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    fpath = dirs['performance'] / 'cross_validation_stability.png'
                    fig.savefig(fpath, dpi=200, bbox_inches='tight')
                    plt.close()
                    generated_files.append(str(fpath))
                    logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 11 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 12: Degradation / Overfitting Histogram (Val - Test gap)
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                ai_df['Degradation'] = ai_df['Val_R2'] - ai_df['Test_R2']
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle('Model Generalization Analysis (Val R² - Test R²)', fontsize=15, fontweight='bold')

                ax = axes[0]
                ai_df['Degradation'].hist(ax=ax, bins=50, color='#FF5722', edgecolor='white', alpha=0.8)
                ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
                ax.axvline(ai_df['Degradation'].mean(), color='red', linestyle='--',
                           label=f"Mean={ai_df['Degradation'].mean():.3f}")
                ax.set_xlabel('Val R² - Test R²  (positive = overfitting)', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title('Degradation Distribution (All Models)', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)

                ax2 = axes[1]
                for mt, grp in ai_df.groupby('Model_Type'):
                    grp['Degradation'].hist(ax=ax2, bins=30, alpha=0.5, label=mt, density=True)
                ax2.axvline(0, color='black', linestyle='-', linewidth=1.5)
                ax2.set_xlabel('Val R² - Test R²', fontsize=11)
                ax2.set_ylabel('Density', fontsize=11)
                ax2.set_title('Degradation by Model Type', fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                fpath = dirs['distributions'] / 'overfitting_degradation.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 12 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 13: Model Type x Target mean Val_R2 bar chart
        # ----------------------------------------------------------------
        if ai_df is not None:
            try:
                targets_sel = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]
                pivot = ai_df[ai_df['Target'].isin(targets_sel)].pivot_table(
                    values='Val_R2', index='Model_Type', columns='Target', aggfunc='mean'
                )
                x = np.arange(len(pivot.index))
                w = 0.25
                colors_t = ['#E91E63', '#3F51B5', '#009688']
                fig, ax = plt.subplots(figsize=(12, 7))
                for i, (target, col) in enumerate(zip(pivot.columns, colors_t)):
                    offset = (i - 1) * w
                    bars = ax.bar(x + offset, pivot[target], w, label=target, color=col, alpha=0.85)
                    ax.bar_label(bars, fmt='%.3f', fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(pivot.index, fontsize=12)
                ax.set_ylabel('Mean Validation R²', fontsize=12)
                ax.set_title('Mean Validation R² by Model Type and Target', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                fpath = dirs['comparisons'] / 'model_type_target_val_r2.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 13 failed: {e}")

        # ----------------------------------------------------------------
        # CHART 14: Project Summary Dashboard
        # ----------------------------------------------------------------
        try:
            reports_dir2 = self._find_reports_dir()
            summary_path = (reports_dir2 / 'final_summary.json') if reports_dir2 else None
            if summary_path and summary_path.exists():
                with open(summary_path, encoding='utf-8') as f:
                    summary = json.load(f)

                fig = plt.figure(figsize=(16, 10))
                fig.suptitle('Nuclear Physics AI Project - Summary Dashboard', fontsize=16, fontweight='bold')

                # Stats text panel
                ax_text = fig.add_subplot(2, 3, 1)
                ax_text.axis('off')
                stats_lines = [
                    f"Total AI Configs:   {summary.get('total_ai_configs', 'N/A'):,}",
                    f"Total ANFIS Configs:{summary.get('total_anfis_records', 'N/A'):,}",
                    f"AI Datasets:        {len(summary.get('ai_datasets', []))}",
                    f"ANFIS Datasets:     {len(summary.get('anfis_datasets', []))}",
                    f"Best AI Val R2:     {summary.get('ai_best_val_r2', 'N/A')}",
                    f"Best ANFIS Val R2:  {summary.get('anfis_best_val_r2', 'N/A')}",
                    f"Targets:            {', '.join(summary.get('targets', []))}",
                ]
                ax_text.text(0.05, 0.95, '\n'.join(stats_lines), transform=ax_text.transAxes,
                             fontsize=10, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(facecolor='lightyellow', alpha=0.8))
                ax_text.set_title('Project Statistics', fontsize=12, fontweight='bold')

                # Per-target best R2 bar chart
                ax_bar = fig.add_subplot(2, 3, 2)
                per_t = summary.get('per_target_best', {})
                t_labels, ai_vals, anfis_vals = [], [], []
                for t, info in per_t.items():
                    t_labels.append(t)
                    ai_vals.append(info.get('AI_best_val_r2', 0) or 0)
                    anfis_vals.append(info.get('ANFIS_best_val_r2', 0) or 0)
                xp = np.arange(len(t_labels))
                ax_bar.bar(xp - 0.2, ai_vals, 0.35, label='AI Best Val', color='#1565C0', alpha=0.85)
                ax_bar.bar(xp + 0.2, anfis_vals, 0.35, label='ANFIS Best Val', color='#2E7D32', alpha=0.85)
                ax_bar.set_xticks(xp)
                ax_bar.set_xticklabels(t_labels, fontsize=10)
                ax_bar.set_ylim(0, 1.1)
                ax_bar.set_title('Best Val R² per Target', fontsize=11, fontweight='bold')
                ax_bar.legend(fontsize=9)
                ax_bar.grid(True, alpha=0.3, axis='y')
                ax_bar.bar_label(ax_bar.containers[0], fmt='%.3f', fontsize=8)
                ax_bar.bar_label(ax_bar.containers[1], fmt='%.3f', fontsize=8)

                # Model type distribution pie
                if ai_df is not None:
                    ax_pie = fig.add_subplot(2, 3, 3)
                    mt_counts = ai_df['Model_Type'].value_counts()
                    ax_pie.pie(mt_counts.values, labels=mt_counts.index,
                               autopct='%1.1f%%', colors=['#2196F3', '#4CAF50', '#FF9800'],
                               startangle=90)
                    ax_pie.set_title('AI Config Distribution', fontsize=11, fontweight='bold')

                # Val R2 CDF by model type
                if ai_df is not None:
                    ax_cdf = fig.add_subplot(2, 3, 4)
                    for mt, grp in ai_df.groupby('Model_Type'):
                        vals = np.sort(grp['Val_R2'].dropna().values)
                        cdf = np.arange(1, len(vals)+1) / len(vals)
                        ax_cdf.plot(vals, cdf, linewidth=2, label=mt)
                    ax_cdf.set_xlabel('Val R²')
                    ax_cdf.set_ylabel('CDF')
                    ax_cdf.set_title('Val R² CDF by Model Type', fontsize=11, fontweight='bold')
                    ax_cdf.legend(fontsize=9)
                    ax_cdf.grid(True, alpha=0.3)
                    ax_cdf.axvline(0.8, color='red', linestyle='--', alpha=0.5)

                # Target distribution pie
                if ai_df is not None:
                    ax_pie2 = fig.add_subplot(2, 3, 5)
                    tc = ai_df['Target'].value_counts()
                    ax_pie2.pie(tc.values, labels=tc.index, autopct='%1.1f%%',
                                colors=['#E91E63', '#3F51B5', '#009688', '#FF9800'],
                                startangle=90)
                    ax_pie2.set_title('Dataset Distribution by Target', fontsize=11, fontweight='bold')

                # Val R2 per target boxplot simplified
                if ai_df is not None:
                    ax_box = fig.add_subplot(2, 3, 6)
                    t_sel = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]
                    box_data = [ai_df[ai_df['Target'] == t]['Val_R2'].dropna().values for t in t_sel]
                    ax_box.boxplot(box_data, labels=t_sel, patch_artist=True)
                    ax_box.set_title('Val R² per Target', fontsize=11, fontweight='bold')
                    ax_box.set_ylabel('Val R²')
                    ax_box.grid(True, alpha=0.3)

                plt.tight_layout()
                fpath = dirs['summary'] / 'project_dashboard.png'
                fig.savefig(fpath, dpi=200, bbox_inches='tight')
                plt.close()
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
        except Exception as e:
            logger.warning(f"  CHART 14 (dashboard) failed: {e}")

        # ----------------------------------------------------------------
        # CHART 15: INTERACTIVE HTML - Val vs Test R2 Scatter (Plotly)
        # ----------------------------------------------------------------
        if ai_df is not None and PLOTLY_AVAILABLE:
            try:
                sample = ai_df.sample(min(3000, len(ai_df)), random_state=42)
                fig_html = px.scatter(
                    sample, x='Val_R2', y='Test_R2',
                    color='Model_Type', symbol='Target',
                    hover_data=['Dataset', 'Train_R2'],
                    title='Interactive: Validation R² vs Test R² (AI Models)',
                    labels={'Val_R2': 'Validation R²', 'Test_R2': 'Test R²'},
                    opacity=0.6, width=1100, height=750
                )
                # Add diagonal line
                mn, mx = 0, 1
                fig_html.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                                   line=dict(color='black', dash='dash', width=1))
                fpath = dirs['interactive'] / 'interactive_val_vs_test_r2.html'
                fig_html.write_html(str(fpath))
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 15 (HTML scatter) failed: {e}")

        # ----------------------------------------------------------------
        # CHART 16: INTERACTIVE HTML - Model Type R2 Distribution (Plotly)
        # ----------------------------------------------------------------
        if ai_df is not None and PLOTLY_AVAILABLE:
            try:
                fig_html = px.box(
                    ai_df, x='Model_Type', y='Val_R2', color='Target',
                    title='Interactive: Validation R² Distribution by Model Type & Target',
                    labels={'Val_R2': 'Validation R²', 'Model_Type': 'Model Type'},
                    width=1100, height=700
                )
                fpath = dirs['interactive'] / 'interactive_r2_boxplot.html'
                fig_html.write_html(str(fpath))
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 16 (HTML boxplot) failed: {e}")

        # ----------------------------------------------------------------
        # CHART 17: INTERACTIVE HTML - Performance Dashboard (Plotly subplots)
        # ----------------------------------------------------------------
        if ai_df is not None and PLOTLY_AVAILABLE:
            try:
                from plotly.subplots import make_subplots as _make_subplots
                fig_dash = _make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Val R² by Model Type', 'Test R² by Target',
                                    'Val vs Test R² Scatter', 'Dataset Size vs Val R²']
                )
                colors_map = {'RF': '#2196F3', 'XGBoost': '#4CAF50', 'DNN': '#FF9800'}
                for mt in ai_df['Model_Type'].unique():
                    grp = ai_df[ai_df['Model_Type'] == mt]
                    fig_dash.add_trace(
                        go.Box(y=grp['Val_R2'], name=mt, marker_color=colors_map.get(mt, '#9C27B0'),
                               showlegend=True),
                        row=1, col=1
                    )
                for t in [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]:
                    grp = ai_df[ai_df['Target'] == t]
                    fig_dash.add_trace(
                        go.Box(y=grp['Test_R2'], name=t, showlegend=True),
                        row=1, col=2
                    )
                sample2 = ai_df.sample(min(1500, len(ai_df)), random_state=1)
                fig_dash.add_trace(
                    go.Scatter(x=sample2['Val_R2'], y=sample2['Test_R2'], mode='markers',
                               marker=dict(size=4, opacity=0.5, color='#607D8B'),
                               name='Models', showlegend=False),
                    row=2, col=1
                )
                sz_grp = ai_df.dropna(subset=['Size']).groupby('Size')['Val_R2'].mean().reset_index()
                fig_dash.add_trace(
                    go.Scatter(x=sz_grp['Size'], y=sz_grp['Val_R2'], mode='lines+markers',
                               marker=dict(size=8), line=dict(width=2), name='Size vs Val R²',
                               showlegend=False),
                    row=2, col=2
                )
                fig_dash.update_layout(
                    title_text='AI Model Performance Dashboard',
                    height=900, width=1200, showlegend=True
                )
                fpath = dirs['interactive'] / 'interactive_performance_dashboard.html'
                fig_dash.write_html(str(fpath))
                generated_files.append(str(fpath))
                logger.info(f"  [OK] {fpath.name}")
            except Exception as e:
                logger.warning(f"  CHART 17 (HTML dashboard) failed: {e}")

        # ================================================================
        # SECTION B: NUCLEAR PHYSICS CHARTS
        # ================================================================
        self._generate_nuclear_charts(dirs, generated_files)

        # ================================================================
        # SECTION C: ANFIS SPECIFIC CHARTS
        # ================================================================
        self._generate_anfis_charts(ai_df, dirs, generated_files)

        # ================================================================
        # SECTION D: ACTUAL vs PREDICTED + RESIDUALS
        # ================================================================
        self._generate_prediction_charts(dirs, generated_files)

        # ================================================================
        # SECTION E: FEATURE ANALYSIS CHARTS
        # ================================================================
        self._generate_feature_charts(ai_df, dirs, generated_files)

        # ================================================================
        # SECTION F: 3D CHARTS
        # ================================================================
        self._generate_3d_charts(ai_df, dirs, generated_files)

        # ================================================================
        # SECTION G: EXTENDED INTERACTIVE HTML
        # ================================================================
        self._generate_interactive_extended(ai_df, dirs, generated_files)

        # ================================================================
        # SECTION H: EMPTY FOLDER CHARTS (predictions, features, anomaly,
        #            robustness, shap, master_report, optimization,
        #            training_metrics)
        # ================================================================
        self._generate_all_empty_folder_charts(ai_df, generated_files)

        return {'files': generated_files, 'count': len(generated_files)}

    # ------------------------------------------------------------------
    # SECTION B: Nuclear Physics Charts
    # ------------------------------------------------------------------
    def _generate_nuclear_charts(self, dirs: Dict, generated_files: list):
        """Nuclear chart visualizations from AAA2 enriched data"""
        aaa2_path = self._find_reports_dir()
        if aaa2_path:
            aaa2_path = aaa2_path.parent / 'generated_datasets' / 'AAA2_enriched_all_nuclei.csv'
        if aaa2_path is None or not aaa2_path.exists():
            return
        try:
            ndf = pd.read_csv(aaa2_path)
        except Exception:
            return

        ndf.columns = [c.strip() for c in ndf.columns]
        mm_col = next((c for c in ndf.columns if 'MAGNETIC' in c.upper()), None)
        qm_col = next((c for c in ndf.columns if 'QUADRUPOLE' in c.upper()), None)

        # Chart N1: Nuclear chart - Z vs N colored by Beta_2
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            sc = ax.scatter(ndf['N'], ndf['Z'], c=pd.to_numeric(ndf['Beta_2'], errors='coerce'),
                            cmap='RdYlGn', s=60, alpha=0.8, edgecolors='k', linewidths=0.3)
            plt.colorbar(sc, ax=ax, label='Beta_2 (Deformation)')
            # Magic number lines
            for mn in [8, 20, 28, 50, 82, 126]:
                ax.axvline(mn, color='blue', linestyle='--', alpha=0.25, linewidth=0.8)
                ax.axhline(mn, color='blue', linestyle='--', alpha=0.25, linewidth=0.8)
            ax.set_xlabel('Neutron Number (N)', fontsize=13)
            ax.set_ylabel('Proton Number (Z)', fontsize=13)
            ax.set_title('Nuclear Chart: Beta_2 Deformation (267 Nuclei)', fontsize=15, fontweight='bold')
            ax.text(0.02, 0.98, 'Dashed lines: magic numbers', transform=ax.transAxes,
                    fontsize=9, va='top', color='blue', alpha=0.7)
            plt.tight_layout()
            fp = dirs['heatmaps'] / 'nuclear_chart_beta2.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Nuclear chart Beta2 failed: {e}")

        # Chart N2: Nuclear chart - Z vs N colored by Magnetic Moment
        if mm_col:
            try:
                mm_vals = pd.to_numeric(ndf[mm_col], errors='coerce')
                vmax = mm_vals.abs().quantile(0.95)
                fig, ax = plt.subplots(figsize=(14, 10))
                sc = ax.scatter(ndf['N'], ndf['Z'], c=mm_vals, cmap='coolwarm',
                                s=60, alpha=0.85, edgecolors='k', linewidths=0.3,
                                vmin=-vmax, vmax=vmax)
                plt.colorbar(sc, ax=ax, label='Magnetic Moment (μN)')
                for mn in [8, 20, 28, 50, 82, 126]:
                    ax.axvline(mn, color='gray', linestyle='--', alpha=0.25, linewidth=0.8)
                    ax.axhline(mn, color='gray', linestyle='--', alpha=0.25, linewidth=0.8)
                ax.set_xlabel('Neutron Number (N)', fontsize=13)
                ax.set_ylabel('Proton Number (Z)', fontsize=13)
                ax.set_title('Nuclear Chart: Magnetic Moment Distribution', fontsize=15, fontweight='bold')
                plt.tight_layout()
                fp = dirs['heatmaps'] / 'nuclear_chart_magnetic_moment.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  Nuclear chart MM failed: {e}")

        # Chart N3: Nuclear chart - colored by Quadrupole Moment
        if qm_col:
            try:
                qm_vals = pd.to_numeric(ndf[qm_col], errors='coerce')
                has_qm = qm_vals.notna()
                fig, ax = plt.subplots(figsize=(14, 10))
                ax.scatter(ndf.loc[~has_qm, 'N'], ndf.loc[~has_qm, 'Z'],
                           c='lightgray', s=40, alpha=0.5, label='No QM data', zorder=1)
                sc = ax.scatter(ndf.loc[has_qm, 'N'], ndf.loc[has_qm, 'Z'],
                                c=qm_vals[has_qm], cmap='PiYG', s=70, alpha=0.9,
                                edgecolors='k', linewidths=0.3, zorder=2)
                plt.colorbar(sc, ax=ax, label='Quadrupole Moment Q (barn)')
                for mn in [8, 20, 28, 50, 82, 126]:
                    ax.axvline(mn, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
                    ax.axhline(mn, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
                ax.set_xlabel('Neutron Number (N)', fontsize=13)
                ax.set_ylabel('Proton Number (Z)', fontsize=13)
                ax.set_title('Nuclear Chart: Quadrupole Moment Distribution', fontsize=15, fontweight='bold')
                ax.legend(fontsize=10)
                plt.tight_layout()
                fp = dirs['heatmaps'] / 'nuclear_chart_quadrupole.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  Nuclear chart QM failed: {e}")

        # Chart N4: Spin & Parity Distribution
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('Nuclear Spin and Parity Distribution', fontsize=14, fontweight='bold')
            # Spin
            spin_counts = ndf['SPIN'].value_counts().sort_index()
            axes[0].bar(spin_counts.index.astype(str), spin_counts.values, color='#3F51B5', alpha=0.8)
            axes[0].set_xlabel('Spin (ℏ)', fontsize=12)
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_title('Spin Distribution', fontsize=12)
            axes[0].grid(True, alpha=0.3, axis='y')
            # Parity
            par_counts = ndf['PARITY'].map({1: '+', -1: '−'}).value_counts()
            axes[1].bar(par_counts.index, par_counts.values,
                        color=['#4CAF50', '#F44336'], alpha=0.85)
            axes[1].set_xlabel('Parity', fontsize=12)
            axes[1].set_ylabel('Count', fontsize=12)
            axes[1].set_title('Parity Distribution', fontsize=12)
            for ax in axes:
                ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            fp = dirs['distributions'] / 'spin_parity_distribution.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Spin/parity failed: {e}")

        # Chart N5: Isotopic chains - Sn (Z=50) and Pb (Z=82)
        if mm_col:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle('Isotopic Chain Analysis', fontsize=14, fontweight='bold')
                for ax, (Z_val, name) in zip(axes, [(50, 'Sn (Z=50)'), (82, 'Pb (Z=82)')]):
                    chain = ndf[ndf['Z'] == Z_val].copy()
                    chain['MM_num'] = pd.to_numeric(chain[mm_col], errors='coerce')
                    chain_sorted = chain.sort_values('N')
                    ax.plot(chain_sorted['N'], chain_sorted['MM_num'], 'o-', color='#E91E63',
                            markersize=8, linewidth=2, label='Magnetic Moment')
                    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
                    ax.set_xlabel('Neutron Number (N)', fontsize=12)
                    ax.set_ylabel('Magnetic Moment (μN)', fontsize=12)
                    ax.set_title(f'{name} Isotopic Chain', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=10)
                plt.tight_layout()
                fp = dirs['scatter'] / 'isotopic_chains_mm.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  Isotopic chains failed: {e}")

        # Chart N6: Magic number effect on properties
        try:
            magic = [2, 8, 20, 28, 50, 82, 126]
            ndf['near_magic'] = (
                ndf['Z'].apply(lambda z: min(abs(z - m) for m in magic)) <= 2
            ) | (
                ndf['N'].apply(lambda n: min(abs(n - m) for m in magic)) <= 2
            )
            ndf['category'] = ndf['near_magic'].map({True: 'Near Magic', False: 'Non-Magic'})
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Magic Number Effect on Nuclear Properties', fontsize=14, fontweight='bold')
            props = [('Beta_2', 'Deformation β₂', axes[0]),
                     (mm_col, 'Magnetic Moment (μN)', axes[1]) if mm_col else (None, None, None),
                     (qm_col, 'Quadrupole Moment (barn)', axes[2]) if qm_col else (None, None, None)]
            colors_cat = {'Near Magic': '#2196F3', 'Non-Magic': '#FF5722'}
            for prop, label, ax in props:
                if prop is None or ax is None:
                    continue
                for cat, grp in ndf.groupby('category'):
                    vals = pd.to_numeric(grp[prop], errors='coerce').dropna()
                    if len(vals) > 0:
                        ax.hist(vals, bins=25, alpha=0.65, label=cat,
                                color=colors_cat[cat], edgecolor='white')
                ax.set_xlabel(label, fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title(label, fontsize=11)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fp = dirs['distributions'] / 'magic_number_effect.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Magic number effect failed: {e}")

        # Chart N7: Mass region analysis
        try:
            ndf['mass_region'] = pd.cut(ndf['A'], bins=[0, 40, 100, 150, 220, 300],
                                         labels=['Light\n(A<40)', 'Medium\n(40-100)',
                                                 'Heavy\n(100-150)', 'Very Heavy\n(150-220)',
                                                 'Super Heavy\n(>220)'])
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Nuclear Properties by Mass Region', fontsize=14, fontweight='bold')
            for ax, (col, lbl) in zip(axes, [
                ('Beta_2', 'β₂ Deformation'),
                (mm_col, 'Magnetic Moment (μN)'),
                (qm_col, 'Quadrupole Moment (barn)')
            ]):
                if col is None:
                    continue
                ndf['_val'] = pd.to_numeric(ndf[col], errors='coerce')
                data_by_reg = [ndf[ndf['mass_region'] == r]['_val'].dropna().values
                               for r in ndf['mass_region'].cat.categories]
                bplot = ax.boxplot([d for d in data_by_reg if len(d) > 0],
                                   labels=[r for r, d in zip(ndf['mass_region'].cat.categories,
                                                             data_by_reg) if len(d) > 0],
                                   patch_artist=True)
                colors_reg = ['#4FC3F7', '#29B6F6', '#0288D1', '#01579B', '#0D47A1']
                for patch, col_c in zip(bplot['boxes'], colors_reg):
                    patch.set_facecolor(col_c); patch.set_alpha(0.7)
                ax.set_ylabel(lbl, fontsize=10)
                ax.set_title(lbl, fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', labelsize=8)
            plt.tight_layout()
            fp = dirs['distributions'] / 'mass_region_properties.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Mass region analysis failed: {e}")

        # Chart N8: Feature correlation heatmap (nuclear properties)
        try:
            num_cols = ['A', 'Z', 'N', 'SPIN', 'Beta_2', 'BE_per_A', 'magic_character',
                        'Z_magic_dist', 'N_magic_dist', 'BE_asymmetry', 'Z_valence', 'N_valence']
            existing = [c for c in num_cols if c in ndf.columns]
            corr_data = ndf[existing].apply(pd.to_numeric, errors='coerce').corr()
            fig, ax = plt.subplots(figsize=(13, 11))
            mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
            sns.heatmap(corr_data, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                        vmin=-1, vmax=1, mask=False, square=True,
                        annot_kws={'size': 8}, linewidths=0.5)
            ax.set_title('Nuclear Feature Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fp = dirs['heatmaps'] / 'nuclear_feature_correlation.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Feature correlation heatmap failed: {e}")

        # Chart N9: Nuclear chart colored by mass region (SEGRE chart style)
        try:
            color_map_reg = {'Light\n(A<40)': '#FF9800', 'Medium\n(40-100)': '#4CAF50',
                             'Heavy\n(100-150)': '#2196F3', 'Very Heavy\n(150-220)': '#9C27B0',
                             'Super Heavy\n(>220)': '#F44336'}
            ndf['mass_region2'] = pd.cut(ndf['A'], bins=[0, 40, 100, 150, 220, 300],
                                          labels=['Light\n(A<40)', 'Medium\n(40-100)',
                                                  'Heavy\n(100-150)', 'Very Heavy\n(150-220)',
                                                  'Super Heavy\n(>220)'])
            fig, ax = plt.subplots(figsize=(14, 10))
            for reg, grp in ndf.groupby('mass_region2'):
                ax.scatter(grp['N'], grp['Z'], label=str(reg).replace('\n', ' '),
                           color=color_map_reg.get(str(reg), '#607D8B'),
                           s=70, alpha=0.85, edgecolors='k', linewidths=0.3)
            for mn in [8, 20, 28, 50, 82, 126]:
                ax.axvline(mn, color='black', linestyle='--', alpha=0.2, linewidth=1)
                ax.axhline(mn, color='black', linestyle='--', alpha=0.2, linewidth=1)
            ax.set_xlabel('Neutron Number (N)', fontsize=13)
            ax.set_ylabel('Proton Number (Z)', fontsize=13)
            ax.set_title('Segre Chart: 267 Nuclei by Mass Region', fontsize=15, fontweight='bold')
            ax.legend(fontsize=10, loc='upper left')
            plt.tight_layout()
            fp = dirs['heatmaps'] / 'segre_chart_mass_regions.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Segre chart failed: {e}")

    # ------------------------------------------------------------------
    # SECTION C: ANFIS Specific Charts
    # ------------------------------------------------------------------
    def _generate_anfis_charts(self, ai_df: Optional[pd.DataFrame], dirs: Dict, generated_files: list):
        """ANFIS-specific visualizations"""
        reports_dir = self._find_reports_dir()
        if reports_dir is None:
            return

        # Try THESIS Excel first, then dedicated ANFIS excel
        anfis_df = None
        thesis_excel = self._find_thesis_excel(reports_dir)
        if thesis_excel:
            try:
                anfis_df = pd.read_excel(thesis_excel, sheet_name='All_ANFIS_Models')
                anfis_df['Target'] = anfis_df['Dataset'].apply(
                    lambda ds: 'MM_QM' if str(ds).startswith('MM_QM')
                    else 'MM' if str(ds).startswith('MM_')
                    else 'QM' if str(ds).startswith('QM_')
                    else 'Beta_2' if str(ds).startswith('Beta_2_')
                    else 'Unknown'
                )
            except Exception:
                pass

        anfis_detail_df = None
        anfis_excel = reports_dir.parent / 'anfis_models' / 'anfis_training_results.xlsx'
        if anfis_excel.exists():
            try:
                anfis_detail_df = pd.read_excel(anfis_excel, sheet_name='All_Results')
            except Exception:
                pass

        # Chart A1: ANFIS Val R2 boxplot by MF Type (from detail)
        if anfis_detail_df is not None and 'MF_Type' in anfis_detail_df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                fig.suptitle('ANFIS Performance by Membership Function Type', fontsize=14, fontweight='bold')
                for ax, metric in zip(axes, ['Val_R2', 'Test_R2']):
                    data_mf = {mf: grp[metric].dropna().values
                               for mf, grp in anfis_detail_df.groupby('MF_Type') if metric in anfis_detail_df}
                    if data_mf:
                        bplot = ax.boxplot(data_mf.values(), labels=data_mf.keys(), patch_artist=True)
                        mf_colors = {'gaussian': '#2196F3', 'bell': '#4CAF50', 'triangle': '#FF9800',
                                     'trapezoid': '#9C27B0', 'subclust': '#F44336'}
                        for patch, lbl in zip(bplot['boxes'], data_mf.keys()):
                            patch.set_facecolor(mf_colors.get(lbl, '#607D8B'))
                            patch.set_alpha(0.75)
                        ax.set_ylabel(metric.replace('_', ' '), fontsize=11)
                        ax.set_title(f'{metric.replace("_"," ")} by MF Type', fontsize=12)
                        ax.axhline(0.8, color='red', linestyle='--', alpha=0.4)
                        ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fp = dirs['comparisons'] / 'anfis_mf_type_comparison.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  ANFIS MF boxplot failed: {e}")

        # Chart A2: ANFIS N_Rules vs Val_R2
        if anfis_detail_df is not None and 'N_Rules' in anfis_detail_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 7))
                mf_colors = {'gaussian': '#2196F3', 'bell': '#4CAF50', 'triangle': '#FF9800',
                             'trapezoid': '#9C27B0', 'subclust': '#F44336'}
                for mf, grp in anfis_detail_df.groupby('MF_Type'):
                    ax.scatter(grp['N_Rules'], grp['Val_R2'],
                               alpha=0.6, s=40, label=mf,
                               color=mf_colors.get(mf, '#607D8B'))
                ax.set_xlabel('Number of Fuzzy Rules', fontsize=12)
                ax.set_ylabel('Validation R²', fontsize=12)
                ax.set_title('ANFIS: Number of Rules vs Validation R²', fontsize=14, fontweight='bold')
                ax.legend(title='MF Type', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.axhline(0.8, color='red', linestyle='--', alpha=0.4, label='R²=0.80')
                plt.tight_layout()
                fp = dirs['scatter'] / 'anfis_nrules_vs_r2.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  ANFIS rules vs R2 failed: {e}")

        # Chart A3: ANFIS Config Comparison (8 configs)
        if anfis_detail_df is not None and 'Config_ID' in anfis_detail_df.columns:
            try:
                cfg_grp = anfis_detail_df.groupby('Config_ID').agg(
                    Val_R2_mean=('Val_R2', 'mean'),
                    Val_R2_std=('Val_R2', 'std'),
                    Test_R2_mean=('Test_R2', 'mean'),
                    Count=('Val_R2', 'count')
                ).reset_index().sort_values('Val_R2_mean', ascending=True)

                fig, ax = plt.subplots(figsize=(12, 8))
                y = range(len(cfg_grp))
                ax.barh(y, cfg_grp['Val_R2_mean'], xerr=cfg_grp['Val_R2_std'],
                        color='#3F51B5', alpha=0.8, capsize=4, label='Val R²')
                ax.scatter(cfg_grp['Test_R2_mean'].values, list(y),
                           color='#F44336', zorder=5, s=70, marker='D', label='Test R²')
                ax.set_yticks(list(y))
                ax.set_yticklabels(cfg_grp['Config_ID'], fontsize=10)
                ax.set_xlabel('Mean R² Score', fontsize=12)
                ax.set_title('ANFIS Configuration Comparison (Mean Val R²)', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.axvline(0.8, color='red', linestyle='--', alpha=0.4)
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                fp = dirs['performance'] / 'anfis_config_comparison.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  ANFIS config comparison failed: {e}")

        # Chart A4: AI vs ANFIS scatter per dataset
        if anfis_df is not None and ai_df is not None:
            try:
                ai_best = ai_df.groupby('Dataset')['Val_R2'].max().reset_index()
                ai_best.columns = ['Dataset', 'AI_Best_Val_R2']
                anfis_best = anfis_df.groupby('Dataset')['Val_R2'].max().reset_index()
                anfis_best.columns = ['Dataset', 'ANFIS_Best_Val_R2']
                merged = ai_best.merge(anfis_best, on='Dataset').dropna()

                if len(merged) > 0:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter(merged['AI_Best_Val_R2'], merged['ANFIS_Best_Val_R2'],
                               alpha=0.65, s=50, color='#607D8B', edgecolors='k', linewidths=0.3)
                    lims = [min(merged[['AI_Best_Val_R2', 'ANFIS_Best_Val_R2']].min()),
                            max(merged[['AI_Best_Val_R2', 'ANFIS_Best_Val_R2']].max())]
                    ax.plot(lims, lims, 'k--', alpha=0.4, label='AI = ANFIS')
                    # Color above/below line
                    above = merged[merged['ANFIS_Best_Val_R2'] > merged['AI_Best_Val_R2']]
                    below = merged[merged['ANFIS_Best_Val_R2'] <= merged['AI_Best_Val_R2']]
                    ax.scatter(above['AI_Best_Val_R2'], above['ANFIS_Best_Val_R2'],
                               color='#4CAF50', alpha=0.7, s=60, label=f'ANFIS wins ({len(above)})')
                    ax.scatter(below['AI_Best_Val_R2'], below['ANFIS_Best_Val_R2'],
                               color='#F44336', alpha=0.7, s=60, label=f'AI wins ({len(below)})')
                    ax.set_xlabel('AI Best Val R²', fontsize=12)
                    ax.set_ylabel('ANFIS Best Val R²', fontsize=12)
                    ax.set_title('AI vs ANFIS Performance per Dataset', fontsize=14, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    fp = dirs['scatter'] / 'ai_vs_anfis_per_dataset_scatter.png'
                    fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                    generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  AI vs ANFIS scatter failed: {e}")

        # Chart A5: ANFIS R2 distribution by Target
        if anfis_detail_df is not None and 'Target' in anfis_detail_df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle('ANFIS R² Distribution by Target', fontsize=14, fontweight='bold')
                for ax, metric in zip(axes, ['Val_R2', 'Test_R2']):
                    for target, grp in anfis_detail_df.groupby('Target'):
                        vals = grp[metric].dropna()
                        if len(vals) > 0:
                            vals.hist(ax=ax, bins=20, alpha=0.5, label=target, density=True)
                    ax.set_xlabel(metric.replace('_', ' '), fontsize=11)
                    ax.set_ylabel('Density', fontsize=11)
                    ax.set_title(metric.replace('_', ' '), fontsize=12)
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fp = dirs['distributions'] / 'anfis_r2_by_target.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  ANFIS R2 by target failed: {e}")

    # ------------------------------------------------------------------
    # SECTION D: Actual vs Predicted + Residual Charts
    # ------------------------------------------------------------------
    def _generate_prediction_charts(self, dirs: Dict, generated_files: list):
        """Load best models and generate actual vs predicted scatter plots"""
        reports_dir = self._find_reports_dir()
        if reports_dir is None:
            return
        outputs_dir = reports_dir.parent
        datasets_dir = outputs_dir / 'generated_datasets'
        models_dir = outputs_dir / 'trained_models'
        if not datasets_dir.exists() or not models_dir.exists():
            return

        import joblib

        for target, prefix, col_hint in [
            ('MM', 'MM_', 'MAGNETIC'),
            ('QM', 'QM_', 'QUADRUPOLE'),
            ('Beta_2', 'Beta_2_', 'Beta_2'),
        ]:
            try:
                # Find best dataset+model by scanning metrics
                best_r2, best_model, best_test_csv, best_features = -999, None, None, None
                for ds_dir in sorted(models_dir.iterdir()):
                    if not ds_dir.is_dir() or not ds_dir.name.startswith(prefix):
                        continue
                    for mt_dir in ds_dir.iterdir():
                        if not mt_dir.is_dir():
                            continue
                        for cfg_dir in mt_dir.iterdir():
                            if not cfg_dir.is_dir():
                                continue
                            mf = cfg_dir / f'metrics_{cfg_dir.name}.json'
                            if not mf.exists():
                                continue
                            with open(mf, encoding='utf-8') as f:
                                m = json.load(f)
                            r2 = m.get('val', {}).get('r2', -999)
                            if r2 > best_r2:
                                pkls = list(cfg_dir.glob('*.pkl'))
                                if pkls:
                                    # Find meta
                                    meta_path = datasets_dir / ds_dir.name / 'metadata.json'
                                    test_path = datasets_dir / ds_dir.name / 'test.csv'
                                    if meta_path.exists() and test_path.exists():
                                        with open(meta_path, encoding='utf-8') as f:
                                            meta = json.load(f)
                                        best_r2 = r2
                                        best_model = pkls[0]
                                        best_test_csv = test_path
                                        best_features = meta.get('feature_names', [])

                if best_model is None or best_test_csv is None:
                    continue

                # Load test data and model
                test_df = pd.read_csv(best_test_csv)
                target_col = next((c for c in test_df.columns if col_hint.upper() in c.upper()), None)
                if target_col is None:
                    # Try direct target name match
                    target_col = next((c for c in test_df.columns if target in c), None)
                if target_col is None:
                    continue

                feat_avail = [f for f in best_features if f in test_df.columns]
                if not feat_avail:
                    continue

                X_test = test_df[feat_avail].apply(pd.to_numeric, errors='coerce').fillna(0).values
                y_true = pd.to_numeric(test_df[target_col], errors='coerce').values
                mask = ~np.isnan(y_true)
                X_test, y_true = X_test[mask], y_true[mask]

                if len(y_true) < 3:
                    continue

                model = joblib.load(best_model)
                y_pred = model.predict(X_test).flatten()

                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                residuals = y_true - y_pred

                # Actual vs Predicted
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                fig.suptitle(f'{target}: Best Model Predictions (Val R²={best_r2:.3f})',
                             fontsize=14, fontweight='bold')

                ax = axes[0]
                ax.scatter(y_true, y_pred, alpha=0.8, s=60, color='#2196F3',
                           edgecolors='k', linewidths=0.4)
                lim = [min(y_true.min(), y_pred.min()) - 0.1,
                       max(y_true.max(), y_pred.max()) + 0.1]
                ax.plot(lim, lim, 'r--', linewidth=2, label='Perfect')
                ax.set_xlabel('Actual Value', fontsize=12)
                ax.set_ylabel('Predicted Value', fontsize=12)
                ax.set_title(f'Actual vs Predicted\nR²={r2:.3f}  MAE={mae:.3f}', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                # Add nucleus labels if available
                if 'NUCLEUS' in test_df.columns:
                    nuc = test_df.loc[mask, 'NUCLEUS'].values
                    for i, (xt, xp, n) in enumerate(zip(y_true, y_pred, nuc)):
                        if abs(xt - xp) > 2 * np.std(residuals):
                            ax.annotate(str(n), (xt, xp), fontsize=7, alpha=0.7)

                # Residual plot
                ax2 = axes[1]
                ax2.scatter(y_pred, residuals, alpha=0.8, s=60, color='#FF5722',
                            edgecolors='k', linewidths=0.4)
                ax2.axhline(0, color='black', linewidth=1.5)
                ax2.axhline(np.std(residuals), color='red', linestyle='--', alpha=0.5, label='+1σ')
                ax2.axhline(-np.std(residuals), color='red', linestyle='--', alpha=0.5, label='-1σ')
                ax2.set_xlabel('Predicted Value', fontsize=12)
                ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
                ax2.set_title('Residual Analysis', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                fp = dirs['scatter'] / f'{target.lower()}_actual_vs_predicted.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")

                # Residual Histogram
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f'{target}: Residual Distribution Analysis', fontsize=13, fontweight='bold')
                axes[0].hist(residuals, bins=20, color='#9C27B0', edgecolor='white', alpha=0.8)
                axes[0].axvline(0, color='black', linewidth=1.5)
                axes[0].axvline(np.mean(residuals), color='red', linestyle='--',
                                label=f'Mean={np.mean(residuals):.3f}')
                axes[0].set_xlabel('Residual', fontsize=11); axes[0].set_ylabel('Count', fontsize=11)
                axes[0].set_title('Residual Histogram', fontsize=11); axes[0].legend(); axes[0].grid(True, alpha=0.3)
                from scipy import stats as sp_stats
                sp_stats.probplot(residuals, dist='norm', plot=axes[1])
                axes[1].set_title('Normal Q-Q Plot of Residuals', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                plt.tight_layout()
                fp = dirs['distributions'] / f'{target.lower()}_residual_analysis.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")

            except Exception as e:
                logger.warning(f"  Prediction chart {target} failed: {e}")

    # ------------------------------------------------------------------
    # SECTION E: Feature Analysis Charts
    # ------------------------------------------------------------------
    def _generate_feature_charts(self, ai_df: Optional[pd.DataFrame], dirs: Dict, generated_files: list):
        """Feature set rankings and analysis"""
        if ai_df is None:
            return

        # Chart F1: Feature set performance ranking per target
        def extract_feature_set(ds):
            parts = str(ds).split('_')
            # Feature code is the 4th segment (after target, size, scenario)
            if len(parts) >= 5:
                return parts[3]
            return 'Unknown'

        try:
            ai_df = ai_df.copy()
            ai_df['FeatureSet'] = ai_df['Dataset'].apply(extract_feature_set)
            targets = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]

            fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 10))
            if len(targets) == 1:
                axes = [axes]
            fig.suptitle('Feature Set Performance Ranking by Target (Mean Val R²)',
                         fontsize=14, fontweight='bold')

            for ax, target in zip(axes, targets):
                fs_grp = (ai_df[ai_df['Target'] == target]
                          .groupby('FeatureSet')['Val_R2']
                          .agg(['mean', 'std', 'count'])
                          .reset_index()
                          .sort_values('mean', ascending=True))
                fs_top = fs_grp.tail(20)  # Top 20 feature sets

                y = range(len(fs_top))
                ax.barh(y, fs_top['mean'], xerr=fs_top['std'],
                        color='#1976D2', alpha=0.8, capsize=3,
                        error_kw={'elinewidth': 1, 'capthick': 1})
                ax.set_yticks(list(y))
                ax.set_yticklabels(fs_top['FeatureSet'], fontsize=9)
                ax.set_xlabel('Mean Val R²', fontsize=11)
                ax.set_title(f'{target}', fontsize=13, fontweight='bold')
                ax.axvline(0.8, color='red', linestyle='--', alpha=0.4)
                ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            fp = dirs['performance'] / 'feature_set_ranking_per_target.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Feature set ranking failed: {e}")

        # Chart F2: Violin plot - Val R2 by model type x target
        try:
            targets_sel = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]
            plot_df = ai_df[ai_df['Target'].isin(targets_sel)].copy()
            fig, ax = plt.subplots(figsize=(14, 8))
            # Create position mapping
            model_types = sorted(plot_df['Model_Type'].unique())
            positions = []
            labels = []
            violin_data = []
            tick_pos = []
            group_width = len(model_types) + 1
            colors_v = {'RF': '#2196F3', 'XGBoost': '#4CAF50', 'DNN': '#FF9800'}

            for t_idx, target in enumerate(targets_sel):
                base = t_idx * group_width
                tick_pos.append(base + len(model_types) / 2)
                labels.append(target)
                for m_idx, mt in enumerate(model_types):
                    data = plot_df[(plot_df['Target'] == target) &
                                   (plot_df['Model_Type'] == mt)]['Val_R2'].dropna().values
                    if len(data) > 2:
                        parts = ax.violinplot(data, positions=[base + m_idx],
                                              widths=0.8, showmeans=True, showmedians=True)
                        for pc in parts['bodies']:
                            pc.set_facecolor(colors_v.get(mt, '#607D8B'))
                            pc.set_alpha(0.65)

            ax.set_xticks(tick_pos)
            ax.set_xticklabels(targets_sel, fontsize=12)
            ax.set_ylabel('Validation R²', fontsize=12)
            ax.set_title('Val R² Violin Plot: Target × Model Type', fontsize=14, fontweight='bold')
            # Legend
            from matplotlib.patches import Patch
            legend_patches = [Patch(facecolor=colors_v.get(mt, '#607D8B'), alpha=0.65, label=mt)
                              for mt in model_types]
            ax.legend(handles=legend_patches, fontsize=11, title='Model Type')
            ax.axhline(0.8, color='red', linestyle='--', alpha=0.4)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            fp = dirs['distributions'] / 'violin_val_r2_target_model.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  Violin plot failed: {e}")

        # Chart F3: R2 thresholds - % achieving R2 > 0.7, 0.8, 0.9
        try:
            targets_sel = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]
            thresholds = [0.7, 0.8, 0.9, 0.95]
            fig, ax = plt.subplots(figsize=(12, 7))
            x = np.arange(len(targets_sel))
            w = 0.18
            threshold_colors = ['#FFF176', '#A5D6A7', '#42A5F5', '#1565C0']
            for i, (thresh, col) in enumerate(zip(thresholds, threshold_colors)):
                pcts = []
                for target in targets_sel:
                    vals = ai_df[ai_df['Target'] == target]['Val_R2'].dropna()
                    pcts.append(100 * (vals >= thresh).sum() / len(vals) if len(vals) > 0 else 0)
                bars = ax.bar(x + (i - 1.5) * w, pcts, w, label=f'Val R² ≥ {thresh}',
                              color=col, alpha=0.9, edgecolor='gray', linewidth=0.5)
                ax.bar_label(bars, fmt='%.0f%%', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(targets_sel, fontsize=12)
            ax.set_ylabel('% of Models Achieving Threshold', fontsize=12)
            ax.set_title('Success Rate by R² Threshold and Target', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.set_ylim(0, 115)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            fp = dirs['comparisons'] / 'r2_threshold_success_rate.png'
            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
        except Exception as e:
            logger.warning(f"  R2 threshold chart failed: {e}")

        # Chart F4: Training Time distribution by model type
        if 'Training_Time_s' in ai_df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle('Model Training Time Analysis', fontsize=14, fontweight='bold')
                colors_mt = {'RF': '#2196F3', 'XGBoost': '#4CAF50', 'DNN': '#FF9800'}
                for mt, grp in ai_df.groupby('Model_Type'):
                    vals = grp['Training_Time_s'].dropna()
                    if len(vals) > 0:
                        vals.hist(ax=axes[0], bins=30, alpha=0.5, label=mt,
                                  color=colors_mt.get(mt, '#607D8B'), density=True)
                axes[0].set_xlabel('Training Time (s)', fontsize=11)
                axes[0].set_ylabel('Density', fontsize=11)
                axes[0].set_title('Training Time Distribution', fontsize=12)
                axes[0].legend(fontsize=10)
                axes[0].grid(True, alpha=0.3)

                tt_grp = ai_df.groupby('Model_Type')['Training_Time_s'].agg(['mean', 'std', 'median']).reset_index()
                x = np.arange(len(tt_grp))
                axes[1].bar(x, tt_grp['mean'], color=[colors_mt.get(m, '#607D8B') for m in tt_grp['Model_Type']],
                            alpha=0.8, yerr=tt_grp['std'], capsize=6)
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(tt_grp['Model_Type'], fontsize=12)
                axes[1].set_ylabel('Mean Training Time (s)', fontsize=11)
                axes[1].set_title('Mean Training Time by Model Type', fontsize=12)
                axes[1].grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                fp = dirs['distributions'] / 'training_time_analysis.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  Training time chart failed: {e}")

    # ------------------------------------------------------------------
    # SECTION F: 3D Charts
    # ------------------------------------------------------------------
    def _generate_3d_charts(self, ai_df: Optional[pd.DataFrame], dirs: Dict, generated_files: list):
        """3D matplotlib and 3D nuclear surface charts"""
        dirs_3d = self.output_dir / '3d_plots'
        dirs_3d.mkdir(parents=True, exist_ok=True)

        # Chart 3D-1: 3D scatter - Val R2 / Test R2 / Training Time
        if ai_df is not None and 'Training_Time_s' in ai_df.columns:
            try:
                sample = ai_df.dropna(subset=['Val_R2', 'Test_R2', 'Training_Time_s']).sample(
                    min(800, len(ai_df)), random_state=42
                )
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')
                colors_3d = {'RF': '#2196F3', 'XGBoost': '#4CAF50', 'DNN': '#FF9800'}
                for mt, grp in sample.groupby('Model_Type'):
                    ax.scatter(grp['Val_R2'], grp['Test_R2'], np.log1p(grp['Training_Time_s']),
                               alpha=0.6, s=30, color=colors_3d.get(mt, '#607D8B'),
                               label=mt, edgecolors='none')
                ax.set_xlabel('Val R²', fontsize=11, labelpad=8)
                ax.set_ylabel('Test R²', fontsize=11, labelpad=8)
                ax.set_zlabel('log(Training Time + 1)', fontsize=11, labelpad=8)
                ax.set_title('3D: Val R² / Test R² / Training Time', fontsize=13, fontweight='bold')
                ax.legend(fontsize=10)
                plt.tight_layout()
                fp = dirs_3d / '3d_val_test_traintime.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  3D scatter failed: {e}")

        # Chart 3D-2: 3D surface - Z, N, Magnetic Moment (from AAA2 data)
        reports_dir = self._find_reports_dir()
        if reports_dir:
            aaa2_path = reports_dir.parent / 'generated_datasets' / 'AAA2_enriched_all_nuclei.csv'
            if aaa2_path.exists():
                try:
                    ndf = pd.read_csv(aaa2_path)
                    mm_col = next((c for c in ndf.columns if 'MAGNETIC' in c.upper()), None)
                    qm_col = next((c for c in ndf.columns if 'QUADRUPOLE' in c.upper()), None)

                    for col, title, fname in [
                        (mm_col, 'Magnetic Moment (μN)', '3d_nuclear_mm_surface.png'),
                        (qm_col, 'Quadrupole Moment (barn)', '3d_nuclear_qm_surface.png'),
                        ('Beta_2', 'Beta_2 Deformation', '3d_nuclear_beta2_surface.png'),
                    ]:
                        if col is None:
                            continue
                        try:
                            ndf['_v'] = pd.to_numeric(ndf[col], errors='coerce')
                            valid = ndf.dropna(subset=['Z', 'N', '_v'])
                            if len(valid) < 5:
                                continue

                            fig = plt.figure(figsize=(14, 10))
                            ax = fig.add_subplot(111, projection='3d')
                            sc = ax.scatter(valid['N'], valid['Z'], valid['_v'],
                                            c=valid['_v'], cmap='viridis', s=50, alpha=0.8)
                            fig.colorbar(sc, ax=ax, shrink=0.5, label=title)
                            ax.set_xlabel('Neutron Number (N)', fontsize=11, labelpad=8)
                            ax.set_ylabel('Proton Number (Z)', fontsize=11, labelpad=8)
                            ax.set_zlabel(title, fontsize=11, labelpad=8)
                            ax.set_title(f'3D Nuclear Chart: {title}', fontsize=13, fontweight='bold')
                            plt.tight_layout()
                            fp = dirs_3d / fname
                            fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                            generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
                        except Exception as e2:
                            logger.warning(f"  3D nuclear {col} failed: {e2}")

                    # Chart 3D-5: 3D scatter - A, Z, N colored by Beta_2
                    try:
                        ndf['_b2'] = pd.to_numeric(ndf['Beta_2'], errors='coerce')
                        fig = plt.figure(figsize=(14, 10))
                        ax = fig.add_subplot(111, projection='3d')
                        sc = ax.scatter(ndf['A'], ndf['Z'], ndf['N'],
                                        c=ndf['_b2'], cmap='RdYlGn', s=40, alpha=0.75)
                        fig.colorbar(sc, ax=ax, shrink=0.5, label='Beta_2 Deformation')
                        ax.set_xlabel('Mass Number (A)', fontsize=11, labelpad=8)
                        ax.set_ylabel('Proton Number (Z)', fontsize=11, labelpad=8)
                        ax.set_zlabel('Neutron Number (N)', fontsize=11, labelpad=8)
                        ax.set_title('3D Nuclear Space: A, Z, N colored by β₂', fontsize=13, fontweight='bold')
                        plt.tight_layout()
                        fp = dirs_3d / '3d_AZN_by_beta2.png'
                        fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                        generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
                    except Exception as e:
                        logger.warning(f"  3D AZN failed: {e}")

                except Exception as e:
                    logger.warning(f"  3D nuclear charts failed: {e}")

        # Chart 3D-6: 3D bar - Target x Model Type x Mean R2
        if ai_df is not None:
            try:
                targets_sel = [t for t in ['MM', 'QM', 'MM_QM', 'Beta_2'] if t in ai_df['Target'].unique()]
                model_types = sorted(ai_df['Model_Type'].unique())
                fig = plt.figure(figsize=(14, 10))
                ax = fig.add_subplot(111, projection='3d')
                colors_3db = ['#2196F3', '#4CAF50', '#FF9800']
                for m_idx, (mt, col) in enumerate(zip(model_types, colors_3db)):
                    for t_idx, target in enumerate(targets_sel):
                        val = ai_df[(ai_df['Model_Type'] == mt) & (ai_df['Target'] == target)]['Val_R2'].mean()
                        if not np.isnan(val):
                            ax.bar3d(t_idx - 0.35 + m_idx * 0.23, m_idx - 0.1, 0,
                                     0.22, 0.2, val, color=col, alpha=0.75)
                ax.set_xticks(range(len(targets_sel)))
                ax.set_xticklabels(targets_sel, fontsize=10)
                ax.set_yticks(range(len(model_types)))
                ax.set_yticklabels(model_types, fontsize=10)
                ax.set_zlabel('Mean Val R²', fontsize=11)
                ax.set_title('3D Bar: Target × Model Type × Mean Val R²', fontsize=12, fontweight='bold')
                plt.tight_layout()
                fp = dirs_3d / '3d_bar_target_model_r2.png'
                fig.savefig(fp, dpi=200, bbox_inches='tight'); plt.close()
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  3D bar chart failed: {e}")

    # ------------------------------------------------------------------
    # SECTION G: Extended Interactive HTML Charts
    # ------------------------------------------------------------------
    def _generate_interactive_extended(self, ai_df: Optional[pd.DataFrame],
                                       dirs: Dict, generated_files: list):
        """Extended Plotly interactive HTML charts"""
        if not PLOTLY_AVAILABLE:
            return

        # Chart I4: Interactive Nuclear Chart (Z vs N colored by Beta_2 with hover)
        reports_dir = self._find_reports_dir()
        if reports_dir:
            aaa2_path = reports_dir.parent / 'generated_datasets' / 'AAA2_enriched_all_nuclei.csv'
            if aaa2_path.exists():
                try:
                    ndf = pd.read_csv(aaa2_path)
                    ndf.columns = [c.strip() for c in ndf.columns]
                    mm_col = next((c for c in ndf.columns if 'MAGNETIC' in c.upper()), None)
                    qm_col = next((c for c in ndf.columns if 'QUADRUPOLE' in c.upper()), None)
                    ndf['Beta_2_num'] = pd.to_numeric(ndf['Beta_2'], errors='coerce')
                    ndf['MM_num'] = pd.to_numeric(ndf.get(mm_col, pd.Series(dtype=float)), errors='coerce') if mm_col else np.nan
                    ndf['QM_num'] = pd.to_numeric(ndf.get(qm_col, pd.Series(dtype=float)), errors='coerce') if qm_col else np.nan

                    hover_txt = ndf.apply(lambda r: (
                        f"Nucleus: {r.get('NUCLEUS','?')}<br>"
                        f"Z={r['Z']}, N={r['N']}, A={r['A']}<br>"
                        f"β₂={r['Beta_2_num']:.3f}<br>"
                        f"MM={r['MM_num']:.3f}<br>"
                        f"Spin={r.get('SPIN','?')}"
                    ), axis=1)

                    fig_nc = go.Figure(go.Scatter(
                        x=ndf['N'], y=ndf['Z'],
                        mode='markers',
                        marker=dict(size=10, color=ndf['Beta_2_num'], colorscale='RdYlGn',
                                    colorbar=dict(title='β₂'), showscale=True,
                                    line=dict(width=0.5, color='black')),
                        text=hover_txt, hoverinfo='text'
                    ))
                    # Magic number lines
                    for mn in [8, 20, 28, 50, 82, 126]:
                        fig_nc.add_vline(x=mn, line_dash='dash', line_color='gray', opacity=0.4)
                        fig_nc.add_hline(y=mn, line_dash='dash', line_color='gray', opacity=0.4)
                    fig_nc.update_layout(
                        title='Interactive Nuclear Chart: β₂ Deformation (267 Nuclei)',
                        xaxis_title='Neutron Number (N)', yaxis_title='Proton Number (Z)',
                        hovermode='closest', width=1100, height=800,
                        font=dict(size=12)
                    )
                    fp = dirs['interactive'] / 'interactive_nuclear_chart.html'
                    fig_nc.write_html(str(fp))
                    generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
                except Exception as e:
                    logger.warning(f"  Interactive nuclear chart failed: {e}")

                # Chart I5: 3D Interactive Nuclear Chart (Plotly)
                try:
                    ndf2 = pd.read_csv(aaa2_path)
                    ndf2.columns = [c.strip() for c in ndf2.columns]
                    mm_col2 = next((c for c in ndf2.columns if 'MAGNETIC' in c.upper()), None)
                    ndf2['MM_num'] = pd.to_numeric(ndf2[mm_col2], errors='coerce') if mm_col2 else 0
                    ndf2['Beta2_num'] = pd.to_numeric(ndf2['Beta_2'], errors='coerce').fillna(0)

                    fig_3d = go.Figure(data=[go.Scatter3d(
                        x=ndf2['N'], y=ndf2['Z'], z=ndf2['MM_num'].fillna(0),
                        mode='markers',
                        marker=dict(size=6, color=ndf2['Beta2_num'],
                                    colorscale='Viridis', opacity=0.8,
                                    colorbar=dict(title='β₂ Deformation'),
                                    line=dict(width=0.3, color='black')),
                        text=[f"Nucleus:{r.get('NUCLEUS','?')}<br>Z={r['Z']},N={r['N']}<br>"
                              f"MM={r['MM_num']:.3f}<br>β₂={r['Beta2_num']:.3f}"
                              for _, r in ndf2.iterrows()],
                        hoverinfo='text'
                    )])
                    fig_3d.update_layout(
                        title='3D Interactive: Z, N, Magnetic Moment (colored by β₂)',
                        scene=dict(
                            xaxis_title='Neutron Number (N)',
                            yaxis_title='Proton Number (Z)',
                            zaxis_title='Magnetic Moment (μN)'
                        ),
                        width=1100, height=800
                    )
                    fp = dirs['interactive'] / 'interactive_3d_nuclear_mm.html'
                    fig_3d.write_html(str(fp))
                    generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
                except Exception as e:
                    logger.warning(f"  Interactive 3D nuclear failed: {e}")

        # Chart I6: Interactive Feature Set Analysis (Plotly)
        if ai_df is not None:
            try:
                ai_df2 = ai_df.copy()

                def _fs(ds):
                    parts = str(ds).split('_')
                    return parts[3] if len(parts) >= 4 else 'Unknown'

                ai_df2['FeatureSet'] = ai_df2['Dataset'].apply(_fs)
                fs_grp = ai_df2.groupby(['FeatureSet', 'Target', 'Model_Type'])['Val_R2'].agg(
                    ['mean', 'std', 'count']).reset_index()
                fs_grp.columns = ['FeatureSet', 'Target', 'Model_Type', 'Mean_Val_R2', 'Std', 'Count']

                fig_fs = px.scatter(
                    fs_grp, x='FeatureSet', y='Mean_Val_R2', color='Target',
                    symbol='Model_Type', size='Count',
                    error_y='Std',
                    title='Interactive: Feature Set Performance by Target & Model Type',
                    labels={'Mean_Val_R2': 'Mean Val R²', 'FeatureSet': 'Feature Set'},
                    hover_data=['Count', 'Std'],
                    width=1200, height=700
                )
                fig_fs.update_layout(xaxis_tickangle=-45)
                fp = dirs['interactive'] / 'interactive_feature_set_analysis.html'
                fig_fs.write_html(str(fp))
                generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  Interactive feature set failed: {e}")

        # Chart I7: Interactive ANFIS vs AI comparison
        if ai_df is not None:
            try:
                reports_dir2 = self._find_reports_dir()
                if reports_dir2:
                    thesis_xl = self._find_thesis_excel(reports_dir2)
                    if thesis_xl:
                        anfis_cmp = pd.read_excel(thesis_xl, sheet_name='AI_vs_ANFIS_Comparison')
                        anfis_cmp['Target'] = anfis_cmp['Dataset'].apply(
                            lambda ds: 'MM' if str(ds).startswith('MM_') and not str(ds).startswith('MM_QM')
                            else 'QM' if str(ds).startswith('QM_')
                            else 'Beta_2' if str(ds).startswith('Beta_2_')
                            else 'MM_QM'
                        )
                        fig_cmp = px.scatter(
                            anfis_cmp.dropna(subset=['AI_Best_Val_R2', 'ANFIS_Best_Val_R2']),
                            x='AI_Best_Val_R2', y='ANFIS_Best_Val_R2',
                            color='Target', hover_data=['Dataset'],
                            title='Interactive: AI vs ANFIS Best Val R² per Dataset',
                            labels={'AI_Best_Val_R2': 'AI Best Val R²',
                                    'ANFIS_Best_Val_R2': 'ANFIS Best Val R²'},
                            width=1000, height=700, opacity=0.7
                        )
                        fig_cmp.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                                          line=dict(dash='dash', color='black', width=1))
                        fig_cmp.update_layout(
                            annotations=[dict(x=0.7, y=0.7, text='AI = ANFIS', showarrow=False,
                                              font=dict(color='black', size=11))]
                        )
                        fp = dirs['interactive'] / 'interactive_ai_vs_anfis.html'
                        fig_cmp.write_html(str(fp))
                        generated_files.append(str(fp)); logger.info(f"  [OK] {fp.name}")
            except Exception as e:
                logger.warning(f"  Interactive AI vs ANFIS failed: {e}")

    # ------------------------------------------------------------------
    # SECTION H: Fill all empty folders with real nuclear data charts
    # ------------------------------------------------------------------
    def _generate_all_empty_folder_charts(self, ai_df: Optional[pd.DataFrame], generated_files: list):
        """
        Generate charts for the 8 empty folders using real AAA2 data and PFAZ9 predictions.
        Folders: predictions, features, anomaly, robustness, shap, master_report,
                 optimization, training_metrics
        """
        # ---- Locate real data ----
        project_root = Path(__file__).resolve().parents[2]
        aaa2_csv = project_root / 'outputs' / 'aaa2_results' / 'aaa2_enriched_with_theory.csv'
        mm_xlsx  = project_root / 'outputs' / 'aaa2_results' / 'AAA2_Complete_MM.xlsx'
        qm_xlsx  = project_root / 'outputs' / 'aaa2_results' / 'AAA2_Complete_QM.xlsx'
        b2_xlsx  = project_root / 'outputs' / 'aaa2_results' / 'AAA2_Complete_Beta_2.xlsx'

        aaa2 = None
        if aaa2_csv.exists():
            try:
                aaa2 = pd.read_csv(aaa2_csv)
                # Normalize column names
                # Find and rename MAGNETIC MOMENT column (encoding-safe)
                rename_map = {}
                for c in aaa2.columns:
                    cu = c.upper()
                    if 'MAGNET' in cu and 'MM' not in rename_map.values():
                        rename_map[c] = 'MM'
                    elif 'QUADRUPOLE' in cu and 'QM' not in rename_map.values():
                        rename_map[c] = 'QM'
                rename_map['Beta_2'] = 'Beta2_exp'
                aaa2.rename(columns=rename_map, inplace=True)
                # Ensure MM/QM/Beta2_exp exist
                if 'MM' not in aaa2.columns:
                    aaa2['MM'] = np.nan
                if 'QM' not in aaa2.columns:
                    aaa2['QM'] = np.nan
                if 'Beta2_exp' not in aaa2.columns:
                    aaa2['Beta2_exp'] = aaa2.get('Beta_2_estimated', np.nan)
                aaa2['MM'] = pd.to_numeric(aaa2['MM'], errors='coerce')
                aaa2['QM'] = pd.to_numeric(aaa2['QM'], errors='coerce')
                aaa2['Beta2_exp'] = pd.to_numeric(aaa2['Beta2_exp'], errors='coerce')
                aaa2['Z'] = pd.to_numeric(aaa2['Z'], errors='coerce')
                aaa2['N'] = pd.to_numeric(aaa2['N'], errors='coerce')
                aaa2['A'] = pd.to_numeric(aaa2['A'], errors='coerce')
                # Mass region
                def _mass_region(a):
                    if pd.isna(a): return 'Unknown'
                    if a < 40: return 'Light (A<40)'
                    if a < 100: return 'Medium (40-100)'
                    if a < 160: return 'Heavy (100-160)'
                    return 'Very Heavy (>160)'
                aaa2['mass_region'] = aaa2['A'].apply(_mass_region)
            except Exception as e:
                logger.warning(f"  Cannot load aaa2_enriched_with_theory.csv: {e}")
                aaa2 = None

        # Load PFAZ9 uncertainty predictions (MM, QM, Beta_2)
        def _load_uncertainty(path, col_name):
            if not path.exists():
                return None
            try:
                df = pd.read_excel(path, sheet_name='Uncertainty')
                df = df[['NUCLEUS', 'Mean_Prediction', 'Std_Prediction', 'CI_Lower', 'CI_Upper', 'CV']].copy()
                df.rename(columns={'Mean_Prediction': col_name+'_pred',
                                   'Std_Prediction':  col_name+'_std',
                                   'CI_Lower':        col_name+'_ci_lo',
                                   'CI_Upper':        col_name+'_ci_hi',
                                   'CV':              col_name+'_cv'}, inplace=True)
                return df
            except Exception as e:
                logger.warning(f"  Cannot load uncertainty from {path.name}: {e}")
                return None

        mm_pred  = _load_uncertainty(mm_xlsx,  'MM')
        qm_pred  = _load_uncertainty(qm_xlsx,  'QM')
        b2_pred  = _load_uncertainty(b2_xlsx, 'B2')

        # Merge predictions onto AAA2
        merged = aaa2.copy() if aaa2 is not None else pd.DataFrame()
        if not merged.empty:
            for pred_df in [mm_pred, qm_pred, b2_pred]:
                if pred_df is not None:
                    merged = merged.merge(pred_df, on='NUCLEUS', how='left')

        # Define output directories
        out_root = self.output_dir
        dir_map = {
            'predictions':    out_root / 'predictions',
            'features':       out_root / 'features',
            'anomaly':        out_root / 'anomaly',
            'robustness':     out_root / 'robustness',
            'shap':           out_root / 'shap',
            'master_report':  out_root / 'master_report',
            'optimization':   out_root / 'optimization',
            'training_metrics': out_root / 'training_metrics',
        }
        for d in dir_map.values():
            d.mkdir(parents=True, exist_ok=True)

        def _save(fig, path):
            try:
                fig.savefig(str(path), dpi=120, bbox_inches='tight')
                plt.close(fig)
                generated_files.append(str(path))
                logger.info(f"  [OK] {path.name}")
            except Exception as e:
                logger.warning(f"  Save failed {path.name}: {e}")
                plt.close(fig)

        MAGIC_N = {2, 8, 20, 28, 50, 82, 126}
        MAGIC_Z = {2, 8, 20, 28, 50, 82}

        # ================================================================
        # PREDICTIONS FOLDER
        # ================================================================
        try:
            if not merged.empty and 'MM_pred' in merged.columns:
                # 1. Bar: all 267 nuclei MM experimental
                fig, ax = plt.subplots(figsize=(18, 5))
                mm_valid = merged.dropna(subset=['MM']).sort_values('A')
                colors = {'Light (A<40)': '#2196F3', 'Medium (40-100)': '#4CAF50',
                          'Heavy (100-160)': '#FF9800', 'Very Heavy (>160)': '#F44336'}
                bar_colors = [colors.get(r, '#9E9E9E') for r in mm_valid['mass_region']]
                ax.bar(range(len(mm_valid)), mm_valid['MM'], color=bar_colors, alpha=0.8, width=0.9)
                ax.axhline(0, color='black', lw=0.8, ls='--')
                ax.set_title('267 Cilek Manyetik Moment (MM) - Deneysel Deger', fontsize=13)
                ax.set_xlabel('Nukleus (A sirali)')
                ax.set_ylabel('Manyetik Moment [nm]')
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                _save(fig, dir_map['predictions'] / 'predictions_mm_experimental_all_nuclei.png')

                # 2. Experimental vs Predicted scatter MM
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                targets = [('MM', 'MM_pred', 'Manyetik Moment [nm]'),
                           ('QM', 'QM_pred', 'Kuadrupol Moment [Q]'),
                           ('Beta2_exp', 'B2_pred', 'Beta_2 Deformasyon')]
                for ax, (exp_col, pred_col, ylabel) in zip(axes, targets):
                    if exp_col in merged.columns and pred_col in merged.columns:
                        sub = merged.dropna(subset=[exp_col, pred_col])
                        ax.scatter(sub[exp_col], sub[pred_col], alpha=0.6, s=30, color='steelblue')
                        mn = min(sub[exp_col].min(), sub[pred_col].min())
                        mx = max(sub[exp_col].max(), sub[pred_col].max())
                        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
                        r2 = np.corrcoef(sub[exp_col], sub[pred_col])[0, 1]**2 if len(sub) > 2 else 0
                        ax.set_title(f'{ylabel}\nR²={r2:.3f}', fontsize=10)
                        ax.set_xlabel('Deneysel')
                        ax.set_ylabel('Tahmin (PFAZ9 Ens.)')
                fig.suptitle('267 Nukleus: Deneysel vs PFAZ9 Tahmin', fontsize=14, fontweight='bold')
                plt.tight_layout()
                _save(fig, dir_map['predictions'] / 'predictions_experimental_vs_pfaz9.png')

                # 3. Nuclear chart colored by predicted MM
                fig, ax = plt.subplots(figsize=(14, 8))
                sub = merged.dropna(subset=['Z', 'N', 'MM_pred'])
                sc = ax.scatter(sub['N'], sub['Z'], c=sub['MM_pred'], cmap='RdBu_r',
                                s=80, alpha=0.9, vmin=-5, vmax=5)
                plt.colorbar(sc, ax=ax, label='Tahmin Manyetik Moment [nm]')
                for mz in MAGIC_Z:
                    ax.axhline(mz, color='gray', lw=0.5, alpha=0.5, ls=':')
                for mn in MAGIC_N:
                    ax.axvline(mn, color='gray', lw=0.5, alpha=0.5, ls=':')
                ax.set_xlabel('Notron Sayisi N')
                ax.set_ylabel('Proton Sayisi Z')
                ax.set_title('Nukleer Harita - PFAZ9 Tahmini Manyetik Moment', fontsize=12)
                _save(fig, dir_map['predictions'] / 'predictions_nuclear_chart_mm_predicted.png')

                # 4. Nuclear chart colored by predicted Beta_2
                if 'B2_pred' in merged.columns:
                    fig, ax = plt.subplots(figsize=(14, 8))
                    sub = merged.dropna(subset=['Z', 'N', 'B2_pred'])
                    sc = ax.scatter(sub['N'], sub['Z'], c=sub['B2_pred'], cmap='coolwarm',
                                    s=80, alpha=0.9, vmin=-0.5, vmax=0.5)
                    plt.colorbar(sc, ax=ax, label='Tahmin Beta_2 Deformasyon')
                    for mz in MAGIC_Z:
                        ax.axhline(mz, color='gray', lw=0.5, alpha=0.5, ls=':')
                    for mn in MAGIC_N:
                        ax.axvline(mn, color='gray', lw=0.5, alpha=0.5, ls=':')
                    ax.set_xlabel('Notron Sayisi N')
                    ax.set_ylabel('Proton Sayisi Z')
                    ax.set_title('Nukleer Harita - PFAZ9 Tahmini Beta_2', fontsize=12)
                    _save(fig, dir_map['predictions'] / 'predictions_nuclear_chart_beta2_predicted.png')

                # 5. MM uncertainty bands (top 60 nuclei sorted by A)
                if 'MM_ci_lo' in merged.columns:
                    fig, ax = plt.subplots(figsize=(18, 6))
                    sub = merged.dropna(subset=['MM_pred', 'MM_ci_lo']).sort_values('A').head(80)
                    x = range(len(sub))
                    ax.fill_between(x, sub['MM_ci_lo'], sub['MM_ci_hi'], alpha=0.3, color='blue', label='95% CI')
                    ax.plot(x, sub['MM_pred'], 'b-', lw=1.5, label='Tahmin Ort.')
                    if 'MM' in sub.columns:
                        ax.scatter(x, sub['MM'], color='red', s=20, zorder=5, label='Deneysel')
                    ax.set_title('MM Tahmin Belirsizlik Bandlari (ilk 80 nukleus, A sirali)', fontsize=11)
                    ax.set_xlabel('Nukleus indexi')
                    ax.set_ylabel('Manyetik Moment [nm]')
                    ax.legend()
                    _save(fig, dir_map['predictions'] / 'predictions_mm_uncertainty_bands.png')

                # 6. QM bar chart
                if 'QM' in merged.columns:
                    fig, ax = plt.subplots(figsize=(18, 5))
                    qm_valid = merged.dropna(subset=['QM']).sort_values('A')
                    bar_colors = [colors.get(r, '#9E9E9E') for r in qm_valid['mass_region']]
                    ax.bar(range(len(qm_valid)), qm_valid['QM'], color=bar_colors, alpha=0.8, width=0.9)
                    ax.axhline(0, color='black', lw=0.8, ls='--')
                    ax.set_title('267 Cilek Kuadrupol Moment (QM) - Deneysel Deger', fontsize=13)
                    ax.set_xlabel('Nukleus (A sirali)')
                    ax.set_ylabel('Kuadrupol Moment [Q]')
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                    _save(fig, dir_map['predictions'] / 'predictions_qm_experimental_all_nuclei.png')

                # 7. Beta_2 bar
                if 'Beta2_exp' in merged.columns:
                    fig, ax = plt.subplots(figsize=(18, 5))
                    b2_valid = merged.dropna(subset=['Beta2_exp']).sort_values('A')
                    bar_colors2 = [colors.get(r, '#9E9E9E') for r in b2_valid['mass_region']]
                    ax.bar(range(len(b2_valid)), b2_valid['Beta2_exp'], color=bar_colors2, alpha=0.8, width=0.9)
                    ax.axhline(0, color='black', lw=0.8, ls='--')
                    ax.set_title('267 Cilek Beta_2 Deformasyon - Deneysel Deger', fontsize=13)
                    ax.set_xlabel('Nukleus (A sirali)')
                    ax.set_ylabel('Beta_2 Deformasyon')
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                    _save(fig, dir_map['predictions'] / 'predictions_beta2_experimental_all_nuclei.png')

        except Exception as e:
            logger.warning(f"  predictions/ charts failed: {e}")

        # ================================================================
        # FEATURES FOLDER
        # ================================================================
        try:
            if aaa2 is not None and len(aaa2) > 10:
                feat_cols = ['Z', 'N', 'A', 'SPIN', 'magic_character', 'BE_per_A',
                             'Beta_2_estimated', 'Z_magic_dist', 'N_magic_dist',
                             'Z_valence', 'N_valence', 'Q0_intrinsic', 'BE_asymmetry',
                             'BE_pairing', 'shell_closure_effect']
                feat_cols = [c for c in feat_cols if c in aaa2.columns]
                feat_df = aaa2[feat_cols + ['MM', 'QM', 'Beta2_exp', 'mass_region']].copy()

                # 1. Feature - target correlation bar
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                for ax, target in zip(axes, ['MM', 'QM', 'Beta2_exp']):
                    if target not in feat_df.columns:
                        continue
                    corrs = {}
                    for fc in feat_cols:
                        sub = feat_df[[fc, target]].dropna()
                        if len(sub) > 5:
                            corrs[fc] = abs(np.corrcoef(sub[fc], sub[target])[0, 1])
                    corrs = pd.Series(corrs).sort_values(ascending=True).tail(12)
                    ax.barh(corrs.index, corrs.values, color='steelblue', alpha=0.8)
                    ax.set_xlabel('|Pearson r|')
                    ax.set_title(f'Feature-{target} Korelasyon', fontsize=10)
                plt.suptitle('Feature - Hedef Korelasyon Analizi', fontsize=13, fontweight='bold')
                plt.tight_layout()
                _save(fig, dir_map['features'] / 'features_correlation_with_targets.png')

                # 2. PCA 2D
                pca_cols = [c for c in feat_cols if feat_df[c].notna().sum() > 50]
                if len(pca_cols) >= 4:
                    pca_data = feat_df[pca_cols].dropna()
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(pca_data)
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(X_scaled)
                    idx = pca_data.index
                    regions = feat_df.loc[idx, 'mass_region']
                    region_colors = {'Light (A<40)': '#2196F3', 'Medium (40-100)': '#4CAF50',
                                     'Heavy (100-160)': '#FF9800', 'Very Heavy (>160)': '#F44336'}
                    fig, ax = plt.subplots(figsize=(10, 7))
                    for region, color in region_colors.items():
                        mask = regions == region
                        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                                   c=color, label=region, alpha=0.75, s=50)
                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                    ax.set_title('PCA - Nukleer Ozellik Uzayi (267 Nukleus)', fontsize=12)
                    ax.legend()
                    _save(fig, dir_map['features'] / 'features_pca_2d_mass_region.png')

                # 3. PCA 3D
                if len(pca_cols) >= 5:
                    pca3 = PCA(n_components=3)
                    pca3_result = pca3.fit_transform(X_scaled)
                    fig = plt.figure(figsize=(11, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    beta2_color = feat_df.loc[idx, 'Beta2_exp'].fillna(0)
                    sc = ax.scatter(pca3_result[:, 0], pca3_result[:, 1], pca3_result[:, 2],
                                    c=beta2_color, cmap='coolwarm', s=40, alpha=0.8)
                    fig.colorbar(sc, ax=ax, label='Beta_2 Deformasyon')
                    ax.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)')
                    ax.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)')
                    ax.set_zlabel(f'PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)')
                    ax.set_title('3D PCA - Beta_2 Renk Kodlu', fontsize=11)
                    _save(fig, dir_map['features'] / 'features_pca_3d_beta2_colored.png')

                # 4. Feature correlation heatmap (features only)
                if len(pca_cols) >= 4:
                    corr_matrix = feat_df[pca_cols].corr()
                    fig, ax = plt.subplots(figsize=(12, 10))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                                cmap='coolwarm', center=0, ax=ax, annot_kws={'size': 7})
                    ax.set_title('Nukleer Feature Korelasyon Matrisi', fontsize=12)
                    plt.tight_layout()
                    _save(fig, dir_map['features'] / 'features_feature_correlation_matrix.png')

                # 5. Z vs N with Beta_2 coloring
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for ax, (target, title, cmap) in zip(axes, [
                    ('MM', 'Manyetik Moment', 'RdBu_r'),
                    ('QM', 'Kuadrupol Moment', 'PiYG'),
                    ('Beta2_exp', 'Beta_2 Deformasyon', 'coolwarm')
                ]):
                    if target not in aaa2.columns:
                        continue
                    sub = aaa2.dropna(subset=['Z', 'N', target])
                    sc = ax.scatter(sub['N'], sub['Z'], c=sub[target], cmap=cmap,
                                    s=50, alpha=0.85)
                    plt.colorbar(sc, ax=ax, label=title)
                    ax.set_xlabel('N')
                    ax.set_ylabel('Z')
                    ax.set_title(f'{title} - Z/N Dagilimi')
                plt.suptitle('Nukleer Ozellikler: Z vs N Haritasi', fontsize=13, fontweight='bold')
                plt.tight_layout()
                _save(fig, dir_map['features'] / 'features_zn_map_all_targets.png')

        except Exception as e:
            logger.warning(f"  features/ charts failed: {e}")

        # ================================================================
        # ANOMALY FOLDER
        # ================================================================
        try:
            if not merged.empty and 'MM_cv' in merged.columns:
                # 1. CV distribution by mass region
                fig, ax = plt.subplots(figsize=(12, 6))
                regions = merged['mass_region'].unique()
                cv_data = [merged[merged['mass_region'] == r]['MM_cv'].dropna().values for r in regions]
                cv_data = [d[d < 500] for d in cv_data]  # clip extreme outliers
                bp = ax.boxplot(cv_data, labels=regions, patch_artist=True)
                colors_box = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
                for patch, color in zip(bp['boxes'], colors_box[:len(regions)]):
                    patch.set_facecolor(color)
                ax.set_title('MM Tahmin Belirsizlik (CV%) - Kutle Bolgesine Gore', fontsize=12)
                ax.set_ylabel('Degisim Katsayisi (CV%)')
                ax.set_xlabel('Kutle Bolgesi')
                _save(fig, dir_map['anomaly'] / 'anomaly_uncertainty_cv_by_mass_region.png')

                # 2. High uncertainty nuclei bar chart
                if 'MM_std' in merged.columns:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    top_unc = merged.nlargest(30, 'MM_std')[['NUCLEUS', 'MM_std']].dropna()
                    ax.barh(range(len(top_unc)), top_unc['MM_std'].values, color='tomato', alpha=0.8)
                    ax.set_yticks(range(len(top_unc)))
                    ax.set_yticklabels(top_unc['NUCLEUS'].values, fontsize=8)
                    ax.set_xlabel('Tahmin Std (50 Model)')
                    ax.set_title('En Yuksek Belirsizlikli 30 Nukleus (MM)', fontsize=12)
                    _save(fig, dir_map['anomaly'] / 'anomaly_top30_high_uncertainty_nuclei.png')

            if aaa2 is not None:
                # 3. Anomaly: Mahalanobis distance with PCA
                feat_cols = ['Z', 'N', 'A', 'magic_character', 'BE_per_A',
                             'Z_magic_dist', 'N_magic_dist']
                feat_cols = [c for c in feat_cols if c in aaa2.columns]
                pca_data = aaa2[feat_cols].dropna()
                if len(pca_data) > 20:
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(pca_data)
                    mean_v = X_s.mean(axis=0)
                    cov_v  = np.cov(X_s.T)
                    try:
                        inv_cov = np.linalg.pinv(cov_v)
                        diff = X_s - mean_v
                        mah_dist = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
                        fig, ax = plt.subplots(figsize=(14, 5))
                        threshold = np.percentile(mah_dist, 95)
                        colors_mah = ['red' if d > threshold else 'steelblue' for d in mah_dist]
                        ax.bar(range(len(mah_dist)), mah_dist, color=colors_mah, alpha=0.7)
                        ax.axhline(threshold, color='red', ls='--', lw=1.5, label=f'95th percentile ({threshold:.2f})')
                        ax.set_title('Mahalanobis Uzakligi - Anormal Cekirdek Tespiti', fontsize=12)
                        ax.set_xlabel('Nukleus Indexi (A sirali)')
                        ax.set_ylabel('Mahalanobis Uzakligi')
                        ax.legend()
                        _save(fig, dir_map['anomaly'] / 'anomaly_mahalanobis_distance.png')
                    except Exception:
                        pass

                # 4. Magic vs non-magic nucleus properties
                if 'is_magic' in aaa2.columns or 'magic_character' in aaa2.columns:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    mc_col = 'magic_character' if 'magic_character' in aaa2.columns else None
                    if mc_col:
                        for ax, (target, label) in zip(axes, [
                            ('MM', 'MM'), ('QM', 'QM'), ('Beta2_exp', 'Beta_2')
                        ]):
                            if target not in aaa2.columns:
                                continue
                            sub = aaa2.dropna(subset=[mc_col, target])
                            magic_vals    = sub[sub[mc_col] > 0.7][target]
                            nonmagic_vals = sub[sub[mc_col] <= 0.7][target]
                            ax.hist(magic_vals, bins=15, alpha=0.6, color='gold', label='Magic (MC>0.7)')
                            ax.hist(nonmagic_vals, bins=15, alpha=0.6, color='steelblue', label='Non-magic')
                            ax.set_title(f'{label} Dagilimi: Magic vs Non-Magic', fontsize=9)
                            ax.legend(fontsize=8)
                    plt.suptitle('Buyusel Sayi Etkisi: Ozellik Dagilimi', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    _save(fig, dir_map['anomaly'] / 'anomaly_magic_nonmagic_property_distribution.png')

        except Exception as e:
            logger.warning(f"  anomaly/ charts failed: {e}")

        # ================================================================
        # ROBUSTNESS FOLDER
        # ================================================================
        try:
            if not merged.empty and 'MM_std' in merged.columns:
                # 1. Prediction std per mass region (all targets)
                fig, axes = plt.subplots(1, 3, figsize=(16, 6))
                for ax, (std_col, title) in zip(axes, [
                    ('MM_std', 'MM Std'), ('QM_std', 'QM Std'), ('B2_std', 'Beta2 Std')
                ]):
                    if std_col not in merged.columns:
                        continue
                    sub = merged.dropna(subset=[std_col])
                    region_stds = sub.groupby('mass_region')[std_col].mean().sort_values()
                    colors_r = ['#2196F3', '#4CAF50', '#FF9800', '#F44336'][:len(region_stds)]
                    ax.barh(region_stds.index, region_stds.values, color=colors_r, alpha=0.8)
                    ax.set_xlabel('Ort. Std (50 Model)')
                    ax.set_title(f'{title} - Kutle Bolgesine Gore')
                plt.suptitle('Model Uyumu: Tahmin Std - Kutle Bolgeleri', fontsize=13, fontweight='bold')
                plt.tight_layout()
                _save(fig, dir_map['robustness'] / 'robustness_prediction_std_by_mass_region.png')

                # 2. CI width distribution
                if 'MM_ci_lo' in merged.columns:
                    merged['MM_ci_width'] = merged['MM_ci_hi'] - merged['MM_ci_lo']
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(merged['MM_ci_width'].dropna(), bins=30, color='steelblue', alpha=0.8, edgecolor='white')
                    ax.axvline(merged['MM_ci_width'].median(), color='red', ls='--', lw=2,
                               label=f'Medyan={merged["MM_ci_width"].median():.3f}')
                    ax.set_title('MM 95% Guven Araligi Genisligi Dagilimi', fontsize=12)
                    ax.set_xlabel('CI Genisligi (nm)')
                    ax.set_ylabel('Frekans')
                    ax.legend()
                    _save(fig, dir_map['robustness'] / 'robustness_mm_ci_width_distribution.png')

                # 3. Model agreement: std vs predicted value
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                for ax, (pred_col, std_col, label) in zip(axes, [
                    ('MM_pred', 'MM_std', 'MM'),
                    ('QM_pred', 'QM_std', 'QM'),
                    ('B2_pred', 'B2_std', 'Beta_2')
                ]):
                    if pred_col not in merged.columns or std_col not in merged.columns:
                        continue
                    sub = merged.dropna(subset=[pred_col, std_col])
                    ax.scatter(sub[pred_col].abs(), sub[std_col], alpha=0.5, s=25, color='purple')
                    ax.set_xlabel(f'|{label} Tahmin|')
                    ax.set_ylabel(f'{label} Std')
                    ax.set_title(f'{label}: Tahmin Buyuklugu vs Belirsizlik')
                plt.suptitle('Model Uyumu: Buyuk Degerler Daha Belirsiz mi?', fontsize=12, fontweight='bold')
                plt.tight_layout()
                _save(fig, dir_map['robustness'] / 'robustness_prediction_vs_uncertainty.png')

                # 4. Z/N position vs MM uncertainty (nuclear chart of uncertainty)
                if 'MM_std' in merged.columns:
                    fig, ax = plt.subplots(figsize=(13, 8))
                    sub = merged.dropna(subset=['Z', 'N', 'MM_std'])
                    sc = ax.scatter(sub['N'], sub['Z'], c=sub['MM_std'], cmap='YlOrRd',
                                    s=80, alpha=0.9)
                    plt.colorbar(sc, ax=ax, label='MM Std (50 Model)')
                    for mz in MAGIC_Z:
                        ax.axhline(mz, color='blue', lw=0.5, alpha=0.4, ls=':')
                    for mn in MAGIC_N:
                        ax.axvline(mn, color='blue', lw=0.5, alpha=0.4, ls=':')
                    ax.set_xlabel('Notron Sayisi N')
                    ax.set_ylabel('Proton Sayisi Z')
                    ax.set_title('Nukleer Harita - MM Tahmin Belirsizligi (50 Model Std)', fontsize=12)
                    _save(fig, dir_map['robustness'] / 'robustness_nuclear_chart_mm_uncertainty.png')

        except Exception as e:
            logger.warning(f"  robustness/ charts failed: {e}")

        # ================================================================
        # SHAP FOLDER (correlation-based feature importance)
        # ================================================================
        try:
            if aaa2 is not None:
                feat_cols_shap = ['Z', 'N', 'A', 'SPIN', 'magic_character', 'BE_per_A',
                                  'Beta_2_estimated', 'Z_magic_dist', 'N_magic_dist',
                                  'Z_valence', 'N_valence', 'Q0_intrinsic', 'BE_asymmetry',
                                  'BE_pairing', 'shell_closure_effect', 'nilsson_epsilon',
                                  'fermi_energy', 'WS_potential_depth']
                feat_cols_shap = [c for c in feat_cols_shap if c in aaa2.columns]

                for target, target_col, label, color in [
                    ('MM', 'MM', 'Manyetik Moment', '#1976D2'),
                    ('QM', 'QM', 'Kuadrupol Moment', '#388E3C'),
                    ('Beta_2', 'Beta2_exp', 'Beta_2 Deformasyon', '#F57C00')
                ]:
                    if target_col not in aaa2.columns:
                        continue
                    corrs = {}
                    for fc in feat_cols_shap:
                        sub = aaa2[[fc, target_col]].dropna()
                        if len(sub) > 10:
                            corrs[fc] = abs(np.corrcoef(sub[fc], sub[target_col])[0, 1])
                    if not corrs:
                        continue
                    corrs = pd.Series(corrs).sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(10, 7))
                    bars = ax.barh(corrs.index, corrs.values, color=color, alpha=0.8)
                    # Add value labels
                    for bar, val in zip(bars, corrs.values):
                        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                                f'{val:.3f}', va='center', fontsize=8)
                    ax.set_xlabel('|Pearson Korelasyon Katsayisi|')
                    ax.set_title(f'Feature Onemi - {label} (Korelasyon Bazli)', fontsize=12)
                    ax.set_xlim(0, max(corrs.values) * 1.2)
                    plt.tight_layout()
                    _save(fig, dir_map['shap'] / f'shap_feature_importance_{target.lower()}.png')

                # Combined: top 10 features for each target (grouped bar)
                all_corrs = {}
                for target_col, label in [('MM', 'MM'), ('QM', 'QM'), ('Beta2_exp', 'Beta2')]:
                    if target_col not in aaa2.columns:
                        continue
                    for fc in feat_cols_shap:
                        sub = aaa2[[fc, target_col]].dropna()
                        if len(sub) > 10:
                            r = abs(np.corrcoef(sub[fc], sub[target_col])[0, 1])
                            all_corrs.setdefault(fc, {})[label] = r
                if all_corrs:
                    df_corrs = pd.DataFrame(all_corrs).T.fillna(0)
                    df_corrs['mean'] = df_corrs.mean(axis=1)
                    df_corrs = df_corrs.nlargest(12, 'mean').drop(columns='mean')
                    fig, ax = plt.subplots(figsize=(13, 7))
                    df_corrs.plot(kind='barh', ax=ax, alpha=0.8,
                                  color=['#1976D2', '#388E3C', '#F57C00'])
                    ax.set_xlabel('|Pearson r|')
                    ax.set_title('Top 12 Feature Onemi - Tum Hedefler Karsilastirmali', fontsize=12)
                    ax.legend(title='Hedef', loc='lower right')
                    plt.tight_layout()
                    _save(fig, dir_map['shap'] / 'shap_combined_feature_importance_all_targets.png')

        except Exception as e:
            logger.warning(f"  shap/ charts failed: {e}")

        # ================================================================
        # MASTER REPORT FOLDER
        # ================================================================
        try:
            if not merged.empty:
                # 1. Comprehensive 6-panel nuclear dashboard
                fig = plt.figure(figsize=(18, 14))
                gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

                # Panel 1: Z/N nuclear chart (Beta_2)
                ax1 = fig.add_subplot(gs[0, 0])
                sub = merged.dropna(subset=['Z', 'N', 'Beta2_exp'])
                sc1 = ax1.scatter(sub['N'], sub['Z'], c=sub['Beta2_exp'],
                                   cmap='coolwarm', s=30, alpha=0.8)
                fig.colorbar(sc1, ax=ax1, label='Beta_2')
                ax1.set_title('Nukleer Harita (Beta_2)', fontsize=9)
                ax1.set_xlabel('N'); ax1.set_ylabel('Z')

                # Panel 2: MM distribution
                ax2 = fig.add_subplot(gs[0, 1])
                mm_data = merged['MM'].dropna()
                ax2.hist(mm_data, bins=30, color='#1976D2', alpha=0.8, edgecolor='white')
                ax2.axvline(mm_data.mean(), color='red', ls='--', lw=1.5,
                            label=f'Ort.={mm_data.mean():.2f}')
                ax2.set_title('MM Dagilimi', fontsize=9)
                ax2.set_xlabel('MM [nm]'); ax2.legend(fontsize=7)

                # Panel 3: QM distribution
                ax3 = fig.add_subplot(gs[0, 2])
                qm_data = merged['QM'].dropna()
                ax3.hist(qm_data, bins=30, color='#388E3C', alpha=0.8, edgecolor='white')
                ax3.axvline(qm_data.mean(), color='red', ls='--', lw=1.5,
                            label=f'Ort.={qm_data.mean():.2f}')
                ax3.set_title('QM Dagilimi', fontsize=9)
                ax3.set_xlabel('QM [Q]'); ax3.legend(fontsize=7)

                # Panel 4: Beta_2 distribution
                ax4 = fig.add_subplot(gs[1, 0])
                b2_data = merged['Beta2_exp'].dropna()
                ax4.hist(b2_data, bins=30, color='#F57C00', alpha=0.8, edgecolor='white')
                ax4.axvline(0, color='black', lw=0.8, ls='-')
                ax4.set_title('Beta_2 Dagilimi', fontsize=9)
                ax4.set_xlabel('Beta_2')

                # Panel 5: MM predicted uncertainty
                if 'MM_std' in merged.columns:
                    ax5 = fig.add_subplot(gs[1, 1])
                    sc5 = ax5.scatter(merged['N'].dropna(), merged['Z'].dropna(),
                                      c=merged.loc[merged['N'].notna(), 'MM_std'],
                                      cmap='YlOrRd', s=30, alpha=0.8)
                    fig.colorbar(sc5, ax=ax5, label='MM Std')
                    ax5.set_title('Model Belirsizligi (MM)', fontsize=9)
                    ax5.set_xlabel('N'); ax5.set_ylabel('Z')

                # Panel 6: Mass region distribution pie
                ax6 = fig.add_subplot(gs[1, 2])
                region_counts = merged['mass_region'].value_counts()
                ax6.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%',
                        colors=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
                ax6.set_title('Kutle Bolgesi Dagilimi', fontsize=9)

                # Panel 7: MM exp vs pred scatter
                if 'MM_pred' in merged.columns:
                    ax7 = fig.add_subplot(gs[2, 0])
                    sub = merged.dropna(subset=['MM', 'MM_pred'])
                    ax7.scatter(sub['MM'], sub['MM_pred'], alpha=0.5, s=25, color='purple')
                    mn, mx = sub['MM'].min(), sub['MM'].max()
                    ax7.plot([mn, mx], [mn, mx], 'r--', lw=1.2)
                    ax7.set_title('MM: Deneysel vs Tahmin', fontsize=9)
                    ax7.set_xlabel('Deneysel'); ax7.set_ylabel('Tahmin')

                # Panel 8: Magic number analysis
                ax8 = fig.add_subplot(gs[2, 1])
                if 'magic_character' in merged.columns and 'MM' in merged.columns:
                    sub = merged.dropna(subset=['magic_character', 'MM'])
                    sub['magic_bin'] = pd.cut(sub['magic_character'],
                                              bins=[0, 0.3, 0.7, 1.0],
                                              labels=['Duzensiz', 'Gecis', 'Kabuklu'])
                    sub.groupby('magic_bin')['MM'].mean().plot(kind='bar', ax=ax8,
                                                               color=['tomato', 'gold', 'steelblue'],
                                                               alpha=0.8)
                    ax8.set_title('Kabuk Yapisi vs Ort. MM', fontsize=9)
                    ax8.set_xlabel('Kabuk Karakteri'); ax8.set_ylabel('Ort. |MM|')

                # Panel 9: QM vs Beta_2 scatter
                ax9 = fig.add_subplot(gs[2, 2])
                sub = merged.dropna(subset=['QM', 'Beta2_exp'])
                sc9 = ax9.scatter(sub['Beta2_exp'], sub['QM'], c=sub['Z'],
                                   cmap='viridis', s=30, alpha=0.7)
                fig.colorbar(sc9, ax=ax9, label='Z')
                ax9.set_xlabel('Beta_2'); ax9.set_ylabel('QM')
                ax9.set_title('QM vs Beta_2 (renk: Z)', fontsize=9)

                fig.suptitle('MASTER RAPOR: 267 Nukleus - Kapsamli Ozet Dashboard',
                             fontsize=14, fontweight='bold', y=0.98)
                _save(fig, dir_map['master_report'] / 'master_comprehensive_nuclear_dashboard.png')

                # 2. Three-target predictions summary
                if 'MM_pred' in merged.columns:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    for ax, (exp_c, pred_c, title) in zip(axes, [
                        ('MM', 'MM_pred', 'Manyetik Moment'),
                        ('QM', 'QM_pred', 'Kuadrupol Moment'),
                        ('Beta2_exp', 'B2_pred', 'Beta_2 Deformasyon')
                    ]):
                        if exp_c not in merged.columns or pred_c not in merged.columns:
                            continue
                        sub = merged.dropna(subset=[exp_c, pred_c])
                        if len(sub) < 3:
                            continue
                        ax.scatter(sub[exp_c], sub[pred_c], alpha=0.6, s=40, color='steelblue')
                        mn, mx = sub[exp_c].min(), sub[exp_c].max()
                        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
                        r2 = np.corrcoef(sub[exp_c], sub[pred_c])[0, 1]**2
                        ax.set_title(f'{title}\nR²={r2:.3f} (n={len(sub)})', fontsize=10)
                        ax.set_xlabel('Deneysel')
                        ax.set_ylabel('PFAZ9 Tahmin')
                    plt.suptitle('PFAZ9 Ensemble: 267 Nukleus Tahmin vs Deneysel',
                                 fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    _save(fig, dir_map['master_report'] / 'master_predictions_vs_experimental.png')

        except Exception as e:
            logger.warning(f"  master_report/ charts failed: {e}")

        # ================================================================
        # OPTIMIZATION FOLDER
        # ================================================================
        try:
            if ai_df is not None and len(ai_df) > 10:
                # 1. R2 by dataset size
                if 'Size' in ai_df.columns and 'Val_R2' in ai_df.columns:
                    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                    for ax, target in zip(axes, ['MM', 'QM', 'Beta_2']):
                        sub = ai_df[ai_df['Target'] == target].dropna(subset=['Size', 'Val_R2'])
                        if len(sub) < 3:
                            continue
                        sub.groupby('Size')['Val_R2'].mean().plot(kind='line', ax=ax, marker='o',
                                                                   color='steelblue', lw=2)
                        ax.fill_between(
                            sub.groupby('Size')['Val_R2'].mean().index,
                            sub.groupby('Size')['Val_R2'].quantile(0.25),
                            sub.groupby('Size')['Val_R2'].quantile(0.75),
                            alpha=0.2, color='steelblue'
                        )
                        ax.set_title(f'{target}: Dataset Boyutu vs Val R²', fontsize=10)
                        ax.set_xlabel('Dataset Boyutu')
                        ax.set_ylabel('Mean Val R²')
                    plt.suptitle('Optimizasyon: Dataset Boyutu Etkisi', fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    _save(fig, dir_map['optimization'] / 'optimization_dataset_size_effect.png')

                # 2. R2 by model type (optimization landscape)
                if 'Model_Type' in ai_df.columns and 'Val_R2' in ai_df.columns:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    pivot = ai_df.pivot_table(index='Model_Type', columns='Target',
                                              values='Val_R2', aggfunc='mean')
                    pivot.plot(kind='bar', ax=ax, alpha=0.8)
                    ax.set_title('Model Tipi Optimizasyonu: Her Hedef icin Mean Val R²', fontsize=12)
                    ax.set_xlabel('Model Tipi')
                    ax.set_ylabel('Mean Val R²')
                    ax.legend(title='Hedef')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    _save(fig, dir_map['optimization'] / 'optimization_model_type_per_target.png')

                # 3. S70 vs S80 scenario comparison
                if 'Scenario' in ai_df.columns:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    for ax, target in zip(axes, ['MM', 'QM', 'Beta_2']):
                        sub = ai_df[ai_df['Target'] == target].dropna(subset=['Val_R2'])
                        if 'Scenario' not in sub.columns:
                            continue
                        s70 = sub[sub['Scenario'] == 'S70']['Val_R2']
                        s80 = sub[sub['Scenario'] == 'S80']['Val_R2']
                        ax.boxplot([s70, s80], labels=['S70', 'S80'], patch_artist=True,
                                   boxprops=dict(facecolor='lightblue'))
                        ax.set_title(f'{target}: Senaryo Karsilastirmasi', fontsize=10)
                        ax.set_ylabel('Val R²')
                    plt.suptitle('S70 vs S80 Senaryo Optimizasyonu', fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    _save(fig, dir_map['optimization'] / 'optimization_scenario_s70_vs_s80.png')

                # 4. Top configurations (best val R2 per model type-target)
                if 'Val_R2' in ai_df.columns and 'Model_Type' in ai_df.columns:
                    fig, ax = plt.subplots(figsize=(14, 8))
                    top_cfgs = ai_df.nlargest(20, 'Val_R2')[['Dataset', 'Model_Type', 'Target', 'Val_R2']].copy()
                    top_cfgs['label'] = top_cfgs['Model_Type'] + '\n' + top_cfgs['Target']
                    colors_target = {'MM': '#1976D2', 'QM': '#388E3C', 'Beta_2': '#F57C00',
                                     'MM_QM': '#9C27B0'}
                    bar_c = [colors_target.get(t, 'gray') for t in top_cfgs['Target']]
                    ax.barh(range(len(top_cfgs)), top_cfgs['Val_R2'].values, color=bar_c, alpha=0.8)
                    ax.set_yticks(range(len(top_cfgs)))
                    ax.set_yticklabels(top_cfgs['label'].values, fontsize=8)
                    ax.set_xlabel('Val R²')
                    ax.set_title('Top 20 Konfigurasyonlari - En Iyi Val R²', fontsize=12)
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors_target.items()]
                    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
                    plt.tight_layout()
                    _save(fig, dir_map['optimization'] / 'optimization_top20_configs.png')

        except Exception as e:
            logger.warning(f"  optimization/ charts failed: {e}")

        # ================================================================
        # TRAINING METRICS FOLDER
        # ================================================================
        try:
            if ai_df is not None and len(ai_df) > 10:
                val_col  = 'Val_R2'  if 'Val_R2'  in ai_df.columns else None
                test_col = 'Test_R2' if 'Test_R2' in ai_df.columns else None
                train_col = 'Train_R2' if 'Train_R2' in ai_df.columns else None

                # 1. Train/Val/Test R2 convergence per model type
                if val_col and test_col and 'Model_Type' in ai_df.columns:
                    fig, ax = plt.subplots(figsize=(13, 6))
                    mt_groups = ai_df.groupby('Model_Type')
                    metrics = {mt: {
                        'Val': g[val_col].mean(),
                        'Test': g[test_col].mean(),
                        'Gap': (g[val_col] - g[test_col]).mean()
                    } for mt, g in mt_groups}
                    df_m = pd.DataFrame(metrics).T
                    df_m[['Val', 'Test']].plot(kind='bar', ax=ax, alpha=0.8,
                                               color=['#1976D2', '#388E3C'])
                    ax.set_title('Model Tipi Bazinda Val/Test R² Karsilastirmasi', fontsize=12)
                    ax.set_xlabel('Model Tipi')
                    ax.set_ylabel('Mean R²')
                    ax.legend(['Val R²', 'Test R²'])
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    _save(fig, dir_map['training_metrics'] / 'training_metrics_val_test_by_model.png')

                # 2. Overfitting analysis (Val-Test gap)
                if val_col and test_col:
                    ai_df_copy = ai_df.copy()
                    ai_df_copy['overfitting_gap'] = ai_df_copy[val_col] - ai_df_copy[test_col]
                    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                    for ax, target in zip(axes, ['MM', 'QM', 'Beta_2']):
                        sub = ai_df_copy[ai_df_copy['Target'] == target].dropna(subset=['overfitting_gap'])
                        if len(sub) < 3:
                            continue
                        ax.hist(sub['overfitting_gap'], bins=25, color='tomato', alpha=0.8, edgecolor='white')
                        ax.axvline(0, color='black', lw=1.5, ls='--')
                        ax.axvline(sub['overfitting_gap'].mean(), color='blue', lw=1.5, ls='--',
                                   label=f'Ort.={sub["overfitting_gap"].mean():.3f}')
                        ax.set_title(f'{target}: Overfitting Gap (Val-Test R²)', fontsize=10)
                        ax.set_xlabel('Val R² - Test R²')
                        ax.legend(fontsize=8)
                    plt.suptitle('Egitim Metrikleri: Overfitting Analizi', fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    _save(fig, dir_map['training_metrics'] / 'training_metrics_overfitting_analysis.png')

                # 3. R2 progression by dataset size (line plot, all targets)
                if val_col and 'Size' in ai_df.columns:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    colors_t = {'MM': '#1976D2', 'QM': '#388E3C', 'Beta_2': '#F57C00'}
                    for target, color in colors_t.items():
                        sub = ai_df[ai_df['Target'] == target].dropna(subset=['Size', val_col])
                        if len(sub) < 3:
                            continue
                        sizes = sorted(sub['Size'].unique())
                        means = [sub[sub['Size'] == s][val_col].mean() for s in sizes]
                        stds  = [sub[sub['Size'] == s][val_col].std() for s in sizes]
                        ax.plot(sizes, means, '-o', color=color, lw=2, label=target)
                        ax.fill_between(sizes,
                                        [m - s for m, s in zip(means, stds)],
                                        [m + s for m, s in zip(means, stds)],
                                        alpha=0.15, color=color)
                    ax.set_title('Dataset Boyutu vs Val R² (Egitim Etkinligi)', fontsize=12)
                    ax.set_xlabel('Dataset Boyutu')
                    ax.set_ylabel('Mean Val R²')
                    ax.legend(title='Hedef')
                    plt.tight_layout()
                    _save(fig, dir_map['training_metrics'] / 'training_metrics_size_vs_r2_progression.png')

                # 4. Training summary: best model per target x metric
                if val_col and test_col and 'Target' in ai_df.columns and 'Model_Type' in ai_df.columns:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    summary_rows = []
                    for target in ['MM', 'QM', 'Beta_2']:
                        sub = ai_df[ai_df['Target'] == target]
                        for model_type in sub['Model_Type'].unique() if 'Model_Type' in sub.columns else []:
                            msub = sub[sub['Model_Type'] == model_type]
                            if len(msub) == 0:
                                continue
                            summary_rows.append({
                                'Target': target, 'Model': model_type,
                                'Val_R2': msub[val_col].mean(),
                                'Test_R2': msub[test_col].mean(),
                            })
                    if summary_rows:
                        df_sum = pd.DataFrame(summary_rows)
                        pivot_sum = df_sum.pivot_table(index='Model', columns='Target',
                                                       values='Val_R2', aggfunc='mean')
                        pivot_sum.plot(kind='barh', ax=ax, alpha=0.8)
                        ax.set_title('Egitim Ozeti: Model x Hedef Ort. Val R²', fontsize=12)
                        ax.set_xlabel('Mean Val R²')
                        ax.legend(title='Hedef')
                        plt.tight_layout()
                        _save(fig, dir_map['training_metrics'] / 'training_metrics_model_target_summary.png')

        except Exception as e:
            logger.warning(f"  training_metrics/ charts failed: {e}")

    def _auto_load_project_data(self) -> Dict:
        """Auto-load model_metrics from PFAZ 6 final_summary.json for basic chart"""
        project_data = {}
        reports_dir = self._find_reports_dir()
        if reports_dir is None:
            return project_data
        summary_path = reports_dir / 'final_summary.json'
        if not summary_path.exists():
            return project_data
        try:
            with open(summary_path, encoding='utf-8') as f:
                summary = json.load(f)
            per_target = summary.get('per_target_best', {})
            model_metrics = {}
            for target, info in per_target.items():
                ai_r2  = info.get('AI_best_test_r2') or info.get('ai', {}).get('test_r2')
                anfis_r2 = info.get('ANFIS_best_test_r2') or info.get('anfis', {}).get('test_r2')
                if ai_r2 is not None:
                    model_metrics[f"AI_{target}"] = {'r2_score': ai_r2, 'mae': 0, 'rmse': 0}
                if anfis_r2 is not None:
                    model_metrics[f"ANFIS_{target}"] = {'r2_score': anfis_r2, 'mae': 0, 'rmse': 0}
            if model_metrics:
                project_data['model_metrics'] = model_metrics
        except Exception as e:
            logger.warning(f"  Could not load final_summary.json: {e}")
        return project_data

    def generate_all_visualizations(self, project_data: Dict):
        """Generate all visualizations from project data"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING ALL VISUALIZATIONS")
        logger.info("="*80)

        generated = []

        # ---- Always run comprehensive auto-generation from PFAZ 6 data ----
        logger.info("\n-> Auto-generating comprehensive visualizations from PFAZ 6 outputs...")
        auto_result = self.auto_generate_from_pfaz6_data()
        n_auto = auto_result.get('count', 0)
        if n_auto > 0:
            generated.append(f'auto_pfaz6 ({n_auto} files)')
            logger.info(f"  [OK] {n_auto} charts generated automatically")

        # ---- project_data-driven visualizations (if caller provides data) ----
        if not project_data:
            project_data = self._auto_load_project_data()

        if 'robustness' in project_data:
            try:
                self.robustness_viz.plot_perturbation_sensitivity(
                    project_data['robustness']['predictions'],
                    project_data['robustness']['results']
                )
                generated.append('robustness')
            except Exception as e:
                logger.warning(f"  Robustness viz failed: {e}")

        if 'shap' in project_data:
            try:
                self.shap_viz.plot_shap_summary(
                    project_data['shap']['features'],
                    project_data['shap']['values'],
                    project_data['shap']['X']
                )
                generated.append('shap')
            except Exception as e:
                logger.warning(f"  SHAP viz failed: {e}")

        if 'anomaly' in project_data:
            try:
                self.anomaly_viz.plot_anomaly_characteristics(
                    project_data['anomaly']['normal'],
                    project_data['anomaly']['anomalous'],
                    project_data['anomaly']['features']
                )
                generated.append('anomaly')
            except Exception as e:
                logger.warning(f"  Anomaly viz failed: {e}")

        if 'master_results' in project_data:
            try:
                self.master_report_viz.plot_master_summary_stats(
                    project_data['master_results']['summary_df']
                )
                generated.append('master_results')
            except Exception as e:
                logger.warning(f"  Master report viz failed: {e}")

        if 'predictions' in project_data:
            try:
                self.prediction_viz.plot_prediction_comparison(
                    project_data['predictions']['experimental'],
                    project_data['predictions']['models']
                )
                generated.append('predictions')
            except Exception as e:
                logger.warning(f"  Predictions viz failed: {e}")

        if 'model_metrics' in project_data:
            try:
                self.model_comparison_viz.plot_model_ranking(project_data['model_metrics'])
                generated.append('model_metrics')
            except Exception as e:
                logger.warning(f"  Model ranking viz failed: {e}")

        if 'training_history' in project_data:
            try:
                self.training_metrics_viz.plot_training_curves(project_data['training_history'])
                generated.append('training_history')
            except Exception as e:
                logger.warning(f"  Training metrics viz failed: {e}")

        # ---- SHAPAnalyzer: en iyi AI modellerinde SHAP onem analizi ----
        try:
            from pfaz_modules.pfaz08_visualization.shap_analysis import SHAPAnalyzer, SHAP_AVAILABLE
            if not SHAP_AVAILABLE:
                logger.info("  [INFO] SHAP kurulu degil — SHAP analizi atlanıyor (pip install shap)")
            else:
                import joblib as _jl
                # Trained models dir'i bul (PFAZ2 ciktisi)
                _models_root = None
                for _cand in [
                    self.output_dir.parent / 'trained_models',
                    self.output_dir.parent.parent / 'outputs' / 'trained_models',
                ]:
                    if _cand.exists():
                        _models_root = _cand
                        break

                if _models_root is None:
                    logger.info("  [INFO] trained_models dizini bulunamadı — SHAP atlanıyor")
                else:
                    _shap_analyzer = SHAPAnalyzer(output_dir=str(self.output_dir / 'shap_analysis'))
                    _shap_done = 0
                    # Her target icin en iyi 1 RF/XGB modeli bul ve SHAP calistir
                    for _target_name in ['MM', 'QM', 'Beta_2']:
                        try:
                            _best_pkl = None
                            _best_r2  = -999.0
                            for _pkl in _models_root.rglob('model_*.pkl'):
                                # Sadece tree-based modeller (RF/XGB/GBM) SHAP ile hizli calisir
                                _pstr = str(_pkl).lower()
                                if not any(_t in _pstr for _t in ('rf', 'xgb', 'gbm', 'lgb')):
                                    continue
                                # Target kontrolu (dosya yolundan)
                                _ds_name = _pkl.parts[-4] if len(_pkl.parts) >= 4 else ''
                                if not _ds_name.startswith(_target_name):
                                    continue
                                _mf = _pkl.parent / f'metrics_{_pkl.parent.name}.json'
                                if not _mf.exists():
                                    continue
                                import json as _js
                                with open(_mf) as _mff:
                                    _met = _js.load(_mff)
                                _vr2 = _met.get('val', {}).get('r2', -999.0)
                                if _vr2 > _best_r2:
                                    _best_r2 = _vr2
                                    _best_pkl = _pkl

                            if _best_pkl is None:
                                continue

                            # Load model + dataset
                            _model = _jl.load(_best_pkl)
                            _ds_dir = _models_root.parent / 'generated_datasets' / _best_pkl.parts[-4]
                            if not _ds_dir.exists():
                                continue
                            import pandas as _pds
                            _train_csv = _ds_dir / 'train.csv'
                            _test_csv  = _ds_dir / 'test.csv'
                            if not (_train_csv.exists() and _test_csv.exists()):
                                continue
                            # Load with metadata for headerless CSV
                            import json as _jsm
                            _meta_f = _ds_dir / 'metadata.json'
                            _col_names = None
                            if _meta_f.exists():
                                with open(_meta_f) as _mf2:
                                    _meta = _jsm.load(_mf2)
                                _feat_cols = _meta.get('feature_names') or _meta.get('feature_columns', [])
                                _tgt_cols  = _meta.get('target_names')  or _meta.get('target_columns',  [])
                                if _feat_cols and _tgt_cols:
                                    _col_names = list(_feat_cols) + list(_tgt_cols)
                            _read_kw = {'header': None, 'names': _col_names} if _col_names else {}
                            _df_tr = _pds.read_csv(_train_csv, **_read_kw)
                            _df_te = _pds.read_csv(_test_csv,  **_read_kw)
                            _target_col = _target_name
                            _feat = [c for c in _df_tr.columns if c not in
                                     {'NUCLEUS', 'MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]', 'Beta_2', _target_col}]
                            if not _feat:
                                continue
                            _X_tr = _df_tr[_feat].fillna(0).values
                            _X_te = _df_te[_feat].fillna(0).values
                            _shap_analyzer.analyze_model(
                                model=_model,
                                X_train=_X_tr,
                                X_test=_X_te,
                                feature_names=_feat,
                                model_name=f'{_target_name}_{_best_pkl.parent.name}'
                            )
                            _shap_done += 1
                        except Exception as _se:
                            logger.warning(f"  [WARNING] SHAP {_target_name}: {_se}")

                    if _shap_done > 0:
                        generated.append(f'shap ({_shap_done} models)')
                        logger.info(f"  [OK] SHAPAnalyzer: {_shap_done} model analiz edildi -> shap_analysis/")
        except Exception as _e:
            logger.warning(f"[WARNING] SHAPAnalyzer basarisiz (devam): {_e}")

        # ---- AnomalyVisualizationsComplete: anomali radar + outlier karsilastirma ----
        try:
            from pfaz_modules.pfaz08_visualization.anomaly_visualizations_complete import AnomalyVisualizationsComplete
            import pandas as _pds_av
            # AAA2 enriched dosyasini bul
            _anom_data = None
            for _cand_av in [
                self.output_dir.parent / 'generated_datasets' / 'AAA2_enriched_all_nuclei.csv',
                self.output_dir.parent.parent / 'outputs' / 'generated_datasets' / 'AAA2_enriched_all_nuclei.csv',
                self.output_dir.parent / 'generated_datasets' / 'AAA2_enriched_all_nuclei.xlsx',
            ]:
                if _cand_av.exists():
                    _anom_data = _pds_av.read_excel(str(_cand_av)) if str(_cand_av).endswith('.xlsx') else _pds_av.read_csv(str(_cand_av))
                    break
            if _anom_data is not None and len(_anom_data) > 0:
                _av = AnomalyVisualizationsComplete(output_dir=str(self.output_dir / 'anomaly_visualizations'))
                _feat_cols_av = [c for c in _anom_data.columns
                                 if c not in {'A', 'Z', 'N', 'NUCLEUS', 'MM', 'QM', 'Beta_2'}
                                 and _anom_data[c].dtype in ('float64', 'int64', 'float32')][:8]
                if not _feat_cols_av:
                    _feat_cols_av = [c for c in _anom_data.select_dtypes('number').columns][:8]
                import numpy as _np_av
                # IQR ile anomali tespiti
                _q1 = _anom_data[_feat_cols_av].quantile(0.25)
                _q3 = _anom_data[_feat_cols_av].quantile(0.75)
                _iqr = _q3 - _q1
                _is_anom = ((_anom_data[_feat_cols_av] < (_q1 - 3.0 * _iqr)) |
                             (_anom_data[_feat_cols_av] > (_q3 + 3.0 * _iqr))).any(axis=1)
                _anom_idx = _np_av.where(_is_anom)[0]
                _anom_sub = _anom_data.iloc[_anom_idx] if len(_anom_idx) > 0 else _anom_data.head(5)
                _av.generate_all_anomaly_plots(
                    anomaly_data=_anom_sub,
                    feature_cols=_feat_cols_av,
                    data=_anom_data,
                    outlier_methods={'IQR': _is_anom.values}
                )
                generated.append('anomaly_visualizations')
                logger.info(f"  [OK] AnomalyVisualizationsComplete: anomali grafikleri -> anomaly_visualizations/")
            else:
                logger.info("  [INFO] AnomalyVisualizationsComplete: AAA2 enriched verisi bulunamadı — atlanıyor")
        except Exception as _e:
            logger.warning(f"[WARNING] AnomalyVisualizationsComplete basarisiz (devam): {_e}")

        # ---- InteractiveHTMLVisualizer: AI sonuclari icin interaktif HTML ----
        try:
            from pfaz_modules.pfaz08_visualization.interactive_html_visualizer import InteractiveHTMLVisualizer
            import pandas as _pds_ih
            _ih_data = None
            for _cand_ih in [
                self.output_dir.parent / 'trained_models' / 'training_summary.xlsx',
                self.output_dir.parent.parent / 'outputs' / 'trained_models' / 'training_summary.xlsx',
                self.output_dir.parent / 'trained_models' / 'ai_training_summary.xlsx',
                self.output_dir.parent.parent / 'outputs' / 'trained_models' / 'ai_training_summary.xlsx',
                self.output_dir.parent / 'final_report' / 'ANFIS_Comprehensive_Report.xlsx',
            ]:
                if _cand_ih.exists():
                    try:
                        _ih_data = _pds_ih.read_excel(str(_cand_ih), sheet_name=0)
                        break
                    except Exception:
                        continue
            if _ih_data is not None and len(_ih_data) > 0:
                _ih = InteractiveHTMLVisualizer(output_dir=str(self.output_dir / 'interactive_html'))
                _ih.create_all_visualizations(_ih_data)
                generated.append('interactive_html')
                logger.info(f"  [OK] InteractiveHTMLVisualizer: HTML grafikleri -> interactive_html/")
            else:
                logger.info("  [INFO] InteractiveHTMLVisualizer: AI training_summary bulunamadı — atlanıyor")
        except Exception as _e:
            logger.warning(f"[WARNING] InteractiveHTMLVisualizer basarisiz (devam): {_e}")

        # ---- LogAnalyticsVisualizationsComplete: log dosyasi analizi ----
        try:
            from pfaz_modules.pfaz08_visualization.log_analytics_visualizations_complete import LogAnalyticsVisualizationsComplete
            import pandas as _pds_la
            import re as _re_la
            from datetime import datetime as _dt_la
            # Log dosyasini bul
            _log_file = None
            for _cand_la in [
                self.output_dir.parent.parent / 'pipeline.log',
                self.output_dir.parent.parent / 'outputs' / 'pipeline.log',
                self.output_dir.parent.parent / 'main.log',
                self.output_dir.parent.parent / 'nucdatav1.log',
            ]:
                if _cand_la.exists():
                    _log_file = _cand_la
                    break
            if _log_file is None:
                # Glob ile herhangi bir .log dosyasi bul
                _log_candidates = list(self.output_dir.parent.parent.glob('*.log'))
                if _log_candidates:
                    _log_file = _log_candidates[0]
            if _log_file is not None and _log_file.exists():
                # Log satırlarını parse et -> DataFrame
                _log_rows = []
                _pat = _re_la.compile(
                    r'(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
                    r'.*?(?P<level>INFO|WARNING|ERROR|DEBUG)'
                    r'\s+(?P<msg>.+)'
                )
                with open(_log_file, encoding='utf-8', errors='ignore') as _lf:
                    for _line in _lf:
                        _m = _pat.search(_line)
                        if _m:
                            try:
                                _ts = _dt_la.strptime(_m.group('ts'), '%Y-%m-%d %H:%M:%S')
                            except Exception:
                                _ts = _dt_la.now()
                            _log_rows.append({
                                'timestamp': _ts,
                                'level': _m.group('level'),
                                'message': _m.group('msg').strip(),
                                'module': _m.group('msg').split(']')[0].strip('[') if ']' in _m.group('msg') else 'main',
                            })
                if _log_rows:
                    _la_df = _pds_la.DataFrame(_log_rows)
                    _la = LogAnalyticsVisualizationsComplete(output_dir=str(self.output_dir / 'log_analytics'))
                    _la.generate_all_log_analytics_plots(log_data=_la_df, metrics_log=_la_df)
                    generated.append('log_analytics')
                    logger.info(f"  [OK] LogAnalyticsVisualizationsComplete: log grafikleri -> log_analytics/")
                else:
                    logger.info("  [INFO] LogAnalyticsVisualizationsComplete: log dosyasi parse edilemedi — atlanıyor")
            else:
                logger.info("  [INFO] LogAnalyticsVisualizationsComplete: .log dosyasi bulunamadı — atlanıyor")
        except Exception as _e:
            logger.warning(f"[WARNING] LogAnalyticsVisualizationsComplete basarisiz (devam): {_e}")

        # ---- MasterReportVisualizationsComplete: ozet istatistikler ----
        try:
            from pfaz_modules.pfaz08_visualization.master_report_visualizations_complete import MasterReportVisualizationsComplete
            import pandas as _pds_mr
            # PFAZ6 ozet verisini bul
            _mr_perf = None
            for _cand_mr in [
                self.output_dir.parent / 'final_report',
                self.output_dir.parent.parent / 'outputs' / 'final_report',
            ]:
                if _cand_mr.exists():
                    for _xlsx in list(_cand_mr.glob('*.xlsx'))[:3]:
                        try:
                            _df_tmp = _pds_mr.read_excel(str(_xlsx), sheet_name=0)
                            if len(_df_tmp) > 3:
                                _mr_perf = _df_tmp
                                break
                        except Exception:
                            continue
                if _mr_perf is not None:
                    break
            if _mr_perf is not None:
                _mr = MasterReportVisualizationsComplete(output_dir=str(self.output_dir / 'master_report_complete'))
                # summary_stats dict'ini DataFrame'den türet
                _num_cols = [c for c in _mr_perf.select_dtypes('number').columns]
                _ss = {col: {'mean': float(_mr_perf[col].mean()), 'std': float(_mr_perf[col].std())}
                       for col in _num_cols[:6]}
                _mr.generate_all_master_report_plots(
                    summary_stats=_ss,
                    model_performance=_mr_perf.head(50)
                )
                generated.append('master_report_complete')
                logger.info(f"  [OK] MasterReportVisualizationsComplete: ozet grafikleri -> master_report_complete/")
            else:
                logger.info("  [INFO] MasterReportVisualizationsComplete: PFAZ6 ozet verisi bulunamadı — atlanıyor")
        except Exception as _e:
            logger.warning(f"[WARNING] MasterReportVisualizationsComplete basarisiz (devam): {_e}")

        # ---- ModelComparisonDashboard: AI + ANFIS karsilastirma panosu ----
        try:
            from pfaz_modules.pfaz08_visualization.model_comparison_dashboard import ModelComparisonDashboard
            _ai_f, _anfis_f = None, None
            for _cand_md in [
                self.output_dir.parent / 'trained_models' / 'training_summary.xlsx',
                self.output_dir.parent.parent / 'outputs' / 'trained_models' / 'training_summary.xlsx',
                self.output_dir.parent / 'trained_models' / 'ai_training_summary.xlsx',
                self.output_dir.parent.parent / 'outputs' / 'trained_models' / 'ai_training_summary.xlsx',
            ]:
                if _cand_md.exists():
                    _ai_f = str(_cand_md)
                    break
            for _cand_af in [
                self.output_dir.parent / 'anfis_models' / 'anfis_training_summary.xlsx',
                self.output_dir.parent.parent / 'outputs' / 'anfis_models' / 'anfis_training_summary.xlsx',
                self.output_dir.parent / 'anfis_models' / 'anfis_training_summary.json',
            ]:
                if _cand_af.exists():
                    _anfis_f = str(_cand_af)
                    break
            if _ai_f or _anfis_f:
                _mcd = ModelComparisonDashboard(output_dir=str(self.output_dir / 'model_comparison_dashboard'))
                _mcd.load_results(ai_results_file=_ai_f, anfis_results_file=_anfis_f)
                if _mcd.results_df is not None and len(_mcd.results_df) > 0:
                    _mcd.create_comparison_dashboard()
                    generated.append('model_comparison_dashboard')
                    logger.info(f"  [OK] ModelComparisonDashboard: karsilastirma panosu -> model_comparison_dashboard/")
                else:
                    logger.info("  [INFO] ModelComparisonDashboard: yüklenen sonuç yok — atlanıyor")
            else:
                logger.info("  [INFO] ModelComparisonDashboard: AI/ANFIS summary dosyasi bulunamadı — atlanıyor")
        except Exception as _e:
            logger.warning(f"[WARNING] ModelComparisonDashboard basarisiz (devam): {_e}")

        logger.info("\n" + "="*80)
        logger.info(f"[OK] ALL VISUALIZATIONS COMPLETED! ({len(generated)} sections, {n_auto} PNG/HTML files)")
        logger.info(f"[OK] Output: {self.output_dir}")
        logger.info("="*80)

        return {
            'success': True,
            'generated_sections': generated,
            'n_sections': len(generated),
            'n_files': n_auto,
            'output_dir': str(self.output_dir),
            'auto_files': auto_result.get('files', []),
        }


def main():
    """Test ve demo"""
    logger.info("\n" + "="*80)
    logger.info("MASTER VISUALIZATION SYSTEM - TEST")
    logger.info("="*80)
    
    # Initialize system
    viz_system = MasterVisualizationSystem('output/visualizations_master')
    
    logger.info(f"\n[OK] System initialized")
    logger.info(f"[OK] Output directory: output/visualizations_master")
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
