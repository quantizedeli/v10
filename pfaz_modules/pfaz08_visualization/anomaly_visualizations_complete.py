# -*- coding: utf-8 -*-
"""
ANOMALY VISUALIZATIONS COMPLETE
================================

Eksik anomaly grafiklerini tamamlayan modül

Yeni Grafikler:
1. anomaly_feature_radar_charts.png - Anomaly nuclei feature profilleri
2. outlier_detection_comparison.png - Farklı outlier detection yöntemleri karşılaştırması

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 8 Completion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plot configuration
PLOT_CONFIG = {
    'dpi': 300,
    'figsize_default': (14, 10),
    'figsize_wide': (18, 10),
    'style': 'seaborn-v0_8-darkgrid'
}

plt.style.use(PLOT_CONFIG['style'])
sns.set_palette("husl")


class AnomalyVisualizationsComplete:
    """Anomaly detection için eksik grafikleri tamamlar"""
    
    def __init__(self, output_dir='visualizations/anomaly_complete'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ AnomalyVisualizationsComplete initialized: {self.output_dir}")
    
    def plot_anomaly_feature_radar_charts(self,
                                         anomaly_data: pd.DataFrame,
                                         feature_cols: List[str],
                                         save_name: str = 'anomaly_feature_radar_charts'):
        """
        Anomaly nuclei için feature radar charts
        
        Args:
            anomaly_data: DataFrame with anomaly nuclei
                Columns: NUCLEUS, A, Z, N, + feature_cols, anomaly_score
            feature_cols: List of feature column names to plot
        """
        logger.info("\n-> Creating anomaly feature radar charts...")
        
        # Select top anomalies
        top_anomalies = anomaly_data.nlargest(6, 'anomaly_score')
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Anomaly Nuclei: Feature Profile Radar Charts\n(Normalized Features)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Limit features for readability
        features_to_plot = feature_cols[:12] if len(feature_cols) > 12 else feature_cols
        n_features = len(features_to_plot)
        
        # Calculate angles for radar
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Create subplots
        for idx, (_, anomaly) in enumerate(top_anomalies.iterrows()):
            ax = plt.subplot(2, 3, idx + 1, projection='polar')
            
            # Get feature values and normalize
            values = []
            for feat in features_to_plot:
                val = anomaly[feat]
                # Normalize to 0-1 range
                if val != 0:
                    values.append(abs(val) / (abs(val) + 1))
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2.5, 
                   color='red', markersize=8, alpha=0.7, label='Anomaly')
            ax.fill(angles, values, alpha=0.25, color='red')
            
            # Plot average profile (reference)
            avg_values = []
            for feat in features_to_plot:
                avg = anomaly_data[feat].mean()
                if avg != 0:
                    avg_values.append(abs(avg) / (abs(avg) + 1))
                else:
                    avg_values.append(0)
            avg_values += avg_values[:1]
            
            ax.plot(angles, avg_values, 'o-', linewidth=1.5, 
                   color='blue', markersize=4, alpha=0.5, 
                   linestyle='--', label='Average')
            
            # Customize
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features_to_plot, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
            ax.grid(True, alpha=0.3)
            
            # Title with nucleus info
            nucleus_name = anomaly['NUCLEUS']
            anomaly_score = anomaly['anomaly_score']
            ax.set_title(f"{nucleus_name}\nAnomaly Score: {anomaly_score:.3f}", 
                        fontsize=10, fontweight='bold', pad=15)
            
            if idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_name}.png")
        return save_path
    
    def plot_outlier_detection_comparison(self,
                                         data: pd.DataFrame,
                                         outlier_methods: Dict[str, np.ndarray],
                                         feature_x: str,
                                         feature_y: str,
                                         save_name: str = 'outlier_detection_comparison'):
        """
        Farklı outlier detection yöntemlerini karşılaştır
        
        Args:
            data: DataFrame with features
            outlier_methods: {
                'IsolationForest': boolean_array,
                'LOF': boolean_array,
                'OneClassSVM': boolean_array,
                'DBSCAN': boolean_array,
                'StatisticalZ': boolean_array
            }
            feature_x: Feature for x-axis
            feature_y: Feature for y-axis
        """
        logger.info("\n-> Creating outlier detection comparison...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Outlier Detection Methods Comparison\n{feature_x} vs {feature_y}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        axes = axes.flatten()
        
        # Get features
        X = data[feature_x].values
        Y = data[feature_y].values
        
        # Plot each method
        method_names = list(outlier_methods.keys())
        
        for idx, method_name in enumerate(method_names):
            ax = axes[idx]
            outliers = outlier_methods[method_name]
            
            # Plot inliers
            inliers = ~outliers
            ax.scatter(X[inliers], Y[inliers], 
                      c='steelblue', s=30, alpha=0.6, 
                      edgecolors='black', linewidth=0.5, label='Inliers')
            
            # Plot outliers
            ax.scatter(X[outliers], Y[outliers], 
                      c='red', s=100, alpha=0.8, 
                      edgecolors='black', linewidth=1.5, 
                      marker='X', label='Outliers')
            
            # Add statistics
            n_outliers = outliers.sum()
            outlier_pct = (n_outliers / len(outliers)) * 100
            
            ax.set_xlabel(feature_x, fontsize=10, fontweight='bold')
            ax.set_ylabel(feature_y, fontsize=10, fontweight='bold')
            ax.set_title(f'{method_name}\nOutliers: {n_outliers} ({outlier_pct:.1f}%)', 
                        fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Consensus plot (last subplot)
        ax = axes[-1]
        
        # Calculate consensus (majority voting)
        consensus_matrix = np.column_stack([outlier_methods[m] for m in method_names])
        consensus_votes = consensus_matrix.sum(axis=1)
        
        # Define consensus levels
        n_methods = len(method_names)
        high_consensus = consensus_votes >= (n_methods * 0.7)  # 70%+ agree
        medium_consensus = (consensus_votes >= (n_methods * 0.4)) & (consensus_votes < (n_methods * 0.7))
        low_consensus = consensus_votes < (n_methods * 0.4)
        
        # Plot with different colors
        ax.scatter(X[low_consensus], Y[low_consensus], 
                  c='lightblue', s=30, alpha=0.6, 
                  edgecolors='black', linewidth=0.5, label='Low Agreement')
        
        ax.scatter(X[medium_consensus], Y[medium_consensus], 
                  c='orange', s=60, alpha=0.7, 
                  edgecolors='black', linewidth=0.8, label='Medium Agreement')
        
        ax.scatter(X[high_consensus], Y[high_consensus], 
                  c='red', s=100, alpha=0.9, 
                  edgecolors='black', linewidth=1.5, 
                  marker='X', label='High Agreement (Outlier)')
        
        n_high = high_consensus.sum()
        high_pct = (n_high / len(high_consensus)) * 100
        
        ax.set_xlabel(feature_x, fontsize=10, fontweight='bold')
        ax.set_ylabel(feature_y, fontsize=10, fontweight='bold')
        ax.set_title(f'Consensus Outliers\n{n_high} nuclei ({high_pct:.1f}%) - High Agreement', 
                    fontsize=11, fontweight='bold', color='darkred')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_name}.png")
        
        # Also create a summary comparison
        self._plot_method_agreement_summary(outlier_methods, consensus_votes)
        
        return save_path
    
    def _plot_method_agreement_summary(self,
                                      outlier_methods: Dict[str, np.ndarray],
                                      consensus_votes: np.ndarray):
        """Create method agreement summary"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Outlier Detection Methods: Agreement Analysis', 
                    fontsize=14, fontweight='bold')
        
        # 1. Method-wise outlier counts
        ax = axes[0]
        method_names = list(outlier_methods.keys())
        outlier_counts = [outlier_methods[m].sum() for m in method_names]
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.9, len(method_names)))
        bars = ax.barh(method_names, outlier_counts, color=colors, 
                      edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Number of Outliers Detected', fontsize=11, fontweight='bold')
        ax.set_title('Outliers Detected by Each Method', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, count in zip(bars, outlier_counts):
            width = bar.get_width()
            ax.text(width + max(outlier_counts)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{int(count)}', ha='left', va='center', 
                   fontsize=10, fontweight='bold')
        
        # 2. Agreement distribution
        ax = axes[1]
        n_methods = len(method_names)
        
        agreement_counts = [(consensus_votes == i).sum() for i in range(n_methods + 1)]
        agreement_labels = [f'{i}/{n_methods}' for i in range(n_methods + 1)]
        
        colors_agree = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n_methods + 1))
        bars = ax.bar(agreement_labels, agreement_counts, color=colors_agree, 
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Number of Methods in Agreement', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Data Points', fontsize=11, fontweight='bold')
        ax.set_title('Agreement Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, agreement_counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{int(count)}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / 'method_agreement_summary.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: method_agreement_summary.png")
    
    def generate_all_anomaly_plots(self,
                                   anomaly_data: pd.DataFrame = None,
                                   feature_cols: List[str] = None,
                                   data: pd.DataFrame = None,
                                   outlier_methods: Dict[str, np.ndarray] = None,
                                   feature_x: str = 'A',
                                   feature_y: str = 'Z'):
        """Generate all 2 anomaly plots"""
        
        logger.info("\n" + "="*70)
        logger.info("GENERATING COMPLETE ANOMALY VISUALIZATIONS")
        logger.info("="*70)
        
        generated_plots = []
        
        # 1. Feature radar charts
        if anomaly_data is not None and feature_cols is not None:
            try:
                path = self.plot_anomaly_feature_radar_charts(
                    anomaly_data, feature_cols
                )
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  ✗ Failed radar charts: {e}")
        
        # 2. Outlier detection comparison
        if data is not None and outlier_methods is not None:
            try:
                path = self.plot_outlier_detection_comparison(
                    data, outlier_methods, feature_x, feature_y
                )
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  ✗ Failed outlier comparison: {e}")
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ ANOMALY VISUALIZATIONS COMPLETE: {len(generated_plots)}/2")
        logger.info("="*70)
        
        return generated_plots


def generate_sample_data():
    """Generate sample data for testing"""
    
    # Sample anomaly data
    n_samples = 100
    feature_cols = [f'Feature_{i}' for i in range(15)]
    
    anomaly_data = pd.DataFrame({
        'NUCLEUS': [f'Nucleus_{i}' for i in range(n_samples)],
        'A': np.random.randint(50, 250, n_samples),
        'Z': np.random.randint(20, 100, n_samples),
        'N': np.random.randint(30, 150, n_samples),
        'anomaly_score': np.random.uniform(0, 1, n_samples)
    })
    
    # Add feature columns
    for feat in feature_cols:
        anomaly_data[feat] = np.random.randn(n_samples)
    
    # Make some nuclei more anomalous
    top_anomalies = anomaly_data.nlargest(10, 'anomaly_score').index
    for feat in feature_cols[:5]:
        anomaly_data.loc[top_anomalies, feat] *= 3
    
    # Sample data for outlier detection
    data = pd.DataFrame({
        'A': np.random.randn(200) * 50 + 150,
        'Z': np.random.randn(200) * 20 + 60,
        'N': np.random.randn(200) * 30 + 90
    })
    
    # Add some outliers
    outlier_indices = np.random.choice(200, 15, replace=False)
    data.loc[outlier_indices, 'A'] += np.random.randn(15) * 100
    data.loc[outlier_indices, 'Z'] += np.random.randn(15) * 40
    
    # Sample outlier methods
    outlier_methods = {}
    for method in ['IsolationForest', 'LOF', 'OneClassSVM', 'DBSCAN', 'StatisticalZ']:
        # Random outliers with some overlap
        n_outliers = np.random.randint(10, 25)
        outliers = np.zeros(200, dtype=bool)
        outliers[np.random.choice(200, n_outliers, replace=False)] = True
        
        # Make true outliers more likely to be detected
        for idx in outlier_indices:
            if np.random.rand() > 0.3:  # 70% chance
                outliers[idx] = True
        
        outlier_methods[method] = outliers
    
    return anomaly_data, feature_cols, data, outlier_methods


if __name__ == "__main__":
    # Test with sample data
    logger.info("\n" + "="*70)
    logger.info("TESTING ANOMALY VISUALIZATIONS COMPLETE")
    logger.info("="*70)
    
    # Generate sample data
    anomaly_data, feature_cols, data, outlier_methods = generate_sample_data()
    
    # Create visualizer
    visualizer = AnomalyVisualizationsComplete('test_output/anomaly')
    
    # Generate all plots
    visualizer.generate_all_anomaly_plots(
        anomaly_data=anomaly_data,
        feature_cols=feature_cols,
        data=data,
        outlier_methods=outlier_methods,
        feature_x='A',
        feature_y='Z'
    )
    
    logger.info("\n✓ Test complete! Check test_output/anomaly/")
