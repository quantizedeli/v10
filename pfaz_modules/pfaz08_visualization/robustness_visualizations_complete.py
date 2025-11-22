# -*- coding: utf-8 -*-
"""
ROBUSTNESS VISUALIZATIONS COMPLETE
===================================

Eksik robustness grafiklerini tamamlayan modül

Yeni Grafikler:
1. noise_sensitivity_detailed.png - Farklı noise seviyeleri detaylı analiz
2. robustness_score_heatmap.png - Model × Target robustness ısı haritası
3. perturbation_impact_analysis.png - Feature perturbation etki analizi

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
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plot configuration
PLOT_CONFIG = {
    'dpi': 300,
    'figsize_default': (14, 10),
    'figsize_wide': (16, 6),
    'style': 'seaborn-v0_8-darkgrid'
}

plt.style.use(PLOT_CONFIG['style'])
sns.set_palette("husl")


class RobustnessVisualizationsComplete:
    """Robustness testleri için eksik grafikleri tamamlar"""
    
    def __init__(self, output_dir='visualizations/robustness_complete'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[OK] RobustnessVisualizationsComplete initialized: {self.output_dir}")
    
    def plot_noise_sensitivity_detailed(self, 
                                       robustness_results: Dict[str, Dict],
                                       save_name: str = 'noise_sensitivity_detailed'):
        """
        Detaylı noise sensitivity analizi
        
        Args:
            robustness_results: {
                'model_id': {
                    'noise_0.01': {'r2': 0.95, 'rmse': 0.12},
                    'noise_0.05': {'r2': 0.92, 'rmse': 0.15},
                    'noise_0.10': {'r2': 0.88, 'rmse': 0.18},
                    'noise_0.20': {'r2': 0.82, 'rmse': 0.22}
                }
            }
        """
        logger.info("\n-> Creating detailed noise sensitivity plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Noise Sensitivity Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Parse results
        noise_levels = []
        models = list(robustness_results.keys())
        
        for model_id, results in robustness_results.items():
            for key in results.keys():
                if key.startswith('noise_'):
                    noise = float(key.split('_')[1])
                    if noise not in noise_levels:
                        noise_levels.append(noise)
        
        noise_levels = sorted(noise_levels)
        
        # 1. R² degradation curves
        ax = axes[0, 0]
        for model_id in models[:10]:  # Top 10 models
            r2_values = []
            for noise in noise_levels:
                key = f'noise_{noise:.2f}'
                if key in robustness_results[model_id]:
                    r2_values.append(robustness_results[model_id][key]['r2'])
                else:
                    r2_values.append(np.nan)
            
            if len(r2_values) > 0:
                ax.plot(noise_levels, r2_values, marker='o', 
                       label=model_id[:15], linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
        ax.set_title('R² Degradation with Noise', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # 2. RMSE increase curves
        ax = axes[0, 1]
        for model_id in models[:10]:
            rmse_values = []
            for noise in noise_levels:
                key = f'noise_{noise:.2f}'
                if key in robustness_results[model_id]:
                    rmse_values.append(robustness_results[model_id][key]['rmse'])
                else:
                    rmse_values.append(np.nan)
            
            if len(rmse_values) > 0:
                ax.plot(noise_levels, rmse_values, marker='s', 
                       label=model_id[:15], linewidth=2, markersize=6)
        
        ax.set_xlabel('Noise Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax.set_title('RMSE Increase with Noise', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # 3. Robustness score distribution
        ax = axes[1, 0]
        robustness_scores = []
        model_labels = []
        
        for model_id in models:
            # Calculate robustness score (average R² across noise levels)
            r2_vals = []
            for noise in noise_levels:
                key = f'noise_{noise:.2f}'
                if key in robustness_results[model_id]:
                    r2_vals.append(robustness_results[model_id][key]['r2'])
            
            if len(r2_vals) > 0:
                robustness_scores.append(np.mean(r2_vals))
                model_labels.append(model_id[:20])
        
        # Sort by robustness
        sorted_indices = np.argsort(robustness_scores)[::-1][:15]
        sorted_scores = [robustness_scores[i] for i in sorted_indices]
        sorted_labels = [model_labels[i] for i in sorted_indices]
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_scores)))
        ax.barh(range(len(sorted_scores)), sorted_scores, color=colors, 
               edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(sorted_scores)))
        ax.set_yticklabels(sorted_labels, fontsize=8)
        ax.set_xlabel('Average R² (Robustness Score)', fontsize=11, fontweight='bold')
        ax.set_title('Top 15 Most Robust Models', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 1.05])
        
        # Add score labels
        for i, score in enumerate(sorted_scores):
            ax.text(score + 0.02, i, f'{score:.3f}', 
                   va='center', fontsize=8, fontweight='bold')
        
        # 4. Noise tolerance box plot
        ax = axes[1, 1]
        
        # Prepare data for box plot
        noise_tolerance_data = []
        noise_labels = []
        
        for noise in noise_levels:
            key = f'noise_{noise:.2f}'
            r2_at_noise = []
            
            for model_id in models:
                if key in robustness_results[model_id]:
                    r2_at_noise.append(robustness_results[model_id][key]['r2'])
            
            if len(r2_at_noise) > 0:
                noise_tolerance_data.append(r2_at_noise)
                noise_labels.append(f'{noise:.2f}')
        
        bp = ax.boxplot(noise_tolerance_data, labels=noise_labels, 
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color boxes
        colors_box = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Noise Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('R² Score Distribution', fontsize=11, fontweight='bold')
        ax.set_title('R² Distribution Across Noise Levels', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] Saved: {save_name}.png")
        return save_path
    
    def plot_robustness_score_heatmap(self,
                                     robustness_matrix: pd.DataFrame,
                                     save_name: str = 'robustness_score_heatmap'):
        """
        Model × Target robustness score heatmap
        
        Args:
            robustness_matrix: DataFrame with models as rows, targets as columns
                               Values are robustness scores (0-1)
        """
        logger.info("\n-> Creating robustness score heatmap...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create heatmap
        im = ax.imshow(robustness_matrix.values, cmap='RdYlGn', 
                      aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(robustness_matrix.columns)))
        ax.set_yticks(np.arange(len(robustness_matrix.index)))
        ax.set_xticklabels(robustness_matrix.columns, fontsize=10, fontweight='bold')
        ax.set_yticklabels(robustness_matrix.index, fontsize=9)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Robustness Score', fontsize=11, fontweight='bold')
        
        # Add value annotations
        for i in range(len(robustness_matrix.index)):
            for j in range(len(robustness_matrix.columns)):
                value = robustness_matrix.iloc[i, j]
                if not np.isnan(value):
                    color = 'white' if value < 0.5 else 'black'
                    text = ax.text(j, i, f'{value:.3f}',
                                 ha='center', va='center', 
                                 color=color, fontsize=8, fontweight='bold')
        
        ax.set_title('Model × Target Robustness Score Heatmap\n(Higher = More Robust)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Property', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.set_xticks(np.arange(len(robustness_matrix.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(robustness_matrix.index)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] Saved: {save_name}.png")
        return save_path
    
    def plot_perturbation_impact_analysis(self,
                                         perturbation_results: Dict[str, Dict],
                                         feature_names: List[str],
                                         save_name: str = 'perturbation_impact_analysis'):
        """
        Feature perturbation impact analysis
        
        Args:
            perturbation_results: {
                'feature1': {
                    'baseline_r2': 0.95,
                    'perturbed_r2': 0.92,
                    'impact': 0.03,
                    'importance': 0.75
                },
                ...
            }
        """
        logger.info("\n-> Creating perturbation impact analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Perturbation Impact Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Extract data
        features = list(perturbation_results.keys())
        impacts = [perturbation_results[f]['impact'] for f in features]
        baseline_r2s = [perturbation_results[f]['baseline_r2'] for f in features]
        perturbed_r2s = [perturbation_results[f]['perturbed_r2'] for f in features]
        
        # Sort by impact
        sorted_indices = np.argsort(impacts)[::-1]
        top_n = min(20, len(sorted_indices))
        
        # 1. Top features by perturbation impact
        ax = axes[0, 0]
        top_features = [features[i] for i in sorted_indices[:top_n]]
        top_impacts = [impacts[i] for i in sorted_indices[:top_n]]
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_impacts)))
        ax.barh(range(len(top_impacts)), top_impacts, color=colors,
               edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(top_impacts)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('R² Drop (Impact)', fontsize=11, fontweight='bold')
        ax.set_title('Top 20 Features by Perturbation Impact', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, impact in enumerate(top_impacts):
            ax.text(impact + 0.001, i, f'{impact:.4f}', 
                   va='center', fontsize=8, fontweight='bold')
        
        # 2. Baseline vs Perturbed scatter
        ax = axes[0, 1]
        scatter = ax.scatter(baseline_r2s, perturbed_r2s, 
                           c=impacts, cmap='Reds', s=100, 
                           alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # Add diagonal line (no impact)
        lims = [0, 1]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='No Impact Line')
        
        ax.set_xlabel('Baseline R²', fontsize=11, fontweight='bold')
        ax.set_ylabel('Perturbed R² (±10%)', fontsize=11, fontweight='bold')
        ax.set_title('Baseline vs Perturbed R²', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Impact', fontsize=10, fontweight='bold')
        
        # 3. Impact distribution
        ax = axes[1, 0]
        ax.hist(impacts, bins=30, color='coral', alpha=0.7, 
               edgecolor='black', linewidth=1.5)
        ax.axvline(np.mean(impacts), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(impacts):.4f}')
        ax.axvline(np.median(impacts), color='blue', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(impacts):.4f}')
        
        ax.set_xlabel('R² Impact', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Perturbation Impacts', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Cumulative impact
        ax = axes[1, 1]
        sorted_impacts = sorted(impacts, reverse=True)
        cumulative_impact = np.cumsum(sorted_impacts)
        cumulative_pct = (cumulative_impact / cumulative_impact[-1]) * 100
        
        ax.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, 
               linewidth=3, color='steelblue', marker='o', markersize=4)
        
        # Add threshold lines
        ax.axhline(80, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='80% Impact')
        ax.axhline(95, color='orange', linestyle='--', linewidth=2, 
                  alpha=0.7, label='95% Impact')
        
        # Find features contributing to 80%
        n_80 = np.where(cumulative_pct >= 80)[0][0] + 1
        ax.axvline(n_80, color='red', linestyle=':', linewidth=2, alpha=0.5)
        ax.text(n_80, 50, f'Top {n_80} features\n= 80% impact', 
               fontsize=9, ha='center', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Impact (%)', fontsize=11, fontweight='bold')
        ax.set_title('Cumulative Perturbation Impact', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] Saved: {save_name}.png")
        return save_path
    
    def generate_all_robustness_plots(self, 
                                      robustness_results: Dict = None,
                                      robustness_matrix: pd.DataFrame = None,
                                      perturbation_results: Dict = None,
                                      feature_names: List[str] = None):
        """Generate all 3 robustness plots"""
        
        logger.info("\n" + "="*70)
        logger.info("GENERATING COMPLETE ROBUSTNESS VISUALIZATIONS")
        logger.info("="*70)
        
        generated_plots = []
        
        # 1. Noise sensitivity detailed
        if robustness_results is not None:
            try:
                path = self.plot_noise_sensitivity_detailed(robustness_results)
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  [FAIL] Failed noise sensitivity: {e}")
        
        # 2. Robustness score heatmap
        if robustness_matrix is not None:
            try:
                path = self.plot_robustness_score_heatmap(robustness_matrix)
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  [FAIL] Failed heatmap: {e}")
        
        # 3. Perturbation impact
        if perturbation_results is not None and feature_names is not None:
            try:
                path = self.plot_perturbation_impact_analysis(
                    perturbation_results, feature_names
                )
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  [FAIL] Failed perturbation: {e}")
        
        logger.info("\n" + "="*70)
        logger.info(f"[OK] ROBUSTNESS VISUALIZATIONS COMPLETE: {len(generated_plots)}/3")
        logger.info("="*70)
        
        return generated_plots


def generate_sample_data():
    """Generate sample data for testing"""
    
    # Sample robustness results
    models = [f'Model_{i}' for i in range(15)]
    noise_levels = [0.01, 0.05, 0.10, 0.20]
    
    robustness_results = {}
    for model in models:
        robustness_results[model] = {}
        base_r2 = np.random.uniform(0.85, 0.98)
        
        for noise in noise_levels:
            degradation = noise * np.random.uniform(1, 3)
            r2 = max(0.5, base_r2 - degradation)
            rmse = 0.1 + noise * np.random.uniform(1, 2)
            
            robustness_results[model][f'noise_{noise:.2f}'] = {
                'r2': r2,
                'rmse': rmse
            }
    
    # Sample robustness matrix
    targets = ['MM', 'QM', 'Beta_2', 'BE']
    robustness_matrix = pd.DataFrame(
        np.random.uniform(0.7, 0.95, (len(models), len(targets))),
        index=models,
        columns=targets
    )
    
    # Sample perturbation results
    features = [f'Feature_{i}' for i in range(30)]
    perturbation_results = {}
    
    for feature in features:
        baseline = np.random.uniform(0.90, 0.98)
        impact = np.random.uniform(0.001, 0.05)
        perturbation_results[feature] = {
            'baseline_r2': baseline,
            'perturbed_r2': baseline - impact,
            'impact': impact,
            'importance': np.random.uniform(0.1, 1.0)
        }
    
    return robustness_results, robustness_matrix, perturbation_results, features


if __name__ == "__main__":
    # Test with sample data
    logger.info("\n" + "="*70)
    logger.info("TESTING ROBUSTNESS VISUALIZATIONS COMPLETE")
    logger.info("="*70)
    
    # Generate sample data
    robustness_results, robustness_matrix, perturbation_results, features = generate_sample_data()
    
    # Create visualizer
    visualizer = RobustnessVisualizationsComplete('test_output/robustness')
    
    # Generate all plots
    visualizer.generate_all_robustness_plots(
        robustness_results=robustness_results,
        robustness_matrix=robustness_matrix,
        perturbation_results=perturbation_results,
        feature_names=features
    )
    
    logger.info("\n[OK] Test complete! Check test_output/robustness/")
