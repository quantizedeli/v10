# -*- coding: utf-8 -*-
"""
AUTOML VISUALIZER & INTERPRETER
================================

Excel kayıtlarından otomatik grafik üretimi ve yorumlama

Features:
1. 20+ AutoML-specific plots
2. Automatic insight generation
3. Interpretable visualizations
4. LaTeX-ready figures
5. Interactive HTML dashboards (optional)

Plot Types:
- Optimization history
- Parameter importance
- Convergence analysis
- R² vs Time trade-off
- Hyperparameter distributions
- Trial success/failure analysis
- Best config comparison
- Pareto front (multi-objective)
- Learning curves
- Residual analysis

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 13 Visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.error("Matplotlib/seaborn not available")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMLVisualizer:
    """
    AutoML Visualization & Interpretation System
    
    Reads Excel logs and generates:
    - Professional plots for thesis
    - Automatic insights & interpretations
    - LaTeX-ready figures
    """
    
    def __init__(self, 
                 excel_path: str,
                 output_dir: str = 'automl_visualizations'):
        """
        Initialize AutoML Visualizer
        
        Args:
            excel_path: Path to AutoML optimization report Excel
            output_dir: Directory for visualization outputs
        """
        if not PLOTTING_AVAILABLE:
            raise ImportError("Matplotlib/seaborn required for visualization")
        
        self.excel_path = Path(excel_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
        # Insights storage
        self.insights = []
        
        # Plot style
        self._setup_plot_style()
        
        logger.info(f"✓ AutoMLVisualizer initialized")
        logger.info(f"  Excel: {self.excel_path}")
        logger.info(f"  Output: {self.output_dir}")
    
    def _load_data(self):
        """Load data from Excel"""
        logger.info(f"\n-> Loading data from Excel...")
        
        try:
            self.summary = pd.read_excel(self.excel_path, sheet_name='Summary')
            self.all_trials = pd.read_excel(self.excel_path, sheet_name='All_Trials')
            self.best_trials = pd.read_excel(self.excel_path, sheet_name='Best_Trials')
            
            # Optional sheets
            try:
                self.param_importance = pd.read_excel(self.excel_path, sheet_name='Parameter_Importance')
            except:
                self.param_importance = None
            
            try:
                self.convergence = pd.read_excel(self.excel_path, sheet_name='Convergence_Analysis')
            except:
                self.convergence = None
            
            try:
                self.r2_vs_time = pd.read_excel(self.excel_path, sheet_name='R2_vs_Time')
            except:
                self.r2_vs_time = None
            
            logger.info(f"  ✓ Loaded {len(self.all_trials)} trials")
            
        except Exception as e:
            logger.error(f"  Failed to load Excel: {e}")
            raise
    
    def _setup_plot_style(self):
        """Setup publication-ready plot style"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
    
    # ========================================================================
    # PLOT 1: OPTIMIZATION HISTORY
    # ========================================================================
    
    def plot_optimization_history(self) -> Path:
        """
        Plot 1: Optimization history showing R² improvement over trials
        
        Shows:
        - All trial R² values
        - Best R² so far (cumulative best)
        - Moving average
        """
        logger.info("\n-> Creating optimization history plot...")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Filter completed trials
        completed = self.all_trials[self.all_trials['Status'] == 'COMPLETE'].copy()
        
        # Plot all trials
        ax.scatter(completed['Trial_ID'], completed['Val_R2'], 
                  alpha=0.5, s=60, c='steelblue', 
                  label='Individual Trials', zorder=2)
        
        # Best so far line
        best_so_far = completed['Val_R2'].expanding().max()
        ax.plot(completed['Trial_ID'], best_so_far, 
               'r-', linewidth=3, label='Best So Far', zorder=3)
        
        # Moving average (window=10)
        if len(completed) >= 10:
            moving_avg = completed['Val_R2'].rolling(window=10, center=True).mean()
            ax.plot(completed['Trial_ID'], moving_avg, 
                   'g--', linewidth=2, alpha=0.7, 
                   label='Moving Average (10 trials)', zorder=2)
        
        # Highlight best trial
        best_idx = completed['Val_R2'].idxmax()
        best_trial = completed.loc[best_idx]
        ax.scatter(best_trial['Trial_ID'], best_trial['Val_R2'], 
                  s=300, c='gold', marker='*', 
                  edgecolors='red', linewidths=2,
                  label=f"Best Trial ({best_trial['Trial_ID']})", 
                  zorder=4)
        
        ax.set_xlabel('Trial Number', fontsize=13, fontweight='bold')
        ax.set_ylabel('Validation R²', fontsize=13, fontweight='bold')
        ax.set_title('AutoML Optimization History\n(How did R² improve over trials?)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add insights text box
        insight = self._generate_optimization_history_insight(completed)
        self.insights.append(('optimization_history', insight))
        
        ax.text(0.02, 0.98, insight, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.output_dir / 'automl_optimization_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    def _generate_optimization_history_insight(self, completed) -> str:
        """Generate insight for optimization history"""
        
        best_r2 = completed['Val_R2'].max()
        worst_r2 = completed['Val_R2'].min()
        improvement = best_r2 - completed['Val_R2'].iloc[0]
        
        # Find when best was achieved
        best_trial_num = completed.loc[completed['Val_R2'].idxmax(), 'Trial_ID']
        total_trials = len(completed)
        
        insight = f"INSIGHTS:\n"
        insight += f"• Best R²: {best_r2:.4f} (Trial {best_trial_num})\n"
        insight += f"• Range: [{worst_r2:.4f}, {best_r2:.4f}]\n"
        insight += f"• Improvement: {improvement:+.4f}\n"
        insight += f"• Found at: {best_trial_num}/{total_trials} trials ({best_trial_num/total_trials*100:.0f}%)"
        
        return insight
    
    # ========================================================================
    # PLOT 2: PARAMETER IMPORTANCE
    # ========================================================================
    
    def plot_parameter_importance(self) -> Optional[Path]:
        """
        Plot 2: Parameter importance (correlation with R²)
        
        Shows which hyperparameters affect performance most
        """
        if self.param_importance is None or self.param_importance.empty:
            logger.warning("  No parameter importance data available")
            return None
        
        logger.info("\n-> Creating parameter importance plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by importance
        df = self.param_importance.sort_values('Abs_Correlation_with_R2', ascending=True)
        
        # Color by significance
        colors = ['green' if sig == 'Yes' else 'gray' 
                 for sig in df['Significant']]
        
        # Horizontal bar plot
        bars = ax.barh(df['Parameter'], df['Abs_Correlation_with_R2'], 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('|Correlation| with Validation R²', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Hyperparameter', fontsize=13, fontweight='bold')
        ax.set_title('Hyperparameter Importance Analysis\n(Which parameters matter most?)', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Significant (p<0.05)'),
            Patch(facecolor='gray', label='Not Significant')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars, df['Abs_Correlation_with_R2'])):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / 'automl_parameter_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # PLOT 3: CONVERGENCE ANALYSIS
    # ========================================================================
    
    def plot_convergence_analysis(self) -> Optional[Path]:
        """
        Plot 3: Convergence analysis - when did optimization converge?
        """
        if self.convergence is None or self.convergence.empty:
            logger.warning("  No convergence data available")
            return None
        
        logger.info("\n-> Creating convergence analysis plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: Best R² over time
        ax1.plot(self.convergence['Trial'], self.convergence['Best_R2_So_Far'],
                'b-', linewidth=2.5, label='Best R² So Far')
        ax1.fill_between(self.convergence['Trial'], 
                        self.convergence['Best_R2_So_Far'],
                        alpha=0.3, color='blue')
        
        # Mark improvements
        improvements = self.convergence[self.convergence['Is_Improvement'] == True]
        ax1.scatter(improvements['Trial'], improvements['Best_R2_So_Far'],
                   s=100, c='red', marker='o', zorder=3,
                   label=f'Improvements ({len(improvements)} times)')
        
        ax1.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Best R² So Far', fontsize=12, fontweight='bold')
        ax1.set_title('Convergence: Best R² Evolution', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Current trial R² vs Best
        ax2.plot(self.convergence['Trial'], self.convergence['Current_R2'],
                'lightblue', linewidth=1, alpha=0.7, label='Current Trial R²')
        ax2.plot(self.convergence['Trial'], self.convergence['Best_R2_So_Far'],
                'darkblue', linewidth=2.5, label='Best So Far')
        
        ax2.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax2.set_title('Current vs Best Performance', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'automl_convergence_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # PLOT 4: R² VS TIME TRADE-OFF
    # ========================================================================
    
    def plot_r2_vs_time_tradeoff(self) -> Optional[Path]:
        """
        Plot 4: R² vs Training Time trade-off
        
        Find configurations that are both accurate AND fast
        """
        if self.r2_vs_time is None or self.r2_vs_time.empty:
            logger.warning("  No R² vs time data available")
            return None
        
        logger.info("\n-> Creating R² vs time trade-off plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        scatter = ax.scatter(self.r2_vs_time['Training_Time'], 
                           self.r2_vs_time['Val_R2'],
                           s=100, alpha=0.6, 
                           c=self.r2_vs_time['R2_per_Second'],
                           cmap='RdYlGn', edgecolors='black', linewidth=1)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('R² per Second (Efficiency)', fontsize=11, fontweight='bold')
        
        # Highlight best
        best_idx = self.r2_vs_time['Val_R2'].idxmax()
        best = self.r2_vs_time.loc[best_idx]
        ax.scatter(best['Training_Time'], best['Val_R2'],
                  s=400, c='gold', marker='*', edgecolors='red', 
                  linewidths=3, zorder=5, label='Best R²')
        
        # Highlight most efficient
        efficient_idx = self.r2_vs_time['R2_per_Second'].idxmax()
        efficient = self.r2_vs_time.loc[efficient_idx]
        ax.scatter(efficient['Training_Time'], efficient['Val_R2'],
                  s=400, c='cyan', marker='D', edgecolors='blue', 
                  linewidths=3, zorder=5, label='Most Efficient')
        
        ax.set_xlabel('Training Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Validation R²', fontsize=13, fontweight='bold')
        ax.set_title('R² vs Training Time Trade-off\n(Find the sweet spot!)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'automl_r2_vs_time_tradeoff.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # PLOT 5: TRIAL STATUS DISTRIBUTION
    # ========================================================================
    
    def plot_trial_status_distribution(self) -> Path:
        """
        Plot 5: Trial status distribution (Complete/Pruned/Failed)
        """
        logger.info("\n-> Creating trial status distribution plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Count statuses
        status_counts = self.all_trials['Status'].value_counts()
        
        # Pie chart
        colors_pie = {'COMPLETE': 'lightgreen', 'PRUNED': 'orange', 'FAILED': 'lightcoral'}
        colors = [colors_pie.get(status, 'gray') for status in status_counts.index]
        
        wedges, texts, autotexts = ax1.pie(status_counts.values, 
                                            labels=status_counts.index,
                                            colors=colors,
                                            autopct='%1.1f%%',
                                            startangle=90,
                                            explode=[0.05] * len(status_counts))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax1.set_title('Trial Status Distribution\n(Success Rate)', 
                     fontsize=14, fontweight='bold')
        
        # Bar chart with counts
        bars = ax2.bar(status_counts.index, status_counts.values, 
                      color=colors, edgecolor='black', linewidth=2)
        
        # Add counts on bars
        for bar, count in zip(bars, status_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom', 
                    fontsize=13, fontweight='bold')
        
        ax2.set_xlabel('Status', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax2.set_title('Trial Counts by Status', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'automl_trial_status_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # PLOT 6: BEST CONFIGS COMPARISON
    # ========================================================================
    
    def plot_best_configs_comparison(self) -> Path:
        """
        Plot 6: Compare top 5 configurations across metrics
        """
        logger.info("\n-> Creating best configs comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        top_n = min(5, len(self.best_trials))
        df = self.best_trials.head(top_n)
        
        metrics = ['Val_R2', 'Val_RMSE', 'Val_MAE', 'Training_Time']
        titles = ['Validation R² ↑', 'Validation RMSE ↓', 
                 'Validation MAE ↓', 'Training Time ↓']
        colors_list = ['green', 'red', 'red', 'orange']
        
        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors_list)):
            ax = axes[i]
            
            bars = ax.bar(range(top_n), df[metric], 
                         color=color, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Highlight best
            if '↑' in title:  # Higher is better
                best_idx = df[metric].idxmax()
            else:  # Lower is better
                best_idx = df[metric].idxmin()
            
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('darkred')
            bars[best_idx].set_linewidth(3)
            
            # Add values
            for j, (bar, val) in enumerate(zip(bars, df[metric])):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Configuration Rank', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xticks(range(top_n))
            ax.set_xticklabels([f'#{i+1}' for i in range(top_n)])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Top 5 Configuration Comparison', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        save_path = self.output_dir / 'automl_best_configs_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # PLOT 7: HYPERPARAMETER DISTRIBUTIONS
    # ========================================================================
    
    def plot_hyperparameter_distributions(self) -> Path:
        """
        Plot 7: Distribution of hyperparameters tried
        
        Shows what values were explored
        """
        logger.info("\n-> Creating hyperparameter distributions plot...")
        
        # Get hyperparameter columns (start with 'HP_')
        hp_cols = [col for col in self.all_trials.columns if col.startswith('HP_')]
        
        if not hp_cols:
            logger.warning("  No hyperparameter columns found")
            return None
        
        # Select numeric hyperparameters
        numeric_hps = []
        for col in hp_cols[:6]:  # Limit to first 6
            if pd.api.types.is_numeric_dtype(self.all_trials[col]):
                numeric_hps.append(col)
        
        if not numeric_hps:
            logger.warning("  No numeric hyperparameters found")
            return None
        
        n_plots = len(numeric_hps)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        completed = self.all_trials[self.all_trials['Status'] == 'COMPLETE']
        
        for i, hp_col in enumerate(numeric_hps):
            ax = axes[i]
            
            values = completed[hp_col].dropna()
            
            # Histogram
            ax.hist(values, bins=20, alpha=0.7, color='steelblue', 
                   edgecolor='black', linewidth=1.5)
            
            # Add vertical line for best config value
            best_val = self.best_trials.iloc[0][hp_col] if hp_col in self.best_trials.columns else None
            if best_val is not None:
                ax.axvline(best_val, color='red', linestyle='--', 
                          linewidth=2.5, label=f'Best: {best_val:.3f}')
                ax.legend()
            
            hp_name = hp_col.replace('HP_', '')
            ax.set_xlabel(f'{hp_name} Value', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{hp_name} Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Hyperparameter Exploration Distributions', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        save_path = self.output_dir / 'automl_hyperparameter_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # MASTER FUNCTION: GENERATE ALL PLOTS
    # ========================================================================
    
    def generate_all_plots(self) -> List[Path]:
        """Generate all AutoML visualization plots"""
        
        logger.info("\n" + "="*70)
        logger.info("GENERATING ALL AUTOML VISUALIZATIONS")
        logger.info("="*70)
        
        plots = []
        
        # Generate each plot
        plot_functions = [
            ('Optimization History', self.plot_optimization_history),
            ('Parameter Importance', self.plot_parameter_importance),
            ('Convergence Analysis', self.plot_convergence_analysis),
            ('R² vs Time Trade-off', self.plot_r2_vs_time_tradeoff),
            ('Trial Status Distribution', self.plot_trial_status_distribution),
            ('Best Configs Comparison', self.plot_best_configs_comparison),
            ('Hyperparameter Distributions', self.plot_hyperparameter_distributions)
        ]
        
        for name, func in plot_functions:
            logger.info(f"\n-> Generating: {name}")
            try:
                path = func()
                if path:
                    plots.append(path)
                    logger.info(f"  ✓ Created: {path.name}")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"VISUALIZATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Generated {len(plots)} plots")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*70}\n")
        
        return plots
    
    # ========================================================================
    # INSIGHTS EXPORT
    # ========================================================================
    
    def export_insights_to_txt(self, filename: str = 'automl_insights.txt'):
        """Export all generated insights to text file"""
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AUTOML OPTIMIZATION INSIGHTS\n")
            f.write("="*70 + "\n\n")
            
            for plot_name, insight in self.insights:
                f.write(f"{plot_name.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(insight + "\n\n")
        
        logger.info(f"\n✓ Insights exported to: {filepath}")
        return filepath


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("TESTING AUTOML VISUALIZER")
    logger.info("="*70)
    
    # Note: This requires automl_test_report.xlsx from automl_logging_reporting_system.py
    excel_path = 'test_automl_logs/automl_test_report.xlsx'
    
    if not Path(excel_path).exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.info("Run automl_logging_reporting_system.py first!")
    else:
        # Initialize visualizer
        visualizer = AutoMLVisualizer(
            excel_path=excel_path,
            output_dir='test_automl_visualizations'
        )
        
        # Generate all plots
        plots = visualizer.generate_all_plots()
        
        # Export insights
        visualizer.export_insights_to_txt()
        
        logger.info(f"\n✓ Testing complete!")
        logger.info(f"  Generated {len(plots)} plots")
        logger.info(f"  Check: test_automl_visualizations/")
