"""
ANFIS Visualization Module
ANFIS Sonuçları Görselleştirme

Özellikler:
1. Training curves (epoch vs error)
2. Membership function plots
3. Surface plots (2D input-output)
4. Prediction scatter plots
5. Residual analysis
6. Method comparison charts

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# ANFIS VISUALIZER
# ============================================================================

class ANFISVisualizer:
    """
    ANFIS sonuçları için görselleştirme
    """
    
    def __init__(self, output_dir='anfis_visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ANFIS Visualizer başlatıldı")
    
    def plot_training_curve(self, history_df, title='ANFIS Training Curve', save_name='training_curve'):
        """
        Training curve plot
        
        Args:
            history_df: DataFrame with 'epoch', 'train_error', 'check_error'
        """
        
        logger.info(f"Training curve oluşturuluyor: {save_name}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(history_df['epoch'], history_df['train_error'], 
                'o-', label='Training Error', linewidth=2, markersize=4)
        ax.plot(history_df['epoch'], history_df['check_error'], 
                's-', label='Validation Error', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Error (RMSE)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Semilogy if range is large
        if history_df['train_error'].max() / history_df['train_error'].min() > 10:
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive plotly version
        fig_plotly = go.Figure()
        
        fig_plotly.add_trace(go.Scatter(
            x=history_df['epoch'],
            y=history_df['train_error'],
            mode='lines+markers',
            name='Training Error',
            line=dict(width=2),
            marker=dict(size=6)
        ))
        
        fig_plotly.add_trace(go.Scatter(
            x=history_df['epoch'],
            y=history_df['check_error'],
            mode='lines+markers',
            name='Validation Error',
            line=dict(width=2),
            marker=dict(size=6)
        ))
        
        fig_plotly.update_layout(
            title=title,
            xaxis_title='Epoch',
            yaxis_title='Error (RMSE)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig_plotly.write_html(self.output_dir / f'{save_name}.html')
        
        logger.info(f"  [OK] {save_name}.png & .html")
    
    def plot_predictions_scatter(self, y_true, y_pred, split_name='test', 
                                 title='ANFIS Predictions', save_name='predictions_scatter'):
        """
        Prediction scatter plot (actual vs predicted)
        """
        
        logger.info(f"Prediction scatter: {save_name}")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Text box
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{title} ({split_name})', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_name}_{split_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] {save_name}_{split_name}.png")
    
    def plot_residuals(self, y_true, y_pred, split_name='test', 
                      title='ANFIS Residuals', save_name='residuals'):
        """
        Residual analysis plots
        """
        
        logger.info(f"Residual plots: {save_name}")
        
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5, s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residual Plot', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'{title} ({split_name})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{save_name}_{split_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] {save_name}_{split_name}.png")
    
    def plot_method_comparison(self, method1_results, method2_results, target_name):
        """
        Method 1 vs Method 2 comparison
        
        Args:
            method1_results: List of result dicts from Method 1
            method2_results: List of result dicts from Method 2
            target_name: Target variable name
        """
        
        logger.info(f"Method comparison: {target_name}")
        
        # Prepare data
        m1_r2 = [r['metrics']['test']['R2'] for r in method1_results]
        m2_r2 = [r['metrics']['test']['R2'] for r in method2_results]
        
        m1_rmse = [r['metrics']['test']['RMSE'] for r in method1_results]
        m2_rmse = [r['metrics']['test']['RMSE'] for r in method2_results]
        
        m1_time = [r['training_time'] for r in method1_results]
        m2_time = [r['training_time'] for r in method2_results]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # R² comparison
        data_r2 = [m1_r2, m2_r2]
        bp1 = axes[0].boxplot(data_r2, labels=['Method 1\n(Layered)', 'Method 2\n(Balanced)'],
                               patch_artist=True)
        for patch, color in zip(bp1['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² Comparison', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # RMSE comparison
        data_rmse = [m1_rmse, m2_rmse]
        bp2 = axes[1].boxplot(data_rmse, labels=['Method 1\n(Layered)', 'Method 2\n(Balanced)'],
                               patch_artist=True)
        for patch, color in zip(bp2['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Training time comparison
        data_time = [m1_time, m2_time]
        bp3 = axes[2].boxplot(data_time, labels=['Method 1\n(Layered)', 'Method 2\n(Balanced)'],
                               patch_artist=True)
        for patch, color in zip(bp3['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        axes[2].set_ylabel('Training Time (s)', fontsize=12)
        axes[2].set_title('Training Time Comparison', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'ANFIS Method Comparison - {target_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / f'method_comparison_{target_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] method_comparison_{target_name}.png")
        
        # Statistics table
        stats_df = pd.DataFrame({
            'Method': ['Method 1', 'Method 2'],
            'R2_mean': [np.mean(m1_r2), np.mean(m2_r2)],
            'R2_std': [np.std(m1_r2), np.std(m2_r2)],
            'RMSE_mean': [np.mean(m1_rmse), np.mean(m2_rmse)],
            'RMSE_std': [np.std(m1_rmse), np.std(m2_rmse)],
            'Time_mean': [np.mean(m1_time), np.mean(m2_time)],
            'Time_std': [np.std(m1_time), np.std(m2_time)]
        })
        
        stats_df.to_csv(self.output_dir / f'method_comparison_{target_name}.csv', index=False)
        logger.info(f"  [OK] method_comparison_{target_name}.csv")
    
    def plot_target_comparison(self, results_by_target):
        """
        Compare ANFIS performance across targets
        
        Args:
            results_by_target: Dict {target_name: [results]}
        """
        
        logger.info("Target comparison plot")
        
        # Prepare data
        data = []
        for target, results in results_by_target.items():
            for result in results:
                data.append({
                    'Target': target,
                    'R2': result['metrics']['test']['R2'],
                    'RMSE': result['metrics']['test']['RMSE'],
                    'MAE': result['metrics']['test']['MAE']
                })
        
        df = pd.DataFrame(data)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # R² by target
        df.boxplot(column='R2', by='Target', ax=axes[0])
        axes[0].set_xlabel('Target', fontsize=12)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² by Target', fontsize=13, fontweight='bold')
        axes[0].get_figure().suptitle('')  # Remove default title
        
        # RMSE by target
        df.boxplot(column='RMSE', by='Target', ax=axes[1])
        axes[1].set_xlabel('Target', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE by Target', fontsize=13, fontweight='bold')
        
        # MAE by target
        df.boxplot(column='MAE', by='Target', ax=axes[2])
        axes[2].set_xlabel('Target', fontsize=12)
        axes[2].set_ylabel('MAE', fontsize=12)
        axes[2].set_title('MAE by Target', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  [OK] target_comparison.png")
        
        # Summary table
        summary = df.groupby('Target').agg({
            'R2': ['mean', 'std', 'min', 'max'],
            'RMSE': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        summary.to_csv(self.output_dir / 'target_comparison_summary.csv')
        logger.info("  [OK] target_comparison_summary.csv")
    
    def create_performance_heatmap(self, results_df):
        """
        Performance heatmap (datasets × metrics)
        
        Args:
            results_df: DataFrame with columns ['dataset_name', 'r2_test', 'rmse_test', 'mae_test']
        """
        
        logger.info("Performance heatmap")
        
        # Select top/bottom datasets
        n_show = min(20, len(results_df))
        top_df = results_df.nlargest(n_show // 2, 'r2_test')
        bottom_df = results_df.nsmallest(n_show // 2, 'r2_test')
        plot_df = pd.concat([top_df, bottom_df])
        
        # Prepare data for heatmap
        heatmap_data = plot_df[['r2_test', 'rmse_test', 'mae_test']].T
        heatmap_data.columns = plot_df['dataset_name'].values
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(12, n_show * 0.6), 6))
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        ax.set_title('ANFIS Performance Heatmap (Top & Bottom)', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  [OK] performance_heatmap.png")


# ============================================================================
# ANFIS BATCH VISUALIZER
# ============================================================================

class ANFISBatchVisualizer:
    """
    Batch ANFIS results visualization
    """
    
    def __init__(self, anfis_results_dir='anfis_results', output_dir='anfis_visualizations'):
        self.anfis_results_dir = Path(anfis_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = ANFISVisualizer(output_dir)
        
        logger.info("ANFIS Batch Visualizer başlatıldı")
    
    def visualize_all_results(self):
        """Tüm ANFIS sonuçlarını görselleştir"""
        
        logger.info("\n" + "="*80)
        logger.info("ANFIS BATCH VISUALIZATION")
        logger.info("="*80)
        
        # Find all result directories
        result_dirs = list(self.anfis_results_dir.rglob('metrics.json'))
        
        logger.info(f"\nFound {len(result_dirs)} ANFIS results")
        
        # Visualize each result
        for i, metrics_file in enumerate(result_dirs):
            result_dir = metrics_file.parent
            
            logger.info(f"\n[{i+1}/{len(result_dirs)}] {result_dir.name}")
            
            try:
                self._visualize_single_result(result_dir)
            except Exception as e:
                logger.error(f"  [FAIL] Visualization error: {e}")
        
        # Overall summary
        self._create_overall_summary()
        
        logger.info("\n" + "="*80)
        logger.info("[OK] BATCH VISUALIZATION TAMAMLANDI")
        logger.info("="*80)
    
    def _visualize_single_result(self, result_dir):
        """Tek bir ANFIS sonucunu görselleştir"""
        
        # Load data
        with open(result_dir / 'metrics.json', encoding='utf-8') as f:
            metrics = json.load(f)
        
        history_df = pd.read_csv(result_dir / 'training_history.csv')
        
        # Predictions
        y_test_pred = pd.read_csv(result_dir / 'predictions_test.csv')['y_pred'].values
        
        # We need y_true - try to load from dataset
        # (simplified - in real case, load from dataset)
        dataset_name = result_dir.name
        
        # Training curve
        viz_dir = result_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        temp_viz = ANFISVisualizer(viz_dir)
        temp_viz.plot_training_curve(history_df, title=f'ANFIS Training: {dataset_name}')
        
        logger.info(f"  [OK] Visualizations created")
    
    def _create_overall_summary(self):
        """Create overall summary visualizations"""
        
        logger.info("\nCreating overall summary...")
        
        # Load all results
        all_results = []
        
        for metrics_file in self.anfis_results_dir.rglob('metrics.json'):
            with open(metrics_file, encoding='utf-8') as f:
                data = json.load(f)
                data['dataset_name'] = metrics_file.parent.name
                all_results.append(data)
        
        if not all_results:
            logger.warning("No results found for summary")
            return
        
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            summary_data.append({
                'dataset_name': result['dataset_name'],
                'r2_train': result['metrics']['train']['R2'],
                'r2_test': result['metrics']['test']['R2'],
                'rmse_test': result['metrics']['test']['RMSE'],
                'mae_test': result['metrics']['test']['MAE'],
                'training_time': result['training_time'],
                'epochs': result['epochs_completed']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_df.to_csv(self.output_dir / 'anfis_overall_summary.csv', index=False)
        
        # Visualizations
        self.visualizer.create_performance_heatmap(summary_df)
        
        # Distribution plots
        self._plot_distributions(summary_df)
        
        logger.info("  [OK] Overall summary created")
    
    def _plot_distributions(self, summary_df):
        """Plot metric distributions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # R² distribution
        axes[0, 0].hist(summary_df['r2_test'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('R² Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('R² Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].axvline(summary_df['r2_test'].mean(), color='r', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # RMSE distribution
        axes[0, 1].hist(summary_df['rmse_test'], bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('RMSE', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('RMSE Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].axvline(summary_df['rmse_test'].mean(), color='r', linestyle='--', linewidth=2, label='Mean')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Training time distribution
        axes[1, 0].hist(summary_df['training_time'], bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Training Time (s)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Training Time Distribution', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # R² train vs test
        axes[1, 1].scatter(summary_df['r2_train'], summary_df['r2_test'], alpha=0.6, edgecolors='k', s=50)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[1, 1].set_xlabel('R² Train', fontsize=12)
        axes[1, 1].set_ylabel('R² Test', fontsize=12)
        axes[1, 1].set_title('Train vs Test R²', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle('ANFIS Overall Performance Distributions', fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  [OK] Distribution plots created")


# ============================================================================
# MAIN TEST
# ============================================================================

def test_anfis_visualizer():
    """Test ANFIS visualizer"""
    
    print("\n" + "="*80)
    print("ANFIS VISUALIZER TEST")
    print("="*80)
    
    # Dummy data
    np.random.seed(42)
    
    # Training history
    epochs = 50
    train_error = np.exp(-np.linspace(0, 3, epochs)) + np.random.rand(epochs) * 0.05
    check_error = np.exp(-np.linspace(0, 2.5, epochs)) + np.random.rand(epochs) * 0.08
    
    history_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_error': train_error,
        'check_error': check_error
    })
    
    # Predictions
    n_samples = 100
    y_true = np.random.randn(n_samples)
    y_pred = y_true + np.random.randn(n_samples) * 0.2
    
    # Visualizer
    viz = ANFISVisualizer(output_dir='test_anfis_viz')
    
    # Training curve
    viz.plot_training_curve(history_df, title='Test ANFIS Training')
    
    # Predictions
    viz.plot_predictions_scatter(y_true, y_pred, split_name='test')
    
    # Residuals
    viz.plot_residuals(y_true, y_pred, split_name='test')
    
    print("\n[OK] Test tamamlandı!")
    print(f"Visualizations saved to: test_anfis_viz/")


if __name__ == "__main__":
    test_anfis_visualizer()