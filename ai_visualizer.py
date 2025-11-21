"""
AI Model Visualization Module
Learning Curves, Residuals, Feature Importance

9. modül - visualization/ klasöründe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIVisualizer:
    """AI model sonuçları için özel görselleştirmeler"""
    
    def __init__(self, output_dir='visualizations/ai_models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_learning_curves(self, history, model_name='DNN'):
        """DNN/BNN için learning curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(history['loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE curve (if available)
        if 'mae' in history:
            axes[1].plot(history['mae'], label='Train MAE')
            axes[1].plot(history['val_mae'], label='Val MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title(f'{model_name} - MAE Curve')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_learning_curves.png', dpi=300)
        plt.close()
        
        logger.info(f"✓ Learning curves: {model_name}")
    
    def plot_feature_importance_comparison(self, importances_dict):
        """Compare feature importance across models"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        df_list = []
        for model_name, importance in importances_dict.items():
            df = pd.DataFrame({
                'Feature': importance['features'],
                'Importance': importance['values'],
                'Model': model_name
            })
            df_list.append(df)
        
        combined_df = pd.concat(df_list)
        
        # Plot
        sns.barplot(data=combined_df, x='Importance', y='Feature', hue='Model', ax=ax)
        ax.set_title('Feature Importance Comparison Across Models', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance_comparison.png', dpi=300)
        plt.close()
        
        logger.info("✓ Feature importance comparison")
    
    def plot_residuals_advanced(self, y_true, y_pred, model_name):
        """Advanced residual analysis"""
        residuals = y_pred - y_true
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Residual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        
        # 2. Histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(residuals, bins=30, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        
        # 3. Q-Q Plot
        ax3 = plt.subplot(2, 3, 3)
        from scipy import stats
        stats.probplot(residuals.flatten(), dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        
        # 4. Actual vs Predicted
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax4.set_xlabel('Actual')
        ax4.set_ylabel('Predicted')
        ax4.set_title('Actual vs Predicted')
        
        # 5. Residual scale-location
        ax5 = plt.subplot(2, 3, 5)
        standardized_residuals = residuals / np.std(residuals)
        ax5.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('√|Standardized Residuals|')
        ax5.set_title('Scale-Location Plot')
        
        # 6. Residual autocorrelation
        ax6 = plt.subplot(2, 3, 6)
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(pd.Series(residuals.flatten()), ax=ax6)
        ax6.set_title('Residual Autocorrelation')
        
        fig.suptitle(f'{model_name} - Advanced Residual Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_residuals_advanced.png', dpi=300)
        plt.close()
        
        logger.info(f"✓ Advanced residuals: {model_name}")


if __name__ == "__main__":
    print("✓ AI Visualizer modülü hazır - visualization/ai_visualizer.py")