"""
Ensemble Evaluator
==================

Tüm ensemble yöntemlerini karşılaştırır ve en iyi performansı sağlayanı seçer:
- Simple Voting
- Weighted Voting (R², RMSE, MAE, Inverse Error)
- Stacking (Ridge, Lasso, RF, GBM)

Author: AI Dataset Training Pipeline
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """
    Ensemble Evaluator
    
    Farklı ensemble yöntemlerini kapsamlı şekilde değerlendirir
    """
    
    def __init__(self, output_dir: str = 'output/ensemble_evaluation'):
        """
        Args:
            output_dir: Değerlendirme raporlarının kaydedileceği klasör
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []  # Tüm ensemble sonuçları
        
        logger.info(f"[OK] EnsembleEvaluator initialized")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def add_ensemble_result(self, 
                           ensemble_name: str,
                           ensemble_result: Dict,
                           predictions: np.ndarray,
                           y_true: np.ndarray):
        """
        Ensemble sonucu ekle
        
        Args:
            ensemble_name: Ensemble ismi
            ensemble_result: Ensemble metadata (method, weights, etc.)
            predictions: Model tahminleri
            y_true: Gerçek değerler
        """
        # Metrikleri hesapla
        r2 = r2_score(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        
        # Residuals
        residuals = y_true - predictions
        
        result = {
            'name': ensemble_name,
            'method': ensemble_result.get('method', 'unknown'),
            'n_models': ensemble_result.get('n_models', 0),
            'model_ids': ensemble_result.get('model_ids', []),
            'weights': ensemble_result.get('weights', None),
            'meta_model_type': ensemble_result.get('meta_model_type', None),
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'y_true': y_true,
            'residuals': residuals
        }
        
        self.results.append(result)
        
        logger.info(f"[OK] Added ensemble: {ensemble_name}")
        logger.info(f"  Method: {result['method']}")
        logger.info(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    def compare_all_ensembles(self) -> pd.DataFrame:
        """
        Tüm ensemble yöntemlerini karşılaştır
        
        Returns:
            Comparison DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPARING ALL ENSEMBLE METHODS")
        logger.info(f"{'='*60}")
        logger.info(f"Total ensembles: {len(self.results)}")
        
        comparison = []
        
        for result in self.results:
            comparison.append({
                'Ensemble': result['name'],
                'Method': result['method'],
                'N_Models': result['n_models'],
                'Meta_Model': result['meta_model_type'] if result['meta_model_type'] else '-',
                'R²': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae']
            })
        
        df = pd.DataFrame(comparison)
        
        # Sırala (R² göre)
        df = df.sort_values('R²', ascending=False)
        
        logger.info(f"\n{df.to_string(index=False)}")
        
        # Save Excel
        excel_path = self.output_dir / 'ensemble_comparison.xlsx'
        df.to_excel(excel_path, index=False)
        logger.info(f"\n[OK] Comparison table saved: {excel_path}")
        
        return df
    
    def select_best_ensemble(self, criterion: str = 'r2') -> Dict:
        """
        En iyi ensemble'ı seç
        
        Args:
            criterion: Seçim kriteri ('r2', 'rmse', 'mae', 'composite')
        
        Returns:
            Best ensemble result
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"SELECTING BEST ENSEMBLE")
        logger.info(f"{'='*60}")
        logger.info(f"Criterion: {criterion}")
        
        if len(self.results) == 0:
            raise ValueError("No ensemble results to select from!")
        
        if criterion == 'r2':
            best_result = max(self.results, key=lambda x: x['r2'])
        
        elif criterion == 'rmse':
            best_result = min(self.results, key=lambda x: x['rmse'])
        
        elif criterion == 'mae':
            best_result = min(self.results, key=lambda x: x['mae'])
        
        elif criterion == 'composite':
            # Composite score: R² yüksek, RMSE düşük, MAE düşük
            for result in self.results:
                result['composite_score'] = (
                    result['r2'] * 0.5 +
                    (1.0 / (result['rmse'] + 0.001)) * 0.25 +
                    (1.0 / (result['mae'] + 0.001)) * 0.25
                )
            best_result = max(self.results, key=lambda x: x['composite_score'])
        
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        logger.info(f"\n[SUCCESS] BEST ENSEMBLE: {best_result['name']}")
        logger.info(f"  Method: {best_result['method']}")
        logger.info(f"  R²: {best_result['r2']:.4f}")
        logger.info(f"  RMSE: {best_result['rmse']:.4f}")
        logger.info(f"  MAE: {best_result['mae']:.4f}")
        
        return best_result
    
    def visualize_ensemble_comparison(self):
        """
        Ensemble karşılaştırma görselleştirmesi
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CREATING VISUALIZATIONS")
        logger.info(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ensemble Methods Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data
        names = [r['name'] for r in self.results]
        r2_scores = [r['r2'] for r in self.results]
        rmse_scores = [r['rmse'] for r in self.results]
        mae_scores = [r['mae'] for r in self.results]
        
        # Plot 1: R² comparison
        ax1 = axes[0, 0]
        bars = ax1.barh(names, r2_scores, color='steelblue')
        ax1.set_xlabel('R² Score', fontweight='bold')
        ax1.set_title('R² Comparison')
        ax1.grid(axis='x', alpha=0.3)
        
        # Highlight best
        best_idx = r2_scores.index(max(r2_scores))
        bars[best_idx].set_color('darkgreen')
        
        # Plot 2: RMSE comparison
        ax2 = axes[0, 1]
        bars = ax2.barh(names, rmse_scores, color='coral')
        ax2.set_xlabel('RMSE', fontweight='bold')
        ax2.set_title('RMSE Comparison (Lower is better)')
        ax2.grid(axis='x', alpha=0.3)
        
        # Highlight best
        best_idx = rmse_scores.index(min(rmse_scores))
        bars[best_idx].set_color('darkgreen')
        
        # Plot 3: MAE comparison
        ax3 = axes[1, 0]
        bars = ax3.barh(names, mae_scores, color='gold')
        ax3.set_xlabel('MAE', fontweight='bold')
        ax3.set_title('MAE Comparison (Lower is better)')
        ax3.grid(axis='x', alpha=0.3)
        
        # Highlight best
        best_idx = mae_scores.index(min(mae_scores))
        bars[best_idx].set_color('darkgreen')
        
        # Plot 4: Scatter R² vs RMSE
        ax4 = axes[1, 1]
        scatter = ax4.scatter(rmse_scores, r2_scores, s=150, c=range(len(names)), 
                             cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Annotate points
        for i, name in enumerate(names):
            ax4.annotate(name, (rmse_scores[i], r2_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('RMSE', fontweight='bold')
        ax4.set_ylabel('R² Score', fontweight='bold')
        ax4.set_title('R² vs RMSE Trade-off')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = self.output_dir / 'ensemble_comparison_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Visualization saved: {plot_path}")
    
    def visualize_predictions(self, ensemble_name: Optional[str] = None):
        """
        Tahmin görselleştirmesi
        
        Args:
            ensemble_name: Görselleştirilecek ensemble (None ise en iyi)
        """
        if ensemble_name is None:
            # En iyiyi seç
            result = self.select_best_ensemble('r2')
        else:
            # İsimle bul
            result = next((r for r in self.results if r['name'] == ensemble_name), None)
            if result is None:
                logger.error(f"Ensemble not found: {ensemble_name}")
                return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"VISUALIZING PREDICTIONS: {result['name']}")
        logger.info(f"{'='*60}")
        
        y_true = result['y_true']
        y_pred = result['predictions']
        residuals = result['residuals']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Predictions: {result['name']}", fontsize=16, fontweight='bold')
        
        # Plot 1: True vs Predicted
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('True Values', fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontweight='bold')
        ax1.set_title(f"True vs Predicted (R²={result['r2']:.4f})")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Residuals
        ax2 = axes[1]
        ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values', fontweight='bold')
        ax2.set_ylabel('Residuals', fontweight='bold')
        ax2.set_title('Residual Plot')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Residual distribution
        ax3 = axes[2]
        ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Residuals', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title(f'Residual Distribution (MAE={result["mae"]:.4f})')
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = self.output_dir / f'predictions_{result["name"]}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Prediction visualization saved: {plot_path}")
    
    def generate_final_report(self) -> Dict:
        """
        Final ensemble değerlendirme raporu
        
        Returns:
            Report dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATING FINAL REPORT")
        logger.info(f"{'='*60}")
        
        # En iyi ensemble
        best_ensemble = self.select_best_ensemble('r2')
        
        # Tüm karşılaştırma
        comparison_df = self.compare_all_ensembles()
        
        # Görselleştirmeler
        self.visualize_ensemble_comparison()
        self.visualize_predictions(best_ensemble['name'])
        
        # Rapor
        report = {
            'total_ensembles': len(self.results),
            'best_ensemble': {
                'name': best_ensemble['name'],
                'method': best_ensemble['method'],
                'r2': best_ensemble['r2'],
                'rmse': best_ensemble['rmse'],
                'mae': best_ensemble['mae']
            },
            'all_results': [
                {
                    'name': r['name'],
                    'method': r['method'],
                    'r2': r['r2'],
                    'rmse': r['rmse'],
                    'mae': r['mae']
                }
                for r in self.results
            ]
        }
        
        # Save JSON
        report_path = self.output_dir / 'ensemble_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n[OK] Final report saved: {report_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ENSEMBLE EVALUATION COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"[SUCCESS] Best Ensemble: {best_ensemble['name']}")
        logger.info(f"   R² = {best_ensemble['r2']:.4f}")
        logger.info(f"   RMSE = {best_ensemble['rmse']:.4f}")
        logger.info(f"   MAE = {best_ensemble['mae']:.4f}")
        logger.info(f"{'='*60}")
        
        return report


def main():
    """Test function"""
    logger.info("="*60)
    logger.info("ENSEMBLE EVALUATOR - TEST")
    logger.info("="*60)
    
    # Mock data
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randn(n_samples) * 2 + 10
    
    # Mock ensemble predictions
    pred1 = y_true + np.random.randn(n_samples) * 0.5  # Simple Voting
    pred2 = y_true + np.random.randn(n_samples) * 0.4  # Weighted R2
    pred3 = y_true + np.random.randn(n_samples) * 0.45  # Weighted RMSE
    pred4 = y_true + np.random.randn(n_samples) * 0.3  # Stacking Ridge
    pred5 = y_true + np.random.randn(n_samples) * 0.35  # Stacking RF
    
    # Evaluator
    evaluator = EnsembleEvaluator('test_ensemble_evaluation')
    
    # Add results
    evaluator.add_ensemble_result(
        'SimpleVoting',
        {'method': 'simple_voting', 'n_models': 5},
        pred1, y_true
    )
    
    evaluator.add_ensemble_result(
        'WeightedVoting_R2',
        {'method': 'weighted_voting', 'n_models': 5, 'weights': [0.2, 0.25, 0.2, 0.15, 0.2]},
        pred2, y_true
    )
    
    evaluator.add_ensemble_result(
        'WeightedVoting_RMSE',
        {'method': 'weighted_voting', 'n_models': 5, 'weights': [0.18, 0.22, 0.25, 0.15, 0.2]},
        pred3, y_true
    )
    
    evaluator.add_ensemble_result(
        'Stacking_Ridge',
        {'method': 'stacking', 'n_models': 5, 'meta_model_type': 'ridge'},
        pred4, y_true
    )
    
    evaluator.add_ensemble_result(
        'Stacking_RF',
        {'method': 'stacking', 'n_models': 5, 'meta_model_type': 'rf'},
        pred5, y_true
    )
    
    # Compare
    comparison = evaluator.compare_all_ensembles()
    
    # Select best
    best = evaluator.select_best_ensemble('r2')
    
    # Generate report
    report = evaluator.generate_final_report()
    
    logger.info("\n[SUCCESS] TEST COMPLETED!")


if __name__ == "__main__":
    main()
