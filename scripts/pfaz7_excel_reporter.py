"""
PFAZ7 Excel Reporter
====================

Ensemble sonuçlarını kapsamlı Excel raporuna dönüştürür

Author: AI Dataset Training Pipeline
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import BarChart, Reference, ScatterChart, Series
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logger.warning("openpyxl not available - using basic Excel output")


class PFAZ7ExcelReporter:
    """
    PFAZ7 Ensemble Results Excel Reporter

    Tüm ensemble yöntemlerinin sonuçlarını profesyonel Excel raporuna dönüştürür
    """

    def __init__(self, output_path: str = "PFAZ7_Ensemble_Results.xlsx"):
        """
        Args:
            output_path: Output Excel file path
        """
        self.output_path = Path(output_path)
        self.ensemble_results = []
        self.base_model_results = []

        logger.info(f"[OK] PFAZ7ExcelReporter initialized")
        logger.info(f"  Output: {self.output_path}")

    def add_ensemble_result(self,
                           ensemble_name: str,
                           method: str,
                           n_models: int,
                           model_ids: List[str],
                           r2: float,
                           rmse: float,
                           mae: float,
                           weights: Optional[List[float]] = None,
                           meta_model_type: Optional[str] = None):
        """Add ensemble result"""

        result = {
            'Ensemble_Name': ensemble_name,
            'Method': method,
            'N_Base_Models': n_models,
            'Base_Models': ', '.join(model_ids),
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Weights': weights if weights else '-',
            'Meta_Model': meta_model_type if meta_model_type else '-'
        }

        self.ensemble_results.append(result)
        logger.info(f"  [OK] Added: {ensemble_name} (R^2={r2:.4f})")

    def add_base_model_result(self,
                             model_id: str,
                             model_type: str,
                             r2: float,
                             rmse: float,
                             mae: float):
        """Add base model result"""

        result = {
            'Model_ID': model_id,
            'Model_Type': model_type,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        }

        self.base_model_results.append(result)

    def generate_report(self):
        """Generate comprehensive Excel report"""

        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATING PFAZ7 EXCEL REPORT")
        logger.info(f"{'='*60}")

        # Create Excel writer
        with pd.ExcelWriter(self.output_path, engine='openpyxl' if OPENPYXL_AVAILABLE else 'xlsxwriter') as writer:

            # Sheet 1: Summary
            self._create_summary_sheet(writer)

            # Sheet 2: All Ensemble Results
            self._create_ensemble_results_sheet(writer)

            # Sheet 3: Base Model Results
            self._create_base_model_results_sheet(writer)

            # Sheet 4: Method Comparison
            self._create_method_comparison_sheet(writer)

            # Sheet 5: Best Performers
            self._create_best_performers_sheet(writer)

            # Sheet 6: Weights Analysis (if available)
            self._create_weights_analysis_sheet(writer)

            # Sheet 7: Statistical Summary
            self._create_statistical_summary_sheet(writer)

            # Sheet 8: Recommendations
            self._create_recommendations_sheet(writer)

        logger.info(f"\n[SUCCESS] Excel report generated: {self.output_path}")
        logger.info(f"   Sheets: 8")
        logger.info(f"   Ensembles: {len(self.ensemble_results)}")
        logger.info(f"   Base Models: {len(self.base_model_results)}")

    def _create_summary_sheet(self, writer):
        """Sheet 1: Summary"""

        summary_data = {
            'Metric': [
                'Project Name',
                'PFAZ Phase',
                'Report Date',
                'Total Ensembles',
                'Total Base Models',
                'Best R²',
                'Best RMSE',
                'Best MAE',
                'Best Ensemble'
            ],
            'Value': [
                'Nuclear Physics AI - Ensemble Learning',
                'PFAZ 7: Ensemble & Meta-Learning',
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                len(self.ensemble_results),
                len(self.base_model_results),
                f"{max([r['R2'] for r in self.ensemble_results]):.4f}" if self.ensemble_results else 'N/A',
                f"{min([r['RMSE'] for r in self.ensemble_results]):.4f}" if self.ensemble_results else 'N/A',
                f"{min([r['MAE'] for r in self.ensemble_results]):.4f}" if self.ensemble_results else 'N/A',
                max(self.ensemble_results, key=lambda x: x['R2'])['Ensemble_Name'] if self.ensemble_results else 'N/A'
            ]
        }

        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Summary', index=False)

    def _create_ensemble_results_sheet(self, writer):
        """Sheet 2: All Ensemble Results"""

        df = pd.DataFrame(self.ensemble_results)
        df = df.sort_values('R2', ascending=False)
        df.to_excel(writer, sheet_name='Ensemble_Results', index=False)

    def _create_base_model_results_sheet(self, writer):
        """Sheet 3: Base Model Results"""

        if self.base_model_results:
            df = pd.DataFrame(self.base_model_results)
            df = df.sort_values('R2', ascending=False)
            df.to_excel(writer, sheet_name='Base_Model_Results', index=False)
        else:
            # Create empty sheet with header
            df = pd.DataFrame(columns=['Model_ID', 'Model_Type', 'R2', 'RMSE', 'MAE'])
            df.to_excel(writer, sheet_name='Base_Model_Results', index=False)

    def _create_method_comparison_sheet(self, writer):
        """Sheet 4: Method Comparison"""

        # Group by method
        methods = {}
        for result in self.ensemble_results:
            method = result['Method']
            if method not in methods:
                methods[method] = []
            methods[method].append(result)

        comparison_data = []
        for method, results in methods.items():
            r2_scores = [r['R2'] for r in results]
            rmse_scores = [r['RMSE'] for r in results]
            mae_scores = [r['MAE'] for r in results]

            comparison_data.append({
                'Method': method,
                'Count': len(results),
                'Best_R2': max(r2_scores),
                'Avg_R2': np.mean(r2_scores),
                'Std_R2': np.std(r2_scores),
                'Best_RMSE': min(rmse_scores),
                'Avg_RMSE': np.mean(rmse_scores),
                'Best_MAE': min(mae_scores),
                'Avg_MAE': np.mean(mae_scores)
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Best_R2', ascending=False)
        df.to_excel(writer, sheet_name='Method_Comparison', index=False)

    def _create_best_performers_sheet(self, writer):
        """Sheet 5: Best Performers"""

        # Top 5 by R²
        top_r2 = sorted(self.ensemble_results, key=lambda x: x['R2'], reverse=True)[:5]

        # Top 5 by RMSE (lowest)
        top_rmse = sorted(self.ensemble_results, key=lambda x: x['RMSE'])[:5]

        # Top 5 by MAE (lowest)
        top_mae = sorted(self.ensemble_results, key=lambda x: x['MAE'])[:5]

        best_data = {
            'Rank': list(range(1, 6)),
            'Top_R2_Ensemble': [r['Ensemble_Name'] for r in top_r2],
            'R2_Score': [r['R2'] for r in top_r2],
            'Top_RMSE_Ensemble': [r['Ensemble_Name'] for r in top_rmse],
            'RMSE_Score': [r['RMSE'] for r in top_rmse],
            'Top_MAE_Ensemble': [r['Ensemble_Name'] for r in top_mae],
            'MAE_Score': [r['MAE'] for r in top_mae]
        }

        df = pd.DataFrame(best_data)
        df.to_excel(writer, sheet_name='Best_Performers', index=False)

    def _create_weights_analysis_sheet(self, writer):
        """Sheet 6: Weights Analysis"""

        weighted_ensembles = [r for r in self.ensemble_results if r['Weights'] != '-']

        if weighted_ensembles:
            weights_data = []
            for result in weighted_ensembles:
                weights_data.append({
                    'Ensemble': result['Ensemble_Name'],
                    'Method': result['Method'],
                    'Weights': str(result['Weights']) if isinstance(result['Weights'], list) else result['Weights'],
                    'R2': result['R2']
                })

            df = pd.DataFrame(weights_data)
            df.to_excel(writer, sheet_name='Weights_Analysis', index=False)
        else:
            # Create empty sheet
            df = pd.DataFrame(columns=['Ensemble', 'Method', 'Weights', 'R2'])
            df.to_excel(writer, sheet_name='Weights_Analysis', index=False)

    def _create_statistical_summary_sheet(self, writer):
        """Sheet 7: Statistical Summary"""

        r2_scores = [r['R2'] for r in self.ensemble_results]
        rmse_scores = [r['RMSE'] for r in self.ensemble_results]
        mae_scores = [r['MAE'] for r in self.ensemble_results]

        stats_data = {
            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
            'R2': [
                len(r2_scores),
                np.mean(r2_scores),
                np.median(r2_scores),
                np.std(r2_scores),
                np.min(r2_scores),
                np.max(r2_scores),
                np.max(r2_scores) - np.min(r2_scores)
            ],
            'RMSE': [
                len(rmse_scores),
                np.mean(rmse_scores),
                np.median(rmse_scores),
                np.std(rmse_scores),
                np.min(rmse_scores),
                np.max(rmse_scores),
                np.max(rmse_scores) - np.min(rmse_scores)
            ],
            'MAE': [
                len(mae_scores),
                np.mean(mae_scores),
                np.median(mae_scores),
                np.std(mae_scores),
                np.min(mae_scores),
                np.max(mae_scores),
                np.max(mae_scores) - np.min(mae_scores)
            ]
        }

        df = pd.DataFrame(stats_data)
        df.to_excel(writer, sheet_name='Statistical_Summary', index=False)

    def _create_recommendations_sheet(self, writer):
        """Sheet 8: Recommendations"""

        best_ensemble = max(self.ensemble_results, key=lambda x: x['R2'])

        recommendations = {
            'Category': [
                'Best Overall Ensemble',
                'Method',
                'Performance (R²)',
                'Recommendation',
                'Use Case',
                'Advantages',
                'Deployment'
            ],
            'Details': [
                best_ensemble['Ensemble_Name'],
                best_ensemble['Method'],
                f"R²={best_ensemble['R2']:.4f}, RMSE={best_ensemble['RMSE']:.4f}",
                f"Use {best_ensemble['Ensemble_Name']} for production predictions",
                'High-accuracy nuclear physics predictions',
                'Combines strengths of multiple models, reduces variance',
                'Deploy via REST API or batch processing'
            ]
        }

        df = pd.DataFrame(recommendations)
        df.to_excel(writer, sheet_name='Recommendations', index=False)


def create_sample_report():
    """Create sample PFAZ7 report with mock data"""

    logger.info("="*60)
    logger.info("CREATING SAMPLE PFAZ7 EXCEL REPORT")
    logger.info("="*60)

    reporter = PFAZ7ExcelReporter("PFAZ7_Ensemble_Results.xlsx")

    # Add base models (mock data)
    base_models = [
        ('Ridge', 'Ridge', 0.92, 0.18, 0.12),
        ('Lasso', 'Lasso', 0.91, 0.19, 0.13),
        ('ElasticNet', 'ElasticNet', 0.93, 0.17, 0.11),
        ('RF', 'RandomForest', 0.94, 0.15, 0.10),
        ('GBM', 'GradientBoosting', 0.95, 0.14, 0.09),
        ('XGBoost', 'XGBoost', 0.96, 0.13, 0.08)
    ]

    for model_id, model_type, r2, rmse, mae in base_models:
        reporter.add_base_model_result(model_id, model_type, r2, rmse, mae)

    # Add ensemble results (mock data)
    ensembles = [
        ('SimpleVoting', 'simple_voting', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.95, 0.14, 0.09, None, None),

        ('WeightedVoting_R2', 'weighted_voting', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.96, 0.13, 0.08, [0.10, 0.12, 0.15, 0.18, 0.20, 0.25], None),

        ('WeightedVoting_RMSE', 'weighted_voting', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.96, 0.12, 0.08, [0.11, 0.13, 0.14, 0.19, 0.21, 0.22], None),

        ('WeightedVoting_InvError', 'weighted_voting', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.95, 0.13, 0.09, [0.12, 0.11, 0.16, 0.18, 0.20, 0.23], None),

        ('Stacking_Ridge', 'stacking', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.97, 0.11, 0.07, None, 'ridge'),

        ('Stacking_Lasso', 'stacking', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.97, 0.11, 0.07, None, 'lasso'),

        ('Stacking_RF', 'stacking', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.98, 0.10, 0.06, None, 'rf'),

        ('Stacking_GBM', 'stacking', 6, ['Ridge', 'Lasso', 'ElasticNet', 'RF', 'GBM', 'XGBoost'],
         0.98, 0.09, 0.06, None, 'gbm')
    ]

    for name, method, n_models, model_ids, r2, rmse, mae, weights, meta_model in ensembles:
        reporter.add_ensemble_result(name, method, n_models, model_ids, r2, rmse, mae, weights, meta_model)

    # Generate report
    reporter.generate_report()

    logger.info("\n[SUCCESS] Sample report created successfully!")
    return reporter


if __name__ == "__main__":
    create_sample_report()
