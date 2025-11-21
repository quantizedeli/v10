"""
Cross-Model Evaluator - TEZ İÇİN KRİTİK!
Ortak Model Performans Analizi

Bu modül, tüm modellerin (AI + ANFIS) ortak performansını analiz eder:
- Hangi çekirdekleri TÜM modeller iyi tahmin etti (Good Nuclei - 50 çekirdek)
- Hangi çekirdekleri TÜM modeller orta tahmin etti (Medium Nuclei - 50 çekirdek)
- Hangi çekirdekleri TÜM modeller kötü tahmin etti (Poor Nuclei - 50 çekirdek)

MM, QM, MM_QM, Beta_2 için ayrı ayrı analiz

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossModelEvaluator:
    """
    Tüm modellerin ortak performans analizi
    
    Usage:
        evaluator = CrossModelEvaluator()
        
        # Tüm model tahminlerini ekle
        evaluator.add_predictions('RandomForest', rf_df)
        evaluator.add_predictions('GradientBoosting', gbm_df)
        evaluator.add_predictions('ANFIS_GAU2MF', anfis_df)
        
        # Analiz yap
        results = evaluator.evaluate_common_performance(target='MM')
        
        # Excel raporu
        evaluator.save_cross_model_report('cross_model_report.xlsx')
    """
    
    def __init__(self, output_dir='reports/cross_model'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model predictions storage
        self.predictions = {}  # {model_name: df with predictions}
        
        # Classification thresholds
        self.thresholds = {
            'good': {'error': 0.1, 'r2': 0.90},    # Error < 0.1 and R² > 0.90
            'medium': {'error': 0.5, 'r2': 0.70},  # 0.1 ≤ Error < 0.5 and 0.70 ≤ R² < 0.90
            'poor': {'error': 0.5, 'r2': 0.70}     # Error ≥ 0.5 or R² < 0.70
        }
        
        # Results storage
        self.results = {}
        
        logger.info("Cross-Model Evaluator initialized")
    
    def add_predictions(self, 
                       model_name: str, 
                       predictions_df: pd.DataFrame,
                       target_col: str = 'target',
                       prediction_col: str = 'prediction',
                       nucleus_col: str = 'nucleus'):
        """
        Add model predictions
        
        Args:
            model_name: Model name (e.g., 'RandomForest', 'ANFIS_GAU2MF')
            predictions_df: DataFrame with columns [nucleus, target, prediction]
            target_col: Target column name
            prediction_col: Prediction column name
            nucleus_col: Nucleus identifier column
        """
        # Validate columns
        required = [nucleus_col, target_col, prediction_col]
        if not all(col in predictions_df.columns for col in required):
            raise ValueError(f"DataFrame must contain: {required}")
        
        # Store predictions
        self.predictions[model_name] = predictions_df[[
            nucleus_col, target_col, prediction_col
        ]].copy()
        
        # Calculate error
        self.predictions[model_name]['error'] = abs(
            self.predictions[model_name][target_col] - 
            self.predictions[model_name][prediction_col]
        )
        
        logger.info(f"✓ Added predictions for {model_name}: {len(predictions_df)} nuclei")
    
    def evaluate_common_performance(self, 
                                   target_name: str = 'MM',
                                   top_n: int = 50) -> Dict:
        """
        Evaluate common performance across all models
        
        Args:
            target_name: Target name (MM, QM, MM_QM, Beta_2)
            top_n: Number of nuclei to select for each category (default: 50)
        
        Returns:
            dict: Results with good/medium/poor nuclei lists
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"CROSS-MODEL PERFORMANCE EVALUATION: {target_name}")
        logger.info(f"{'='*80}")
        
        if len(self.predictions) == 0:
            raise ValueError("No predictions added! Use add_predictions() first.")
        
        # Get common nuclei across all models
        common_nuclei = self._get_common_nuclei()
        logger.info(f"Common nuclei across all models: {len(common_nuclei)}")
        
        # Calculate aggregate performance for each nucleus
        nucleus_performance = self._calculate_aggregate_performance(
            common_nuclei, target_name
        )
        
        # Classify nuclei
        classification = self._classify_nuclei(nucleus_performance)
        
        # Select top N for each category
        results = self._select_top_nuclei(classification, top_n)
        
        # Add model agreement analysis
        results['model_agreement'] = self._analyze_model_agreement(
            results, nucleus_performance
        )
        
        # Store results
        self.results[target_name] = results
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"RESULTS SUMMARY - {target_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Good Nuclei (all models accurate): {len(results['good_nuclei'])}")
        logger.info(f"Medium Nuclei (all models moderate): {len(results['medium_nuclei'])}")
        logger.info(f"Poor Nuclei (all models inaccurate): {len(results['poor_nuclei'])}")
        logger.info(f"Model Agreement Score: {results['model_agreement']['overall_agreement']:.3f}")
        
        return results
    
    def _get_common_nuclei(self) -> List[str]:
        """Get nuclei common to all models"""
        nucleus_sets = [
            set(df['nucleus'].unique()) 
            for df in self.predictions.values()
        ]
        
        common = set.intersection(*nucleus_sets)
        return sorted(list(common))
    
    def _calculate_aggregate_performance(self, 
                                        common_nuclei: List[str],
                                        target_name: str) -> pd.DataFrame:
        """
        Calculate aggregate performance metrics for each nucleus
        
        Returns:
            DataFrame with columns: [nucleus, mean_error, std_error, mean_r2, agreement_score]
        """
        logger.info(f"Calculating aggregate performance for {len(common_nuclei)} nuclei...")
        
        nucleus_data = []
        
        for nucleus in common_nuclei:
            nucleus_errors = []
            nucleus_r2_scores = []
            
            # Get predictions from all models for this nucleus
            for model_name, df in self.predictions.items():
                nucleus_pred = df[df['nucleus'] == nucleus]
                
                if len(nucleus_pred) == 0:
                    continue
                
                error = float(nucleus_pred['error'].iloc[0])
                nucleus_errors.append(error)
                
                # Calculate individual R² (approximate using error)
                # R² ≈ 1 - (error / target)²
                target_val = float(nucleus_pred['target'].iloc[0])
                if abs(target_val) > 1e-6:
                    r2_approx = 1 - (error / abs(target_val))**2
                else:
                    r2_approx = 0.0
                
                nucleus_r2_scores.append(r2_approx)
            
            if len(nucleus_errors) == 0:
                continue
            
            # Aggregate metrics
            mean_error = np.mean(nucleus_errors)
            std_error = np.std(nucleus_errors)
            mean_r2 = np.mean(nucleus_r2_scores)
            
            # Agreement score (lower std = higher agreement)
            agreement_score = 1.0 / (1.0 + std_error)
            
            nucleus_data.append({
                'nucleus': nucleus,
                'mean_error': mean_error,
                'std_error': std_error,
                'mean_r2': mean_r2,
                'agreement_score': agreement_score,
                'n_models': len(nucleus_errors)
            })
        
        df = pd.DataFrame(nucleus_data)
        logger.info(f"  ✓ Aggregate performance calculated for {len(df)} nuclei")
        
        return df
    
    def _classify_nuclei(self, nucleus_performance: pd.DataFrame) -> Dict:
        """
        Classify nuclei into good/medium/poor based on aggregate performance
        
        Returns:
            dict: {'good': df, 'medium': df, 'poor': df}
        """
        good_mask = (
            (nucleus_performance['mean_error'] < self.thresholds['good']['error']) &
            (nucleus_performance['mean_r2'] > self.thresholds['good']['r2'])
        )
        
        poor_mask = (
            (nucleus_performance['mean_error'] >= self.thresholds['poor']['error']) |
            (nucleus_performance['mean_r2'] < self.thresholds['poor']['r2'])
        )
        
        medium_mask = ~(good_mask | poor_mask)
        
        classification = {
            'good': nucleus_performance[good_mask].copy(),
            'medium': nucleus_performance[medium_mask].copy(),
            'poor': nucleus_performance[poor_mask].copy()
        }
        
        logger.info(f"  Classification:")
        logger.info(f"    Good: {len(classification['good'])} nuclei")
        logger.info(f"    Medium: {len(classification['medium'])} nuclei")
        logger.info(f"    Poor: {len(classification['poor'])} nuclei")
        
        return classification
    
    def _select_top_nuclei(self, classification: Dict, top_n: int) -> Dict:
        """
        Select top N nuclei from each category
        
        Good: lowest mean_error
        Medium: medium mean_error
        Poor: highest mean_error
        """
        results = {}
        
        # Good nuclei (best performers)
        good_df = classification['good'].nsmallest(top_n, 'mean_error')
        results['good_nuclei'] = good_df['nucleus'].tolist()
        results['good_stats'] = {
            'mean_error': float(good_df['mean_error'].mean()),
            'mean_r2': float(good_df['mean_r2'].mean()),
            'mean_agreement': float(good_df['agreement_score'].mean())
        }
        
        # Medium nuclei (middle performers)
        medium_df = classification['medium'].sort_values('mean_error')
        # Select middle 50
        start_idx = max(0, len(medium_df) // 2 - top_n // 2)
        end_idx = start_idx + top_n
        medium_df = medium_df.iloc[start_idx:end_idx]
        results['medium_nuclei'] = medium_df['nucleus'].tolist()
        results['medium_stats'] = {
            'mean_error': float(medium_df['mean_error'].mean()),
            'mean_r2': float(medium_df['mean_r2'].mean()),
            'mean_agreement': float(medium_df['agreement_score'].mean())
        }
        
        # Poor nuclei (worst performers)
        poor_df = classification['poor'].nlargest(top_n, 'mean_error')
        results['poor_nuclei'] = poor_df['nucleus'].tolist()
        results['poor_stats'] = {
            'mean_error': float(poor_df['mean_error'].mean()),
            'mean_r2': float(poor_df['mean_r2'].mean()),
            'mean_agreement': float(poor_df['agreement_score'].mean())
        }
        
        logger.info(f"  ✓ Selected top {top_n} nuclei for each category")
        
        return results
    
    def _analyze_model_agreement(self, 
                                 results: Dict,
                                 nucleus_performance: pd.DataFrame) -> Dict:
        """
        Analyze agreement between models
        
        Returns:
            dict: Agreement metrics
        """
        # Overall agreement score (average of all nuclei)
        overall_agreement = float(nucleus_performance['agreement_score'].mean())
        
        # Agreement by category
        good_nuclei_perf = nucleus_performance[
            nucleus_performance['nucleus'].isin(results['good_nuclei'])
        ]
        medium_nuclei_perf = nucleus_performance[
            nucleus_performance['nucleus'].isin(results['medium_nuclei'])
        ]
        poor_nuclei_perf = nucleus_performance[
            nucleus_performance['nucleus'].isin(results['poor_nuclei'])
        ]
        
        agreement = {
            'overall_agreement': overall_agreement,
            'good_nuclei_agreement': float(good_nuclei_perf['agreement_score'].mean()) if len(good_nuclei_perf) > 0 else 0,
            'medium_nuclei_agreement': float(medium_nuclei_perf['agreement_score'].mean()) if len(medium_nuclei_perf) > 0 else 0,
            'poor_nuclei_agreement': float(poor_nuclei_perf['agreement_score'].mean()) if len(poor_nuclei_perf) > 0 else 0
        }
        
        logger.info(f"  Model Agreement:")
        logger.info(f"    Overall: {agreement['overall_agreement']:.3f}")
        logger.info(f"    Good Nuclei: {agreement['good_nuclei_agreement']:.3f}")
        logger.info(f"    Medium Nuclei: {agreement['medium_nuclei_agreement']:.3f}")
        logger.info(f"    Poor Nuclei: {agreement['poor_nuclei_agreement']:.3f}")
        
        return agreement
    
    def save_cross_model_report(self, output_file: str = 'cross_model_report.xlsx'):
        """
        Save comprehensive cross-model report to Excel
        
        Args:
            output_file: Output Excel file name
        """
        if not self.results:
            logger.warning("No results to save! Run evaluate_common_performance() first.")
            return
        
        output_path = self.output_dir / output_file
        
        logger.info(f"\nSaving cross-model report to {output_path}...")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            self._write_summary_sheet(writer)
            
            # Sheets 2-4: Good/Medium/Poor nuclei for each target
            for target_name, results in self.results.items():
                self._write_nuclei_sheets(writer, target_name, results)
            
            # Sheet 5: Model Agreement Matrix
            self._write_agreement_matrix(writer)
            
            # Sheet 6: Detailed Statistics
            self._write_detailed_stats(writer)
        
        logger.info(f"✓ Cross-model report saved: {output_path}")
    
    def _write_summary_sheet(self, writer):
        """Write summary sheet"""
        summary_data = []
        
        for target_name, results in self.results.items():
            summary_data.append({
                'Target': target_name,
                'N_Models': len(self.predictions),
                'Good_Nuclei_Count': len(results['good_nuclei']),
                'Medium_Nuclei_Count': len(results['medium_nuclei']),
                'Poor_Nuclei_Count': len(results['poor_nuclei']),
                'Good_Mean_Error': results['good_stats']['mean_error'],
                'Medium_Mean_Error': results['medium_stats']['mean_error'],
                'Poor_Mean_Error': results['poor_stats']['mean_error'],
                'Overall_Agreement': results['model_agreement']['overall_agreement']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Summary', index=False)
    
    def _write_nuclei_sheets(self, writer, target_name, results):
        """Write sheets for good/medium/poor nuclei"""
        categories = ['good', 'medium', 'poor']
        
        for category in categories:
            nuclei_list = results[f'{category}_nuclei']
            stats = results[f'{category}_stats']
            
            # Get detailed predictions for these nuclei
            detailed_data = []
            
            for nucleus in nuclei_list:
                nucleus_info = {'Nucleus': nucleus}
                
                # Add target value (experimental)
                first_model = list(self.predictions.keys())[0]
                nucleus_data = self.predictions[first_model][
                    self.predictions[first_model]['nucleus'] == nucleus
                ]
                if len(nucleus_data) > 0:
                    nucleus_info['Experimental_Value'] = nucleus_data['target'].iloc[0]
                
                # Add predictions from each model
                for model_name, df in self.predictions.items():
                    nucleus_pred = df[df['nucleus'] == nucleus]
                    if len(nucleus_pred) > 0:
                        nucleus_info[f'{model_name}_Pred'] = nucleus_pred['prediction'].iloc[0]
                        nucleus_info[f'{model_name}_Error'] = nucleus_pred['error'].iloc[0]
                        
                        # Delta = Prediction - Experimental
                        nucleus_info[f'{model_name}_Delta'] = (
                            nucleus_pred['prediction'].iloc[0] - nucleus_pred['target'].iloc[0]
                        )
                
                detailed_data.append(nucleus_info)
            
            df_detailed = pd.DataFrame(detailed_data)
            
            # Add statistics row
            stats_row = {'Nucleus': 'MEAN', 'Experimental_Value': '---'}
            for col in df_detailed.columns:
                if col not in ['Nucleus', 'Experimental_Value']:
                    if df_detailed[col].dtype in [np.float64, np.int64]:
                        stats_row[col] = df_detailed[col].mean()
            
            df_detailed = pd.concat([df_detailed, pd.DataFrame([stats_row])], ignore_index=True)
            
            sheet_name = f'{target_name}_{category.capitalize()}'[:31]
            df_detailed.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _write_agreement_matrix(self, writer):
        """Write model agreement matrix"""
        # Create pairwise agreement matrix between models
        model_names = list(self.predictions.keys())
        n_models = len(model_names)
        
        agreement_matrix = np.zeros((n_models, n_models))
        
        # Calculate pairwise correlation of errors
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Get common nuclei
                    df1 = self.predictions[model1]
                    df2 = self.predictions[model2]
                    
                    merged = df1.merge(df2, on='nucleus', suffixes=('_1', '_2'))
                    
                    if len(merged) > 0:
                        corr = merged['error_1'].corr(merged['error_2'])
                        agreement_matrix[i, j] = corr
        
        df_matrix = pd.DataFrame(agreement_matrix, 
                                index=model_names, 
                                columns=model_names)
        
        df_matrix.to_excel(writer, sheet_name='Model_Agreement_Matrix')
    
    def _write_detailed_stats(self, writer):
        """Write detailed statistics"""
        stats_data = []
        
        for model_name, df in self.predictions.items():
            stats_data.append({
                'Model': model_name,
                'N_Predictions': len(df),
                'Mean_Error': df['error'].mean(),
                'Std_Error': df['error'].std(),
                'Min_Error': df['error'].min(),
                'Max_Error': df['error'].max(),
                'Median_Error': df['error'].median()
            })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Detailed_Statistics', index=False)
    
    def visualize_results(self, target_name: str = 'MM'):
        """
        Create visualizations for cross-model results
        
        Args:
            target_name: Target name
        """
        if target_name not in self.results:
            logger.warning(f"No results for {target_name}. Run evaluate_common_performance() first.")
            return
        
        results = self.results[target_name]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Category Distribution
        ax = axes[0, 0]
        categories = ['Good', 'Medium', 'Poor']
        counts = [
            len(results['good_nuclei']),
            len(results['medium_nuclei']),
            len(results['poor_nuclei'])
        ]
        colors = ['green', 'orange', 'red']
        ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_title(f'Nucleus Classification Distribution - {target_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Mean Error by Category
        ax = axes[0, 1]
        mean_errors = [
            results['good_stats']['mean_error'],
            results['medium_stats']['mean_error'],
            results['poor_stats']['mean_error']
        ]
        ax.bar(categories, mean_errors, color=colors, alpha=0.7)
        ax.set_title(f'Mean Error by Category - {target_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Agreement Score by Category
        ax = axes[1, 0]
        agreement_scores = [
            results['good_stats']['mean_agreement'],
            results['medium_stats']['mean_agreement'],
            results['poor_stats']['mean_agreement']
        ]
        ax.bar(categories, agreement_scores, color=colors, alpha=0.7)
        ax.set_title(f'Model Agreement by Category - {target_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Agreement Score')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Overall Statistics
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
        CROSS-MODEL ANALYSIS - {target_name}
        {'='*50}
        
        Number of Models: {len(self.predictions)}
        
        Good Nuclei:
          Count: {len(results['good_nuclei'])}
          Mean Error: {results['good_stats']['mean_error']:.4f}
          Mean R²: {results['good_stats']['mean_r2']:.4f}
        
        Medium Nuclei:
          Count: {len(results['medium_nuclei'])}
          Mean Error: {results['medium_stats']['mean_error']:.4f}
          Mean R²: {results['medium_stats']['mean_r2']:.4f}
        
        Poor Nuclei:
          Count: {len(results['poor_nuclei'])}
          Mean Error: {results['poor_stats']['mean_error']:.4f}
          Mean R²: {results['poor_stats']['mean_r2']:.4f}
        
        Overall Agreement: {results['model_agreement']['overall_agreement']:.3f}
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
               fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save
        viz_path = self.output_dir / f'cross_model_visualization_{target_name}.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Visualization saved: {viz_path}")


def main():
    """Test function"""
    logger.info("="*80)
    logger.info("CROSS-MODEL EVALUATOR TEST")
    logger.info("="*80)
    
    # Create dummy predictions for 3 models
    np.random.seed(42)
    n_nuclei = 200
    
    nuclei_names = [f"Nucleus_{i}" for i in range(n_nuclei)]
    true_values = np.random.uniform(-5, 5, n_nuclei)
    
    # Model 1: Good predictions
    pred1 = true_values + np.random.normal(0, 0.1, n_nuclei)
    df1 = pd.DataFrame({
        'nucleus': nuclei_names,
        'target': true_values,
        'prediction': pred1
    })
    
    # Model 2: Medium predictions
    pred2 = true_values + np.random.normal(0, 0.3, n_nuclei)
    df2 = pd.DataFrame({
        'nucleus': nuclei_names,
        'target': true_values,
        'prediction': pred2
    })
    
    # Model 3: Variable predictions
    pred3 = true_values + np.random.normal(0, 0.5, n_nuclei)
    df3 = pd.DataFrame({
        'nucleus': nuclei_names,
        'target': true_values,
        'prediction': pred3
    })
    
    # Create evaluator
    evaluator = CrossModelEvaluator('test_cross_model')
    
    # Add predictions
    evaluator.add_predictions('Model_1', df1)
    evaluator.add_predictions('Model_2', df2)
    evaluator.add_predictions('Model_3', df3)
    
    # Evaluate
    results = evaluator.evaluate_common_performance(target_name='MM', top_n=50)
    
    # Save report
    evaluator.save_cross_model_report('test_cross_model_report.xlsx')
    
    # Visualize
    evaluator.visualize_results('MM')
    
    print("\n✓ Test completed!")


if __name__ == "__main__":
    main()
