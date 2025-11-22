"""
Control Group Generator & Nucleus Prediction Analysis
Modellerin Hangi Çekirdekleri Tahmin Edebildiği/Edemediği Analizi

ÖNEMLİ ÖZELLIKLER:
1. Her çekirdek için tüm modellerin tahmin accuracy'si
2. Hangi çekirdekler genelde zor (poor prediction)
3. Hangi çekirdekler genelde kolay (good prediction)
4. Model bazlı başarı analizi
5. Çekirdek özellikleri ile tahmin başarısı korelasyonu
6. Kapsamlı Excel raporu

ÇIKTILAR:
- nucleus_prediction_analysis.xlsx (multi-sheet)
- poorly_predicted_nuclei.csv
- well_predicted_nuclei.csv
- model_nucleus_heatmap.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONTROL GROUP GENERATOR
# ============================================================================

class ControlGroupGenerator:
    """
    Nucleus-level prediction analysis
    
    Analyzes:
    - Which nuclei are consistently mispredicted by all models
    - Which nuclei are easy to predict
    - Model-specific weak spots
    - Correlation between nucleus properties and prediction accuracy
    """
    
    def __init__(self, output_dir='control_group_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.nucleus_predictions = defaultdict(lambda: defaultdict(list))
        # nucleus_predictions[nucleus_name][model_name] = [errors...]
        
        self.nucleus_properties = {}
        # nucleus_properties[nucleus_name] = {A, Z, N, ...}
        
        logger.info("Control Group Generator initialized")
    
    def record_prediction(self, nucleus_name, model_name, y_true, y_pred, nucleus_properties=None):
        """
        Record single nucleus prediction
        
        Args:
            nucleus_name: e.g., 'Li-6', 'C-12'
            model_name: e.g., 'RandomForest', 'XGBoost'
            y_true: True value
            y_pred: Predicted value
            nucleus_properties: Dict with A, Z, N, etc.
        """
        
        # Calculate error
        error = abs(y_pred - y_true)
        relative_error = error / (abs(y_true) + 1e-10)
        
        # Store
        self.nucleus_predictions[nucleus_name][model_name].append({
            'y_true': y_true,
            'y_pred': y_pred,
            'error': error,
            'relative_error': relative_error
        })
        
        # Store properties (first time only)
        if nucleus_properties and nucleus_name not in self.nucleus_properties:
            self.nucleus_properties[nucleus_name] = nucleus_properties
    
    def record_batch_predictions(self, nucleus_list, model_name, y_true_array, y_pred_array, properties_df=None):
        """
        Record batch of predictions
        
        Args:
            nucleus_list: List of nucleus names
            model_name: Model name
            y_true_array: Array of true values
            y_pred_array: Array of predictions
            properties_df: DataFrame with nucleus properties
        """
        
        for i, nucleus in enumerate(nucleus_list):
            y_true = y_true_array[i] if len(y_true_array.shape) == 1 else y_true_array[i, 0]
            y_pred = y_pred_array[i] if len(y_pred_array.shape) == 1 else y_pred_array[i, 0]
            
            # Get properties if available
            props = None
            if properties_df is not None and nucleus in properties_df.index:
                props = properties_df.loc[nucleus].to_dict()
            
            self.record_prediction(nucleus, model_name, y_true, y_pred, props)
    
    def analyze_all(self):
        """
        Comprehensive analysis of all predictions
        
        Returns:
            dict: Analysis results
        """
        
        logger.info("\n" + "="*80)
        logger.info("CONTROL GROUP ANALYSIS")
        logger.info("="*80)
        
        if not self.nucleus_predictions:
            logger.warning("No predictions recorded!")
            return {}
        
        analysis = {}
        
        # 1. Per-nucleus statistics
        logger.info("\n-> Computing per-nucleus statistics...")
        analysis['nucleus_stats'] = self._compute_nucleus_statistics()
        
        # 2. Per-model statistics
        logger.info("\n-> Computing per-model statistics...")
        analysis['model_stats'] = self._compute_model_statistics()
        
        # 3. Identify problem nuclei
        logger.info("\n-> Identifying problem nuclei...")
        analysis['problem_nuclei'] = self._identify_problem_nuclei()
        
        # 4. Identify easy nuclei
        logger.info("\n-> Identifying easy nuclei...")
        analysis['easy_nuclei'] = self._identify_easy_nuclei()
        
        # 5. Property correlation
        logger.info("\n-> Analyzing property correlations...")
        analysis['property_correlation'] = self._analyze_property_correlation()
        
        # 6. Generate reports
        logger.info("\n-> Generating reports...")
        self._generate_comprehensive_report(analysis)
        
        logger.info("\n" + "="*80)
        logger.info("[OK] CONTROL GROUP ANALYSIS COMPLETE")
        logger.info("="*80)
        
        return analysis
    
    def _compute_nucleus_statistics(self):
        """Compute statistics per nucleus across all models"""
        
        nucleus_stats = []
        
        for nucleus, model_results in self.nucleus_predictions.items():
            # Aggregate across all models
            all_errors = []
            all_relative_errors = []
            model_count = 0
            
            for model, predictions in model_results.items():
                errors = [p['error'] for p in predictions]
                rel_errors = [p['relative_error'] for p in predictions]
                
                all_errors.extend(errors)
                all_relative_errors.extend(rel_errors)
                model_count += 1
            
            # Statistics
            stats = {
                'nucleus': nucleus,
                'n_models': model_count,
                'n_predictions': len(all_errors),
                'mean_error': np.mean(all_errors),
                'std_error': np.std(all_errors),
                'median_error': np.median(all_errors),
                'max_error': np.max(all_errors),
                'mean_relative_error': np.mean(all_relative_errors),
                'prediction_difficulty': self._categorize_difficulty(np.mean(all_relative_errors))
            }
            
            # Add properties if available
            if nucleus in self.nucleus_properties:
                props = self.nucleus_properties[nucleus]
                stats.update({
                    'A': props.get('A'),
                    'Z': props.get('Z'),
                    'N': props.get('N'),
                    'is_magic': props.get('is_magic', False),
                    'is_doubly_magic': props.get('is_doubly_magic', False)
                })
            
            nucleus_stats.append(stats)
        
        return pd.DataFrame(nucleus_stats)
    
    def _categorize_difficulty(self, relative_error):
        """Categorize prediction difficulty"""
        if relative_error < 0.05:
            return 'Easy'
        elif relative_error < 0.15:
            return 'Medium'
        elif relative_error < 0.30:
            return 'Hard'
        else:
            return 'Very Hard'
    
    def _compute_model_statistics(self):
        """Compute statistics per model across all nuclei"""
        
        model_stats = defaultdict(lambda: {
            'nuclei_predicted': set(),
            'errors': [],
            'relative_errors': []
        })
        
        for nucleus, model_results in self.nucleus_predictions.items():
            for model, predictions in model_results.items():
                model_stats[model]['nuclei_predicted'].add(nucleus)
                
                for pred in predictions:
                    model_stats[model]['errors'].append(pred['error'])
                    model_stats[model]['relative_errors'].append(pred['relative_error'])
        
        # Convert to DataFrame
        model_summary = []
        for model, stats in model_stats.items():
            model_summary.append({
                'model': model,
                'n_nuclei': len(stats['nuclei_predicted']),
                'mean_error': np.mean(stats['errors']),
                'std_error': np.std(stats['errors']),
                'median_error': np.median(stats['errors']),
                'mean_relative_error': np.mean(stats['relative_errors']),
                'q25_error': np.percentile(stats['errors'], 25),
                'q75_error': np.percentile(stats['errors'], 75)
            })
        
        return pd.DataFrame(model_summary)
    
    def _identify_problem_nuclei(self, threshold=0.25):
        """
        Identify nuclei that are consistently hard to predict
        
        Args:
            threshold: Relative error threshold for "problem"
        """
        
        nucleus_stats = self._compute_nucleus_statistics()
        
        # Filter problem nuclei
        problem = nucleus_stats[nucleus_stats['mean_relative_error'] > threshold].copy()
        problem = problem.sort_values('mean_relative_error', ascending=False)
        
        logger.info(f"  Found {len(problem)} problem nuclei (rel_error > {threshold})")
        
        return problem
    
    def _identify_easy_nuclei(self, threshold=0.05):
        """
        Identify nuclei that are easy to predict
        
        Args:
            threshold: Relative error threshold for "easy"
        """
        
        nucleus_stats = self._compute_nucleus_statistics()
        
        # Filter easy nuclei
        easy = nucleus_stats[nucleus_stats['mean_relative_error'] < threshold].copy()
        easy = easy.sort_values('mean_relative_error')
        
        logger.info(f"  Found {len(easy)} easy nuclei (rel_error < {threshold})")
        
        return easy
    
    def _analyze_property_correlation(self):
        """Analyze correlation between nucleus properties and prediction error"""
        
        if not self.nucleus_properties:
            return {}
        
        nucleus_stats = self._compute_nucleus_statistics()
        
        # Check which properties are available
        available_props = []
        for prop in ['A', 'Z', 'N']:
            if prop in nucleus_stats.columns and nucleus_stats[prop].notna().any():
                available_props.append(prop)
        
        if not available_props:
            return {}
        
        correlations = {}
        
        for prop in available_props:
            # Remove NaN
            valid_data = nucleus_stats[[prop, 'mean_relative_error']].dropna()
            
            if len(valid_data) > 10:
                corr = valid_data[prop].corr(valid_data['mean_relative_error'])
                correlations[prop] = float(corr)
        
        # Check categorical properties
        if 'is_magic' in nucleus_stats.columns:
            magic_nuclei = nucleus_stats[nucleus_stats['is_magic'] == True]
            non_magic_nuclei = nucleus_stats[nucleus_stats['is_magic'] == False]
            
            if len(magic_nuclei) > 0 and len(non_magic_nuclei) > 0:
                correlations['magic_vs_non_magic'] = {
                    'magic_mean_error': float(magic_nuclei['mean_relative_error'].mean()),
                    'non_magic_mean_error': float(non_magic_nuclei['mean_relative_error'].mean())
                }
        
        return correlations
    
    def _generate_comprehensive_report(self, analysis):
        """Generate comprehensive Excel report"""
        
        excel_file = self.output_dir / 'nucleus_prediction_analysis.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Nucleus Statistics
            nucleus_stats = analysis['nucleus_stats']
            nucleus_stats.to_excel(writer, sheet_name='Nucleus_Statistics', index=False)
            
            # Sheet 2: Model Statistics
            model_stats = analysis['model_stats']
            model_stats.to_excel(writer, sheet_name='Model_Statistics', index=False)
            
            # Sheet 3: Problem Nuclei
            problem_nuclei = analysis['problem_nuclei']
            problem_nuclei.to_excel(writer, sheet_name='Problem_Nuclei', index=False)
            
            # Sheet 4: Easy Nuclei
            easy_nuclei = analysis['easy_nuclei']
            easy_nuclei.to_excel(writer, sheet_name='Easy_Nuclei', index=False)
            
            # Sheet 5: Summary
            summary_data = {
                'Metric': [
                    'Total Nuclei Analyzed',
                    'Total Models',
                    'Problem Nuclei (rel_error > 0.25)',
                    'Easy Nuclei (rel_error < 0.05)',
                    'Average Relative Error',
                    'Median Relative Error'
                ],
                'Value': [
                    len(nucleus_stats),
                    len(model_stats),
                    len(problem_nuclei),
                    len(easy_nuclei),
                    nucleus_stats['mean_relative_error'].mean(),
                    nucleus_stats['mean_relative_error'].median()
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"[OK] Excel report: {excel_file}")
        
        # Save CSVs
        problem_nuclei.to_csv(self.output_dir / 'poorly_predicted_nuclei.csv', index=False)
        easy_nuclei.to_csv(self.output_dir / 'well_predicted_nuclei.csv', index=False)
        
        # Generate heatmap
        self._generate_heatmap(analysis)
        
        # Generate text summary
        self._generate_text_summary(analysis)
    
    def _generate_heatmap(self, analysis):
        """Generate model × nucleus heatmap"""
        
        # Create pivot table
        data = []
        for nucleus, model_results in self.nucleus_predictions.items():
            for model, predictions in model_results.items():
                mean_error = np.mean([p['relative_error'] for p in predictions])
                data.append({
                    'Nucleus': nucleus,
                    'Model': model,
                    'Relative_Error': mean_error
                })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Pivot
        pivot = df.pivot_table(values='Relative_Error', index='Model', columns='Nucleus', aggfunc='mean')
        
        # Limit to top 50 nuclei by variance (most interesting)
        if pivot.shape[1] > 50:
            variance = pivot.var(axis=0)
            top_nuclei = variance.nlargest(50).index
            pivot = pivot[top_nuclei]
        
        # Plot
        fig, ax = plt.subplots(figsize=(max(20, pivot.shape[1] * 0.3), max(8, pivot.shape[0] * 0.5)))
        
        sns.heatmap(pivot, annot=False, cmap='RdYlGn_r', center=0.15,
                   cbar_kws={'label': 'Relative Error'}, ax=ax)
        
        ax.set_title('Model × Nucleus Prediction Error Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Nucleus', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'model_nucleus_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Heatmap: {self.output_dir / 'model_nucleus_heatmap.png'}")
    
    def _generate_text_summary(self, analysis):
        """Generate text summary"""
        
        txt_file = self.output_dir / 'analysis_summary.txt'
        
        with open(txt_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NUCLEUS PREDICTION ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            nucleus_stats = analysis['nucleus_stats']
            model_stats = analysis['model_stats']
            problem = analysis['problem_nuclei']
            easy = analysis['easy_nuclei']
            
            f.write(f"Total Nuclei: {len(nucleus_stats)}\n")
            f.write(f"Total Models: {len(model_stats)}\n")
            f.write(f"Problem Nuclei: {len(problem)}\n")
            f.write(f"Easy Nuclei: {len(easy)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("TOP 10 HARDEST NUCLEI TO PREDICT\n")
            f.write("="*80 + "\n\n")
            
            for i, row in problem.head(10).iterrows():
                f.write(f"{row['nucleus']}\n")
                f.write(f"  Mean Relative Error: {row['mean_relative_error']:.4f}\n")
                f.write(f"  Difficulty: {row['prediction_difficulty']}\n")
                if 'A' in row and pd.notna(row['A']):
                    f.write(f"  A={int(row['A'])}, Z={int(row['Z'])}, N={int(row['N'])}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("TOP 10 EASIEST NUCLEI TO PREDICT\n")
            f.write("="*80 + "\n\n")
            
            for i, row in easy.head(10).iterrows():
                f.write(f"{row['nucleus']}\n")
                f.write(f"  Mean Relative Error: {row['mean_relative_error']:.4f}\n")
                if 'A' in row and pd.notna(row['A']):
                    f.write(f"  A={int(row['A'])}, Z={int(row['Z'])}, N={int(row['N'])}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("MODEL PERFORMANCE RANKING\n")
            f.write("="*80 + "\n\n")
            
            sorted_models = model_stats.sort_values('mean_relative_error')
            for i, row in sorted_models.iterrows():
                f.write(f"{row['model']}\n")
                f.write(f"  Mean Relative Error: {row['mean_relative_error']:.4f}\n")
                f.write(f"  Nuclei Predicted: {row['n_nuclei']}\n\n")
            
            # Correlations
            if 'property_correlation' in analysis and analysis['property_correlation']:
                f.write("="*80 + "\n")
                f.write("PROPERTY CORRELATIONS WITH ERROR\n")
                f.write("="*80 + "\n\n")
                
                for prop, corr in analysis['property_correlation'].items():
                    if isinstance(corr, dict):
                        f.write(f"{prop}:\n")
                        for k, v in corr.items():
                            f.write(f"  {k}: {v:.4f}\n")
                    else:
                        f.write(f"{prop}: {corr:.4f}\n")
        
        logger.info(f"[OK] Text summary: {txt_file}")


# ============================================================================
# TEST
# ============================================================================

def test_control_group():
    """Test control group generator"""
    
    print("\n" + "="*80)
    print("CONTROL GROUP GENERATOR TEST")
    print("="*80)
    
    generator = ControlGroupGenerator(output_dir='test_control_group')
    
    # Simulate predictions
    nuclei = ['Li-6', 'Li-7', 'C-12', 'O-16', 'Fe-56', 'U-238']
    models = ['RF', 'XGBoost', 'DNN']
    
    np.random.seed(42)
    
    for nucleus in nuclei:
        # Simulate properties
        A = int(nucleus.split('-')[1])
        Z = {'Li': 3, 'C': 6, 'O': 8, 'Fe': 26, 'U': 92}[nucleus.split('-')[0]]
        N = A - Z
        
        props = {'A': A, 'Z': Z, 'N': N, 'is_magic': Z in [2, 8, 20, 28, 50, 82]}
        
        # Simulate difficulty (some nuclei harder than others)
        base_difficulty = np.random.uniform(0.05, 0.3)
        if nucleus in ['U-238', 'Fe-56']:
            base_difficulty *= 2  # Harder
        
        for model in models:
            # Simulate 5 predictions per nucleus
            for _ in range(5):
                y_true = np.random.uniform(0, 10)
                y_pred = y_true + np.random.normal(0, base_difficulty * y_true)
                
                generator.record_prediction(nucleus, model, y_true, y_pred, props)
    
    # Analyze
    analysis = generator.analyze_all()
    
    print("\n[OK] Test complete!")
    print(f"Output: test_control_group/")


if __name__ == "__main__":
    test_control_group()
    print("\n[SUCCESS] CONTROL GROUP GENERATOR COMPLETE")
    print("Location: dataset_generation/control_group_generator.py")