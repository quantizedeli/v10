"""
Model Comparison Dashboard
Interactive Model Performance Comparison

Features:
1. Side-by-side model comparison
2. Interactive plots (Plotly)
3. Statistical tests
4. Performance ranking
5. Best model recommendation
6. Export reports

ORTA ÖNCELİK #8 [SUCCESS]

Location: visualization/model_comparison_dashboard.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    plotly = None
    go = None
    make_subplots = None
    px = None
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - interactive plots disabled")

# Statistical tests
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL COMPARISON DASHBOARD
# ============================================================================

class ModelComparisonDashboard:
    """
    Interactive Model Comparison Dashboard
    
    Compares ALL models:
    - AI Models (RF, GBM, XGBoost, DNN, BNN, PINN, Ensemble)
    - ANFIS Models (Method 1 & 2)
    
    Features:
    - Side-by-side comparison
    - Statistical significance tests
    - Performance ranking
    - Interactive visualizations
    - Recommendations
    """
    
    def __init__(self, output_dir='dashboard'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_df = None
        self.comparison_report = {}
        
        logger.info("Model Comparison Dashboard initialized")
    
    def load_results(self, ai_results_file=None, anfis_results_file=None):
        """
        Load model results from files
        
        Args:
            ai_results_file: AI training summary (CSV/Excel)
            anfis_results_file: ANFIS training summary (CSV/Excel)
        """
        
        logger.info("Loading model results...")
        
        all_results = []
        
        # Load AI results
        if ai_results_file and Path(ai_results_file).exists():
            logger.info(f"  Loading AI results: {ai_results_file}")
            
            if str(ai_results_file).endswith('.xlsx'):
                ai_df = pd.read_excel(ai_results_file)
            else:
                ai_df = pd.read_csv(ai_results_file)
            
            ai_df['Model_Category'] = 'AI'
            all_results.append(ai_df)
        
        # Load ANFIS results
        if anfis_results_file and Path(anfis_results_file).exists():
            logger.info(f"  Loading ANFIS results: {anfis_results_file}")
            
            if str(anfis_results_file).endswith('.xlsx'):
                anfis_df = pd.read_excel(anfis_results_file)
            else:
                anfis_df = pd.read_csv(anfis_results_file)
            
            anfis_df['Model_Category'] = 'ANFIS'
            all_results.append(anfis_df)
        
        # Combine
        if all_results:
            self.results_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"[OK] Loaded {len(self.results_df)} results")
        else:
            logger.warning("[WARNING] No results loaded!")
            self.results_df = pd.DataFrame()
        
        return self.results_df
    
    def create_comparison_dashboard(self, target=None):
        """
        Create complete comparison dashboard
        
        Args:
            target: Specific target to analyze (None = all)
        """
        
        if self.results_df is None or len(self.results_df) == 0:
            logger.error("No results loaded!")
            return
        
        logger.info("\n" + "="*80)
        logger.info("CREATING MODEL COMPARISON DASHBOARD")
        logger.info("="*80)
        
        # Filter by target if specified
        if target:
            df = self.results_df[self.results_df['Target'] == target].copy()
            logger.info(f"Analyzing target: {target}")
        else:
            df = self.results_df.copy()
            logger.info("Analyzing all targets")
        
        # 1. Summary Statistics
        logger.info("\n-> Computing summary statistics...")
        self._compute_summary_stats(df)
        
        # 2. Statistical Tests
        logger.info("\n-> Running statistical tests...")
        self._run_statistical_tests(df)
        
        # 3. Performance Ranking
        logger.info("\n-> Creating performance ranking...")
        self._create_performance_ranking(df)
        
        # 4. Interactive Visualizations
        if PLOTLY_AVAILABLE:
            logger.info("\n-> Creating interactive visualizations...")
            self._create_interactive_plots(df)
        
        # 5. Static Visualizations
        logger.info("\n-> Creating static visualizations...")
        self._create_static_plots(df)
        
        # 6. Recommendations
        logger.info("\n-> Generating recommendations...")
        self._generate_recommendations(df)
        
        # 7. Export Report
        logger.info("\n-> Exporting report...")
        self._export_report()
        
        logger.info("\n" + "="*80)
        logger.info("[OK] DASHBOARD COMPLETED")
        logger.info("="*80)
    
    def _compute_summary_stats(self, df):
        """Compute summary statistics"""
        
        # Group by model
        summary = df.groupby('Model').agg({
            'R2_test': ['mean', 'std', 'min', 'max', 'count'],
            'RMSE_test': ['mean', 'std', 'min', 'max'],
            'MAE_test': ['mean', 'std', 'min', 'max'],
            'Training_Time': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Save
        summary.to_csv(self.output_dir / 'summary_statistics.csv', index=False)
        
        self.comparison_report['summary_statistics'] = summary
        
        logger.info(f"  [OK] Summary statistics: {len(summary)} models")
        
        # Print top 5
        print("\n" + "="*80)
        print("TOP 5 MODELS (by R² mean):")
        print("="*80)
        top5 = summary.nlargest(5, 'R2_test_mean')[['Model', 'R2_test_mean', 'R2_test_std', 'RMSE_test_mean']]
        print(top5.to_string(index=False))
    
    def _run_statistical_tests(self, df):
        """Run statistical significance tests"""
        
        # Get unique models
        models = df['Model'].unique()
        
        if len(models) < 2:
            logger.warning("  [WARNING] Need at least 2 models for statistical tests")
            return
        
        # Pairwise t-tests
        logger.info("  Running pairwise t-tests...")
        
        test_results = []
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                r2_1 = df[df['Model'] == model1]['R2_test'].dropna()
                r2_2 = df[df['Model'] == model2]['R2_test'].dropna()
                
                if len(r2_1) > 1 and len(r2_2) > 1:
                    t_stat, p_value = stats.ttest_ind(r2_1, r2_2)
                    
                    test_results.append({
                        'Model_1': model1,
                        'Model_2': model2,
                        'Mean_1': r2_1.mean(),
                        'Mean_2': r2_2.mean(),
                        'Diff': r2_1.mean() - r2_2.mean(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'Significant': 'Yes' if p_value < 0.05 else 'No'
                    })
        
        test_df = pd.DataFrame(test_results)
        test_df = test_df.sort_values('p_value')
        
        # Save
        test_df.to_csv(self.output_dir / 'statistical_tests.csv', index=False)
        
        self.comparison_report['statistical_tests'] = test_df
        
        logger.info(f"  [OK] Completed {len(test_df)} pairwise tests")
        
        # Print significant differences
        sig_tests = test_df[test_df['Significant'] == 'Yes']
        if len(sig_tests) > 0:
            print("\n" + "="*80)
            print("SIGNIFICANT DIFFERENCES (p < 0.05):")
            print("="*80)
            print(sig_tests[['Model_1', 'Model_2', 'Diff', 'p_value']].head(10).to_string(index=False))
    
    def _create_performance_ranking(self, df):
        """Create comprehensive performance ranking"""
        
        # Compute composite score
        # Normalize metrics to [0, 1]
        df_rank = df.groupby('Model').agg({
            'R2_test': 'mean',
            'RMSE_test': 'mean',
            'MAE_test': 'mean',
            'Training_Time': 'mean'
        }).reset_index()
        
        # Normalize R² (higher is better)
        df_rank['R2_norm'] = df_rank['R2_test']
        
        # Normalize RMSE (lower is better)
        df_rank['RMSE_norm'] = 1 - (df_rank['RMSE_test'] - df_rank['RMSE_test'].min()) / (df_rank['RMSE_test'].max() - df_rank['RMSE_test'].min() + 1e-10)
        
        # Normalize MAE (lower is better)
        df_rank['MAE_norm'] = 1 - (df_rank['MAE_test'] - df_rank['MAE_test'].min()) / (df_rank['MAE_test'].max() - df_rank['MAE_test'].min() + 1e-10)
        
        # Normalize Time (lower is better, but less important)
        df_rank['Time_norm'] = 1 - (df_rank['Training_Time'] - df_rank['Training_Time'].min()) / (df_rank['Training_Time'].max() - df_rank['Training_Time'].min() + 1e-10)
        
        # Composite score (weighted)
        df_rank['Composite_Score'] = (
            df_rank['R2_norm'] * 0.5 +
            df_rank['RMSE_norm'] * 0.25 +
            df_rank['MAE_norm'] * 0.15 +
            df_rank['Time_norm'] * 0.10
        )
        
        # Rank
        df_rank = df_rank.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        df_rank['Rank'] = range(1, len(df_rank) + 1)
        
        # Save
        df_rank.to_csv(self.output_dir / 'performance_ranking.csv', index=False)
        
        self.comparison_report['ranking'] = df_rank
        
        logger.info(f"  [OK] Performance ranking created")
        
        # Print ranking
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE RANKING:")
        print("="*80)
        print(df_rank[['Rank', 'Model', 'R2_test', 'RMSE_test', 'Composite_Score']].to_string(index=False))
    
    def _create_interactive_plots(self, df):
        """Create interactive Plotly visualizations"""
        
        viz_dir = self.output_dir / 'interactive'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Performance Comparison
        self._plot_interactive_performance(df, viz_dir)
        
        # 2. Scatter Matrix
        self._plot_interactive_scatter_matrix(df, viz_dir)
        
        # 3. Time vs Performance
        self._plot_interactive_time_vs_perf(df, viz_dir)
        
        # 4. Box Plots
        self._plot_interactive_boxplots(df, viz_dir)
        
        logger.info(f"  [OK] Interactive plots saved: {viz_dir}")
    
    def _plot_interactive_performance(self, df, viz_dir):
        """Interactive performance bar chart"""
        
        model_means = df.groupby('Model')['R2_test'].mean().sort_values(ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_means.index,
            y=model_means.values,
            marker=dict(
                color=model_means.values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="R²")
            ),
            text=[f'{v:.4f}' for v in model_means.values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Average R² Score',
            height=600,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.write_html(viz_dir / 'performance_comparison.html')
    
    def _plot_interactive_scatter_matrix(self, df, viz_dir):
        """Interactive scatter matrix"""
        
        model_summary = df.groupby('Model').agg({
            'R2_test': 'mean',
            'RMSE_test': 'mean',
            'MAE_test': 'mean',
            'Training_Time': 'mean'
        }).reset_index()
        
        fig = px.scatter_matrix(
            model_summary,
            dimensions=['R2_test', 'RMSE_test', 'MAE_test', 'Training_Time'],
            color='Model',
            title='Model Metrics Scatter Matrix',
            height=800
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.write_html(viz_dir / 'scatter_matrix.html')
    
    def _plot_interactive_time_vs_perf(self, df, viz_dir):
        """Interactive time vs performance scatter"""
        
        fig = go.Figure()
        
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            
            fig.add_trace(go.Scatter(
                x=model_data['Training_Time'],
                y=model_data['R2_test'],
                mode='markers',
                name=model,
                marker=dict(size=10, opacity=0.7),
                text=[f"Dataset: {d}" for d in model_data.get('Dataset', [''] * len(model_data))],
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x:.2f}s<br>R²: %{y:.4f}<br>%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Training Time vs Performance',
            xaxis_title='Training Time (seconds)',
            yaxis_title='R² Score',
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
        
        fig.write_html(viz_dir / 'time_vs_performance.html')
    
    def _plot_interactive_boxplots(self, df, viz_dir):
        """Interactive box plots"""
        
        fig = go.Figure()
        
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]['R2_test']
            
            fig.add_trace(go.Box(
                y=model_data,
                name=model,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title='R² Distribution by Model',
            yaxis_title='R² Score',
            height=600,
            template='plotly_white',
            showlegend=True
        )
        
        fig.write_html(viz_dir / 'r2_boxplots.html')
    
    def _create_static_plots(self, df):
        """Create static matplotlib plots"""
        
        viz_dir = self.output_dir / 'static'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Performance bars with error bars
        self._plot_performance_with_errors(df, viz_dir)
        
        # 2. Heatmap
        self._plot_performance_heatmap(df, viz_dir)
        
        # 3. Violin plots
        self._plot_violin_plots(df, viz_dir)
        
        logger.info(f"  [OK] Static plots saved: {viz_dir}")
    
    def _plot_performance_with_errors(self, df, viz_dir):
        """Bar plot with error bars"""
        
        model_stats = df.groupby('Model').agg({
            'R2_test': ['mean', 'std']
        })
        
        model_stats.columns = ['mean', 'std']
        model_stats = model_stats.sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(model_stats))
        
        bars = ax.bar(x_pos, model_stats['mean'], yerr=model_stats['std'],
                     capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Color bars by performance
        colors = plt.cm.RdYlGn(model_stats['mean'] / model_stats['mean'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_stats.index, rotation=45, ha='right')
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Model Performance with Standard Deviation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_with_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self, df, viz_dir):
        """Performance heatmap"""
        
        # Pivot: Model × Metric
        metrics = ['R2_test', 'RMSE_test', 'MAE_test']
        
        heatmap_data = df.groupby('Model')[metrics].mean()
        
        # Normalize each metric
        for col in metrics:
            if col == 'R2_test':
                heatmap_data[col] = heatmap_data[col]  # Higher is better
            else:
                heatmap_data[col] = 1 - (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min() + 1e-10)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_data) * 0.4)))
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Normalized Score'}, ax=ax, linewidths=1)
        
        ax.set_title('Model Performance Heatmap (Normalized)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_violin_plots(self, df, viz_dir):
        """Violin plots for R² distribution"""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sort models by median R²
        model_order = df.groupby('Model')['R2_test'].median().sort_values(ascending=False).index
        
        sns.violinplot(data=df, x='Model', y='R2_test', order=model_order, ax=ax, inner='box')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('R² Score Distribution by Model', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'r2_violin_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_recommendations(self, df):
        """Generate model recommendations"""
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'recommendations': []
        }
        
        # Best overall
        model_means = df.groupby('Model')['R2_test'].mean().sort_values(ascending=False)
        best_model = model_means.index[0]
        best_r2 = model_means.values[0]
        
        recommendations['summary']['best_model'] = best_model
        recommendations['summary']['best_r2'] = float(best_r2)
        
        recommendations['recommendations'].append({
            'type': 'best_overall',
            'model': best_model,
            'reason': f'Highest average R² score: {best_r2:.4f}'
        })
        
        # Fastest model
        fastest = df.groupby('Model')['Training_Time'].mean().sort_values().index[0]
        recommendations['recommendations'].append({
            'type': 'fastest',
            'model': fastest,
            'reason': 'Shortest average training time'
        })
        
        # Best accuracy-speed tradeoff
        if 'Composite_Score' in self.comparison_report.get('ranking', {}).columns:
            best_tradeoff = self.comparison_report['ranking'].iloc[0]['Model']
            recommendations['recommendations'].append({
                'type': 'best_tradeoff',
                'model': best_tradeoff,
                'reason': 'Best balance of accuracy and speed'
            })
        
        # Save
        with open(self.output_dir / 'recommendations.json', 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2)
        
        self.comparison_report['recommendations'] = recommendations
        
        logger.info("  [OK] Recommendations generated")
        
        # Print
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        for rec in recommendations['recommendations']:
            print(f"\n{rec['type'].upper()}:")
            print(f"  Model: {rec['model']}")
            print(f"  Reason: {rec['reason']}")
    
    def _export_report(self):
        """Export comprehensive report"""
        
        # JSON report
        report_file = self.output_dir / 'comparison_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_report, f, indent=2, default=str)
        
        logger.info(f"  [OK] Report exported: {report_file}")
        
        # Summary text
        summary_file = self.output_dir / 'comparison_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON DASHBOARD SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'summary_statistics' in self.comparison_report:
                f.write("Top 5 Models:\n")
                f.write("-"*80 + "\n")
                top5 = self.comparison_report['summary_statistics'].nlargest(5, 'R2_test_mean')
                f.write(top5[['Model', 'R2_test_mean', 'R2_test_std']].to_string(index=False))
                f.write("\n\n")
            
            if 'recommendations' in self.comparison_report:
                f.write("Recommendations:\n")
                f.write("-"*80 + "\n")
                for rec in self.comparison_report['recommendations']['recommendations']:
                    f.write(f"\n{rec['type'].upper()}: {rec['model']}\n")
                    f.write(f"  {rec['reason']}\n")
        
        logger.info(f"  [OK] Summary exported: {summary_file}")


# ============================================================================
# MAIN TEST
# ============================================================================

def test_dashboard():
    """Test model comparison dashboard"""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON DASHBOARD TEST")
    print("="*80)
    
    # Create dummy results
    np.random.seed(42)
    
    models = ['RF', 'GBM', 'XGBoost', 'DNN', 'BNN', 'ANFIS-M1', 'ANFIS-M2']
    n_datasets = 20
    
    results = []
    for model in models:
        for i in range(n_datasets):
            # Simulate performance
            base_r2 = np.random.beta(5, 2)  # Biased towards high values
            
            results.append({
                'Model': model,
                'Dataset': f'Dataset_{i}',
                'Target': 'MM',
                'R2_test': base_r2,
                'RMSE_test': np.random.uniform(0.05, 0.3),
                'MAE_test': np.random.uniform(0.03, 0.2),
                'Training_Time': np.random.uniform(10, 200)
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)
    
    # Create dashboard
    dashboard = ModelComparisonDashboard(output_dir='test_dashboard')
    dashboard.results_df = results_df
    dashboard.create_comparison_dashboard()
    
    print("\n[OK] Dashboard test completed!")
    print(f"Output: test_dashboard/")


if __name__ == "__main__":
    test_dashboard()
    print("\n[SUCCESS] Model Comparison Dashboard - ORTA ÖNCELİK #8 COMPLETE")
    print("Location: visualization/model_comparison_dashboard.py")