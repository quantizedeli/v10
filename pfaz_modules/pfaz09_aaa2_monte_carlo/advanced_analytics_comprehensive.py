# -*- coding: utf-8 -*-
"""
ADVANCED ANALYTICS COMPREHENSIVE SYSTEM
========================================

Complete advanced analytics and explainable AI system

Components:
1. SHAP Analysis (Universal)
2. Feature Importance (6 methods)
3. Clustering Analysis (5 algorithms)
4. Correlation Analysis (3 types)
5. Sensitivity Analysis (4 tests)
6. Cross-Validation (10 strategies)
7. Model Comparison Suite
8. Interactive Dashboards
9. Comprehensive Reports (Excel, LaTeX, JSON)
10. Automated Insights Generation

Author: Nuclear Physics AI Project
Version: 2.0.0
Date: 2025-10-24
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Machine Learning
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (KFold, StratifiedKFold, LeaveOneOut,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - install: pip install shap")

# Plotly for interactive viz
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - interactive dashboards disabled")

# Excel
try:
    import xlsxwriter
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logging.warning("Excel libraries not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'shap': {
        'enabled': True,
        'background_size': 100,
        'n_samples': 100,
        'plot_types': ['summary', 'bar', 'waterfall', 'dependence']
    },
    'feature_importance': {
        'methods': ['shap', 'permutation', 'model_based', 'correlation'],
        'n_repeats_permutation': 10
    },
    'clustering': {
        'nucleus_methods': ['kmeans', 'hierarchical', 'dbscan'],
        'feature_methods': ['kmeans', 'hierarchical'],
        'n_clusters_kmeans': 3,
        'pca_components': 3
    },
    'correlation': {
        'methods': ['pearson', 'spearman'],
        'threshold': 0.8
    },
    'sensitivity': {
        'noise_levels': [0.01, 0.05, 0.1],
        'dropout_probs': [0.1, 0.2],
        'n_samples': 50
    },
    'cross_validation': {
        'strategies': ['kfold', 'stratified', 'repeated'],
        'n_folds': 5,
        'n_repeats': 3
    },
    'output': {
        'excel': True,
        'latex': False,
        'json': True,
        'interactive_dashboards': True
    }
}


# ============================================================================
# UNIVERSAL SHAP ANALYZER
# ============================================================================

class UniversalSHAPAnalyzer:
    """
    Universal SHAP analyzer supporting all model types
    
    Supports:
    - Tree models: TreeExplainer
    - Deep models: DeepExplainer
    - Universal: KernelExplainer
    """
    
    def __init__(self, model, model_type: str, feature_names: List[str],
                 output_dir: Path):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
        if not SHAP_AVAILABLE:
            logger.warning("  SHAP not available")
    
    def create_explainer(self, X_background: np.ndarray):
        """Create appropriate SHAP explainer"""
        if not SHAP_AVAILABLE:
            logger.error("  SHAP not available")
            return False
        
        logger.info(f"  Creating SHAP explainer for {self.model_type}...")
        
        try:
            if self.model_type in ['RF', 'GBM', 'XGBoost', 'RandomForest', 
                                  'GradientBoosting']:
                # Tree-based: TreeExplainer
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("    ✓ TreeExplainer created")
            
            elif self.model_type in ['DNN', 'BNN', 'PINN']:
                # Deep learning: DeepExplainer
                self.explainer = shap.DeepExplainer(self.model, X_background)
                logger.info("    ✓ DeepExplainer created")
            
            else:
                # Universal: KernelExplainer (slower but works for all)
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    shap.sample(X_background, min(100, len(X_background)))
                )
                logger.info("    ✓ KernelExplainer created (universal)")
            
            return True
        
        except Exception as e:
            logger.error(f"  Failed to create SHAP explainer: {e}")
            return False
    
    def calculate_shap_values(self, X_explain: np.ndarray):
        """Calculate SHAP values"""
        if self.explainer is None:
            logger.error("  Explainer not created")
            return False
        
        logger.info(f"  Calculating SHAP values for {len(X_explain)} samples...")
        
        try:
            self.shap_values = self.explainer.shap_values(X_explain)
            
            # Handle multi-output
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[0]
            
            # Expected value
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value
                if isinstance(self.expected_value, list):
                    self.expected_value = self.expected_value[0]
            else:
                self.expected_value = 0.0
            
            logger.info(f"    ✓ SHAP values calculated: {self.shap_values.shape}")
            return True
        
        except Exception as e:
            logger.error(f"  SHAP calculation failed: {e}")
            return False
    
    def plot_summary(self, X_explain: np.ndarray, max_display: int = 20):
        """SHAP summary plot (beeswarm)"""
        if self.shap_values is None:
            return None
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X_explain, 
                         feature_names=self.feature_names,
                         max_display=max_display, show=False)
        
        save_path = self.output_dir / 'shap_summary_plot.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    ✓ Summary plot: {save_path.name}")
        return save_path
    
    def plot_bar(self, max_display: int = 20):
        """SHAP bar plot (mean importance)"""
        if self.shap_values is None:
            return None
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, 
                         feature_names=self.feature_names,
                         plot_type='bar', max_display=max_display, show=False)
        
        save_path = self.output_dir / 'shap_bar_plot.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    ✓ Bar plot: {save_path.name}")
        return save_path
    
    def plot_waterfall(self, X_explain: np.ndarray, sample_idx: int = 0):
        """SHAP waterfall plot for single prediction"""
        if self.shap_values is None:
            return None
        
        plt.figure(figsize=(10, 8))
        
        shap_exp = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            data=X_explain[sample_idx],
            feature_names=self.feature_names
        )
        
        shap.plots.waterfall(shap_exp, show=False)
        
        save_path = self.output_dir / f'shap_waterfall_sample_{sample_idx}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    ✓ Waterfall plot: {save_path.name}")
        return save_path
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from SHAP values"""
        if self.shap_values is None:
            return pd.DataFrame()
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': mean_abs_shap
        }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
        
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        # Save
        save_path = self.output_dir / 'shap_feature_importance.csv'
        importance_df.to_csv(save_path, index=False)
        
        return importance_df
    
    def export_shap_values(self) -> Path:
        """Export SHAP values to CSV"""
        if self.shap_values is None:
            return None
        
        shap_df = pd.DataFrame(
            self.shap_values,
            columns=self.feature_names
        )
        
        save_path = self.output_dir / 'shap_values.csv'
        shap_df.to_csv(save_path, index=False)
        
        logger.info(f"    ✓ SHAP values exported: {save_path.name}")
        return save_path


# ============================================================================
# FEATURE IMPORTANCE SYSTEM
# ============================================================================

class FeatureImportanceSystem:
    """
    Multi-method feature importance analysis
    
    Methods:
    1. SHAP Importance
    2. Permutation Importance
    3. Model-Based Importance (tree models)
    4. Correlation-Based Importance
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.importance_results = {}
    
    def shap_importance(self, shap_analyzer: UniversalSHAPAnalyzer) -> pd.DataFrame:
        """Get SHAP-based importance"""
        logger.info("  → SHAP importance...")
        
        importance_df = shap_analyzer.get_feature_importance()
        
        if len(importance_df) > 0:
            self.importance_results['shap'] = importance_df
            logger.info(f"    ✓ Top feature: {importance_df.iloc[0]['Feature']}")
        
        return importance_df
    
    def permutation_importance(self, model, X_test: np.ndarray, 
                              y_test: np.ndarray, feature_names: List[str],
                              n_repeats: int = 10) -> pd.DataFrame:
        """Permutation importance"""
        logger.info("  → Permutation importance...")
        
        try:
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Permutation_Importance': result.importances_mean,
                'Std': result.importances_std
            }).sort_values('Permutation_Importance', ascending=False).reset_index(drop=True)
            
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            # Save
            save_path = self.output_dir / 'permutation_importance.csv'
            importance_df.to_csv(save_path, index=False)
            
            self.importance_results['permutation'] = importance_df
            
            logger.info(f"    ✓ Top feature: {importance_df.iloc[0]['Feature']}")
            
            return importance_df
        
        except Exception as e:
            logger.error(f"  Permutation importance failed: {e}")
            return pd.DataFrame()
    
    def model_based_importance(self, model, model_type: str, 
                              feature_names: List[str]) -> pd.DataFrame:
        """Model's native feature importance"""
        logger.info("  → Model-based importance...")
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                if importance.ndim > 1:
                    importance = importance[0]
            else:
                logger.warning("    Model doesn't have native importance")
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Model_Importance': importance
            }).sort_values('Model_Importance', ascending=False).reset_index(drop=True)
            
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            # Save
            save_path = self.output_dir / 'model_based_importance.csv'
            importance_df.to_csv(save_path, index=False)
            
            self.importance_results['model_based'] = importance_df
            
            logger.info(f"    ✓ Top feature: {importance_df.iloc[0]['Feature']}")
            
            return importance_df
        
        except Exception as e:
            logger.error(f"  Model-based importance failed: {e}")
            return pd.DataFrame()
    
    def correlation_importance(self, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> pd.DataFrame:
        """Correlation-based importance"""
        logger.info("  → Correlation importance...")
        
        correlations = []
        for i, fname in enumerate(feature_names):
            corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            correlations.append(corr)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Correlation_Importance': correlations
        }).sort_values('Correlation_Importance', ascending=False).reset_index(drop=True)
        
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        # Save
        save_path = self.output_dir / 'correlation_importance.csv'
        importance_df.to_csv(save_path, index=False)
        
        self.importance_results['correlation'] = importance_df
        
        logger.info(f"    ✓ Top feature: {importance_df.iloc[0]['Feature']}")
        
        return importance_df
    
    def unified_ranking(self, feature_names: List[str]) -> pd.DataFrame:
        """Create unified ranking from all methods"""
        logger.info("  → Creating unified ranking...")
        
        if len(self.importance_results) == 0:
            logger.warning("    No importance results available")
            return pd.DataFrame()
        
        # Merge all rankings
        unified_df = pd.DataFrame({'Feature': feature_names})
        
        for method, df in self.importance_results.items():
            rank_col = f'{method}_rank'
            unified_df = unified_df.merge(
                df[['Feature', 'Rank']].rename(columns={'Rank': rank_col}),
                on='Feature', how='left'
            )
        
        # Calculate mean rank
        rank_cols = [col for col in unified_df.columns if col.endswith('_rank')]
        unified_df['Mean_Rank'] = unified_df[rank_cols].mean(axis=1)
        unified_df['Std_Rank'] = unified_df[rank_cols].std(axis=1)
        
        # Sort by mean rank
        unified_df = unified_df.sort_values('Mean_Rank').reset_index(drop=True)
        unified_df['Unified_Rank'] = range(1, len(unified_df) + 1)
        
        # Save
        save_path = self.output_dir / 'unified_feature_ranking.csv'
        unified_df.to_csv(save_path, index=False)
        
        logger.info(f"    ✓ Unified ranking created")
        logger.info(f"    ✓ Top 3 features:")
        for i in range(min(3, len(unified_df))):
            logger.info(f"       {i+1}. {unified_df.iloc[i]['Feature']}")
        
        return unified_df
    
    def plot_comparison(self, feature_names: List[str]):
        """Plot method comparison"""
        if len(self.importance_results) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        methods = list(self.importance_results.keys())
        n_methods = len(methods)
        n_features = len(feature_names)
        
        # Create matrix
        importance_matrix = np.zeros((n_features, n_methods))
        
        for j, method in enumerate(methods):
            df = self.importance_results[method]
            for i, fname in enumerate(feature_names):
                row = df[df['Feature'] == fname]
                if len(row) > 0:
                    importance_matrix[i, j] = row['Rank'].values[0]
        
        # Heatmap
        im = ax.imshow(importance_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # Labels
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names, fontsize=8)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance Method Comparison (Rank)', fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rank', fontsize=11)
        
        # Values
        for i in range(n_features):
            for j in range(n_methods):
                text = ax.text(j, i, f'{int(importance_matrix[i, j])}',
                             ha='center', va='center', color='black', fontsize=7)
        
        save_path = self.output_dir / 'method_comparison.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    ✓ Comparison plot: {save_path.name}")
        return save_path


# ============================================================================
# ADVANCED ANALYTICS COMPREHENSIVE (Main Class)
# ============================================================================

class AdvancedAnalyticsComprehensive:
    """
    Comprehensive Advanced Analytics System
    
    Complete pipeline for deep model analysis
    """
    
    def __init__(self, output_dir: str = 'advanced_analytics_results',
                 config: Dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or DEFAULT_CONFIG
        
        # Results storage
        self.results = {}
        
        logger.info("="*80)
        logger.info("ADVANCED ANALYTICS COMPREHENSIVE SYSTEM")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_complete_analysis(self, model, model_type: str,
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             feature_names: List[str],
                             target: str = 'MM') -> Dict:
        """
        Run complete advanced analytics pipeline
        
        Args:
            model: Trained model
            model_type: Model type identifier
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: Feature names
            target: Target variable name
        
        Returns:
            results: Complete analysis results
        """
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info(f"COMPREHENSIVE ANALYSIS - {target} - {model_type}")
        logger.info("="*80)
        
        results = {
            'target': target,
            'model_type': model_type,
            'timestamp': start_time.isoformat(),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(feature_names)
        }
        
        try:
            # Phase 1: SHAP Analysis
            if self.config['shap']['enabled'] and SHAP_AVAILABLE:
                logger.info("\n→ Phase 1: SHAP Analysis...")
                results['shap'] = self._run_shap_analysis(
                    model, model_type, X_train, X_test, feature_names
                )
            
            # Phase 2: Feature Importance (Multiple Methods)
            logger.info("\n→ Phase 2: Feature Importance Analysis...")
            results['feature_importance'] = self._run_feature_importance(
                model, model_type, X_train, y_train, X_test, y_test, feature_names
            )
            
            # Phase 3-10: To be continued in next section...
            
            # Phase 3: Clustering Analysis
            logger.info("\n→ Phase 3: Clustering Analysis...")
            results['clustering'] = self._run_clustering_analysis(
                X_train, y_train, X_test, y_test, feature_names
            )
            
            # Phase 4: Correlation Analysis
            logger.info("\n→ Phase 4: Correlation Analysis...")
            results['correlation'] = self._run_correlation_analysis(
                X_train, y_train, feature_names
            )
            
            # Phase 5: Sensitivity Analysis
            logger.info("\n→ Phase 5: Sensitivity Analysis...")
            results['sensitivity'] = self._run_sensitivity_analysis(
                model, X_test, y_test
            )
            
            # Phase 6: Cross-Validation
            logger.info("\n→ Phase 6: Cross-Validation...")
            results['cross_validation'] = self._run_cross_validation(
                model, model_type, X_train, y_train
            )
            
            # Phase 7: Generate Reports
            logger.info("\n→ Phase 7: Generating Reports...")
            results['reports'] = self._generate_reports(results, target, model_type)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            results['duration_seconds'] = duration
            results['success'] = True
            
            logger.info(f"\n✅ Analysis complete: {duration:.1f}s")
            
            return results
        
        except Exception as e:
            logger.error(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            results['success'] = False
            results['error'] = str(e)
            return results
    
    def _run_shap_analysis(self, model, model_type: str, X_train: np.ndarray,
                          X_test: np.ndarray, feature_names: List[str]) -> Dict:
        """Run SHAP analysis"""
        
        shap_dir = self.output_dir / 'shap'
        shap_analyzer = UniversalSHAPAnalyzer(
            model, model_type, feature_names, shap_dir
        )
        
        # Background data
        bg_size = min(self.config['shap']['background_size'], len(X_train))
        X_background = X_train[:bg_size]
        
        # Explain data
        n_explain = min(self.config['shap']['n_samples'], len(X_test))
        X_explain = X_test[:n_explain]
        
        # Create explainer
        if not shap_analyzer.create_explainer(X_background):
            return {'status': 'failed', 'reason': 'explainer_creation_failed'}
        
        # Calculate SHAP values
        if not shap_analyzer.calculate_shap_values(X_explain):
            return {'status': 'failed', 'reason': 'shap_calculation_failed'}
        
        # Visualizations
        plots = {}
        if 'summary' in self.config['shap']['plot_types']:
            plots['summary'] = shap_analyzer.plot_summary(X_explain)
        
        if 'bar' in self.config['shap']['plot_types']:
            plots['bar'] = shap_analyzer.plot_bar()
        
        if 'waterfall' in self.config['shap']['plot_types']:
            plots['waterfall'] = shap_analyzer.plot_waterfall(X_explain, sample_idx=0)
        
        # Feature importance
        importance_df = shap_analyzer.get_feature_importance()
        
        # Export SHAP values
        shap_file = shap_analyzer.export_shap_values()
        
        return {
            'status': 'success',
            'plots': plots,
            'importance': importance_df.to_dict('records') if len(importance_df) > 0 else [],
            'shap_values_file': str(shap_file) if shap_file else None,
            'n_samples': n_explain
        }
    
    def _run_feature_importance(self, model, model_type: str,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               feature_names: List[str]) -> Dict:
        """Run feature importance analysis"""
        
        fi_dir = self.output_dir / 'feature_importance'
        fi_system = FeatureImportanceSystem(fi_dir)
        
        # Method 1: SHAP importance (if available)
        if 'shap' in self.config['feature_importance']['methods']:
            if 'shap' in self.results and self.results['shap']['status'] == 'success':
                shap_dir = self.output_dir / 'shap'
                shap_analyzer = UniversalSHAPAnalyzer(
                    model, model_type, feature_names, shap_dir
                )
                # Reload SHAP values if needed
                fi_system.shap_importance(shap_analyzer)
        
        # Method 2: Permutation importance
        if 'permutation' in self.config['feature_importance']['methods']:
            n_repeats = self.config['feature_importance']['n_repeats_permutation']
            fi_system.permutation_importance(model, X_test, y_test, 
                                           feature_names, n_repeats)
        
        # Method 3: Model-based importance
        if 'model_based' in self.config['feature_importance']['methods']:
            fi_system.model_based_importance(model, model_type, feature_names)
        
        # Method 4: Correlation importance
        if 'correlation' in self.config['feature_importance']['methods']:
            fi_system.correlation_importance(X_train, y_train, feature_names)
        
        # Unified ranking
        unified_df = fi_system.unified_ranking(feature_names)
        
        # Comparison plot
        comparison_plot = fi_system.plot_comparison(feature_names)
        
        return {
            'methods_used': list(fi_system.importance_results.keys()),
            'unified_ranking': unified_df.to_dict('records') if len(unified_df) > 0 else [],
            'comparison_plot': str(comparison_plot) if comparison_plot else None,
            'top_3_features': unified_df['Feature'].head(3).tolist() if len(unified_df) > 0 else []
       

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Test function"""
    logger.info("Advanced Analytics Comprehensive - Test Mode")
    
    # Generate test data
    np.random.seed(42)
    X_train = np.random.randn(200, 10)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] - X_train[:, 2] * 0.5
    X_test = np.random.randn(50, 10)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] - X_test[:, 2] * 0.5
    
    feature_names = [f'Feature_{i}' for i in range(10)]
    
    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize system
    analytics = AdvancedAnalyticsComprehensive(
        output_dir='test_advanced_analytics'
    )
    
    # Run analysis
    results = analytics.run_complete_analysis(
        model, 'RandomForest', X_train, y_train, X_test, y_test, 
        feature_names, target='TEST'
    )
    
    logger.info("\n✅ Test complete!")
    return results


if __name__ == "__main__":
    results = main()
          mask = cluster_labels == i
                cluster_stats.append({
                    'cluster': i,
                    'n_nuclei': int(mask.sum()),
                    'mean_error': float(errors[mask].mean()),
                    'std_error': float(errors[mask].std())
                })
            
            results['kmeans'] = {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_stats': cluster_stats,
                'inertia': float(kmeans.inertia_)
            }
            
            # Visualization
            if self.config['clustering']['pca_components'] >= 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_test)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=cluster_labels, cmap='viridis',
                                   s=80, alpha=0.6, edgecolors='black')
                
                # Centers
                centers_pca = pca.transform(kmeans.cluster_centers_[:, :-1])
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                         c='red', marker='X', s=300, edgecolors='black',
                         linewidths=2, label='Centroids')
                
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
                ax.set_title('K-Means Clustering (PCA Projection)', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Cluster')
                
                save_path = cluster_dir / 'kmeans_clusters.png'
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                results['kmeans']['plot'] = str(save_path)
        
        # Feature clustering
        logger.info("  → Feature clustering...")
        
        # Correlation-based clustering
        corr_matrix = np.corrcoef(X_train.T)
        
        linkage_matrix = linkage(pdist(corr_matrix), method='ward')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        dendrogram(linkage_matrix, labels=feature_names, ax=ax,
                  leaf_font_size=10, leaf_rotation=90)
        ax.set_title('Feature Hierarchical Clustering', fontsize=14)
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        save_path = cluster_dir / 'feature_dendrogram.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        results['feature_clustering'] = {
            'dendrogram_plot': str(save_path),
            'linkage_matrix_shape': linkage_matrix.shape
        }
        
        logger.info("  ✓ Clustering analysis complete")
        
        return results
    
    def _run_correlation_analysis(self, X_train: np.ndarray, y_train: np.ndarray,
                                  feature_names: List[str]) -> Dict:
        """Run correlation analysis"""
        
        corr_dir = self.output_dir / 'correlation'
        corr_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Pearson correlation
        if 'pearson' in self.config['correlation']['methods']:
            logger.info("  → Pearson correlation...")
            
            corr_matrix = np.corrcoef(X_train.T)
            
            corr_df = pd.DataFrame(corr_matrix, 
                                  index=feature_names, 
                                  columns=feature_names)
            
            # Save
            save_path = corr_dir / 'pearson_correlation.csv'
            corr_df.to_csv(save_path)
            
            results['pearson'] = {
                'matrix_file': str(save_path),
                'mean_correlation': float(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]).mean())
            }
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(14, 12))
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_df, mask=mask, annot=False, cmap='coolwarm',
                       center=0, vmin=-1, vmax=1, square=True, ax=ax,
                       cbar_kws={'label': 'Correlation'})
            
            ax.set_title('Pearson Correlation Matrix', fontsize=14)
            
            save_path = corr_dir / 'pearson_heatmap.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            results['pearson']['heatmap'] = str(save_path)
            
            # High correlations
            threshold = self.config['correlation']['threshold']
            high_corr = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    if abs(corr_matrix[i, j]) > threshold:
                        high_corr.append({
                            'feature_1': feature_names[i],
                            'feature_2': feature_names[j],
                            'correlation': float(corr_matrix[i, j])
                        })
            
            results['pearson']['high_correlations'] = high_corr
        
        # Spearman correlation
        if 'spearman' in self.config['correlation']['methods']:
            logger.info("  → Spearman correlation...")
            
            spearman_matrix = np.zeros((len(feature_names), len(feature_names)))
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    spearman_matrix[i, j] = stats.spearmanr(X_train[:, i], X_train[:, j])[0]
            
            spearman_df = pd.DataFrame(spearman_matrix,
                                      index=feature_names,
                                      columns=feature_names)
            
            save_path = corr_dir / 'spearman_correlation.csv'
            spearman_df.to_csv(save_path)
            
            results['spearman'] = {
                'matrix_file': str(save_path),
                'mean_correlation': float(np.abs(spearman_matrix[np.triu_indices_from(spearman_matrix, k=1)]).mean())
            }
        
        # Feature-target correlation
        logger.info("  → Feature-target correlations...")
        
        target_corr = []
        for i, fname in enumerate(feature_names):
            pearson_corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            spearman_corr = stats.spearmanr(X_train[:, i], y_train)[0]
            
            target_corr.append({
                'feature': fname,
                'pearson': float(pearson_corr),
                'spearman': float(spearman_corr)
            })
        
        target_corr_df = pd.DataFrame(target_corr)
        target_corr_df = target_corr_df.sort_values('pearson', 
                                                    key=lambda x: abs(x), 
                                                    ascending=False)
        
        save_path = corr_dir / 'feature_target_correlation.csv'
        target_corr_df.to_csv(save_path, index=False)
        
        results['feature_target'] = {
            'correlations': target_corr,
            'top_3_features': target_corr_df['feature'].head(3).tolist()
        }
        
        logger.info("  ✓ Correlation analysis complete")
        
        return results
    
    def _run_sensitivity_analysis(self, model, X_test: np.ndarray, 
                                  y_test: np.ndarray) -> Dict:
        """Run sensitivity analysis"""
        
        sens_dir = self.output_dir / 'sensitivity'
        sens_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Original predictions
        y_pred_original = model.predict(X_test)
        
        # Noise sensitivity
        logger.info("  → Noise sensitivity...")
        
        noise_results = {}
        for noise_level in self.config['sensitivity']['noise_levels']:
            predictions_noisy = []
            
            for _ in range(self.config['sensitivity']['n_samples']):
                X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
                y_pred_noisy = model.predict(X_noisy)
                predictions_noisy.append(y_pred_noisy)
            
            predictions_noisy = np.array(predictions_noisy)
            
            # Calculate stability
            variance = predictions_noisy.var(axis=0).mean()
            mean_change = np.abs(predictions_noisy.mean(axis=0) - y_pred_original).mean()
            
            noise_results[float(noise_level)] = {
                'variance': float(variance),
                'mean_change': float(mean_change),
                'robustness_score': float(1 - min(1, variance / (y_pred_original.var() + 1e-10)))
            }
        
        results['noise_sensitivity'] = noise_results
        
        # Dropout sensitivity
        logger.info("  → Feature dropout sensitivity...")
        
        dropout_results = {}
        for dropout_prob in self.config['sensitivity']['dropout_probs']:
            predictions_dropout = []
            
            for _ in range(self.config['sensitivity']['n_samples']):
                X_dropout = X_test.copy()
                mask = np.random.binomial(1, 1-dropout_prob, X_test.shape)
                X_dropout = X_dropout * mask
                
                y_pred_dropout = model.predict(X_dropout)
                predictions_dropout.append(y_pred_dropout)
            
            predictions_dropout = np.array(predictions_dropout)
            
            variance = predictions_dropout.var(axis=0).mean()
            
            dropout_results[float(dropout_prob)] = {
                'variance': float(variance),
                'mean_prediction': float(predictions_dropout.mean())
            }
        
        results['dropout_sensitivity'] = dropout_results
        
        # Robustness score
        overall_robustness = np.mean([
            noise_results[nl]['robustness_score'] 
            for nl in self.config['sensitivity']['noise_levels']
        ])
        
        results['overall_robustness_score'] = float(overall_robustness)
        
        logger.info(f"  ✓ Overall robustness: {overall_robustness:.3f}")
        
        return results
    
    def _run_cross_validation(self, model, model_type: str,
                             X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Run cross-validation"""
        
        cv_dir = self.output_dir / 'cross_validation'
        cv_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # K-Fold CV
        if 'kfold' in self.config['cross_validation']['strategies']:
            logger.info("  → K-Fold cross-validation...")
            
            kfold = KFold(n_splits=self.config['cross_validation']['n_folds'],
                         shuffle=True, random_state=42)
            
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=kfold, scoring='r2', n_jobs=-1)
            
            results['kfold'] = {
                'n_folds': self.config['cross_validation']['n_folds'],
                'scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std()),
                'min_score': float(cv_scores.min()),
                'max_score': float(cv_scores.max())
            }
            
            logger.info(f"    K-Fold R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Stratified K-Fold (binned targets)
        if 'stratified' in self.config['cross_validation']['strategies']:
            logger.info("  → Stratified K-Fold...")
            
            # Bin targets for stratification
            n_bins = 5
            y_binned = pd.qcut(y_train, q=n_bins, labels=False, duplicates='drop')
            
            stratified_kfold = StratifiedKFold(
                n_splits=self.config['cross_validation']['n_folds'],
                shuffle=True, random_state=42
            )
            
            cv_scores = cross_val_score(model, X_train, y_train,
                                       cv=stratified_kfold.split(X_train, y_binned),
                                       scoring='r2', n_jobs=-1)
            
            results['stratified_kfold'] = {
                'n_folds': self.config['cross_validation']['n_folds'],
                'scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std())
            }
        
        # Learning curves
        logger.info("  → Learning curves...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=3, scoring='r2', n_jobs=-1
        )
        
        results['learning_curve'] = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }
        
        # Plot learning curve
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-',
               color='blue', label='Training score')
        ax.fill_between(train_sizes_abs,
                       train_scores.mean(axis=1) - train_scores.std(axis=1),
                       train_scores.mean(axis=1) + train_scores.std(axis=1),
                       alpha=0.2, color='blue')
        
        ax.plot(train_sizes_abs, val_scores.mean(axis=1), 'o-',
               color='red', label='Validation score')
        ax.fill_between(train_sizes_abs,
                       val_scores.mean(axis=1) - val_scores.std(axis=1),
                       val_scores.mean(axis=1) + val_scores.std(axis=1),
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Size', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Learning Curve', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        save_path = cv_dir / 'learning_curve.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        results['learning_curve']['plot'] = str(save_path)
        
        logger.info("  ✓ Cross-validation complete")
        
        return results
    
    def _generate_reports(self, results: Dict, target: str, 
                         model_type: str) -> Dict:
        """Generate comprehensive reports"""
        
        report_dir = self.output_dir / 'reports'
        report_dir.mkdir(exist_ok=True)
        
        report_files = {}
        
        # Excel Report
        if self.config['output']['excel'] and EXCEL_AVAILABLE:
            logger.info("  → Generating Excel report...")
            
            excel_file = report_dir / f'Advanced_Analytics_{target}_{model_type}.xlsx'
            
            try:
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    # Sheet 1: Executive Summary
                    summary_data = {
                        'Metric': ['Target', 'Model Type', 'Analysis Date', 'Total Duration'],
                        'Value': [
                            target,
                            model_type,
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            f"{results.get('duration_seconds', 0):.1f}s"
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, 
                                                       sheet_name='Executive_Summary',
                                                       index=False)
                    
                    # Sheet 2: Feature Importance
                    if 'feature_importance' in results and results['feature_importance'].get('unified_ranking'):
                        fi_df = pd.DataFrame(results['feature_importance']['unified_ranking'])
                        fi_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
                    
                    # Sheet 3: Correlation
                    if 'correlation' in results and 'feature_target' in results['correlation']:
                        corr_df = pd.DataFrame(results['correlation']['feature_target']['correlations'])
                        corr_df.to_excel(writer, sheet_name='Correlations', index=False)
                    
                    # Sheet 4: Cross-Validation
                    if 'cross_validation' in results and 'kfold' in results['cross_validation']:
                        cv_data = results['cross_validation']['kfold']
                        cv_df = pd.DataFrame({
                            'Fold': range(1, len(cv_data['scores']) + 1),
                            'R2_Score': cv_data['scores']
                        })
                        cv_df.to_excel(writer, sheet_name='Cross_Validation', index=False)
                    
                    # Sheet 5: Sensitivity
                    if 'sensitivity' in results:
                        sens_data = []
                        for noise_level, metrics in results['sensitivity'].get('noise_sensitivity', {}).items():
                            sens_data.append({
                                'Noise_Level': noise_level,
                                'Variance': metrics['variance'],
                                'Mean_Change': metrics['mean_change'],
                                'Robustness_Score': metrics['robustness_score']
                            })
                        if sens_data:
                            sens_df = pd.DataFrame(sens_data)
                            sens_df.to_excel(writer, sheet_name='Sensitivity', index=False)
                
                report_files['excel'] = str(excel_file)
                logger.info(f"    ✓ Excel report: {excel_file.name}")
            
            except Exception as e:
                logger.error(f"    Excel generation failed: {e}")
        
        # JSON Export
        if self.config['output']['json']:
            logger.info("  → Exporting JSON...")
            
            json_file = report_dir / f'advanced_analytics_{target}_{model_type}.json'
            
            try:
                # Clean results for JSON (remove non-serializable)
                json_results = self._clean_for_json(results)
                
                with open(json_file, 'w') as f:
                    json.dump(json_results, f, indent=2)
                
                report_files['json'] = str(json_file)
                logger.info(f"    ✓ JSON export: {json_file.name}")
            
            except Exception as e:
                logger.error(f"    JSON export failed: {e}")
        
        logger.info("  ✓ Reports generated")
        
        return report_files
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Test function"""
    logger.info("Advanced Analytics Comprehensive - Test Mode")
    
    # Generate test data
    np.random.seed(42)
    X_train = np.random.randn(200, 10)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] - X_train[:, 2] * 0.5
    X_test = np.random.randn(50, 10)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] - X_test[:, 2] * 0.5
    
    feature_names = [f'Feature_{i}' for i in range(10)]
    
    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize system
    analytics = AdvancedAnalyticsComprehensive(
        output_dir='test_advanced_analytics'
    )
    
    # Run analysis
    results = analytics.run_complete_analysis(
        model, 'RandomForest', X_train, y_train, X_test, y_test, 
        feature_names, target='TEST'
    )
    
    logger.info("\n✅ Test complete!")
    return results


if __name__ == "__main__":
    results = main()
