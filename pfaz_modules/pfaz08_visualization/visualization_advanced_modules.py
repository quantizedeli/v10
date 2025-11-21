"""
ADVANCED VISUALIZATION MODULES
===============================

10. DATA CATALOG VISUALIZATIONS
11. REPORTS VISUALIZATIONS  
12. LOG ANALYTICS VISUALIZATIONS
13. ADVANCED 3D CLUSTERING & ANALYSIS
14. ENSEMBLE & STACKING VISUALIZATIONS
15. PRODUCTION READINESS DASHBOARDS

Author: Nuclear Physics AI Project
Date: October 2025
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLOT_CONFIG = {
    'dpi': 300,
    'figsize_default': (14, 10),
    'figsize_small': (10, 6),
    'figsize_large': (18, 12),
    'figsize_3d': (16, 12),
    'style': 'seaborn-v0_8-darkgrid'
}


# =============================================================================
# 10. DATA CATALOG VISUALIZATIONS
# =============================================================================

class DataCatalogVisualizer:
    """Veri kataloğunun görselleştirilmesi"""
    
    def __init__(self, output_dir='visualizations/data_catalog'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_dataset_overview(self,
                             datasets_info: Dict[str, Dict],
                             save_name: str = 'dataset_overview'):
        """Veri setleri özet görselleştirmesi"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Data Catalog Overview', fontsize=16, fontweight='bold')
        
        dataset_names = list(datasets_info.keys())
        n_nuclei = [datasets_info[d].get('n_nuclei', 0) for d in dataset_names]
        n_features = [datasets_info[d].get('n_features', 0) for d in dataset_names]
        data_quality = [datasets_info[d].get('quality_score', 0) for d in dataset_names]
        creation_dates = [datasets_info[d].get('creation_date', '') for d in dataset_names]
        
        # 1. Dataset sizes
        ax = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_names)))
        ax.barh(dataset_names, n_nuclei, color=colors, edgecolor='black', linewidth=1)
        ax.set_xlabel('Number of Nuclei')
        ax.set_title('Dataset Sizes')
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(n_nuclei):
            ax.text(v, i, f' {v}', va='center', fontsize=9)
        
        # 2. Feature counts
        ax = axes[0, 1]
        ax.bar(dataset_names, n_features, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Counts')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(n_features):
            ax.text(i, v, f' {v}', ha='center', va='bottom', fontsize=9)
        
        # 3. Data quality scores
        ax = axes[1, 0]
        colors_quality = ['green' if q >= 0.8 else 'orange' if q >= 0.6 else 'red' for q in data_quality]
        ax.barh(dataset_names, data_quality, color=colors_quality, alpha=0.7, edgecolor='black', linewidth=1)
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good')
        ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair')
        ax.set_xlabel('Quality Score')
        ax.set_title('Data Quality Assessment')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 1])
        
        # 4. Timeline
        ax = axes[1, 1]
        ax.text(0.5, 0.9, 'Dataset Timeline', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
        y_pos = 0.8
        for name, date in zip(dataset_names, creation_dates):
            ax.text(0.05, y_pos, f'• {name}: {date}', fontsize=10, transform=ax.transAxes)
            y_pos -= 0.15
        ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_data_quality_metrics(self,
                                 quality_metrics: Dict[str, Dict[str, float]],
                                 save_name: str = 'data_quality'):
        """Veri kalitesi metriklerinin heatmap'i"""
        datasets = list(quality_metrics.keys())
        metrics = list(next(iter(quality_metrics.values())).keys())
        
        # Create matrix
        data_matrix = np.array([[quality_metrics[d].get(m, 0) for m in metrics] for d in datasets])
        
        fig, ax = plt.subplots(figsize=self.config['figsize_large'])
        
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(datasets)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(datasets)
        
        # Add values
        for i in range(len(datasets)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Data Quality Metrics Heatmap', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Quality Score')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_feature_statistics_catalog(self,
                                       datasets_features: Dict[str, pd.DataFrame],
                                       save_name: str = 'feature_statistics'):
        """Datasets'te feature istatistikleri"""
        fig = plt.figure(figsize=self.config['figsize_large'])
        
        datasets = list(datasets_features.keys())
        n_datasets = len(datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig.suptitle('Feature Statistics Across Datasets', fontsize=16, fontweight='bold')
        
        for idx, dataset in enumerate(datasets):
            ax = fig.add_subplot(n_rows, n_cols, idx+1)
            
            df = datasets_features[dataset]
            
            # Feature statistics
            feature_names = df.columns[:10]  # Top 10 features
            feature_means = df[feature_names].mean()
            feature_stds = df[feature_names].std()
            
            x_pos = np.arange(len(feature_names))
            ax.bar(x_pos, feature_means, yerr=feature_stds, capsize=5, 
                  color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Mean Value')
            ax.set_title(f'{dataset}\n(n={len(df)} samples)')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 11. REPORTS VISUALIZATIONS
# =============================================================================

class ReportsVisualizer:
    """Raporların görselleştirilmesi"""
    
    def __init__(self, output_dir='visualizations/reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_report_metrics_summary(self,
                                   reports_data: Dict[str, Dict],
                                   save_name: str = 'reports_summary'):
        """Raporlardaki metrikler özeti"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Reports Metrics Summary', fontsize=16, fontweight='bold')
        
        reports = list(reports_data.keys())
        
        # 1. Report completion status
        ax = axes[0, 0]
        completion_status = [reports_data[r].get('completion_percentage', 0) for r in reports]
        colors_status = ['green' if c >= 95 else 'orange' if c >= 70 else 'red' for c in completion_status]
        ax.barh(reports, completion_status, color=colors_status, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xlabel('Completion %')
        ax.set_title('Report Completion Status')
        ax.set_xlim([0, 100])
        ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Section quality
        ax = axes[0, 1]
        if all('section_quality' in reports_data[r] for r in reports):
            sections = list(next(iter(reports_data.values()))['section_quality'].keys())
            for report in reports:
                qualities = [reports_data[report]['section_quality'].get(s, 0) for s in sections]
                ax.plot(sections, qualities, marker='o', label=report, linewidth=2, alpha=0.7)
            ax.set_ylabel('Quality Score')
            ax.set_title('Section Quality Comparison')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 3. Issues and errors
        ax = axes[1, 0]
        issues = [len(reports_data[r].get('issues', [])) for r in reports]
        errors = [len(reports_data[r].get('errors', [])) for r in reports]
        x_pos = np.arange(len(reports))
        width = 0.35
        ax.bar(x_pos - width/2, issues, width, label='Issues', alpha=0.7, edgecolor='black')
        ax.bar(x_pos + width/2, errors, width, label='Errors', alpha=0.7, edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(reports, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Count')
        ax.set_title('Issues and Errors')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Report statistics
        ax = axes[1, 1]
        report_sizes = [reports_data[r].get('file_size_mb', 0) for r in reports]
        ax.bar(reports, report_sizes, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_ylabel('Size (MB)')
        ax.set_title('Report File Sizes')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_report_generation_timeline(self,
                                       report_timestamps: Dict[str, List[datetime]],
                                       save_name: str = 'generation_timeline'):
        """Raporların oluşturulma zaman çizelgesi"""
        fig, axes = plt.subplots(2, 1, figsize=self.config['figsize_large'])
        fig.suptitle('Report Generation Timeline', fontsize=16, fontweight='bold')
        
        # 1. Chronological timeline
        ax = axes[0]
        all_timestamps = []
        all_labels = []
        all_colors = []
        colors_palette = plt.cm.Set3(np.linspace(0, 1, len(report_timestamps)))
        
        for idx, (report_name, timestamps) in enumerate(report_timestamps.items()):
            if timestamps:
                all_timestamps.extend(timestamps)
                all_labels.extend([report_name] * len(timestamps))
                all_colors.extend([colors_palette[idx]] * len(timestamps))
        
        if all_timestamps:
            scatter = ax.scatter(all_timestamps, all_labels, c=range(len(all_timestamps)), 
                               cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Date/Time')
            ax.set_ylabel('Report')
            ax.set_title('Report Generation Timeline')
            ax.grid(True, alpha=0.3, axis='x')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Generation frequency
        ax = axes[1]
        reports = list(report_timestamps.keys())
        generation_counts = [len(timestamps) for timestamps in report_timestamps.values()]
        
        colors_freq = ['green' if c >= 5 else 'orange' if c >= 3 else 'red' for c in generation_counts]
        ax.barh(reports, generation_counts, color=colors_freq, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xlabel('Generation Count')
        ax.set_title('Report Generation Frequency')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 12. LOG ANALYTICS VISUALIZATIONS
# =============================================================================

class LogAnalyticsVisualizer:
    """Log kayıtlarının analiz ve görselleştirilmesi"""
    
    def __init__(self, output_dir='visualizations/log_analytics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_log_statistics(self,
                           log_data: pd.DataFrame,
                           save_name: str = 'log_statistics'):
        """Log istatistikleri"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Log Analytics: Statistics', fontsize=16, fontweight='bold')
        
        # 1. Log level distribution
        ax = axes[0, 0]
        if 'level' in log_data.columns:
            level_counts = log_data['level'].value_counts()
            colors_level = {'ERROR': 'red', 'WARNING': 'orange', 'INFO': 'blue', 'DEBUG': 'gray'}
            colors = [colors_level.get(level, 'gray') for level in level_counts.index]
            ax.barh(level_counts.index, level_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_xlabel('Count')
            ax.set_title('Log Level Distribution')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Errors and warnings over time
        ax = axes[0, 1]
        if 'timestamp' in log_data.columns and 'level' in log_data.columns:
            log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])
            log_data = log_data.sort_values('timestamp')
            
            errors = (log_data['level'] == 'ERROR').cumsum()
            warnings = (log_data['level'] == 'WARNING').cumsum()
            
            ax.plot(log_data['timestamp'], errors, label='Errors', linewidth=2, marker='o', markersize=3)
            ax.plot(log_data['timestamp'], warnings, label='Warnings', linewidth=2, marker='s', markersize=3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Count')
            ax.set_title('Errors & Warnings Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Module distribution
        ax = axes[1, 0]
        if 'module' in log_data.columns:
            module_counts = log_data['module'].value_counts().head(10)
            ax.barh(module_counts.index, module_counts.values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_xlabel('Log Count')
            ax.set_title('Top 10 Modules (by log frequency)')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Log message patterns
        ax = axes[1, 1]
        log_counts_by_level = log_data['level'].value_counts()
        total_logs = log_counts_by_level.sum()
        percentages = (log_counts_by_level / total_logs * 100).values
        
        wedges, texts, autotexts = ax.pie(percentages, labels=log_counts_by_level.index, 
                                          autopct='%1.1f%%', colors=['red', 'orange', 'blue', 'gray'][:len(log_counts_by_level)])
        ax.set_title('Log Level Distribution %')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_error_analysis(self,
                           error_logs: pd.DataFrame,
                           save_name: str = 'error_analysis'):
        """Hata analizi detaylı"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Log Analytics: Error Analysis', fontsize=16, fontweight='bold')
        
        # 1. Error types
        ax = axes[0, 0]
        if 'error_type' in error_logs.columns:
            error_types = error_logs['error_type'].value_counts()
            ax.barh(error_types.index, error_types.values, color='red', alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_xlabel('Frequency')
            ax.set_title('Error Types')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Error timeline
        ax = axes[0, 1]
        if 'timestamp' in error_logs.columns:
            error_logs_copy = error_logs.copy()
            error_logs_copy['timestamp'] = pd.to_datetime(error_logs_copy['timestamp'])
            errors_per_hour = error_logs_copy.groupby(error_logs_copy['timestamp'].dt.floor('H')).size()
            
            ax.plot(errors_per_hour.index, errors_per_hour.values, marker='o', linewidth=2, color='red', alpha=0.7)
            ax.fill_between(errors_per_hour.index, errors_per_hour.values, alpha=0.3, color='red')
            ax.set_xlabel('Time')
            ax.set_ylabel('Error Count')
            ax.set_title('Errors Over Time (hourly)')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Error severity
        ax = axes[1, 0]
        if 'severity' in error_logs.columns:
            severity_counts = error_logs['severity'].value_counts()
            colors_sev = {'CRITICAL': 'darkred', 'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
            colors = [colors_sev.get(sev, 'gray') for sev in severity_counts.index]
            ax.bar(severity_counts.index, severity_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_ylabel('Count')
            ax.set_title('Error Severity Distribution')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Top error messages
        ax = axes[1, 1]
        if 'message' in error_logs.columns:
            top_errors = error_logs['message'].value_counts().head(5)
            messages = [msg[:30] + '...' if len(msg) > 30 else msg for msg in top_errors.index]
            ax.barh(messages, top_errors.values, color='darkred', alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_xlabel('Frequency')
            ax.set_title('Top 5 Error Messages')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_performance_metrics_from_logs(self,
                                          logs_with_metrics: pd.DataFrame,
                                          save_name: str = 'performance_logs'):
        """Log'lardan çıkarılan performans metriklerinin analizi"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Performance Metrics from Logs', fontsize=16, fontweight='bold')
        
        # 1. Execution times
        ax = axes[0, 0]
        if 'execution_time' in logs_with_metrics.columns:
            execution_times = logs_with_metrics['execution_time'].dropna()
            ax.hist(execution_times, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(execution_times.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {execution_times.mean():.2f}s')
            ax.axvline(execution_times.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {execution_times.median():.2f}s')
            ax.set_xlabel('Execution Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title('Execution Times Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Memory usage
        ax = axes[0, 1]
        if 'memory_used' in logs_with_metrics.columns:
            memory_used = logs_with_metrics['memory_used'].dropna()
            ax.plot(range(len(memory_used)), memory_used, marker='o', linewidth=1, markersize=3, color='purple', alpha=0.7)
            ax.fill_between(range(len(memory_used)), memory_used, alpha=0.3, color='purple')
            ax.set_xlabel('Log Entry')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Memory Usage Over Time')
            ax.grid(True, alpha=0.3)
        
        # 3. CPU usage
        ax = axes[1, 0]
        if 'cpu_usage' in logs_with_metrics.columns:
            cpu_usage = logs_with_metrics['cpu_usage'].dropna()
            ax.plot(range(len(cpu_usage)), cpu_usage, marker='s', linewidth=1, markersize=3, color='orange', alpha=0.7)
            ax.fill_between(range(len(cpu_usage)), cpu_usage, alpha=0.3, color='orange')
            ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='High Threshold')
            ax.set_xlabel('Log Entry')
            ax.set_ylabel('CPU Usage (%)')
            ax.set_title('CPU Usage Over Time')
            ax.set_ylim([0, 100])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Performance summary
        ax = axes[1, 1]
        summary_text = "Performance Summary:\n\n"
        if 'execution_time' in logs_with_metrics.columns:
            summary_text += f"Avg Execution Time: {logs_with_metrics['execution_time'].mean():.2f}s\n"
            summary_text += f"Max Execution Time: {logs_with_metrics['execution_time'].max():.2f}s\n"
        if 'memory_used' in logs_with_metrics.columns:
            summary_text += f"Peak Memory: {logs_with_metrics['memory_used'].max():.2f}MB\n"
        if 'cpu_usage' in logs_with_metrics.columns:
            summary_text += f"Avg CPU: {logs_with_metrics['cpu_usage'].mean():.1f}%\n"
            summary_text += f"Max CPU: {logs_with_metrics['cpu_usage'].max():.1f}%"
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               transform=ax.transAxes, family='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 13. ADVANCED 3D CLUSTERING & ANALYSIS
# =============================================================================

class Advanced3DVisualizer:
    """İleri 3D görselleştirmeler"""
    
    def __init__(self, output_dir='visualizations/advanced_3d'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_3d_clustering_analysis(self,
                                   X: np.ndarray,
                                   clusters: np.ndarray,
                                   feature_names: List[str] = None,
                                   save_name: str = '3d_clustering'):
        """3D Clustering analizi"""
        if X.shape[1] < 3:
            logger.warning("Need at least 3 features for 3D plotting")
            return
        
        fig = plt.figure(figsize=self.config['figsize_3d'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique clusters
        unique_clusters = np.unique(clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        # Plot clusters
        for idx, cluster_id in enumerate(unique_clusters):
            cluster_mask = clusters == cluster_id
            ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1], X[cluster_mask, 2],
                      c=[colors[idx]], label=f'Cluster {cluster_id}', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Labels
        if feature_names and len(feature_names) >= 3:
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_zlabel(feature_names[2])
        else:
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
        
        ax.set_title('3D Clustering Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_3d_density_landscape(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 feature_indices: Tuple[int, int, int] = (0, 1, 2),
                                 save_name: str = '3d_density'):
        """3D yoğunluk haritası"""
        fig = plt.figure(figsize=self.config['figsize_3d'])
        ax = fig.add_subplot(111, projection='3d')
        
        idx1, idx2, idx3 = feature_indices
        
        # Surface plot with scatter
        scatter = ax.scatter(X[:, idx1], X[:, idx2], X[:, idx3], c=y, cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'Feature {idx1}')
        ax.set_ylabel(f'Feature {idx2}')
        ax.set_zlabel(f'Feature {idx3}')
        ax.set_title('3D Density Landscape', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Target Value')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_hierarchical_clustering_dendrogram(self,
                                               X: np.ndarray,
                                               method: str = 'ward',
                                               sample_size: int = 100,
                                               save_name: str = 'dendrogram'):
        """Hiyerarşik clustering dendrogramı"""
        # Sample if too large
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        # Compute linkage
        Z = linkage(X_scaled, method=method)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.config['figsize_large'])
        dendrogram(Z, ax=ax, no_labels=True)
        
        ax.set_title(f'Hierarchical Clustering Dendrogram ({method})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 14. ENSEMBLE & STACKING VISUALIZATIONS
# =============================================================================

class EnsembleVisualizationExtended:
    """Ensemble ve stacking modellerin görselleştirilmesi"""
    
    def __init__(self, output_dir='visualizations/ensemble'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def plot_ensemble_comparison(self,
                                ensemble_results: Dict[str, Dict],
                                base_models: Dict[str, Dict],
                                save_name: str = 'ensemble_comparison'):
        """Ensemble vs Base Models karşılaştırması"""
        fig, axes = plt.subplots(2, 2, figsize=self.config['figsize_large'])
        fig.suptitle('Ensemble vs Base Models Comparison', fontsize=16, fontweight='bold')
        
        ensemble_names = list(ensemble_results.keys())
        base_model_names = list(base_models.keys())
        
        # 1. R² Scores
        ax = axes[0, 0]
        ensemble_r2 = [ensemble_results[e].get('r2', 0) for e in ensemble_names]
        base_r2 = [base_models[b].get('r2', 0) for b in base_model_names]
        
        x_pos = np.arange(len(ensemble_names) + len(base_model_names))
        values = ensemble_r2 + base_r2
        colors = ['green']*len(ensemble_names) + ['blue']*len(base_model_names)
        labels = ensemble_names + base_model_names
        
        ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=max(base_r2), color='blue', linestyle='--', alpha=0.5, linewidth=1)
        
        # 2. MAE Comparison
        ax = axes[0, 1]
        ensemble_mae = [ensemble_results[e].get('mae', float('inf')) for e in ensemble_names]
        base_mae = [base_models[b].get('mae', float('inf')) for b in base_model_names]
        
        values_mae = ensemble_mae + base_mae
        ax.bar(x_pos, values_mae, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('MAE')
        ax.set_title('MAE Comparison (lower is better)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Improvement percentages
        ax = axes[1, 0]
        if base_r2:
            best_base_r2 = max(base_r2)
            improvements = [(r2 - best_base_r2) / best_base_r2 * 100 for r2 in ensemble_r2]
            colors_imp = ['green' if imp >= 0 else 'red' for imp in improvements]
            ax.barh(ensemble_names, improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=1)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Improvement (%)')
            ax.set_title('R² Improvement over Best Base Model')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Prediction variance
        ax = axes[1, 1]
        ensemble_vars = [ensemble_results[e].get('prediction_variance', 0) for e in ensemble_names]
        base_vars = [base_models[b].get('prediction_variance', 0) for b in base_model_names]
        
        values_var = ensemble_vars + base_vars
        ax.bar(x_pos, values_var, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Prediction Variance')
        ax.set_title('Prediction Variance (lower is more stable)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")
    
    def plot_stacking_architecture(self,
                                  base_models: List[str],
                                  meta_model: str,
                                  performance_data: Dict,
                                  save_name: str = 'stacking_architecture'):
        """Stacking mimarisinin gösterimi"""
        fig, ax = plt.subplots(figsize=self.config['figsize_large'])
        
        # Create visualization (simplified version)
        ax.text(0.5, 0.95, 'Stacking Architecture', ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        # Base models layer
        y_base = 0.7
        ax.text(0.5, y_base, 'Base Models', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        n_base = len(base_models)
        for i, model in enumerate(base_models):
            x_pos = (i + 1) / (n_base + 1)
            y_pos = y_base - 0.15
            
            # Box for model
            box = plt.Rectangle((x_pos - 0.08, y_pos - 0.05), 0.16, 0.1, 
                               transform=ax.transAxes, fill=True, facecolor='lightblue', 
                               edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(x_pos, y_pos, model, ha='center', va='center', fontsize=9, transform=ax.transAxes)
            
            # Arrows to meta model
            ax.annotate('', xy=(0.5, 0.3), xytext=(x_pos, y_pos),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
        
        # Meta model layer
        meta_box = plt.Rectangle((0.35, 0.2), 0.3, 0.08, 
                                transform=ax.transAxes, fill=True, facecolor='lightgreen', 
                                edgecolor='black', linewidth=2)
        ax.add_patch(meta_box)
        ax.text(0.5, 0.24, f'Meta Model: {meta_model}', ha='center', va='center', 
               fontsize=11, fontweight='bold', transform=ax.transAxes)
        
        # Output
        output_box = plt.Rectangle((0.4, 0.05), 0.2, 0.08,
                                  transform=ax.transAxes, fill=True, facecolor='lightyellow',
                                  edgecolor='black', linewidth=1)
        ax.add_patch(output_box)
        ax.arrow(0.5, 0.2, 0, -0.05, head_width=0.02, head_length=0.01, fc='black', ec='black', transform=ax.transAxes)
        ax.text(0.5, 0.09, 'Final Prediction', ha='center', va='center', fontsize=10, transform=ax.transAxes)
        
        ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# 15. PRODUCTION READINESS DASHBOARDS
# =============================================================================

class ProductionReadinessDashboard:
    """Production hazırlık dashboard'u"""
    
    def __init__(self, output_dir='visualizations/production_readiness'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = PLOT_CONFIG
    
    def create_production_readiness_report(self,
                                          model_metrics: Dict,
                                          robustness_scores: Dict,
                                          resource_requirements: Dict,
                                          save_name: str = 'production_readiness'):
        """Production hazırlık raporu"""
        fig = plt.figure(figsize=self.config['figsize_large'])
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Production Readiness Assessment', fontsize=18, fontweight='bold')
        
        # 1. Performance Checklist
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        checklist_items = [
            ('Model R² > 0.90', model_metrics.get('r2', 0) > 0.90),
            ('MAE < 0.05', model_metrics.get('mae', float('inf')) < 0.05),
            ('RMSE < 0.10', model_metrics.get('rmse', float('inf')) < 0.10),
            ('Cross-validation Stable', robustness_scores.get('cv_stability', 0) > 0.85),
            ('Overfitting Check', robustness_scores.get('overfitting_score', 0) > 0.8),
        ]
        
        ax1.text(0.05, 0.95, 'Performance Checklist', fontsize=12, fontweight='bold', transform=ax1.transAxes)
        for i, (item, status) in enumerate(checklist_items):
            y_pos = 0.8 - i * 0.15
            symbol = '✓' if status else '✗'
            color = 'green' if status else 'red'
            ax1.text(0.1, y_pos, f'{symbol} {item}', fontsize=11, transform=ax1.transAxes, color=color, fontweight='bold')
        
        # 2. Model Performance Gauge
        ax2 = fig.add_subplot(gs[1, 0])
        performance_score = model_metrics.get('r2', 0)
        colors_gauge = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        ranges = np.array([0, 0.7, 0.8, 0.9, 0.95, 1.0])
        
        for i in range(len(colors_gauge)):
            ax2.barh(0, ranges[i+1] - ranges[i], left=ranges[i], color=colors_gauge[i], height=0.3)
        
        ax2.arrow(performance_score, 0.2, 0, 0.08, head_width=0.02, head_length=0.02, fc='black', ec='black')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([-0.2, 0.5])
        ax2.set_xticks([0, 0.7, 0.8, 0.9, 0.95, 1.0])
        ax2.set_xticklabels(['0.0', '0.7', '0.8', '0.9', '0.95', '1.0'], fontsize=8)
        ax2.set_yticks([])
        ax2.set_title('Model Performance Score', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.text(0.5, -0.15, f'R² Score: {performance_score:.3f}', ha='center', fontsize=10, transform=ax2.transData)
        
        # 3. Robustness Score
        ax3 = fig.add_subplot(gs[1, 1])
        robustness_avg = np.mean(list(robustness_scores.values()))
        ax3.bar(['Robustness'], [robustness_avg], color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylim([0, 1])
        ax3.set_ylabel('Score')
        ax3.set_title('Overall Robustness', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.text(0, robustness_avg + 0.05, f'{robustness_avg:.2f}', ha='center', fontsize=11, fontweight='bold')
        
        # 4. Resource Requirements
        ax4 = fig.add_subplot(gs[1, 2])
        resources = list(resource_requirements.keys())
        values = list(resource_requirements.values())
        
        ax4.barh(resources, values, color=['red' if v > 50 else 'orange' if v > 30 else 'green' for v in values],
                alpha=0.7, edgecolor='black', linewidth=1)
        ax4.set_xlabel('Value')
        ax4.set_title('Resource Requirements', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Deployment Readiness
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        readiness_score = (performance_score + robustness_avg) / 2
        if readiness_score >= 0.95:
            status = "🟢 READY FOR PRODUCTION"
            color = 'green'
        elif readiness_score >= 0.85:
            status = "🟡 READY WITH MONITORING"
            color = 'orange'
        else:
            status = "🔴 NOT READY - REQUIRES IMPROVEMENTS"
            color = 'red'
        
        ax5.text(0.5, 0.8, 'Production Deployment Status', ha='center', fontsize=12, fontweight='bold', transform=ax5.transAxes)
        ax5.text(0.5, 0.4, status, ha='center', fontsize=14, fontweight='bold', color=color, transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat' if color == 'orange' else 'lightgreen' if color == 'green' else 'lightcoral',
                         alpha=0.8))
        ax5.text(0.5, 0.05, f'Overall Readiness Score: {readiness_score:.2%}', ha='center', fontsize=11, 
                transform=ax5.transAxes, style='italic')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'{save_name}.png', dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {save_name}.png")


# =============================================================================
# INTEGRATION & MAIN
# =============================================================================

def main():
    """Test ve demo"""
    logger.info("\n" + "="*80)
    logger.info("ADVANCED VISUALIZATION MODULES - TEST")
    logger.info("="*80)
    
    # Initialize visualizers
    logger.info("\nInitializing advanced visualizers...")
    
    visualizers = {
        'DataCatalog': DataCatalogVisualizer(),
        'Reports': ReportsVisualizer(),
        'LogAnalytics': LogAnalyticsVisualizer(),
        'Advanced3D': Advanced3DVisualizer(),
        'Ensemble': EnsembleVisualizationExtended(),
        'ProductionReadiness': ProductionReadinessDashboard()
    }
    
    logger.info("\n✓ Advanced visualizers initialized successfully!")
    logger.info("\nAvailable modules:")
    for name in visualizers.keys():
        logger.info(f"  - {name}Visualizer")


if __name__ == "__main__":
    main()
