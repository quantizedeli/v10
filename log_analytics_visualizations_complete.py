# -*- coding: utf-8 -*-
"""
LOG ANALYTICS VISUALIZATIONS COMPLETE
======================================

Eksik log analytics grafiklerini tamamlayan modül

Yeni Grafikler:
1. training_progress_timeline.png - Training sürecinin zaman çizelgesi
2. error_warning_analysis.png - Error ve warning dağılımı
3. module_activity_heatmap.png - Modül aktivite ısı haritası
4. performance_over_time.png - Metrikler zaman içinde
5. resource_usage_monitoring.png - CPU/GPU/RAM kullanımı

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 8 Completion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter, HourLocator
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plot configuration
PLOT_CONFIG = {
    'dpi': 300,
    'figsize_default': (16, 10),
    'figsize_wide': (18, 8),
    'style': 'seaborn-v0_8-darkgrid'
}

plt.style.use(PLOT_CONFIG['style'])
sns.set_palette("husl")


class LogAnalyticsVisualizationsComplete:
    """Log analytics için eksik grafikleri tamamlar"""
    
    def __init__(self, output_dir='visualizations/log_analytics_complete'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ LogAnalyticsVisualizationsComplete initialized: {self.output_dir}")
    
    def plot_training_progress_timeline(self,
                                       log_data: pd.DataFrame,
                                       save_name: str = 'training_progress_timeline'):
        """
        Training progress timeline from logs
        
        Args:
            log_data: DataFrame with columns:
                timestamp, level, module, message, phase, target, model_id
        """
        logger.info("\n→ Creating training progress timeline...")
        
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        fig.suptitle('Training Progress Timeline\nEnd-to-End Project Execution', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Parse timestamps
        log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])
        log_data = log_data.sort_values('timestamp')
        
        # 1. Phase progression
        ax = axes[0]
        phases = log_data['phase'].unique()
        phase_colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
        phase_color_map = dict(zip(phases, phase_colors))
        
        y_pos = 0
        for phase in phases:
            phase_logs = log_data[log_data['phase'] == phase]
            if len(phase_logs) > 0:
                start_time = phase_logs['timestamp'].min()
                end_time = phase_logs['timestamp'].max()
                duration = (end_time - start_time).total_seconds() / 3600  # hours
                
                ax.barh(y_pos, duration, left=start_time, height=0.8,
                       color=phase_color_map[phase], edgecolor='black', linewidth=1.5,
                       label=phase, alpha=0.7)
                
                # Add phase name
                mid_time = start_time + (end_time - start_time) / 2
                ax.text(mid_time, y_pos, phase, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
                
                y_pos += 1
        
        ax.set_yticks(range(len(phases)))
        ax.set_yticklabels(phases, fontsize=9)
        ax.set_ylabel('Phase', fontsize=10, fontweight='bold')
        ax.set_title('(A) Phase Execution Timeline', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Training events (models completed)
        ax = axes[1]
        
        # Extract model completion events
        model_events = log_data[log_data['message'].str.contains('completed|trained', case=False, na=False)]
        
        if len(model_events) > 0:
            # Cumulative models trained
            model_events = model_events.sort_values('timestamp')
            cumulative_models = np.arange(1, len(model_events) + 1)
            
            ax.plot(model_events['timestamp'], cumulative_models, 
                   linewidth=2.5, color='green', marker='o', markersize=4,
                   label='Models Completed')
            
            # Mark milestones
            milestones = [int(len(cumulative_models) * p) for p in [0.25, 0.5, 0.75, 1.0]]
            for milestone in milestones:
                if milestone > 0 and milestone <= len(cumulative_models):
                    idx = milestone - 1
                    ax.axhline(cumulative_models[idx], color='red', linestyle='--',
                             alpha=0.5, linewidth=1)
                    ax.text(model_events['timestamp'].iloc[0], cumulative_models[idx],
                           f'{cumulative_models[idx]} models', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_ylabel('Cumulative Models', fontsize=10, fontweight='bold')
        ax.set_title('(B) Model Training Progress', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. Success/Failure rate over time
        ax = axes[2]
        
        # Calculate success rate in time windows
        window_size = pd.Timedelta(hours=2)
        time_windows = pd.date_range(log_data['timestamp'].min(), 
                                     log_data['timestamp'].max(), 
                                     freq=window_size)
        
        success_rates = []
        failure_rates = []
        window_centers = []
        
        for i in range(len(time_windows) - 1):
            window_logs = log_data[(log_data['timestamp'] >= time_windows[i]) & 
                                  (log_data['timestamp'] < time_windows[i+1])]
            
            if len(window_logs) > 0:
                n_success = len(window_logs[window_logs['level'] == 'INFO'])
                n_error = len(window_logs[window_logs['level'] == 'ERROR'])
                total = n_success + n_error
                
                if total > 0:
                    success_rates.append(n_success / total * 100)
                    failure_rates.append(n_error / total * 100)
                    window_centers.append(time_windows[i] + window_size/2)
        
        if len(success_rates) > 0:
            ax.fill_between(window_centers, 0, success_rates, 
                           color='green', alpha=0.3, label='Success Rate')
            ax.fill_between(window_centers, success_rates, 
                           np.array(success_rates) + np.array(failure_rates),
                           color='red', alpha=0.3, label='Error Rate')
            
            ax.plot(window_centers, success_rates, linewidth=2, color='darkgreen')
            ax.plot(window_centers, np.array(success_rates) + np.array(failure_rates), 
                   linewidth=2, color='darkred')
        
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title('(C) Success/Error Rate Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Format x-axis
        date_formatter = DateFormatter('%H:%M')
        for ax in axes:
            ax.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_name}.png")
        return save_path
    
    def plot_error_warning_analysis(self,
                                   log_data: pd.DataFrame,
                                   save_name: str = 'error_warning_analysis'):
        """Error and warning distribution analysis"""
        
        logger.info("\n→ Creating error/warning analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error & Warning Analysis\nSystem Reliability Metrics', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Error/Warning counts by module
        ax = axes[0, 0]
        error_warning_logs = log_data[log_data['level'].isin(['ERROR', 'WARNING'])]
        
        if len(error_warning_logs) > 0:
            module_counts = error_warning_logs.groupby(['module', 'level']).size().unstack(fill_value=0)
            
            module_counts.plot(kind='barh', stacked=True, ax=ax,
                              color=['red', 'orange'], edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel('Count', fontsize=10, fontweight='bold')
            ax.set_ylabel('Module', fontsize=10, fontweight='bold')
            ax.set_title('(A) Errors & Warnings by Module', fontsize=12, fontweight='bold')
            ax.legend(['ERROR', 'WARNING'], loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Timeline of errors and warnings
        ax = axes[0, 1]
        
        if len(error_warning_logs) > 0:
            error_warning_logs = error_warning_logs.sort_values('timestamp')
            error_warning_logs['timestamp'] = pd.to_datetime(error_warning_logs['timestamp'])
            
            errors = error_warning_logs[error_warning_logs['level'] == 'ERROR']
            warnings = error_warning_logs[error_warning_logs['level'] == 'WARNING']
            
            # Cumulative plot
            errors_cum = np.arange(1, len(errors) + 1)
            warnings_cum = np.arange(1, len(warnings) + 1)
            
            if len(errors) > 0:
                ax.plot(errors['timestamp'], errors_cum, linewidth=2.5, 
                       color='red', marker='X', markersize=6, label='Errors')
            
            if len(warnings) > 0:
                ax.plot(warnings['timestamp'], warnings_cum, linewidth=2.5,
                       color='orange', marker='o', markersize=4, label='Warnings')
            
            ax.set_xlabel('Time', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cumulative Count', fontsize=10, fontweight='bold')
            ax.set_title('(B) Errors & Warnings Timeline', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            date_formatter = DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Error frequency heatmap (by hour and module)
        ax = axes[1, 0]
        
        if len(error_warning_logs) > 0:
            error_warning_logs['hour'] = pd.to_datetime(error_warning_logs['timestamp']).dt.hour
            
            heatmap_data = error_warning_logs.groupby(['hour', 'module']).size().unstack(fill_value=0)
            
            if len(heatmap_data) > 0:
                sns.heatmap(heatmap_data.T, cmap='YlOrRd', annot=True, fmt='d',
                           cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)
                
                ax.set_xlabel('Hour of Day', fontsize=10, fontweight='bold')
                ax.set_ylabel('Module', fontsize=10, fontweight='bold')
                ax.set_title('(C) Error/Warning Heatmap (by Hour)', fontsize=12, fontweight='bold')
        
        # 4. Top error messages
        ax = axes[1, 1]
        ax.axis('off')
        
        if len(error_warning_logs) > 0:
            # Extract and count error messages
            error_msgs = errors['message'].value_counts().head(10) if len(errors) > 0 else pd.Series()
            
            text = "TOP 10 ERROR MESSAGES\n" + "="*50 + "\n\n"
            
            for idx, (msg, count) in enumerate(error_msgs.items(), 1):
                # Truncate long messages
                msg_short = msg[:60] + "..." if len(msg) > 60 else msg
                text += f"{idx}. [{count}x] {msg_short}\n\n"
            
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='mistyrose',
                            alpha=0.9, edgecolor='red', linewidth=2))
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_name}.png")
        return save_path
    
    def plot_module_activity_heatmap(self,
                                    log_data: pd.DataFrame,
                                    save_name: str = 'module_activity_heatmap'):
        """Module activity heatmap over time"""
        
        logger.info("\n→ Creating module activity heatmap...")
        
        fig, ax = plt.subplots(figsize=(18, 10))
        fig.suptitle('Module Activity Heatmap\nSystem Component Usage Over Time', 
                    fontsize=16, fontweight='bold')
        
        # Parse timestamps and create time windows
        log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])
        log_data = log_data.sort_values('timestamp')
        
        # Create 30-minute windows
        window_size = pd.Timedelta(minutes=30)
        time_windows = pd.date_range(log_data['timestamp'].min(),
                                     log_data['timestamp'].max(),
                                     freq=window_size)
        
        modules = log_data['module'].unique()[:20]  # Top 20 modules
        
        # Create activity matrix
        activity_matrix = np.zeros((len(modules), len(time_windows)-1))
        
        for i, module in enumerate(modules):
            module_logs = log_data[log_data['module'] == module]
            
            for j in range(len(time_windows) - 1):
                window_logs = module_logs[(module_logs['timestamp'] >= time_windows[j]) &
                                         (module_logs['timestamp'] < time_windows[j+1])]
                activity_matrix[i, j] = len(window_logs)
        
        # Plot heatmap
        im = ax.imshow(activity_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_yticks(np.arange(len(modules)))
        ax.set_yticklabels(modules, fontsize=9)
        
        # X-axis: time labels
        time_labels = [t.strftime('%H:%M') for t in time_windows[:-1:max(1, len(time_windows)//10)]]
        x_ticks = np.arange(0, len(time_windows)-1, max(1, len(time_windows)//10))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=9)
        
        ax.set_xlabel('Time (30-min windows)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Module', fontsize=11, fontweight='bold')
        ax.set_title('Log Activity by Module and Time', fontsize=13, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Log Count', fontsize=10, fontweight='bold')
        
        # Add grid
        ax.set_xticks(np.arange(len(time_windows)-1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(modules)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_name}.png")
        return save_path
    
    def plot_performance_over_time(self,
                                  metrics_log: pd.DataFrame,
                                  save_name: str = 'performance_over_time'):
        """
        Performance metrics over time
        
        Args:
            metrics_log: DataFrame with columns:
                timestamp, model_id, target, r2, rmse, mae, training_time
        """
        logger.info("\n→ Creating performance over time plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Metrics Over Time\nTraining Evolution', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        metrics_log['timestamp'] = pd.to_datetime(metrics_log['timestamp'])
        metrics_log = metrics_log.sort_values('timestamp')
        
        # 1. R² progression
        ax = axes[0, 0]
        
        # Running average R²
        window = 50
        if len(metrics_log) >= window:
            rolling_r2 = metrics_log['r2'].rolling(window=window).mean()
            
            ax.scatter(metrics_log['timestamp'], metrics_log['r2'], 
                      alpha=0.3, s=20, c='steelblue', label='Individual Models')
            ax.plot(metrics_log['timestamp'], rolling_r2, 
                   linewidth=3, color='red', label=f'Rolling Avg (n={window})')
            
            # Add trend line
            from scipy import stats
            x_numeric = (metrics_log['timestamp'] - metrics_log['timestamp'].min()).dt.total_seconds()
            slope, intercept, _, _, _ = stats.linregress(x_numeric, metrics_log['r2'])
            trend = slope * x_numeric + intercept
            ax.plot(metrics_log['timestamp'], trend, 
                   linewidth=2, color='green', linestyle='--', label='Trend')
        
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=10, fontweight='bold')
        ax.set_title('(A) R² Score Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        date_formatter = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(date_formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. RMSE progression
        ax = axes[0, 1]
        
        if len(metrics_log) >= window:
            rolling_rmse = metrics_log['rmse'].rolling(window=window).mean()
            
            ax.scatter(metrics_log['timestamp'], metrics_log['rmse'],
                      alpha=0.3, s=20, c='coral', label='Individual Models')
            ax.plot(metrics_log['timestamp'], rolling_rmse,
                   linewidth=3, color='red', label=f'Rolling Avg (n={window})')
        
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=10, fontweight='bold')
        ax.set_title('(B) RMSE Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(date_formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Training time trend
        ax = axes[1, 0]
        
        if len(metrics_log) >= window:
            rolling_time = metrics_log['training_time'].rolling(window=window).mean()
            
            ax.scatter(metrics_log['timestamp'], metrics_log['training_time'],
                      alpha=0.3, s=20, c='green', label='Individual Models')
            ax.plot(metrics_log['timestamp'], rolling_time,
                   linewidth=3, color='darkgreen', label=f'Rolling Avg (n={window})')
        
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=10, fontweight='bold')
        ax.set_title('(C) Training Time Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(date_formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Performance by target over time
        ax = axes[1, 1]
        
        for target in metrics_log['target'].unique():
            target_data = metrics_log[metrics_log['target'] == target]
            
            if len(target_data) >= 10:
                rolling = target_data['r2'].rolling(window=min(20, len(target_data)//2)).mean()
                ax.plot(target_data['timestamp'], rolling, 
                       linewidth=2.5, marker='o', markersize=4, label=target)
        
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('R² Score (Rolling Avg)', fontsize=10, fontweight='bold')
        ax.set_title('(D) Performance by Target', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        ax.xaxis.set_major_formatter(date_formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_name}.png")
        return save_path
    
    def plot_resource_usage_monitoring(self,
                                      resource_log: pd.DataFrame,
                                      save_name: str = 'resource_usage_monitoring'):
        """
        Resource usage monitoring (CPU, GPU, RAM)
        
        Args:
            resource_log: DataFrame with columns:
                timestamp, cpu_percent, gpu_percent, ram_gb, gpu_ram_gb
        """
        logger.info("\n→ Creating resource usage monitoring plot...")
        
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        fig.suptitle('Resource Usage Monitoring\nSystem Resource Utilization', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        resource_log['timestamp'] = pd.to_datetime(resource_log['timestamp'])
        resource_log = resource_log.sort_values('timestamp')
        
        # 1. CPU usage
        ax = axes[0]
        ax.fill_between(resource_log['timestamp'], 0, resource_log['cpu_percent'],
                       color='steelblue', alpha=0.3)
        ax.plot(resource_log['timestamp'], resource_log['cpu_percent'],
               linewidth=2, color='darkblue', label='CPU Usage')
        
        # Add threshold lines
        ax.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='High (80%)')
        ax.axhline(95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical (95%)')
        
        ax.set_ylabel('CPU Usage (%)', fontsize=10, fontweight='bold')
        ax.set_title('(A) CPU Utilization', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # 2. GPU usage
        ax = axes[1]
        ax.fill_between(resource_log['timestamp'], 0, resource_log['gpu_percent'],
                       color='green', alpha=0.3)
        ax.plot(resource_log['timestamp'], resource_log['gpu_percent'],
               linewidth=2, color='darkgreen', label='GPU Usage')
        
        ax.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(95, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_ylabel('GPU Usage (%)', fontsize=10, fontweight='bold')
        ax.set_title('(B) GPU Utilization', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # 3. RAM usage
        ax = axes[2]
        
        # System RAM
        ax.fill_between(resource_log['timestamp'], 0, resource_log['ram_gb'],
                       color='coral', alpha=0.3, label='System RAM')
        ax.plot(resource_log['timestamp'], resource_log['ram_gb'],
               linewidth=2, color='darkred')
        
        # GPU RAM
        if 'gpu_ram_gb' in resource_log.columns:
            ax.fill_between(resource_log['timestamp'], 
                           resource_log['ram_gb'], 
                           resource_log['ram_gb'] + resource_log['gpu_ram_gb'],
                           color='purple', alpha=0.3, label='GPU RAM')
            ax.plot(resource_log['timestamp'], 
                   resource_log['ram_gb'] + resource_log['gpu_ram_gb'],
                   linewidth=2, color='darkviolet')
        
        ax.set_xlabel('Time', fontsize=10, fontweight='bold')
        ax.set_ylabel('RAM Usage (GB)', fontsize=10, fontweight='bold')
        ax.set_title('(C) Memory Utilization', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        date_formatter = DateFormatter('%H:%M')
        for ax in axes:
            ax.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Saved: {save_name}.png")
        return save_path
    
    def generate_all_log_analytics_plots(self,
                                         log_data: pd.DataFrame = None,
                                         metrics_log: pd.DataFrame = None,
                                         resource_log: pd.DataFrame = None):
        """Generate all 5 log analytics plots"""
        
        logger.info("\n" + "="*70)
        logger.info("GENERATING COMPLETE LOG ANALYTICS VISUALIZATIONS")
        logger.info("="*70)
        
        generated_plots = []
        
        if log_data is not None:
            # 1. Training progress timeline
            try:
                path = self.plot_training_progress_timeline(log_data)
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  ✗ Failed timeline: {e}")
            
            # 2. Error/warning analysis
            try:
                path = self.plot_error_warning_analysis(log_data)
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  ✗ Failed error analysis: {e}")
            
            # 3. Module activity heatmap
            try:
                path = self.plot_module_activity_heatmap(log_data)
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  ✗ Failed heatmap: {e}")
        
        # 4. Performance over time
        if metrics_log is not None:
            try:
                path = self.plot_performance_over_time(metrics_log)
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  ✗ Failed performance: {e}")
        
        # 5. Resource monitoring
        if resource_log is not None:
            try:
                path = self.plot_resource_usage_monitoring(resource_log)
                generated_plots.append(path)
            except Exception as e:
                logger.error(f"  ✗ Failed resource: {e}")
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ LOG ANALYTICS VISUALIZATIONS COMPLETE: {len(generated_plots)}/5")
        logger.info("="*70)
        
        return generated_plots


def generate_sample_data():
    """Generate sample log data for testing"""
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=10)
    n_logs = 1000
    timestamps = [start_time + timedelta(seconds=i*36) for i in range(n_logs)]
    
    phases = ['PFAZ1', 'PFAZ2', 'PFAZ3', 'PFAZ4', 'PFAZ5']
    modules = ['data_loader', 'model_trainer', 'anfis_trainer', 'evaluator', 
               'visualizer', 'reporter', 'ensemble_builder']
    levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
    
    # Log data
    log_data = pd.DataFrame({
        'timestamp': timestamps,
        'level': np.random.choice(levels, n_logs, p=[0.7, 0.15, 0.10, 0.05]),
        'module': np.random.choice(modules, n_logs),
        'message': [f'Log message {i}' for i in range(n_logs)],
        'phase': np.random.choice(phases, n_logs),
        'target': np.random.choice(['MM', 'QM', 'Beta_2'], n_logs),
        'model_id': [f'Model_{i%50}' for i in range(n_logs)]
    })
    
    # Metrics log
    n_models = 200
    model_timestamps = np.sort(np.random.choice(timestamps, n_models, replace=False))
    
    metrics_log = pd.DataFrame({
        'timestamp': model_timestamps,
        'model_id': [f'Model_{i}' for i in range(n_models)],
        'target': np.random.choice(['MM', 'QM', 'Beta_2'], n_models),
        'r2': np.random.uniform(0.80, 0.98, n_models),
        'rmse': np.random.uniform(0.08, 0.30, n_models),
        'mae': np.random.uniform(0.05, 0.20, n_models),
        'training_time': np.random.uniform(10, 180, n_models)
    })
    
    # Resource log
    n_resource = 500
    resource_timestamps = [start_time + timedelta(seconds=i*72) for i in range(n_resource)]
    
    resource_log = pd.DataFrame({
        'timestamp': resource_timestamps,
        'cpu_percent': np.random.uniform(20, 95, n_resource),
        'gpu_percent': np.random.uniform(10, 90, n_resource),
        'ram_gb': np.random.uniform(8, 28, n_resource),
        'gpu_ram_gb': np.random.uniform(2, 10, n_resource)
    })
    
    return log_data, metrics_log, resource_log


if __name__ == "__main__":
    # Test with sample data
    logger.info("\n" + "="*70)
    logger.info("TESTING LOG ANALYTICS VISUALIZATIONS COMPLETE")
    logger.info("="*70)
    
    # Generate sample data
    log_data, metrics_log, resource_log = generate_sample_data()
    
    # Create visualizer
    visualizer = LogAnalyticsVisualizationsComplete('test_output/log_analytics')
    
    # Generate all plots
    visualizer.generate_all_log_analytics_plots(
        log_data=log_data,
        metrics_log=metrics_log,
        resource_log=resource_log
    )
    
    logger.info("\n✓ Test complete! Check test_output/log_analytics/")
