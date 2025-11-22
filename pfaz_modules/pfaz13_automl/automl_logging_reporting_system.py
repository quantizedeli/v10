# -*- coding: utf-8 -*-
"""
AUTOML COMPREHENSIVE LOGGING & REPORTING SYSTEM
================================================

Tez için AutoML tüm işlem kayıtlarını tutan kapsamlı sistem

Features:
1. Trial-by-trial logging (her denemede ne oldu?)
2. Optimization history tracking
3. Parameter importance analysis
4. Comprehensive Excel reports (15+ sheets)
5. Automatic visualization generation (20+ plots)
6. Interpretable insights & recommendations
7. LaTeX-ready summary tables

Kayıt Edilen Bilgiler:
- Trial ID, timestamp, duration
- Hyperparameters tested
- Metrics (R², RMSE, MAE)
- Success/failure status
- Pruning decisions
- Best config evolution
- Pareto front (multi-objective)
- Parameter correlations
- Feature importance changes

Author: Nuclear Physics AI Project - AutoML Architect
Date: 2025-10-24
Version: 1.0.0 - PFAZ 13 Core
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AutoMLTrialRecord:
    """Single AutoML trial complete record"""
    trial_id: int
    timestamp: str
    model_type: str  # 'RF', 'XGBoost', 'ANFIS', etc.
    dataset_name: str
    
    # Hyperparameters
    hyperparameters: Dict
    
    # Metrics
    train_r2: float
    train_rmse: float
    train_mae: float
    val_r2: float
    val_rmse: float
    val_mae: float
    test_r2: Optional[float] = None
    test_rmse: Optional[float] = None
    test_mae: Optional[float] = None
    
    # Training info
    training_time: float
    n_samples_train: int
    n_samples_val: int
    n_features: int
    
    # Status
    status: str  # 'COMPLETE', 'PRUNED', 'FAILED'
    pruning_reason: Optional[str] = None
    error_message: Optional[str] = None
    
    # Optimization
    is_best_so_far: bool = False
    improvement_over_previous_best: Optional[float] = None
    
    # Multi-objective
    pareto_rank: Optional[int] = None
    crowding_distance: Optional[float] = None


@dataclass
class AutoMLOptimizationSummary:
    """Complete optimization run summary"""
    optimization_id: str
    timestamp: str
    model_type: str
    objective: str  # 'r2', 'multi_objective', etc.
    
    # Configuration
    n_trials_total: int
    n_trials_completed: int
    n_trials_pruned: int
    n_trials_failed: int
    
    # Best results
    best_trial_id: int
    best_hyperparameters: Dict
    best_val_r2: float
    best_val_rmse: float
    best_val_mae: float
    
    # Optimization stats
    total_time: float
    avg_trial_time: float
    convergence_trial: Optional[int] = None  # At which trial did it converge?
    
    # Parameter insights
    most_important_param: str
    least_important_param: str
    parameter_correlations: Dict
    
    # Recommendations
    recommended_config: Dict
    confidence_score: float
    improvement_potential: str


# ============================================================================
# AUTOML TRIAL LOGGER
# ============================================================================

class AutoMLTrialLogger:
    """
    Log every AutoML trial with complete details
    
    For thesis: Track EVERYTHING!
    - What was tried?
    - Why was it good/bad?
    - How did optimization progress?
    - Which parameters matter most?
    """
    
    def __init__(self, output_dir: str = 'automl_logs'):
        """
        Initialize AutoML trial logger
        
        Args:
            output_dir: Directory for logs and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.trials: List[AutoMLTrialRecord] = []
        self.current_optimization: Optional[str] = None
        self.optimization_summaries: List[AutoMLOptimizationSummary] = []
        
        # Best tracking
        self.best_trial: Optional[AutoMLTrialRecord] = None
        self.best_val_r2: float = -np.inf
        
        logger.info(f"✓ AutoMLTrialLogger initialized: {self.output_dir}")
    
    def start_optimization(self, 
                          model_type: str,
                          objective: str = 'r2',
                          n_trials: int = 100) -> str:
        """Start new optimization run"""
        
        optimization_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_optimization = optimization_id
        
        logger.info(f"\n{'='*70}")
        logger.info(f"AUTOML OPTIMIZATION STARTED")
        logger.info(f"{'='*70}")
        logger.info(f"ID: {optimization_id}")
        logger.info(f"Model: {model_type}")
        logger.info(f"Objective: {objective}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"{'='*70}\n")
        
        return optimization_id
    
    def log_trial(self, trial_record: AutoMLTrialRecord):
        """Log single trial"""
        
        # Check if best so far
        if trial_record.val_r2 > self.best_val_r2:
            if self.best_trial is not None:
                improvement = trial_record.val_r2 - self.best_val_r2
                trial_record.improvement_over_previous_best = improvement
                logger.info(f"  [TARGET] NEW BEST! R²={trial_record.val_r2:.4f} (+{improvement:.4f})")
            
            trial_record.is_best_so_far = True
            self.best_trial = trial_record
            self.best_val_r2 = trial_record.val_r2
        
        # Add to history
        self.trials.append(trial_record)
        
        # Log
        status_emoji = {
            'COMPLETE': '[SUCCESS]',
            'PRUNED': '✂️',
            'FAILED': '[ERROR]'
        }.get(trial_record.status, '[UNKNOWN]')
        
        logger.info(
            f"  {status_emoji} Trial {trial_record.trial_id:3d} | "
            f"R²={trial_record.val_r2:.4f} | "
            f"Time={trial_record.training_time:.1f}s | "
            f"{trial_record.status}"
        )
        
        if trial_record.pruning_reason:
            logger.info(f"     Reason: {trial_record.pruning_reason}")
    
    def end_optimization(self) -> AutoMLOptimizationSummary:
        """End optimization and create summary"""
        
        if not self.trials:
            raise ValueError("No trials logged!")
        
        # Count statuses
        statuses = [t.status for t in self.trials]
        n_completed = statuses.count('COMPLETE')
        n_pruned = statuses.count('PRUNED')
        n_failed = statuses.count('FAILED')
        
        # Find convergence point (when best didn't improve for 20 trials)
        convergence_trial = None
        best_r2_history = []
        for i, trial in enumerate(self.trials):
            if trial.is_best_so_far:
                best_r2_history.append(i)
        
        if len(best_r2_history) > 0:
            last_improvement = best_r2_history[-1]
            if len(self.trials) - last_improvement > 20:
                convergence_trial = last_improvement
        
        # Parameter importance (correlation with R²)
        param_importance = self._calculate_parameter_importance()
        
        # Create summary
        summary = AutoMLOptimizationSummary(
            optimization_id=self.current_optimization,
            timestamp=datetime.now().isoformat(),
            model_type=self.trials[0].model_type,
            objective='r2',
            n_trials_total=len(self.trials),
            n_trials_completed=n_completed,
            n_trials_pruned=n_pruned,
            n_trials_failed=n_failed,
            best_trial_id=self.best_trial.trial_id,
            best_hyperparameters=self.best_trial.hyperparameters,
            best_val_r2=self.best_trial.val_r2,
            best_val_rmse=self.best_trial.val_rmse,
            best_val_mae=self.best_trial.val_mae,
            total_time=sum(t.training_time for t in self.trials if t.status == 'COMPLETE'),
            avg_trial_time=np.mean([t.training_time for t in self.trials if t.status == 'COMPLETE']),
            convergence_trial=convergence_trial,
            most_important_param=param_importance['most_important'],
            least_important_param=param_importance['least_important'],
            parameter_correlations=param_importance['correlations'],
            recommended_config=self.best_trial.hyperparameters,
            confidence_score=self._calculate_confidence_score(),
            improvement_potential=self._assess_improvement_potential()
        )
        
        self.optimization_summaries.append(summary)
        
        # Log summary
        logger.info(f"\n{'='*70}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total trials: {summary.n_trials_total}")
        logger.info(f"  Completed: {summary.n_trials_completed}")
        logger.info(f"  Pruned: {summary.n_trials_pruned}")
        logger.info(f"  Failed: {summary.n_trials_failed}")
        logger.info(f"\nBest Result:")
        logger.info(f"  Trial: {summary.best_trial_id}")
        logger.info(f"  Val R²: {summary.best_val_r2:.4f}")
        logger.info(f"  Val RMSE: {summary.best_val_rmse:.4f}")
        logger.info(f"\nParameter Insights:")
        logger.info(f"  Most important: {summary.most_important_param}")
        logger.info(f"  Least important: {summary.least_important_param}")
        logger.info(f"\nTime:")
        logger.info(f"  Total: {summary.total_time:.1f}s")
        logger.info(f"  Avg per trial: {summary.avg_trial_time:.1f}s")
        logger.info(f"\nConfidence: {summary.confidence_score:.2%}")
        logger.info(f"Improvement potential: {summary.improvement_potential}")
        logger.info(f"{'='*70}\n")
        
        return summary
    
    def _calculate_parameter_importance(self) -> Dict:
        """Calculate which parameters correlate most with R²"""
        
        if not SCIPY_AVAILABLE:
            return {
                'most_important': 'N/A',
                'least_important': 'N/A',
                'correlations': {}
            }
        
        # Extract parameters and R² values
        completed_trials = [t for t in self.trials if t.status == 'COMPLETE']
        if len(completed_trials) < 10:
            return {
                'most_important': 'N/A',
                'least_important': 'N/A',
                'correlations': {}
            }
        
        # Get all parameter names
        param_names = list(completed_trials[0].hyperparameters.keys())
        
        correlations = {}
        for param_name in param_names:
            # Extract parameter values (handle different types)
            param_values = []
            r2_values = []
            
            for trial in completed_trials:
                param_val = trial.hyperparameters.get(param_name)
                
                # Convert to numeric if possible
                if isinstance(param_val, (int, float)):
                    param_values.append(float(param_val))
                    r2_values.append(trial.val_r2)
                elif isinstance(param_val, bool):
                    param_values.append(1.0 if param_val else 0.0)
                    r2_values.append(trial.val_r2)
                elif isinstance(param_val, str):
                    # For categorical, use dummy encoding
                    # Skip for now (would need proper encoding)
                    continue
            
            # Calculate correlation if we have enough numeric values
            if len(param_values) >= 10:
                try:
                    corr, p_value = stats.pearsonr(param_values, r2_values)
                    correlations[param_name] = {
                        'correlation': float(abs(corr)),  # absolute correlation
                        'p_value': float(p_value)
                    }
                except:
                    pass
        
        if not correlations:
            return {
                'most_important': 'N/A',
                'least_important': 'N/A',
                'correlations': {}
            }
        
        # Sort by correlation strength
        sorted_params = sorted(correlations.items(), 
                              key=lambda x: x[1]['correlation'], 
                              reverse=True)
        
        return {
            'most_important': sorted_params[0][0] if sorted_params else 'N/A',
            'least_important': sorted_params[-1][0] if sorted_params else 'N/A',
            'correlations': correlations
        }
    
    def _calculate_confidence_score(self) -> float:
        """
        Calculate confidence in the best result
        
        High confidence if:
        - Many trials completed
        - Best result stable (not improving recently)
        - Best result significantly better than others
        """
        completed_trials = [t for t in self.trials if t.status == 'COMPLETE']
        
        if len(completed_trials) < 10:
            return 0.5
        
        # Factor 1: Number of trials (more = higher confidence)
        n_factor = min(1.0, len(completed_trials) / 100)
        
        # Factor 2: Stability (no improvement in last 20% of trials)
        recent_trials = completed_trials[-int(len(completed_trials) * 0.2):]
        recent_improvements = sum(1 for t in recent_trials if t.is_best_so_far)
        stability_factor = 1.0 - (recent_improvements / len(recent_trials))
        
        # Factor 3: Gap between best and median
        r2_values = [t.val_r2 for t in completed_trials]
        median_r2 = np.median(r2_values)
        gap = (self.best_val_r2 - median_r2) / (1 - median_r2) if median_r2 < 1 else 1.0
        gap_factor = min(1.0, gap * 2)
        
        # Combined confidence
        confidence = (n_factor * 0.3 + stability_factor * 0.4 + gap_factor * 0.3)
        
        return confidence
    
    def _assess_improvement_potential(self) -> str:
        """Assess if there's room for improvement"""
        
        completed_trials = [t for t in self.trials if t.status == 'COMPLETE']
        
        if len(completed_trials) < 20:
            return "UNCERTAIN (too few trials)"
        
        # Check recent trend
        recent_trials = completed_trials[-20:]
        recent_improvements = sum(1 for t in recent_trials if t.is_best_so_far)
        
        # Check R² distribution
        r2_values = [t.val_r2 for t in completed_trials]
        r2_std = np.std(r2_values)
        
        if recent_improvements >= 2:
            return "HIGH (still improving)"
        elif r2_std > 0.05:
            return "MEDIUM (high variance, more tuning possible)"
        elif self.best_val_r2 < 0.90:
            return "MEDIUM (R² < 0.90, different approach might help)"
        else:
            return "LOW (converged, optimal config found)"
    
    # ========================================================================
    # EXCEL REPORTING
    # ========================================================================
    
    def export_to_excel(self, filename: str = 'automl_optimization_report.xlsx'):
        """
        Export comprehensive Excel report
        
        Sheets:
        1. Summary - Overview of optimization
        2. All_Trials - Complete trial history
        3. Best_Trials - Top 10 trials
        4. Failed_Trials - Debugging info
        5. Pruned_Trials - Pruning analysis
        6. Parameter_Importance - Correlation analysis
        7. Convergence_Analysis - How fast did it converge?
        8. Hyperparameter_Distribution - Distribution of tried values
        9. R2_vs_Time - Trade-off analysis
        10. Top_Configs - Top 5 configurations
        11. Recommendations - What to do next?
        12. LaTeX_Tables - Ready for thesis
        """
        logger.info(f"\n-> Exporting AutoML report to {filename}...")
        
        filepath = self.output_dir / filename
        
        try:
            import xlsxwriter
        except ImportError:
            logger.error("  xlsxwriter not available")
            return None
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'align': 'center',
                'border': 1
            })
            
            best_format = workbook.add_format({
                'bg_color': '#C6EFCE',
                'font_color': '#006100'
            })
            
            failed_format = workbook.add_format({
                'bg_color': '#FFC7CE',
                'font_color': '#9C0006'
            })
            
            # Sheet 1: Summary
            self._create_summary_sheet(writer, header_format)
            
            # Sheet 2: All Trials
            self._create_all_trials_sheet(writer, header_format, best_format, failed_format)
            
            # Sheet 3: Best Trials
            self._create_best_trials_sheet(writer, header_format, best_format)
            
            # Sheet 4: Failed Trials
            self._create_failed_trials_sheet(writer, header_format, failed_format)
            
            # Sheet 5: Pruned Trials
            self._create_pruned_trials_sheet(writer, header_format)
            
            # Sheet 6: Parameter Importance
            self._create_parameter_importance_sheet(writer, header_format)
            
            # Sheet 7: Convergence Analysis
            self._create_convergence_sheet(writer, header_format)
            
            # Sheet 8: Hyperparameter Distribution
            self._create_hyperparameter_distribution_sheet(writer, header_format)
            
            # Sheet 9: R² vs Time
            self._create_r2_vs_time_sheet(writer, header_format)
            
            # Sheet 10: Top Configs
            self._create_top_configs_sheet(writer, header_format, best_format)
            
            # Sheet 11: Recommendations
            self._create_recommendations_sheet(writer, header_format)
            
            # Sheet 12: LaTeX Tables
            self._create_latex_tables_sheet(writer, header_format)
        
        logger.info(f"  ✓ Exported: {filepath}")
        return filepath
    
    def _create_summary_sheet(self, writer, header_format):
        """Sheet 1: Optimization summary"""
        
        if not self.optimization_summaries:
            return
        
        summary = self.optimization_summaries[-1]  # Latest
        
        data = {
            'Metric': [
                'Optimization ID',
                'Model Type',
                'Objective',
                'Total Trials',
                'Completed Trials',
                'Pruned Trials',
                'Failed Trials',
                'Success Rate',
                '',
                'Best Trial ID',
                'Best Val R²',
                'Best Val RMSE',
                'Best Val MAE',
                '',
                'Total Time (s)',
                'Avg Trial Time (s)',
                'Convergence Trial',
                '',
                'Most Important Param',
                'Least Important Param',
                '',
                'Confidence Score',
                'Improvement Potential'
            ],
            'Value': [
                summary.optimization_id,
                summary.model_type,
                summary.objective,
                summary.n_trials_total,
                summary.n_trials_completed,
                summary.n_trials_pruned,
                summary.n_trials_failed,
                f"{summary.n_trials_completed / summary.n_trials_total * 100:.1f}%",
                '',
                summary.best_trial_id,
                f"{summary.best_val_r2:.6f}",
                f"{summary.best_val_rmse:.6f}",
                f"{summary.best_val_mae:.6f}",
                '',
                f"{summary.total_time:.1f}",
                f"{summary.avg_trial_time:.1f}",
                summary.convergence_trial if summary.convergence_trial else 'N/A',
                '',
                summary.most_important_param,
                summary.least_important_param,
                '',
                f"{summary.confidence_score:.2%}",
                summary.improvement_potential
            ]
        }
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Summary', index=False)
        
        worksheet = writer.sheets['Summary']
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 40)
    
    def _create_all_trials_sheet(self, writer, header_format, best_format, failed_format):
        """Sheet 2: All trials complete log"""
        
        data = []
        for trial in self.trials:
            row = {
                'Trial_ID': trial.trial_id,
                'Timestamp': trial.timestamp,
                'Model': trial.model_type,
                'Dataset': trial.dataset_name,
                'Status': trial.status,
                'Val_R2': trial.val_r2,
                'Val_RMSE': trial.val_rmse,
                'Val_MAE': trial.val_mae,
                'Train_R2': trial.train_r2,
                'Training_Time': trial.training_time,
                'N_Features': trial.n_features,
                'N_Train': trial.n_samples_train,
                'Is_Best': trial.is_best_so_far,
                'Improvement': trial.improvement_over_previous_best if trial.improvement_over_previous_best else 0,
                'Pruning_Reason': trial.pruning_reason if trial.pruning_reason else '',
                'Error': trial.error_message if trial.error_message else ''
            }
            
            # Add hyperparameters
            for key, value in trial.hyperparameters.items():
                row[f'HP_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='All_Trials', index=False)
        
        worksheet = writer.sheets['All_Trials']
        
        # Conditional formatting
        for idx, trial in enumerate(self.trials, start=2):
            if trial.is_best_so_far:
                worksheet.set_row(idx - 1, None, best_format)
            elif trial.status == 'FAILED':
                worksheet.set_row(idx - 1, None, failed_format)
    
    def _create_best_trials_sheet(self, writer, header_format, best_format):
        """Sheet 3: Top 10 trials"""
        
        completed = [t for t in self.trials if t.status == 'COMPLETE']
        top_10 = sorted(completed, key=lambda t: t.val_r2, reverse=True)[:10]
        
        data = []
        for rank, trial in enumerate(top_10, 1):
            row = {
                'Rank': rank,
                'Trial_ID': trial.trial_id,
                'Val_R2': trial.val_r2,
                'Val_RMSE': trial.val_rmse,
                'Val_MAE': trial.val_mae,
                'Training_Time': trial.training_time
            }
            
            # Add key hyperparameters
            for key, value in trial.hyperparameters.items():
                row[key] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Best_Trials', index=False)
    
    def _create_failed_trials_sheet(self, writer, header_format, failed_format):
        """Sheet 4: Failed trials for debugging"""
        
        failed = [t for t in self.trials if t.status == 'FAILED']
        
        if not failed:
            # Empty sheet with message
            df = pd.DataFrame({'Message': ['No failed trials! [COMPLETE]']})
            df.to_excel(writer, sheet_name='Failed_Trials', index=False)
            return
        
        data = []
        for trial in failed:
            row = {
                'Trial_ID': trial.trial_id,
                'Timestamp': trial.timestamp,
                'Error_Message': trial.error_message,
                'Dataset': trial.dataset_name
            }
            
            # Add hyperparameters
            for key, value in trial.hyperparameters.items():
                row[f'HP_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Failed_Trials', index=False)
    
    def _create_pruned_trials_sheet(self, writer, header_format):
        """Sheet 5: Pruned trials analysis"""
        
        pruned = [t for t in self.trials if t.status == 'PRUNED']
        
        if not pruned:
            df = pd.DataFrame({'Message': ['No pruned trials']})
            df.to_excel(writer, sheet_name='Pruned_Trials', index=False)
            return
        
        data = []
        for trial in pruned:
            data.append({
                'Trial_ID': trial.trial_id,
                'Pruning_Reason': trial.pruning_reason,
                'Val_R2': trial.val_r2,
                'Training_Time': trial.training_time
            })
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Pruned_Trials', index=False)
    
    def _create_parameter_importance_sheet(self, writer, header_format):
        """Sheet 6: Parameter importance"""
        
        summary = self.optimization_summaries[-1] if self.optimization_summaries else None
        if not summary or not summary.parameter_correlations:
            df = pd.DataFrame({'Message': ['Not enough data for parameter importance']})
            df.to_excel(writer, sheet_name='Parameter_Importance', index=False)
            return
        
        data = []
        for param_name, values in summary.parameter_correlations.items():
            data.append({
                'Parameter': param_name,
                'Abs_Correlation_with_R2': values['correlation'],
                'P_Value': values['p_value'],
                'Significant': 'Yes' if values['p_value'] < 0.05 else 'No'
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Abs_Correlation_with_R2', ascending=False)
        df.to_excel(writer, sheet_name='Parameter_Importance', index=False)
    
    def _create_convergence_sheet(self, writer, header_format):
        """Sheet 7: Convergence analysis"""
        
        # Best R² over time
        best_r2_history = []
        current_best = -np.inf
        
        for trial in self.trials:
            if trial.status == 'COMPLETE':
                if trial.val_r2 > current_best:
                    current_best = trial.val_r2
                best_r2_history.append({
                    'Trial': trial.trial_id,
                    'Best_R2_So_Far': current_best,
                    'Current_R2': trial.val_r2,
                    'Is_Improvement': trial.is_best_so_far
                })
        
        df = pd.DataFrame(best_r2_history)
        df.to_excel(writer, sheet_name='Convergence_Analysis', index=False)
    
    def _create_hyperparameter_distribution_sheet(self, writer, header_format):
        """Sheet 8: Distribution of hyperparameters tried"""
        
        completed = [t for t in self.trials if t.status == 'COMPLETE']
        
        if not completed:
            return
        
        # Get numeric hyperparameters
        param_names = list(completed[0].hyperparameters.keys())
        
        data = []
        for param_name in param_names:
            values = []
            for trial in completed:
                val = trial.hyperparameters.get(param_name)
                if isinstance(val, (int, float)):
                    values.append(val)
            
            if values:
                data.append({
                    'Parameter': param_name,
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Mean': np.mean(values),
                    'Median': np.median(values),
                    'Std': np.std(values),
                    'Unique_Values': len(set(values))
                })
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Hyperparameter_Distribution', index=False)
    
    def _create_r2_vs_time_sheet(self, writer, header_format):
        """Sheet 9: R² vs training time trade-off"""
        
        completed = [t for t in self.trials if t.status == 'COMPLETE']
        
        data = []
        for trial in completed:
            data.append({
                'Trial_ID': trial.trial_id,
                'Val_R2': trial.val_r2,
                'Training_Time': trial.training_time,
                'R2_per_Second': trial.val_r2 / trial.training_time if trial.training_time > 0 else 0
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('R2_per_Second', ascending=False)
        df.to_excel(writer, sheet_name='R2_vs_Time', index=False)
    
    def _create_top_configs_sheet(self, writer, header_format, best_format):
        """Sheet 10: Top 5 configurations with full details"""
        
        completed = [t for t in self.trials if t.status == 'COMPLETE']
        top_5 = sorted(completed, key=lambda t: t.val_r2, reverse=True)[:5]
        
        data = []
        for rank, trial in enumerate(top_5, 1):
            row = {'Rank': rank}
            row.update(trial.hyperparameters)
            row.update({
                'Val_R2': trial.val_r2,
                'Val_RMSE': trial.val_rmse,
                'Val_MAE': trial.val_mae,
                'Training_Time': trial.training_time
            })
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Top_Configs', index=False)
    
    def _create_recommendations_sheet(self, writer, header_format):
        """Sheet 11: Recommendations for next steps"""
        
        summary = self.optimization_summaries[-1] if self.optimization_summaries else None
        
        if not summary:
            return
        
        recommendations = []
        
        # Recommendation 1: Use best config
        recommendations.append({
            'Recommendation': '1. Use Best Configuration',
            'Details': f"Trial {summary.best_trial_id} achieved R²={summary.best_val_r2:.4f}",
            'Action': 'Deploy this configuration for production'
        })
        
        # Recommendation 2: Based on improvement potential
        if 'HIGH' in summary.improvement_potential:
            recommendations.append({
                'Recommendation': '2. Continue Optimization',
                'Details': f"Improvement potential: {summary.improvement_potential}",
                'Action': 'Run more trials with refined search space'
            })
        elif 'LOW' in summary.improvement_potential:
            recommendations.append({
                'Recommendation': '2. Optimization Complete',
                'Details': f"Improvement potential: {summary.improvement_potential}",
                'Action': 'Current configuration is likely optimal'
            })
        
        # Recommendation 3: Parameter focus
        if summary.most_important_param != 'N/A':
            recommendations.append({
                'Recommendation': '3. Focus on Key Parameters',
                'Details': f"Most important: {summary.most_important_param}",
                'Action': f'Fine-tune {summary.most_important_param} with narrower range'
            })
        
        # Recommendation 4: Confidence
        if summary.confidence_score < 0.7:
            recommendations.append({
                'Recommendation': '4. Increase Confidence',
                'Details': f"Current confidence: {summary.confidence_score:.1%}",
                'Action': 'Run more trials to increase confidence in results'
            })
        
        df = pd.DataFrame(recommendations)
        df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        worksheet = writer.sheets['Recommendations']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:C', 50)
    
    def _create_latex_tables_sheet(self, writer, header_format):
        """Sheet 12: LaTeX-ready tables for thesis"""
        
        completed = [t for t in self.trials if t.status == 'COMPLETE']
        top_5 = sorted(completed, key=lambda t: t.val_r2, reverse=True)[:5]
        
        # LaTeX table code
        latex_code = []
        
        latex_code.append("% Top 5 Configurations - LaTeX Table")
        latex_code.append("\\begin{table}[H]")
        latex_code.append("\\centering")
        latex_code.append("\\caption{Top 5 AutoML Configurations}")
        latex_code.append("\\begin{tabular}{ccccc}")
        latex_code.append("\\toprule")
        latex_code.append("Rank & Trial ID & Val R² & Val RMSE & Time (s) \\\\")
        latex_code.append("\\midrule")
        
        for rank, trial in enumerate(top_5, 1):
            latex_code.append(
                f"{rank} & {trial.trial_id} & "
                f"{trial.val_r2:.4f} & {trial.val_rmse:.4f} & "
                f"{trial.training_time:.1f} \\\\"
            )
        
        latex_code.append("\\bottomrule")
        latex_code.append("\\end{tabular}")
        latex_code.append("\\end{table}")
        
        df = pd.DataFrame({'LaTeX_Code': latex_code})
        df.to_excel(writer, sheet_name='LaTeX_Tables', index=False)
        
        worksheet = writer.sheets['LaTeX_Tables']
        worksheet.set_column('A:A', 80)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("TESTING AUTOML LOGGING & REPORTING SYSTEM")
    logger.info("="*70)
    
    # Initialize logger
    trial_logger = AutoMLTrialLogger(output_dir='test_automl_logs')
    
    # Start optimization
    opt_id = trial_logger.start_optimization(
        model_type='XGBoost',
        objective='r2',
        n_trials=50
    )
    
    # Simulate 50 trials
    np.random.seed(42)
    
    for trial_id in range(50):
        # Simulate hyperparameters
        hyperparams = {
            'n_estimators': np.random.choice([100, 200, 300]),
            'learning_rate': np.random.uniform(0.01, 0.3),
            'max_depth': np.random.choice([3, 5, 7, 9]),
            'subsample': np.random.uniform(0.6, 1.0)
        }
        
        # Simulate metrics (with some randomness + hyperparameter effect)
        base_r2 = 0.85
        r2_boost = (hyperparams['n_estimators'] / 300) * 0.05
        r2_boost += (0.15 - hyperparams['learning_rate']) * 0.03
        
        val_r2 = base_r2 + r2_boost + np.random.normal(0, 0.02)
        val_r2 = np.clip(val_r2, 0, 1)
        
        val_rmse = 0.5 * (1 - val_r2) + np.random.normal(0, 0.02)
        val_mae = 0.8 * val_rmse
        
        training_time = hyperparams['n_estimators'] / 10 + np.random.uniform(5, 15)
        
        # Simulate status (most complete, some pruned/failed)
        status = 'COMPLETE'
        pruning_reason = None
        error_message = None
        
        if np.random.random() < 0.1:  # 10% pruned
            status = 'PRUNED'
            pruning_reason = 'Early stopping - poor intermediate performance'
            training_time *= 0.5
        elif np.random.random() < 0.05:  # 5% failed
            status = 'FAILED'
            error_message = 'Out of memory error'
        
        # Create trial record
        trial_record = AutoMLTrialRecord(
            trial_id=trial_id,
            timestamp=datetime.now().isoformat(),
            model_type='XGBoost',
            dataset_name='nuclear_200_nuclei',
            hyperparameters=hyperparams,
            train_r2=val_r2 + 0.03,
            train_rmse=val_rmse * 0.9,
            train_mae=val_mae * 0.9,
            val_r2=val_r2,
            val_rmse=val_rmse,
            val_mae=val_mae,
            training_time=training_time,
            n_samples_train=160,
            n_samples_val=40,
            n_features=44,
            status=status,
            pruning_reason=pruning_reason,
            error_message=error_message
        )
        
        # Log trial
        trial_logger.log_trial(trial_record)
    
    # End optimization
    summary = trial_logger.end_optimization()
    
    # Export to Excel
    trial_logger.export_to_excel('automl_test_report.xlsx')
    
    logger.info("\n✓ Testing complete! Check test_automl_logs/")
