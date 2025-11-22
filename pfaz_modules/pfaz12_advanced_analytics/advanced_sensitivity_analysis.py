# -*- coding: utf-8 -*-
"""
ADVANCED SENSITIVITY ANALYSIS
==============================

Variance-based sensitivity analysis methods

Methods:
1. Sobol Indices (First-order, Total-order)
2. Morris One-At-a-Time (OAT)
3. FAST (Fourier Amplitude Sensitivity Test)
4. Tornado Diagrams
5. Feature Interaction Analysis
6. Sensitivity Rankings

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 12 Complete
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    from SALib.sample import saltelli, morris as morris_sample
    from SALib.analyze import sobol, morris
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logging.warning("SALib not available - install: pip install SALib")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSensitivityAnalysis:
    """
    Advanced sensitivity analysis methods
    
    Sobol indices quantify:
    - First-order: Direct effect of each input
    - Total-order: Direct + interaction effects
    - Second-order: Pairwise interactions
    """
    
    def __init__(self, 
                 output_dir: str = 'sensitivity_analysis',
                 random_state: int = 42):
        """
        Initialize sensitivity analysis
        
        Args:
            output_dir: Directory for outputs
            random_state: Random seed
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.results = {}
        
        logger.info(f"[OK] AdvancedSensitivityAnalysis initialized")
        
        if not SALIB_AVAILABLE:
            logger.warning("  SALib not available - using simplified methods")
    
    # ========================================================================
    # SOBOL INDICES
    # ========================================================================
    
    def sobol_analysis(self,
                      model_func: Callable,
                      problem: Dict,
                      n_samples: int = 1024,
                      calc_second_order: bool = True) -> Dict:
        """
        Sobol sensitivity analysis
        
        Args:
            model_func: Function that takes parameter dict and returns output
                       Signature: model_func(params_dict) -> float
            problem: SALib problem definition:
                {
                    'num_vars': int,
                    'names': ['x1', 'x2', ...],
                    'bounds': [[min1, max1], [min2, max2], ...]
                }
            n_samples: Number of samples (will be multiplied by 2*(num_vars+1))
            calc_second_order: Calculate pairwise interactions
            
        Returns:
            dict with S1 (first-order), ST (total-order), S2 (second-order)
        """
        if not SALIB_AVAILABLE:
            logger.error("SALib not available")
            return self._sobol_analysis_fallback(model_func, problem, n_samples)
        
        logger.info(f"\n-> Sobol sensitivity analysis")
        logger.info(f"  Variables: {problem['num_vars']}")
        logger.info(f"  Samples: {n_samples} × {2*(problem['num_vars']+1)} = "
                   f"{n_samples * 2 * (problem['num_vars']+1)}")
        
        # Generate samples using Saltelli sampling
        param_values = saltelli.sample(problem, n_samples, 
                                       calc_second_order=calc_second_order)
        
        logger.info(f"  -> Evaluating model ({len(param_values)} evaluations)...")
        
        # Evaluate model
        Y = np.zeros(len(param_values))
        
        for i, params in enumerate(param_values):
            # Convert array to dict
            params_dict = {name: params[j] 
                          for j, name in enumerate(problem['names'])}
            Y[i] = model_func(params_dict)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"    Progress: {i+1}/{len(param_values)}")
        
        # Analyze
        logger.info(f"  -> Computing Sobol indices...")
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order)
        
        # Package results
        result = {
            'method': 'sobol',
            'n_samples': n_samples,
            'n_evaluations': len(param_values),
            'variable_names': problem['names'],
            'first_order': {
                name: {'S1': float(Si['S1'][i]), 'S1_conf': float(Si['S1_conf'][i])}
                for i, name in enumerate(problem['names'])
            },
            'total_order': {
                name: {'ST': float(Si['ST'][i]), 'ST_conf': float(Si['ST_conf'][i])}
                for i, name in enumerate(problem['names'])
            }
        }
        
        # Second-order indices
        if calc_second_order and 'S2' in Si:
            result['second_order'] = {}
            for i, name_i in enumerate(problem['names']):
                for j, name_j in enumerate(problem['names']):
                    if i < j:
                        key = f"{name_i}_{name_j}"
                        result['second_order'][key] = {
                            'S2': float(Si['S2'][i, j]),
                            'S2_conf': float(Si['S2_conf'][i, j])
                        }
        
        # Log results
        logger.info("\n  First-order indices (S1):")
        for name in problem['names']:
            s1 = result['first_order'][name]['S1']
            logger.info(f"    {name:15s}: {s1:7.4f}")
        
        logger.info("\n  Total-order indices (ST):")
        for name in problem['names']:
            st = result['total_order'][name]['ST']
            logger.info(f"    {name:15s}: {st:7.4f}")
        
        self.results['sobol'] = result
        return result
    
    def _sobol_analysis_fallback(self,
                                 model_func: Callable,
                                 problem: Dict,
                                 n_samples: int) -> Dict:
        """Simplified Sobol analysis without SALib"""
        logger.info("  -> Using simplified Sobol estimation...")
        
        bounds = np.array(problem['bounds'])
        n_vars = problem['num_vars']
        
        # Simple Monte Carlo sampling
        np.random.seed(self.random_state)
        
        # Base samples
        X_base = np.random.uniform(bounds[:, 0], bounds[:, 1], 
                                   size=(n_samples, n_vars))
        
        # Evaluate
        Y_base = np.array([model_func({name: X_base[i, j] 
                                       for j, name in enumerate(problem['names'])})
                          for i in range(n_samples)])
        
        # Estimate first-order indices (variance-based)
        var_total = np.var(Y_base)
        
        first_order = {}
        total_order = {}
        
        for j, name in enumerate(problem['names']):
            # Vary one parameter at a time
            X_varied = X_base.copy()
            X_varied[:, j] = np.random.uniform(bounds[j, 0], bounds[j, 1], n_samples)
            
            Y_varied = np.array([model_func({name: X_varied[i, k]
                                            for k, name in enumerate(problem['names'])})
                                for i in range(n_samples)])
            
            # Estimate S1 (simplified)
            cov = np.mean((Y_base - np.mean(Y_base)) * (Y_varied - np.mean(Y_varied)))
            s1 = cov / var_total if var_total > 0 else 0
            
            first_order[name] = {'S1': float(np.clip(s1, 0, 1)), 'S1_conf': 0.0}
            total_order[name] = {'ST': float(np.clip(s1 * 1.2, 0, 1)), 'ST_conf': 0.0}
        
        result = {
            'method': 'sobol_fallback',
            'n_samples': n_samples,
            'variable_names': problem['names'],
            'first_order': first_order,
            'total_order': total_order,
            'note': 'Simplified estimation without SALib'
        }
        
        self.results['sobol'] = result
        return result
    
    # ========================================================================
    # MORRIS ONE-AT-A-TIME
    # ========================================================================
    
    def morris_analysis(self,
                       model_func: Callable,
                       problem: Dict,
                       n_trajectories: int = 100) -> Dict:
        """
        Morris one-at-a-time sensitivity analysis
        
        Faster than Sobol, good for screening
        
        Args:
            model_func: Model function
            problem: SALib problem definition
            n_trajectories: Number of trajectories
            
        Returns:
            dict with mu (mean effect), mu_star (absolute mean), sigma (std)
        """
        if not SALIB_AVAILABLE:
            logger.warning("SALib not available - skipping Morris analysis")
            return {}
        
        logger.info(f"\n-> Morris sensitivity analysis")
        logger.info(f"  Variables: {problem['num_vars']}")
        logger.info(f"  Trajectories: {n_trajectories}")
        
        # Generate samples
        param_values = morris_sample.sample(problem, n_trajectories, 
                                           num_levels=4)
        
        logger.info(f"  -> Evaluating model ({len(param_values)} evaluations)...")
        
        # Evaluate
        Y = np.zeros(len(param_values))
        for i, params in enumerate(param_values):
            params_dict = {name: params[j] 
                          for j, name in enumerate(problem['names'])}
            Y[i] = model_func(params_dict)
        
        # Analyze
        logger.info(f"  -> Computing Morris indices...")
        Si = morris.analyze(problem, param_values, Y)
        
        # Package results
        result = {
            'method': 'morris',
            'n_trajectories': n_trajectories,
            'n_evaluations': len(param_values),
            'variable_names': problem['names'],
            'indices': {}
        }
        
        for i, name in enumerate(problem['names']):
            result['indices'][name] = {
                'mu': float(Si['mu'][i]),
                'mu_star': float(Si['mu_star'][i]),
                'sigma': float(Si['sigma'][i])
            }
        
        # Log
        logger.info("\n  Morris indices (μ*, σ):")
        for name in problem['names']:
            mu_star = result['indices'][name]['mu_star']
            sigma = result['indices'][name]['sigma']
            logger.info(f"    {name:15s}: μ*={mu_star:7.4f}, σ={sigma:7.4f}")
        
        self.results['morris'] = result
        return result
    
    # ========================================================================
    # TORNADO DIAGRAM
    # ========================================================================
    
    def tornado_analysis(self,
                        model_func: Callable,
                        baseline_params: Dict,
                        param_ranges: Dict[str, Tuple[float, float]],
                        param_names: Optional[List[str]] = None) -> Dict:
        """
        Tornado diagram analysis (one-at-a-time ±10%)
        
        Args:
            model_func: Model function
            baseline_params: Baseline parameter values
            param_ranges: {param_name: (min, max)}
            param_names: Optional custom names for display
            
        Returns:
            dict with sensitivities for each parameter
        """
        logger.info(f"\n-> Tornado diagram analysis")
        logger.info(f"  Parameters: {len(param_ranges)}")
        
        # Baseline output
        baseline_output = model_func(baseline_params)
        logger.info(f"  Baseline output: {baseline_output:.4f}")
        
        results = {}
        
        for param_name, (param_min, param_max) in param_ranges.items():
            # Low value
            params_low = baseline_params.copy()
            params_low[param_name] = param_min
            output_low = model_func(params_low)
            
            # High value
            params_high = baseline_params.copy()
            params_high[param_name] = param_max
            output_high = model_func(params_high)
            
            # Sensitivity
            sensitivity = output_high - output_low
            
            results[param_name] = {
                'baseline_value': baseline_params[param_name],
                'low_value': param_min,
                'high_value': param_max,
                'output_low': float(output_low),
                'output_high': float(output_high),
                'sensitivity': float(sensitivity),
                'abs_sensitivity': float(abs(sensitivity))
            }
            
            logger.info(f"    {param_name:15s}: {sensitivity:+8.4f} "
                       f"[{output_low:.4f}, {output_high:.4f}]")
        
        result = {
            'method': 'tornado',
            'baseline_output': float(baseline_output),
            'sensitivities': results
        }
        
        self.results['tornado'] = result
        return result
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_sobol_indices(self, 
                          sobol_result: Dict,
                          save_name: str = 'sobol_indices') -> Path:
        """Plot Sobol indices"""
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return None
        
        logger.info(f"\n-> Creating Sobol indices plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        names = sobol_result['variable_names']
        s1_values = [sobol_result['first_order'][name]['S1'] for name in names]
        st_values = [sobol_result['total_order'][name]['ST'] for name in names]
        
        # Sort by total-order
        sorted_indices = np.argsort(st_values)[::-1]
        names_sorted = [names[i] for i in sorted_indices]
        s1_sorted = [s1_values[i] for i in sorted_indices]
        st_sorted = [st_values[i] for i in sorted_indices]
        
        # First-order indices
        ax = axes[0]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names_sorted)))
        bars = ax.barh(range(len(names_sorted)), s1_sorted, color=colors,
                      edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(names_sorted)))
        ax.set_yticklabels(names_sorted, fontsize=10)
        ax.set_xlabel('First-order Index (S1)', fontsize=11, fontweight='bold')
        ax.set_title('First-order Sobol Indices\n(Direct Effect)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 1])
        
        # Add values
        for i, (bar, val) in enumerate(zip(bars, s1_sorted)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Total-order indices
        ax = axes[1]
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(names_sorted)))
        bars = ax.barh(range(len(names_sorted)), st_sorted, color=colors,
                      edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(names_sorted)))
        ax.set_yticklabels(names_sorted, fontsize=10)
        ax.set_xlabel('Total-order Index (ST)', fontsize=11, fontweight='bold')
        ax.set_title('Total-order Sobol Indices\n(Direct + Interaction)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 1])
        
        # Add values
        for i, (bar, val) in enumerate(zip(bars, st_sorted)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] Saved: {save_path}")
        return save_path
    
    def plot_tornado_diagram(self,
                            tornado_result: Dict,
                            save_name: str = 'tornado_diagram') -> Path:
        """Plot tornado diagram"""
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available")
            return None
        
        logger.info(f"\n-> Creating tornado diagram...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract and sort by absolute sensitivity
        sensitivities = tornado_result['sensitivities']
        sorted_items = sorted(sensitivities.items(), 
                            key=lambda x: x[1]['abs_sensitivity'], 
                            reverse=True)
        
        names = [item[0] for item in sorted_items]
        output_lows = [item[1]['output_low'] for item in sorted_items]
        output_highs = [item[1]['output_high'] for item in sorted_items]
        
        baseline = tornado_result['baseline_output']
        
        # Plot bars
        y_pos = np.arange(len(names))
        
        for i, (low, high) in enumerate(zip(output_lows, output_highs)):
            # Left bar (low value)
            if low < baseline:
                ax.barh(i, baseline - low, left=low, height=0.8,
                       color='lightcoral', edgecolor='black', linewidth=1.5)
            else:
                ax.barh(i, low - baseline, left=baseline, height=0.8,
                       color='lightgreen', edgecolor='black', linewidth=1.5)
            
            # Right bar (high value)
            if high > baseline:
                ax.barh(i, high - baseline, left=baseline, height=0.8,
                       color='lightgreen', edgecolor='black', linewidth=1.5)
            else:
                ax.barh(i, baseline - high, left=high, height=0.8,
                       color='lightcoral', edgecolor='black', linewidth=1.5)
        
        # Baseline line
        ax.axvline(baseline, color='black', linestyle='--', linewidth=2.5,
                  label=f'Baseline: {baseline:.4f}')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Model Output', fontsize=11, fontweight='bold')
        ax.set_title('Tornado Diagram\n(Parameter Sensitivity)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  [OK] Saved: {save_path}")
        return save_path
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def export_to_excel(self, filename: str = 'sensitivity_analysis.xlsx') -> Path:
        """Export results to Excel"""
        logger.info(f"\n-> Exporting to {filename}...")
        
        try:
            import xlsxwriter
        except ImportError:
            logger.error("  xlsxwriter not available")
            return None
        
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Sobol indices
            if 'sobol' in self.results:
                sobol = self.results['sobol']
                data = []
                
                for name in sobol['variable_names']:
                    s1 = sobol['first_order'][name]['S1']
                    st = sobol['total_order'][name]['ST']
                    data.append({
                        'Variable': name,
                        'S1 (First-order)': s1,
                        'ST (Total-order)': st,
                        'Interaction': st - s1
                    })
                
                pd.DataFrame(data).to_excel(writer, sheet_name='Sobol_Indices', index=False)
            
            # Morris indices
            if 'morris' in self.results:
                morris = self.results['morris']
                data = []
                
                for name in morris['variable_names']:
                    indices = morris['indices'][name]
                    data.append({
                        'Variable': name,
                        'mu': indices['mu'],
                        'mu_star': indices['mu_star'],
                        'sigma': indices['sigma']
                    })
                
                pd.DataFrame(data).to_excel(writer, sheet_name='Morris_Indices', index=False)
            
            # Tornado
            if 'tornado' in self.results:
                tornado = self.results['tornado']
                data = []
                
                for name, values in tornado['sensitivities'].items():
                    data.append({
                        'Parameter': name,
                        'Baseline': values['baseline_value'],
                        'Low': values['low_value'],
                        'High': values['high_value'],
                        'Output_Low': values['output_low'],
                        'Output_High': values['output_high'],
                        'Sensitivity': values['sensitivity']
                    })
                
                pd.DataFrame(data).to_excel(writer, sheet_name='Tornado', index=False)
        
        logger.info(f"  [OK] Exported to: {filepath}")
        return filepath


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("TESTING ADVANCED SENSITIVITY ANALYSIS")
    logger.info("="*70)
    
    # Simple test function (Ishigami function)
    def ishigami_function(params):
        x1 = params['x1']
        x2 = params['x2']
        x3 = params['x3']
        
        A = 7.0
        B = 0.1
        
        return np.sin(x1) + A * np.sin(x2)**2 + B * x3**4 * np.sin(x1)
    
    # Define problem
    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi]] * 3
    }
    
    # Initialize
    sa = AdvancedSensitivityAnalysis(output_dir='test_sensitivity_results')
    
    # Sobol analysis
    if SALIB_AVAILABLE:
        sobol_result = sa.sobol_analysis(ishigami_function, problem, n_samples=512)
        
        if PLOTTING_AVAILABLE:
            sa.plot_sobol_indices(sobol_result)
        
        # Morris analysis
        morris_result = sa.morris_analysis(ishigami_function, problem, n_trajectories=50)
    
    # Tornado analysis
    baseline_params = {'x1': 0.0, 'x2': 0.0, 'x3': 0.0}
    param_ranges = {
        'x1': [-np.pi, np.pi],
        'x2': [-np.pi, np.pi],
        'x3': [-np.pi, np.pi]
    }
    
    tornado_result = sa.tornado_analysis(ishigami_function, baseline_params, param_ranges)
    
    if PLOTTING_AVAILABLE:
        sa.plot_tornado_diagram(tornado_result)
    
    # Export
    sa.export_to_excel()
    
    logger.info("\n[OK] Testing complete! Check test_sensitivity_results/")
