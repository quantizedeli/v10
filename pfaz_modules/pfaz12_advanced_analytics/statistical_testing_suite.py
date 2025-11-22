# -*- coding: utf-8 -*-
"""
STATISTICAL TESTING SUITE
==========================

Comprehensive statistical testing for model comparison

Tests Included:
1. Paired t-test (compare two models)
2. One-way ANOVA (compare multiple models)
3. Wilcoxon signed-rank test (non-parametric alternative to t-test)
4. Friedman test (non-parametric alternative to ANOVA)
5. Post-hoc tests (Bonferroni, Tukey HSD)
6. Effect sizes (Cohen's d, eta-squared, Cliff's delta)
7. Statistical power analysis
8. Multiple comparison corrections

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 12 Complete
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available - some tests disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalTestingSuite:
    """
    Comprehensive statistical testing for model comparison
    
    Use cases:
    - Compare two models on same dataset
    - Compare multiple models across datasets
    - Validate statistical significance of improvements
    - Generate publication-ready test results
    """
    
    def __init__(self, output_dir: str = 'statistical_tests', alpha: float = 0.05):
        """
        Initialize statistical testing suite
        
        Args:
            output_dir: Directory for outputs
            alpha: Significance level (default 0.05)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        
        self.results = {}
        
        logger.info(f"[OK] StatisticalTestingSuite initialized (α={alpha})")
    
    # ========================================================================
    # PAIRED TESTS (Two Models)
    # ========================================================================
    
    def paired_t_test(self, 
                     scores_a: np.ndarray, 
                     scores_b: np.ndarray,
                     model_a_name: str = 'Model A',
                     model_b_name: str = 'Model B') -> Dict:
        """
        Paired t-test for comparing two models
        
        Use when: Same dataset, normally distributed differences
        
        Args:
            scores_a: Scores from model A (e.g., R² values)
            scores_b: Scores from model B on same data
            
        Returns:
            dict with statistic, p_value, significant, effect_size
        """
        logger.info(f"\n-> Running paired t-test: {model_a_name} vs {model_b_name}")
        
        # Validate inputs
        if len(scores_a) != len(scores_b):
            raise ValueError("Score arrays must have same length")
        
        # Calculate differences
        differences = scores_a - scores_b
        
        # Perform test
        statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Effect size (Cohen's d for paired samples)
        effect_size = self.cohens_d_paired(scores_a, scores_b)
        
        # Confidence interval for mean difference
        ci = stats.t.interval(1 - self.alpha, 
                             len(differences) - 1,
                             loc=np.mean(differences),
                             scale=stats.sem(differences))
        
        result = {
            'test': 'paired_t_test',
            'model_a': model_a_name,
            'model_b': model_b_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'mean_diff': float(np.mean(differences)),
            'effect_size_cohens_d': float(effect_size),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_samples': len(scores_a),
            'interpretation': self._interpret_effect_size(effect_size)
        }
        
        logger.info(f"  Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        logger.info(f"  Significant: {result['significant']}, Effect: {result['interpretation']}")
        
        self.results['paired_t_test'] = result
        return result
    
    def wilcoxon_test(self,
                     scores_a: np.ndarray,
                     scores_b: np.ndarray,
                     model_a_name: str = 'Model A',
                     model_b_name: str = 'Model B') -> Dict:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        
        Use when: Same dataset, but differences not normally distributed
        
        Args:
            scores_a: Scores from model A
            scores_b: Scores from model B
            
        Returns:
            dict with statistic, p_value, significant, effect_size
        """
        logger.info(f"\n-> Running Wilcoxon test: {model_a_name} vs {model_b_name}")
        
        # Perform test
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        
        # Effect size (Cliff's delta for non-parametric)
        effect_size = self.cliffs_delta(scores_a, scores_b)
        
        result = {
            'test': 'wilcoxon',
            'model_a': model_a_name,
            'model_b': model_b_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'median_diff': float(np.median(scores_a - scores_b)),
            'effect_size_cliffs_delta': float(effect_size),
            'n_samples': len(scores_a),
            'interpretation': self._interpret_cliffs_delta(effect_size)
        }
        
        logger.info(f"  Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        logger.info(f"  Significant: {result['significant']}, Effect: {result['interpretation']}")
        
        self.results['wilcoxon'] = result
        return result
    
    # ========================================================================
    # MULTIPLE COMPARISON TESTS
    # ========================================================================
    
    def one_way_anova(self,
                     scores_dict: Dict[str, np.ndarray]) -> Dict:
        """
        One-way ANOVA for comparing multiple models
        
        Use when: Comparing 3+ models, normally distributed scores
        
        Args:
            scores_dict: {'Model1': scores1, 'Model2': scores2, ...}
            
        Returns:
            dict with F_statistic, p_value, significant, eta_squared
        """
        logger.info(f"\n-> Running one-way ANOVA ({len(scores_dict)} models)")
        
        model_names = list(scores_dict.keys())
        scores_list = [scores_dict[name] for name in model_names]
        
        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*scores_list)
        
        # Effect size (eta-squared)
        eta_squared = self.eta_squared(scores_list)
        
        result = {
            'test': 'one_way_anova',
            'models': model_names,
            'n_models': len(model_names),
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'eta_squared': float(eta_squared),
            'interpretation': self._interpret_eta_squared(eta_squared)
        }
        
        logger.info(f"  F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")
        logger.info(f"  Significant: {result['significant']}, η²: {eta_squared:.4f}")
        
        self.results['one_way_anova'] = result
        
        # If significant, run post-hoc tests
        if result['significant'] and STATSMODELS_AVAILABLE:
            logger.info("  -> Running post-hoc Tukey HSD...")
            posthoc = self.tukey_hsd_posthoc(scores_dict)
            result['posthoc'] = posthoc
        
        return result
    
    def friedman_test(self,
                     scores_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Friedman test (non-parametric alternative to repeated measures ANOVA)
        
        Use when: Comparing 3+ models, non-normal distributions
        
        Args:
            scores_dict: {'Model1': scores1, 'Model2': scores2, ...}
            All score arrays must have same length (same datasets)
            
        Returns:
            dict with statistic, p_value, significant
        """
        logger.info(f"\n-> Running Friedman test ({len(scores_dict)} models)")
        
        model_names = list(scores_dict.keys())
        scores_list = [scores_dict[name] for name in model_names]
        
        # Validate same length
        lengths = [len(s) for s in scores_list]
        if len(set(lengths)) > 1:
            raise ValueError("All score arrays must have same length for Friedman test")
        
        # Perform test
        statistic, p_value = stats.friedmanchisquare(*scores_list)
        
        result = {
            'test': 'friedman',
            'models': model_names,
            'n_models': len(model_names),
            'n_samples': lengths[0],
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha
        }
        
        logger.info(f"  Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        logger.info(f"  Significant: {result['significant']}")
        
        self.results['friedman'] = result
        
        # If significant, run pairwise Wilcoxon with correction
        if result['significant']:
            logger.info("  -> Running pairwise Wilcoxon tests...")
            pairwise = self.pairwise_wilcoxon(scores_dict)
            result['pairwise'] = pairwise
        
        return result
    
    # ========================================================================
    # POST-HOC TESTS
    # ========================================================================
    
    def tukey_hsd_posthoc(self, scores_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Tukey HSD post-hoc test after ANOVA
        
        Args:
            scores_dict: {'Model1': scores1, 'Model2': scores2, ...}
            
        Returns:
            dict with pairwise comparisons
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("  statsmodels not available - skipping Tukey HSD")
            return {}
        
        # Prepare data for Tukey
        data = []
        groups = []
        for model_name, scores in scores_dict.items():
            data.extend(scores)
            groups.extend([model_name] * len(scores))
        
        # Run Tukey HSD
        tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=self.alpha)
        
        # Parse results
        comparisons = []
        for i in range(len(tukey.summary().data) - 1):  # Skip header
            row = tukey.summary().data[i + 1]
            comparisons.append({
                'group1': row[0],
                'group2': row[1],
                'meandiff': float(row[2]),
                'p_adj': float(row[3]),
                'ci_lower': float(row[4]),
                'ci_upper': float(row[5]),
                'significant': row[6] == 'True'
            })
        
        result = {
            'test': 'tukey_hsd',
            'n_comparisons': len(comparisons),
            'comparisons': comparisons
        }
        
        logger.info(f"    Tukey HSD: {len(comparisons)} pairwise comparisons")
        
        return result
    
    def pairwise_wilcoxon(self, 
                         scores_dict: Dict[str, np.ndarray],
                         correction: str = 'bonferroni') -> Dict:
        """
        Pairwise Wilcoxon tests with multiple comparison correction
        
        Args:
            scores_dict: {'Model1': scores1, 'Model2': scores2, ...}
            correction: 'bonferroni', 'holm', 'fdr_bh', etc.
            
        Returns:
            dict with pairwise comparisons
        """
        model_names = list(scores_dict.keys())
        comparisons = []
        p_values = []
        
        # All pairwise combinations
        for name_a, name_b in combinations(model_names, 2):
            scores_a = scores_dict[name_a]
            scores_b = scores_dict[name_b]
            
            statistic, p_value = stats.wilcoxon(scores_a, scores_b)
            
            comparisons.append({
                'model_a': name_a,
                'model_b': name_b,
                'statistic': float(statistic),
                'p_value': float(p_value)
            })
            p_values.append(p_value)
        
        # Apply correction
        if STATSMODELS_AVAILABLE:
            reject, p_adjusted, _, _ = multipletests(p_values, 
                                                     alpha=self.alpha,
                                                     method=correction)
            
            for i, comp in enumerate(comparisons):
                comp['p_adjusted'] = float(p_adjusted[i])
                comp['significant'] = bool(reject[i])
        else:
            # Manual Bonferroni
            alpha_bonf = self.alpha / len(comparisons)
            for i, comp in enumerate(comparisons):
                comp['p_adjusted'] = min(1.0, p_values[i] * len(comparisons))
                comp['significant'] = p_values[i] < alpha_bonf
        
        result = {
            'test': 'pairwise_wilcoxon',
            'correction': correction,
            'n_comparisons': len(comparisons),
            'comparisons': comparisons
        }
        
        logger.info(f"    Pairwise Wilcoxon: {len(comparisons)} comparisons ({correction})")
        
        return result
    
    # ========================================================================
    # EFFECT SIZES
    # ========================================================================
    
    def cohens_d_paired(self, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Cohen's d for paired samples"""
        differences = scores_a - scores_b
        return np.mean(differences) / np.std(differences, ddof=1)
    
    def cohens_d_independent(self, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Cohen's d for independent samples"""
        n_a, n_b = len(scores_a), len(scores_b)
        var_a, var_b = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        return (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
    
    def eta_squared(self, scores_list: List[np.ndarray]) -> float:
        """Eta-squared effect size for ANOVA"""
        all_scores = np.concatenate(scores_list)
        grand_mean = np.mean(all_scores)
        
        # Between-group sum of squares
        ss_between = sum([len(scores) * (np.mean(scores) - grand_mean)**2 
                         for scores in scores_list])
        
        # Total sum of squares
        ss_total = np.sum((all_scores - grand_mean)**2)
        
        return ss_between / ss_total if ss_total > 0 else 0.0
    
    def cliffs_delta(self, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Cliff's delta effect size (non-parametric)"""
        n_a, n_b = len(scores_a), len(scores_b)
        
        # Count dominances
        dominance = 0
        for a in scores_a:
            for b in scores_b:
                if a > b:
                    dominance += 1
                elif a < b:
                    dominance -= 1
        
        return dominance / (n_a * n_b)
    
    # ========================================================================
    # INTERPRETATION HELPERS
    # ========================================================================
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta-squared"""
        if eta < 0.01:
            return 'negligible'
        elif eta < 0.06:
            return 'small'
        elif eta < 0.14:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta"""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return 'negligible'
        elif abs_delta < 0.33:
            return 'small'
        elif abs_delta < 0.474:
            return 'medium'
        else:
            return 'large'
    
    # ========================================================================
    # COMPREHENSIVE COMPARISON
    # ========================================================================
    
    def compare_models_comprehensive(self,
                                    scores_dict: Dict[str, np.ndarray],
                                    paired: bool = True) -> Dict:
        """
        Comprehensive model comparison with all appropriate tests
        
        Args:
            scores_dict: {'Model1': scores1, 'Model2': scores2, ...}
            paired: Whether comparisons are paired (same datasets)
            
        Returns:
            dict with all test results
        """
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE MODEL COMPARISON")
        logger.info("="*70)
        
        results = {
            'n_models': len(scores_dict),
            'paired': paired,
            'alpha': self.alpha
        }
        
        model_names = list(scores_dict.keys())
        
        # Two models: paired or independent tests
        if len(scores_dict) == 2:
            scores_a = scores_dict[model_names[0]]
            scores_b = scores_dict[model_names[1]]
            
            if paired:
                results['parametric'] = self.paired_t_test(
                    scores_a, scores_b, model_names[0], model_names[1]
                )
                results['non_parametric'] = self.wilcoxon_test(
                    scores_a, scores_b, model_names[0], model_names[1]
                )
            else:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(scores_a, scores_b)
                effect_size = self.cohens_d_independent(scores_a, scores_b)
                
                results['parametric'] = {
                    'test': 'independent_t_test',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < self.alpha,
                    'effect_size': float(effect_size)
                }
                
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(scores_a, scores_b)
                
                results['non_parametric'] = {
                    'test': 'mann_whitney_u',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < self.alpha
                }
        
        # Multiple models: ANOVA or Friedman
        else:
            results['parametric'] = self.one_way_anova(scores_dict)
            
            if paired:
                results['non_parametric'] = self.friedman_test(scores_dict)
            else:
                # Kruskal-Wallis (non-parametric ANOVA for independent samples)
                scores_list = [scores_dict[name] for name in model_names]
                statistic, p_value = stats.kruskal(*scores_list)
                
                results['non_parametric'] = {
                    'test': 'kruskal_wallis',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < self.alpha
                }
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*70)
        logger.info(f"Parametric test: {results['parametric']['test']}")
        logger.info(f"  p-value: {results['parametric']['p_value']:.4f}")
        logger.info(f"  Significant: {results['parametric']['significant']}")
        logger.info(f"\nNon-parametric test: {results['non_parametric']['test']}")
        logger.info(f"  p-value: {results['non_parametric']['p_value']:.4f}")
        logger.info(f"  Significant: {results['non_parametric']['significant']}")
        
        self.results['comprehensive'] = results
        return results
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def export_to_excel(self, filename: str = 'statistical_tests.xlsx') -> Path:
        """Export all test results to Excel"""
        logger.info(f"\n-> Exporting results to {filename}...")
        
        try:
            import xlsxwriter
        except ImportError:
            logger.error("  xlsxwriter not available")
            return None
        
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = []
            for test_name, result in self.results.items():
                if isinstance(result, dict) and 'p_value' in result:
                    summary_data.append({
                        'Test': result.get('test', test_name),
                        'P-Value': result['p_value'],
                        'Significant': result['significant'],
                        'Effect_Size': result.get('effect_size_cohens_d', 
                                                  result.get('eta_squared', 'N/A'))
                    })
            
            if summary_data:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed sheets for each test
            for test_name, result in self.results.items():
                if isinstance(result, dict):
                    # Flatten dict for Excel
                    flat_result = self._flatten_dict(result)
                    df = pd.DataFrame([flat_result])
                    
                    # Truncate sheet name to 31 chars (Excel limit)
                    sheet_name = test_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"  [OK] Exported to: {filepath}")
        return filepath
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("TESTING STATISTICAL TESTING SUITE")
    logger.info("="*70)
    
    # Generate sample data
    np.random.seed(42)
    
    # Scenario 1: Compare two models (paired)
    model_a_scores = np.random.normal(0.90, 0.05, 30)
    model_b_scores = model_a_scores + np.random.normal(0.02, 0.03, 30)  # Slight improvement
    
    suite = StatisticalTestingSuite(output_dir='test_statistical_results')
    
    # Paired t-test
    suite.paired_t_test(model_a_scores, model_b_scores, 'Model_A', 'Model_B')
    
    # Wilcoxon test
    suite.wilcoxon_test(model_a_scores, model_b_scores, 'Model_A', 'Model_B')
    
    # Scenario 2: Compare multiple models
    scores_dict = {
        'RF': np.random.normal(0.85, 0.05, 30),
        'GBM': np.random.normal(0.88, 0.04, 30),
        'XGBoost': np.random.normal(0.90, 0.03, 30),
        'DNN': np.random.normal(0.89, 0.04, 30)
    }
    
    # ANOVA
    suite.one_way_anova(scores_dict)
    
    # Friedman test
    suite.friedman_test(scores_dict)
    
    # Comprehensive comparison
    suite.compare_models_comprehensive(scores_dict, paired=True)
    
    # Export
    suite.export_to_excel()
    
    logger.info("\n[OK] Testing complete! Check test_statistical_results/")
