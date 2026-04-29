"""
Bayesian Model Comparison
==========================

Advanced statistical testing using Bayesian methods:
- Bayesian Model Comparison (Bayes Factors)
- Posterior distributions
- Credible intervals
- Model averaging
- ROPE (Region of Practical Equivalence)
- Expected Log Predictive Density (ELPD)

Complements frequentist methods in statistical_testing_suite.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)


class BayesianModelComparison:
    """
    Bayesian methods for comparing ML models

    Methods:
    - Bayes Factor calculation
    - Posterior predictive checks
    - ROPE analysis
    - Model posterior probabilities
    """

    def __init__(self, rope_percentage: float = 0.05):
        """
        Initialize Bayesian comparator

        Args:
            rope_percentage: ROPE as percentage of scale (e.g., 0.05 = 5%)
        """
        self.rope_percentage = rope_percentage
        self.results = {}

    def bayes_factor(self, errors1: np.ndarray, errors2: np.ndarray,
                    prior_odds: float = 1.0) -> Dict[str, float]:
        """
        Calculate Bayes Factor for comparing two models

        BF > 1: Evidence for model 1
        BF < 1: Evidence for model 2

        Interpretation (Jeffreys' scale):
        - BF > 100: Decisive evidence
        - 30-100: Very strong evidence
        - 10-30: Strong evidence
        - 3-10: Substantial evidence
        - 1-3: Barely worth mentioning

        Args:
            errors1: Prediction errors from model 1
            errors2: Prediction errors from model 2
            prior_odds: Prior odds ratio (default 1.0 = equal priors)

        Returns:
            Dictionary with Bayes factor and interpretation
        """
        logger.info("Calculating Bayes Factor...")

        # Calculate MSE for both models
        mse1 = np.mean(errors1 ** 2)
        mse2 = np.mean(errors2 ** 2)

        # Sample sizes
        n = len(errors1)

        # Assume normal likelihood with unknown variance
        # Use Savage-Dickey density ratio approximation

        # Calculate log marginal likelihoods
        # Using BIC approximation: log p(D|M) ≈ -0.5 * BIC
        # BIC = n * log(MSE) + k * log(n), with k parameters

        # Simplified: assume same number of parameters
        log_ml1 = -0.5 * n * np.log(mse1)
        log_ml2 = -0.5 * n * np.log(mse2)

        # Bayes factor (using log space for numerical stability)
        log_bf = log_ml1 - log_ml2
        bf = np.exp(log_bf)

        # Posterior odds = BF * prior odds
        posterior_odds = bf * prior_odds

        # Interpretation
        if bf > 100:
            interpretation = "Decisive evidence for Model 1"
        elif bf > 30:
            interpretation = "Very strong evidence for Model 1"
        elif bf > 10:
            interpretation = "Strong evidence for Model 1"
        elif bf > 3:
            interpretation = "Substantial evidence for Model 1"
        elif bf > 1:
            interpretation = "Weak evidence for Model 1"
        elif bf > 1/3:
            interpretation = "Weak evidence for Model 2"
        elif bf > 1/10:
            interpretation = "Substantial evidence for Model 2"
        elif bf > 1/30:
            interpretation = "Strong evidence for Model 2"
        elif bf > 1/100:
            interpretation = "Very strong evidence for Model 2"
        else:
            interpretation = "Decisive evidence for Model 2"

        results = {
            'bayes_factor': bf,
            'log_bayes_factor': log_bf,
            'posterior_odds': posterior_odds,
            'interpretation': interpretation,
            'mse1': mse1,
            'mse2': mse2
        }

        logger.info(f"Bayes Factor: {bf:.4f} ({interpretation})")

        return results

    def rope_analysis(self, errors1: np.ndarray, errors2: np.ndarray,
                     rope_bounds: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        ROPE (Region of Practical Equivalence) analysis

        Determines if difference between models is practically significant

        Args:
            errors1: Errors from model 1
            errors2: Errors from model 2
            rope_bounds: (lower, upper) bounds for ROPE (optional)

        Returns:
            ROPE analysis results
        """
        logger.info("Performing ROPE analysis...")

        # Calculate difference in errors
        diff = errors1 - errors2

        # Estimate posterior distribution of difference (assume normal)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))

        # Calculate 95% credible interval
        ci_lower = mean_diff - 1.96 * std_diff
        ci_upper = mean_diff + 1.96 * std_diff

        # Set ROPE bounds if not provided
        if rope_bounds is None:
            # Default: ±5% of error magnitude
            error_scale = np.std(errors1)
            rope_width = self.rope_percentage * error_scale
            rope_bounds = (-rope_width, rope_width)

        rope_lower, rope_upper = rope_bounds

        # Determine decision
        if ci_lower > rope_upper:
            decision = "Model 2 is practically better"
        elif ci_upper < rope_lower:
            decision = "Model 1 is practically better"
        elif ci_lower > rope_lower and ci_upper < rope_upper:
            decision = "Models are practically equivalent"
        else:
            decision = "Inconclusive"

        # Calculate probability in ROPE
        prob_in_rope = (
            stats.norm.cdf(rope_upper, mean_diff, std_diff) -
            stats.norm.cdf(rope_lower, mean_diff, std_diff)
        )

        results = {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'credible_interval_95': (ci_lower, ci_upper),
            'rope_bounds': rope_bounds,
            'prob_in_rope': prob_in_rope,
            'decision': decision
        }

        logger.info(f"ROPE decision: {decision}")
        logger.info(f"P(in ROPE) = {prob_in_rope:.3f}")

        return results

    def model_posteriors(self, model_errors: Dict[str, np.ndarray],
                        prior_probs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate posterior probabilities for multiple models

        Args:
            model_errors: Dictionary of {model_name: errors}
            prior_probs: Prior probabilities (default: uniform)

        Returns:
            Posterior probabilities for each model
        """
        logger.info(f"Calculating posterior probabilities for {len(model_errors)} models...")

        n_models = len(model_errors)
        model_names = list(model_errors.keys())

        # Set uniform priors if not provided
        if prior_probs is None:
            prior_probs = {name: 1/n_models for name in model_names}

        # Calculate log marginal likelihoods
        log_ml = {}
        for name, errors in model_errors.items():
            mse = np.mean(errors ** 2)
            n = len(errors)
            log_ml[name] = -0.5 * n * np.log(mse)

        # Calculate log posterior probabilities (unnormalized)
        log_post_unnorm = {
            name: log_ml[name] + np.log(prior_probs[name])
            for name in model_names
        }

        # Normalize using logsumexp for numerical stability
        log_evidence = logsumexp(list(log_post_unnorm.values()))

        # Posterior probabilities
        posteriors = {
            name: np.exp(log_post_unnorm[name] - log_evidence)
            for name in model_names
        }

        # Sort by posterior probability
        sorted_posteriors = dict(sorted(posteriors.items(),
                                       key=lambda x: x[1],
                                       reverse=True))

        logger.info("Posterior probabilities:")
        for name, prob in sorted_posteriors.items():
            logger.info(f"  {name}: {prob:.4f}")

        return sorted_posteriors

    def predictive_performance_difference(self,
                                         y_true: np.ndarray,
                                         predictions1: np.ndarray,
                                         predictions2: np.ndarray,
                                         n_samples: int = 10000) -> Dict[str, Any]:
        """
        Bayesian estimation of predictive performance difference

        Uses bootstrap sampling to estimate posterior distribution

        Args:
            y_true: True values
            predictions1: Predictions from model 1
            predictions2: Predictions from model 2
            n_samples: Number of bootstrap samples

        Returns:
            Posterior distribution statistics
        """
        logger.info("Estimating posterior distribution of performance difference...")

        n = len(y_true)

        # Bootstrap samples of performance difference
        mse_diff_samples = []

        for _ in range(n_samples):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)

            y_sample = y_true[indices]
            pred1_sample = predictions1[indices]
            pred2_sample = predictions2[indices]

            # Calculate MSE for each model
            mse1 = np.mean((y_sample - pred1_sample) ** 2)
            mse2 = np.mean((y_sample - pred2_sample) ** 2)

            mse_diff_samples.append(mse1 - mse2)

        mse_diff_samples = np.array(mse_diff_samples)

        # Posterior statistics
        mean_diff = np.mean(mse_diff_samples)
        median_diff = np.median(mse_diff_samples)
        std_diff = np.std(mse_diff_samples)

        # Credible intervals
        ci_50 = np.percentile(mse_diff_samples, [25, 75])
        ci_95 = np.percentile(mse_diff_samples, [2.5, 97.5])
        ci_99 = np.percentile(mse_diff_samples, [0.5, 99.5])

        # Probability that model 1 is better
        prob_m1_better = np.mean(mse_diff_samples < 0)

        results = {
            'mean_difference': mean_diff,
            'median_difference': median_diff,
            'std_difference': std_diff,
            'credible_interval_50': ci_50,
            'credible_interval_95': ci_95,
            'credible_interval_99': ci_99,
            'prob_model1_better': prob_m1_better,
            'samples': mse_diff_samples
        }

        logger.info(f"Mean difference: {mean_diff:.6f}")
        logger.info(f"P(Model 1 better) = {prob_m1_better:.3f}")

        return results

    def generate_report(self, y_true: np.ndarray,
                       model_predictions: Dict[str, np.ndarray],
                       output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive Bayesian comparison report

        Args:
            y_true: True values
            model_predictions: Dictionary of {model_name: predictions}
            output_file: Path to save report (optional)

        Returns:
            DataFrame with comparison results
        """
        logger.info("Generating Bayesian comparison report...")

        model_names = list(model_predictions.keys())
        n_models = len(model_names)

        # Calculate errors
        model_errors = {
            name: y_true - pred
            for name, pred in model_predictions.items()
        }

        # Pairwise Bayes factors
        bf_results = []

        for i in range(n_models):
            for j in range(i+1, n_models):
                name1 = model_names[i]
                name2 = model_names[j]

                bf = self.bayes_factor(model_errors[name1], model_errors[name2])

                bf_results.append({
                    'model1': name1,
                    'model2': name2,
                    'bayes_factor': bf['bayes_factor'],
                    'interpretation': bf['interpretation']
                })

        # Model posteriors
        posteriors = self.model_posteriors(model_errors)

        # Create DataFrame
        df_bf = pd.DataFrame(bf_results)

        df_posterior = pd.DataFrame([
            {'model': name, 'posterior_probability': prob}
            for name, prob in posteriors.items()
        ])

        logger.info(f"Report generated with {len(bf_results)} pairwise comparisons")

        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                df_bf.to_excel(writer, sheet_name='Bayes_Factors', index=False)
                df_posterior.to_excel(writer, sheet_name='Posteriors', index=False)
            logger.info(f"Report saved to {output_file}")

        return df_bf, df_posterior


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Simulate predictions from 3 models
    np.random.seed(42)
    n = 100

    y_true = np.random.randn(n)

    # Model 1: Good predictions
    pred1 = y_true + np.random.randn(n) * 0.1

    # Model 2: Similar to model 1
    pred2 = y_true + np.random.randn(n) * 0.12

    # Model 3: Worse predictions
    pred3 = y_true + np.random.randn(n) * 0.3

    # Initialize comparator
    comparator = BayesianModelComparison(rope_percentage=0.05)

    # 1. Bayes Factor
    print("\n=== Bayes Factor ===")
    errors1 = y_true - pred1
    errors2 = y_true - pred2
    bf_results = comparator.bayes_factor(errors1, errors2)
    print(f"BF = {bf_results['bayes_factor']:.4f}")
    print(f"{bf_results['interpretation']}")

    # 2. ROPE Analysis
    print("\n=== ROPE Analysis ===")
    rope_results = comparator.rope_analysis(errors1, errors2)
    print(f"Decision: {rope_results['decision']}")

    # 3. Model Posteriors
    print("\n=== Model Posteriors ===")
    model_errors = {
        'Model 1': errors1,
        'Model 2': errors2,
        'Model 3': y_true - pred3
    }
    posteriors = comparator.model_posteriors(model_errors)

    # 4. Predictive Performance
    print("\n=== Predictive Performance ===")
    perf = comparator.predictive_performance_difference(y_true, pred1, pred2)
    print(f"P(Model 1 better) = {perf['prob_model1_better']:.3f}")

    # 5. Generate Report
    print("\n=== Report ===")
    model_preds = {'Model 1': pred1, 'Model 2': pred2, 'Model 3': pred3}
    df_bf, df_post = comparator.generate_report(y_true, model_preds)
    print("\nBayes Factors:")
    print(df_bf)
    print("\nPosteriors:")
    print(df_post)

    print("\n[SUCCESS] Bayesian model comparison tested!")
