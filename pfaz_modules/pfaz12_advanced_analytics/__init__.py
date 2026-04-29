# PFAZ 12: Advanced Analytics
# =============================

try:
    from .statistical_testing_suite import StatisticalTestingSuite
    STATISTICAL_TESTING_SUITE_AVAILABLE = True
except ImportError:
    StatisticalTestingSuite = None
    STATISTICAL_TESTING_SUITE_AVAILABLE = False

try:
    from .bayesian_model_comparison import BayesianModelComparison
    BAYESIAN_MODEL_COMPARISON_AVAILABLE = True
except ImportError:
    BayesianModelComparison = None
    BAYESIAN_MODEL_COMPARISON_AVAILABLE = False

try:
    from .bootstrap_confidence_intervals import BootstrapConfidenceIntervals
    BOOTSTRAP_CONFIDENCE_INTERVALS_AVAILABLE = True
except ImportError:
    BootstrapConfidenceIntervals = None
    BOOTSTRAP_CONFIDENCE_INTERVALS_AVAILABLE = False

try:
    from .advanced_sensitivity_analysis import AdvancedSensitivityAnalysis
    ADVANCED_SENSITIVITY_ANALYSIS_AVAILABLE = True
except ImportError:
    AdvancedSensitivityAnalysis = None
    ADVANCED_SENSITIVITY_ANALYSIS_AVAILABLE = False

__all__ = [
    'StatisticalTestingSuite',      'STATISTICAL_TESTING_SUITE_AVAILABLE',
    'BayesianModelComparison',      'BAYESIAN_MODEL_COMPARISON_AVAILABLE',
    'BootstrapConfidenceIntervals', 'BOOTSTRAP_CONFIDENCE_INTERVALS_AVAILABLE',
    'AdvancedSensitivityAnalysis',  'ADVANCED_SENSITIVITY_ANALYSIS_AVAILABLE',
]
