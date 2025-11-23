# PFAZ 12: Advanced Analytics
# =============================

# Main pipeline (ACTIVE)
from .advanced_analytics_comprehensive import AdvancedAnalyticsComprehensive

#  ACTIVATED MODULES: Advanced Analytics Tools (3 modüller aktif edildi)
try:
    from .advanced_sensitivity_analysis import AdvancedSensitivityAnalysis
    ADVANCED_SENSITIVITY_ANALYSIS_AVAILABLE = True
except ImportError:
    AdvancedSensitivityAnalysis = None
    ADVANCED_SENSITIVITY_ANALYSIS_AVAILABLE = False

try:
    from .bootstrap_confidence_intervals import BootstrapConfidenceIntervals
    BOOTSTRAP_CONFIDENCE_INTERVALS_AVAILABLE = True
except ImportError:
    BootstrapConfidenceIntervals = None
    BOOTSTRAP_CONFIDENCE_INTERVALS_AVAILABLE = False

try:
    from .statistical_testing_suite import StatisticalTestingSuite
    STATISTICAL_TESTING_SUITE_AVAILABLE = True
except ImportError:
    StatisticalTestingSuite = None
    STATISTICAL_TESTING_SUITE_AVAILABLE = False

__all__ = [
    'AdvancedAnalyticsComprehensive',
    'AdvancedSensitivityAnalysis', 'ADVANCED_SENSITIVITY_ANALYSIS_AVAILABLE',
    'BootstrapConfidenceIntervals', 'BOOTSTRAP_CONFIDENCE_INTERVALS_AVAILABLE',
    'StatisticalTestingSuite', 'STATISTICAL_TESTING_SUITE_AVAILABLE',
]
