# PFAZ 9: AAA2 Control Group & Monte Carlo
# ==========================================

# Main pipeline (ACTIVE)
from .aaa2_control_group_complete_v4 import AAA2ControlGroupAnalyzerComplete

#  ACTIVATED MODULES: Advanced Analytics (4 modüller aktif edildi)
try:
    from .aaa2_control_group_comprehensive import AAA2ControlGroupComprehensive
    AAA2_CONTROL_GROUP_COMPREHENSIVE_AVAILABLE = True
except ImportError:
    AAA2ControlGroupComprehensive = None
    AAA2_CONTROL_GROUP_COMPREHENSIVE_AVAILABLE = False

try:
    from .aaa2_quality_checker import AAA2QualityChecker
    AAA2_QUALITY_CHECKER_AVAILABLE = True
except ImportError:
    AAA2QualityChecker = None
    AAA2_QUALITY_CHECKER_AVAILABLE = False

try:
    from .advanced_analytics_comprehensive import AdvancedAnalyticsComprehensive
    ADVANCED_ANALYTICS_COMPREHENSIVE_AVAILABLE = True
except ImportError:
    AdvancedAnalyticsComprehensive = None
    ADVANCED_ANALYTICS_COMPREHENSIVE_AVAILABLE = False

try:
    from .monte_carlo_simulation_system import MonteCarloSimulationSystem
    MONTE_CARLO_SIMULATION_AVAILABLE = True
except ImportError:
    MonteCarloSimulationSystem = None
    MONTE_CARLO_SIMULATION_AVAILABLE = False

__all__ = [
    'AAA2ControlGroupAnalyzerComplete',
    'AAA2ControlGroupComprehensive', 'AAA2_CONTROL_GROUP_COMPREHENSIVE_AVAILABLE',
    'AAA2QualityChecker', 'AAA2_QUALITY_CHECKER_AVAILABLE',
    'AdvancedAnalyticsComprehensive', 'ADVANCED_ANALYTICS_COMPREHENSIVE_AVAILABLE',
    'MonteCarloSimulationSystem', 'MONTE_CARLO_SIMULATION_AVAILABLE',
]
