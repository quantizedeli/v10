# PFAZ 13: AutoML Integration Modules
# ====================================

# Main pipeline (ACTIVE)
from .automl_hyperparameter_optimizer import AutoMLHyperparameterOptimizer

#  ACTIVATED MODULES: AutoML Tools
try:
    from .automl_visualizer import AutoMLVisualizer
    AUTOML_VISUALIZER_AVAILABLE = True
except ImportError:
    AutoMLVisualizer = None
    AUTOML_VISUALIZER_AVAILABLE = False

try:
    from .automl_feature_engineer import AutoMLFeatureEngineer
    AUTOML_FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    AutoMLFeatureEngineer = None
    AUTOML_FEATURE_ENGINEER_AVAILABLE = False

try:
    from .automl_optimizer import AutoMLOptimizer
    AUTOML_OPTIMIZER_AVAILABLE = True
except ImportError:
    AutoMLOptimizer = None
    AUTOML_OPTIMIZER_AVAILABLE = False

try:
    from .automl_anfis_optimizer import AutoMLANFISOptimizer
    AUTOML_ANFIS_OPTIMIZER_AVAILABLE = True
except ImportError:
    AutoMLANFISOptimizer = None
    AUTOML_ANFIS_OPTIMIZER_AVAILABLE = False

try:
    from .automl_logging_reporting_system import AutoMLLoggingReportingSystem
    AUTOML_LOGGING_AVAILABLE = True
except ImportError:
    AutoMLLoggingReportingSystem = None
    AUTOML_LOGGING_AVAILABLE = False

__all__ = [
    # Main
    'AutoMLHyperparameterOptimizer',
    # Visualization
    'AutoMLVisualizer',
    'AUTOML_VISUALIZER_AVAILABLE',
    # Feature Engineering
    'AutoMLFeatureEngineer',
    'AUTOML_FEATURE_ENGINEER_AVAILABLE',
    # Optimizer
    'AutoMLOptimizer',
    'AUTOML_OPTIMIZER_AVAILABLE',
    # ANFIS Optimizer
    'AutoMLANFISOptimizer',
    'AUTOML_ANFIS_OPTIMIZER_AVAILABLE',
    # Logging & Reporting
    'AutoMLLoggingReportingSystem',
    'AUTOML_LOGGING_AVAILABLE',
]
