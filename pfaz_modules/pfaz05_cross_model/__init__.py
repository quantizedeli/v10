# PFAZ 5: Cross-Model Analysis
# ==============================

# Main pipeline (ACTIVE)
from .cross_model_evaluator import CrossModelEvaluator

#  ACTIVATED MODULES: Cross-Model Tools (2 mod�ller aktif edildi)
try:
    from .best_model_selector import BestModelSelector
    BEST_MODEL_SELECTOR_AVAILABLE = True
except ImportError:
    BestModelSelector = None
    BEST_MODEL_SELECTOR_AVAILABLE = False

try:
    from .faz5_complete_cross_model import CompleteCrossModelAnalyzer
    COMPLETE_CROSS_MODEL_AVAILABLE = True
except ImportError:
    CompleteCrossModelAnalyzer = None
    COMPLETE_CROSS_MODEL_AVAILABLE = False

try:
    from .faz5_cross_model_analysis import CrossModelAnalysisPipeline
    CROSS_MODEL_ANALYSIS_AVAILABLE = True
except ImportError:
    CrossModelAnalysisPipeline = None
    CROSS_MODEL_ANALYSIS_AVAILABLE = False

# ✅ ACTIVATED: Moved from root directory
try:
    from .optimizer_comparison_reporter import OptimizerComparisonReporter
    OPTIMIZER_COMPARISON_REPORTER_AVAILABLE = True
except ImportError:
    OptimizerComparisonReporter = None
    OPTIMIZER_COMPARISON_REPORTER_AVAILABLE = False

__all__ = [
    'CrossModelEvaluator',
    'BestModelSelector', 'BEST_MODEL_SELECTOR_AVAILABLE',
    'CompleteCrossModelAnalyzer', 'COMPLETE_CROSS_MODEL_AVAILABLE',
    'CrossModelAnalysisPipeline', 'CROSS_MODEL_ANALYSIS_AVAILABLE',
    # Moved from root
    'OptimizerComparisonReporter', 'OPTIMIZER_COMPARISON_REPORTER_AVAILABLE',
]
