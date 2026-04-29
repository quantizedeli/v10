# PFAZ 3: ANFIS Training Modules
# ===============================

# Main pipeline (ACTIVE)
from .anfis_parallel_trainer_v2 import ANFISParallelTrainerV2

#  ACTIVATED MODULES: ANFIS Components (9 mod�ller aktif edildi)
try:
    from .anfis_adaptive_strategy import ANFISAdaptiveStrategy
    ANFIS_ADAPTIVE_STRATEGY_AVAILABLE = True
except ImportError:
    ANFISAdaptiveStrategy = None
    ANFIS_ADAPTIVE_STRATEGY_AVAILABLE = False

try:
    from .anfis_config_manager import ANFISConfigManager
    ANFIS_CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    ANFISConfigManager = None
    ANFIS_CONFIG_MANAGER_AVAILABLE = False

try:
    from .anfis_dataset_selector import ANFISDatasetSelector
    ANFIS_DATASET_SELECTOR_AVAILABLE = True
except ImportError:
    ANFISDatasetSelector = None
    ANFIS_DATASET_SELECTOR_AVAILABLE = False

try:
    from .anfis_model_saver import ANFISModelSaver
    ANFIS_MODEL_SAVER_AVAILABLE = True
except ImportError:
    ANFISModelSaver = None
    ANFIS_MODEL_SAVER_AVAILABLE = False

try:
    from .anfis_performance_analyzer import ANFISPerformanceAnalyzer
    ANFIS_PERFORMANCE_ANALYZER_AVAILABLE = True
except ImportError:
    ANFISPerformanceAnalyzer = None
    ANFIS_PERFORMANCE_ANALYZER_AVAILABLE = False

try:
    from .anfis_robustness_tester import ANFISRobustnessTester
    ANFIS_ROBUSTNESS_TESTER_AVAILABLE = True
except ImportError:
    ANFISRobustnessTester = None
    ANFIS_ROBUSTNESS_TESTER_AVAILABLE = False

try:
    from .anfis_visualizer import ANFISVisualizer
    ANFIS_VISUALIZER_AVAILABLE = True
except ImportError:
    ANFISVisualizer = None
    ANFIS_VISUALIZER_AVAILABLE = False

try:
    from .matlab_anfis_trainer import MATLABAnfisTrainer
    MatlabANFISTrainer = MATLABAnfisTrainer  # backward compat alias
    MATLAB_ANFIS_TRAINER_AVAILABLE = True
except ImportError as _e:
    import logging as _logging
    _logging.warning(f"MATLAB ANFIS trainer not available: {_e}")
    MATLABAnfisTrainer = None
    MatlabANFISTrainer = None
    MATLAB_ANFIS_TRAINER_AVAILABLE = False

__all__ = [
    'ANFISParallelTrainerV2',
    'ANFISAdaptiveStrategy', 'ANFIS_ADAPTIVE_STRATEGY_AVAILABLE',
    'ANFISConfigManager', 'ANFIS_CONFIG_MANAGER_AVAILABLE',
    'ANFISDatasetSelector', 'ANFIS_DATASET_SELECTOR_AVAILABLE',
    'ANFISModelSaver', 'ANFIS_MODEL_SAVER_AVAILABLE',
    'ANFISPerformanceAnalyzer', 'ANFIS_PERFORMANCE_ANALYZER_AVAILABLE',
    'ANFISRobustnessTester', 'ANFIS_ROBUSTNESS_TESTER_AVAILABLE',
    'ANFISVisualizer', 'ANFIS_VISUALIZER_AVAILABLE',
    'MATLABAnfisTrainer', 'MatlabANFISTrainer', 'MATLAB_ANFIS_TRAINER_AVAILABLE',
]
