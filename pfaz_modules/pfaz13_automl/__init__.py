# PFAZ 13: AutoML Integration Modules
# ====================================
#
# AutoML akışı:
# 1. automl_optimizer.py     — Optuna ile hiperparametre optimizasyonu (RF/XGB/LGB/CB/SVR/DNN)
# 2. automl_anfis_optimizer  — ANFIS konfigürasyon optimizasyonu (PFAZ3 ile uyumlu)
# 3. automl_feature_engineer — Polinomial + fizik esinli özellik mühendisliği
# 4. automl_visualizer       — Optuna grafik ve raporları
# 5. automl_logging_*        — Trial-by-trial kayıt sistemi
#
# NOT: automl_hyperparameter_optimizer.py ile automl_optimizer.py
#      ikisi de AutoMLOptimizer sınıfını tanımlar.
#      automl_optimizer.py daha kapsamlı (LGB/CB/SVR desteği, MM_QM).
#      automl_hyperparameter_optimizer.py eski sürüm, backward-compat için tutuluyor.

# Primary optimizer (multi-model, MM_QM, safe optuna import)
try:
    from .automl_optimizer import AutoMLOptimizer, OPTUNA_AVAILABLE, optimize_all_targets
    AUTOML_OPTIMIZER_AVAILABLE = True
except ImportError:
    AutoMLOptimizer = None
    OPTUNA_AVAILABLE = False
    optimize_all_targets = None
    AUTOML_OPTIMIZER_AVAILABLE = False

# Legacy hyperparameter optimizer (backward-compat alias)
try:
    from .automl_hyperparameter_optimizer import AutoMLOptimizer as AutoMLHyperparameterOptimizer
    AUTOML_HYPERPARAMETER_OPTIMIZER_AVAILABLE = True
except ImportError:
    AutoMLHyperparameterOptimizer = None
    AUTOML_HYPERPARAMETER_OPTIMIZER_AVAILABLE = False

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
    from .automl_anfis_optimizer import AutoMLANFISOptimizer
    AUTOML_ANFIS_OPTIMIZER_AVAILABLE = True
except ImportError:
    AutoMLANFISOptimizer = None
    AUTOML_ANFIS_OPTIMIZER_AVAILABLE = False

try:
    from .automl_logging_reporting_system import AutoMLLoggingReportingSystem
    AUTOML_LOGGING_AVAILABLE = True
except Exception:
    AutoMLLoggingReportingSystem = None
    AUTOML_LOGGING_AVAILABLE = False

try:
    from .automl_retraining_loop import AutoMLRetrainingLoop
    AUTOML_RETRAINING_AVAILABLE = True
except ImportError:
    AutoMLRetrainingLoop = None
    AUTOML_RETRAINING_AVAILABLE = False

__all__ = [
    # Primary
    'AutoMLOptimizer', 'OPTUNA_AVAILABLE', 'AUTOML_OPTIMIZER_AVAILABLE',
    'optimize_all_targets',
    # Legacy alias
    'AutoMLHyperparameterOptimizer', 'AUTOML_HYPERPARAMETER_OPTIMIZER_AVAILABLE',
    # Visualization
    'AutoMLVisualizer', 'AUTOML_VISUALIZER_AVAILABLE',
    # Feature Engineering
    'AutoMLFeatureEngineer', 'AUTOML_FEATURE_ENGINEER_AVAILABLE',
    # ANFIS Optimizer
    'AutoMLANFISOptimizer', 'AUTOML_ANFIS_OPTIMIZER_AVAILABLE',
    # Logging & Reporting
    'AutoMLLoggingReportingSystem', 'AUTOML_LOGGING_AVAILABLE',
    # Retraining Loop
    'AutoMLRetrainingLoop', 'AUTOML_RETRAINING_AVAILABLE',
]
