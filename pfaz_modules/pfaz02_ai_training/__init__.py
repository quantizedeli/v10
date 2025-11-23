# PFAZ 2: AI Training Modules
# ==========================

# Main pipeline (ACTIVE)
from .parallel_ai_trainer import ParallelAITrainer

#  ACTIVATED MODULES: Optional/Utility Tools
try:
    from .hyperparameter_tuner import HyperparameterTuner
    HYPERPARAMETER_TUNER_AVAILABLE = True
except ImportError:
    HyperparameterTuner = None
    HYPERPARAMETER_TUNER_AVAILABLE = False

try:
    from .model_validator import CrossValidationAnalyzer, RobustnessTestSuite
    MODEL_VALIDATOR_AVAILABLE = True
except ImportError:
    CrossValidationAnalyzer = None
    RobustnessTestSuite = None
    MODEL_VALIDATOR_AVAILABLE = False

try:
    from .overfitting_detector import OverfittingDetector, OverfittingMetrics
    OVERFITTING_DETECTOR_AVAILABLE = True
except ImportError:
    OverfittingDetector = None
    OverfittingMetrics = None
    OVERFITTING_DETECTOR_AVAILABLE = False

try:
    from .advanced_models import (
        BayesianNeuralNetwork,
        PINN,
        TransferLearningModel,
        EnsembleHybridModel
    )
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    BayesianNeuralNetwork = None
    PINN = None
    TransferLearningModel = None
    EnsembleHybridModel = None
    ADVANCED_MODELS_AVAILABLE = False

try:
    from .model_trainer import ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    ModelTrainer = None
    MODEL_TRAINER_AVAILABLE = False

__all__ = [
    # Main
    'ParallelAITrainer',
    # Hyperparameter Tuning
    'HyperparameterTuner',
    'HYPERPARAMETER_TUNER_AVAILABLE',
    # Model Validation
    'CrossValidationAnalyzer',
    'RobustnessTestSuite',
    'MODEL_VALIDATOR_AVAILABLE',
    # Overfitting Detection
    'OverfittingDetector',
    'OverfittingMetrics',
    'OVERFITTING_DETECTOR_AVAILABLE',
    # Advanced Models
    'BayesianNeuralNetwork',
    'PINN',
    'TransferLearningModel',
    'EnsembleHybridModel',
    'ADVANCED_MODELS_AVAILABLE',
    # Alternative Trainer
    'ModelTrainer',
    'MODEL_TRAINER_AVAILABLE',
]
