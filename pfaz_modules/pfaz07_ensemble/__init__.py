# PFAZ 7: Ensemble & Meta-Models
# ================================

# Main pipeline (ACTIVE)
from .pfaz7_production_complete import run_pfaz7_production

#  ACTIVATED MODULES: Ensemble Tools (6 modüller aktif edildi)
try:
    from .ensemble_evaluator import EnsembleEvaluator
    ENSEMBLE_EVALUATOR_AVAILABLE = True
except ImportError:
    EnsembleEvaluator = None
    ENSEMBLE_EVALUATOR_AVAILABLE = False

try:
    from .ensemble_model_builder import EnsembleModelBuilder
    ENSEMBLE_MODEL_BUILDER_AVAILABLE = True
except ImportError:
    EnsembleModelBuilder = None
    ENSEMBLE_MODEL_BUILDER_AVAILABLE = False

try:
    from .faz7_ensemble_pipeline import Faz7EnsemblePipeline
    FAZ7_ENSEMBLE_PIPELINE_AVAILABLE = True
except ImportError:
    Faz7EnsemblePipeline = None
    FAZ7_ENSEMBLE_PIPELINE_AVAILABLE = False

try:
    from .pfaz7_complete_ensemble_pipeline import CompletePFAZ7EnsemblePipeline
    COMPLETE_ENSEMBLE_PIPELINE_AVAILABLE = True
except ImportError:
    CompletePFAZ7EnsemblePipeline = None
    COMPLETE_ENSEMBLE_PIPELINE_AVAILABLE = False

try:
    from .pfaz7_ensemble import PFAZ7Ensemble
    PFAZ7_ENSEMBLE_AVAILABLE = True
except ImportError:
    PFAZ7Ensemble = None
    PFAZ7_ENSEMBLE_AVAILABLE = False

try:
    from .stacking_meta_learner import StackingMetaLearner
    STACKING_META_LEARNER_AVAILABLE = True
except ImportError:
    StackingMetaLearner = None
    STACKING_META_LEARNER_AVAILABLE = False

__all__ = [
    'run_pfaz7_production',
    'EnsembleEvaluator', 'ENSEMBLE_EVALUATOR_AVAILABLE',
    'EnsembleModelBuilder', 'ENSEMBLE_MODEL_BUILDER_AVAILABLE',
    'Faz7EnsemblePipeline', 'FAZ7_ENSEMBLE_PIPELINE_AVAILABLE',
    'CompletePFAZ7EnsemblePipeline', 'COMPLETE_ENSEMBLE_PIPELINE_AVAILABLE',
    'PFAZ7Ensemble', 'PFAZ7_ENSEMBLE_AVAILABLE',
    'StackingMetaLearner', 'STACKING_META_LEARNER_AVAILABLE',
]
