# PFAZ 4: Unknown Nuclei Predictions
# ====================================

# Main pipeline (ACTIVE)
from .unknown_nuclei_predictor import UnknownNucleiPredictor

# Wired: GeneralizationAnalyzer called from UnknownNucleiPredictor.predict_unknown_nuclei()
try:
    from .generalization_analyzer import GeneralizationAnalyzer
    GENERALIZATION_ANALYZER_AVAILABLE = True
except ImportError:
    GeneralizationAnalyzer = None
    GENERALIZATION_ANALYZER_AVAILABLE = False

# Removed: AllNucleiPredictor (replaced by SingleNucleusPredictor + UnknownNucleiPredictor)
# Removed: UnknownNucleiSplitter (PFAZ1 already creates train/val/test splits)

__all__ = [
    'UnknownNucleiPredictor',
    'GeneralizationAnalyzer', 'GENERALIZATION_ANALYZER_AVAILABLE',
]
