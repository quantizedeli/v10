# PFAZ 4: Unknown Nuclei Predictions
# ====================================

# Main pipeline (ACTIVE)
from .unknown_nuclei_predictor import UnknownNucleiPredictor

#  ACTIVATED MODULES: Unknown Predictions Tools (3 modüller aktif edildi)
try:
    from .all_nuclei_predictor import AllNucleiPredictor
    ALL_NUCLEI_PREDICTOR_AVAILABLE = True
except ImportError:
    AllNucleiPredictor = None
    ALL_NUCLEI_PREDICTOR_AVAILABLE = False

try:
    from .generalization_analyzer import GeneralizationAnalyzer
    GENERALIZATION_ANALYZER_AVAILABLE = True
except ImportError:
    GeneralizationAnalyzer = None
    GENERALIZATION_ANALYZER_AVAILABLE = False

try:
    from .unknown_nuclei_splitter import UnknownNucleiSplitter
    UNKNOWN_NUCLEI_SPLITTER_AVAILABLE = True
except ImportError:
    UnknownNucleiSplitter = None
    UNKNOWN_NUCLEI_SPLITTER_AVAILABLE = False

__all__ = [
    'UnknownNucleiPredictor',
    'AllNucleiPredictor', 'ALL_NUCLEI_PREDICTOR_AVAILABLE',
    'GeneralizationAnalyzer', 'GENERALIZATION_ANALYZER_AVAILABLE',
    'UnknownNucleiSplitter', 'UNKNOWN_NUCLEI_SPLITTER_AVAILABLE',
]
