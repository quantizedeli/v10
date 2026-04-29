# PFAZ 1: Dataset Generation Modules
# ====================================

# Main pipeline (ACTIVE)
from .dataset_generation_pipeline_v2 import DatasetGenerationPipelineV2

#  ACTIVATED MODULES: Data Processing & Enrichment (7 modüller aktif edildi)
try:
    from .control_group_generator import ControlGroupGenerator
    CONTROL_GROUP_GENERATOR_AVAILABLE = True
except ImportError:
    ControlGroupGenerator = None
    CONTROL_GROUP_GENERATOR_AVAILABLE = False

try:
    from .data_enricher import DataEnricher
    DATA_ENRICHER_AVAILABLE = True
except ImportError:
    DataEnricher = None
    DATA_ENRICHER_AVAILABLE = False

try:
    from .data_loader import DataLoader
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DataLoader = None
    DATA_LOADER_AVAILABLE = False

try:
    from .data_quality_modules import DataQualityChecker
    DATA_QUALITY_MODULES_AVAILABLE = True
except ImportError:
    DataQualityChecker = None
    DATA_QUALITY_MODULES_AVAILABLE = False

try:
    from .dataset_generator import DatasetGenerator
    DATASET_GENERATOR_AVAILABLE = True
except ImportError:
    DatasetGenerator = None
    DATASET_GENERATOR_AVAILABLE = False

try:
    from .nuclei_distribution_analyzer import NucleiDistributionAnalyzer
    NUCLEI_DISTRIBUTION_ANALYZER_AVAILABLE = True
except ImportError:
    NucleiDistributionAnalyzer = None
    NUCLEI_DISTRIBUTION_ANALYZER_AVAILABLE = False

try:
    from .qm_filter_manager import QMFilterManager
    QM_FILTER_MANAGER_AVAILABLE = True
except ImportError:
    QMFilterManager = None
    QM_FILTER_MANAGER_AVAILABLE = False

__all__ = [
    'DatasetGenerationPipelineV2',
    'ControlGroupGenerator', 'CONTROL_GROUP_GENERATOR_AVAILABLE',
    'DataEnricher', 'DATA_ENRICHER_AVAILABLE',
    'DataLoader', 'DATA_LOADER_AVAILABLE',
    'DataQualityChecker', 'DATA_QUALITY_MODULES_AVAILABLE',
    'DatasetGenerator', 'DATASET_GENERATOR_AVAILABLE',
    'NucleiDistributionAnalyzer', 'NUCLEI_DISTRIBUTION_ANALYZER_AVAILABLE',
    'QMFilterManager', 'QM_FILTER_MANAGER_AVAILABLE',
]
