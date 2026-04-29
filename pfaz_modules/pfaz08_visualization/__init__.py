# PFAZ 8: Visualization Modules
# ==============================

# Main pipeline (ACTIVE)
from .visualization_master_system import MasterVisualizationSystem
VisualizationMasterSystem = MasterVisualizationSystem  # alias for backward compat

#  ACTIVATED MODULES: Visualization Tools (10 mod�ller aktif edildi)
try:
    from .shap_analysis import SHAPAnalyzer
    SHAP_ANALYSIS_AVAILABLE = True
except ImportError:
    SHAPAnalyzer = None
    SHAP_ANALYSIS_AVAILABLE = False

try:
    from .ai_visualizer import AIVisualizer
    AI_VISUALIZER_AVAILABLE = True
except ImportError:
    AIVisualizer = None
    AI_VISUALIZER_AVAILABLE = False

try:
    from .anomaly_visualizations_complete import AnomalyVisualizationSystem
    ANOMALY_VIZ_AVAILABLE = True
except ImportError:
    AnomalyVisualizationSystem = None
    ANOMALY_VIZ_AVAILABLE = False

try:
    from .log_analytics_visualizations_complete import LogAnalyticsVisualizer
    LOG_ANALYTICS_VIZ_AVAILABLE = True
except ImportError:
    LogAnalyticsVisualizer = None
    LOG_ANALYTICS_VIZ_AVAILABLE = False

try:
    from .model_comparison_dashboard import ModelComparisonDashboard
    MODEL_COMPARISON_AVAILABLE = True
except ImportError:
    ModelComparisonDashboard = None
    MODEL_COMPARISON_AVAILABLE = False

try:
    from .interactive_html_visualizer import InteractiveHTMLVisualizer
    INTERACTIVE_HTML_AVAILABLE = True
except ImportError:
    InteractiveHTMLVisualizer = None
    INTERACTIVE_HTML_AVAILABLE = False

try:
    from .robustness_visualizations_complete import RobustnessVisualizationSystem
    ROBUSTNESS_VIZ_AVAILABLE = True
except ImportError:
    RobustnessVisualizationSystem = None
    ROBUSTNESS_VIZ_AVAILABLE = False

try:
    from .master_report_visualizations_complete import MasterReportVisualizationSystem
    MASTER_REPORT_VIZ_AVAILABLE = True
except ImportError:
    MasterReportVisualizationSystem = None
    MASTER_REPORT_VIZ_AVAILABLE = False

try:
    from .visualization_system import VisualizationSystem
    VISUALIZATION_SYSTEM_AVAILABLE = True
except ImportError:
    VisualizationSystem = None
    VISUALIZATION_SYSTEM_AVAILABLE = False

try:
    from .visualization_advanced_modules import AdvancedVisualizationModules
    ADVANCED_VIZ_MODULES_AVAILABLE = True
except ImportError:
    AdvancedVisualizationModules = None
    ADVANCED_VIZ_MODULES_AVAILABLE = False

__all__ = [
    'MasterVisualizationSystem',
    'VisualizationMasterSystem',
    'SHAPAnalyzer', 'SHAP_ANALYSIS_AVAILABLE',
    'AIVisualizer', 'AI_VISUALIZER_AVAILABLE',
    'AnomalyVisualizationSystem', 'ANOMALY_VIZ_AVAILABLE',
    'LogAnalyticsVisualizer', 'LOG_ANALYTICS_VIZ_AVAILABLE',
    'ModelComparisonDashboard', 'MODEL_COMPARISON_AVAILABLE',
    'InteractiveHTMLVisualizer', 'INTERACTIVE_HTML_AVAILABLE',
    'RobustnessVisualizationSystem', 'ROBUSTNESS_VIZ_AVAILABLE',
    'MasterReportVisualizationSystem', 'MASTER_REPORT_VIZ_AVAILABLE',
    'VisualizationSystem', 'VISUALIZATION_SYSTEM_AVAILABLE',
    'AdvancedVisualizationModules', 'ADVANCED_VIZ_MODULES_AVAILABLE',
]
