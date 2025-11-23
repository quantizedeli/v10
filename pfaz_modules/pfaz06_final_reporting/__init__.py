# PFAZ 6: Final Reporting Modules
# =================================

# Main pipeline (ACTIVE)
from .pfaz6_final_reporting import FinalReportingPipeline

#  ACTIVATED MODULES: Reporting Tools (4 modüller aktif edildi)
try:
    from .excel_charts import ExcelChartGenerator
    EXCEL_CHARTS_AVAILABLE = True
except ImportError:
    ExcelChartGenerator = None
    EXCEL_CHARTS_AVAILABLE = False

try:
    from .advanced_analysis_reporting_manager import AdvancedAnalysisReportingManager
    ADVANCED_ANALYSIS_REPORTING_AVAILABLE = True
except ImportError:
    AdvancedAnalysisReportingManager = None
    ADVANCED_ANALYSIS_REPORTING_AVAILABLE = False

try:
    from .latex_generator import LatexReportGenerator
    LATEX_GENERATOR_AVAILABLE = True
except ImportError:
    LatexReportGenerator = None
    LATEX_GENERATOR_AVAILABLE = False

try:
    from .comprehensive_excel_reporter import ComprehensiveExcelReporter
    COMPREHENSIVE_EXCEL_REPORTER_AVAILABLE = True
except ImportError:
    ComprehensiveExcelReporter = None
    COMPREHENSIVE_EXCEL_REPORTER_AVAILABLE = False

try:
    from .excel_formatter import ExcelFormatter
    EXCEL_FORMATTER_AVAILABLE = True
except ImportError:
    ExcelFormatter = None
    EXCEL_FORMATTER_AVAILABLE = False

try:
    from .reports_comprehensive_module import ReportsComprehensiveModule
    REPORTS_COMPREHENSIVE_AVAILABLE = True
except ImportError:
    ReportsComprehensiveModule = None
    REPORTS_COMPREHENSIVE_AVAILABLE = False

__all__ = [
    'FinalReportingPipeline',
    'ExcelChartGenerator', 'EXCEL_CHARTS_AVAILABLE',
    'AdvancedAnalysisReportingManager', 'ADVANCED_ANALYSIS_REPORTING_AVAILABLE',
    'LatexReportGenerator', 'LATEX_GENERATOR_AVAILABLE',
    'ComprehensiveExcelReporter', 'COMPREHENSIVE_EXCEL_REPORTER_AVAILABLE',
    'ExcelFormatter', 'EXCEL_FORMATTER_AVAILABLE',
    'ReportsComprehensiveModule', 'REPORTS_COMPREHENSIVE_AVAILABLE',
]
