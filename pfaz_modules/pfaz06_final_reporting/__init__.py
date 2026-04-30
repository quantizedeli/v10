# PFAZ 6: Final Reporting Modules
# =================================

# Main pipeline (ACTIVE)
from .pfaz6_final_reporting import FinalReportingPipeline

# Wired modules (called from FinalReportingPipeline.run_complete_pipeline())
try:
    from .excel_charts import ExcelChartGenerator
    EXCEL_CHARTS_AVAILABLE = True
except ImportError:
    ExcelChartGenerator = None
    EXCEL_CHARTS_AVAILABLE = False

# AdvancedAnalysisReportingManager kaldırıldı — FinalReportingPipeline ile örtüşüyor,
# pipeline'da çağrılmıyor ve bakım yükü yaratıyor.

try:
    from .latex_generator import LaTeXReportGenerator
    LATEX_GENERATOR_AVAILABLE = True
except ImportError:
    LaTeXReportGenerator = None
    LATEX_GENERATOR_AVAILABLE = False

# Wired: ComprehensiveExcelReporter called from run_complete_pipeline()
try:
    from .comprehensive_excel_reporter import ComprehensiveExcelReporter
    COMPREHENSIVE_EXCEL_REPORTER_AVAILABLE = True
except ImportError:
    ComprehensiveExcelReporter = None
    COMPREHENSIVE_EXCEL_REPORTER_AVAILABLE = False

try:
    from .reports_comprehensive_module import ReportsComprehensiveModule
    REPORTS_COMPREHENSIVE_AVAILABLE = True
except ImportError:
    ReportsComprehensiveModule = None
    REPORTS_COMPREHENSIVE_AVAILABLE = False

# Removed: ExcelFormatter / AdvancedExcelFormatter (simple wrapper, never used in pipeline)
# Replaced by: ExcelStandardizer (genel amaçlı formatlama)
try:
    from .excel_standardizer import ExcelStandardizer, autosize_and_header, add_r2_color_scale, color_cell
    EXCEL_STANDARDIZER_AVAILABLE = True
except ImportError:
    ExcelStandardizer = None
    autosize_and_header = None
    add_r2_color_scale = None
    color_cell = None
    EXCEL_STANDARDIZER_AVAILABLE = False

__all__ = [
    'FinalReportingPipeline',
    'ExcelChartGenerator', 'EXCEL_CHARTS_AVAILABLE',
    'LaTeXReportGenerator', 'LATEX_GENERATOR_AVAILABLE',
    'ComprehensiveExcelReporter', 'COMPREHENSIVE_EXCEL_REPORTER_AVAILABLE',
    'ReportsComprehensiveModule', 'REPORTS_COMPREHENSIVE_AVAILABLE',
    'ExcelStandardizer', 'EXCEL_STANDARDIZER_AVAILABLE',
    'autosize_and_header', 'add_r2_color_scale', 'color_cell',
]
