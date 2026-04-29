# PFAZ 11: Production Deployment

# ================================



# Main pipeline (ACTIVE)

try:
    from .production_model_serving import ModelServingManager
    ProductionModelServer = ModelServingManager  # backward compat alias
    PRODUCTION_MODEL_SERVER_AVAILABLE = True
except ImportError as _e:
    import logging as _logging
    _logging.warning(f'ProductionModelServer not available: {_e}')
    ModelServingManager = None
    ProductionModelServer = None
    PRODUCTION_MODEL_SERVER_AVAILABLE = False



#  ACTIVATED MODULES: Production Tools (3 modüller aktif edildi)

try:

    from .production_cicd_pipeline import ProductionCICDPipeline

    PRODUCTION_CICD_AVAILABLE = True

except ImportError:

    ProductionCICDPipeline = None

    PRODUCTION_CICD_AVAILABLE = False



try:

    from .production_web_interface import ProductionWebInterface

    PRODUCTION_WEB_INTERFACE_AVAILABLE = True

except ImportError:

    ProductionWebInterface = None

    PRODUCTION_WEB_INTERFACE_AVAILABLE = False



try:

    from .production_monitoring_system import ProductionMonitoringSystem

    PRODUCTION_MONITORING_AVAILABLE = True

except ImportError:

    ProductionMonitoringSystem = None

    PRODUCTION_MONITORING_AVAILABLE = False



__all__ = [

    'ProductionModelServer',

    'ProductionCICDPipeline', 'PRODUCTION_CICD_AVAILABLE',

    'ProductionWebInterface', 'PRODUCTION_WEB_INTERFACE_AVAILABLE',

    'ProductionMonitoringSystem', 'PRODUCTION_MONITORING_AVAILABLE',

]

