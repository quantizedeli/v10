# Utils Module
# =============
# Utility modules moved from root directory

# ✅ ACTIVATED: Utility Modules
try:
    from .smart_cache import SmartCache
    SMART_CACHE_AVAILABLE = True
except ImportError:
    SmartCache = None
    SMART_CACHE_AVAILABLE = False

try:
    from .checkpoint_manager import CheckpointManager
    CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    CheckpointManager = None
    CHECKPOINT_MANAGER_AVAILABLE = False

try:
    from .ai_model_checkpoint import AIModelCheckpoint
    AI_MODEL_CHECKPOINT_AVAILABLE = True
except ImportError:
    AIModelCheckpoint = None
    AI_MODEL_CHECKPOINT_AVAILABLE = False

try:
    from .file_io_utils import FileIOUtils
    FILE_IO_UTILS_AVAILABLE = True
except ImportError:
    FileIOUtils = None
    FILE_IO_UTILS_AVAILABLE = False

__all__ = [
    'SmartCache', 'SMART_CACHE_AVAILABLE',
    'CheckpointManager', 'CHECKPOINT_MANAGER_AVAILABLE',
    'AIModelCheckpoint', 'AI_MODEL_CHECKPOINT_AVAILABLE',
    'FileIOUtils', 'FILE_IO_UTILS_AVAILABLE',
]
