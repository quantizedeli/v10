"""
Unified Comprehensive Logging System
=====================================

Provides structured logging with:
- JSON-formatted logs for machine parsing
- Real-time training metrics tracking
- Per-module log level configuration
- Performance monitoring
- Automatic log rotation
- Integration with TensorBoard and MLflow
"""

import logging
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback

try:
    import numpy as np
except ImportError:
    np = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics
        if hasattr(record, 'phase'):
            log_data['phase'] = record.phase
        if hasattr(record, 'model'):
            log_data['model'] = record.model
        if hasattr(record, 'duration'):
            log_data['duration'] = record.duration

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }

    def __init__(self, use_color: bool = True):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors"""
        if self.use_color:
            levelname = record.levelname
            record.levelname = f"{self.COLORS.get(levelname, '')}{levelname}{self.COLORS['RESET']}"

        result = super().format(record)

        # Add metrics if present
        if hasattr(record, 'metrics'):
            metrics_str = ', '.join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in record.metrics.items())
            result += f" | {metrics_str}"

        return result


class MetricsLogger:
    """Tracks and logs training metrics"""

    def __init__(self, log_dir: Path, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.use_tensorboard = use_tensorboard and SummaryWriter is not None

        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        else:
            self.tb_writer = None

        self.current_epoch = 0
        self.start_time = time.time()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None,
                    phase: str = 'train', model_name: str = None):
        """Log metrics to file and TensorBoard"""
        if step is None:
            step = self.current_epoch

        # Create metrics record
        record = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'phase': phase,
            'metrics': metrics,
            'elapsed_time': time.time() - self.start_time
        }

        if model_name:
            record['model'] = model_name

        # Write to JSONL file
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')

        # Write to TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                tag = f"{phase}/{model_name}/{name}" if model_name else f"{phase}/{name}"
                self.tb_writer.add_scalar(tag, value, step)

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float] = None):
        """Log hyperparameters"""
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics or {})

        # Write to file
        hparams_file = self.log_dir / 'hyperparameters.json'
        with open(hparams_file, 'w', encoding='utf-8') as f:
            json.dumps(hparams, f, indent=2, default=str)

    def log_model_graph(self, model, input_sample):
        """Log model architecture to TensorBoard"""
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_sample)
            except Exception as e:
                logging.warning(f"Could not log model graph: {e}")

    def close(self):
        """Close TensorBoard writer"""
        if self.tb_writer:
            self.tb_writer.close()


class UnifiedLogger:
    """Unified logging system for the entire pipeline"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.loggers: Dict[str, logging.Logger] = {}
        self.metrics_logger: Optional[MetricsLogger] = None
        self.log_dir: Optional[Path] = None
        self._initialized = True

    def setup(self, log_dir: Union[str, Path],
              default_level: int = logging.INFO,
              use_structured: bool = True,
              use_color: bool = True,
              use_tensorboard: bool = True,
              module_levels: Optional[Dict[str, int]] = None):
        """
        Setup unified logging system

        Args:
            log_dir: Directory for log files
            default_level: Default logging level
            use_structured: Use JSON structured logging for files
            use_color: Use colored output for console
            use_tensorboard: Enable TensorBoard logging
            module_levels: Per-module log levels, e.g., {'pfaz02': logging.DEBUG}
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create metrics logger
        self.metrics_logger = MetricsLogger(
            self.log_dir / 'metrics',
            use_tensorboard=use_tensorboard
        )

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

        # Remove existing handlers
        root_logger.handlers.clear()

        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(default_level)
        console_handler.setFormatter(HumanReadableFormatter(use_color=use_color))
        root_logger.addHandler(console_handler)

        # File handler (structured JSON)
        if use_structured:
            structured_file = self.log_dir / 'structured.jsonl'
            structured_handler = RotatingFileHandler(
                structured_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10
            )
            structured_handler.setLevel(logging.DEBUG)
            structured_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(structured_handler)

        # Human-readable file handler
        readable_file = self.log_dir / 'app.log'
        readable_handler = TimedRotatingFileHandler(
            readable_file,
            when='midnight',
            interval=1,
            backupCount=30
        )
        readable_handler.setLevel(default_level)
        readable_handler.setFormatter(HumanReadableFormatter(use_color=False))
        root_logger.addHandler(readable_handler)

        # Error file handler
        error_file = self.log_dir / 'errors.log'
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)

        # Set module-specific levels
        if module_levels:
            for module_name, level in module_levels.items():
                logger = logging.getLogger(module_name)
                logger.setLevel(level)

        logging.info("Unified logging system initialized",
                    extra={'phase': 'initialization', 'log_dir': str(self.log_dir)})

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a module"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]

    def log_training_start(self, model_name: str, config: Dict[str, Any]):
        """Log training start with configuration"""
        logger = self.get_logger('training')
        logger.info(f"Starting training: {model_name}",
                   extra={'model': model_name, 'config': config})

        if self.metrics_logger:
            self.metrics_logger.log_hyperparameters(config)

    def log_training_step(self, model_name: str, epoch: int,
                         metrics: Dict[str, float], phase: str = 'train'):
        """Log a training step with metrics"""
        logger = self.get_logger('training')
        logger.info(f"Epoch {epoch} {phase}",
                   extra={'model': model_name, 'metrics': metrics, 'phase': phase})

        if self.metrics_logger:
            self.metrics_logger.log_metrics(metrics, step=epoch,
                                           phase=phase, model_name=model_name)

    def log_training_end(self, model_name: str, final_metrics: Dict[str, float],
                        duration: float):
        """Log training completion"""
        logger = self.get_logger('training')
        logger.info(f"Training completed: {model_name}",
                   extra={
                       'model': model_name,
                       'metrics': final_metrics,
                       'duration': duration
                   })

    def log_phase_start(self, phase_name: str, config: Dict[str, Any] = None):
        """Log PFAZ phase start"""
        logger = self.get_logger('pfaz')
        logger.info(f"Starting phase: {phase_name}",
                   extra={'phase': phase_name, 'config': config or {}})

    def log_phase_end(self, phase_name: str, duration: float,
                     results: Dict[str, Any] = None):
        """Log PFAZ phase completion"""
        logger = self.get_logger('pfaz')
        logger.info(f"Phase completed: {phase_name}",
                   extra={
                       'phase': phase_name,
                       'duration': duration,
                       'results': results or {}
                   })

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context"""
        logger = self.get_logger('error')
        logger.error(f"Error occurred: {str(error)}",
                    exc_info=True,
                    extra=context or {})

    def log_performance(self, operation: str, duration: float,
                       metrics: Dict[str, Any] = None):
        """Log performance metrics"""
        logger = self.get_logger('performance')
        logger.info(f"Performance: {operation}",
                   extra={
                       'operation': operation,
                       'duration': duration,
                       'metrics': metrics or {}
                   })

    def close(self):
        """Close all handlers and cleanup"""
        if self.metrics_logger:
            self.metrics_logger.close()

        # Close all handlers
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)


# Global singleton instance
_unified_logger = UnifiedLogger()


def setup_logging(log_dir: Union[str, Path] = './logs',
                 default_level: int = logging.INFO,
                 use_structured: bool = True,
                 use_color: bool = True,
                 use_tensorboard: bool = True,
                 module_levels: Optional[Dict[str, int]] = None):
    """Setup unified logging (convenience function)"""
    _unified_logger.setup(
        log_dir=log_dir,
        default_level=default_level,
        use_structured=use_structured,
        use_color=use_color,
        use_tensorboard=use_tensorboard,
        module_levels=module_levels
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger (convenience function)"""
    return _unified_logger.get_logger(name)


def log_training_metrics(model_name: str, epoch: int,
                        metrics: Dict[str, float], phase: str = 'train'):
    """Log training metrics (convenience function)"""
    _unified_logger.log_training_step(model_name, epoch, metrics, phase)


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log an error (convenience function)"""
    _unified_logger.log_error(error, context)


# Context manager for timing operations
class LoggedOperation:
    """Context manager for logging timed operations"""

    def __init__(self, operation_name: str, logger_name: str = 'performance'):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(f"Completed: {self.operation_name}",
                           extra={'duration': duration})
        else:
            self.logger.error(f"Failed: {self.operation_name}",
                            exc_info=(exc_type, exc_val, exc_tb),
                            extra={'duration': duration})

        return False  # Don't suppress exceptions


# Example usage
if __name__ == '__main__':
    # Setup logging
    setup_logging(
        log_dir='./logs',
        default_level=logging.INFO,
        module_levels={
            'pfaz02': logging.DEBUG,  # Debug level for training phase
            'performance': logging.DEBUG
        }
    )

    # Get logger
    logger = get_logger('example')

    # Log messages
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning")

    # Log with extra fields
    logger.info("Training step", extra={
        'model': 'RandomForest',
        'metrics': {'r2': 0.95, 'rmse': 0.12}
    })

    # Use context manager for timing
    with LoggedOperation('data_loading'):
        time.sleep(1)  # Simulate work

    # Log training metrics
    log_training_metrics(
        model_name='XGBoost',
        epoch=10,
        metrics={'r2': 0.92, 'rmse': 0.15, 'mae': 0.10},
        phase='validation'
    )

    # Log error
    try:
        raise ValueError("Example error")
    except ValueError as e:
        log_error(e, context={'phase': 'testing', 'model': 'TestModel'})
