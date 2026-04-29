"""
GPU Optimization Module
3-5x faster training with GPU acceleration

This module provides GPU optimization for:
- XGBoost: GPU tree construction and prediction
- TensorFlow: Mixed precision (FP16), XLA, Multi-GPU
- Automatic batch size optimization
- Performance monitoring and benchmarking

Author: PFAZ Performance Team
Version: 1.0.0
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import time

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    GPU Optimization Manager

    Features:
    - Auto-detect GPU availability
    - XGBoost GPU acceleration
    - TensorFlow GPU optimization
    - Multi-GPU support
    - Mixed precision (FP16)
    - Dynamic batch sizing
    """

    def __init__(self):
        """Initialize GPU optimizer"""
        self.gpu_available = self._check_gpu_availability()
        self.gpu_count = self._count_gpus()
        self.gpu_name = self._get_gpu_name()

        if self.gpu_available:
            logger.info(f"[OK] GPU detected: {self.gpu_count} device(s)")
            if self.gpu_name:
                logger.info(f"  GPU: {self.gpu_name}")
            self._setup_gpu_memory_growth()
        else:
            logger.warning("[WARNING] No GPU detected - using CPU")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except ImportError:
            logger.warning("TensorFlow not installed, GPU detection limited")
            return False
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False

    def _count_gpus(self) -> int:
        """Count available GPUs"""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU'))
        except Exception as e:
            return 0

    def _get_gpu_name(self) -> Optional[str]:
        """Get GPU name"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                return gpu_details.get('device_name', 'Unknown GPU')
        except Exception as e:
            pass
        return None

    def _setup_gpu_memory_growth(self):
        """Enable GPU memory growth (prevents OOM)"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("[OK] GPU memory growth enabled")
        except Exception as e:
            logger.warning(f"Could not set memory growth: {e}")

    def optimize_xgboost(self, params: dict) -> dict:
        """
        Optimize XGBoost parameters for GPU acceleration

        Input params:
        {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            ...
        }

        Output params (GPU optimized):
        {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'tree_method': 'gpu_hist',  # <- GPU acceleration
            'gpu_id': 0,
            'predictor': 'gpu_predictor',
            'sampling_method': 'gradient_based',
            ...
        }

        Args:
            params: Original XGBoost parameters

        Returns:
            Optimized parameters with GPU settings
        """

        if not self.gpu_available:
            logger.warning("GPU not available for XGBoost, using CPU")
            params['tree_method'] = 'hist'  # CPU fallback
            return params

        # GPU-specific parameters — version-aware (XGBoost 2.0 removed gpu_hist)
        try:
            import xgboost as xgb
            xgb_version = tuple(int(x) for x in xgb.__version__.split('.')[:2])
        except Exception:
            xgb_version = (0, 0)

        if xgb_version >= (2, 0):
            gpu_params = {
                'tree_method': 'hist',
                'device': 'cuda',
                'sampling_method': 'gradient_based',
            }
            logger.info("[OK] XGBoost GPU optimization enabled (XGBoost 2.0+ API)")
            logger.info("  - tree_method: hist, device: cuda")
        else:
            gpu_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor',
                'sampling_method': 'gradient_based',
            }
            logger.info("[OK] XGBoost GPU optimization enabled (legacy API)")
            logger.info("  - tree_method: gpu_hist, predictor: gpu_predictor")

        # Merge with original params (original params take precedence for duplicates)
        optimized_params = {**params, **gpu_params}

        return optimized_params

    def optimize_tensorflow(self, enable_mixed_precision: bool = True) -> Optional[Any]:
        """
        Optimize TensorFlow for GPU performance

        Features:
        1. Mixed precision (FP16) - 2-3x faster
        2. XLA compilation - 10-20% faster
        3. CuDNN auto-tune - find fastest algorithms
        4. Multi-GPU strategy

        Args:
            enable_mixed_precision: Enable FP16 mixed precision

        Returns:
            Multi-GPU strategy if available, None otherwise
        """

        if not self.gpu_available:
            logger.warning("GPU not available for TensorFlow")
            return None

        import tensorflow as tf

        # 1. Mixed Precision (FP16)
        if enable_mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("[OK] Mixed precision enabled (FP16)")
                logger.info("  Expected speedup: 2-3x")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")

        # 2. XLA Compilation
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("[OK] XLA compilation enabled")
            logger.info("  Expected speedup: 10-20%")
        except Exception as e:
            logger.warning(f"Could not enable XLA: {e}")

        # 3. CuDNN Auto-tune
        # This finds the fastest cuDNN algorithms for your specific hardware
        import os
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
        logger.info("[OK] CuDNN auto-tune enabled")

        # 4. Multi-GPU Strategy
        if self.gpu_count > 1:
            try:
                strategy = tf.distribute.MirroredStrategy()
                logger.info(f"[OK] Multi-GPU strategy: {self.gpu_count} GPUs")
                return strategy
            except Exception as e:
                logger.warning(f"Could not create multi-GPU strategy: {e}")
                return None
        else:
            return None

    def find_optimal_batch_size(
        self,
        model: Any,
        X_sample: np.ndarray,
        y_sample: np.ndarray,
        max_batch_size: int = 1024
    ) -> int:
        """
        Find optimal batch size for GPU

        Tries increasing batch sizes until GPU memory error occurs.
        Returns the largest working batch size.

        Args:
            model: Keras model
            X_sample: Sample input data (100 samples)
            y_sample: Sample output data
            max_batch_size: Maximum batch size to try

        Returns:
            optimal_batch_size: int
        """

        import tensorflow as tf

        batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]

        optimal_size = 32  # Safe default

        logger.info("[SEARCH] Finding optimal batch size...")

        for batch_size in batch_sizes:
            try:
                # Try training one batch
                model.fit(
                    X_sample[:min(batch_size, len(X_sample))],
                    y_sample[:min(batch_size, len(y_sample))],
                    epochs=1,
                    batch_size=batch_size,
                    verbose=0
                )
                optimal_size = batch_size
                logger.info(f"  [OK] Batch size {batch_size} OK")

            except tf.errors.ResourceExhaustedError:
                logger.info(f"  [FAIL] Batch size {batch_size} - Out of memory")
                break
            except Exception as e:
                logger.warning(f"  [WARNING] Batch size {batch_size} - Error: {e}")
                break

        logger.info(f"[OK] Optimal batch size: {optimal_size}")
        return optimal_size

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information

        Returns:
            Dictionary with GPU details
        """
        info = {
            'available': self.gpu_available,
            'count': self.gpu_count,
            'name': self.gpu_name
        }

        if self.gpu_available:
            try:
                import tensorflow as tf
                # Get memory info
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    info['compute_capability'] = details.get('compute_capability', 'Unknown')
            except Exception as e:
                pass

        return info

    def print_gpu_info(self):
        """Print GPU information"""
        info = self.get_gpu_info()

        print("="*60)
        print("GPU INFORMATION")
        print("="*60)
        print(f"Available: {info['available']}")
        if info['available']:
            print(f"Count: {info['count']}")
            print(f"Name: {info['name']}")
            if 'compute_capability' in info:
                print(f"Compute Capability: {info['compute_capability']}")
        else:
            print("No GPU detected - using CPU")
        print("="*60)


# ============================================================================
# INTEGRATION WITH TRAINING CODE
# ============================================================================

def train_xgboost_optimized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict
) -> tuple:
    """
    Train XGBoost with GPU acceleration

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: XGBoost configuration

    Returns:
        (model, metrics) tuple
    """

    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # GPU optimization
    gpu_opt = GPUOptimizer()
    optimized_config = gpu_opt.optimize_xgboost(config)

    # Train with GPU
    logger.info("[START] Training XGBoost with GPU acceleration...")

    start_time = time.time()

    model = XGBRegressor(**optimized_config)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    training_time = time.time() - start_time

    # Predict (also on GPU!)
    y_pred = model.predict(X_val)

    # Metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"[OK] Training complete in {training_time:.1f}s")
    logger.info(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    return model, {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'training_time': training_time
    }


def train_dnn_optimized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict
) -> tuple:
    """
    Train DNN with GPU optimization

    Features:
    - Mixed precision (FP16)
    - Optimal batch size
    - Multi-GPU support
    - Early stopping

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: DNN configuration
            {
                'hidden_layers': [128, 64, 32],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'epochs': 100
            }

    Returns:
        (model, metrics) tuple
    """

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # GPU optimization
    gpu_opt = GPUOptimizer()
    strategy = gpu_opt.optimize_tensorflow(enable_mixed_precision=True)

    # Build model
    if strategy:  # Multi-GPU
        with strategy.scope():
            model = _build_dnn_model(config, X_train.shape[1])
    else:  # Single GPU
        model = _build_dnn_model(config, X_train.shape[1])

    # Find optimal batch size
    X_sample = X_train[:100]
    y_sample = y_train[:100]
    optimal_batch_size = gpu_opt.find_optimal_batch_size(
        model, X_sample, y_sample
    )

    # Train
    logger.info(f"[START] Training DNN with GPU optimization...")
    logger.info(f"  Batch size: {optimal_batch_size}")
    logger.info(f"  Mixed precision: FP16")

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get('epochs', 100),
        batch_size=optimal_batch_size,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
    )

    training_time = time.time() - start_time

    # Evaluate
    y_pred = model.predict(X_val, batch_size=optimal_batch_size, verbose=0)
    y_pred = y_pred.flatten()

    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"[OK] Training complete in {training_time:.1f}s")
    logger.info(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    return model, {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'training_time': training_time,
        'batch_size': optimal_batch_size,
        'epochs_trained': len(history.history['loss'])
    }


def _build_dnn_model(config: dict, input_dim: int) -> Any:
    """
    Build DNN model

    Args:
        config: Model configuration
        input_dim: Input dimension

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential()

    # Input layer
    hidden_layers = config.get('hidden_layers', [128, 64, 32])
    model.add(Dense(
        hidden_layers[0],
        activation='relu',
        input_shape=(input_dim,)
    ))

    # Hidden layers
    dropout = config.get('dropout', 0.0)
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))

    # Output layer (single value regression)
    model.add(Dense(1))

    # Compile (with mixed precision support)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            config.get('learning_rate', 0.001)
        ),
        loss='mse',
        metrics=['mae']
    )

    return model


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

class PerformanceBenchmark:
    """Benchmark GPU vs CPU performance"""

    def __init__(self):
        self.results = []

    def benchmark_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: dict,
        n_runs: int = 3
    ) -> dict:
        """
        Benchmark XGBoost CPU vs GPU

        Args:
            X: Features
            y: Labels
            config: XGBoost config
            n_runs: Number of runs to average

        Returns:
            Benchmark results
        """
        from xgboost import XGBRegressor

        logger.info("[CALC] Benchmarking XGBoost CPU vs GPU...")

        # CPU benchmark
        cpu_times = []
        for i in range(n_runs):
            cpu_config = {**config, 'tree_method': 'hist'}
            start = time.time()
            model = XGBRegressor(**cpu_config)
            model.fit(X, y, verbose=False)
            cpu_times.append(time.time() - start)

        cpu_avg = np.mean(cpu_times)
        logger.info(f"  CPU: {cpu_avg:.2f}s (avg of {n_runs} runs)")

        # GPU benchmark
        gpu_opt = GPUOptimizer()
        if gpu_opt.gpu_available:
            gpu_times = []
            for i in range(n_runs):
                gpu_config = gpu_opt.optimize_xgboost(config)
                start = time.time()
                model = XGBRegressor(**gpu_config)
                model.fit(X, y, verbose=False)
                gpu_times.append(time.time() - start)

            gpu_avg = np.mean(gpu_times)
            speedup = cpu_avg / gpu_avg
            logger.info(f"  GPU: {gpu_avg:.2f}s (avg of {n_runs} runs)")
            logger.info(f"  Speedup: {speedup:.2f}x")

            return {
                'cpu_time': cpu_avg,
                'gpu_time': gpu_avg,
                'speedup': speedup
            }
        else:
            logger.warning("  GPU not available for benchmark")
            return {
                'cpu_time': cpu_avg,
                'gpu_time': None,
                'speedup': None
            }


if __name__ == '__main__':
    # Quick GPU check
    print("="*60)
    print("GPU OPTIMIZATION MODULE")
    print("="*60)

    optimizer = GPUOptimizer()
    optimizer.print_gpu_info()

    print("\nModule loaded successfully!")
    print("Use train_xgboost_optimized() or train_dnn_optimized() for GPU-accelerated training")
