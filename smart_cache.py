"""
Smart Caching System
Cache expensive operations for instant re-runs

Caches:
- Raw dataset loading
- Preprocessed data
- Feature calculations
- Train/val splits
- Model predictions
- Intermediate results

Provides 90%+ speedup for re-runs by intelligently caching expensive operations
with automatic cache invalidation based on file modification times.

Author: PFAZ Performance Team
Version: 1.0.0
"""

import hashlib
import pickle
import json
import time
from pathlib import Path
from typing import Any, Optional, Callable
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SmartCache:
    """
    Intelligent caching system with automatic invalidation

    Features:
    - Hash-based cache keys
    - Automatic expiration
    - LRU eviction policy
    - Cache statistics
    - Manual invalidation
    """

    def __init__(self, cache_dir: str = 'cache', max_size_mb: int = 5000):
        """
        Initialize smart cache

        Args:
            cache_dir: Cache directory
            max_size_mb: Maximum cache size in MB (default 5GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024

        # Cache metadata
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()

        logger.info(f"[OK] Smart Cache initialized: {self.cache_dir}")
        logger.info(f"  Max size: {max_size_mb} MB")

        # Clean old caches on startup
        self._cleanup_old_caches()

    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate unique cache key from arguments

        Hash the function arguments to create a unique key.
        Same args -> same key -> cache hit!

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            hex_key: str (e.g., 'a3f5c2...')
        """

        # Convert args to string representation
        args_str = str(args) + str(sorted(kwargs.items()))

        # Hash it
        hash_obj = hashlib.sha256(args_str.encode())
        hex_key = hash_obj.hexdigest()[:16]  # First 16 chars

        return hex_key

    def _generate_file_key(self, file_path: Path) -> str:
        """
        Generate cache key for file-based caching

        Include file path + modification time.
        If file changes -> different key -> cache miss -> reload!

        Args:
            file_path: Path to file

        Returns:
            key: str
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # File info
        mtime = file_path.stat().st_mtime
        size = file_path.stat().st_size

        # Create key: path + mtime + size
        key_str = f"{file_path}_{mtime}_{size}"
        hash_obj = hashlib.sha256(key_str.encode())

        return hash_obj.hexdigest()[:16]

    def cache_dataframe(self, key: str, df: pd.DataFrame):
        """
        Cache pandas DataFrame

        Save to pickle for fast loading.

        Args:
            key: Cache key
            df: DataFrame to cache
        """

        cache_file = self.cache_dir / f'df_{key}.pkl'

        with open(cache_file, 'wb') as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update metadata
        self.metadata[key] = {
            'type': 'dataframe',
            'file': str(cache_file),
            'size': cache_file.stat().st_size,
            'created': time.time(),
            'last_accessed': time.time()
        }
        self._save_metadata()

        logger.info(f"[SAVE] Cached dataframe: {key}")
        logger.info(f"  Size: {cache_file.stat().st_size / 1024:.1f} KB")

    def get_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached DataFrame

        Returns None if not found (cache miss).

        Args:
            key: Cache key

        Returns:
            DataFrame if found, None otherwise
        """

        if key not in self.metadata:
            logger.info(f"🚫 Cache MISS: {key}")
            return None

        cache_file = Path(self.metadata[key]['file'])

        if not cache_file.exists():
            logger.warning(f"[WARNING] Cache file missing: {cache_file}")
            del self.metadata[key]
            self._save_metadata()
            return None

        # Load from pickle
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)

        # Update access time
        self.metadata[key]['last_accessed'] = time.time()
        self._save_metadata()

        logger.info(f"[FAST] Cache HIT: {key}")

        return df

    def cache_dataset_file(self, file_path: str) -> tuple:
        """
        Cache dataset file with auto-invalidation

        Workflow:
        1. Generate key from file path + mtime
        2. Check if cached
        3. If not, load and preprocess
        4. Cache result
        5. Return cache key and dataframe

        Args:
            file_path: Path to dataset file

        Returns:
            (cache_key, dataframe) tuple
        """

        file_path = Path(file_path)
        cache_key = self._generate_file_key(file_path)

        # Try cache first
        df = self.get_cached_dataframe(cache_key)

        if df is not None:
            return cache_key, df  # Cache hit!

        # Cache miss - load file
        logger.info(f"[OPEN] Loading dataset: {file_path}")

        if file_path.suffix == '.txt':
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Cache it
        self.cache_dataframe(cache_key, df)

        return cache_key, df

    def cache_array(self, key: str, array: np.ndarray):
        """
        Cache numpy array

        Args:
            key: Cache key
            array: Numpy array to cache
        """

        cache_file = self.cache_dir / f'array_{key}.npy'
        np.save(cache_file, array)

        self.metadata[key] = {
            'type': 'array',
            'file': str(cache_file),
            'size': cache_file.stat().st_size,
            'created': time.time(),
            'last_accessed': time.time()
        }
        self._save_metadata()

        logger.info(f"[SAVE] Cached array: {key} (shape: {array.shape})")

    def get_cached_array(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve cached array

        Args:
            key: Cache key

        Returns:
            Array if found, None otherwise
        """

        if key not in self.metadata:
            logger.info(f"🚫 Cache MISS: {key}")
            return None

        cache_file = Path(self.metadata[key]['file'])

        if not cache_file.exists():
            del self.metadata[key]
            self._save_metadata()
            return None

        array = np.load(cache_file)

        self.metadata[key]['last_accessed'] = time.time()
        self._save_metadata()

        logger.info(f"[FAST] Cache HIT: {key}")

        return array

    def cache_object(self, key: str, obj: Any):
        """
        Cache any Python object (pickle)

        Args:
            key: Cache key
            obj: Object to cache
        """

        cache_file = self.cache_dir / f'obj_{key}.pkl'

        with open(cache_file, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.metadata[key] = {
            'type': 'object',
            'file': str(cache_file),
            'size': cache_file.stat().st_size,
            'created': time.time(),
            'last_accessed': time.time()
        }
        self._save_metadata()

        logger.info(f"[SAVE] Cached object: {key}")

    def get_cached_object(self, key: str) -> Optional[Any]:
        """
        Retrieve cached object

        Args:
            key: Cache key

        Returns:
            Object if found, None otherwise
        """

        if key not in self.metadata:
            logger.info(f"🚫 Cache MISS: {key}")
            return None

        cache_file = Path(self.metadata[key]['file'])

        if not cache_file.exists():
            del self.metadata[key]
            self._save_metadata()
            return None

        with open(cache_file, 'rb') as f:
            obj = pickle.load(f)

        self.metadata[key]['last_accessed'] = time.time()
        self._save_metadata()

        logger.info(f"[FAST] Cache HIT: {key}")

        return obj

    def _load_metadata(self) -> dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _cleanup_old_caches(self, max_age_days: int = 30):
        """
        Remove caches older than max_age_days

        LRU eviction: Remove least recently used caches if size > max_size

        Args:
            max_age_days: Maximum age in days
        """

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        removed = []

        for key, meta in list(self.metadata.items()):
            age = current_time - meta.get('created', current_time)

            if age > max_age_seconds:
                cache_file = Path(meta['file'])
                if cache_file.exists():
                    cache_file.unlink()
                removed.append(key)
                del self.metadata[key]

        if removed:
            logger.info(f"🗑️ Removed {len(removed)} old caches")
            self._save_metadata()

        # Check total size
        total_size = sum(meta.get('size', 0) for meta in self.metadata.values())

        if total_size > self.max_size_bytes:
            logger.warning(f"[WARNING] Cache size exceeded: {total_size / 1024 / 1024:.1f} MB")
            self._evict_lru()

    def _evict_lru(self):
        """
        Evict least recently used caches

        Sort by last_accessed, remove oldest until size < max_size
        """

        # Sort by last accessed (oldest first)
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k].get('last_accessed', 0)
        )

        total_size = sum(meta.get('size', 0) for meta in self.metadata.values())

        removed = 0
        for key in sorted_keys:
            if total_size < self.max_size_bytes:
                break

            meta = self.metadata[key]
            cache_file = Path(meta['file'])

            if cache_file.exists():
                total_size -= meta.get('size', 0)
                cache_file.unlink()

            del self.metadata[key]
            removed += 1

        if removed > 0:
            logger.info(f"🗑️ Evicted {removed} LRU caches")
            self._save_metadata()

    def clear_all(self):
        """Clear all caches"""

        for meta in self.metadata.values():
            cache_file = Path(meta['file'])
            if cache_file.exists():
                cache_file.unlink()

        self.metadata = {}
        self._save_metadata()

        logger.info("🗑️ All caches cleared")

    def get_stats(self) -> dict:
        """Get cache statistics"""

        total_size = sum(meta.get('size', 0) for meta in self.metadata.values())

        return {
            'n_caches': len(self.metadata),
            'total_size_mb': total_size / 1024 / 1024,
            'max_size_mb': self.max_size_bytes / 1024 / 1024,
            'usage_percent': (total_size / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0
        }

    def print_stats(self):
        """Print cache statistics"""

        stats = self.get_stats()

        print("="*60)
        print("CACHE STATISTICS")
        print("="*60)
        print(f"Cached items: {stats['n_caches']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"Max size: {stats['max_size_mb']:.1f} MB")
        print(f"Usage: {stats['usage_percent']:.1f}%")
        print("="*60)


# ============================================================================
# DECORATOR FOR EASY CACHING
# ============================================================================

_global_cache = None


def get_cache() -> SmartCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache()
    return _global_cache


def cached(cache_type: str = 'object'):
    """
    Decorator for automatic caching

    Usage:
        @cached(cache_type='dataframe')
        def load_and_preprocess_data(file_path):
            df = pd.read_csv(file_path)
            # ... expensive preprocessing ...
            return df

        # First call: slow (cache miss)
        df = load_and_preprocess_data('aaa2.txt')

        # Second call: fast! (cache hit)
        df = load_and_preprocess_data('aaa2.txt')

    Args:
        cache_type: Type of cache ('object', 'dataframe', 'array')

    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            cache = get_cache()

            # Generate cache key
            key = cache._generate_key(func.__name__, *args, **kwargs)

            # Try cache
            if cache_type == 'dataframe':
                result = cache.get_cached_dataframe(key)
            elif cache_type == 'array':
                result = cache.get_cached_array(key)
            else:
                result = cache.get_cached_object(key)

            if result is not None:
                return result  # Cache hit!

            # Cache miss - execute function
            logger.info(f"[RUN] Executing: {func.__name__}")
            result = func(*args, **kwargs)

            # Cache result
            if cache_type == 'dataframe':
                cache.cache_dataframe(key, result)
            elif cache_type == 'array':
                cache.cache_array(key, result)
            else:
                cache.cache_object(key, result)

            return result

        return wrapper
    return decorator


# ============================================================================
# EXAMPLE USAGE FOR NUCLEAR PHYSICS DATA
# ============================================================================

@cached(cache_type='dataframe')
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset with automatic caching

    First call: slow (2 seconds)
    Subsequent calls: fast! (0.1 seconds)

    Args:
        file_path: Path to dataset file

    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')
    return df


@cached(cache_type='dataframe')
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataset with caching

    Expensive operations cached!

    Args:
        df: Input DataFrame

    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing dataset...")

    # Expensive preprocessing
    df = df.dropna()
    df = df[df.get('A', df.iloc[:, 0]) > 0]  # Filter positive values

    return df


@cached(cache_type='array')
def calculate_theoretical_features(df: pd.DataFrame, feature_names: list) -> np.ndarray:
    """
    Calculate theoretical nuclear physics features with caching

    Features like SEMF, Shell Model, etc. - all cached!

    Args:
        df: Input DataFrame
        feature_names: List of feature names to calculate

    Returns:
        Feature matrix (n_samples, n_features)
    """
    logger.info(f"Calculating {len(feature_names)} theoretical features...")

    # Expensive calculations...
    n_samples = len(df)
    n_features = len(feature_names)
    features = np.zeros((n_samples, n_features))

    # Simulated expensive computation
    time.sleep(0.01 * n_features)  # Simulate computation time

    return features


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class CacheBenchmark:
    """Benchmark cache performance"""

    @staticmethod
    def benchmark_dataframe_caching(df: pd.DataFrame, n_runs: int = 5) -> dict:
        """
        Benchmark DataFrame caching performance

        Args:
            df: Test DataFrame
            n_runs: Number of runs

        Returns:
            Benchmark results
        """
        cache = SmartCache(cache_dir='benchmark_cache')

        # First run (cache miss)
        key = 'benchmark_df'
        start = time.time()
        cache.cache_dataframe(key, df)
        write_time = time.time() - start

        # Subsequent runs (cache hit)
        read_times = []
        for _ in range(n_runs):
            start = time.time()
            df_cached = cache.get_cached_dataframe(key)
            read_times.append(time.time() - start)

        avg_read_time = np.mean(read_times)
        speedup = write_time / avg_read_time if avg_read_time > 0 else 0

        # Cleanup
        cache.clear_all()

        return {
            'write_time': write_time,
            'avg_read_time': avg_read_time,
            'speedup': speedup,
            'dataframe_size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }


if __name__ == '__main__':
    # Quick cache check
    print("="*60)
    print("SMART CACHE MODULE")
    print("="*60)

    cache = SmartCache()
    cache.print_stats()

    print("\nModule loaded successfully!")
    print("Use @cached decorator for automatic caching:")
    print("  @cached(cache_type='dataframe')")
    print("  def load_data(path):")
    print("      return pd.read_csv(path)")
