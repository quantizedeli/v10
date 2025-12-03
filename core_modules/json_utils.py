"""
JSON Utility Functions
======================

Provides robust JSON serialization utilities for the Nuclear Physics AI Project.
Handles common serialization issues like tuple keys, numpy types, Path objects, etc.

Author: Nuclear Physics AI Project
Version: 1.0.0
Date: 2025-12-03
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects for JSON serialization.
    Converts problematic types to JSON-compatible types.

    Handles:
    - Tuple keys in dictionaries → strings
    - NumPy types (int64, float64, ndarray) → Python types
    - Path objects → strings
    - Other non-serializable objects → strings

    Args:
        obj: Object to sanitize (dict, list, or primitive)

    Returns:
        JSON-serializable version of obj

    Examples:
        >>> sanitize_for_json({(1, 2): "value"})
        {"(1, 2)": "value"}

        >>> sanitize_for_json({"key": np.int64(42)})
        {"key": 42}
    """
    if isinstance(obj, dict):
        # Convert any tuple keys to string representations
        sanitized = {}
        for key, value in obj.items():
            # Convert tuple keys to string
            if isinstance(key, tuple):
                key = str(key)
            # Ensure key is JSON-serializable
            elif not isinstance(key, (str, int, float, bool, type(None))):
                key = str(key)
            # Recursively sanitize value
            sanitized[key] = sanitize_for_json(value)
        return sanitized

    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]

    # NumPy integer types
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)

    # NumPy floating types
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)

    # NumPy boolean
    elif isinstance(obj, np.bool_):
        return bool(obj)

    # NumPy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # Path objects
    elif isinstance(obj, Path):
        return str(obj)

    # Already JSON-serializable primitives
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Fallback: convert to string
    else:
        return str(obj)


def safe_json_dump(obj: Any, filepath: Union[str, Path], indent: int = 2, **kwargs) -> None:
    """
    Safely dump object to JSON file with automatic sanitization.

    Args:
        obj: Object to save
        filepath: Output file path
        indent: JSON indentation (default: 2)
        **kwargs: Additional arguments for json.dump

    Raises:
        IOError: If file cannot be written
    """
    sanitized = sanitize_for_json(obj)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sanitized, f, indent=indent, **kwargs)


def safe_json_dumps(obj: Any, indent: int = 2, **kwargs) -> str:
    """
    Safely convert object to JSON string with automatic sanitization.

    Args:
        obj: Object to serialize
        indent: JSON indentation (default: 2)
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string
    """
    sanitized = sanitize_for_json(obj)
    return json.dumps(sanitized, indent=indent, **kwargs)


def validate_json_serializable(obj: Any, path: str = "root") -> List[str]:
    """
    Validate if an object is JSON serializable and report issues.

    Args:
        obj: Object to validate
        path: Current path in object hierarchy (for error reporting)

    Returns:
        List of error messages (empty if valid)

    Examples:
        >>> errors = validate_json_serializable({"key": np.int64(42)})
        >>> if errors:
        ...     print("Issues found:", errors)
    """
    issues = []

    try:
        json.dumps(obj)
        return []  # Valid!
    except (TypeError, ValueError) as e:
        pass  # Continue to detailed checking

    if isinstance(obj, dict):
        for key, value in obj.items():
            # Check key
            if not isinstance(key, (str, int, float, bool, type(None))):
                issues.append(f"{path}: Invalid key type {type(key).__name__} for key {key}")

            # Check value recursively
            key_str = str(key) if not isinstance(key, str) else key
            issues.extend(validate_json_serializable(value, f"{path}.{key_str}"))

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            issues.extend(validate_json_serializable(item, f"{path}[{i}]"))

    elif isinstance(obj, (np.ndarray, np.integer, np.floating, np.bool_)):
        issues.append(f"{path}: NumPy type {type(obj).__name__} not JSON-serializable")

    elif isinstance(obj, Path):
        issues.append(f"{path}: Path object not JSON-serializable")

    elif not isinstance(obj, (str, int, float, bool, type(None))):
        issues.append(f"{path}: Type {type(obj).__name__} not JSON-serializable")

    return issues


# Convenience exports
__all__ = [
    'sanitize_for_json',
    'safe_json_dump',
    'safe_json_dumps',
    'validate_json_serializable'
]


if __name__ == "__main__":
    # Quick test
    print("JSON Utils Module - Self Test")
    print("=" * 50)

    test_data = {
        "string": "hello",
        "int": 42,
        "float": 3.14,
        "numpy_int": np.int64(100),
        "numpy_float": np.float64(2.718),
        "path": Path("/tmp/test.txt"),
        (1, 2): "tuple_key",
        "nested": {
            "array": np.array([1, 2, 3]),
            "list": [1, 2, np.int32(3)]
        }
    }

    print("Original data has issues:")
    issues = validate_json_serializable(test_data)
    for issue in issues:
        print(f"  - {issue}")

    print("\nSanitized data:")
    sanitized = sanitize_for_json(test_data)
    print(json.dumps(sanitized, indent=2))

    print("\n✅ JSON Utils Module ready!")
