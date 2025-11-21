"""
Pytest Configuration & Fixtures
================================

Shared test fixtures and configuration for all tests

Author: Nuclear Physics AI QA Team
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Project root directory"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_path(project_root):
    """Path to config.json"""
    return project_root / "config.json"


@pytest.fixture(scope="session")
def data_file_path(project_root):
    """Path to aaa2.txt data file"""
    return project_root / "aaa2.txt"


@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {
        "project_info": {
            "name": "Nuclear Physics AI Project",
            "version": "2.0.0"
        },
        "data": {
            "input_file": "aaa2.txt",
            "total_nuclei": 267
        }
    }


@pytest.fixture
def sample_nuclei_data():
    """Sample nuclei data for testing"""
    import numpy as np
    return {
        'A': np.array([12, 16, 28, 40]),
        'Z': np.array([6, 8, 14, 20]),
        'N': np.array([6, 8, 14, 20]),
        'MM_exp': np.array([0.0, 0.0, 0.0, 0.0]),
        'QM_exp': np.array([0.0, 0.0, 0.0, 0.0])
    }


# Test configuration
def pytest_configure(config):
    """Pytest configuration"""
    config.addinivalue_line(
        "markers", "smoke: Smoke tests (quick sanity checks)"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests (individual functions)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (multiple components)"
    )
    config.addinivalue_line(
        "markers", "system: System tests (end-to-end)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (may take minutes)"
    )
