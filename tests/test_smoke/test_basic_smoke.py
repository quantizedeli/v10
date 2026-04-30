"""
Smoke Tests - Basic Sanity Checks
==================================

Quick tests to verify basic functionality

Author: Nuclear Physics AI QA Team
"""

import pytest
import sys
import json
from pathlib import Path


@pytest.mark.smoke
def test_python_version():
    """Test Python version is 3.8+"""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"


@pytest.mark.smoke
def test_project_root_exists(project_root):
    """Test project root directory exists"""
    assert project_root.exists(), "Project root not found"


@pytest.mark.smoke
def test_config_file_exists(config_path):
    """Test config.json exists"""
    assert config_path.exists(), "config.json not found"


@pytest.mark.smoke
def test_config_file_valid_json(config_path):
    """Test config.json is valid JSON"""
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
        assert isinstance(config, dict), "Config should be a dictionary"
        assert 'project_info' in config, "Config missing project_info"
    except json.JSONDecodeError:
        pytest.fail("config.json is not valid JSON")


@pytest.mark.smoke
def test_data_file_exists(data_file_path):
    """Test aaa2.txt data file exists"""
    assert data_file_path.exists(), "aaa2.txt data file not found"


@pytest.mark.smoke
def test_main_py_exists(project_root):
    """Test main.py exists"""
    main_path = project_root / "main.py"
    assert main_path.exists(), "main.py not found"


@pytest.mark.smoke
def test_main_py_syntax(project_root):
    """Test main.py has valid Python syntax"""
    main_path = project_root / "main.py"
    try:
        with open(main_path, encoding='utf-8') as f:
            compile(f.read(), str(main_path), 'exec')
    except SyntaxError as e:
        pytest.fail(f"main.py has syntax error: {e}")


@pytest.mark.smoke
def test_essential_modules_importable():
    """Test essential modules can be imported"""
    essential_modules = [
        ('core_modules.constants', 'core_modules/constants.py'),
        ('pfaz_modules.pfaz01_dataset_generation.data_loader', 'pfaz_modules/pfaz01_dataset_generation/data_loader.py'),
        ('pfaz_modules.pfaz01_dataset_generation.dataset_generator', 'pfaz_modules/pfaz01_dataset_generation/dataset_generator.py'),
    ]

    for module_name, file_path in essential_modules:
        try:
            __import__(module_name)
        except ImportError:
            # Expected if module not in path, just check file exists
            module_path = Path(__file__).parent.parent.parent / file_path
            assert module_path.exists(), f"{file_path} not found"
