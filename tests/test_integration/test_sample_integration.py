"""
Integration Tests - Sample Tests
=================================

Tests for component interactions

Author: Nuclear Physics AI QA Team
"""

import pytest
import json
from pathlib import Path


@pytest.mark.integration
def test_config_and_data_integration(config_path, data_file_path):
    """Test config file references correct data file"""
    with open(config_path) as f:
        config = json.load(f)

    # Get data file name from config
    data_file_name = config.get('data', {}).get('input_file', 'aaa2.txt')

    # Check if file exists
    expected_path = data_file_path.parent / data_file_name
    assert expected_path.exists(), f"Data file {data_file_name} not found"


@pytest.mark.integration
@pytest.mark.slow
def test_data_loading_pipeline(data_file_path):
    """Test basic data loading pipeline"""
    # This is a placeholder - actual implementation would load data
    assert data_file_path.exists(), "Data file should exist"

    # Check file is not empty
    assert data_file_path.stat().st_size > 0, "Data file should not be empty"

    # Check file has content (basic check)
    with open(data_file_path) as f:
        first_line = f.readline()
        assert len(first_line) > 0, "Data file should have content"
