# QA Testing Infrastructure

## Overview

This directory contains the Quality Assurance (QA) testing infrastructure for the Nuclear Physics AI Project.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration & shared fixtures
├── test_smoke/              # Smoke tests (quick sanity checks)
│   └── test_basic_smoke.py
├── test_units/              # Unit tests (individual functions)
│   └── test_sample_unit.py
├── test_integration/        # Integration tests (component interactions)
│   └── test_sample_integration.py
├── test_system/             # System tests (end-to-end workflows)
├── fixtures/                # Test data fixtures
└── reports/                 # Test reports output
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Smoke tests only (fastest)
pytest tests/ -m smoke

# Unit tests
pytest tests/ -m unit

# Integration tests
pytest tests/ -m integration

# System tests
pytest tests/ -m system
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run Verbose
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_smoke/test_basic_smoke.py -v
```

## Test Markers

- `@pytest.mark.smoke` - Quick sanity checks
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.system` - System tests
- `@pytest.mark.slow` - Slow tests (may take minutes)

## Adding New Tests

### Example Unit Test

```python
import pytest

@pytest.mark.unit
def test_my_function():
    result = my_function(input_data)
    assert result == expected_output
```

### Example Integration Test

```python
import pytest

@pytest.mark.integration
def test_pipeline_integration():
    # Test multiple components working together
    data = load_data()
    processed = process_data(data)
    result = analyze(processed)
    assert result.is_valid()
```

## Fixtures

Common fixtures are defined in `conftest.py`:
- `project_root` - Project root directory
- `config_path` - Path to config.json
- `data_file_path` - Path to aaa2.txt
- `sample_config` - Sample configuration dictionary
- `sample_nuclei_data` - Sample nuclei data for testing

## Continuous Integration

Tests are automatically run on every commit via GitHub Actions (when configured).

## Test Coverage Goals

- **Smoke Tests:** 100% (all basic checks pass)
- **Unit Tests:** 50%+ code coverage
- **Integration Tests:** Critical paths covered
- **System Tests:** End-to-end workflows validated

## Current Status

✅ Infrastructure setup complete
⚠️ Test suite in development
📝 Add tests as you implement features

## Best Practices

1. **Write tests first** (TDD approach when possible)
2. **Keep tests fast** (use mocking for slow operations)
3. **Test edge cases** (not just happy path)
4. **Use descriptive names** (`test_should_raise_error_when_input_invalid`)
5. **One assertion per test** (when practical)
6. **Use fixtures** (avoid code duplication)
7. **Clean up after tests** (use fixtures with yield)

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- QA_ENGINEER_MODULE_DESIGN.md (detailed QA strategy)
