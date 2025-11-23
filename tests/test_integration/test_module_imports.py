"""
Integration Tests: Module Import Tests
=======================================

Test that all activated modules can be imported successfully.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPFAZ01Import:
    """Test PFAZ 01 module imports"""

    def test_main_pipeline_import(self):
        """Test main pipeline import"""
        from pfaz_modules.pfaz01_dataset_generation import DatasetGenerationPipelineV2
        assert DatasetGenerationPipelineV2 is not None

    def test_optional_modules_available(self):
        """Test optional modules are importable"""
        import pfaz_modules.pfaz01_dataset_generation as pfaz01

        # Check availability flags exist
        assert hasattr(pfaz01, 'CONTROL_GROUP_GENERATOR_AVAILABLE')
        assert hasattr(pfaz01, 'DATA_ENRICHER_AVAILABLE')


class TestPFAZ02Import:
    """Test PFAZ 02 module imports"""

    def test_main_pipeline_import(self):
        """Test main pipeline import"""
        from pfaz_modules.pfaz02_ai_training import ParallelAITrainer
        assert ParallelAITrainer is not None

    def test_optional_modules_available(self):
        """Test optional modules are importable"""
        import pfaz_modules.pfaz02_ai_training as pfaz02

        # Check availability flags exist
        assert hasattr(pfaz02, 'HYPERPARAMETER_TUNER_AVAILABLE')
        assert hasattr(pfaz02, 'MODEL_VALIDATOR_AVAILABLE')
        assert hasattr(pfaz02, 'OVERFITTING_DETECTOR_AVAILABLE')
        assert hasattr(pfaz02, 'ADVANCED_MODELS_AVAILABLE')

    def test_moved_modules_available(self):
        """Test modules moved from root are available"""
        import pfaz_modules.pfaz02_ai_training as pfaz02

        assert hasattr(pfaz02, 'GPU_OPTIMIZATION_AVAILABLE')
        assert hasattr(pfaz02, 'TRAINING_UTILS_AVAILABLE')
        assert hasattr(pfaz02, 'ROBUSTNESS_TESTER_AVAILABLE')


class TestPFAZ13Import:
    """Test PFAZ 13 module imports"""

    def test_main_pipeline_import(self):
        """Test main pipeline import"""
        from pfaz_modules.pfaz13_automl import AutoMLHyperparameterOptimizer
        assert AutoMLHyperparameterOptimizer is not None

    def test_optional_modules_available(self):
        """Test optional modules are importable"""
        import pfaz_modules.pfaz13_automl as pfaz13

        # Check availability flags exist
        assert hasattr(pfaz13, 'AUTOML_VISUALIZER_AVAILABLE')
        assert hasattr(pfaz13, 'AUTOML_FEATURE_ENGINEER_AVAILABLE')
        assert hasattr(pfaz13, 'AUTOML_OPTIMIZER_AVAILABLE')


class TestUtilsImport:
    """Test utils module imports"""

    def test_utils_module_import(self):
        """Test utils module can be imported"""
        import utils
        assert utils is not None

    def test_utils_availability_flags(self):
        """Test utils availability flags"""
        import utils

        assert hasattr(utils, 'SMART_CACHE_AVAILABLE')
        assert hasattr(utils, 'CHECKPOINT_MANAGER_AVAILABLE')
        assert hasattr(utils, 'AI_MODEL_CHECKPOINT_AVAILABLE')


class TestScriptsImport:
    """Test scripts module imports"""

    def test_scripts_module_import(self):
        """Test scripts module can be imported"""
        import scripts
        assert scripts is not None

    def test_scripts_examples_import(self):
        """Test scripts.examples module can be imported"""
        import scripts.examples
        assert scripts.examples is not None


class TestAllPFAZImports:
    """Test all PFAZ modules can be imported"""

    @pytest.mark.parametrize("pfaz_num,module_name", [
        (1, 'pfaz01_dataset_generation'),
        (2, 'pfaz02_ai_training'),
        (3, 'pfaz03_anfis_training'),
        (4, 'pfaz04_unknown_predictions'),
        (5, 'pfaz05_cross_model'),
        (6, 'pfaz06_final_reporting'),
        (7, 'pfaz07_ensemble'),
        (8, 'pfaz08_visualization'),
        (9, 'pfaz09_aaa2_monte_carlo'),
        (10, 'pfaz10_thesis_compilation'),
        (11, 'pfaz11_production'),
        (12, 'pfaz12_advanced_analytics'),
        (13, 'pfaz13_automl'),
    ])
    def test_pfaz_import(self, pfaz_num, module_name):
        """Test individual PFAZ module import"""
        module = __import__(f'pfaz_modules.{module_name}', fromlist=['*'])
        assert module is not None

        # Check __all__ exists
        assert hasattr(module, '__all__')
        assert len(module.__all__) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
