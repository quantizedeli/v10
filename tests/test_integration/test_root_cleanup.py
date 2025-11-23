"""
Integration Tests: Root Cleanup Verification
=============================================

Verify that root directory cleanup was successful.
"""

import pytest
from pathlib import Path


class TestRootCleanup:
    """Test root directory cleanup"""

    def test_main_py_exists(self):
        """Test main.py exists in root"""
        assert Path('main.py').exists()

    def test_run_complete_pipeline_exists(self):
        """Test run_complete_pipeline.py exists"""
        assert Path('run_complete_pipeline.py').exists()

    def test_fix_scripts_deleted(self):
        """Test fix scripts were deleted"""
        fix_scripts = [
            'fix_all_emojis.py',
            'check_and_fix_emojis.py',
            'fix_numeric_comparison_errors.py',
            'test_data_validation_fix.py',
        ]

        for script in fix_scripts:
            assert not Path(script).exists(), f"{script} should have been deleted"

    def test_duplicates_deleted(self):
        """Test duplicate files were deleted"""
        duplicates = [
            'parallel_trainer.py',
            'dataset_generation_pipeline_v2.py',
            'adaptive_strategy.py',
        ]

        for dup in duplicates:
            assert not Path(dup).exists(), f"{dup} should have been deleted (duplicate)"

    def test_scripts_directory_exists(self):
        """Test scripts directory was created"""
        assert Path('scripts').exists()
        assert Path('scripts').is_dir()

    def test_scripts_examples_directory_exists(self):
        """Test scripts/examples directory was created"""
        assert Path('scripts/examples').exists()
        assert Path('scripts/examples').is_dir()

    def test_utils_directory_exists(self):
        """Test utils directory exists"""
        assert Path('utils').exists()
        assert Path('utils').is_dir()

    def test_moved_to_scripts(self):
        """Test files were moved to scripts"""
        moved_scripts = [
            'scripts/check_pfaz_completeness.py',
            'scripts/log_parser.py',
            'scripts/generate_sample_data.py',
            'scripts/create_pfaz7_xlsx.py',
            'scripts/pfaz7_excel_reporter.py',
            'scripts/examples/example_performance_pipeline.py',
            'scripts/examples/example_usage.py',
        ]

        for script in moved_scripts:
            assert Path(script).exists(), f"{script} should exist"

    def test_moved_to_utils(self):
        """Test files were moved to utils"""
        moved_utils = [
            'utils/smart_cache.py',
            'utils/checkpoint_manager.py',
            'utils/ai_model_checkpoint.py',
        ]

        for util in moved_utils:
            assert Path(util).exists(), f"{util} should exist"

    def test_moved_to_pfaz02(self):
        """Test files were moved to PFAZ 02"""
        moved_pfaz02 = [
            'pfaz_modules/pfaz02_ai_training/gpu_optimization.py',
            'pfaz_modules/pfaz02_ai_training/training_utils_v2.py',
            'pfaz_modules/pfaz02_ai_training/robustness_tester.py',
        ]

        for file in moved_pfaz02:
            assert Path(file).exists(), f"{file} should exist"

    def test_moved_to_pfaz05(self):
        """Test files were moved to PFAZ 05"""
        assert Path('pfaz_modules/pfaz05_cross_model/optimizer_comparison_reporter.py').exists()

    def test_root_python_files_count(self):
        """Test root has minimal Python files"""
        root_py_files = list(Path('.').glob('*.py'))
        root_py_files = [f for f in root_py_files if f.is_file()]

        # Should have only main.py and run_complete_pipeline.py
        assert len(root_py_files) <= 3, f"Too many files in root: {[f.name for f in root_py_files]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
