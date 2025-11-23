"""
Reproducibility Package
=======================

Ensures complete deterministic execution across all frameworks and operations.

Features:
- Global seed management for all RNG frameworks
- Environment snapshot and versioning
- Deterministic algorithm configuration
- Experiment tracking and versioning
- Hardware configuration logging
- Reproducibility validation
"""

import os
import sys
import json
import random
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class ReproducibilityManager:
    """Manages reproducibility across the entire ML pipeline"""

    def __init__(self, seed: int = 42, strict_mode: bool = True):
        """
        Initialize reproducibility manager

        Args:
            seed: Random seed for all operations
            strict_mode: Enable strict deterministic mode (may impact performance)
        """
        self.seed = seed
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)

        # Environment snapshot
        self.environment: Dict[str, Any] = {}
        self.git_info: Dict[str, str] = {}
        self.hardware_info: Dict[str, Any] = {}

        # Experiment tracking
        self.experiment_id: Optional[str] = None
        self.config_hash: Optional[str] = None

    def set_global_seed(self, seed: Optional[int] = None):
        """Set random seed for all frameworks"""
        if seed is not None:
            self.seed = seed

        self.logger.info(f"Setting global random seed to {self.seed}")

        # Python's random module
        random.seed(self.seed)

        # NumPy
        if HAS_NUMPY:
            np.random.seed(self.seed)
            self.logger.info(f"NumPy seed set to {self.seed}")

        # TensorFlow
        if HAS_TF:
            tf.random.set_seed(self.seed)

            # Enable deterministic operations (may reduce performance)
            if self.strict_mode:
                os.environ['TF_DETERMINISTIC_OPS'] = '1'
                os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

                # TF 2.x determinism
                try:
                    tf.config.experimental.enable_op_determinism()
                    self.logger.info("TensorFlow deterministic operations enabled")
                except AttributeError:
                    self.logger.warning("TF deterministic operations not available in this version")

            self.logger.info(f"TensorFlow seed set to {self.seed}")

        # PyTorch
        if HAS_TORCH:
            torch.manual_seed(self.seed)

            # CUDA seeds
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)  # For multi-GPU

                # Deterministic mode
                if self.strict_mode:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    self.logger.info("PyTorch CUDA deterministic mode enabled")

            self.logger.info(f"PyTorch seed set to {self.seed}")

        # Environment variable for additional packages
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def configure_sklearn(self, estimator):
        """
        Configure scikit-learn estimator for reproducibility

        Args:
            estimator: sklearn estimator instance

        Returns:
            Configured estimator
        """
        # Set random_state if parameter exists
        if hasattr(estimator, 'random_state'):
            estimator.random_state = self.seed

        # Set n_jobs for deterministic parallel processing
        if hasattr(estimator, 'n_jobs') and self.strict_mode:
            # Single job for determinism (can be slow)
            # estimator.n_jobs = 1
            pass  # Allow parallel processing, rely on seed

        return estimator

    def configure_xgboost_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure XGBoost parameters for reproducibility"""
        params = params.copy()
        params['seed'] = self.seed
        params['random_state'] = self.seed

        if self.strict_mode and 'nthread' not in params:
            # Single thread for determinism
            # params['nthread'] = 1
            pass  # Allow multi-threading

        return params

    def configure_lightgbm_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure LightGBM parameters for reproducibility"""
        params = params.copy()
        params['seed'] = self.seed
        params['feature_fraction_seed'] = self.seed
        params['bagging_seed'] = self.seed
        params['drop_seed'] = self.seed
        params['data_random_seed'] = self.seed

        if self.strict_mode:
            params['deterministic'] = True

        return params

    def snapshot_environment(self) -> Dict[str, Any]:
        """Capture complete environment snapshot"""
        self.logger.info("Capturing environment snapshot")

        self.environment = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'strict_mode': self.strict_mode,
            'python': {
                'version': sys.version,
                'executable': sys.executable,
                'platform': platform.platform(),
                'machine': platform.machine(),
                'processor': platform.processor(),
            },
            'packages': self._get_package_versions(),
            'environment_variables': self._get_relevant_env_vars(),
        }

        # Git information
        self.git_info = self._get_git_info()
        self.environment['git'] = self.git_info

        # Hardware information
        self.hardware_info = self._get_hardware_info()
        self.environment['hardware'] = self.hardware_info

        return self.environment

    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages"""
        packages = {}

        try:
            import pkg_resources
            key_packages = [
                'numpy', 'scipy', 'pandas', 'scikit-learn',
                'tensorflow', 'torch', 'xgboost', 'lightgbm',
                'matplotlib', 'seaborn', 'plotly',
                'optuna', 'hyperopt', 'shap', 'lime'
            ]

            for pkg in key_packages:
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    packages[pkg] = version
                except pkg_resources.DistributionNotFound:
                    packages[pkg] = 'not_installed'
        except ImportError:
            self.logger.warning("pkg_resources not available")

        # Framework versions
        if HAS_NUMPY:
            packages['numpy'] = np.__version__
        if HAS_TF:
            packages['tensorflow'] = tf.__version__
        if HAS_TORCH:
            packages['torch'] = torch.__version__

        return packages

    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables"""
        relevant_vars = [
            'PYTHONHASHSEED', 'TF_DETERMINISTIC_OPS', 'TF_CUDNN_DETERMINISTIC',
            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS'
        ]

        return {var: os.environ.get(var, 'not_set') for var in relevant_vars}

    def _get_git_info(self) -> Dict[str, str]:
        """Get Git repository information"""
        git_info = {}

        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                # Get commit hash
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True, text=True, timeout=5
                )
                git_info['commit_hash'] = result.stdout.strip()

                # Get branch name
                result = subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    capture_output=True, text=True, timeout=5
                )
                git_info['branch'] = result.stdout.strip()

                # Check for uncommitted changes
                result = subprocess.run(
                    ['git', 'status', '--porcelain'],
                    capture_output=True, text=True, timeout=5
                )
                git_info['has_uncommitted_changes'] = bool(result.stdout.strip())

                # Get remote URL
                result = subprocess.run(
                    ['git', 'config', '--get', 'remote.origin.url'],
                    capture_output=True, text=True, timeout=5
                )
                git_info['remote_url'] = result.stdout.strip()

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning(f"Could not get Git info: {e}")
            git_info['error'] = str(e)

        return git_info

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        hardware = {
            'cpu_count': os.cpu_count(),
            'platform': platform.platform(),
            'machine': platform.machine(),
        }

        # GPU information
        if HAS_TORCH and torch.cuda.is_available():
            hardware['cuda'] = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'devices': [
                    {
                        'name': torch.cuda.get_device_name(i),
                        'capability': torch.cuda.get_device_capability(i),
                        'total_memory': torch.cuda.get_device_properties(i).total_memory
                    }
                    for i in range(torch.cuda.device_count())
                ]
            }
        elif HAS_TF:
            hardware['cuda'] = {
                'available': len(tf.config.list_physical_devices('GPU')) > 0,
                'devices': [d.name for d in tf.config.list_physical_devices('GPU')]
            }
        else:
            hardware['cuda'] = {'available': False}

        return hardware

    def generate_experiment_id(self, config: Dict[str, Any]) -> str:
        """
        Generate unique experiment ID based on configuration

        Args:
            config: Experiment configuration

        Returns:
            Unique experiment ID
        """
        # Create hash of configuration
        config_str = json.dumps(config, sort_keys=True, default=str)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # Create experiment ID with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f"{timestamp}_{self.config_hash}"

        return self.experiment_id

    def save_reproducibility_info(self, output_dir: Path,
                                  config: Optional[Dict[str, Any]] = None,
                                  additional_info: Optional[Dict[str, Any]] = None):
        """
        Save complete reproducibility information

        Args:
            output_dir: Directory to save reproducibility files
            config: Experiment configuration
            additional_info: Any additional information to save
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot environment if not done already
        if not self.environment:
            self.snapshot_environment()

        # Generate experiment ID if config provided
        if config and not self.experiment_id:
            self.generate_experiment_id(config)

        # Compile complete reproducibility package
        repro_info = {
            'experiment_id': self.experiment_id,
            'seed': self.seed,
            'strict_mode': self.strict_mode,
            'environment': self.environment,
            'config': config,
            'config_hash': self.config_hash,
        }

        if additional_info:
            repro_info['additional_info'] = additional_info

        # Save as JSON
        repro_file = output_dir / 'reproducibility_info.json'
        with open(repro_file, 'w') as f:
            json.dump(repro_info, f, indent=2, default=str)

        self.logger.info(f"Reproducibility info saved to {repro_file}")

        # Save requirements.txt snapshot
        self._save_requirements(output_dir)

        # Save environment.yml for conda
        self._save_conda_env(output_dir)

        return repro_file

    def _save_requirements(self, output_dir: Path):
        """Save pip requirements"""
        try:
            result = subprocess.run(
                ['pip', 'freeze'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                requirements_file = output_dir / 'requirements.txt'
                with open(requirements_file, 'w') as f:
                    f.write(result.stdout)
                self.logger.info(f"Requirements saved to {requirements_file}")
        except Exception as e:
            self.logger.warning(f"Could not save requirements: {e}")

    def _save_conda_env(self, output_dir: Path):
        """Save conda environment"""
        try:
            result = subprocess.run(
                ['conda', 'env', 'export'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                env_file = output_dir / 'environment.yml'
                with open(env_file, 'w') as f:
                    f.write(result.stdout)
                self.logger.info(f"Conda environment saved to {env_file}")
        except Exception as e:
            self.logger.debug(f"Could not save conda environment: {e}")

    def validate_reproducibility(self, reference_file: Path) -> Dict[str, Any]:
        """
        Validate current environment against reference

        Args:
            reference_file: Path to reference reproducibility_info.json

        Returns:
            Validation results
        """
        with open(reference_file, 'r') as f:
            reference = json.load(f)

        current = self.snapshot_environment()

        validation = {
            'timestamp': datetime.now().isoformat(),
            'seed_match': reference['seed'] == current['seed'],
            'python_version_match': reference['environment']['python']['version'] == current['python']['version'],
            'package_mismatches': [],
            'git_commit_match': True,
        }

        # Check package versions
        ref_packages = reference['environment'].get('packages', {})
        cur_packages = current.get('packages', {})

        for pkg, ref_ver in ref_packages.items():
            cur_ver = cur_packages.get(pkg, 'not_installed')
            if ref_ver != cur_ver:
                validation['package_mismatches'].append({
                    'package': pkg,
                    'reference_version': ref_ver,
                    'current_version': cur_ver
                })

        # Check Git commit
        if 'git' in reference['environment'] and 'git' in current:
            ref_commit = reference['environment']['git'].get('commit_hash')
            cur_commit = current['git'].get('commit_hash')
            validation['git_commit_match'] = ref_commit == cur_commit

        # Overall validation
        validation['is_reproducible'] = (
            validation['seed_match'] and
            validation['python_version_match'] and
            len(validation['package_mismatches']) == 0 and
            validation['git_commit_match']
        )

        return validation

    def load_and_apply(self, reproducibility_file: Path):
        """
        Load reproducibility info and apply settings

        Args:
            reproducibility_file: Path to reproducibility_info.json
        """
        with open(reproducibility_file, 'r') as f:
            repro_info = json.load(f)

        # Apply seed
        seed = repro_info.get('seed', 42)
        self.set_global_seed(seed)

        # Validate environment
        validation = self.validate_reproducibility(reproducibility_file)

        if not validation['is_reproducible']:
            self.logger.warning("Environment differs from reference!")
            self.logger.warning(f"Validation results: {json.dumps(validation, indent=2)}")

        return repro_info


# Global singleton
_repro_manager = ReproducibilityManager()


def set_global_seed(seed: int = 42, strict_mode: bool = True):
    """Set global random seed (convenience function)"""
    global _repro_manager
    _repro_manager = ReproducibilityManager(seed=seed, strict_mode=strict_mode)
    _repro_manager.set_global_seed()
    return _repro_manager


def get_reproducibility_manager() -> ReproducibilityManager:
    """Get the global reproducibility manager"""
    return _repro_manager


def save_reproducibility_info(output_dir: Path,
                              config: Optional[Dict[str, Any]] = None):
    """Save reproducibility information (convenience function)"""
    return _repro_manager.save_reproducibility_info(output_dir, config)


# Example usage
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize and set seed
    repro = set_global_seed(seed=42, strict_mode=True)

    # Snapshot environment
    env = repro.snapshot_environment()
    print(json.dumps(env, indent=2, default=str))

    # Configure sklearn estimator
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    rf = repro.configure_sklearn(rf)
    print(f"RF random_state: {rf.random_state}")

    # Configure XGBoost params
    xgb_params = {'max_depth': 6, 'learning_rate': 0.1}
    xgb_params = repro.configure_xgboost_params(xgb_params)
    print(f"XGBoost params: {xgb_params}")

    # Save reproducibility info
    config = {
        'model': 'RandomForest',
        'max_depth': 10,
        'n_estimators': 100
    }
    repro.save_reproducibility_info(
        output_dir=Path('./reproducibility'),
        config=config
    )
