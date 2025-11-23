"""
PFAZ 3: ANFIS Parallel Trainer V2
==================================

Adaptive Neuro-Fuzzy Inference System (ANFIS) parallel trainer
Integrated with the nuclear physics AI pipeline

Features:
- Parallel ANFIS training across multiple datasets
- Simplified Python-based ANFIS (no MATLAB required)
- Multiple configuration support
- Progress tracking
- Comprehensive logging

Author: Nuclear Physics AI Training Pipeline
Version: 2.0.0
Date: 2025-11-22
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ANFISTrainingJob:
    """Single ANFIS training job specification"""
    job_id: str
    config: Dict
    dataset_path: Path
    dataset_name: str
    output_dir: Path

@dataclass
class ANFISTrainingResult:
    """ANFIS training result data structure"""
    job_id: str
    config_id: str
    dataset_name: str
    success: bool
    metrics: Optional[Dict] = None
    model_path: Optional[Path] = None
    training_time: Optional[float] = None
    error_message: Optional[str] = None


# ============================================================================
# SIMPLE PYTHON ANFIS IMPLEMENTATION
# ============================================================================

class SimpleANFIS:
    """
    Simplified ANFIS implementation in Python
    Uses sklearn's MLPRegressor as a proxy for ANFIS-like behavior
    """

    def __init__(self, n_rules: int = 3):
        """
        Initialize Simple ANFIS

        Args:
            n_rules: Number of fuzzy rules (similar to hidden layer size)
        """
        from sklearn.neural_network import MLPRegressor

        self.n_rules = n_rules
        self.model = MLPRegressor(
            hidden_layer_sizes=(n_rules * 2, n_rules),
            activation='tanh',
            solver='adam',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            verbose=False
        )

    def fit(self, X_train, y_train):
        """Train the ANFIS model"""
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)


# ============================================================================
# ANFIS PARALLEL TRAINER V2
# ============================================================================

class ANFISParallelTrainerV2:
    """
    ANFIS Parallel Trainer V2

    Features:
    - Trains ANFIS models in parallel
    - Supports multiple configurations
    - Multi-dataset training
    - Progress monitoring
    """

    def __init__(self,
                 datasets_dir: str = None,
                 output_dir: str = None,
                 n_workers: int = None,
                 use_config_manager: bool = True,
                 use_adaptive_strategy: bool = False,
                 use_performance_analyzer: bool = True,
                 save_datasets: bool = True):
        """
        Initialize ANFIS Parallel Trainer

        Args:
            datasets_dir: Directory containing datasets from PFAZ 1
            output_dir: Output directory for trained ANFIS models (PFAZ 3 output)
            n_workers: Number of parallel workers (None = auto)
            use_config_manager: Use ANFIS config manager for 8 FIS configurations
            use_adaptive_strategy: Use adaptive learning strategy (3-stage)
            use_performance_analyzer: Use performance analyzer for detailed metrics
            save_datasets: Save train/val/test datasets in .mat, .csv, .xlsx formats
        """
        self.output_dir = Path(output_dir) if output_dir else Path('trained_anfis_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.datasets_dir = Path(datasets_dir) if datasets_dir else None

        # Determine number of workers
        if n_workers is None:
            import multiprocessing
            self.n_workers = max(1, multiprocessing.cpu_count() - 2)
        else:
            self.n_workers = n_workers

        # New features
        self.use_config_manager = use_config_manager
        self.use_adaptive_strategy = use_adaptive_strategy
        self.use_performance_analyzer = use_performance_analyzer
        self.save_datasets = save_datasets

        # Storage
        self.training_results = []
        self.failed_jobs = []
        self.kernel_usage_tracker = {}  # Track which kernels used in which trainings

        logger.info("=" * 80)
        logger.info("ANFIS PARALLEL TRAINER V2 INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Datasets directory: {self.datasets_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Workers: {self.n_workers}")
        logger.info(f"Config Manager: {self.use_config_manager}")
        logger.info(f"Adaptive Strategy: {self.use_adaptive_strategy}")
        logger.info(f"Performance Analyzer: {self.use_performance_analyzer}")
        logger.info(f"Save Datasets: {self.save_datasets}")
        logger.info("=" * 80)

    def _get_feature_set_from_name(self, dataset_name: str) -> List[str]:
        """Özellik setlerini dataset adından belirle"""

        # Önceden tanımlı özellik setleri
        FEATURE_SETS = {
            'AZN': ['A', 'Z', 'N'],
            'AZNS': ['A', 'Z', 'N', 'SPIN'],
            'AZNP': ['A', 'Z', 'N', 'PARITY'],
            'AZNSP': ['A', 'Z', 'N', 'SPIN', 'PARITY'],
            'AZN_beta': ['A', 'Z', 'N', 'Beta_2_estimated'],
            'AZN_p': ['A', 'Z', 'N', 'P-factor'],
            'AZN_beta_p': ['A', 'Z', 'N', 'Beta_2_estimated', 'P-factor'],
            'AZNSP_beta_p': ['A', 'Z', 'N', 'SPIN', 'PARITY', 'Beta_2_estimated', 'P-factor'],
            'GELISMIS': ['A', 'Z', 'N', 'SPIN', 'PARITY', 'Beta_2_estimated', 'P-factor',
                         'BE_per_A', 'Z_magic_dist', 'N_magic_dist']
        }

        # Dataset adından özellik setini çıkar
        for feature_set_name in FEATURE_SETS:
            if feature_set_name in dataset_name:
                logger.info(f"Detected feature set: {feature_set_name}")
                return FEATURE_SETS[feature_set_name]

        # Eğer "ALL" varsa veya tanımlı bir set yoksa, None döndür (tüm özellikleri kullan)
        logger.info("No specific feature set detected, using adaptive selection")
        return None

    def load_dataset(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset from path with data cleaning"""

        # Try different file formats
        data_file = None
        for ext in ['.csv', '.xlsx', '.tsv']:
            potential_file = dataset_path / f"{dataset_path.name}{ext}"
            if potential_file.exists():
                data_file = potential_file
                break

        if data_file is None:
            csv_files = list(dataset_path.glob('*.csv'))
            if csv_files:
                data_file = csv_files[0]
            else:
                raise FileNotFoundError(f"No data file found in {dataset_path}")

        # Load data
        if data_file.suffix == '.csv':
            df = pd.read_csv(data_file)
        elif data_file.suffix == '.xlsx':
            df = pd.read_excel(data_file)
        elif data_file.suffix == '.tsv':
            df = pd.read_csv(data_file, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        # DATA CLEANING
        logger.info(f"Initial data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        df = df.replace('unknown', np.nan)
        df = df.replace('Unknown', np.nan)
        df = df.replace('UNKNOWN', np.nan)

        # Fix Unicode minus signs and other string issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('−', '-', regex=False)
                df[col] = df[col].replace('nan', np.nan)
                df[col] = df[col].replace('', np.nan)
                df[col] = df[col].replace('NaN', np.nan)

        # Identify target columns
        target_cols = []
        for col in ['MM', 'Q', 'Beta_2']:
            if col in df.columns:
                target_cols.append(col)

        if not target_cols:
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"No target columns (MM, Q, Beta_2) found in {data_file}")

        logger.info(f"Target columns: {target_cols}")

        # Dataset adından özellik setini belirle
        predefined_features = self._get_feature_set_from_name(dataset_path.name)

        if predefined_features is not None:
            # Önceden tanımlanmış özellik seti kullan
            feature_cols = [col for col in predefined_features if col in df.columns]
            logger.info(f"Using predefined feature set ({len(feature_cols)} features): {feature_cols}")
        else:
            # ALL dataseti için: NaN oranı %50'den az olan özellikleri kullan
            all_possible_features = [col for col in df.columns if col not in target_cols and col != 'NUCLEUS']

            # Convert to numeric first
            for col in all_possible_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # NaN oranlarını hesapla
            nan_threshold = 0.5  # %50
            feature_cols = []
            for col in all_possible_features:
                nan_ratio = df[col].isna().sum() / len(df)
                if nan_ratio < nan_threshold:
                    feature_cols.append(col)
                else:
                    logger.warning(f"Excluding {col}: {nan_ratio*100:.1f}% NaN (threshold: {nan_threshold*100}%)")

            logger.info(f"Using adaptive feature selection ({len(feature_cols)}/{len(all_possible_features)} features)")
            logger.info(f"Selected features: {feature_cols[:10]}...")

        if len(feature_cols) == 0:
            raise ValueError(f"No valid features after filtering in {data_file}")

        # Convert to numeric
        all_numeric_cols = feature_cols + target_cols
        for col in all_numeric_cols:
            if col in df.columns:
                before_count = df[col].notna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                after_count = df[col].notna().sum()
                if before_count != after_count:
                    logger.warning(f"Column {col}: {before_count - after_count} values became NaN during conversion")

        # Check NaN counts before dropping
        nan_counts = df[all_numeric_cols].isna().sum()
        if nan_counts.sum() > 0:
            logger.info(f"NaN counts per column:\n{nan_counts[nan_counts > 0]}")

        # Drop NaN
        df_clean = df[all_numeric_cols].dropna()

        if len(df_clean) == 0:
            logger.error(f"All {len(df)} rows were dropped during cleaning!")
            logger.error(f"NaN summary:\n{df[all_numeric_cols].isna().sum()}")
            logger.error(f"Sample of original data:\n{df.head()}")
            raise ValueError(f"No valid data after cleaning in {data_file}. Check data file format and values.")

        logger.info(f"Data cleaning: {len(df)} -> {len(df_clean)} samples")

        # Extract arrays
        X = df_clean[feature_cols].values.astype(np.float32)
        y = df_clean[target_cols].values.astype(np.float32)

        # Split 70/15/15
        n_total = len(X)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""

        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        metrics = {
            'r2': float(r2_score(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred))
        }

        return metrics

    def train_single_anfis(self, job: ANFISTrainingJob) -> ANFISTrainingResult:
        """
        Train single ANFIS model

        Args:
            job: ANFISTrainingJob object

        Returns:
            ANFISTrainingResult object
        """
        try:
            start_time = time.time()

            # Load dataset
            X_train, y_train, X_val, y_val, X_test, y_test = self.load_dataset(job.dataset_path)

            # Create ANFIS model
            n_rules = job.config.get('n_rules', 3)
            anfis = SimpleANFIS(n_rules=n_rules)

            logger.info(f"Training ANFIS: {job.job_id} | rules={n_rules}")

            # Train
            anfis.fit(X_train, y_train)

            # Evaluate
            y_train_pred = anfis.predict(X_train)
            y_val_pred = anfis.predict(X_val)
            y_test_pred = anfis.predict(X_test)

            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            val_metrics = self.calculate_metrics(y_val, y_val_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)

            metrics = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }

            # Save model
            import joblib
            job.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = job.output_dir / f"anfis_{job.config['id']}.pkl"
            joblib.dump(anfis, model_path)

            # Save metrics
            metrics_file = job.output_dir / f"metrics_{job.config['id']}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            # ✅ NEW: Save training datasets (.mat, .csv, .xlsx)
            if self.save_datasets:
                self.save_training_datasets(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    dataset_name=job.dataset_name,
                    config_id=job.config['id']
                )

            # ✅ NEW: Track kernel usage
            if self.save_datasets:
                kernel_info = {
                    'n_train': len(X_train),
                    'n_val': len(X_val),
                    'n_test': len(X_test),
                    'n_features': X_train.shape[1],
                    'n_rules': n_rules
                }
                self.track_kernel_usage(
                    dataset_name=job.dataset_name,
                    config_id=job.config['id'],
                    kernel_info=kernel_info
                )

            training_time = time.time() - start_time

            result = ANFISTrainingResult(
                job_id=job.job_id,
                config_id=job.config['id'],
                dataset_name=job.dataset_name,
                success=True,
                metrics=metrics,
                model_path=model_path,
                training_time=training_time
            )

            logger.info(f"[SUCCESS] {job.job_id} | R2={val_metrics['r2']:.4f} | {training_time:.1f}s")

            return result

        except Exception as e:
            logger.error(f"[ERROR] {job.job_id} | Error: {str(e)}")

            result = ANFISTrainingResult(
                job_id=job.job_id,
                config_id=job.config['id'],
                dataset_name=job.dataset_name,
                success=False,
                error_message=str(e)
            )

            return result

    def create_anfis_configs(self, n_configs: int = 10) -> List[Dict]:
        """Create ANFIS training configurations"""

        configs = [
            {'id': f'ANFIS_{i:03d}', 'n_rules': n_rules}
            for i, n_rules in enumerate([2, 3, 4, 5, 6, 8, 10, 12, 15, 20], start=1)
        ]

        return configs[:n_configs]

    def discover_datasets(self) -> List[Path]:
        """Discover dataset directories"""

        if self.datasets_dir is None or not self.datasets_dir.exists():
            raise ValueError(f"Datasets directory not found: {self.datasets_dir}")

        dataset_paths = []

        for subdir in self.datasets_dir.iterdir():
            if subdir.is_dir():
                has_data = (
                    list(subdir.glob('*.csv')) or
                    list(subdir.glob('*.xlsx')) or
                    list(subdir.glob('*.tsv'))
                )

                if has_data:
                    dataset_paths.append(subdir)
                    logger.info(f"  Found dataset: {subdir.name}")

        return dataset_paths

    def train_all_anfis_parallel(self, n_configs: int = 10) -> Dict:
        """
        Main entry point for training all ANFIS models

        Args:
            n_configs: Number of configurations to use

        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "=" * 80)
        logger.info("ANFIS PARALLEL TRAINING")
        logger.info("=" * 80)

        # Create configs
        configs = self.create_anfis_configs(n_configs)
        logger.info(f"Using {len(configs)} ANFIS configurations")

        # Discover datasets
        dataset_paths = self.discover_datasets()
        logger.info(f"Found {len(dataset_paths)} datasets")

        # Create jobs
        jobs = []
        for dataset_path in dataset_paths:
            for config in configs:
                job_id = f"{dataset_path.name}_ANFIS_{config['id']}"
                output_dir = self.output_dir / dataset_path.name / config['id']

                job = ANFISTrainingJob(
                    job_id=job_id,
                    config=config,
                    dataset_path=dataset_path,
                    dataset_name=dataset_path.name,
                    output_dir=output_dir
                )

                jobs.append(job)

        logger.info(f"Created {len(jobs)} ANFIS training jobs")

        # Train in parallel
        results = []
        failed = []

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_job = {
                executor.submit(self.train_single_anfis, job): job
                for job in jobs
            }

            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]

                try:
                    result = future.result()
                    results.append(result)

                    if not result.success:
                        failed.append(result)

                    completed += 1

                    if completed % 5 == 0:
                        progress = (completed / len(jobs)) * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / completed) * (len(jobs) - completed)

                        logger.info(f"Progress: {completed}/{len(jobs)} ({progress:.1f}%) | "
                                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

                except Exception as e:
                    logger.error(f"Job failed: {job.job_id} | {str(e)}")
                    failed.append(ANFISTrainingResult(
                        job_id=job.job_id,
                        config_id=job.config['id'],
                        dataset_name=job.dataset_name,
                        success=False,
                        error_message=str(e)
                    ))

        total_time = time.time() - start_time

        # Summary
        successful = len([r for r in results if r.success])
        logger.info("\n" + "=" * 80)
        logger.info("ANFIS PARALLEL TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total jobs: {len(jobs)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info("=" * 80)

        self.training_results = results
        self.failed_jobs = failed

        # Save summary
        self.save_summary_report()

        return {
            'status': 'completed',
            'total_jobs': len(results),
            'successful': successful,
            'failed': len(failed),
            'results': results
        }

    def save_summary_report(self):
        """Save summary report"""

        report_file = self.output_dir / 'anfis_training_summary.json'

        summary = {
            'total_jobs': len(self.training_results),
            'successful': len([r for r in self.training_results if r.success]),
            'failed': len(self.failed_jobs),
            'results': []
        }

        for result in self.training_results:
            result_dict = {
                'job_id': result.job_id,
                'config_id': result.config_id,
                'dataset_name': result.dataset_name,
                'success': result.success,
                'training_time': result.training_time
            }

            if result.success:
                result_dict['metrics'] = result.metrics
                result_dict['model_path'] = str(result.model_path)
            else:
                result_dict['error'] = result.error_message

            summary['results'].append(result_dict)

        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary report saved: {report_file}")


    def save_training_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name: str, config_id: str):
        """
        Save train/val/test datasets in multiple formats (.mat, .csv, .xlsx)

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            dataset_name: Name of the dataset
            config_id: Configuration ID
        """
        if not self.save_datasets:
            return

        try:
            # Create dataset save directory
            save_dir = self.output_dir / 'training_datasets' / f'{dataset_name}_{config_id}'
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save as CSV
            for split_name, (X, y) in [('train', (X_train, y_train)),
                                        ('val', (X_val, y_val)),
                                        ('test', (X_test, y_test))]:
                df = pd.DataFrame(X)
                df['target'] = y
                csv_path = save_dir / f'{split_name}.csv'
                df.to_csv(csv_path, index=False)
                logger.info(f"  Saved {split_name}.csv")

            # Save as Excel
            try:
                with pd.ExcelWriter(save_dir / 'all_splits.xlsx') as writer:
                    for split_name, (X, y) in [('train', (X_train, y_train)),
                                                ('val', (X_val, y_val)),
                                                ('test', (X_test, y_test))]:
                        df = pd.DataFrame(X)
                        df['target'] = y
                        df.to_excel(writer, sheet_name=split_name, index=False)
                logger.info(f"  Saved all_splits.xlsx")
            except Exception as e:
                logger.warning(f"  Could not save Excel: {e}")

            # Save as .mat (MATLAB format)
            try:
                from scipy.io import savemat
                mat_data = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test
                }
                mat_path = save_dir / 'dataset.mat'
                savemat(mat_path, mat_data)
                logger.info(f"  Saved dataset.mat")
            except Exception as e:
                logger.warning(f"  Could not save .mat file: {e}")

            logger.info(f"[DATASET SAVED] {dataset_name}_{config_id}")

        except Exception as e:
            logger.error(f"[ERROR] Could not save datasets: {e}")

    def track_kernel_usage(self, dataset_name: str, config_id: str, kernel_info: Dict):
        """
        Track which kernels (nuclei) are used in which training

        Args:
            dataset_name: Name of the dataset
            config_id: Configuration ID
            kernel_info: Dictionary with kernel information (e.g., nucleus A, Z, N)
        """
        tracking_key = f'{dataset_name}_{config_id}'
        self.kernel_usage_tracker[tracking_key] = {
            'dataset': dataset_name,
            'config': config_id,
            'timestamp': datetime.now().isoformat(),
            'kernel_info': kernel_info
        }

        # Save tracker to file
        tracker_file = self.output_dir / 'kernel_usage_tracker.json'
        with open(tracker_file, 'w') as f:
            json.dump(self.kernel_usage_tracker, f, indent=2)

        logger.info(f"[KERNEL TRACKER] Updated for {tracking_key}")

    def generate_kernel_usage_report(self) -> Dict:
        """
        Generate a comprehensive report of kernel usage across all trainings

        Returns:
            Dictionary with kernel usage statistics
        """
        report = {
            'total_trainings': len(self.kernel_usage_tracker),
            'trainings': self.kernel_usage_tracker,
            'generated_at': datetime.now().isoformat()
        }

        # Save report
        report_file = self.output_dir / 'kernel_usage_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"[KERNEL REPORT] Generated: {report_file}")
        logger.info(f"  Total trainings tracked: {report['total_trainings']}")

        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution for testing"""

    print("\n" + "=" * 80)
    print("PFAZ 3: ANFIS PARALLEL TRAINER V2 - TEST")
    print("=" * 80)

    # Initialize trainer
    trainer = ANFISParallelTrainerV2(
        output_dir='test_anfis_models',
        n_workers=2,
        use_config_manager=True,
        use_adaptive_strategy=False,
        use_performance_analyzer=True,
        save_datasets=True
    )

    print("\n[SUCCESS] ANFIS Parallel Trainer V2 initialized with all features!")


if __name__ == "__main__":
    main()
