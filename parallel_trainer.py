"""
Parallel ANFIS Training for Multiple Datasets
ANFIS-specific: Uses train.csv, check.csv, test.csv structure
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelANFISTrainer:
    """
    Parallel ANFIS training for multiple datasets
    Works with ANFIS dataset structure: train.csv, check.csv, test.csv
    """
    
    def __init__(self, n_workers: int = None, force_maximum: bool = False):
        """
        Args:
            n_workers: Number of parallel workers (None = auto)
            force_maximum: Use all available cores minus 1
        """
        if force_maximum:
            self.n_workers = cpu_count() - 1
        elif n_workers:
            self.n_workers = n_workers
        else:
            self.n_workers = max(1, cpu_count() - 2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 PARALLEL ANFIS TRAINER INITIALIZED")
        logger.info(f"{'='*60}")
        logger.info(f"   Workers: {self.n_workers}/{cpu_count()} cores")
        logger.info(f"   Maximum mode: {force_maximum}")
        logger.info(f"{'='*60}\n")
    
    def train_multiple_datasets(self,
                               dataset_paths: List[Path],
                               configs: List[Dict],
                               output_base_dir: Path) -> List[Dict]:
        """
        Train ANFIS on multiple datasets with multiple configs in parallel
        
        Args:
            dataset_paths: List of Path objects to dataset directories
                          Each should contain: train.csv, check.csv, test.csv
            configs: List of FIS configurations from ANFISConfigManager
            output_base_dir: Base directory for outputs
            
        Returns:
            List of training results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"PARALLEL ANFIS TRAINING STARTING")
        logger.info(f"{'='*60}")
        logger.info(f"   Datasets: {len(dataset_paths)}")
        logger.info(f"   Configs per dataset: {len(configs)}")
        logger.info(f"   Total jobs: {len(dataset_paths) * len(configs)}")
        logger.info(f"   Workers: {self.n_workers}")
        logger.info(f"{'='*60}\n")
        
        # Validate datasets
        valid_datasets = []
        for ds_path in dataset_paths:
            if self._validate_dataset(ds_path):
                valid_datasets.append(ds_path)
            else:
                logger.warning(f"⚠️  Invalid dataset: {ds_path.name}")
        
        logger.info(f"Valid datasets: {len(valid_datasets)}/{len(dataset_paths)}")
        
        # Create job list
        jobs = []
        for ds_idx, ds_path in enumerate(valid_datasets):
            for cfg in configs:
                output_dir = output_base_dir / ds_path.name / cfg['id']
                output_dir.mkdir(parents=True, exist_ok=True)
                
                jobs.append({
                    'dataset_idx': ds_idx,
                    'dataset_path': ds_path,
                    'dataset_name': ds_path.name,
                    'config': cfg,
                    'output_dir': output_dir
                })
        
        logger.info(f"Total jobs created: {len(jobs)}")
        
        # Parallel execution
        start_time = time.time()
        
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(self._train_single_job, jobs),
                total=len(jobs),
                desc="🔄 Training ANFIS models",
                unit="job",
                ncols=100
            ))
        
        elapsed = time.time() - start_time
        
        # Statistics
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ PARALLEL TRAINING COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"   Total jobs: {len(results)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total time: {elapsed/60:.2f} minutes")
        logger.info(f"   Avg time/job: {elapsed/len(results):.2f} seconds")
        logger.info(f"{'='*60}\n")
        
        return results
    
    @staticmethod
    def _validate_dataset(dataset_path: Path) -> bool:
        """Validate ANFIS dataset structure"""
        required_files = ['train.csv', 'check.csv', 'test.csv']
        return all((dataset_path / f).exists() for f in required_files)
    
    @staticmethod
    def _train_single_job(job: Dict) -> Dict:
        """
        Train single ANFIS model (worker function)
        This runs in separate process
        """
        try:
            from matlab_anfis_trainer import MATLABAnfisTrainer
            
            dataset_path = job['dataset_path']
            config = job['config']
            output_dir = job['output_dir']
            
            # File paths
            train_file = dataset_path / 'train.csv'
            check_file = dataset_path / 'check.csv'
            test_file = dataset_path / 'test.csv'
            
            # Initialize trainer
            trainer = MATLABAnfisTrainer()
            
            # Load data
            import pandas as pd
            train_df = pd.read_csv(train_file)
            check_df = pd.read_csv(check_file)
            test_df = pd.read_csv(test_file)
            
            # Prepare data (assume last column is target)
            X_train = train_df.iloc[:, :-1].values
            y_train = train_df.iloc[:, -1].values
            X_val = check_df.iloc[:, :-1].values
            y_val = check_df.iloc[:, -1].values
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values
            
            # Train ANFIS with specific config
            result = trainer.train_anfis(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_clusters=config.get('mfs', 3),
                max_epochs=config.get('epochs', 100)
            )
            
            # Add job info
            result['success'] = True
            result['dataset_idx'] = job['dataset_idx']
            result['dataset_name'] = job['dataset_name']
            result['config_id'] = config['id']
            result['config_name'] = config['name']
            
            # Test evaluation
            y_test_pred = trainer.predict(X_test)
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            result['test_mae'] = float(mean_absolute_error(y_test, y_test_pred))
            result['test_rmse'] = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            result['test_r2'] = float(r2_score(y_test, y_test_pred))
            
            # Save model
            model_file = output_dir / f"model_{config['id']}.fis"
            trainer.save_fis(str(model_file))
            result['model_file'] = str(model_file)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Job failed: {job['dataset_name']} - {job['config']['id']}: {e}")
            return {
                'success': False,
                'dataset_idx': job['dataset_idx'],
                'dataset_name': job['dataset_name'],
                'config_id': job['config']['id'],
                'config_name': job['config']['name'],
                'error': str(e)
            }
    
    def train_with_robustness(self,
                             dataset_paths: List[Path],
                             configs: List[Dict],
                             output_base_dir: Path,
                             source_data_path: str = None) -> List[Dict]:
        """
        Train with iterative robustness testing
        
        Args:
            dataset_paths: Dataset directories
            configs: FIS configurations
            output_base_dir: Output directory
            source_data_path: Path to source data for nucleus replacement
            
        Returns:
            List of results with robustness info
        """
        from robustness_tester import RobustnessTester
        
        logger.info("🛡️  Training with robustness testing enabled")
        
        # Create robustness tester
        tester = RobustnessTester(config={
            'max_iterations': 5,
            'outlier_threshold': 2.0,
            'retrain_thresholds': {
                'critical': 0.3,
                'poor': 0.5,
                'acceptable': 0.7,
                'good': 0.85
            }
        })
        
        # Load source data if provided
        source_data = None
        if source_data_path:
            import pandas as pd
            source_data = pd.read_csv(source_data_path, sep='\t')
            logger.info(f"Source data loaded: {len(source_data)} samples")
        
        # Modified job creation with robustness
        jobs = []
        for ds_idx, ds_path in enumerate(dataset_paths):
            if not self._validate_dataset(ds_path):
                continue
                
            for cfg in configs:
                output_dir = output_base_dir / ds_path.name / cfg['id']
                output_dir.mkdir(parents=True, exist_ok=True)
                
                jobs.append({
                    'dataset_idx': ds_idx,
                    'dataset_path': ds_path,
                    'dataset_name': ds_path.name,
                    'config': cfg,
                    'output_dir': output_dir,
                    'tester': tester,
                    'source_data': source_data
                })
        
        # Train with robustness
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(self._train_with_robustness_worker, jobs),
                total=len(jobs),
                desc="🛡️  Training with robustness",
                unit="job"
            ))
        
        return results
    
    @staticmethod
    def _train_with_robustness_worker(job: Dict) -> Dict:
        """Worker for robustness training"""
        # Similar to _train_single_job but with robustness
        # Implementation would include iterative cleaning
        pass


if __name__ == "__main__":
    # Test parallel trainer
    logger.info("Testing Parallel ANFIS Trainer")
    
    # Mock dataset paths
    base_path = Path("test_datasets")
    dataset_paths = [base_path / f"dataset_{i}" for i in range(10)]
    
    # Mock configs
    configs = [
        {'id': 'CFG001', 'name': 'Grid_2MF_Gauss', 'mfs': 2, 'epochs': 50},
        {'id': 'CFG002', 'name': 'Grid_2MF_Bell', 'mfs': 2, 'epochs': 50},
    ]
    
    # Initialize
    trainer = ParallelANFISTrainer(force_maximum=True)
    
    logger.info("✅ Parallel trainer initialized successfully")
    logger.info(f"Ready to train {len(dataset_paths)} datasets with {len(configs)} configs")
    logger.info(f"Total jobs: {len(dataset_paths) * len(configs)}")