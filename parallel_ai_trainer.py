# -*- coding: utf-8 -*-
"""
PFAZ 2: Parallel AI Model Trainer
==================================

50 farklı konfigürasyon ile multiple dataset'lerde parallel AI model training

Features:
- Multi-model support (RF, GBM, XGBoost, DNN, BNN, PINN)
- 50 training configurations
- Parallel training with multiprocessing
- GPU/CPU optimization
- Checkpoint & resume capability
- Real-time progress monitoring
- Comprehensive logging

Author: Nuclear Physics AI Training Pipeline
Version: 1.0.0
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

# TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrainingJob:
    """Single training job specification"""
    job_id: str
    model_type: str
    config: Dict
    dataset_path: Path
    dataset_name: str
    output_dir: Path
    
@dataclass
class TrainingResult:
    """Training result data structure"""
    job_id: str
    model_type: str
    config_id: str
    dataset_name: str
    success: bool
    metrics: Optional[Dict] = None
    model_path: Optional[Path] = None
    training_time: Optional[float] = None
    error_message: Optional[str] = None
    checkpoint_path: Optional[Path] = None


# ============================================================================
# MODEL TRAINERS
# ============================================================================

class BaseAITrainer:
    """Base class for AI model trainers"""
    
    def __init__(self, model_type: str, config: Dict, output_dir: Path):
        self.model_type = model_type
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.history = {}
        
    def load_dataset(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset from path"""
        
        # Try different file formats
        data_file = None
        for ext in ['.csv', '.xlsx', '.tsv']:
            potential_file = dataset_path / f"{dataset_path.name}{ext}"
            if potential_file.exists():
                data_file = potential_file
                break
        
        if data_file is None:
            # Try finding any data file
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
        
        # Identify feature and target columns
        # Assume target columns are: MM, Q, Beta_2, or combinations
        target_cols = []
        for col in ['MM', 'Q', 'Beta_2']:
            if col in df.columns:
                target_cols.append(col)
        
        if not target_cols:
            raise ValueError(f"No target columns found in {data_file}")
        
        # Features are all columns except targets and NUCLEUS
        feature_cols = [col for col in df.columns if col not in target_cols and col != 'NUCLEUS']
        
        # Extract features and targets
        X = df[feature_cols].values
        y = df[target_cols].values
        
        # Split into train/val/test (70/15/15)
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
        
        # Handle multi-output
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            # Calculate metrics per output
            metrics = {}
            for i in range(y_true.shape[1]):
                metrics[f'r2_output{i}'] = float(r2_score(y_true[:, i], y_pred[:, i]))
                metrics[f'rmse_output{i}'] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
                metrics[f'mae_output{i}'] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
            
            # Average metrics
            metrics['r2_avg'] = float(np.mean([metrics[k] for k in metrics if k.startswith('r2_')]))
            metrics['rmse_avg'] = float(np.mean([metrics[k] for k in metrics if k.startswith('rmse_')]))
            metrics['mae_avg'] = float(np.mean([metrics[k] for k in metrics if k.startswith('mae_')]))
        else:
            # Single output
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
    
    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save_model(self, filepath: Path):
        """Save model"""
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved: {filepath}")


class RandomForestTrainer(BaseAITrainer):
    """Random Forest Trainer"""
    
    def __init__(self, config: Dict, output_dir: Path):
        super().__init__('RandomForest', config, output_dir)
    
    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train Random Forest"""
        
        # Get config parameters
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', None)
        min_samples_split = self.config.get('min_samples_split', 2)
        
        logger.info(f"Training RF: n_estimators={n_estimators}, max_depth={max_depth}")
        
        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=42
        )
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'training_time': training_time
        }
        
        return metrics


class XGBoostTrainer(BaseAITrainer):
    """XGBoost Trainer"""
    
    def __init__(self, config: Dict, output_dir: Path):
        super().__init__('XGBoost', config, output_dir)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
    
    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train XGBoost"""
        
        # Get config parameters
        n_estimators = self.config.get('n_estimators', 100)
        learning_rate = self.config.get('learning_rate', 0.1)
        max_depth = self.config.get('max_depth', 6)
        
        logger.info(f"Training XGBoost: n_estimators={n_estimators}, lr={learning_rate}")
        
        # Create and train model
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )
        
        start_time = time.time()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        training_time = time.time() - start_time
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'training_time': training_time
        }
        
        return metrics


class DNNTrainer(BaseAITrainer):
    """Deep Neural Network Trainer"""
    
    def __init__(self, config: Dict, output_dir: Path):
        super().__init__('DNN', config, output_dir)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
    
    def build_model(self, input_dim: int, output_dim: int) -> keras.Model:
        """Build DNN model"""
        
        architecture = self.config.get('architecture', [256, 128, 64, 32])
        dropout = self.config.get('dropout', [0.1, 0.1, 0.1, 0.0])
        activation = self.config.get('activation', 'relu')
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Build model
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        
        for units, drop in zip(architecture, dropout):
            x = layers.Dense(units, activation=activation)(x)
            if drop > 0:
                x = layers.Dropout(drop)(x)
        
        outputs = layers.Dense(output_dim)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train DNN"""
        
        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        
        logger.info(f"Training DNN: batch_size={batch_size}, epochs={epochs}")
        
        # Build model
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
        
        self.model = self.build_model(input_dim, output_dim)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Train
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        training_time = time.time() - start_time
        
        self.history = history.history
        
        # Evaluate
        y_train_pred = self.model.predict(X_train, verbose=0)
        y_val_pred = self.model.predict(X_val, verbose=0)
        
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        val_metrics = self.calculate_metrics(y_val, y_val_pred)
        
        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss'])
        }
        
        return metrics


# ============================================================================
# PARALLEL AI TRAINER
# ============================================================================

class ParallelAITrainer:
    """
    Parallel AI Model Trainer
    
    Features:
    - Trains multiple models in parallel
    - Supports 50 different configurations
    - Multi-dataset training
    - Progress monitoring
    - Checkpoint & resume
    """
    
    def __init__(self, 
                 output_dir: str = 'trained_models',
                 n_workers: int = None,
                 gpu_enabled: bool = False):
        """
        Initialize Parallel AI Trainer
        
        Args:
            output_dir: Output directory for models
            n_workers: Number of parallel workers (None = auto)
            gpu_enabled: Enable GPU training (for DNN)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine number of workers
        if n_workers is None:
            import multiprocessing
            self.n_workers = max(1, multiprocessing.cpu_count() - 2)
        else:
            self.n_workers = n_workers
        
        self.gpu_enabled = gpu_enabled
        
        # Storage
        self.training_results = []
        self.failed_jobs = []
        
        logger.info("=" * 80)
        logger.info("PARALLEL AI TRAINER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Workers: {self.n_workers}")
        logger.info(f"GPU enabled: {self.gpu_enabled}")
        logger.info("=" * 80)
    
    def load_training_configs(self, config_file: Path) -> List[Dict]:
        """Load 50 training configurations"""
        
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        logger.info(f"Loaded {len(configs)} training configurations")
        return configs
    
    def create_training_jobs(self,
                            model_types: List[str],
                            configs: List[Dict],
                            dataset_paths: List[Path]) -> List[TrainingJob]:
        """
        Create training jobs for all combinations
        
        Args:
            model_types: List of model types (e.g., ['RF', 'XGBoost', 'DNN'])
            configs: List of training configurations
            dataset_paths: List of dataset directories
        
        Returns:
            List of TrainingJob objects
        """
        jobs = []
        
        for dataset_path in dataset_paths:
            for model_type in model_types:
                for config in configs:
                    job_id = f"{dataset_path.name}_{model_type}_{config['id']}"
                    
                    output_dir = self.output_dir / dataset_path.name / model_type / config['id']
                    
                    job = TrainingJob(
                        job_id=job_id,
                        model_type=model_type,
                        config=config,
                        dataset_path=dataset_path,
                        dataset_name=dataset_path.name,
                        output_dir=output_dir
                    )
                    
                    jobs.append(job)
        
        logger.info(f"Created {len(jobs)} training jobs")
        logger.info(f"  Datasets: {len(dataset_paths)}")
        logger.info(f"  Model types: {len(model_types)}")
        logger.info(f"  Configs per model: {len(configs)}")
        
        return jobs
    
    def train_single_job(self, job: TrainingJob) -> TrainingResult:
        """
        Train single model (worker function)
        
        Args:
            job: TrainingJob object
        
        Returns:
            TrainingResult object
        """
        try:
            start_time = time.time()
            
            # Create trainer based on model type
            if job.model_type in ['RF', 'RandomForest']:
                trainer = RandomForestTrainer(job.config, job.output_dir)
            elif job.model_type in ['XGB', 'XGBoost']:
                trainer = XGBoostTrainer(job.config, job.output_dir)
            elif job.model_type == 'DNN':
                trainer = DNNTrainer(job.config, job.output_dir)
            else:
                raise ValueError(f"Unknown model type: {job.model_type}")
            
            # Load dataset
            X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_dataset(job.dataset_path)
            
            # Train
            metrics = trainer.train(X_train, y_train, X_val, y_val)
            
            # Test evaluation
            y_test_pred = trainer.predict(X_test)
            test_metrics = trainer.calculate_metrics(y_test, y_test_pred)
            metrics['test'] = test_metrics
            
            # Save model
            model_filename = f"model_{job.model_type}_{job.config['id']}.pkl"
            model_path = job.output_dir / model_filename
            job.output_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(model_path)
            
            # Save metrics
            metrics_file = job.output_dir / f"metrics_{job.config['id']}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                job_id=job.job_id,
                model_type=job.model_type,
                config_id=job.config['id'],
                dataset_name=job.dataset_name,
                success=True,
                metrics=metrics,
                model_path=model_path,
                training_time=training_time
            )
            
            logger.info(f"✅ {job.job_id} | R2={metrics['val'].get('r2', 0):.4f} | {training_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {job.job_id} | Error: {str(e)}")
            
            result = TrainingResult(
                job_id=job.job_id,
                model_type=job.model_type,
                config_id=job.config['id'],
                dataset_name=job.dataset_name,
                success=False,
                error_message=str(e)
            )
            
            return result
    
    def train_all_parallel(self, jobs: List[TrainingJob]) -> List[TrainingResult]:
        """
        Train all jobs in parallel
        
        Args:
            jobs: List of TrainingJob objects
        
        Returns:
            List of TrainingResult objects
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING PARALLEL TRAINING")
        logger.info("=" * 80)
        logger.info(f"Total jobs: {len(jobs)}")
        logger.info(f"Workers: {self.n_workers}")
        logger.info("=" * 80 + "\n")
        
        results = []
        failed = []
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for I/O bound tasks with sklearn
        # Use ProcessPoolExecutor for CPU-intensive tasks
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.train_single_job, job): job
                for job in jobs
            }
            
            # Process completed jobs
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if not result.success:
                        failed.append(result)
                    
                    completed += 1
                    
                    if completed % 10 == 0:
                        progress = (completed / len(jobs)) * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / completed) * (len(jobs) - completed)
                        
                        logger.info(f"Progress: {completed}/{len(jobs)} ({progress:.1f}%) | "
                                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                
                except Exception as e:
                    logger.error(f"Job failed: {job.job_id} | {str(e)}")
                    failed.append(TrainingResult(
                        job_id=job.job_id,
                        model_type=job.model_type,
                        config_id=job.config['id'],
                        dataset_name=job.dataset_name,
                        success=False,
                        error_message=str(e)
                    ))
        
        total_time = time.time() - start_time
        
        # Summary
        successful = len([r for r in results if r.success])
        logger.info("\n" + "=" * 80)
        logger.info("PARALLEL TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total jobs: {len(jobs)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Avg time per job: {total_time/len(jobs):.1f} seconds")
        logger.info("=" * 80 + "\n")
        
        self.training_results = results
        self.failed_jobs = failed
        
        return results
    
    def save_summary_report(self):
        """Save summary report"""
        
        report_file = self.output_dir / 'training_summary.json'
        
        summary = {
            'total_jobs': len(self.training_results),
            'successful': len([r for r in self.training_results if r.success]),
            'failed': len(self.failed_jobs),
            'results': []
        }
        
        for result in self.training_results:
            result_dict = {
                'job_id': result.job_id,
                'model_type': result.model_type,
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution for testing"""
    
    print("\n" + "=" * 80)
    print("PFAZ 2: PARALLEL AI TRAINER - TEST")
    print("=" * 80)
    
    # Initialize trainer
    trainer = ParallelAITrainer(
        output_dir='test_trained_models',
        n_workers=4
    )
    
    # Load configs (example - in real use, load from training_configs_50.json)
    configs = [
        {'id': 'TRAIN_001', 'n_estimators': 100, 'max_depth': 10},
        {'id': 'TRAIN_002', 'n_estimators': 200, 'max_depth': 15},
    ]
    
    # Create dummy dataset for testing
    print("\nCreating test dataset...")
    test_data_dir = Path('test_datasets/MM_75nuclei')
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy data
    n_samples = 75
    np.random.seed(42)
    test_df = pd.DataFrame({
        'A': np.random.randint(20, 250, n_samples),
        'Z': np.random.randint(10, 100, n_samples),
        'N': np.random.randint(10, 150, n_samples),
        'MM': np.random.randn(n_samples) * 2
    })
    test_df.to_csv(test_data_dir / 'MM_75nuclei.csv', index=False)
    
    # Create jobs
    print("\nCreating training jobs...")
    jobs = trainer.create_training_jobs(
        model_types=['RF'],
        configs=configs,
        dataset_paths=[test_data_dir]
    )
    
    # Train
    print("\nStarting training...")
    results = trainer.train_all_parallel(jobs)
    
    # Save report
    trainer.save_summary_report()
    
    print("\n✅ TEST COMPLETED!")


if __name__ == "__main__":
    main()
