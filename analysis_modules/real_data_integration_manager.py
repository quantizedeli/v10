"""
FAZ 8: Real Data Integration Manager
=====================================

Gerçek ANFIS veri setleriyle ensemble sistemi entegrasyonu:
1. ANFIS veri setlerini yükle
2. AI modellerini eğit
3. Ensemble modelleri oluştur
4. Kapsamlı değerlendirme
5. Production-ready results

Author: AI Dataset Training Pipeline
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import FAZ 7 modules
import sys
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/outputs')

from ensemble_model_builder import EnsembleModelBuilder
from stacking_meta_learner import StackingMetaLearner
from ensemble_evaluator import EnsembleEvaluator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class RealDataIntegrationManager:
    """
    Real Data Integration Manager
    
    Gerçek ANFIS veri setleriyle ensemble sistemini entegre eder
    """
    
    def __init__(self, 
                 anfis_data_dir: str,
                 output_dir: str = 'output/faz8_real_data_integration'):
        """
        Args:
            anfis_data_dir: ANFIS veri setlerinin bulunduğu klasör
            output_dir: Çıktıların kaydedileceği klasör
        """
        self.anfis_data_dir = Path(anfis_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {}  # Dataset ID -> Dataset info
        self.trained_models = {}  # Model ID -> Model info
        self.ensemble_results = {}  # Ensemble ID -> Results
        
        logger.info("="*80)
        logger.info("FAZ 8: REAL DATA INTEGRATION MANAGER")
        logger.info("="*80)
        logger.info(f"ANFIS data directory: {self.anfis_data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_anfis_datasets(self, 
                           target: str = 'MM',
                           n_datasets: int = 10) -> Dict:
        """
        ANFIS veri setlerini yükle
        
        Args:
            target: Hedef değişken (MM, QM, MM_QM, Beta_2)
            n_datasets: Yüklenecek dataset sayısı
        
        Returns:
            Loaded datasets info
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"LOADING ANFIS DATASETS - Target: {target}")
        logger.info(f"{'='*60}")
        
        # ANFIS datasets klasörünü tara - sadece _train.csv dosyalarını bul
        train_files = list(self.anfis_data_dir.glob(f"*{target}*_train.csv"))
        
        logger.info(f"Found {len(train_files)} datasets with target '{target}'")
        
        # İlk n dataset'i yükle
        loaded_datasets = []
        
        for i, train_path in enumerate(train_files[:n_datasets]):
            try:
                # Dataset ID'sini çıkar (_train.csv'yi kaldır)
                dataset_id = train_path.stem.replace('_train', '')
                
                # CSV dosyalarını yükle
                val_path = train_path.parent / f"{dataset_id}_check.csv"
                test_path = train_path.parent / f"{dataset_id}_test.csv"
                
                if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
                    logger.warning(f"  Skipping {dataset_id} - missing files")
                    continue
                
                # Load data
                df_train = pd.read_csv(train_path)
                df_val = pd.read_csv(val_path)
                df_test = pd.read_csv(test_path)
                
                # Extract features and target
                # Assume last column is target
                feature_cols = df_train.columns[:-1].tolist()
                target_col = df_train.columns[-1]
                
                X_train = df_train[feature_cols].values
                y_train = df_train[target_col].values
                
                X_val = df_val[feature_cols].values
                y_val = df_val[target_col].values
                
                X_test = df_test[feature_cols].values
                y_test = df_test[target_col].values
                
                dataset_info = {
                    'dataset_id': dataset_id,
                    'target': target,
                    'n_features': len(feature_cols),
                    'feature_names': feature_cols,
                    'n_train': len(X_train),
                    'n_val': len(X_val),
                    'n_test': len(X_test),
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_test': X_test,
                    'y_test': y_test
                }
                
                self.datasets[dataset_id] = dataset_info
                loaded_datasets.append(dataset_id)
                
                logger.info(f"  ✓ Loaded: {dataset_id}")
                logger.info(f"    Features: {len(feature_cols)}, Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            except Exception as e:
                logger.error(f"  ✗ Error loading {dataset_path.name}: {str(e)}")
                continue
        
        logger.info(f"\n✓ Loaded {len(loaded_datasets)} datasets successfully")
        
        return {
            'target': target,
            'n_datasets': len(loaded_datasets),
            'dataset_ids': loaded_datasets
        }
    
    def train_base_models_on_dataset(self, 
                                    dataset_id: str,
                                    model_types: List[str] = None) -> Dict:
        """
        Bir dataset üzerinde base modelleri eğit
        
        Args:
            dataset_id: Dataset ID
            model_types: Eğitilecek model tipleri
        
        Returns:
            Training results
        """
        if model_types is None:
            model_types = ['Ridge', 'Lasso', 'RF', 'GBM', 'MLP']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING BASE MODELS - Dataset: {dataset_id}")
        logger.info(f"{'='*60}")
        
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not loaded!")
        
        dataset_info = self.datasets[dataset_id]
        X_train = dataset_info['X_train']
        y_train = dataset_info['y_train']
        X_val = dataset_info['X_val']
        y_val = dataset_info['y_val']
        
        # Import sklearn models
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        trained_models = {}
        
        for model_type in model_types:
            logger.info(f"\n→ Training {model_type}...")
            
            try:
                # Create model
                if model_type == 'Ridge':
                    model = Ridge(alpha=1.0)
                elif model_type == 'Lasso':
                    model = Lasso(alpha=0.1)
                elif model_type == 'RF':
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                elif model_type == 'GBM':
                    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
                elif model_type == 'MLP':
                    model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                # Train
                model.fit(X_train, y_train)
                
                # Validate
                y_pred_val = model.predict(X_val)
                r2 = r2_score(y_val, y_pred_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                mae = mean_absolute_error(y_val, y_pred_val)
                
                model_id = f"{dataset_id}_{model_type}"
                
                model_info = {
                    'model_id': model_id,
                    'dataset_id': dataset_id,
                    'model_type': model_type,
                    'model': model,
                    'n_features': dataset_info['n_features'],
                    'feature_names': dataset_info['feature_names'],
                    'val_r2': r2,
                    'val_rmse': rmse,
                    'val_mae': mae
                }
                
                trained_models[model_id] = model_info
                self.trained_models[model_id] = model_info
                
                logger.info(f"  ✓ {model_type}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            except Exception as e:
                logger.error(f"  ✗ Error training {model_type}: {str(e)}")
                continue
        
        logger.info(f"\n✓ Trained {len(trained_models)} models on {dataset_id}")
        
        return trained_models
    
    def create_ensemble_for_dataset(self, 
                                   dataset_id: str,
                                   ensemble_methods: List[str] = None) -> Dict:
        """
        Bir dataset için ensemble modelleri oluştur
        
        Args:
            dataset_id: Dataset ID
            ensemble_methods: Ensemble yöntemleri
        
        Returns:
            Ensemble results
        """
        if ensemble_methods is None:
            ensemble_methods = ['simple_voting', 'weighted_voting_r2', 'stacking_ridge']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CREATING ENSEMBLES - Dataset: {dataset_id}")
        logger.info(f"{'='*60}")
        
        # Get dataset info
        dataset_info = self.datasets[dataset_id]
        X_test = dataset_info['X_test']
        y_test = dataset_info['y_test']
        X_val = dataset_info['X_val']
        y_val = dataset_info['y_val']
        X_train = dataset_info['X_train']
        y_train = dataset_info['y_train']
        
        # Get models for this dataset
        model_ids = [mid for mid in self.trained_models.keys() if mid.startswith(dataset_id)]
        
        if len(model_ids) < 2:
            logger.warning(f"Not enough models for ensemble (need >=2, got {len(model_ids)})")
            return {}
        
        logger.info(f"Creating ensembles from {len(model_ids)} base models")
        
        ensemble_results = {}
        
        # Initialize ensemble builder
        if 'simple_voting' in ensemble_methods or 'weighted_voting_r2' in ensemble_methods:
            builder = EnsembleModelBuilder(self.output_dir / 'voting_ensembles' / dataset_id)
            
            # Add models
            for model_id in model_ids:
                model_info = self.trained_models[model_id]
                builder.add_model(
                    model_id=model_id,
                    model=model_info['model'],
                    metadata={'model_type': model_info['model_type'], 
                             'features': model_info['feature_names']},
                    X_val=X_val,
                    y_val=y_val
                )
        
        # Simple Voting
        if 'simple_voting' in ensemble_methods:
            logger.info("\n→ Creating Simple Voting...")
            result = builder.create_simple_voting(
                model_ids=model_ids,
                X_test=X_test,
                y_test=y_test
            )
            ensemble_results['simple_voting'] = result
        
        # Weighted Voting
        if 'weighted_voting_r2' in ensemble_methods:
            logger.info("\n→ Creating Weighted Voting (R²)...")
            result = builder.create_weighted_voting(
                model_ids=model_ids,
                X_test=X_test,
                y_test=y_test,
                optimization_method='r2'
            )
            ensemble_results['weighted_voting_r2'] = result
        
        # Stacking
        if 'stacking_ridge' in ensemble_methods:
            logger.info("\n→ Creating Stacking (Ridge)...")
            stacker = StackingMetaLearner(
                meta_model_type='ridge',
                cv_folds=5,
                output_dir=self.output_dir / 'stacking_models' / dataset_id
            )
            
            # Add models
            for model_id in model_ids:
                model_info = self.trained_models[model_id]
                stacker.add_base_model(
                    model_id,
                    model_info['model'],
                    {'model_type': model_info['model_type'], 
                     'features': model_info['feature_names']}
                )
            
            # Train stacking
            stacker.generate_oof_predictions(X_train, y_train)
            stacker.train_meta_model(y_train)
            result = stacker.evaluate(X_test, y_test)
            ensemble_results['stacking_ridge'] = result
        
        self.ensemble_results[dataset_id] = ensemble_results
        
        logger.info(f"\n✓ Created {len(ensemble_results)} ensembles for {dataset_id}")
        
        return ensemble_results
    
    def evaluate_all_ensembles(self) -> pd.DataFrame:
        """
        Tüm ensemble'ları değerlendir ve karşılaştır
        
        Returns:
            Comparison DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING ALL ENSEMBLES")
        logger.info(f"{'='*60}")
        
        all_results = []
        
        for dataset_id, ensemble_results in self.ensemble_results.items():
            dataset_info = self.datasets[dataset_id]
            
            for ensemble_name, result in ensemble_results.items():
                all_results.append({
                    'Dataset': dataset_id,
                    'Target': dataset_info['target'],
                    'N_Features': dataset_info['n_features'],
                    'N_Test': dataset_info['n_test'],
                    'Ensemble': ensemble_name,
                    'Method': result.get('method', ensemble_name),
                    'R²': result.get('r2', np.nan),
                    'RMSE': result.get('rmse', np.nan),
                    'MAE': result.get('mae', np.nan)
                })
        
        df = pd.DataFrame(all_results)
        df = df.sort_values(['Dataset', 'R²'], ascending=[True, False])
        
        logger.info(f"\n{df.to_string(index=False)}")
        
        # Save
        save_path = self.output_dir / 'all_ensembles_comparison.xlsx'
        df.to_excel(save_path, index=False)
        logger.info(f"\n✓ Comparison saved: {save_path}")
        
        return df
    
    def generate_final_report(self) -> Dict:
        """
        Final rapor oluştur
        
        Returns:
            Report dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATING FINAL REPORT")
        logger.info(f"{'='*60}")
        
        # Comparison
        comparison_df = self.evaluate_all_ensembles()
        
        # Best ensemble per dataset
        best_per_dataset = comparison_df.groupby('Dataset').apply(
            lambda x: x.nlargest(1, 'R²')
        ).reset_index(drop=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST ENSEMBLE PER DATASET:")
        logger.info(f"{'='*60}")
        logger.info(f"\n{best_per_dataset.to_string(index=False)}")
        
        # Overall best
        overall_best = comparison_df.nlargest(1, 'R²').iloc[0]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OVERALL BEST ENSEMBLE:")
        logger.info(f"{'='*60}")
        logger.info(f"Dataset: {overall_best['Dataset']}")
        logger.info(f"Ensemble: {overall_best['Ensemble']}")
        logger.info(f"R² = {overall_best['R²']:.4f}")
        logger.info(f"RMSE = {overall_best['RMSE']:.4f}")
        logger.info(f"MAE = {overall_best['MAE']:.4f}")
        
        # Report
        report = {
            'n_datasets': len(self.datasets),
            'n_models_trained': len(self.trained_models),
            'n_ensembles_created': sum(len(e) for e in self.ensemble_results.values()),
            'overall_best': {
                'dataset': overall_best['Dataset'],
                'ensemble': overall_best['Ensemble'],
                'r2': float(overall_best['R²']),
                'rmse': float(overall_best['RMSE']),
                'mae': float(overall_best['MAE'])
            },
            'best_per_dataset': best_per_dataset.to_dict('records')
        }
        
        # Save JSON
        report_path = self.output_dir / 'faz8_final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved: {report_path}")
        
        return report


def main():
    """FAZ 8 Test"""
    logger.info("="*80)
    logger.info("FAZ 8: REAL DATA INTEGRATION - TEST")
    logger.info("="*80)
    
    # Create mock ANFIS datasets for testing
    logger.info("\n📊 Creating mock ANFIS datasets...")
    
    # Create test directory
    test_anfis_dir = Path('/home/claude/test_anfis_data')
    test_anfis_dir.mkdir(exist_ok=True)
    
    # Create 3 mock datasets
    np.random.seed(42)
    for i in range(3):
        dataset_id = f"MM_75_S70_anomalisiz_AZN_none_random_{i}"
        
        # Generate mock data
        n_train, n_val, n_test = 150, 50, 50
        n_features = 3
        
        X_train = np.random.randn(n_train, n_features)
        X_val = np.random.randn(n_val, n_features)
        X_test = np.random.randn(n_test, n_features)
        
        def target_fn(X):
            return 2*X[:, 0] + X[:, 1] - 0.5*X[:, 2] + np.random.randn(len(X))*0.3
        
        y_train = target_fn(X_train)
        y_val = target_fn(X_val)
        y_test = target_fn(X_test)
        
        # Save CSVs
        df_train = pd.DataFrame(X_train, columns=[f'F{j}' for j in range(n_features)])
        df_train['MM'] = y_train
        df_train.to_csv(test_anfis_dir / f"{dataset_id}_train.csv", index=False)
        
        df_val = pd.DataFrame(X_val, columns=[f'F{j}' for j in range(n_features)])
        df_val['MM'] = y_val
        df_val.to_csv(test_anfis_dir / f"{dataset_id}_check.csv", index=False)
        
        df_test = pd.DataFrame(X_test, columns=[f'F{j}' for j in range(n_features)])
        df_test['MM'] = y_test
        df_test.to_csv(test_anfis_dir / f"{dataset_id}_test.csv", index=False)
    
    logger.info(f"✓ Created 3 mock datasets in {test_anfis_dir}")
    
    # Initialize manager
    manager = RealDataIntegrationManager(
        anfis_data_dir=test_anfis_dir,
        output_dir='/home/claude/output/faz8_test'
    )
    
    # Load datasets
    load_info = manager.load_anfis_datasets(target='MM', n_datasets=3)
    
    # Train models on each dataset
    for dataset_id in load_info['dataset_ids']:
        manager.train_base_models_on_dataset(
            dataset_id,
            model_types=['Ridge', 'Lasso', 'RF']
        )
    
    # Create ensembles
    for dataset_id in load_info['dataset_ids']:
        manager.create_ensemble_for_dataset(
            dataset_id,
            ensemble_methods=['simple_voting', 'weighted_voting_r2', 'stacking_ridge']
        )
    
    # Final evaluation
    report = manager.generate_final_report()
    
    logger.info("\n" + "="*80)
    logger.info("✅ FAZ 8 TEST COMPLETED!")
    logger.info("="*80)
    
    return report


if __name__ == "__main__":
    report = main()
