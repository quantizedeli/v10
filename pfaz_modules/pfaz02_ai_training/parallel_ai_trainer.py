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
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')
warnings.filterwarnings('ignore', message='.*joblib.*')

# Sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

# LightGBM (optional)
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available — install: pip install lightgbm")

# CatBoost (optional)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available — install: pip install catboost")

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

# Minimum val_R2 to save a model .pkl (Poor: <0.5, Failed: <0 — not saved)
R2_MIN_SAVE_THRESHOLD = 0.5

# Import SeedTracker (after logging is configured)
try:
    from .seed_tracker import SeedTracker
    SEED_TRACKER_AVAILABLE = True
except ImportError:
    SEED_TRACKER_AVAILABLE = False
    logger.warning("SeedTracker not available")


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
        
    def _get_feature_set_from_name(self, dataset_name: str) -> List[str]:
        """Feature set kolonlarini dataset adindan belirle.

        Yeni format: {TARGET}_{SIZE}_{SCENARIO}_{FEATURE_CODE}_{SCALING}_{SAMPLING}[_NoAnomaly]
        Ornek: MM_75_S70_AZS_NoScaling_Random

        Dataset dosyasi zaten sadece ilgili feature kolonlarini iceriyor.
        None dondurerek adaptive selection'in dogru kolonlari sec. birakilir.

        NOT: substring matching YAPILMAZ. 'AZN' kodu 'AZNNP' icinde de bulunur
        ve yanlis eslemeye yol acar.
        """
        # Her zaman None dondur: dataset CSV'i zaten dogru feature'lari iceriyor.
        # load_dataset'teki adaptive selection (target + NUCLEUS disindaki tum kolonlar)
        # yeni format icin en guvenilir yontemdir.
        logger.debug(f"Adaptive feature selection for: {dataset_name}")
        return None

    def load_dataset(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load dataset from path with pre-split train/val/test files

        NEW STRUCTURE (from PFAZ 1):
        dataset_path/
            ├── train.csv / train.xlsx
            ├── val.csv / val.xlsx
            └── test.csv / test.xlsx

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """

        # Check for train/val/test split files
        # CSV oncelikli — headerless pure-number format; xlsx fallback (eski datasetler)
        split_files = {}
        for split_name in ['train', 'val', 'test']:
            csv_file  = dataset_path / f"{split_name}.csv"
            xlsx_file = dataset_path / f"{split_name}.xlsx"

            if csv_file.exists():
                split_files[split_name] = csv_file
            elif xlsx_file.exists():
                split_files[split_name] = xlsx_file
            else:
                raise FileNotFoundError(f"Missing {split_name} file in {dataset_path}")

        logger.info(f"Loading pre-split dataset from: {dataset_path}")
        logger.info(f"  Train: {split_files['train'].name}")
        logger.info(f"  Val:   {split_files['val'].name}")
        logger.info(f"  Test:  {split_files['test'].name}")

        # metadata.json'dan sutun adlarini al (yeni format: headerless CSV)
        _col_names = None
        _meta_file = dataset_path / 'metadata.json'
        if _meta_file.exists():
            try:
                with open(_meta_file, encoding='utf-8') as _f:
                    _meta = json.load(_f)
                _feat = _meta.get('feature_names') or _meta.get('feature_columns', [])
                _tgt  = _meta.get('target_names')  or _meta.get('target_columns',  [])
                if _feat and _tgt:
                    _col_names = list(_feat) + list(_tgt)
                    logger.info(f"  metadata.json: {len(_feat)} feature + {len(_tgt)} target = {len(_col_names)} sutun")
            except Exception as _e:
                logger.warning(f"  metadata.json okunamadi: {_e}")

        # Load each split
        splits = {}
        for split_name, file_path in split_files.items():
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_path.suffix == '.csv':
                if _col_names:
                    # Yeni format: headerless CSV — metadata'dan sutun adlari atanir
                    df = pd.read_csv(file_path, header=None, names=_col_names,
                                     encoding='utf-8')
                else:
                    # Eski format: baslik satiri var (geriye donuk uyumluluk)
                    df = pd.read_csv(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            splits[split_name] = df
            logger.info(f"  Loaded {split_name}: {len(df)} samples")

        # Process train split to determine columns (same logic for all splits)
        df = splits['train']

        # DATA CLEANING - Fix invalid values
        logger.info(f"Initial data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # 1. Replace 'unknown' strings with NaN
        df = df.replace('unknown', np.nan)
        df = df.replace('Unknown', np.nan)
        df = df.replace('UNKNOWN', np.nan)

        # 2. Fix Unicode minus signs (U+2212 '−' to regular '-')
        for col in df.columns:
            if df[col].dtype == 'object':
                # Also handle 'nan' string, empty strings
                df[col] = df[col].astype(str).str.replace('−', '-', regex=False)
                df[col] = df[col].replace('nan', np.nan)
                df[col] = df[col].replace('', np.nan)
                df[col] = df[col].replace('NaN', np.nan)

        # 3. Identify feature and target columns based on dataset name
        # Map simplified names to possible actual column names
        # IMPORTANT: These are REAL experimental values from aaa2.txt, not theoretical predictions
        # - Beta_2: Real deformation parameter (from experimental data)
        # - Beta_2_estimated: Theoretical estimate (excluded as feature to prevent leakage)
        # - MM/MAGNETIC MOMENT: Real magnetic moment (from experimental data)
        # - schmidt_moment: Theoretical estimate (excluded as feature to prevent leakage)
        # - Q/QUADRUPOLE MOMENT: Real quadrupole moment (from experimental data)
        # - Q0_intrinsic: Theoretical estimate (excluded as feature to prevent leakage)
        target_map = {
            'MM': ['MM', 'MAGNETIC MOMENT [µ]', 'MAGNETIC MOMENT [μ]'],
            'Q': ['Q', 'QM', 'QUADRUPOLE MOMENT [Q]'],
            'Beta_2': ['Beta_2', 'BETA_2']  # Real experimental Beta_2, NOT Beta_2_estimated
        }

        # Determine which targets to use based on dataset name
        dataset_name = dataset_path.name
        requested_targets = []

        if 'MM_QM' in dataset_name or 'MM-QM' in dataset_name:
            # Both MM and Q targets
            requested_targets = ['MM', 'Q']
        elif 'MM' in dataset_name:
            # Only MM target
            requested_targets = ['MM']
        elif 'QM' in dataset_name or '_Q_' in dataset_name:
            # Only Q target
            requested_targets = ['Q']
        elif 'Beta_2' in dataset_name or 'BETA_2' in dataset_name:
            # Only Beta_2 target
            requested_targets = ['Beta_2']
        else:
            # Default: try to find any available target
            logger.warning(f"Could not determine target from dataset name '{dataset_name}', using all available targets")
            requested_targets = list(target_map.keys())

        logger.info(f"Dataset: {dataset_name} -> Requested targets: {requested_targets}")

        # Find actual column names for requested targets
        target_cols = []
        for simple_name in requested_targets:
            if simple_name in target_map:
                for col_name in target_map[simple_name]:
                    if col_name in df.columns:
                        target_cols.append(col_name)
                        logger.info(f"Found target: {simple_name} -> {col_name}")
                        break
                else:
                    logger.warning(f"Target '{simple_name}' requested but not found in data columns")

        if not target_cols:
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"No target columns found for {requested_targets} in {data_file}")

        logger.info(f"Target columns: {target_cols}")

        # 4. Dataset adından özellik setini belirle
        predefined_features = self._get_feature_set_from_name(dataset_path.name)

        if predefined_features is not None:
            # Önceden tanımlanmış özellik seti kullan
            feature_cols = [col for col in predefined_features if col in df.columns]
            logger.info(f"Using predefined feature set ({len(feature_cols)} features): {feature_cols}")
        else:
            # ALL dataseti için: NaN oranı %50'den az olan özellikleri kullan
            all_possible_features = [col for col in df.columns if col not in target_cols and col != 'NUCLEUS']

            # Exclude CATEGORICAL features (string values, cannot be used in numeric models)
            # These are descriptive labels derived from other features, not useful for ML
            categorical_features = ['deformation_type', 'nucleus_collective_type']
            all_possible_features = [col for col in all_possible_features if col not in categorical_features]
            if any(col in df.columns for col in categorical_features):
                found_categorical = [col for col in categorical_features if col in df.columns]
                logger.info(f"Excluded {len(found_categorical)} categorical features (string values): {found_categorical}")

            # CRITICAL: Prevent data leakage - exclude theoretical predictions of targets
            # These features are direct theoretical calculations/estimates of the targets
            # Using them would give artificially high R² scores
            leakage_features = []

            if 'Beta_2' in requested_targets:
                # Beta_2_estimated: direct theoretical estimate of Beta_2
                # Q0_intrinsic: intrinsic quadrupole moment, calculated from Beta_2_estimated
                # rotational_param: depends on deformation
                # moment_of_inertia: calculated from deformation
                # E_2plus: first excited 2+ state energy, related to deformation
                # vib_frequency: vibrational frequency, related to deformation
                leakage_features.extend([
                    'Beta_2_estimated',  # Direct estimate
                    'Q0_intrinsic',      # Derived from Beta_2
                    'rotational_param',  # Deformation-dependent
                    'moment_of_inertia', # Deformation-dependent
                    'E_2plus',           # Deformation-dependent
                    'vib_frequency'      # Deformation-dependent
                ])

            if 'MM' in requested_targets:
                # schmidt_moment: theoretical magnetic moment prediction
                leakage_features.extend(['schmidt_moment'])

            if 'Q' in requested_targets:
                # Q0_intrinsic: intrinsic quadrupole moment (theoretical)
                # Note: If Beta_2 is also a target, Q0_intrinsic is already excluded
                if 'Q0_intrinsic' not in leakage_features:
                    leakage_features.append('Q0_intrinsic')

            # Remove leakage features
            if leakage_features:
                all_possible_features = [col for col in all_possible_features if col not in leakage_features]
                logger.info(f"Excluded {len(leakage_features)} features to prevent data leakage: {leakage_features}")

            # Convert to numeric first
            for col in all_possible_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # NaN oranlarını hesapla
            nan_threshold = 0.5  # %50
            feature_cols = []
            excluded_high_nan = []
            schmidt_related = []  # Schmidt features excluded (expected for odd-odd nuclei)

            for col in all_possible_features:
                nan_ratio = df[col].isna().sum() / len(df)
                if nan_ratio < nan_threshold:
                    feature_cols.append(col)
                else:
                    # Check if this is a Schmidt-related feature (expected to be NaN for odd-odd nuclei)
                    if 'schmidt' in col.lower():
                        schmidt_related.append(col)
                    else:
                        excluded_high_nan.append((col, nan_ratio))

            # Report excluded features
            if schmidt_related:
                logger.info(f"Excluded {len(schmidt_related)} Schmidt features (not applicable for odd-odd nuclei): {', '.join(schmidt_related)}")
            if excluded_high_nan:
                for col, nan_ratio in excluded_high_nan:
                    logger.warning(f"Excluding {col}: {nan_ratio*100:.1f}% NaN (threshold: {nan_threshold*100}%)")

            logger.info(f"Using adaptive feature selection ({len(feature_cols)}/{len(all_possible_features)} features)")
            logger.info(f"Selected features: {feature_cols[:10]}...")

        if len(feature_cols) == 0:
            raise ValueError(f"No valid features after filtering in {dataset_path}")

        # 5-7. Process each split with same pipeline
        # Train imputer on train set, apply to val and test
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')

        processed_splits = {}

        for split_name in ['train', 'val', 'test']:
            df_split = splits[split_name].copy()

            # Convert to numeric
            all_numeric_cols = feature_cols + target_cols
            for col in all_numeric_cols:
                if col in df_split.columns:
                    df_split[col] = pd.to_numeric(df_split[col], errors='coerce')

            # Drop rows with NaN targets
            df_clean = df_split[all_numeric_cols].dropna(subset=target_cols)

            # Impute features
            if split_name == 'train':
                # Fit imputer on train
                df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
            else:
                # Transform val/test with train imputer
                df_clean[feature_cols] = imputer.transform(df_clean[feature_cols])

            # Extract X, y
            X = df_clean[feature_cols].values.astype(np.float32)
            y = df_clean[target_cols].values.astype(np.float32)

            processed_splits[split_name] = (X, y)
            logger.info(f"  Processed {split_name}: {len(X)} samples")

        X_train, y_train = processed_splits['train']
        X_val, y_val = processed_splits['val']
        X_test, y_test = processed_splits['test']

        # DNN minimum dataset boyutu filtresi
        # DNN, küçük veri setlerinde (< DNN_MIN_SAMPLES) genellikle ıraksar veya overfitting yapar.
        # Bu eşiğin altındaki eğitimler atlanır; zaman ve disk alanı tasarrufu sağlanır.
        DNN_MIN_SAMPLES = 80  # eğitim seti için minimum satır sayısı
        if self.model_type == 'DNN' and len(X_train) < DNN_MIN_SAMPLES:
            raise ValueError(
                f"[DNN-SKIP] Eğitim seti çok küçük ({len(X_train)} < {DNN_MIN_SAMPLES}). "
                f"DNN bu dataset için atlanıyor."
            )

        logger.info(f"Final: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

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
        
        # Random seed
        random_seed = 42

        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=random_seed
        )

        logger.info(f"  Random seed: {random_seed}")
        
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


def _detect_xgb_gpu() -> dict:
    """XGBoost surumune gore GPU parametrelerini dondur."""
    try:
        import xgboost as xgb
        ver = tuple(int(x) for x in xgb.__version__.split('.')[:2])
        import subprocess, sys
        # Hizli CUDA kontrolu: kucuk deneme modeli
        test = xgb.XGBRegressor(n_estimators=2, device='cuda', verbosity=0)
        import numpy as _np
        test.fit(_np.array([[1], [2]]), _np.array([1, 2]))
        if ver >= (2, 0):
            return {'device': 'cuda'}
        else:
            return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
    except Exception:
        return {}  # CPU fallback


_XGB_GPU_PARAMS: dict = None  # lazy-init, modül seviyesinde cache


class XGBoostTrainer(BaseAITrainer):
    """XGBoost Trainer"""

    def __init__(self, config: Dict, output_dir: Path, gpu_enabled: bool = False):
        super().__init__('XGBoost', config, output_dir)
        self.gpu_enabled = gpu_enabled
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")

    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train XGBoost"""
        global _XGB_GPU_PARAMS

        # Get config parameters
        n_estimators = self.config.get('n_estimators', 100)
        learning_rate = self.config.get('learning_rate', 0.1)
        max_depth = self.config.get('max_depth', 6)

        # GPU params (lazy detect once)
        gpu_params = {}
        if self.gpu_enabled:
            if _XGB_GPU_PARAMS is None:
                _XGB_GPU_PARAMS = _detect_xgb_gpu()
            gpu_params = _XGB_GPU_PARAMS
            if gpu_params:
                logger.info(f"[XGB] GPU aktif: {gpu_params}")
            else:
                logger.info("[XGB] GPU bulunamadi, CPU kullaniliyor")

        logger.info(f"Training XGBoost: n_estimators={n_estimators}, lr={learning_rate}")

        # Random seed
        random_seed = 42

        xgb_kwargs = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_seed,
            **gpu_params
        )
        if not gpu_params:
            xgb_kwargs['n_jobs'] = -1

        # Create and train model
        self.model = XGBRegressor(**xgb_kwargs)

        logger.info(f"  Random seed: {random_seed}")
        
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


class LightGBMTrainer(BaseAITrainer):
    """LightGBM Trainer — fast gradient boosting, handles missing values natively"""

    def __init__(self, config: Dict, output_dir: Path, gpu_enabled: bool = False):
        super().__init__('LightGBM', config, output_dir)
        self.gpu_enabled = gpu_enabled
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available — install: pip install lightgbm")

    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        n_estimators = self.config.get('n_estimators', 200)
        learning_rate = self.config.get('learning_rate', 0.05)
        max_depth = self.config.get('max_depth', -1)  # -1 = no limit
        num_leaves = self.config.get('num_leaves', 31)
        random_seed = 42

        # GPU desteği: LightGBM GPU build gerektirir; yoksa sessizce CPU'ya duser
        lgbm_device = 'cpu'
        if self.gpu_enabled:
            try:
                _test = LGBMRegressor(n_estimators=2, device='gpu', verbosity=-1)
                import numpy as _np
                _test.fit(_np.array([[1], [2]]), _np.array([1.0, 2.0]))
                lgbm_device = 'gpu'
                logger.info("[LGB] GPU aktif")
            except Exception:
                logger.info("[LGB] GPU build bulunamadi, CPU kullaniliyor")

        logger.info(f"Training LightGBM: n_estimators={n_estimators}, lr={learning_rate}, leaves={num_leaves}, device={lgbm_device}")

        # Multi-output: wrap with MultiOutputRegressor
        is_multi = y_train.ndim > 1 and y_train.shape[1] > 1
        base_model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_seed,
            device=lgbm_device,
            n_jobs=(-1 if lgbm_device == 'cpu' else 1),
            verbose=-1
        )
        self.model = MultiOutputRegressor(base_model, n_jobs=1) if is_multi else base_model

        start_time = time.time()
        if is_multi:
            self.model.fit(X_train, y_train)
        else:
            y_1d = y_train.flatten() if y_train.ndim > 1 else y_train
            self.model.fit(X_train, y_1d)
        training_time = time.time() - start_time

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        return {
            'train': self.calculate_metrics(y_train, y_train_pred),
            'val': self.calculate_metrics(y_val, y_val_pred),
            'training_time': training_time
        }


class CatBoostTrainer(BaseAITrainer):
    """CatBoost Trainer — handles categorical features and small datasets well"""

    def __init__(self, config: Dict, output_dir: Path):
        super().__init__('CatBoost', config, output_dir)
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available — install: pip install catboost")

    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        n_estimators = self.config.get('n_estimators', 200)
        learning_rate = self.config.get('learning_rate', 0.05)
        depth = self.config.get('max_depth', 6)
        random_seed = 42

        logger.info(f"Training CatBoost: iterations={n_estimators}, lr={learning_rate}, depth={depth}")

        is_multi = y_train.ndim > 1 and y_train.shape[1] > 1
        base_model = CatBoostRegressor(
            iterations=n_estimators,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=0,
            loss_function='RMSE'
        )
        self.model = MultiOutputRegressor(base_model, n_jobs=1) if is_multi else base_model

        start_time = time.time()
        y_train_1d = y_train.flatten() if (not is_multi and y_train.ndim > 1) else y_train
        self.model.fit(X_train, y_train_1d, eval_set=(X_val, y_val.flatten() if (not is_multi and y_val.ndim > 1) else y_val) if not is_multi else None)
        training_time = time.time() - start_time

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        return {
            'train': self.calculate_metrics(y_train, y_train_pred),
            'val': self.calculate_metrics(y_val, y_val_pred),
            'training_time': training_time
        }


class SVRTrainer(BaseAITrainer):
    """SVR Trainer — Support Vector Regression, good for small nuclear datasets"""

    def __init__(self, config: Dict, output_dir: Path):
        super().__init__('SVR', config, output_dir)

    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        from sklearn.preprocessing import StandardScaler
        kernel = self.config.get('kernel', 'rbf')
        C = self.config.get('C', 10.0)
        epsilon = self.config.get('epsilon', 0.1)
        gamma = self.config.get('gamma', 'scale')

        logger.info(f"Training SVR: kernel={kernel}, C={C}, epsilon={epsilon}")

        # SVR requires scaled features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        is_multi = y_train.ndim > 1 and y_train.shape[1] > 1
        base_model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.model = MultiOutputRegressor(base_model, n_jobs=-1) if is_multi else base_model

        start_time = time.time()
        y_train_1d = y_train.flatten() if (not is_multi and y_train.ndim > 1) else y_train
        self.model.fit(X_train_scaled, y_train_1d)
        training_time = time.time() - start_time

        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)

        return {
            'train': self.calculate_metrics(y_train, y_train_pred),
            'val': self.calculate_metrics(y_val, y_val_pred),
            'training_time': training_time
        }

    def predict(self, X):
        if not hasattr(self, 'scaler'):
            raise ValueError("Scaler not fitted")
        return self.model.predict(self.scaler.transform(X))


class DNNTrainer(BaseAITrainer):
    """Deep Neural Network Trainer"""

    def __init__(self, config: Dict, output_dir: Path, gpu_enabled: bool = False):
        super().__init__('DNN', config, output_dir)

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        self.gpu_enabled = gpu_enabled
        self._configure_gpu()
    
    def _configure_gpu(self):
        """Configure GPU settings for TensorFlow"""
        if not self.gpu_enabled:
            # Disable GPU, use CPU only
            tf.config.set_visible_devices([], 'GPU')
            logger.info("[DNN] GPU DISABLED - Using CPU only")
        else:
            # Enable GPU with memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"[DNN] GPU ENABLED - Found {len(gpus)} GPU(s)")
                except RuntimeError as e:
                    logger.warning(f"[DNN] GPU config error: {e}")
            else:
                logger.warning("[DNN] GPU enabled but no GPU found - Using CPU")

    def build_model(self, input_dim: int, output_dim: int) -> keras.Model:
        """Build DNN model with BatchNorm and L2 regularization for numerical stability"""
        from tensorflow.keras import regularizers

        architecture = self.config.get('architecture', [128, 64, 32])
        dropout = self.config.get('dropout', [0.2, 0.2, 0.1])
        activation = self.config.get('activation', 'relu')
        learning_rate = self.config.get('learning_rate', 0.001)
        l2_reg = self.config.get('l2_reg', 1e-4)

        # Pad dropout list if shorter than architecture
        while len(dropout) < len(architecture):
            dropout.append(0.0)

        # Build model
        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        for units, drop in zip(architecture, dropout):
            x = layers.Dense(units, activation=None,
                             kernel_regularizer=regularizers.l2(l2_reg))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            if drop > 0:
                x = layers.Dropout(drop)(x)

        outputs = layers.Dense(output_dim)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # clipnorm=1.0 prevents gradient explosion; clipvalue as extra safety
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
            clipvalue=0.5
        )
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

        return model

    def train(self, X_train, y_train, X_val, y_val) -> Dict:
        """Train DNN with feature AND target scaling to prevent R2 divergence"""

        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 200)
        early_stopping_patience = self.config.get('early_stopping_patience', 20)
        random_seed = 42

        logger.info(f"Training DNN: batch_size={batch_size}, epochs={epochs}, seed={random_seed}")

        # Set random seed for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        # CRITICAL: Scale BOTH features AND targets for neural networks
        # Without target scaling, large-magnitude targets (MM, QM) cause extreme R2 values
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.y_scaler = StandardScaler()
        y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_val_2d = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        y_train_scaled = self.y_scaler.fit_transform(y_train_2d)
        y_val_scaled = self.y_scaler.transform(y_val_2d)

        logger.info(f"Features and targets scaled using StandardScaler (mean=0, std=1)")
        logger.info(f"Target stats — mean: {self.y_scaler.mean_}, std: {self.y_scaler.scale_}")

        # Build model
        input_dim = X_train.shape[1]
        output_dim = y_train_scaled.shape[1]

        self.model = self.build_model(input_dim, output_dim)

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            min_delta=1e-4
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )

        # Train
        start_time = time.time()
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        training_time = time.time() - start_time

        self.history = history.history

        # Evaluate — inverse transform to original scale for meaningful R2
        y_train_pred_scaled = self.model.predict(X_train_scaled, verbose=0)
        y_val_pred_scaled = self.model.predict(X_val_scaled, verbose=0)

        y_train_pred = self.y_scaler.inverse_transform(y_train_pred_scaled)
        y_val_pred = self.y_scaler.inverse_transform(y_val_pred_scaled)

        train_metrics = self.calculate_metrics(y_train_2d, y_train_pred)
        val_metrics = self.calculate_metrics(y_val_2d, y_val_pred)

        # DIVERGENCE DETECTION: Flag if DNN diverged
        val_r2 = val_metrics.get('r2', val_metrics.get('r2_avg', float('nan')))
        diverged = False
        if not np.isnan(val_r2) and val_r2 < -2.0:
            diverged = True
            logger.warning(
                f"[DNN DIVERGED] val_R2={val_r2:.4f} < -2.0 threshold. "
                f"Model is saved but marked as diverged. "
                f"Check: target variance, dataset size, or reduce learning_rate."
            )

        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'diverged': diverged
        }

        return metrics

    def predict(self, X):
        """Make predictions with feature AND target inverse scaling"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        if not hasattr(self, 'scaler'):
            raise ValueError("Feature scaler not fitted yet")
        if not hasattr(self, 'y_scaler'):
            raise ValueError("Target scaler not fitted yet")

        X_scaled = self.scaler.transform(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        return self.y_scaler.inverse_transform(y_pred_scaled)


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
                 datasets_dir: str = None,
                 models_dir: str = None,
                 training_config_path: str = None,
                 output_dir: str = None,
                 n_workers: int = None,
                 gpu_enabled: bool = False,
                 use_hyperparameter_tuning: bool = False,
                 use_model_validation: bool = True,
                 use_advanced_models: bool = False,
                 use_parallel_training: bool = None):
        """
        Initialize Parallel AI Trainer

        Args:
            datasets_dir: Directory containing datasets from PFAZ 1
            models_dir: Output directory for trained models (PFAZ 2 output)
            training_config_path: Path to training configurations JSON file
            output_dir: Alternative to models_dir (for backward compatibility)
            n_workers: Number of parallel workers (None = auto)
            gpu_enabled: Enable GPU training (for DNN)
            use_hyperparameter_tuning: Enable hyperparameter tuning with Optuna
            use_model_validation: Enable cross-validation and robustness testing
            use_advanced_models: Enable advanced models (BNN, PINN)
            use_parallel_training: Use parallel training (None = prompt user, True = parallel, False = sequential)
        """
        # Handle both parameter styles
        if models_dir is not None:
            self.output_dir = Path(models_dir)
        elif output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path('trained_models')

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set datasets directory
        self.datasets_dir = Path(datasets_dir) if datasets_dir else None

        # Set training config path
        self.training_config_path = Path(training_config_path) if training_config_path else None

        # Determine number of workers
        if n_workers is None:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # GPU varsa model eğitimi GPU'da; CPU işçisi sayısını cpu-1 yap (min 2)
            # GPU yoksa cpu-1 yap (sistem responsiveness için 1 core serbest bırak)
            self.n_workers = max(2, cpu_count - 1)
        else:
            self.n_workers = n_workers

        self.gpu_enabled = gpu_enabled
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.use_model_validation = use_model_validation
        self.use_advanced_models = use_advanced_models

        # PARALLEL TRAINING MODE
        # If None, will prompt user in train_all_models_parallel()
        self.use_parallel_training = use_parallel_training

        # Storage
        self.training_results = []
        self.failed_jobs = []

        # SEED TRACKER: Track all random seeds for reproducibility
        if SEED_TRACKER_AVAILABLE:
            self.seed_tracker = SeedTracker(output_dir=str(self.output_dir / 'seed_reports'))
            logger.info("[ACTIVATED] Seed Tracker initialized")
        else:
            self.seed_tracker = None
            logger.warning("[SKIP] Seed Tracker not available")

        # ACTIVATED MODULES: Import optional modules
        self.overfitting_detector = None
        self.hyperparameter_tuner = None
        self.advanced_models_trainer = None

        if self.use_hyperparameter_tuning:
            try:
                from pfaz_modules.pfaz02_ai_training.hyperparameter_tuner import HyperparameterTuner
                logger.info("[ACTIVATED] Hyperparameter Tuner loaded")
            except ImportError as e:
                logger.warning(f"[SKIP] Hyperparameter Tuner not available: {e}")
                self.use_hyperparameter_tuning = False

        if self.use_advanced_models:
            try:
                from pfaz_modules.pfaz02_ai_training.advanced_models import BayesianNeuralNetwork, PINN
                logger.info("[ACTIVATED] Advanced Models (BNN, PINN) loaded")
            except ImportError as e:
                logger.warning(f"[SKIP] Advanced Models not available: {e}")
                self.use_advanced_models = False

        logger.info("=" * 80)
        logger.info("PARALLEL AI TRAINER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Datasets directory: {self.datasets_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Training config: {self.training_config_path}")
        logger.info(f"Workers: {self.n_workers}")
        logger.info(f"GPU enabled: {self.gpu_enabled}")
        logger.info(f"Hyperparameter tuning: {self.use_hyperparameter_tuning}")
        logger.info(f"Model validation: {self.use_model_validation}")
        logger.info(f"Advanced models: {self.use_advanced_models}")
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
        skipped_dnn_jobs = 0
        skipped_datasets = 0

        for dataset_path in dataset_paths:
            # CRITICAL: Check if dataset actually exists with required files
            if not dataset_path.exists():
                logger.warning(f"Skipping {dataset_path.name}: directory not found")
                skipped_datasets += 1
                continue

            # Check for required split files (train/val/test)
            has_train = (dataset_path / "train.xlsx").exists() or (dataset_path / "train.csv").exists()
            has_val = (dataset_path / "val.xlsx").exists() or (dataset_path / "val.csv").exists()
            has_test = (dataset_path / "test.xlsx").exists() or (dataset_path / "test.csv").exists()

            if not (has_train and has_val and has_test):
                logger.warning(f"Skipping {dataset_path.name}: missing train/val/test files (train={has_train}, val={has_val}, test={has_test})")
                skipped_datasets += 1
                continue

            # Extract number of nuclei from dataset name
            dataset_name = dataset_path.name
            nuclei_count = None

            import re
            # New format: MM_75_S70_AZS_... or Beta_2_100_S70_...
            # Second numeric token after the target prefix is the size
            match_new = re.search(r'(?:MM_QM|Beta_2|MM|QM)_(\d+)_', dataset_name)
            if match_new:
                nuclei_count = int(match_new.group(1))
            else:
                # Legacy format: "MM_100nuclei", "Beta_2_75nuclei"
                match_old = re.search(r'(\d+)nuclei', dataset_name)
                if match_old:
                    nuclei_count = int(match_old.group(1))

            if 'ALL' in dataset_name:
                nuclei_count = 999  # ALL size -> enough for DNN

            for model_type in model_types:
                # SKIP DNN for small/medium datasets (< 200 nuclei)
                # Nuclear datasets are small (75-200 samples) - DNN overfits, RF/XGBoost preferred
                if model_type == 'DNN' and (nuclei_count is None or nuclei_count < 200):
                    logger.info(f"Skipping DNN for {dataset_name}: only {nuclei_count} nuclei (DNN requires 100+)")
                    skipped_dnn_jobs += len(configs)
                    continue

                for config in configs:
                    job_id = f"{dataset_path.name}_{model_type}_{config['id']}"

                    output_dir = self.output_dir / dataset_path.name / model_type / config['id']

                    # Resume: skip already-trained jobs (model file exists)
                    model_filename = f"model_{model_type}_{config['id']}.pkl"
                    if (output_dir / model_filename).exists():
                        logger.info(f"[SKIP] Already trained: {job_id}")
                        continue

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
        logger.info(f"  Datasets requested: {len(dataset_paths)}")
        logger.info(f"  Datasets valid: {len(dataset_paths) - skipped_datasets}")
        logger.info(f"  Model types: {len(model_types)}")
        logger.info(f"  Configs per model: {len(configs)}")
        if skipped_datasets > 0:
            logger.warning(f"  Skipped {skipped_datasets} datasets (missing or incomplete)")
        if skipped_dnn_jobs > 0:
            logger.info(f"  Skipped {skipped_dnn_jobs} DNN jobs (small datasets < 100 nuclei)")

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
                trainer = XGBoostTrainer(job.config, job.output_dir, gpu_enabled=self.gpu_enabled)
            elif job.model_type in ['LGB', 'LightGBM']:
                trainer = LightGBMTrainer(job.config, job.output_dir, gpu_enabled=self.gpu_enabled)
            elif job.model_type in ['CB', 'CatBoost']:
                trainer = CatBoostTrainer(job.config, job.output_dir)
            elif job.model_type == 'SVR':
                trainer = SVRTrainer(job.config, job.output_dir)
            elif job.model_type == 'DNN':
                trainer = DNNTrainer(job.config, job.output_dir, gpu_enabled=self.gpu_enabled)
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
            
            # DIVERGENCE CHECK: Skip saving model if DNN diverged (R2 < -2)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            diverged = metrics.get('diverged', False)
            if diverged:
                logger.warning(
                    f"[DIVERGED] {job.job_id} — DNN diverged (val_R2 < -2.0). "
                    f"Model NOT saved. Marked as failed in summary."
                )
                metrics_file = job.output_dir / f"metrics_{job.config['id']}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                return TrainingResult(
                    job_id=job.job_id,
                    model_type=job.model_type,
                    config_id=job.config['id'],
                    dataset_name=job.dataset_name,
                    success=False,
                    metrics=metrics,
                    training_time=time.time() - start_time,
                    error_message=f"DNN diverged: val_R2={metrics.get('val',{}).get('r2', 'N/A')}"
                )

            # QUALITY FILTER: Skip saving model if val_R2 < 0.5 (Poor or Failed category)
            _val_m = metrics.get('val', {})
            _val_r2 = _val_m.get('r2', _val_m.get('r2_avg', None))
            if isinstance(_val_r2, float) and _val_r2 < R2_MIN_SAVE_THRESHOLD:
                logger.warning(
                    f"[POOR] {job.job_id} — val_R2={_val_r2:.4f} < {R2_MIN_SAVE_THRESHOLD} "
                    f"(Poor/Failed category). Model NOT saved."
                )
                metrics_file = job.output_dir / f"metrics_{job.config['id']}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                return TrainingResult(
                    job_id=job.job_id,
                    model_type=job.model_type,
                    config_id=job.config['id'],
                    dataset_name=job.dataset_name,
                    success=False,
                    metrics=metrics,
                    training_time=time.time() - start_time,
                    error_message=f"Poor/Failed: val_R2={_val_r2:.4f} < {R2_MIN_SAVE_THRESHOLD}"
                )

            # Save model
            model_filename = f"model_{job.model_type}_{job.config['id']}.pkl"
            model_path = job.output_dir / model_filename
            trainer.save_model(model_path)

            # Save metrics
            metrics_file = job.output_dir / f"metrics_{job.config['id']}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            # SEED TRACKING: Record seed used for this model
            if hasattr(self, 'seed_tracker') and self.seed_tracker:
                self.seed_tracker.add_seed(
                    operation='model_training',
                    seed=42,  # Currently hardcoded
                    dataset_name=job.dataset_name,
                    model_name=job.model_type,
                    config_id=job.config['id'],
                    details={
                        'config': job.config,
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'test_samples': len(X_test)
                    }
                )

            # ✅ ACTIVATED: Model Validation (Cross-validation)
            if self.use_model_validation:
                try:
                    import numpy as np
                    X_combined = np.vstack([X_train, X_val])
                    y_combined = np.concatenate([y_train, y_val])

                    # CRITICAL: If parallel training is used, use n_jobs=1 for CV to avoid deadlock
                    cv_n_jobs = 1 if self.use_parallel_training else -1

                    cv_results = self.run_model_validation(
                        model=trainer.model,
                        model_name=f"{job.model_type}_{job.config['id']}",
                        X=X_combined,
                        y=y_combined,
                        cv_folds=5,
                        cv_n_jobs=cv_n_jobs
                    )

                    if cv_results.get('status') == 'completed':
                        cv_file = job.output_dir / f"cv_results_{job.config['id']}.json"
                        with open(cv_file, 'w') as f:
                            json.dump(cv_results, f, indent=2)
                        logger.info(f"  [CV] Saved cross-validation results")
                except Exception as e:
                    logger.warning(f"  [CV] Validation failed: {e}")

            # ✅ ACTIVATED: Overfitting Detection
            try:
                from pfaz_modules.pfaz02_ai_training.overfitting_detector import OverfittingDetector

                detector = OverfittingDetector(output_dir=str(job.output_dir))
                overfitting_results = detector.analyze_training_metrics(
                    train_metrics=metrics['train'],
                    val_metrics=metrics['val'],
                    test_metrics=metrics.get('test', {})
                )

                if overfitting_results:
                    overfitting_file = job.output_dir / f"overfitting_analysis_{job.config['id']}.json"
                    with open(overfitting_file, 'w') as f:
                        json.dump(overfitting_results, f, indent=2)
                    logger.info(f"  [OVERFITTING] Analysis saved - Severity: {overfitting_results.get('severity', 'N/A')}")
            except Exception as e:
                logger.warning(f"  [OVERFITTING] Detection failed: {e}")

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

            logger.info(f"[SUCCESS] {job.job_id} | R2={metrics['val'].get('r2', 0):.4f} | {training_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] {job.job_id} | Error: {str(e)}")
            
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
        # Defensive: Avoid division by zero when no jobs
        if len(jobs) > 0:
            logger.info(f"Avg time per job: {total_time/len(jobs):.1f} seconds")
        else:
            logger.info("Avg time per job: N/A (no jobs executed)")
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

        # ✅ Excel özet raporu — tüm model sonuçları tek tabloda (başarısız dahil)
        try:
            import pandas as pd
            rows = []
            for result in self.training_results:
                m = result.metrics or {}
                train_m = m.get('train', {})
                val_m = m.get('val', {})
                test_m = m.get('test', {})
                val_r2  = val_m.get('r2', val_m.get('r2_avg', float('nan')))
                test_r2 = test_m.get('r2', test_m.get('r2_avg', float('nan')))
                _err = result.error_message or ''
                if result.success:
                    _status = 'OK'
                elif 'Poor/Failed' in _err:
                    _status = 'POOR_R2_FILTER'
                elif 'diverged' in _err.lower():
                    _status = 'DIVERGED'
                elif _err:
                    _status = f'ERROR: {_err[:80]}'
                else:
                    _status = 'FAILED'
                rows.append({
                    'Model_Type':      result.model_type,
                    'Config_ID':       result.config_id,
                    'Dataset':         result.dataset_name,
                    'PKL_Saved':       result.success,
                    'Status_Note':     _status,
                    'Train_R2':        train_m.get('r2', train_m.get('r2_avg', float('nan'))),
                    'Train_RMSE':      train_m.get('rmse', train_m.get('rmse_avg', float('nan'))),
                    'Train_MAE':       train_m.get('mae',  train_m.get('mae_avg',  float('nan'))),
                    'Val_R2':          val_r2,
                    'Val_RMSE':        val_m.get('rmse', val_m.get('rmse_avg', float('nan'))),
                    'Val_MAE':         val_m.get('mae',  val_m.get('mae_avg',  float('nan'))),
                    'Test_R2':         test_r2,
                    'Test_RMSE':       test_m.get('rmse', test_m.get('rmse_avg', float('nan'))),
                    'Test_MAE':        test_m.get('mae',  test_m.get('mae_avg',  float('nan'))),
                    'Training_Time_s': result.training_time,
                    'Error':           result.error_message if not result.success else '',
                })

            if rows:
                df = pd.DataFrame(rows)
                excel_file = self.output_dir / 'training_results_summary.xlsx'
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='All_Results', index=False)
                    for mtype in df['Model_Type'].unique():
                        sub = df[df['Model_Type'] == mtype]
                        sheet = mtype[:31]
                        sub.to_excel(writer, sheet_name=sheet, index=False)
                logger.info(f"[OK] Excel özet raporu: {excel_file}")
        except Exception as e:
            logger.warning(f"[WARNING] Excel rapor oluşturulamadı: {e}")

    def train_all_models_parallel(self, n_configs: int = 50, use_parallel: bool = None) -> Dict:
        """
        Main entry point for training all models with multiple configurations

        This method:
        1. Loads or creates training configurations
        2. Discovers datasets from datasets_dir
        3. Creates training jobs for all combinations
        4. Trains all models in parallel or sequentially (prompts user if not specified)
        5. Saves summary report

        Args:
            n_configs: Number of configurations to use (max 50)
            use_parallel: Whether to use parallel training (None = use self.use_parallel_training or prompt)

        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAIN ALL MODELS - PARALLEL EXECUTION")
        logger.info("=" * 80)

        # Determine parallel vs sequential training mode
        if use_parallel is None:
            use_parallel = self.use_parallel_training

        # If still None, prompt user
        if use_parallel is None:
            logger.info("\n" + "=" * 80)
            logger.info("PARALEL EĞİTİM SEÇENEĞİ")
            logger.info("=" * 80)
            logger.info("Cross-validation ile paralel eğitim kullanılırsa deadlock riski vardır.")
            logger.info("Seçenekler:")
            logger.info("  1) PARALEL EĞİTİM (hızlı ama CV sıralı olacak)")
            logger.info("  2) SIRALI EĞİTİM (yavaş ama CV paralel olabilir)")
            logger.info("=" * 80)

            while True:
                choice = input("\nSeçiminiz (1 veya 2): ").strip()
                if choice == '1':
                    use_parallel = True
                    logger.info("\n✅ PARALEL EĞİTİM SEÇİLDİ")
                    logger.info("   - Model eğitimleri paralel çalışacak")
                    logger.info("   - Cross-validation sıralı çalışacak (n_jobs=1)")
                    break
                elif choice == '2':
                    use_parallel = False
                    logger.info("\n✅ SIRALI EĞİTİM SEÇİLDİ")
                    logger.info("   - Model eğitimleri sıralı çalışacak")
                    logger.info("   - Cross-validation paralel çalışabilir (n_jobs=-1)")
                    break
                else:
                    logger.warning("Geçersiz seçim! Lütfen 1 veya 2 girin.")

        # Store the choice
        self.use_parallel_training = use_parallel
        logger.info(f"\nTraining mode: {'PARALLEL' if use_parallel else 'SEQUENTIAL'}")
        logger.info("=" * 80)

        # Step 1: Load or create training configurations
        if self.training_config_path and self.training_config_path.exists():
            configs = self.load_training_configs(self.training_config_path)
        else:
            logger.warning(f"Training config not found: {self.training_config_path}")
            logger.info("Creating default training configurations...")
            configs = self._create_default_configs(n_configs)

        # Limit to n_configs
        configs = configs[:n_configs]
        logger.info(f"Using {len(configs)} training configurations")

        # Step 2: Discover datasets
        if self.datasets_dir is None or not self.datasets_dir.exists():
            raise ValueError(f"Datasets directory not found: {self.datasets_dir}")

        dataset_paths = self._discover_datasets(self.datasets_dir)
        logger.info(f"Found {len(dataset_paths)} datasets")

        # Step 3: Define model types to train
        # Base models — always included if available
        model_types = ['RF']
        if XGBOOST_AVAILABLE:
            model_types.append('XGBoost')
        if LIGHTGBM_AVAILABLE:
            model_types.append('LightGBM')
        if CATBOOST_AVAILABLE:
            model_types.append('CatBoost')
        # SVR — always available (sklearn)
        model_types.append('SVR')
        # DNN only if TF available AND use_advanced_models=True
        # For nuclear datasets (75-200 samples), DNN needs explicit opt-in
        if TF_AVAILABLE and self.use_advanced_models:
            model_types.append('DNN')
        elif TF_AVAILABLE and not self.use_advanced_models:
            logger.info("DNN available (TF installed) but disabled (use_advanced_models=False)")

        logger.info(f"Model types: {model_types}")

        # Step 4: Create training jobs
        jobs = self.create_training_jobs(
            model_types=model_types,
            configs=configs,
            dataset_paths=dataset_paths
        )

        # Step 5: Train all models
        if self.use_parallel_training and self.n_workers > 1:
            logger.info(f"\n🚀 Starting PARALLEL training with {self.n_workers} workers...")
            results = self.train_all_parallel(jobs)
        else:
            # Sequential training
            logger.info("\n🐢 Starting SEQUENTIAL training (one job at a time)...")
            results = []
            for i, job in enumerate(jobs):
                logger.info(f"Training job {i+1}/{len(jobs)}: {job.job_id}")
                result = self.train_single_job(job)
                results.append(result)

        # Step 6: Save summary report
        self.save_summary_report()

        # Step 7: Save seed tracking report
        if hasattr(self, 'seed_tracker') and self.seed_tracker:
            try:
                self.seed_tracker.print_summary()
                self.seed_tracker.save_to_excel('seed_tracking_report.xlsx')
                self.seed_tracker.save_to_json('seed_tracking_report.json')
                logger.info("[OK] Seed tracking reports saved")
            except Exception as e:
                logger.warning(f"[WARNING] Could not save seed tracking report: {e}")

        # Step 8: NuclearPatternAnalyzer — nükleer desen analizi
        try:
            from pfaz_modules.pfaz12_advanced_analytics.nuclear_pattern_analyzer import NuclearPatternAnalyzer
            from pathlib import Path as _Path
            _npa_candidates = [
                'aaa2.txt',
                str(self.output_dir / 'aaa2.txt'),
            ]
            # enriched CSV'yi de dene
            _enriched_candidates = list(_Path('outputs').glob('**/aaa2_enriched*.csv')) + \
                                   list(_Path('outputs').glob('**/aaa2_enriched*.xlsx'))
            _npa_data = None
            for _c in _npa_candidates:
                if _Path(_c).exists():
                    _npa_data = _c
                    break
            if _npa_data is None and _enriched_candidates:
                _npa_data = str(_enriched_candidates[0])
            if _npa_data:
                _npa = NuclearPatternAnalyzer(
                    data_path=_npa_data,
                    output_dir=str(self.output_dir / 'nuclear_patterns')
                )
                _npa.run_all()
                logger.info("[OK] NuclearPatternAnalyzer: Excel + grafikler -> nuclear_patterns/")
            else:
                logger.warning("[WARNING] NuclearPatternAnalyzer: veri dosyası bulunamadı (aaa2.txt yok)")
        except Exception as _npa_e:
            logger.warning(f"[WARNING] NuclearPatternAnalyzer basarisiz (devam): {_npa_e}")

        # Step 9: Return summary
        successful = len([r for r in results if r.success])
        failed = len([r for r in results if not r.success])

        return {
            'status': 'completed',
            'total_jobs': len(results),
            'successful': successful,
            'failed': failed,
            'results': results,
            'summary_file': str(self.output_dir / 'training_summary.json')
        }

    def _discover_datasets(self, datasets_dir: Path) -> List[Path]:
        """
        Discover dataset directories in datasets_dir

        Args:
            datasets_dir: Directory to search for datasets

        Returns:
            List of dataset directory paths
        """
        dataset_paths = []

        # Directories to exclude (not datasets, but metadata/reports)
        excluded_dirs = {'quality_reports', 'metadata', 'reports', 'logs', '__pycache__', '.git'}

        # Look for subdirectories that contain data files
        for subdir in datasets_dir.iterdir():
            if subdir.is_dir():
                # Skip excluded directories
                if subdir.name in excluded_dirs:
                    logger.debug(f"  Skipping non-dataset directory: {subdir.name}")
                    continue

                # Check if directory contains CSV, XLSX, or TSV files
                has_data = (
                    list(subdir.glob('*.csv')) or
                    list(subdir.glob('*.xlsx')) or
                    list(subdir.glob('*.tsv'))
                )

                if has_data:
                    dataset_paths.append(subdir)
                    logger.info(f"  Found dataset: {subdir.name}")

        if not dataset_paths:
            logger.warning(f"No datasets found in {datasets_dir}")

        return dataset_paths

    def _create_default_configs(self, n_configs: int = 50) -> List[Dict]:
        """
        Create default training configurations

        Args:
            n_configs: Number of configurations to create

        Returns:
            List of configuration dictionaries
        """
        configs = []

        # Random Forest configs (20 configs)
        rf_configs = [
            {'id': f'RF_{i:03d}', 'n_estimators': n_est, 'max_depth': depth, 'min_samples_split': split}
            for i, (n_est, depth, split) in enumerate([
                (50, 5, 2), (50, 10, 2), (50, 15, 2), (50, None, 2),
                (100, 5, 2), (100, 10, 2), (100, 15, 2), (100, None, 2),
                (200, 5, 2), (200, 10, 2), (200, 15, 2), (200, None, 2),
                (100, 10, 5), (100, 10, 10), (200, 15, 5), (200, 15, 10),
                (150, 8, 3), (150, 12, 4), (300, 10, 2), (300, 20, 5)
            ], start=1)
        ]

        # XGBoost configs (15 configs)
        xgb_configs = [
            {'id': f'XGB_{i:03d}', 'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
            for i, (n_est, lr, depth) in enumerate([
                (50, 0.1, 3), (50, 0.1, 6), (50, 0.3, 3), (50, 0.3, 6),
                (100, 0.1, 3), (100, 0.1, 6), (100, 0.3, 3), (100, 0.3, 6),
                (200, 0.05, 5), (200, 0.1, 5), (200, 0.2, 5),
                (150, 0.15, 4), (150, 0.2, 6), (300, 0.1, 4), (300, 0.05, 8)
            ], start=21)
        ]

        # DNN configs (15 configs)
        dnn_configs = [
            {
                'id': f'DNN_{i:03d}',
                'architecture': arch,
                'dropout': dropout,
                'learning_rate': lr,
                'batch_size': batch,
                'epochs': 100,
                'early_stopping_patience': 15
            }
            for i, (arch, dropout, lr, batch) in enumerate([
                ([128, 64, 32], [0.1, 0.1, 0.0], 0.001, 32),
                ([256, 128, 64], [0.1, 0.1, 0.1], 0.001, 32),
                ([512, 256, 128, 64], [0.2, 0.2, 0.1, 0.1], 0.001, 64),
                ([128, 64], [0.1, 0.0], 0.001, 32),      # DNN_039: lr 0.01->0.001 (divergence fix)
                ([256, 128], [0.2, 0.1], 0.001, 64),
                ([512, 256], [0.2, 0.2], 0.0005, 64),
                ([256, 256, 128], [0.1, 0.1, 0.1], 0.001, 32),
                ([128, 128, 64, 32], [0.1, 0.1, 0.1, 0.0], 0.001, 32),
                ([512, 256, 128], [0.3, 0.2, 0.1], 0.0005, 128),
                ([256, 128, 64, 32], [0.2, 0.1, 0.1, 0.0], 0.001, 64),
                ([128, 64, 32, 16], [0.1, 0.1, 0.1, 0.0], 0.001, 32),  # DNN_046: lr 0.002->0.001
                ([512, 384, 256], [0.2, 0.2, 0.1], 0.001, 64),
                ([384, 256, 128, 64], [0.2, 0.2, 0.1, 0.1], 0.001, 64),
                ([256, 192, 128, 64], [0.1, 0.1, 0.1, 0.1], 0.001, 32),
                ([512, 512, 256, 128], [0.3, 0.2, 0.2, 0.1], 0.0003, 128)
            ], start=36)
        ]

        # Combine all configs
        all_configs = rf_configs + xgb_configs + dnn_configs

        # Return requested number
        return all_configs[:n_configs]

    def run_hyperparameter_tuning(self, model_type: str, X_train, y_train, X_val, y_val, n_trials: int = 50) -> Dict:
        """
        Run hyperparameter tuning using Optuna

        Args:
            model_type: Model type (RF, XGBoost, DNN)
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of Optuna trials

        Returns:
            Dictionary with best parameters and metrics
        """
        try:
            from pfaz_modules.pfaz02_ai_training.hyperparameter_tuner import HyperparameterTuner

            logger.info(f"\n[HYPERPARAMETER TUNING] Starting for {model_type} with {n_trials} trials")

            tuning_dir = self.output_dir / 'hyperparameter_tuning'
            tuner = HyperparameterTuner(model_type, output_dir=str(tuning_dir), n_trials=n_trials)

            # Run tuning
            best_params = tuner.tune(X_train, y_train, X_val, y_val)

            logger.info(f"[HYPERPARAMETER TUNING] Best parameters: {best_params}")

            return {
                'status': 'completed',
                'best_params': best_params,
                'model_type': model_type,
                'n_trials': n_trials
            }

        except ImportError as e:
            logger.warning(f"[HYPERPARAMETER TUNING] Optuna not available: {e}")
            return {'status': 'skipped', 'reason': 'optuna_not_available'}
        except Exception as e:
            logger.error(f"[HYPERPARAMETER TUNING] Error: {e}")
            return {'status': 'failed', 'error': str(e)}

    def run_model_validation(self, model, model_name: str, X, y, cv_folds: int = 5, cv_n_jobs: int = 1) -> Dict:
        """
        Run cross-validation and robustness testing

        Args:
            model: Trained model
            model_name: Model name
            X, y: Data for validation
            cv_folds: Number of CV folds
            cv_n_jobs: Number of parallel jobs for CV (default=1 to avoid nested parallelization deadlock)

        Returns:
            Dictionary with validation results
        """
        try:
            from pfaz_modules.pfaz02_ai_training.model_validator import CrossValidationAnalyzer

            logger.info(f"\n[MODEL VALIDATION] Starting for {model_name}")

            validation_dir = self.output_dir / 'model_validation'
            validator = CrossValidationAnalyzer(model, model_name, output_dir=str(validation_dir))

            # Run cross-validation with controlled n_jobs
            cv_results = validator.run_cv(X, y, cv=cv_folds, n_jobs=cv_n_jobs)

            logger.info(f"[MODEL VALIDATION] CV Results: {cv_results}")

            return {
                'status': 'completed',
                'cv_results': cv_results,
                'model_name': model_name,
                'cv_folds': cv_folds,
                'cv_n_jobs': cv_n_jobs
            }

        except ImportError as e:
            logger.warning(f"[MODEL VALIDATION] Validator not available: {e}")
            return {'status': 'skipped', 'reason': 'validator_not_available'}
        except Exception as e:
            logger.error(f"[MODEL VALIDATION] Error: {e}")
            return {'status': 'failed', 'error': str(e)}


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
    
    print("\n[SUCCESS] TEST COMPLETED!")


if __name__ == "__main__":
    main()
