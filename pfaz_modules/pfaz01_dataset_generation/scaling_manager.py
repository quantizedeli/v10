"""
Scaling Manager for PFAZ1 Dataset Generation
=============================================

Manages data scaling with physics-aware feature handling:
- NoScaling: No transformation (raw data)
- Standard: StandardScaler (Z-score normalization)
- Robust: RobustScaler (median and IQR)

CRITICAL: Discrete/categorical features are NEVER scaled!
- A, Z, N: Discrete counts
- SPIN: Discrete quantum number
- PARITY: Binary (-1, +1)
- magic_character: Binary (0, 1)

Author: Nuclear Physics AI Project
Version: 1.0.0 (FAZ 3)
Date: 2025-11-23
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# Features that should NEVER be scaled (discrete/categorical)
DISCRETE_FEATURES = [
    'NUCLEUS',  # String ID
    'A',  # Mass number (discrete)
    'Z',  # Proton count (discrete)
    'N',  # Neutron count (discrete)
    'Nn',  # Neutron count (alternative name)
    'Np',  # Proton count (alternative name)
    'SPIN',  # Discrete quantum number (0, 0.5, 1, 1.5, 2, ...)
    'PARITY',  # Binary (-1, +1)
    'magic_character',  # Binary (0, 1)
    'magic_n',  # Binary
    'magic_p',  # Binary
    'magic_np',  # Binary
]


class ScalingManager:
    """
    Manages data scaling for nuclear physics datasets.

    Supports 3 scaling methods:
    1. NoScaling: No transformation
    2. Standard: (X - mean) / std
    3. Robust: (X - median) / IQR

    Features:
    - Automatic exclusion of discrete/categorical features
    - Fit on train, transform on train/val/test
    - Save/load scaler parameters
    - Inverse transform capability
    - JSON export of scaling metadata
    """

    AVAILABLE_METHODS = ['NoScaling', 'Standard', 'Robust', 'MinMax']

    def __init__(self, method: str = 'NoScaling'):
        """
        Initialize ScalingManager.

        Args:
            method: Scaling method ('NoScaling', 'Standard', 'Robust', 'MinMax')
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Invalid scaling method: {method}. "
                f"Available: {self.AVAILABLE_METHODS}"
            )

        self.method = method
        self.scaler_params = {}
        self.features_to_scale = []
        self.features_excluded = []
        self.is_fitted = False

        logger.info(f"ScalingManager initialized with method: {method}")

    def _identify_features_to_scale(self, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
        """
        Identify which features should be scaled and which should be excluded.

        Args:
            feature_cols: All feature column names

        Returns:
            (features_to_scale, features_excluded)
        """
        features_to_scale = []
        features_excluded = []

        for col in feature_cols:
            if col in DISCRETE_FEATURES:
                features_excluded.append(col)
            else:
                features_to_scale.append(col)

        logger.info(f"Features to scale: {len(features_to_scale)}")
        logger.info(f"Features excluded (discrete): {len(features_excluded)}")

        return features_to_scale, features_excluded

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        Fit scaler on training data.

        Args:
            df: Training DataFrame
            feature_cols: Feature column names to consider
        """
        if self.method == 'NoScaling':
            logger.info("NoScaling selected - no fitting needed")
            self.features_to_scale = []
            self.features_excluded = feature_cols
            self.is_fitted = True
            return

        # Identify which features to scale
        self.features_to_scale, self.features_excluded = self._identify_features_to_scale(feature_cols)

        if len(self.features_to_scale) == 0:
            logger.warning("No features to scale!")
            self.is_fitted = True
            return

        # Extract data for continuous features only
        X = df[self.features_to_scale].values

        # Calculate scaler parameters
        if self.method == 'Standard':
            # StandardScaler: (X - mean) / std
            self.scaler_params = {
                'mean': np.mean(X, axis=0).tolist(),
                'std': np.std(X, axis=0).tolist(),
                'var': np.var(X, axis=0).tolist()
            }
            logger.info("Standard scaler fitted (mean, std)")

        elif self.method == 'Robust':
            # RobustScaler: (X - median) / IQR
            self.scaler_params = {
                'median': np.median(X, axis=0).tolist(),
                'q25': np.percentile(X, 25, axis=0).tolist(),
                'q75': np.percentile(X, 75, axis=0).tolist(),
                'iqr': (np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)).tolist()
            }
            logger.info("Robust scaler fitted (median, IQR)")

        elif self.method == 'MinMax':
            # MinMaxScaler: (X - min) / (max - min)  →  [0, 1]
            self.scaler_params = {
                'min': np.min(X, axis=0).tolist(),
                'max': np.max(X, axis=0).tolist(),
                'range': (np.max(X, axis=0) - np.min(X, axis=0)).tolist()
            }
            logger.info("MinMax scaler fitted (min, max) -> [0, 1]")

        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted scaler.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted! Call fit() first.")

        if self.method == 'NoScaling':
            return df.copy()

        # Create copy
        df_scaled = df.copy()

        if len(self.features_to_scale) == 0:
            logger.warning("No features to scale, returning original data")
            return df_scaled

        # Extract continuous features
        X = df[self.features_to_scale].values

        # Apply scaling
        if self.method == 'Standard':
            mean = np.array(self.scaler_params['mean'])
            std = np.array(self.scaler_params['std'])
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            X_scaled = (X - mean) / std

        elif self.method == 'Robust':
            median = np.array(self.scaler_params['median'])
            iqr = np.array(self.scaler_params['iqr'])
            # Avoid division by zero
            iqr = np.where(iqr == 0, 1.0, iqr)
            X_scaled = (X - median) / iqr

        elif self.method == 'MinMax':
            min_val = np.array(self.scaler_params['min'])
            range_val = np.array(self.scaler_params['range'])
            # Avoid division by zero
            range_val = np.where(range_val == 0, 1.0, range_val)
            X_scaled = (X - min_val) / range_val

        # Update only the scaled features
        df_scaled[self.features_to_scale] = X_scaled

        return df_scaled

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Training DataFrame
            feature_cols: Feature column names

        Returns:
            Transformed DataFrame
        """
        self.fit(df, feature_cols)
        return self.transform(df)

    def inverse_transform(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.

        Args:
            df_scaled: Scaled DataFrame

        Returns:
            Original-scale DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted!")

        if self.method == 'NoScaling':
            return df_scaled.copy()

        # Create copy
        df_original = df_scaled.copy()

        if len(self.features_to_scale) == 0:
            return df_original

        # Extract scaled features
        X_scaled = df_scaled[self.features_to_scale].values

        # Apply inverse scaling
        if self.method == 'Standard':
            mean = np.array(self.scaler_params['mean'])
            std = np.array(self.scaler_params['std'])
            X_original = X_scaled * std + mean

        elif self.method == 'Robust':
            median = np.array(self.scaler_params['median'])
            iqr = np.array(self.scaler_params['iqr'])
            X_original = X_scaled * iqr + median

        elif self.method == 'MinMax':
            min_val = np.array(self.scaler_params['min'])
            range_val = np.array(self.scaler_params['range'])
            X_original = X_scaled * range_val + min_val

        # Update features
        df_original[self.features_to_scale] = X_original

        return df_original

    def get_metadata(self) -> Dict:
        """
        Get scaling metadata for reproducibility.

        Returns:
            Metadata dictionary
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted!")

        metadata = {
            'method': self.method,
            'features_scaled': self.features_to_scale,
            'features_excluded': self.features_excluded,
            'n_features_scaled': len(self.features_to_scale),
            'n_features_excluded': len(self.features_excluded),
            'scaler_params': self.scaler_params if self.method != 'NoScaling' else {}
        }

        return metadata

    def save_metadata_json(self, filepath: str):
        """
        Save scaling metadata to JSON file.

        Args:
            filepath: Output JSON file path
        """
        metadata = self.get_metadata()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Scaling metadata saved to: {filepath}")

    @staticmethod
    def load_from_metadata(filepath: str) -> 'ScalingManager':
        """
        Load ScalingManager from saved metadata.

        Args:
            filepath: JSON metadata file path

        Returns:
            ScalingManager instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        scaler = ScalingManager(method=metadata['method'])
        scaler.features_to_scale = metadata['features_scaled']
        scaler.features_excluded = metadata['features_excluded']
        scaler.scaler_params = metadata['scaler_params']
        scaler.is_fitted = True

        logger.info(f"ScalingManager loaded from: {filepath}")
        return scaler


# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

def test_scaling_manager():
    """Test ScalingManager with sample data."""
    print("\n" + "=" * 80)
    print("SCALING MANAGER TEST")
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame({
        'A': np.random.randint(20, 250, n_samples),  # Discrete - should NOT be scaled
        'Z': np.random.randint(10, 100, n_samples),  # Discrete - should NOT be scaled
        'N': np.random.randint(10, 150, n_samples),  # Discrete - should NOT be scaled
        'SPIN': np.random.choice([0, 0.5, 1, 1.5, 2], n_samples),  # Discrete - should NOT be scaled
        'PARITY': np.random.choice([-1, 1], n_samples),  # Discrete - should NOT be scaled
        'BE_volume': np.random.randn(n_samples) * 500 + 2500,  # Continuous - should be scaled
        'BE_surface': np.random.randn(n_samples) * 100 - 500,  # Continuous - should be scaled
        'MM': np.random.randn(n_samples) * 2,  # Continuous - should be scaled
        'Beta_2': np.random.uniform(-0.3, 0.4, n_samples)  # Continuous - should be scaled
    })

    # Split into train/test
    train_df = df[:70]
    test_df = df[70:]

    feature_cols = ['A', 'Z', 'N', 'SPIN', 'PARITY', 'BE_volume', 'BE_surface', 'MM', 'Beta_2']

    # Test 1: NoScaling
    print("\n-> Test 1: NoScaling")
    scaler_none = ScalingManager(method='NoScaling')
    train_scaled = scaler_none.fit_transform(train_df, feature_cols)
    print(f"  Train data unchanged: {train_df.equals(train_scaled)}")

    # Test 2: Standard Scaling
    print("\n-> Test 2: Standard Scaling")
    scaler_std = ScalingManager(method='Standard')
    train_std = scaler_std.fit_transform(train_df, feature_cols)
    test_std = scaler_std.transform(test_df)

    print(f"  Features scaled: {scaler_std.features_to_scale}")
    print(f"  Features excluded: {scaler_std.features_excluded}")
    print(f"  Discrete features unchanged (A): {(train_df['A'] == train_std['A']).all()}")
    print(f"  Continuous features scaled (BE_volume): {not (train_df['BE_volume'] == train_std['BE_volume']).all()}")
    print(f"  Mean of scaled BE_volume: {train_std['BE_volume'].mean():.6f} (should be ~0)")
    print(f"  Std of scaled BE_volume: {train_std['BE_volume'].std():.6f} (should be ~1)")

    # Test 3: Robust Scaling
    print("\n-> Test 3: Robust Scaling")
    scaler_robust = ScalingManager(method='Robust')
    train_robust = scaler_robust.fit_transform(train_df, feature_cols)
    test_robust = scaler_robust.transform(test_df)

    print(f"  Features scaled: {scaler_robust.features_to_scale}")
    print(f"  Discrete features unchanged (SPIN): {(train_df['SPIN'] == train_robust['SPIN']).all()}")
    print(f"  Continuous features scaled (MM): {not (train_df['MM'] == train_robust['MM']).all()}")

    # Test 4: Inverse Transform
    print("\n-> Test 4: Inverse Transform")
    train_recovered = scaler_std.inverse_transform(train_std)
    diff = np.abs(train_df[scaler_std.features_to_scale].values - train_recovered[scaler_std.features_to_scale].values).max()
    print(f"  Max difference after inverse transform: {diff:.10f} (should be ~0)")

    # Test 5: Metadata
    print("\n-> Test 5: Metadata")
    metadata = scaler_std.get_metadata()
    print(f"  Method: {metadata['method']}")
    print(f"  N features scaled: {metadata['n_features_scaled']}")
    print(f"  N features excluded: {metadata['n_features_excluded']}")

    # Test 6: Save/Load
    print("\n-> Test 6: Save/Load Metadata")
    test_path = Path("test_scaler_metadata.json")
    scaler_std.save_metadata_json(str(test_path))
    scaler_loaded = ScalingManager.load_from_metadata(str(test_path))
    print(f"  Loaded method: {scaler_loaded.method}")
    print(f"  Loaded features: {scaler_loaded.features_to_scale == scaler_std.features_to_scale}")
    test_path.unlink()  # Clean up

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_scaling_manager()
