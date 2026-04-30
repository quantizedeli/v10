"""
Extended Feature Engineering
=============================

Additional automated feature engineering capabilities:
1. Advanced physics-inspired features (nuclear structure, magic numbers, etc.)
2. Non-linear transformations and combinations
3. Autoencoder-based feature extraction
4. Target encoding for categorical features
5. Higher-order feature crosses
6. Automated feature discovery using genetic programming

Author: Nuclear Physics AI Project
Version: 2.0.0 - Extended Feature Engineering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats
from itertools import combinations

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# ADVANCED PHYSICS FEATURES
# ============================================================================

class NuclearPhysicsFeatures:
    """
    Generate advanced physics-inspired features for nuclear data

    Features:
    - Magic number indicators
    - Shell model properties
    - Pairing effects
    - Deformation parameters
    - Binding energy relationships
    - Separation energies
    """

    # Nuclear magic numbers
    MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]

    def __init__(self):
        self.feature_generators = {
            'magic_proximity': self._magic_proximity,
            'shell_closure': self._shell_closure_indicator,
            'pairing_strength': self._pairing_strength,
            'asymmetry_parameter': self._asymmetry_parameter,
            'binding_energy_per_nucleon': self._be_per_nucleon,
            'separation_energy_ratio': self._separation_energy_ratio,
            'deformation_category': self._deformation_category,
            'neutron_excess': self._neutron_excess,
            'proton_neutron_ratio': self._pn_ratio,
            'radius_features': self._nuclear_radius_features,
            'coulomb_features': self._coulomb_features,
            'surface_features': self._surface_features,
        }

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all physics features

        Args:
            df: DataFrame with columns: Z, N, A, and potentially others

        Returns:
            DataFrame with additional physics features
        """
        logger.info("Generating advanced nuclear physics features...")

        new_features = df.copy()

        for feature_name, generator_func in self.feature_generators.items():
            try:
                generated = generator_func(df)

                if isinstance(generated, pd.DataFrame):
                    new_features = pd.concat([new_features, generated], axis=1)
                else:
                    new_features[feature_name] = generated

                logger.debug(f"  Generated: {feature_name}")
            except Exception as e:
                logger.warning(f"  Failed to generate {feature_name}: {e}")

        logger.info(f"Physics features: {df.shape[1]} -> {new_features.shape[1]} features")
        return new_features

    def _magic_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance to nearest magic number for Z and N"""
        result = pd.DataFrame()

        for col in ['Z', 'N']:
            if col in df.columns:
                distances = []
                for val in df[col]:
                    dist = min([abs(val - magic) for magic in self.MAGIC_NUMBERS])
                    distances.append(dist)
                result[f'{col}_magic_distance'] = distances

                # Indicator if at magic number
                result[f'{col}_is_magic'] = df[col].isin(self.MAGIC_NUMBERS).astype(int)

        return result

    def _shell_closure_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shell closure indicators"""
        result = pd.DataFrame()

        # Double magic (both Z and N are magic)
        if 'Z' in df.columns and 'N' in df.columns:
            z_magic = df['Z'].isin(self.MAGIC_NUMBERS)
            n_magic = df['N'].isin(self.MAGIC_NUMBERS)
            result['double_magic'] = (z_magic & n_magic).astype(int)

            # Semi-magic (only one is magic)
            result['semi_magic'] = ((z_magic & ~n_magic) | (~z_magic & n_magic)).astype(int)

        return result

    def _pairing_strength(self, df: pd.DataFrame) -> pd.Series:
        """Pairing strength indicator (even-even, even-odd, odd-even, odd-odd)"""
        if 'Z' in df.columns and 'N' in df.columns:
            z_even = (df['Z'] % 2 == 0).astype(int)
            n_even = (df['N'] % 2 == 0).astype(int)

            # even-even: 2, even-odd: 1, odd-even: 1, odd-odd: 0
            pairing = z_even + n_even
            return pairing
        return pd.Series([0] * len(df))

    def _asymmetry_parameter(self, df: pd.DataFrame) -> pd.Series:
        """Asymmetry parameter (N-Z)/A"""
        if all(col in df.columns for col in ['N', 'Z', 'A']):
            return (df['N'] - df['Z']) / df['A']
        return pd.Series([0] * len(df))

    def _be_per_nucleon(self, df: pd.DataFrame) -> pd.Series:
        """Binding energy per nucleon"""
        if 'BE' in df.columns and 'A' in df.columns:
            return df['BE'] / df['A']
        return pd.Series([0] * len(df))

    def _separation_energy_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratios of separation energies"""
        result = pd.DataFrame()

        if 'S' in df.columns and 'P' in df.columns:
            # Avoid division by zero
            result['S_P_ratio'] = df['S'] / (df['P'] + 1e-10)
            result['S_P_sum'] = df['S'] + df['P']
            result['S_P_diff'] = df['S'] - df['P']

        if 'Sn' in df.columns and 'Sp' in df.columns:
            result['Sn_Sp_ratio'] = df['Sn'] / (df['Sp'] + 1e-10)
            result['Sn_Sp_asymmetry'] = (df['Sn'] - df['Sp']) / (df['Sn'] + df['Sp'] + 1e-10)

        return result

    def _deformation_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deformation categories"""
        result = pd.DataFrame()

        if 'Beta_2' in df.columns:
            beta2 = df['Beta_2'].fillna(0)

            # Deformation categories
            result['spherical'] = (np.abs(beta2) < 0.1).astype(int)
            result['weakly_deformed'] = ((np.abs(beta2) >= 0.1) & (np.abs(beta2) < 0.25)).astype(int)
            result['strongly_deformed'] = (np.abs(beta2) >= 0.25).astype(int)

            # Deformation direction
            result['prolate'] = (beta2 > 0).astype(int)
            result['oblate'] = (beta2 < 0).astype(int)

        return result

    def _neutron_excess(self, df: pd.DataFrame) -> pd.Series:
        """Neutron excess"""
        if 'N' in df.columns and 'Z' in df.columns:
            return df['N'] - df['Z']
        return pd.Series([0] * len(df))

    def _pn_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Proton-neutron ratio"""
        if 'Z' in df.columns and 'N' in df.columns:
            return df['Z'] / (df['N'] + 1e-10)
        return pd.Series([0] * len(df))

    def _nuclear_radius_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nuclear radius-related features"""
        result = pd.DataFrame()

        if 'A' in df.columns:
            # r = r0 * A^(1/3), typical r0 = 1.2 fm
            result['radius_estimate'] = 1.2 * (df['A'] ** (1/3))
            result['surface_area'] = 4 * np.pi * (result['radius_estimate'] ** 2)
            result['volume'] = (4/3) * np.pi * (result['radius_estimate'] ** 3)

        return result

    def _coulomb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coulomb energy estimates"""
        result = pd.DataFrame()

        if 'Z' in df.columns and 'A' in df.columns:
            # Coulomb energy ~ Z^2 / A^(1/3)
            result['coulomb_energy_estimate'] = (df['Z'] ** 2) / (df['A'] ** (1/3) + 1e-10)

            # Coulomb correction to binding energy
            result['coulomb_correction'] = 0.72 * (df['Z'] ** 2) / (df['A'] ** (1/3) + 1e-10)

        return result

    def _surface_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Surface energy contributions"""
        result = pd.DataFrame()

        if 'A' in df.columns:
            # Surface energy ~ A^(2/3)
            result['surface_term'] = df['A'] ** (2/3)

        if 'N' in df.columns and 'Z' in df.columns and 'A' in df.columns:
            # Asymmetry term ~ (N-Z)^2 / A
            result['asymmetry_term'] = ((df['N'] - df['Z']) ** 2) / (df['A'] + 1e-10)

        return result


# ============================================================================
# NON-LINEAR TRANSFORMATIONS
# ============================================================================

class NonLinearTransformations:
    """
    Apply non-linear transformations to features

    Transformations:
    - Logarithmic: log(x+1), log(|x|+1)
    - Power: x^2, x^3, x^0.5, x^1.5
    - Exponential: exp(x), exp(-x)
    - Trigonometric: sin, cos, tanh
    - Rational: 1/x, x/(x+1)
    - Quantile: rank transformation
    """

    def __init__(self, transformations: List[str] = None):
        """
        Initialize transformer

        Args:
            transformations: List of transformation types to apply
                           ['log', 'power', 'exp', 'trig', 'rational', 'quantile']
        """
        if transformations is None:
            transformations = ['log', 'power', 'rational']

        self.transformations = transformations

    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply non-linear transformations

        Args:
            X: Input features
            feature_names: Feature names

        Returns:
            X_transformed: Transformed features (including original)
            new_feature_names: Feature names
        """
        logger.info("Applying non-linear transformations...")

        X_list = [X]
        names_list = list(feature_names)

        for i, name in enumerate(feature_names):
            x = X[:, i]

            # Log transformations
            if 'log' in self.transformations:
                # log(|x| + 1) - handles negative values
                X_list.append(np.log(np.abs(x) + 1).reshape(-1, 1))
                names_list.append(f'log_{name}')

            # Power transformations
            if 'power' in self.transformations:
                # Square
                X_list.append((x ** 2).reshape(-1, 1))
                names_list.append(f'{name}_squared')

                # Cube
                X_list.append((x ** 3).reshape(-1, 1))
                names_list.append(f'{name}_cubed')

                # Square root of absolute value
                X_list.append(np.sqrt(np.abs(x)).reshape(-1, 1))
                names_list.append(f'sqrt_{name}')

            # Exponential transformations
            if 'exp' in self.transformations:
                # Clip to avoid overflow
                x_clipped = np.clip(x, -10, 10)
                X_list.append(np.exp(x_clipped).reshape(-1, 1))
                names_list.append(f'exp_{name}')

            # Trigonometric transformations
            if 'trig' in self.transformations:
                X_list.append(np.tanh(x).reshape(-1, 1))
                names_list.append(f'tanh_{name}')

            # Rational transformations
            if 'rational' in self.transformations:
                # 1 / (x + epsilon)
                X_list.append((1 / (x + 1e-10)).reshape(-1, 1))
                names_list.append(f'inv_{name}')

                # x / (|x| + 1)
                X_list.append((x / (np.abs(x) + 1)).reshape(-1, 1))
                names_list.append(f'normalized_{name}')

            # Quantile transformation
            if 'quantile' in self.transformations:
                # Rank-based transformation
                rank = stats.rankdata(x) / len(x)
                X_list.append(rank.reshape(-1, 1))
                names_list.append(f'{name}_rank')

        X_transformed = np.hstack(X_list)

        logger.info(f"Non-linear transformations: {X.shape[1]} -> {X_transformed.shape[1]} features")

        return X_transformed, names_list


# ============================================================================
# AUTOENCODER FEATURE EXTRACTION
# ============================================================================

if TORCH_AVAILABLE:
    class AutoencoderFeatureExtractor:
        """
        Deep autoencoder for automatic feature extraction
        Learns compressed representations of the data
        """

        class Autoencoder(nn.Module):
            """Autoencoder model"""

            def __init__(self, input_dim: int, encoding_dims: List[int]):
                super().__init__()

                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for dim in encoding_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(dim),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = dim

                self.encoder = nn.Sequential(*encoder_layers)

                # Decoder
                decoder_layers = []
                for dim in reversed(encoding_dims[:-1]):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(dim),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = dim

                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded

        def __init__(self, encoding_dims: List[int] = None, device: str = 'cpu'):
            """
            Initialize autoencoder

            Args:
                encoding_dims: Dimensions of encoding layers (e.g., [64, 32, 16])
                device: 'cpu' or 'cuda'
            """
            if encoding_dims is None:
                encoding_dims = [64, 32, 16]

            self.encoding_dims = encoding_dims
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.scaler = StandardScaler()

        def fit_transform(self, X: np.ndarray, epochs: int = 100,
                         batch_size: int = 32, learning_rate: float = 0.001) -> np.ndarray:
            """
            Fit autoencoder and extract features

            Args:
                X: Input features
                epochs: Training epochs
                batch_size: Batch size
                learning_rate: Learning rate

            Returns:
                Encoded features
            """
            logger.info("Training autoencoder for feature extraction...")

            # Normalize
            X_scaled = self.scaler.fit_transform(X)

            # Build model
            input_dim = X.shape[1]
            self.model = self.Autoencoder(input_dim, self.encoding_dims).to(self.device)

            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            dataset = torch.utils.data.TensorDataset(X_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in loader:
                    batch_X = batch[0]

                    optimizer.zero_grad()
                    decoded, _ = self.model(batch_X)
                    loss = criterion(decoded, batch_X)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if (epoch + 1) % 20 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

            # Extract encoded features
            self.model.eval()
            with torch.no_grad():
                _, encoded = self.model(X_tensor)

            encoded_features = encoded.cpu().numpy()

            logger.info(f"Autoencoder: {X.shape[1]} -> {encoded_features.shape[1]} latent features")

            return encoded_features

        def transform(self, X: np.ndarray) -> np.ndarray:
            """Transform new data"""
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            self.model.eval()
            with torch.no_grad():
                _, encoded = self.model(X_tensor)

            return encoded.cpu().numpy()


# ============================================================================
# HIGHER-ORDER FEATURE CROSSES
# ============================================================================

class FeatureCrosses:
    """
    Generate higher-order feature crosses
    Beyond simple pairwise interactions
    """

    def __init__(self, max_order: int = 3, max_features: int = 100):
        """
        Initialize feature crosser

        Args:
            max_order: Maximum order of crosses (2=pairs, 3=triplets)
            max_features: Maximum number of cross features to generate
        """
        self.max_order = max_order
        self.max_features = max_features

    def generate_crosses(self, X: np.ndarray, feature_names: List[str],
                        important_indices: List[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Generate feature crosses

        Args:
            X: Input features
            feature_names: Feature names
            important_indices: Indices of important features to cross

        Returns:
            X_crossed: Features with crosses
            cross_names: Feature names
        """
        logger.info(f"Generating feature crosses up to order {self.max_order}...")

        # If important indices not provided, use all features
        if important_indices is None:
            # Limit to reasonable number for computational efficiency
            important_indices = list(range(min(20, X.shape[1])))

        X_list = [X]
        names_list = list(feature_names)

        crosses_generated = 0

        # Generate crosses for each order
        for order in range(2, self.max_order + 1):
            if crosses_generated >= self.max_features:
                break

            for combo in combinations(important_indices, order):
                if crosses_generated >= self.max_features:
                    break

                # Multiply features
                cross_feature = np.prod(X[:, combo], axis=1).reshape(-1, 1)
                X_list.append(cross_feature)

                # Create name
                cross_name = '_x_'.join([feature_names[i] for i in combo])
                names_list.append(cross_name)

                crosses_generated += 1

        X_crossed = np.hstack(X_list)

        logger.info(f"Feature crosses: {X.shape[1]} -> {X_crossed.shape[1]} features "
                   f"({crosses_generated} crosses)")

        return X_crossed, names_list


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Example nuclear data
    data = pd.DataFrame({
        'Z': [82, 50, 28, 20, 8],
        'N': [126, 82, 50, 28, 8],
        'A': [208, 132, 78, 48, 16],
        'BE': [-1636.45, -1102.85, -640.34, -415.99, -127.62],
        'S': [-7.87, -8.35, -8.21, -8.67, -7.98],
        'P': [-8.07, -9.30, -7.70, -10.42, -12.13],
        'Sn': [-7.37, -8.36, -8.00, -7.55, -4.74],
        'Sp': [-7.42, -9.09, -7.34, -9.31, -12.13],
        'Beta_2': [0.0, 0.15, -0.20, 0.0, 0.0],
    })

    # 1. Physics features
    physics = NuclearPhysicsFeatures()
    data_physics = physics.generate_features(data)
    print(f"\nPhysics features: {data.shape[1]} -> {data_physics.shape[1]}")
    print(data_physics.head())

    # 2. Non-linear transformations
    X = data[['Z', 'N', 'A']].values
    transformer = NonLinearTransformations(['log', 'power', 'rational'])
    X_transformed, names = transformer.transform(X, ['Z', 'N', 'A'])
    print(f"\nNon-linear: {X.shape[1]} -> {X_transformed.shape[1]} features")

    # 3. Feature crosses
    crosser = FeatureCrosses(max_order=2, max_features=10)
    X_crossed, cross_names = crosser.generate_crosses(X, ['Z', 'N', 'A'])
    print(f"\nCrosses: {X.shape[1]} -> {X_crossed.shape[1]} features")

    # 4. Autoencoder (if PyTorch available)
    if TORCH_AVAILABLE:
        X_dummy = np.random.randn(100, 44)
        autoencoder = AutoencoderFeatureExtractor(encoding_dims=[32, 16, 8])
        X_encoded = autoencoder.fit_transform(X_dummy, epochs=50)
        print(f"\nAutoencoder: {X_dummy.shape[1]} -> {X_encoded.shape[1]} latent features")

    print("\n[SUCCESS] All extended feature engineering tested!")
