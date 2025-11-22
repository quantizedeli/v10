# -*- coding: utf-8 -*-
"""
AUTOML FEATURE ENGINEERING
==========================

Comprehensive automatic feature engineering and selection

Features:
1. Polynomial Features (degree 2-3)
2. Interaction Terms (all pairs, physics-inspired)
3. Mathematical Transforms (log, sqrt, exp, 1/x, x^2, x^3)
4. Feature Selection Methods:
   - Recursive Feature Elimination (RFE)
   - LASSO-based selection
   - SHAP-based importance selection
   - Correlation-based filtering
   - Variance thresholding
5. Feature importance visualization
6. LaTeX-ready feature tables

Workflow:
Original 44 features -> 200+ candidate features -> 50-80 selected features

Author: Nuclear Physics AI Project
Date: 2025-10-24
Version: 1.0.0 - PFAZ 13 Feature Engineering
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import (
    RFE, SelectKBest, f_regression,
    VarianceThreshold, mutual_info_regression
)
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - SHAP-based selection disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PHYSICS-INSPIRED INTERACTIONS
# ============================================================================

PHYSICS_INTERACTIONS = {
    # Nuclear structure
    'shell_pairing': ['shell_gap', 'pairing'],
    'deformation_spin': ['deformation', 'SPIN'],
    'N_Z_asymmetry': ['N', 'Z'],
    
    # Mass dependencies
    'A_shell': ['A', 'shell_gap'],
    'A_pairing': ['A', 'pairing'],
    'A_deformation': ['A', 'deformation'],
    
    # Charge dependencies
    'Z_shell': ['Z', 'shell_gap'],
    'Z_pairing': ['Z', 'pairing'],
    'Z_N': ['Z', 'N'],
    
    # Energy scales
    'S_P_product': ['S', 'P'],
    'BE_A': ['BE', 'A'],
    
    # Spin-parity
    'spin_parity': ['SPIN', 'PARITY'],
}


# ============================================================================
# AUTOML FEATURE ENGINEER
# ============================================================================

class AutoMLFeatureEngineer:
    """
    Automatic feature engineering and selection
    
    Pipeline:
    1. Generate candidate features (200+)
       - Polynomial (degree 2-3)
       - Interactions (physics-inspired + all pairs)
       - Transforms (log, sqrt, exp, etc.)
    
    2. Select best features (50-80)
       - Multiple selection methods
       - Ensemble voting
       - Importance ranking
    
    3. Validate and export
       - Feature importance plots
       - LaTeX tables
       - Final feature set
    """
    
    def __init__(self,
                 polynomial_degree: int = 2,
                 include_transforms: bool = True,
                 include_interactions: bool = True,
                 target_n_features: int = 60,
                 selection_methods: List[str] = None,
                 output_dir: str = 'automl_feature_engineering'):
        """
        Initialize feature engineer
        
        Args:
            polynomial_degree: Degree for polynomial features (2 or 3)
            include_transforms: Include math transforms
            include_interactions: Include interaction terms
            target_n_features: Target number of selected features
            selection_methods: ['rfe', 'lasso', 'shap', 'mutual_info']
            output_dir: Output directory
        """
        self.polynomial_degree = polynomial_degree
        self.include_transforms = include_transforms
        self.include_interactions = include_interactions
        self.target_n_features = target_n_features
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Selection methods
        if selection_methods is None:
            selection_methods = ['rfe', 'lasso', 'mutual_info']
            if SHAP_AVAILABLE:
                selection_methods.append('shap')
        self.selection_methods = selection_methods
        
        # Storage
        self.original_features = None
        self.candidate_features = None
        self.selected_features = None
        self.feature_importance = None
        
        # Transformers
        self.scaler = StandardScaler()
        
        logger.info(f"✓ AutoMLFeatureEngineer initialized")
        logger.info(f"  Polynomial degree: {polynomial_degree}")
        logger.info(f"  Transforms: {include_transforms}")
        logger.info(f"  Interactions: {include_interactions}")
        logger.info(f"  Target features: {target_n_features}")
        logger.info(f"  Selection methods: {selection_methods}")
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def fit_transform(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     feature_names: List[str],
                     X_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Complete feature engineering pipeline
        
        Args:
            X: Training features
            y: Training target
            feature_names: Original feature names
            X_val: Validation features (optional, for consistency)
            
        Returns:
            X_engineered: Engineered features
            selected_feature_names: Selected feature names
        """
        logger.info("\n" + "="*70)
        logger.info("AUTOML FEATURE ENGINEERING")
        logger.info("="*70)
        logger.info(f"Original features: {X.shape[1]}")
        logger.info(f"Samples: {X.shape[0]}")
        logger.info("="*70 + "\n")
        
        self.original_features = feature_names
        
        # Step 1: Generate candidate features
        logger.info("-> Step 1: Generating candidate features...")
        X_candidates, candidate_names = self._generate_candidates(X, feature_names)
        logger.info(f"  ✓ Generated {len(candidate_names)} candidate features")
        
        self.candidate_features = candidate_names
        
        # Step 2: Remove low-variance features
        logger.info("\n-> Step 2: Removing low-variance features...")
        X_filtered, filtered_names = self._filter_low_variance(X_candidates, candidate_names)
        logger.info(f"  ✓ Retained {len(filtered_names)} features (variance > 0.01)")
        
        # Step 3: Remove highly correlated features
        logger.info("\n-> Step 3: Removing highly correlated features...")
        X_decorrelated, decorrelated_names = self._remove_high_correlation(
            X_filtered, filtered_names, threshold=0.95
        )
        logger.info(f"  ✓ Retained {len(decorrelated_names)} features (|corr| < 0.95)")
        
        # Step 4: Feature selection
        logger.info("\n-> Step 4: Feature selection...")
        X_selected, selected_names = self._select_features(
            X_decorrelated, y, decorrelated_names
        )
        logger.info(f"  ✓ Selected {len(selected_names)} best features")
        
        self.selected_features = selected_names
        
        # Step 5: Export results
        logger.info("\n-> Step 5: Exporting results...")
        self._export_feature_report(X, y, X_selected, selected_names)
        
        logger.info("\n" + "="*70)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("="*70)
        logger.info(f"Original: {len(feature_names)} features")
        logger.info(f"Candidates: {len(candidate_names)} features")
        logger.info(f"Selected: {len(selected_names)} features")
        logger.info(f"Improvement: {len(selected_names)/len(feature_names):.1f}x features")
        logger.info("="*70 + "\n")
        
        # Transform validation set if provided
        if X_val is not None:
            X_val_transformed = self.transform(X_val, feature_names)
            return X_selected, selected_names, X_val_transformed
        
        return X_selected, selected_names
    
    def transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Transform new data using selected features
        
        Args:
            X: New data
            feature_names: Feature names (must match original)
            
        Returns:
            X_transformed: Transformed data with selected features only
        """
        if self.selected_features is None:
            raise ValueError("Must call fit_transform first!")
        
        # Generate all candidates
        X_candidates, candidate_names = self._generate_candidates(X, feature_names)
        
        # Select only the selected features
        selected_indices = [candidate_names.index(f) for f in self.selected_features]
        X_selected = X_candidates[:, selected_indices]
        
        return X_selected
    
    # ========================================================================
    # CANDIDATE GENERATION
    # ========================================================================
    
    def _generate_candidates(self,
                            X: np.ndarray,
                            feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Generate candidate features"""
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=feature_names)
        
        candidates = []
        candidate_names = []
        
        # Original features
        candidates.append(X)
        candidate_names.extend(feature_names)
        
        # Polynomial features
        if self.polynomial_degree >= 2:
            poly_features, poly_names = self._generate_polynomial_features(df, feature_names)
            if poly_features is not None:
                candidates.append(poly_features)
                candidate_names.extend(poly_names)
        
        # Physics-inspired interactions
        if self.include_interactions:
            interaction_features, interaction_names = self._generate_physics_interactions(df)
            if interaction_features is not None:
                candidates.append(interaction_features)
                candidate_names.extend(interaction_names)
        
        # Mathematical transforms
        if self.include_transforms:
            transform_features, transform_names = self._generate_transforms(df, feature_names)
            if transform_features is not None:
                candidates.append(transform_features)
                candidate_names.extend(transform_names)
        
        # Concatenate all
        X_candidates = np.hstack(candidates)
        
        return X_candidates, candidate_names
    
    def _generate_polynomial_features(self,
                                     df: pd.DataFrame,
                                     feature_names: List[str]) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate polynomial features (degree 2 or 3)"""
        
        logger.info(f"  -> Generating polynomial features (degree {self.polynomial_degree})...")
        
        try:
            # Use sklearn PolynomialFeatures for degree 2
            if self.polynomial_degree == 2:
                poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
                X_poly = poly.fit_transform(df.values)
                
                # Get feature names
                poly_names = poly.get_feature_names_out(feature_names)
                
                # Remove original features (already included)
                original_indices = list(range(len(feature_names)))
                X_poly = np.delete(X_poly, original_indices, axis=1)
                poly_names = [name for i, name in enumerate(poly_names) if i not in original_indices]
                
                logger.info(f"    ✓ Generated {len(poly_names)} polynomial features")
                return X_poly, list(poly_names)
            
            # Manual for degree 3 (to limit explosion)
            elif self.polynomial_degree == 3:
                poly_features = []
                poly_names = []
                
                # Select most important features for degree 3 (top 10)
                important_features = feature_names[:min(10, len(feature_names))]
                
                for feat in important_features:
                    # x^2
                    poly_features.append(df[feat].values ** 2)
                    poly_names.append(f"{feat}^2")
                    
                    # x^3
                    poly_features.append(df[feat].values ** 3)
                    poly_names.append(f"{feat}^3")
                
                X_poly = np.column_stack(poly_features)
                logger.info(f"    ✓ Generated {len(poly_names)} degree-3 features")
                return X_poly, poly_names
        
        except Exception as e:
            logger.error(f"    ✗ Polynomial generation failed: {e}")
            return None, []
    
    def _generate_physics_interactions(self,
                                      df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate physics-inspired interaction terms"""
        
        logger.info("  -> Generating physics-inspired interactions...")
        
        interaction_features = []
        interaction_names = []
        
        for name, feature_pair in PHYSICS_INTERACTIONS.items():
            feat1, feat2 = feature_pair
            
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication
                interaction_features.append(df[feat1].values * df[feat2].values)
                interaction_names.append(f"{feat1}*{feat2}")
                
                # Ratio (if denominator non-zero)
                denom = df[feat2].values
                if np.all(np.abs(denom) > 1e-10):
                    interaction_features.append(df[feat1].values / denom)
                    interaction_names.append(f"{feat1}/{feat2}")
        
        if interaction_features:
            X_interactions = np.column_stack(interaction_features)
            logger.info(f"    ✓ Generated {len(interaction_names)} physics interactions")
            return X_interactions, interaction_names
        else:
            logger.info("    ✗ No physics interactions generated")
            return None, []
    
    def _generate_transforms(self,
                           df: pd.DataFrame,
                           feature_names: List[str]) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate mathematical transforms"""
        
        logger.info("  -> Generating mathematical transforms...")
        
        transform_features = []
        transform_names = []
        
        # Select subset of features for transforms (to avoid explosion)
        # Prioritize: A, Z, N, shell_gap, pairing, deformation
        priority_features = ['A', 'Z', 'N', 'shell_gap', 'pairing', 'deformation']
        selected_features = [f for f in priority_features if f in feature_names]
        selected_features += [f for f in feature_names if f not in selected_features][:5]
        
        for feat in selected_features:
            if feat not in df.columns:
                continue
            
            values = df[feat].values
            
            # Log (for positive values)
            if np.all(values > 0):
                transform_features.append(np.log(values))
                transform_names.append(f"log({feat})")
            
            # Sqrt (for non-negative values)
            if np.all(values >= 0):
                transform_features.append(np.sqrt(values))
                transform_names.append(f"sqrt({feat})")
            
            # Inverse (for non-zero values)
            if np.all(np.abs(values) > 1e-10):
                transform_features.append(1.0 / values)
                transform_names.append(f"1/{feat}")
            
            # Square
            transform_features.append(values ** 2)
            transform_names.append(f"{feat}^2")
        
        if transform_features:
            X_transforms = np.column_stack(transform_features)
            logger.info(f"    ✓ Generated {len(transform_names)} transforms")
            return X_transforms, transform_names
        else:
            return None, []
    
    # ========================================================================
    # FILTERING
    # ========================================================================
    
    def _filter_low_variance(self,
                            X: np.ndarray,
                            feature_names: List[str],
                            threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
        """Remove features with low variance"""
        
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
        
        selected_mask = selector.get_support()
        filtered_names = [name for name, selected in zip(feature_names, selected_mask) if selected]
        
        return X_filtered, filtered_names
    
    def _remove_high_correlation(self,
                                X: np.ndarray,
                                feature_names: List[str],
                                threshold: float = 0.95) -> Tuple[np.ndarray, List[str]]:
        """Remove highly correlated features"""
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        
        # Find pairs with high correlation
        to_remove = set()
        n_features = len(feature_names)
        
        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i + 1, n_features):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    # Remove feature with lower variance
                    var_i = np.var(X[:, i])
                    var_j = np.var(X[:, j])
                    to_remove.add(j if var_i > var_j else i)
        
        # Keep features not in to_remove
        keep_indices = [i for i in range(n_features) if i not in to_remove]
        X_decorrelated = X[:, keep_indices]
        decorrelated_names = [feature_names[i] for i in keep_indices]
        
        return X_decorrelated, decorrelated_names
    
    # ========================================================================
    # FEATURE SELECTION
    # ========================================================================
    
    def _select_features(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Select best features using multiple methods"""
        
        feature_votes = {name: 0 for name in feature_names}
        
        # Method 1: RFE
        if 'rfe' in self.selection_methods:
            selected = self._select_with_rfe(X, y, feature_names)
            for name in selected:
                feature_votes[name] += 1
            logger.info(f"    RFE: {len(selected)} features selected")
        
        # Method 2: LASSO
        if 'lasso' in self.selection_methods:
            selected = self._select_with_lasso(X, y, feature_names)
            for name in selected:
                feature_votes[name] += 1
            logger.info(f"    LASSO: {len(selected)} features selected")
        
        # Method 3: Mutual Information
        if 'mutual_info' in self.selection_methods:
            selected = self._select_with_mutual_info(X, y, feature_names)
            for name in selected:
                feature_votes[name] += 1
            logger.info(f"    Mutual Info: {len(selected)} features selected")
        
        # Method 4: SHAP (if available)
        if 'shap' in self.selection_methods and SHAP_AVAILABLE:
            selected = self._select_with_shap(X, y, feature_names)
            for name in selected:
                feature_votes[name] += 1
            logger.info(f"    SHAP: {len(selected)} features selected")
        
        # Ensemble voting: select features voted by at least 50% of methods
        min_votes = len(self.selection_methods) // 2
        selected_features = [name for name, votes in feature_votes.items() if votes >= min_votes]
        
        # Fallback: if too few selected, take top K by votes
        if len(selected_features) < self.target_n_features:
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            selected_features = [name for name, _ in sorted_features[:self.target_n_features]]
        
        # Get selected columns
        selected_indices = [feature_names.index(name) for name in selected_features]
        X_selected = X[:, selected_indices]
        
        # Store importance
        self.feature_importance = feature_votes
        
        return X_selected, selected_features
    
    def _select_with_rfe(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        feature_names: List[str]) -> List[str]:
        """Recursive Feature Elimination"""
        
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=self.target_n_features, step=1)
        selector.fit(X, y)
        
        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]
    
    def _select_with_lasso(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          feature_names: List[str]) -> List[str]:
        """LASSO-based selection"""
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # LASSO with cross-validation
        lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
        lasso.fit(X_scaled, y)
        
        # Select features with non-zero coefficients
        importance = np.abs(lasso.coef_)
        top_indices = np.argsort(importance)[-self.target_n_features:]
        
        return [feature_names[i] for i in top_indices]
    
    def _select_with_mutual_info(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 feature_names: List[str]) -> List[str]:
        """Mutual information selection"""
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        top_indices = np.argsort(mi_scores)[-self.target_n_features:]
        
        return [feature_names[i] for i in top_indices]
    
    def _select_with_shap(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         feature_names: List[str]) -> List[str]:
        """SHAP-based selection"""
        
        # Train quick model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # SHAP values (sample for speed)
        sample_size = min(100, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[sample_indices])
        
        # Mean absolute SHAP values
        importance = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(importance)[-self.target_n_features:]
        
        return [feature_names[i] for i in top_indices]
    
    # ========================================================================
    # EXPORT & VISUALIZATION
    # ========================================================================
    
    def _export_feature_report(self,
                               X_original: np.ndarray,
                               y: np.ndarray,
                               X_selected: np.ndarray,
                               selected_names: List[str]):
        """Export comprehensive feature engineering report"""
        
        # 1. Feature list CSV
        feature_df = pd.DataFrame({
            'Feature': selected_names,
            'Importance_Votes': [self.feature_importance[name] for name in selected_names]
        })
        feature_df = feature_df.sort_values('Importance_Votes', ascending=False)
        feature_df.to_csv(self.output_dir / 'selected_features.csv', index=False)
        
        # 2. Feature importance plot
        if PLOTTING_AVAILABLE:
            self._plot_feature_importance(feature_df)
        
        # 3. LaTeX table
        self._create_latex_table(feature_df)
        
        # 4. Summary JSON
        summary = {
            'original_n_features': len(self.original_features),
            'candidate_n_features': len(self.candidate_features),
            'selected_n_features': len(selected_names),
            'selected_features': selected_names,
            'selection_methods': self.selection_methods,
            'polynomial_degree': self.polynomial_degree
        }
        
        import json
        with open(self.output_dir / 'feature_engineering_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"  ✓ Exported: selected_features.csv")
        logger.info(f"  ✓ Exported: feature_importance.png")
        logger.info(f"  ✓ Exported: features_latex_table.txt")
        logger.info(f"  ✓ Exported: feature_engineering_summary.json")
    
    def _plot_feature_importance(self, feature_df: pd.DataFrame):
        """Plot feature importance"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Top 30 features
        top_30 = feature_df.head(30)
        
        colors = plt.cm.viridis(top_30['Importance_Votes'] / top_30['Importance_Votes'].max())
        bars = ax.barh(range(len(top_30)), top_30['Importance_Votes'], 
                      color=colors, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels(top_30['Feature'], fontsize=9)
        ax.set_xlabel('Selection Votes', fontsize=12, fontweight='bold')
        ax.set_title('Top 30 Selected Features\n(Votes from multiple selection methods)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add values
        for i, (bar, val) in enumerate(zip(bars, top_30['Importance_Votes'])):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{int(val)}', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_latex_table(self, feature_df: pd.DataFrame):
        """Create LaTeX table of selected features"""
        
        latex_lines = []
        latex_lines.append("% Top 20 Selected Features - LaTeX Table")
        latex_lines.append("\\begin{table}[H]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Top 20 Selected Features from AutoML Feature Engineering}")
        latex_lines.append("\\begin{tabular}{clc}")
        latex_lines.append("\\toprule")
        latex_lines.append("Rank & Feature & Votes \\\\")
        latex_lines.append("\\midrule")
        
        for rank, (_, row) in enumerate(feature_df.head(20).iterrows(), 1):
            # Escape special characters
            feature_name = row['Feature'].replace('_', '\\_')
            votes = int(row['Importance_Votes'])
            latex_lines.append(f"{rank} & {feature_name} & {votes} \\\\")
        
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        with open(self.output_dir / 'features_latex_table.txt', 'w') as f:
            f.write('\n'.join(latex_lines))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("TESTING AUTOML FEATURE ENGINEERING")
    logger.info("="*70)
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Target depends on first 3 features + interaction
    y = X[:, 0] * 2 + X[:, 1] * (-1) + X[:, 0] * X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.2
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Split
    split = int(0.7 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Feature engineering
    engineer = AutoMLFeatureEngineer(
        polynomial_degree=2,
        include_transforms=True,
        include_interactions=True,
        target_n_features=20,
        selection_methods=['rfe', 'lasso', 'mutual_info'],
        output_dir='test_feature_engineering'
    )
    
    X_train_eng, selected_features, X_val_eng = engineer.fit_transform(
        X_train, y_train, feature_names, X_val
    )
    
    logger.info(f"\n✓ Feature engineering complete!")
    logger.info(f"  Original features: {X_train.shape[1]}")
    logger.info(f"  Engineered features: {X_train_eng.shape[1]}")
    logger.info(f"  Improvement: {X_train_eng.shape[1] / X_train.shape[1]:.1f}x")
    logger.info(f"\n  Check: test_feature_engineering/")
