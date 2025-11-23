"""
Feature Combination Manager
============================

Önceden tanımlı feature setlerini yöneten sistem.
Dokümantasyonda belirtilen feature kombinasyonlarını sağlar.

Feature Sets:
- Basic: 6 features (A, Z, N, SPIN, PARITY, P_FACTOR)
- Extended: 21 features (Basic + Physics features)
- Full: 44 features (All available features)
- ANFIS_Compact: 5 features (Optimized for ANFIS, 32 rules)
- ANFIS_Standard: 8 features (256 rules)
- ANFIS_Extended: 10 features (1024 rules)

Author: Nuclear Physics AI Project
Version: 1.0.0
Date: 2025-11-23
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureCombinationManager:
    """
    Feature kombinasyonlarını yöneten sınıf.

    Dokümantasyonda tanımlanan feature setlerini sağlar ve yönetir.
    """

    # Feature Set Definitions
    FEATURE_SETS = {
        'Basic': {
            'features': ['A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR'],
            'count': 6,
            'category': 'Standard',
            'description': 'Basic nuclear properties',
            'recommended_for': 'Quick tests, initial experiments',
            'complexity': 'Simple'
        },

        'Basic_SEMF': {
            'features': [
                'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
                'BE_volume', 'BE_surface', 'BE_coulomb', 'BE_asymmetry',
                'BE_pairing', 'BE_total'
            ],
            'count': 12,
            'category': 'Standard',
            'description': 'Basic + SEMF calculations',
            'recommended_for': 'SEMF-based analysis',
            'complexity': 'Standard'
        },

        'Basic_Shell': {
            'features': [
                'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
                'shell_closure_Z', 'shell_closure_N', 'magic_character',
                'shell_gap_Z', 'shell_gap_N', 'valence_nucleons'
            ],
            'count': 12,
            'category': 'Standard',
            'description': 'Basic + Shell model features',
            'recommended_for': 'Shell model analysis',
            'complexity': 'Standard'
        },

        'Physics_Optimized': {
            'features': [
                'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
                'BE_per_A', 'S_n_approx', 'S_p_approx',
                'magic_character', 'shell_gap_Z', 'shell_gap_N'
            ],
            'count': 12,
            'category': 'Standard',
            'description': 'Best 12 physics features (curated)',
            'recommended_for': 'Optimal performance',
            'complexity': 'Standard'
        },

        'Extended': {
            'features': [
                # Basic
                'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
                # SEMF
                'BE_volume', 'BE_surface', 'BE_coulomb', 'BE_asymmetry', 'BE_pairing', 'BE_total', 'BE_per_A',
                # Shell
                'shell_closure_Z', 'shell_closure_N', 'magic_character', 'shell_gap_Z', 'shell_gap_N',
                # Additional
                'nuclear_radius', 'S_n_approx', 'S_p_approx'
            ],
            'count': 21,
            'category': 'Advanced',
            'description': 'Extended physics features',
            'recommended_for': 'Deep analysis, advanced models',
            'complexity': 'Advanced'
        },

        'Full': {
            'features': 'ALL',  # Special marker for all features
            'count': 44,  # Will be determined at runtime
            'category': 'Advanced',
            'description': 'All available features',
            'recommended_for': 'Maximum information, ensemble models',
            'complexity': 'Advanced'
        },

        'ANFIS_Compact': {
            'features': ['A', 'Z', 'SPIN', 'magic_character', 'BE_per_A'],
            'count': 5,
            'category': 'ANFIS',
            'description': 'Optimized for ANFIS with 2 MFs per input (32 rules)',
            'recommended_for': 'ANFIS training, interpretable models',
            'complexity': 'ANFIS',
            'anfis_rules': 32,
            'anfis_mf_count': 2
        },

        'ANFIS_Standard': {
            'features': [
                'A', 'Z', 'N', 'SPIN', 'PARITY',
                'magic_character', 'BE_per_A', 'shell_gap_Z'
            ],
            'count': 8,
            'category': 'ANFIS',
            'description': 'Optimized for ANFIS with 2 MFs per input (256 rules)',
            'recommended_for': 'ANFIS training, moderate complexity',
            'complexity': 'ANFIS',
            'anfis_rules': 256,
            'anfis_mf_count': 2
        },

        'ANFIS_Extended': {
            'features': [
                'A', 'Z', 'N', 'SPIN', 'PARITY',
                'magic_character', 'BE_per_A', 'shell_gap_Z', 'shell_gap_N', 'S_n_approx'
            ],
            'count': 10,
            'category': 'ANFIS',
            'description': 'Maximum for ANFIS with 2 MFs per input (1024 rules)',
            'recommended_for': 'ANFIS training, maximum complexity',
            'complexity': 'ANFIS',
            'anfis_rules': 1024,
            'anfis_mf_count': 2
        }
    }

    def __init__(self):
        """Initialize Feature Combination Manager"""
        logger.info("Feature Combination Manager initialized")
        logger.info(f"Available feature sets: {list(self.FEATURE_SETS.keys())}")

    def get_feature_set(self,
                       set_name: str,
                       available_columns: List[str],
                       target_cols: List[str]) -> List[str]:
        """
        Belirtilen feature set için feature listesini döndür

        Args:
            set_name: Feature set adı (Basic, Extended, Full, etc.)
            available_columns: DataFrame'de mevcut sütunlar
            target_cols: Target sütunları (exclude edilecek)

        Returns:
            Feature listesi
        """
        if set_name not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {set_name}. Available: {list(self.FEATURE_SETS.keys())}")

        feature_def = self.FEATURE_SETS[set_name]

        # Special case: 'Full' means all available features
        if feature_def['features'] == 'ALL':
            # All columns except targets and NUCLEUS
            features = [col for col in available_columns
                       if col not in target_cols and col != 'NUCLEUS']
            logger.info(f"Feature set '{set_name}': {len(features)} features (all available)")
            return features

        # Normal case: use predefined feature list
        requested_features = feature_def['features']

        # Filter to only include features that exist in the dataframe
        available_features = [f for f in requested_features if f in available_columns]
        missing_features = [f for f in requested_features if f not in available_columns]

        if missing_features:
            logger.warning(f"Feature set '{set_name}': {len(missing_features)} features not available: {missing_features}")

        if not available_features:
            raise ValueError(f"Feature set '{set_name}': No features available in dataframe!")

        logger.info(f"Feature set '{set_name}': {len(available_features)}/{len(requested_features)} features available")

        return available_features

    def get_feature_set_info(self, set_name: str) -> Dict:
        """
        Feature set hakkında bilgi döndür

        Args:
            set_name: Feature set adı

        Returns:
            Feature set metadata
        """
        if set_name not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {set_name}")

        return self.FEATURE_SETS[set_name].copy()

    def list_feature_sets(self, category: Optional[str] = None) -> List[str]:
        """
        Mevcut feature setlerini listele

        Args:
            category: Filter by category (Standard, Advanced, ANFIS)

        Returns:
            Feature set isimleri listesi
        """
        if category is None:
            return list(self.FEATURE_SETS.keys())

        return [name for name, info in self.FEATURE_SETS.items()
                if info['category'] == category]

    def get_recommended_sets_for_target(self, target: str) -> List[str]:
        """
        Target'a göre önerilen feature setlerini döndür

        Args:
            target: Target name (MM, QM, MM_QM, Beta_2)

        Returns:
            Önerilen feature set isimleri
        """
        # Target-specific recommendations
        recommendations = {
            'MM': ['Basic', 'Physics_Optimized', 'Extended', 'Full'],
            'QM': ['Basic', 'Physics_Optimized', 'Extended', 'Full'],
            'MM_QM': ['Extended', 'Full'],  # Dual target needs more features
            'Beta_2': ['Basic_Shell', 'Physics_Optimized', 'Extended', 'Full']
        }

        return recommendations.get(target, ['Basic', 'Extended', 'Full'])

    def save_feature_combinations_json(self, output_path: str):
        """
        Feature kombinasyonlarını JSON dosyasına kaydet

        Args:
            output_path: JSON dosya yolu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON
        export_data = {
            'feature_sets': {},
            'metadata': {
                'version': '1.0.0',
                'total_sets': len(self.FEATURE_SETS),
                'categories': list(set(info['category'] for info in self.FEATURE_SETS.values()))
            }
        }

        for name, info in self.FEATURE_SETS.items():
            export_data['feature_sets'][name] = {
                'features': info['features'] if info['features'] != 'ALL' else 'ALL_AVAILABLE',
                'count': info['count'],
                'category': info['category'],
                'description': info['description'],
                'recommended_for': info['recommended_for'],
                'complexity': info['complexity']
            }

            # Add ANFIS-specific info if exists
            if 'anfis_rules' in info:
                export_data['feature_sets'][name]['anfis_rules'] = info['anfis_rules']
                export_data['feature_sets'][name]['anfis_mf_count'] = info['anfis_mf_count']

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"[OK] Feature combinations saved to: {output_path}")

    def create_feature_set_summary_table(self) -> pd.DataFrame:
        """
        Feature setlerinin özet tablosunu oluştur

        Returns:
            DataFrame with feature set summary
        """
        data = []

        for name, info in self.FEATURE_SETS.items():
            row = {
                'Feature_Set': name,
                'Count': info['count'],
                'Category': info['category'],
                'Complexity': info['complexity'],
                'Description': info['description'],
                'Recommended_For': info['recommended_for']
            }

            if 'anfis_rules' in info:
                row['ANFIS_Rules'] = info['anfis_rules']
                row['ANFIS_MF_Count'] = info['anfis_mf_count']
            else:
                row['ANFIS_Rules'] = '-'
                row['ANFIS_MF_Count'] = '-'

            data.append(row)

        df = pd.DataFrame(data)

        # Sort by category and count
        category_order = {'Standard': 1, 'Advanced': 2, 'ANFIS': 3}
        df['_sort'] = df['Category'].map(category_order)
        df = df.sort_values(['_sort', 'Count']).drop('_sort', axis=1)

        return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_feature_sets() -> List[str]:
    """
    Varsayılan feature setlerini döndür (Faz 1 için)

    Returns:
        ['Basic', 'Extended', 'Full']
    """
    return ['Basic', 'Extended', 'Full']


def get_anfis_feature_sets() -> List[str]:
    """
    ANFIS için optimize edilmiş feature setlerini döndür

    Returns:
        ['ANFIS_Compact', 'ANFIS_Standard', 'ANFIS_Extended']
    """
    return ['ANFIS_Compact', 'ANFIS_Standard', 'ANFIS_Extended']


def get_all_standard_feature_sets() -> List[str]:
    """
    Standart (≤4 input) feature setlerini döndür

    Returns:
        ['Basic', 'Basic_SEMF', 'Basic_Shell', 'Physics_Optimized']
    """
    return ['Basic', 'Basic_SEMF', 'Basic_Shell', 'Physics_Optimized']


# ============================================================================
# MAIN TEST
# ============================================================================

def test_feature_combination_manager():
    """Test Feature Combination Manager"""

    print("\n" + "="*80)
    print("FEATURE COMBINATION MANAGER TEST")
    print("="*80)

    # Initialize manager
    manager = FeatureCombinationManager()

    # Test 1: List all feature sets
    print("\n-> Test 1: List all feature sets")
    all_sets = manager.list_feature_sets()
    print(f"  Total feature sets: {len(all_sets)}")
    print(f"  Names: {all_sets}")

    # Test 2: List by category
    print("\n-> Test 2: List by category")
    for category in ['Standard', 'Advanced', 'ANFIS']:
        sets = manager.list_feature_sets(category=category)
        print(f"  {category}: {sets}")

    # Test 3: Get feature set info
    print("\n-> Test 3: Get feature set info")
    for set_name in ['Basic', 'Extended', 'ANFIS_Compact']:
        info = manager.get_feature_set_info(set_name)
        print(f"  {set_name}:")
        print(f"    Features: {info['count']}")
        print(f"    Category: {info['category']}")
        print(f"    Description: {info['description']}")

    # Test 4: Get features for a set
    print("\n-> Test 4: Get features for a set")
    # Simulate available columns
    available_cols = [
        'NUCLEUS', 'A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR',
        'BE_volume', 'BE_surface', 'BE_coulomb', 'BE_asymmetry', 'BE_pairing', 'BE_total', 'BE_per_A',
        'shell_closure_Z', 'shell_closure_N', 'magic_character', 'shell_gap_Z', 'shell_gap_N',
        'nuclear_radius', 'S_n_approx', 'S_p_approx',
        'MM', 'Q'
    ]
    target_cols = ['MM']

    for set_name in ['Basic', 'Extended', 'Full']:
        features = manager.get_feature_set(set_name, available_cols, target_cols)
        print(f"  {set_name}: {len(features)} features")
        if len(features) <= 10:
            print(f"    {features}")

    # Test 5: Create summary table
    print("\n-> Test 5: Create summary table")
    summary_df = manager.create_feature_set_summary_table()
    print(summary_df.to_string(index=False))

    # Test 6: Save to JSON
    print("\n-> Test 6: Save to JSON")
    manager.save_feature_combinations_json('test_feature_combinations.json')

    # Test 7: Get recommendations
    print("\n-> Test 7: Get recommendations for targets")
    for target in ['MM', 'QM', 'MM_QM', 'Beta_2']:
        recommended = manager.get_recommended_sets_for_target(target)
        print(f"  {target}: {recommended}")

    print("\n[OK] Feature Combination Manager test completed!")


if __name__ == "__main__":
    test_feature_combination_manager()
