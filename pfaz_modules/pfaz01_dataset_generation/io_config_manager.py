"""
Input-Output Configuration Manager
===================================

Dataset input-output konfigürasyonlarını yöneten sistem.
Dokümantasyonda belirtilen I/O kombinasyonlarını sağlar.

Configurations:
- 2In1Out: 2 input → 1 output (minimal)
- 3In1Out: 3 input → 1 output (basic)
- 3In2Out: 3 input → 2 output (dual target)
- 4In1Out: 4 input → 1 output (standard)
- 5InAdv: 5-10 input → 1-2 output (advanced)
- 10InAdv: 10-20 input → 1-2 output (full features)
- 20InAdv: 20-44 input → 1-2 output (maximum info)

Author: Nuclear Physics AI Project
Version: 1.0.0
Date: 2025-11-23
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputOutputConfigManager:
    """
    Input-Output konfigürasyonlarını yöneten sınıf.

    Dokümantasyonda tanımlanan I/O kombinasyonlarını sağlar ve yönetir.
    """

    # I/O Configuration Definitions
    IO_CONFIGS = {
        '2In1Out': {
            'input_range': (2, 2),
            'output_count': 1,
            'category': 'Standard',
            'complexity': 'Simple',
            'recommended_targets': ['Beta_2'],
            'description': 'Minimal configuration (A, Z → Beta_2)',
            'anfis_feasible': True,
            'anfis_rules_2mf': 4,  # 2^2
            'anfis_rules_3mf': 9   # 3^2
        },

        '3In1Out': {
            'input_range': (3, 3),
            'output_count': 1,
            'category': 'Standard',
            'complexity': 'Standard',
            'recommended_targets': ['MM', 'QM', 'Beta_2'],
            'description': 'Basic configuration (A, Z, N → target)',
            'anfis_feasible': True,
            'anfis_rules_2mf': 8,   # 2^3
            'anfis_rules_3mf': 27   # 3^3
        },

        '3In2Out': {
            'input_range': (3, 3),
            'output_count': 2,
            'category': 'Standard',
            'complexity': 'Dual',
            'recommended_targets': ['MM_QM'],
            'description': 'Dual target (A, Z, N → MM, QM)',
            'anfis_feasible': True,
            'anfis_rules_2mf': 8,
            'anfis_rules_3mf': 27
        },

        '4In1Out': {
            'input_range': (4, 4),
            'output_count': 1,
            'category': 'Standard',
            'complexity': 'Advanced',
            'recommended_targets': ['Beta_2', 'MM', 'QM'],
            'description': 'Standard advanced (A, Z, N, SPIN → target)',
            'anfis_feasible': True,
            'anfis_rules_2mf': 16,  # 2^4
            'anfis_rules_3mf': 81   # 3^4
        },

        '5InAdv': {
            'input_range': (5, 10),
            'output_count': 1,  # or 2 for dual targets
            'category': 'Advanced',
            'complexity': 'Advanced',
            'recommended_targets': ['MM', 'QM', 'MM_QM', 'Beta_2'],
            'description': 'Advanced physics features (5-10 inputs)',
            'anfis_feasible': False,  # Too many rules
            'anfis_rules_2mf': None,
            'anfis_rules_3mf': None
        },

        '10InAdv': {
            'input_range': (10, 20),
            'output_count': 1,
            'category': 'Advanced',
            'complexity': 'Full',
            'recommended_targets': ['MM', 'QM', 'MM_QM', 'Beta_2'],
            'description': 'Full feature set (10-20 inputs)',
            'anfis_feasible': False,
            'anfis_rules_2mf': None,
            'anfis_rules_3mf': None
        },

        '20InAdv': {
            'input_range': (20, 44),
            'output_count': 1,
            'category': 'Advanced',
            'complexity': 'Maximum',
            'recommended_targets': ['MM', 'QM', 'MM_QM', 'Beta_2'],
            'description': 'Maximum information (20-44 inputs)',
            'anfis_feasible': False,
            'anfis_rules_2mf': None,
            'anfis_rules_3mf': None
        }
    }

    # Feature set to I/O config mapping
    # SHAP-bazli yeni setler + legacy setler
    FEATURE_SET_TO_CONFIG = {
        # --- Legacy sets ---
        'Basic': '3In1Out',
        'Basic_SEMF': '3In1Out',
        'Basic_Shell': '3In1Out',
        'Physics_Optimized': '3In1Out',
        'ANFIS_Compact': '3In1Out',
        'ANFIS_Standard': '3In1Out',
        'ANFIS_Extended': '4In1Out',
        'Extended': 'AUTO',
        'Full': 'AUTO',

        # --- Common 3-input SHAP sets ---
        'AZN':    '3In1Out',
        'AZS':    '3In1Out',
        'AZMC':   '3In1Out',
        'AZBEPA': '3In1Out',
        'AZB2E':  '3In1Out',

        # --- Common 4-input SHAP sets ---
        'AZNS':   '4In1Out',

        # --- MM 3-input ---
        'ASMC':    '3In1Out',
        'AMCBEPA': '3In1Out',

        # --- MM 4-input ---
        'AZSMC':    '4In1Out',
        'AZSBEPA':  '4In1Out',
        'AZMCBEPA': '4In1Out',
        'AZSB2E':   '4In1Out',

        # --- MM 5-input ---
        'AZSMCBEPA': '5InAdv',
        'AZSMCB2E':  '5InAdv',

        # --- QM 3-input ---
        'ZB2EMC':   '3In1Out',
        'B2EMCBEA': '3In1Out',

        # --- QM 4-input ---
        'AZB2EMC':  '4In1Out',
        'ZB2EMCS':  '4In1Out',
        'AZB2EBEA': '4In1Out',

        # --- QM 5-input ---
        'AZB2EMCS':   '5InAdv',
        'AZB2EMCBEA': '5InAdv',

        # --- Beta_2 3-input ---
        'MCZMNM':  '3In1Out',
        'AZVNV':   '3In1Out',
        'ZMNMBEA': '3In1Out',

        # --- Beta_2 4-input ---
        'MCZMNMZV':  '4In1Out',
        'MCZMNMBEA': '4In1Out',
        'AMCZMNM':   '4In1Out',
        'ZVNVZMNM':  '4In1Out',

        # --- Beta_2 5-input ---
        'MCZMNMZVNV': '5InAdv',
        'AMCZMNMBEA': '5InAdv',

        # --- NnNp sets (raw aaa2.txt valence columns) ---
        'NNPMC':   '3In1Out',   # 3-input
        'AZNNP':   '4In1Out',   # 4-input
        'ZNNPMC':  '4In1Out',   # 4-input
        'AZNNPMC': '5InAdv',    # 5-input
        'AZSNNNP': '5InAdv',    # 5-input
    }

    def __init__(self):
        """Initialize I/O Config Manager"""
        logger.info("Input-Output Configuration Manager initialized")
        logger.info(f"Available I/O configs: {list(self.IO_CONFIGS.keys())}")

    def get_config_for_feature_set(self,
                                   feature_set_name: str,
                                   n_features: int,
                                   target: str) -> str:
        """
        Feature set ve feature sayısına göre uygun I/O config döndür

        Args:
            feature_set_name: Feature set adı
            n_features: Feature sayısı
            target: Target adı (MM, QM, MM_QM, Beta_2)

        Returns:
            I/O config adı (örn: '3In1Out', '5InAdv')
        """
        # Check if feature set has a fixed mapping
        if feature_set_name in self.FEATURE_SET_TO_CONFIG:
            config = self.FEATURE_SET_TO_CONFIG[feature_set_name]

            # If AUTO, determine based on feature count
            if config == 'AUTO':
                return self._auto_detect_config(n_features, target)
            else:
                return config

        # If not mapped, auto-detect based on feature count
        return self._auto_detect_config(n_features, target)

    def _auto_detect_config(self, n_features: int, target: str) -> str:
        """
        Feature sayısına göre otomatik I/O config belirle

        Args:
            n_features: Feature sayısı
            target: Target adı

        Returns:
            I/O config adı
        """
        # Dual target için özel durum
        if target == 'MM_QM':
            if n_features <= 4:
                return '3In2Out'
            else:
                return '5InAdv'  # Advanced dual target

        # Single target
        if n_features <= 2:
            return '2In1Out'
        elif n_features <= 3:
            return '3In1Out'
        elif n_features <= 4:
            return '4In1Out'
        elif n_features <= 10:
            return '5InAdv'
        elif n_features <= 20:
            return '10InAdv'
        else:
            return '20InAdv'

    def get_config_info(self, config_name: str) -> Dict:
        """
        I/O config hakkında bilgi döndür

        Args:
            config_name: I/O config adı

        Returns:
            Config metadata
        """
        if config_name not in self.IO_CONFIGS:
            raise ValueError(f"Unknown I/O config: {config_name}")

        return self.IO_CONFIGS[config_name].copy()

    def list_configs(self, category: Optional[str] = None) -> List[str]:
        """
        Mevcut I/O config'leri listele

        Args:
            category: Filter by category (Standard, Advanced)

        Returns:
            I/O config isimleri listesi
        """
        if category is None:
            return list(self.IO_CONFIGS.keys())

        return [name for name, info in self.IO_CONFIGS.items()
                if info['category'] == category]

    def is_anfis_feasible(self, config_name: str) -> bool:
        """
        ANFIS için uygun mu kontrol et

        Args:
            config_name: I/O config adı

        Returns:
            True if ANFIS feasible
        """
        if config_name not in self.IO_CONFIGS:
            return False

        return self.IO_CONFIGS[config_name]['anfis_feasible']

    def get_anfis_rule_count(self, config_name: str, mf_count: int = 2) -> Optional[int]:
        """
        ANFIS için kural sayısını döndür

        Args:
            config_name: I/O config adı
            mf_count: Membership function count per input (2 or 3)

        Returns:
            Rule count (None if not ANFIS feasible)
        """
        if config_name not in self.IO_CONFIGS:
            return None

        config = self.IO_CONFIGS[config_name]

        if not config['anfis_feasible']:
            return None

        if mf_count == 2:
            return config['anfis_rules_2mf']
        elif mf_count == 3:
            return config['anfis_rules_3mf']
        else:
            return None

    def save_io_configs_json(self, output_path: str):
        """
        I/O konfigürasyonlarını JSON dosyasına kaydet

        Args:
            output_path: JSON dosya yolu
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            'io_configs': self.IO_CONFIGS,
            'feature_set_mappings': self.FEATURE_SET_TO_CONFIG,
            'metadata': {
                'version': '1.0.0',
                'total_configs': len(self.IO_CONFIGS),
                'anfis_feasible_configs': sum(1 for c in self.IO_CONFIGS.values() if c['anfis_feasible'])
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"[OK] I/O configurations saved to: {output_path}")


# ============================================================================
# SCENARIO SYSTEM
# ============================================================================

class ScenarioManager:
    """
    Dataset split scenario'larını yöneten sınıf.

    Scenarios:
    - S70: 70% train, 15% val, 15% test
    - S80: 80% train, 10% val, 10% test
    """

    SCENARIOS = {
        'S70': {
            'train': 0.70,
            'val': 0.15,
            'test': 0.15,
            'description': 'Standard split (70/15/15)',
            'recommended_for': 'Balanced validation'
        },
        'S80': {
            'train': 0.80,
            'val': 0.10,
            'test': 0.10,
            'description': 'More training data (80/10/10)',
            'recommended_for': 'Small datasets, need more training data'
        }
    }

    def __init__(self):
        """Initialize Scenario Manager"""
        logger.info("Scenario Manager initialized")
        logger.info(f"Available scenarios: {list(self.SCENARIOS.keys())}")

    def get_split_ratios(self, scenario: str) -> Tuple[float, float, float]:
        """
        Scenario'ya göre split ratios döndür

        Args:
            scenario: Scenario adı (S70, S80)

        Returns:
            (train_ratio, val_ratio, test_ratio)
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self.SCENARIOS.keys())}")

        s = self.SCENARIOS[scenario]
        return (s['train'], s['val'], s['test'])

    def get_scenario_info(self, scenario: str) -> Dict:
        """
        Scenario bilgisi döndür

        Args:
            scenario: Scenario adı

        Returns:
            Scenario metadata
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")

        return self.SCENARIOS[scenario].copy()

    def calculate_split_counts(self,
                               total_count: int,
                               scenario: str) -> Tuple[int, int, int]:
        """
        Total sayıya göre train/val/test sayılarını hesapla

        Args:
            total_count: Toplam örnek sayısı
            scenario: Scenario adı

        Returns:
            (n_train, n_val, n_test)
        """
        train_ratio, val_ratio, test_ratio = self.get_split_ratios(scenario)

        n_train = int(total_count * train_ratio)
        n_val = int(total_count * val_ratio)
        n_test = total_count - n_train - n_val  # Remaining goes to test

        return (n_train, n_val, n_test)


# ============================================================================
# MAIN TEST
# ============================================================================

def test_io_config_manager():
    """Test I/O Config Manager"""

    print("\n" + "="*80)
    print("INPUT-OUTPUT CONFIGURATION MANAGER TEST")
    print("="*80)

    # Initialize manager
    io_manager = InputOutputConfigManager()

    # Test 1: List all configs
    print("\n-> Test 1: List all I/O configs")
    all_configs = io_manager.list_configs()
    print(f"  Total configs: {len(all_configs)}")
    print(f"  Names: {all_configs}")

    # Test 2: List by category
    print("\n-> Test 2: List by category")
    for category in ['Standard', 'Advanced']:
        configs = io_manager.list_configs(category=category)
        print(f"  {category}: {configs}")

    # Test 3: Get config for feature sets
    print("\n-> Test 3: Get config for feature sets")
    test_cases = [
        ('Basic', 6, 'MM'),
        ('Extended', 21, 'MM'),
        ('Full', 44, 'QM'),
        ('ANFIS_Compact', 5, 'MM')
    ]

    for feature_set, n_features, target in test_cases:
        config = io_manager.get_config_for_feature_set(feature_set, n_features, target)
        print(f"  {feature_set} ({n_features} features, {target}) → {config}")

    # Test 4: ANFIS feasibility
    print("\n-> Test 4: ANFIS feasibility")
    for config in ['3In1Out', '4In1Out', '5InAdv', '10InAdv']:
        feasible = io_manager.is_anfis_feasible(config)
        if feasible:
            rules_2mf = io_manager.get_anfis_rule_count(config, 2)
            rules_3mf = io_manager.get_anfis_rule_count(config, 3)
            print(f"  {config}: ANFIS ✓ (2MF={rules_2mf} rules, 3MF={rules_3mf} rules)")
        else:
            print(f"  {config}: ANFIS ✗ (too many inputs)")

    # Test 5: Scenario Manager
    print("\n-> Test 5: Scenario Manager")
    scenario_manager = ScenarioManager()

    for scenario in ['S70', 'S80']:
        ratios = scenario_manager.get_split_ratios(scenario)
        counts = scenario_manager.calculate_split_counts(100, scenario)
        info = scenario_manager.get_scenario_info(scenario)

        print(f"  {scenario}:")
        print(f"    Ratios: {ratios}")
        print(f"    Counts (n=100): Train={counts[0]}, Val={counts[1]}, Test={counts[2]}")
        print(f"    Description: {info['description']}")

    # Test 6: Save to JSON
    print("\n-> Test 6: Save to JSON")
    io_manager.save_io_configs_json('test_io_configs.json')

    print("\n[OK] I/O Config Manager test completed!")


if __name__ == "__main__":
    test_io_config_manager()
