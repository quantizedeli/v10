#!/usr/bin/env python3
"""
Comprehensive Dataset Generation Script
Binlerce dataset oluşturmak için geliştirilmiş script

Bu script dataset_generation_config.yaml dosyasını okuyarak
dinamik olarak binlerce farklı feature kombinasyonuyla dataset üretir.

Kullanım:
    python generate_comprehensive_datasets.py [--config CONFIG_FILE] [--preset PRESET_NAME]

Örnekler:
    # Default config ile
    python generate_comprehensive_datasets.py

    # Custom config ile
    python generate_comprehensive_datasets.py --config my_config.yaml

    # Preset kullanarak
    python generate_comprehensive_datasets.py --preset quick_test

    # Dry-run (sadece kaç dataset oluşturulacağını göster)
    python generate_comprehensive_datasets.py --dry-run

Versiyon: 2.0.0
Tarih: 2025-11-23
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core_modules.constants import get_dynamic_feature_sets, TARGETS

# Optional import (only needed for actual generation)
try:
    from pfaz_modules.pfaz01_dataset_generation.dataset_generation_pipeline_v2 import DatasetGenerationPipeline
except ImportError:
    DatasetGenerationPipeline = None


def load_config(config_file='dataset_generation_config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_file}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return None


def apply_preset(config, preset_name):
    """Apply a preset configuration"""
    if 'presets' not in config or preset_name not in config['presets']:
        logging.warning(f"Preset '{preset_name}' not found in config")
        return config

    preset = config['presets'][preset_name]

    # Apply preset settings
    for key_path, value in preset.items():
        keys = key_path.split('.')
        current = config

        # Navigate to the nested key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value

    logging.info(f"Applied preset: {preset_name}")
    return config


def estimate_total_datasets(config):
    """
    Estimate total number of datasets that will be generated

    Returns:
        (total_count, breakdown_dict)
    """
    # Get feature sets count
    mode = config['feature_sets']['mode']
    max_sets = config['limits'].get('max_total_datasets', None)

    # Estimate feature sets
    if mode == 'standard':
        n_feature_sets = 9  # Legacy sets
    elif mode == 'extended':
        n_feature_sets = min(200, max_sets) if max_sets else 200
    elif mode == 'comprehensive':
        # Will be limited by max_total_datasets
        n_feature_sets = min(1000, max_sets) if max_sets else 1000
    elif mode == 'targeted':
        n_feature_sets = len(config['targets']['enabled']) * 5  # 5 levels per target

    # Calculate combinations
    n_targets = len(config['targets']['enabled'])
    n_nucleus_counts = len(config['dataset_params']['nucleus_counts'])
    n_scenarios = len(config['dataset_params']['scenarios'])
    n_anomaly = len(config['dataset_params']['anomaly_modes'])
    n_scaling = len(config['dataset_params']['scaling_methods'])
    n_sampling = len(config['dataset_params']['sampling_methods'])

    # Total = targets × nucleus_counts × scenarios × anomaly × features × scaling × sampling
    total_theoretical = (n_targets * n_nucleus_counts * n_scenarios *
                        n_anomaly * n_feature_sets * n_scaling * n_sampling)

    # Apply limit
    max_limit = config['limits'].get('max_total_datasets', None)
    total_actual = min(total_theoretical, max_limit) if max_limit else total_theoretical

    breakdown = {
        'targets': n_targets,
        'nucleus_counts': n_nucleus_counts,
        'scenarios': n_scenarios,
        'anomaly_modes': n_anomaly,
        'feature_sets': n_feature_sets,
        'scaling_methods': n_scaling,
        'sampling_methods': n_sampling,
        'theoretical_total': total_theoretical,
        'actual_total': total_actual,
        'limited_by_max': max_limit and total_actual < total_theoretical
    }

    return total_actual, breakdown


def generate_datasets(config, dry_run=False):
    """
    Generate datasets based on configuration

    Args:
        config: Configuration dictionary
        dry_run: If True, only show what would be generated
    """
    # Setup logging
    log_level = getattr(logging, config.get('general', {}).get('log_level', 'INFO'))
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("=" * 80)
    logging.info("COMPREHENSIVE DATASET GENERATION v2.0.0")
    logging.info("=" * 80)

    # Estimate total datasets
    total, breakdown = estimate_total_datasets(config)

    logging.info("\n📊 GENERATION PLAN:")
    logging.info(f"  Targets: {breakdown['targets']}")
    logging.info(f"  Nucleus counts: {breakdown['nucleus_counts']}")
    logging.info(f"  Scenarios: {breakdown['scenarios']}")
    logging.info(f"  Anomaly modes: {breakdown['anomaly_modes']}")
    logging.info(f"  Feature sets: {breakdown['feature_sets']}")
    logging.info(f"  Scaling methods: {breakdown['scaling_methods']}")
    logging.info(f"  Sampling methods: {breakdown['sampling_methods']}")
    logging.info(f"\n  THEORETICAL TOTAL: {breakdown['theoretical_total']:,} datasets")
    logging.info(f"  ACTUAL TOTAL (with limits): {breakdown['actual_total']:,} datasets")

    if breakdown['limited_by_max']:
        logging.warning(f"  ⚠️  Limited by max_total_datasets = {config['limits']['max_total_datasets']:,}")

    if dry_run:
        logging.info("\n🔍 DRY RUN MODE - No datasets will be generated")
        logging.info("=" * 80)
        return

    # Confirm with user
    logging.info(f"\n⏳ This will generate {breakdown['actual_total']:,} datasets")
    logging.info("   This may take a considerable amount of time and disk space.")

    # For actual generation, we would proceed here
    logging.info("\n✅ Configuration validated and ready for generation")
    logging.info("=" * 80)

    # TODO: Implement actual dataset generation loop
    # This would involve:
    # 1. For each target
    # 2.   Get feature sets for this target
    # 3.   For each (nucleus_count, scenario, anomaly, feature_set, scaling, sampling)
    # 4.     Generate dataset
    # 5.     Save to disk
    # 6.     Update catalog

    logging.warning("\n⚠️  Actual generation not yet implemented in this version")
    logging.info("Please use the existing DatasetGenerationPipeline with the new feature sets")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive nuclear physics datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default config
  python generate_comprehensive_datasets.py

  # Use a preset
  python generate_comprehensive_datasets.py --preset quick_test

  # Dry run to see what would be generated
  python generate_comprehensive_datasets.py --dry-run

  # Custom config file
  python generate_comprehensive_datasets.py --config my_config.yaml
        """
    )

    parser.add_argument(
        '--config',
        default='dataset_generation_config.yaml',
        help='Path to configuration file (default: dataset_generation_config.yaml)'
    )

    parser.add_argument(
        '--preset',
        choices=['quick_test', 'standard_research', 'comprehensive', 'target_focused'],
        help='Use a predefined preset'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without actually generating'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if config is None:
        sys.exit(1)

    # Apply preset if specified
    if args.preset:
        config = apply_preset(config, args.preset)
    elif 'active_preset' in config and config['active_preset']:
        config = apply_preset(config, config['active_preset'])

    # Set verbose
    if args.verbose:
        if 'general' not in config:
            config['general'] = {}
        config['general']['log_level'] = 'DEBUG'

    # Generate datasets
    generate_datasets(config, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
