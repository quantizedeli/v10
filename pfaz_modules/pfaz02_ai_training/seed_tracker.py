# -*- coding: utf-8 -*-
"""
Seed Tracker for Reproducibility
==================================

Her dataset ve model için kullanılan seed'leri izler ve Excel'e kaydeder.

Author: Nuclear Physics AI Project
Version: 1.0.0
Date: 2025-12-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeedTracker:
    """
    Random seed tracking system for reproducibility

    Tracks all random seeds used in:
    - Dataset generation (sampling)
    - Model training (RF, XGBoost, DNN)
    - Cross-validation splits
    - Data shuffling
    """

    def __init__(self, output_dir: str = 'seed_reports'):
        """
        Initialize seed tracker

        Args:
            output_dir: Directory to save seed reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seeds = []

        logger.info(f"SeedTracker initialized: {self.output_dir}")

    def add_seed(self,
                operation: str,
                seed: int,
                dataset_name: str = None,
                model_name: str = None,
                config_id: str = None,
                details: Dict = None):
        """
        Record a seed usage

        Args:
            operation: Type of operation (e.g., 'dataset_sampling', 'model_training', 'cv_split')
            seed: Random seed value
            dataset_name: Name of dataset
            model_name: Name of model
            config_id: Configuration ID
            details: Additional details
        """
        seed_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'seed': seed,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'config_id': config_id,
            'details': json.dumps(details) if details else '{}'
        }

        self.seeds.append(seed_entry)
        logger.debug(f"Seed recorded: {operation} | seed={seed} | dataset={dataset_name}")

    def get_summary(self) -> Dict:
        """
        Get summary statistics

        Returns:
            Dictionary with summary info
        """
        if not self.seeds:
            return {
                'total_seeds': 0,
                'by_operation': {},
                'by_dataset': {},
                'by_model': {}
            }

        df = pd.DataFrame(self.seeds)

        summary = {
            'total_seeds': len(df),
            'unique_seeds': df['seed'].nunique(),
            'by_operation': df.groupby('operation').size().to_dict(),
            'by_dataset': df.groupby('dataset_name').size().to_dict() if 'dataset_name' in df else {},
            'by_model': df.groupby('model_name').size().to_dict() if 'model_name' in df else {}
        }

        return summary

    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()

        logger.info("\n" + "=" * 80)
        logger.info("SEED TRACKER SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total seeds recorded: {summary['total_seeds']}")
        logger.info(f"Unique seeds: {summary['unique_seeds']}")

        if summary['total_seeds'] > 0:
            logger.info("\nBy Operation:")
            for op, count in sorted(summary['by_operation'].items()):
                logger.info(f"  {op}: {count}")

            if summary['by_dataset']:
                logger.info("\nBy Dataset:")
                for ds, count in sorted(summary['by_dataset'].items()):
                    if ds:
                        logger.info(f"  {ds}: {count}")

        logger.info("=" * 80)

    def save_to_excel(self, filename: str = 'seed_tracking_report.xlsx'):
        """
        Save seed tracking report to Excel

        Args:
            filename: Output filename
        """
        if not self.seeds:
            logger.warning("[WARNING] No seeds to save")
            return

        df = pd.DataFrame(self.seeds)
        output_path = self.output_dir / filename

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: All Seeds
            df_sorted = df.sort_values(['operation', 'dataset_name', 'timestamp'])
            df_sorted.to_excel(writer, sheet_name='All_Seeds', index=False)

            # Sheet 2: Summary by Operation
            op_summary = df.groupby('operation').agg({
                'seed': ['count', 'nunique', lambda x: list(x.unique())]
            }).reset_index()
            op_summary.columns = ['Operation', 'Count', 'Unique_Seeds', 'Seed_Values']
            op_summary.to_excel(writer, sheet_name='By_Operation', index=False)

            # Sheet 3: Summary by Dataset
            if 'dataset_name' in df.columns:
                ds_summary = df[df['dataset_name'].notna()].groupby('dataset_name').agg({
                    'seed': ['count', 'nunique', lambda x: list(x.unique())]
                }).reset_index()
                ds_summary.columns = ['Dataset', 'Count', 'Unique_Seeds', 'Seed_Values']
                ds_summary.to_excel(writer, sheet_name='By_Dataset', index=False)

            # Sheet 4: Summary by Model
            if 'model_name' in df.columns:
                model_summary = df[df['model_name'].notna()].groupby('model_name').agg({
                    'seed': ['count', 'nunique', lambda x: list(x.unique())]
                }).reset_index()
                model_summary.columns = ['Model', 'Count', 'Unique_Seeds', 'Seed_Values']
                model_summary.to_excel(writer, sheet_name='By_Model', index=False)

            # Sheet 5: Detailed by Operation
            for operation in df['operation'].unique():
                op_df = df[df['operation'] == operation].copy()
                sheet_name = f"Op_{operation}"[:31]  # Excel max 31 chars
                op_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"[OK] Seed tracking report saved: {output_path}")
        logger.info(f"     Total seeds: {len(df)}")
        logger.info(f"     Sheets created: {4 + len(df['operation'].unique())}")

    def save_to_json(self, filename: str = 'seed_tracking_report.json'):
        """
        Save seed tracking to JSON

        Args:
            filename: Output filename
        """
        if not self.seeds:
            logger.warning("[WARNING] No seeds to save")
            return

        output_path = self.output_dir / filename
        summary = self.get_summary()

        output_data = {
            'summary': summary,
            'seeds': self.seeds,
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"[OK] Seed tracking JSON saved: {output_path}")

    def clear(self):
        """Clear all seed records"""
        self.seeds = []
        logger.info("SeedTracker cleared")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_global_seed(seed: int, tracker: Optional[SeedTracker] = None):
    """
    Set global random seed for reproducibility

    Args:
        seed: Random seed value
        tracker: Optional SeedTracker to record this seed
    """
    # Set seeds for all libraries
    np.random.seed(seed)

    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
        random = None

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
        tf = None

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
        torch = None

    # Record in tracker
    if tracker:
        tracker.add_seed(
            operation='global_seed',
            seed=seed,
            details={'libraries': ['numpy', 'random', 'tensorflow', 'torch']}
        )

    logger.info(f"Global seed set to {seed}")


# ============================================================================
# MAIN TEST
# ============================================================================

def test_seed_tracker():
    """Test the seed tracker"""

    print("\n" + "=" * 80)
    print("SEED TRACKER TEST")
    print("=" * 80)

    tracker = SeedTracker(output_dir='test_seed_reports')

    # Simulate various seed usages
    tracker.add_seed(
        operation='dataset_sampling',
        seed=42,
        dataset_name='MM_75nuclei',
        details={'sampling_method': 'stratified', 'n_samples': 75}
    )

    tracker.add_seed(
        operation='model_training',
        seed=42,
        dataset_name='MM_75nuclei',
        model_name='RandomForest',
        config_id='RF_001',
        details={'n_estimators': 100, 'max_depth': 10}
    )

    tracker.add_seed(
        operation='cv_split',
        seed=42,
        dataset_name='MM_75nuclei',
        model_name='RandomForest',
        config_id='RF_001',
        details={'cv_folds': 5, 'stratified': True}
    )

    tracker.add_seed(
        operation='dataset_sampling',
        seed=42,
        dataset_name='Beta_2_100nuclei',
        details={'sampling_method': 'random', 'n_samples': 100}
    )

    tracker.add_seed(
        operation='model_training',
        seed=42,
        dataset_name='Beta_2_100nuclei',
        model_name='XGBoost',
        config_id='XGB_001',
        details={'n_estimators': 100, 'learning_rate': 0.1}
    )

    # Print summary
    tracker.print_summary()

    # Save reports
    tracker.save_to_excel('test_seed_tracking_report.xlsx')
    tracker.save_to_json('test_seed_tracking_report.json')

    # Test global seed setting
    set_global_seed(42, tracker)

    print("\n[OK] Seed tracker test completed!")


if __name__ == "__main__":
    test_seed_tracker()
