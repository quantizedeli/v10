# -*- coding: utf-8 -*-
"""
PFAZ 4: Unknown Nuclei Splitter
================================

Split datasets into known/unknown nuclei for generalization testing

Features:
- Stratified splitting by A, Z ranges
- Configurable split ratios
- Metadata generation
- Multiple dataset support

Author: Nuclear Physics AI Training Pipeline
Version: 1.0.0
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnknownNucleiSplitter:
    """
    Split datasets into known/unknown nuclei
    
    Features:
    - Stratified random split
    - Configurable ratios
    - Metadata generation
    """
    
    def __init__(self,
                 split_ratio: float = 0.7,
                 random_state: int = 42,
                 output_dir: str = 'unknown_nuclei_splits'):
        """
        Args:
            split_ratio: Ratio of known nuclei (0.7 = 70% known, 30% unknown)
            random_state: Random seed
            output_dir: Output directory
        """
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.split_metadata = {
            'split_ratio': split_ratio,
            'random_state': random_state,
            'datasets': []
        }
        
        logger.info("=" * 80)
        logger.info("UNKNOWN NUCLEI SPLITTER INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Split ratio: {split_ratio:.0%} known / {1-split_ratio:.0%} unknown")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)
    
    def split_dataset(self, dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split single dataset into known/unknown
        
        Returns:
            (known_df, unknown_df)
        """
        # Load dataset
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_excel(dataset_path)
        
        logger.info(f"\nSplitting: {dataset_path.name}")
        logger.info(f"Total nuclei: {len(df)}")
        
        # Stratification by A ranges for better distribution
        if 'A' in df.columns:
            df['A_bin'] = pd.cut(df['A'], bins=5, labels=False)
            stratify = df['A_bin']
        else:
            stratify = None
        
        # Split
        known_df, unknown_df = train_test_split(
            df,
            train_size=self.split_ratio,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Remove temporary column
        if 'A_bin' in known_df.columns:
            known_df = known_df.drop('A_bin', axis=1)
            unknown_df = unknown_df.drop('A_bin', axis=1)
        
        logger.info(f"Known nuclei: {len(known_df)} ({len(known_df)/len(df)*100:.1f}%)")
        logger.info(f"Unknown nuclei: {len(unknown_df)} ({len(unknown_df)/len(df)*100:.1f}%)")
        
        return known_df, unknown_df
    
    def split_all_datasets(self, datasets_dir: Path):
        """Split all datasets in directory"""
        
        datasets_dir = Path(datasets_dir)
        
        # Find all dataset files
        dataset_files = list(datasets_dir.glob('*.csv')) + list(datasets_dir.glob('*.xlsx'))
        dataset_files = [f for f in dataset_files if 'ALL' not in f.name]  # Skip ALL datasets
        
        logger.info(f"\nFound {len(dataset_files)} datasets to split")
        
        for dataset_file in dataset_files:
            try:
                known_df, unknown_df = self.split_dataset(dataset_file)
                
                # Save
                dataset_name = dataset_file.stem
                known_file = self.output_dir / f"{dataset_name}_known.csv"
                unknown_file = self.output_dir / f"{dataset_name}_unknown.csv"
                
                known_df.to_csv(known_file, index=False)
                unknown_df.to_csv(unknown_file, index=False)
                
                # Update metadata
                self.split_metadata['datasets'].append({
                    'name': dataset_name,
                    'total': len(known_df) + len(unknown_df),
                    'known': len(known_df),
                    'unknown': len(unknown_df),
                    'known_file': str(known_file),
                    'unknown_file': str(unknown_file)
                })
                
                logger.info(f"[SUCCESS] Saved: {known_file.name}, {unknown_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to split {dataset_file.name}: {e}")
        
        # Save metadata
        metadata_file = self.output_dir / 'split_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.split_metadata, f, indent=2)
        
        logger.info(f"\n[SUCCESS] Metadata saved: {metadata_file}")
        logger.info(f"[SUCCESS] Total datasets split: {len(self.split_metadata['datasets'])}")


def main():
    """Test execution"""
    
    print("\n" + "=" * 80)
    print("PFAZ 4: UNKNOWN NUCLEI SPLITTER - TEST")
    print("=" * 80)
    
    # Create test dataset
    test_data_dir = Path('test_datasets_for_split')
    test_data_dir.mkdir(exist_ok=True)
    
    # Generate test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'A': np.random.randint(20, 250, 100),
        'Z': np.random.randint(10, 100, 100),
        'N': np.random.randint(10, 150, 100),
        'MM': np.random.randn(100) * 2
    })
    test_df.to_csv(test_data_dir / 'MM_100nuclei.csv', index=False)
    
    # Initialize splitter
    splitter = UnknownNucleiSplitter(
        split_ratio=0.7,
        output_dir='test_unknown_splits'
    )
    
    # Split datasets
    print("\nSplitting datasets...")
    splitter.split_all_datasets(test_data_dir)
    
    print("\n[SUCCESS] TEST COMPLETED!")


if __name__ == "__main__":
    main()
