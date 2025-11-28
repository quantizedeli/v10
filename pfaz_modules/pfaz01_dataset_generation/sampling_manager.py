"""
Sampling Manager for PFAZ1 Dataset Generation
=============================================

Manages sampling strategies for nuclear physics datasets:
- Random: Random sampling with reproducible seed
- Stratified: Stratified sampling by A, Z, or magic numbers

Ensures representative distribution of nuclear properties.

Author: Nuclear Physics AI Project
Version: 1.0.0 (FAZ 3)
Date: 2025-11-23
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# Magic numbers for protons and neutrons
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]


class SamplingManager:
    """
    Manages sampling strategies for nuclear datasets.

    Supported methods:
    1. Random: Simple random sampling with seed
    2. Stratified: Stratified by mass number ranges (A)
    3. StratifiedMagic: Ensure magic number representation
    4. StratifiedHybrid: Combined A-based and magic-based stratification
    """

    AVAILABLE_METHODS = ['Random', 'Stratified', 'StratifiedMagic', 'StratifiedHybrid']

    def __init__(self, method: str = 'Random', random_seed: int = 42):
        """
        Initialize SamplingManager.

        Args:
            method: Sampling method ('Random', 'Stratified', 'StratifiedMagic', 'StratifiedHybrid')
            random_seed: Random seed for reproducibility
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Invalid sampling method: {method}. "
                f"Available: {self.AVAILABLE_METHODS}"
            )

        self.method = method
        self.random_seed = random_seed

        logger.info(f"SamplingManager initialized: method={method}, seed={random_seed}")

    def sample(self,
               df: pd.DataFrame,
               n_samples: int,
               group_col: str = 'A',
               ensure_magic: bool = True) -> pd.DataFrame:
        """
        Sample from DataFrame using configured method.

        Args:
            df: Source DataFrame
            n_samples: Number of samples to draw
            group_col: Column to use for stratification (default: 'A')
            ensure_magic: Ensure magic number nuclei are included (for Stratified methods)

        Returns:
            Sampled DataFrame
        """
        if n_samples > len(df):
            logger.warning(
                f"Requested {n_samples} samples but only {len(df)} available. "
                f"Returning all data."
            )
            return df.copy()

        if self.method == 'Random':
            return self._random_sample(df, n_samples)

        elif self.method == 'Stratified':
            return self._stratified_sample_by_ranges(df, n_samples, group_col)

        elif self.method == 'StratifiedMagic':
            return self._stratified_sample_magic(df, n_samples)

        elif self.method == 'StratifiedHybrid':
            return self._stratified_sample_hybrid(df, n_samples, group_col)

    def _random_sample(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        Simple random sampling.

        Args:
            df: Source DataFrame
            n_samples: Number of samples

        Returns:
            Sampled DataFrame
        """
        return df.sample(n=n_samples, random_state=self.random_seed).reset_index(drop=True)

    def _stratified_sample_by_ranges(self,
                                      df: pd.DataFrame,
                                      n_samples: int,
                                      group_col: str = 'A') -> pd.DataFrame:
        """
        Stratified sampling by mass number ranges.

        Creates bins for A values and samples proportionally from each bin.

        Args:
            df: Source DataFrame
            n_samples: Number of samples
            group_col: Column to stratify by (default: 'A')

        Returns:
            Sampled DataFrame
        """
        if group_col not in df.columns:
            logger.warning(f"Column '{group_col}' not found, falling back to random sampling")
            return self._random_sample(df, n_samples)

        # Create mass number bins
        # Light: A < 60
        # Medium: 60 <= A < 150
        # Heavy: A >= 150
        bins = [0, 60, 150, 300]
        labels = ['Light', 'Medium', 'Heavy']

        df_copy = df.copy()
        df_copy['_mass_bin'] = pd.cut(df_copy[group_col], bins=bins, labels=labels)

        # Calculate samples per bin (proportional)
        bin_counts = df_copy['_mass_bin'].value_counts()
        bin_props = bin_counts / len(df_copy)

        sampled_dfs = []
        total_sampled = 0

        for bin_label in labels:
            bin_df = df_copy[df_copy['_mass_bin'] == bin_label]

            if len(bin_df) == 0:
                continue

            # Calculate how many samples from this bin
            n_from_bin = int(np.round(n_samples * bin_props[bin_label]))
            n_from_bin = min(n_from_bin, len(bin_df))  # Can't sample more than available

            if total_sampled + n_from_bin > n_samples:
                n_from_bin = n_samples - total_sampled

            if n_from_bin > 0:
                sampled_bin = bin_df.sample(n=n_from_bin, random_state=self.random_seed + hash(bin_label) % 1000)
                sampled_dfs.append(sampled_bin)
                total_sampled += n_from_bin

        # If we haven't reached n_samples due to rounding, fill from largest bin
        if total_sampled < n_samples:
            remaining = n_samples - total_sampled
            largest_bin = bin_counts.idxmax()
            bin_df = df_copy[df_copy['_mass_bin'] == largest_bin]
            # Get samples not already selected
            already_sampled = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
            remaining_df = bin_df[~bin_df.index.isin(already_sampled.index)]

            if len(remaining_df) >= remaining:
                extra = remaining_df.sample(n=remaining, random_state=self.random_seed + 999)
                sampled_dfs.append(extra)

        result = pd.concat(sampled_dfs).drop(columns=['_mass_bin']).reset_index(drop=True)

        logger.info(
            f"Stratified sampling by {group_col}: "
            f"sampled {len(result)} from {len(df)}"
        )

        return result

    def _is_magic(self, value: int) -> bool:
        """Check if a number is a magic number."""
        return value in MAGIC_NUMBERS

    def _stratified_sample_magic(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """
        Stratified sampling ensuring magic number representation.

        Ensures that nuclei with magic Z or N are included.

        Args:
            df: Source DataFrame
            n_samples: Number of samples

        Returns:
            Sampled DataFrame
        """
        if 'Z' not in df.columns or 'N' not in df.columns:
            logger.warning("Z or N column missing, falling back to random sampling")
            return self._random_sample(df, n_samples)

        df_copy = df.copy()

        # Identify magic nuclei
        df_copy['_is_magic_z'] = df_copy['Z'].apply(self._is_magic)
        df_copy['_is_magic_n'] = df_copy['N'].apply(self._is_magic)
        df_copy['_is_magic'] = df_copy['_is_magic_z'] | df_copy['_is_magic_n']

        magic_df = df_copy[df_copy['_is_magic']]
        non_magic_df = df_copy[~df_copy['_is_magic']]

        # Allocate ~30% of samples to magic nuclei (if available)
        n_magic_target = min(int(n_samples * 0.3), len(magic_df))
        n_non_magic = n_samples - n_magic_target

        sampled_dfs = []

        # Sample magic nuclei
        if n_magic_target > 0 and len(magic_df) > 0:
            magic_sampled = magic_df.sample(
                n=min(n_magic_target, len(magic_df)),
                random_state=self.random_seed
            )
            sampled_dfs.append(magic_sampled)

        # Sample non-magic nuclei
        if n_non_magic > 0 and len(non_magic_df) > 0:
            non_magic_sampled = non_magic_df.sample(
                n=min(n_non_magic, len(non_magic_df)),
                random_state=self.random_seed + 1
            )
            sampled_dfs.append(non_magic_sampled)

        # If we don't have enough, fill from the other category
        total_sampled = sum(len(s) for s in sampled_dfs)
        if total_sampled < n_samples:
            remaining = n_samples - total_sampled
            if len(non_magic_df) > n_non_magic:
                extra_df = non_magic_df[~non_magic_df.index.isin(sampled_dfs[-1].index if sampled_dfs else [])]
                if len(extra_df) > 0:
                    extra = extra_df.sample(n=min(remaining, len(extra_df)), random_state=self.random_seed + 2)
                    sampled_dfs.append(extra)

        result = pd.concat(sampled_dfs).drop(
            columns=['_is_magic_z', '_is_magic_n', '_is_magic']
        ).reset_index(drop=True)

        n_magic_sampled = sum(result['Z'].apply(self._is_magic) | result['N'].apply(self._is_magic))
        logger.info(
            f"Stratified magic sampling: {len(result)} samples "
            f"({n_magic_sampled} magic, {len(result) - n_magic_sampled} non-magic)"
        )

        return result

    def _stratified_sample_hybrid(self,
                                   df: pd.DataFrame,
                                   n_samples: int,
                                   group_col: str = 'A') -> pd.DataFrame:
        """
        Hybrid stratified sampling: both A-based bins AND magic numbers.

        Args:
            df: Source DataFrame
            n_samples: Number of samples
            group_col: Column for mass-based stratification

        Returns:
            Sampled DataFrame
        """
        if 'Z' not in df.columns or 'N' not in df.columns:
            logger.warning("Z or N missing, using simple stratified sampling")
            return self._stratified_sample_by_ranges(df, n_samples, group_col)

        # Step 1: Ensure magic nuclei representation (~20% of samples)
        df_copy = df.copy()
        df_copy['_is_magic_z'] = df_copy['Z'].apply(self._is_magic)
        df_copy['_is_magic_n'] = df_copy['N'].apply(self._is_magic)
        df_copy['_is_magic'] = df_copy['_is_magic_z'] | df_copy['_is_magic_n']

        magic_df = df_copy[df_copy['_is_magic']]
        non_magic_df = df_copy[~df_copy['_is_magic']]

        n_magic_target = min(int(n_samples * 0.2), len(magic_df))
        n_non_magic = n_samples - n_magic_target

        sampled_dfs = []

        # Sample magic nuclei
        if n_magic_target > 0 and len(magic_df) > 0:
            magic_sampled = magic_df.sample(
                n=min(n_magic_target, len(magic_df)),
                random_state=self.random_seed
            )
            sampled_dfs.append(magic_sampled)

        # Step 2: Sample remaining from A-based bins
        if n_non_magic > 0 and len(non_magic_df) > 0:
            # Use A-based stratification for non-magic nuclei
            bins = [0, 60, 150, 300]
            labels = ['Light', 'Medium', 'Heavy']
            non_magic_df['_mass_bin'] = pd.cut(non_magic_df[group_col], bins=bins, labels=labels)

            bin_counts = non_magic_df['_mass_bin'].value_counts()
            bin_props = bin_counts / len(non_magic_df)

            for bin_label in labels:
                bin_df = non_magic_df[non_magic_df['_mass_bin'] == bin_label]
                if len(bin_df) == 0:
                    continue

                n_from_bin = int(np.round(n_non_magic * bin_props[bin_label]))
                n_from_bin = min(n_from_bin, len(bin_df))

                if n_from_bin > 0:
                    sampled_bin = bin_df.sample(
                        n=n_from_bin,
                        random_state=self.random_seed + hash(bin_label) % 1000
                    )
                    sampled_dfs.append(sampled_bin)

        result = pd.concat(sampled_dfs).drop(
            columns=['_is_magic_z', '_is_magic_n', '_is_magic', '_mass_bin'],
            errors='ignore'
        ).reset_index(drop=True)

        n_magic_sampled = sum(result['Z'].apply(self._is_magic) | result['N'].apply(self._is_magic))
        logger.info(
            f"Hybrid stratified sampling: {len(result)} samples "
            f"({n_magic_sampled} magic, {len(result) - n_magic_sampled} non-magic)"
        )

        return result

    def get_sampling_info(self) -> Dict:
        """
        Get sampling configuration info.

        Returns:
            Dictionary with sampling details
        """
        return {
            'method': self.method,
            'random_seed': self.random_seed,
            'description': {
                'Random': 'Simple random sampling',
                'Stratified': 'Stratified by mass number (A) ranges',
                'StratifiedMagic': 'Ensures magic number nuclei representation',
                'StratifiedHybrid': 'Combined A-based and magic-based stratification'
            }[self.method]
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sampling_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate statistics about a sampled dataset.

    Args:
        df: Sampled DataFrame

    Returns:
        Statistics dictionary
    """
    stats = {
        'n_samples': len(df),
        'A_range': [int(df['A'].min()), int(df['A'].max())] if 'A' in df.columns else None,
        'Z_range': [int(df['Z'].min()), int(df['Z'].max())] if 'Z' in df.columns else None,
        'N_range': [int(df['N'].min()), int(df['N'].max())] if 'N' in df.columns else None,
    }

    if 'Z' in df.columns and 'N' in df.columns:
        n_magic_z = df['Z'].apply(lambda z: z in MAGIC_NUMBERS).sum()
        n_magic_n = df['N'].apply(lambda n: n in MAGIC_NUMBERS).sum()
        n_magic_total = ((df['Z'].apply(lambda z: z in MAGIC_NUMBERS)) |
                         (df['N'].apply(lambda n: n in MAGIC_NUMBERS))).sum()

        stats['magic_nuclei'] = {
            'magic_Z': int(n_magic_z),
            'magic_N': int(n_magic_n),
            'magic_total': int(n_magic_total),
            'magic_percentage': float(n_magic_total / len(df) * 100)
        }

    return stats


if __name__ == "__main__":
    print("SamplingManager module loaded successfully!")
    print(f"Available methods: {SamplingManager.AVAILABLE_METHODS}")
    print(f"Magic numbers: {MAGIC_NUMBERS}")
