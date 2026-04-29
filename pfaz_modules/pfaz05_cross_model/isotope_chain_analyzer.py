# -*- coding: utf-8 -*-
"""
Isotope Chain Correlation Analyzer
====================================

Identifies nuclei in isotopic chains where target values change suddenly,
then correlates those changes with nuclear structure features:
  - Magic numbers (N, Z = 2, 8, 20, 28, 50, 82, 126)
  - Shell closures / shell gaps
  - Valence nucleon numbers
  - Deformation (Beta_2)

Usage:
    analyzer = IsotopeChainAnalyzer(aaa2_txt_path='aaa2.txt')
    results = analyzer.run_full_analysis()
    analyzer.save_excel_report('isotope_chain_report.xlsx')

Author: Nuclear Physics AI Project
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Magic numbers for protons and neutrons
MAGIC_NUMBERS = {2, 8, 20, 28, 50, 82, 126}

# Target column names as they appear in aaa2.txt
TARGET_COLS = {
    'MM':     'MAGNETIC MOMENT [µ]',
    'QM':     'QUADRUPOLE MOMENT [Q]',
    'Beta_2': 'Beta_2',
}

# Threshold multiplier for "sudden change" detection (relative to chain std)
SUDDEN_CHANGE_SIGMA = 1.5


def _is_magic(n: int) -> bool:
    return int(n) in MAGIC_NUMBERS


def _magic_distance(n: int) -> int:
    """Distance to nearest magic number."""
    return min(abs(int(n) - m) for m in MAGIC_NUMBERS)


def _shell_region(n: int) -> str:
    """Classify N or Z by shell region."""
    n = int(n)
    if n <= 2:
        return '1s (≤2)'
    elif n <= 8:
        return '1p (3-8)'
    elif n <= 20:
        return '1d2s (9-20)'
    elif n <= 28:
        return 'fp (21-28)'
    elif n <= 50:
        return 'fp-g (29-50)'
    elif n <= 82:
        return 'g-h (51-82)'
    elif n <= 126:
        return 'h-i (83-126)'
    else:
        return 'superheavy (>126)'


class IsotopeChainAnalyzer:
    """
    Analyzes isotope chains for sudden changes in nuclear target values.

    For each element Z, walks through its isotopes (sorted by N) and
    detects sudden changes in MM, QM, and Beta_2 values.
    """

    def __init__(self, aaa2_txt_path: str = 'aaa2.txt',
                 output_dir: str = 'outputs/isotope_chain_analysis'):
        self.aaa2_path = Path(aaa2_txt_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_df: Optional[pd.DataFrame] = None
        self.chain_results: Dict[str, pd.DataFrame] = {}
        self.sudden_changes: Dict[str, pd.DataFrame] = {}
        self.summary: Dict = {}

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load aaa2.txt with flexible column detection."""
        if not self.aaa2_path.exists():
            raise FileNotFoundError(f"aaa2.txt not found: {self.aaa2_path}")

        # Try space/tab-separated first
        try:
            df = pd.read_csv(self.aaa2_path, sep=r'\s+', comment='#', encoding='utf-8')
        except Exception:
            df = pd.read_csv(self.aaa2_path, encoding='utf-8')

        logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
        self.raw_df = df
        return df

    def _get_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Return the first candidate column that exists in df."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # -------------------------------------------------------------------------
    # Chain Analysis
    # -------------------------------------------------------------------------

    def analyze_isotope_chains(self) -> Dict[str, pd.DataFrame]:
        """
        For each target, compute per-element isotope chains and find sudden changes.

        Returns:
            Dict {target_key: DataFrame with all chain rows + change metadata}
        """
        if self.raw_df is None:
            self.load_data()

        df = self.raw_df.copy()

        # Resolve column names
        z_col = self._get_column(df, ['Z', 'z', 'Protons'])
        n_col = self._get_column(df, ['N', 'Nn', 'n', 'Neutrons'])
        a_col = self._get_column(df, ['A', 'a', 'Mass'])
        nuc_col = self._get_column(df, ['NUCLEUS', 'nucleus', 'Nucleus', 'Name'])

        if z_col is None or n_col is None:
            raise ValueError("Could not find Z/N columns in aaa2.txt")

        # Convert to numeric
        for col in [z_col, n_col, a_col]:
            if col:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for target_key, raw_col in TARGET_COLS.items():
            col = self._get_column(df, [raw_col, target_key])
            if col is None:
                logger.warning(f"Target column not found: {raw_col} — skipping {target_key}")
                continue

            df[col] = pd.to_numeric(df[col], errors='coerce')
            valid = df[[z_col, n_col, col]].dropna()

            if valid.empty:
                continue

            logger.info(f"Analyzing isotope chains for {target_key} ({len(valid)} nuclei)")

            rows = []
            z_groups = valid.groupby(z_col)

            for z_val, group in z_groups:
                chain = group.sort_values(n_col).reset_index(drop=True)
                if len(chain) < 2:
                    continue

                values = chain[col].values
                ns = chain[n_col].values

                # Compute finite differences (dV/dN)
                diffs = np.diff(values)
                diff_ns = ns[1:]  # N value where change occurs

                chain_std = np.std(values)
                chain_mean = np.mean(values)
                threshold = SUDDEN_CHANGE_SIGMA * chain_std if chain_std > 0 else 1e-10

                for i, (diff, n_after) in enumerate(zip(diffs, diff_ns)):
                    n_before = ns[i]
                    v_before = values[i]
                    v_after = values[i + 1]

                    is_sudden = abs(diff) > threshold
                    a_after = int(z_val + n_after) if a_col else None

                    # Look up NUCLEUS name
                    nucleus_id = ''
                    if nuc_col:
                        matches = df[(df[z_col] == z_val) & (df[n_col] == n_after)]
                        if not matches.empty:
                            nucleus_id = str(matches.iloc[0][nuc_col])

                    rows.append({
                        'Target':           target_key,
                        'Z':                int(z_val),
                        'N_before':         int(n_before),
                        'N_after':          int(n_after),
                        'A_after':          a_after,
                        'NUCLEUS':          nucleus_id,
                        'Value_before':     round(float(v_before), 6),
                        'Value_after':      round(float(v_after), 6),
                        'Delta':            round(float(diff), 6),
                        'Abs_Delta':        round(abs(float(diff)), 6),
                        'Chain_Std':        round(float(chain_std), 6),
                        'Delta_over_Sigma': round(abs(diff) / chain_std if chain_std > 0 else 0, 4),
                        'Is_Sudden_Change': is_sudden,
                        'Z_is_Magic':       _is_magic(z_val),
                        'N_before_is_Magic':_is_magic(n_before),
                        'N_after_is_Magic': _is_magic(n_after),
                        'Z_Magic_Dist':     _magic_distance(z_val),
                        'N_Magic_Dist':     _magic_distance(n_after),
                        'Z_Shell_Region':   _shell_region(z_val),
                        'N_Shell_Region':   _shell_region(n_after),
                        'Chain_N_first':    int(ns[0]),
                        'Chain_N_last':     int(ns[-1]),
                        'Chain_Length':     len(chain),
                        'Chain_Mean':       round(float(chain_mean), 6),
                    })

            if rows:
                target_df = pd.DataFrame([r for r in rows if r['Target'] == target_key])
                self.chain_results[target_key] = target_df

                sudden = target_df[target_df['Is_Sudden_Change']].copy()
                sudden = sudden.sort_values('Delta_over_Sigma', ascending=False)
                self.sudden_changes[target_key] = sudden

                logger.info(f"  {target_key}: {len(target_df)} transitions, "
                            f"{len(sudden)} sudden changes")

        return self.chain_results

    # -------------------------------------------------------------------------
    # Feature Correlation at Sudden Changes
    # -------------------------------------------------------------------------

    def compute_sudden_change_stats(self) -> Dict:
        """
        Summarize sudden change statistics per target.

        Returns stats dict with:
          - total_transitions, n_sudden, sudden_pct
          - magic_N_sudden_pct: % of sudden changes where N_after is magic
          - magic_Z_sudden_pct: % of sudden changes where Z is magic
          - top_shell_regions: most common shell region at sudden changes
        """
        stats = {}

        for target_key, sudden_df in self.sudden_changes.items():
            all_df = self.chain_results.get(target_key, pd.DataFrame())
            if all_df.empty or sudden_df.empty:
                continue

            n_total = len(all_df)
            n_sudden = len(sudden_df)

            magic_n = sudden_df['N_after_is_Magic'].sum()
            magic_z = sudden_df['Z_is_Magic'].sum()

            top_n_regions = sudden_df['N_Shell_Region'].value_counts().head(3).to_dict()
            top_z_regions = sudden_df['Z_Shell_Region'].value_counts().head(3).to_dict()

            # Cross-correlation: magic number presence vs non-magic
            magic_present = sudden_df[sudden_df['N_after_is_Magic'] | sudden_df['Z_is_Magic']]
            magic_absent  = sudden_df[~(sudden_df['N_after_is_Magic'] | sudden_df['Z_is_Magic'])]

            mean_delta_magic = float(magic_present['Abs_Delta'].mean()) if not magic_present.empty else 0
            mean_delta_nonmagic = float(magic_absent['Abs_Delta'].mean()) if not magic_absent.empty else 0

            stats[target_key] = {
                'total_transitions': int(n_total),
                'n_sudden':          int(n_sudden),
                'sudden_pct':        round(n_sudden / max(1, n_total) * 100, 2),
                'magic_N_in_sudden': int(magic_n),
                'magic_Z_in_sudden': int(magic_z),
                'magic_N_pct':       round(magic_n / max(1, n_sudden) * 100, 2),
                'magic_Z_pct':       round(magic_z / max(1, n_sudden) * 100, 2),
                'top_N_shell_regions': top_n_regions,
                'top_Z_shell_regions': top_z_regions,
                'mean_delta_at_magic':    round(mean_delta_magic, 6),
                'mean_delta_at_nonmagic': round(mean_delta_nonmagic, 6),
                'magic_amplification':    round(mean_delta_magic / max(1e-9, mean_delta_nonmagic), 4),
            }

        self.summary = stats
        return stats

    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------

    def run_full_analysis(self) -> Dict:
        """Run complete isotope chain analysis pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info("ISOTOPE CHAIN CORRELATION ANALYSIS")
        logger.info("=" * 70)

        self.analyze_isotope_chains()
        stats = self.compute_sudden_change_stats()

        for target, s in stats.items():
            logger.info(f"\n[{target}]")
            logger.info(f"  Transitions: {s['total_transitions']}, Sudden: {s['n_sudden']} ({s['sudden_pct']:.1f}%)")
            logger.info(f"  Magic-N at sudden: {s['magic_N_in_sudden']} ({s['magic_N_pct']:.1f}%)")
            logger.info(f"  Magic-Z at sudden: {s['magic_Z_in_sudden']} ({s['magic_Z_pct']:.1f}%)")
            logger.info(f"  Mean |Δ| at magic: {s['mean_delta_at_magic']:.4f} vs non-magic: {s['mean_delta_at_nonmagic']:.4f} "
                        f"(×{s['magic_amplification']:.2f})")

        return {'chain_results': self.chain_results,
                'sudden_changes': self.sudden_changes,
                'stats': stats}

    def save_excel_report(self, output_path: str = None) -> Path:
        """
        Save isotope chain analysis to Excel.

        Sheets:
          - Summary: per-target stats table
          - {TARGET}_SuddenChanges: sudden change nuclei with feature context
          - {TARGET}_AllTransitions: full chain data
        """
        try:
            import openpyxl  # noqa
        except ImportError:
            pass

        if output_path is None:
            from datetime import datetime
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'isotope_chain_analysis_{ts}.xlsx'

        output_path = Path(output_path)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            if self.summary:
                summary_rows = []
                for target, s in self.summary.items():
                    summary_rows.append({
                        'Target':                    target,
                        'Total_Transitions':         s['total_transitions'],
                        'Sudden_Changes':            s['n_sudden'],
                        'Sudden_Pct':                s['sudden_pct'],
                        'Magic_N_Count':             s['magic_N_in_sudden'],
                        'Magic_N_Pct':               s['magic_N_pct'],
                        'Magic_Z_Count':             s['magic_Z_in_sudden'],
                        'Magic_Z_Pct':               s['magic_Z_pct'],
                        'Mean_Delta_at_Magic':        s['mean_delta_at_magic'],
                        'Mean_Delta_at_NonMagic':     s['mean_delta_at_nonmagic'],
                        'Magic_Amplification':        s['magic_amplification'],
                        'Top_N_Shell_Regions':        str(s['top_N_shell_regions']),
                    })
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name='IsoChain_Summary', index=False)

            # Sheets per target
            for target_key in self.sudden_changes:
                sudden = self.sudden_changes[target_key]
                if not sudden.empty:
                    sheet_name = f'{target_key}_SuddenChanges'[:31]
                    sudden.to_excel(writer, sheet_name=sheet_name, index=False)

                all_df = self.chain_results.get(target_key, pd.DataFrame())
                if not all_df.empty:
                    sheet_name2 = f'{target_key}_AllTransitions'[:31]
                    all_df.to_excel(writer, sheet_name=sheet_name2, index=False)

        logger.info(f"[OK] Isotope chain report: {output_path}")
        return output_path

    def get_sudden_changes_flat(self) -> pd.DataFrame:
        """Return all sudden changes across all targets as a single flat DataFrame."""
        frames = list(self.sudden_changes.values())
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ['Target', 'Delta_over_Sigma'], ascending=[True, False])
