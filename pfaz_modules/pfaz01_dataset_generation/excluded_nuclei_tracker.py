"""
Excluded Nuclei Tracker
=======================

Filtrelenen çekirdekleri takip eden ve raporlayan sistem.
Tüm filtreleme nedenlerini detaylı olarak kaydeder ve Excel raporları oluşturur.

Author: Nuclear Physics AI Project
Version: 1.0.0
Date: 2025-11-23
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core_modules.json_utils import sanitize_for_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcludedNucleiTracker:
    """
    Filtrelenen çekirdekleri takip eder ve detaylı raporlar oluşturur.

    Bu sınıf, dataset oluşturma sürecinde çıkartılan tüm çekirdekleri,
    nedenleriyle birlikte kaydeder ve kapsamlı Excel raporları üretir.
    """

    # Filtreleme neden kodları
    REASON_CODES = {
        'MISSING_TARGET': 'Hedef değişken eksik',
        'QM_REQUIRED': 'QM gerekli ama yok',
        'ODD_A_MM_ZERO': 'Odd-A ama MM=0 (fiziksel hata)',
        'OUTLIER_REMOVED': 'Outlier tespit edildi',
        'PHYSICAL_VIOLATION': 'Fiziksel kısıt ihlali',
        'INVALID_SPIN': 'Geçersiz spin değeri',
        'INVALID_PARITY': 'Geçersiz parite',
        'RANGE_VIOLATION': 'Değer aralığı dışında',
        'INSUFFICIENT_FEATURES': 'Yeterli özellik yok',
        'DUPLICATE': 'Duplikasyon',
        'MISSING_QM_FOR_FEATURES': 'Q-dependent features için QM gerekli',
        'MISSING_BETA2': 'Beta_2 değeri eksik',
        'INVALID_VALUE': 'Geçersiz değer',
        'DATA_QUALITY_ISSUE': 'Veri kalite sorunu'
    }

    def __init__(self):
        """Initialize the tracker"""
        self.excluded = []
        logger.info("ExcludedNucleiTracker başlatıldı")

    def add_exclusion(self,
                     nucleus: str,
                     reason: str,
                     target: str,
                     a: Optional[int] = None,
                     z: Optional[int] = None,
                     n: Optional[int] = None,
                     details: Optional[Dict] = None):
        """
        Filtrelenen çekirdeği kaydet

        Args:
            nucleus: Çekirdek adı (örn: '7wHf176')
            reason: Filtreleme nedeni (REASON_CODES'dan biri)
            target: Hangi target için filtrelendi
            a: Kütle numarası
            z: Proton sayısı
            n: Nötron sayısı
            details: Ek detaylar (dictionary)
        """
        exclusion_entry = {
            'NUCLEUS': nucleus,
            'A': a,
            'Z': z,
            'N': n,
            'Target': target,
            'Reason': reason,
            'Reason_Description': self.REASON_CODES.get(reason, 'Unknown reason'),
            'Details': json.dumps(details) if details else '{}',
            'Timestamp': datetime.now().isoformat()
        }

        self.excluded.append(exclusion_entry)

        # Log the exclusion
        logger.debug(f"Excluded: {nucleus} (Target: {target}, Reason: {reason})")

    def add_bulk_exclusions(self,
                           nuclei_list: List[str],
                           reason: str,
                           target: str,
                           df: Optional[pd.DataFrame] = None,
                           details_dict: Optional[Dict] = None):
        """
        Birden fazla çekirdeği toplu olarak kaydet

        Args:
            nuclei_list: Çekirdek adları listesi
            reason: Filtreleme nedeni
            target: Target adı
            df: Orijinal DataFrame (A,Z,N bilgileri için)
            details_dict: Her çekirdek için detaylar {nucleus: details}
        """
        for nucleus in nuclei_list:
            a, z, n = None, None, None

            # DataFrame'den A,Z,N bilgilerini al
            if df is not None and 'NUCLEUS' in df.columns:
                nucleus_row = df[df['NUCLEUS'] == nucleus]
                if not nucleus_row.empty:
                    a = int(nucleus_row.iloc[0]['A']) if 'A' in df.columns else None
                    z = int(nucleus_row.iloc[0]['Z']) if 'Z' in df.columns else None
                    n = int(nucleus_row.iloc[0]['N']) if 'N' in df.columns else None

            # Detayları al
            details = details_dict.get(nucleus) if details_dict else None

            self.add_exclusion(
                nucleus=nucleus,
                reason=reason,
                target=target,
                a=a,
                z=z,
                n=n,
                details=details
            )

        logger.info(f"[TRACKER] Added {len(nuclei_list)} exclusions (Target: {target}, Reason: {reason})")

    def get_summary(self) -> Dict:
        """
        Filtreleme özeti

        Returns:
            Dictionary with summary statistics
        """
        if not self.excluded:
            return {
                'total_excluded': 0,
                'by_reason': {},
                'by_target': {},
                'by_reason_and_target': {}
            }

        df = pd.DataFrame(self.excluded)

        # Convert multi-level groupby to JSON-serializable format
        reason_target_grouped = df.groupby(['Reason', 'Target']).size()
        reason_target_dict = {f"{reason}_{target}": count
                              for (reason, target), count in reason_target_grouped.items()}

        summary = {
            'total_excluded': len(df),
            'by_reason': df.groupby('Reason').size().to_dict(),
            'by_target': df.groupby('Target').size().to_dict(),
            'by_reason_and_target': reason_target_dict  # Now JSON-serializable
        }

        return summary

    def print_summary(self):
        """Özeti konsola yazdır"""
        summary = self.get_summary()

        logger.info("\n" + "=" * 80)
        logger.info("EXCLUDED NUCLEI SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total excluded: {summary['total_excluded']}")

        if summary['total_excluded'] > 0:
            logger.info("\nBy Reason:")
            for reason, count in sorted(summary['by_reason'].items(), key=lambda x: -x[1]):
                logger.info(f"  {reason}: {count}")

            logger.info("\nBy Target:")
            for target, count in sorted(summary['by_target'].items()):
                logger.info(f"  {target}: {count}")

        logger.info("=" * 80)

    def save_to_excel(self, output_path: str):
        """
        Excel dosyasına detaylı rapor kaydet

        Args:
            output_path: Excel dosya yolu
        """
        if not self.excluded:
            logger.warning("[WARNING] No excluded nuclei to save")
            return

        df = pd.DataFrame(self.excluded)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: All Excluded
            df_sorted = df.sort_values(['Target', 'Reason', 'NUCLEUS'])
            df_sorted.to_excel(writer, sheet_name='All_Excluded', index=False)

            # Sheet 2: Summary by Reason
            reason_summary = df.groupby(['Reason', 'Reason_Description']).size().reset_index(name='Count')
            reason_summary = reason_summary.sort_values('Count', ascending=False)
            reason_summary.to_excel(writer, sheet_name='By_Reason', index=False)

            # Sheet 3: Summary by Target
            target_summary = df.groupby('Target').size().reset_index(name='Count')
            target_summary = target_summary.sort_values('Count', ascending=False)
            target_summary.to_excel(writer, sheet_name='By_Target', index=False)

            # Sheet 4: Cross-tabulation (Reason × Target)
            cross_tab = pd.crosstab(df['Reason'], df['Target'], margins=True)
            cross_tab.to_excel(writer, sheet_name='Reason_x_Target')

            # Sheets 5+: Detailed sheets for each reason
            for reason in sorted(df['Reason'].unique()):
                reason_df = df[df['Reason'] == reason].copy()

                # Sheet name (max 31 characters for Excel)
                sheet_name = f"Reason_{reason}"[:31]

                # Sort by target and nucleus
                reason_df = reason_df.sort_values(['Target', 'NUCLEUS'])

                reason_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Sheets N+: Detailed sheets for each target
            for target in sorted(df['Target'].unique()):
                target_df = df[df['Target'] == target].copy()

                # Sheet name
                sheet_name = f"Target_{target}"[:31]

                # Sort by reason and nucleus
                target_df = target_df.sort_values(['Reason', 'NUCLEUS'])

                target_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"[OK] Excluded nuclei report saved: {output_path}")
        logger.info(f"     Total excluded: {len(df)}")
        logger.info(f"     Sheets created: {3 + len(df['Reason'].unique()) + len(df['Target'].unique())}")

    def save_to_csv(self, output_path: str):
        """
        CSV dosyasına kaydet (basit format)

        Args:
            output_path: CSV dosya yolu
        """
        if not self.excluded:
            logger.warning("[WARNING] No excluded nuclei to save")
            return

        df = pd.DataFrame(self.excluded)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"[OK] Excluded nuclei saved to CSV: {output_path}")

    def save_to_json(self, output_path: str):
        """
        JSON dosyasına kaydet

        Args:
            output_path: JSON dosya yolu
        """
        if not self.excluded:
            logger.warning("[WARNING] No excluded nuclei to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_summary()

        output_data = {
            'summary': summary,
            'excluded_nuclei': self.excluded,
            'timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sanitize_for_json(output_data), f, indent=2)

        logger.info(f"[OK] Excluded nuclei saved to JSON: {output_path}")

    def clear(self):
        """Tüm kayıtları temizle"""
        self.excluded = []
        logger.info("ExcludedNucleiTracker cleared")


# ============================================================================
# MAIN TEST
# ============================================================================

def test_excluded_nuclei_tracker():
    """Test the tracker"""

    print("\n" + "="*80)
    print("EXCLUDED NUCLEI TRACKER TEST")
    print("="*80)

    tracker = ExcludedNucleiTracker()

    # Add some test exclusions
    tracker.add_exclusion(
        nucleus='7wHf176',
        reason='MISSING_TARGET',
        target='MM',
        a=176, z=72, n=104,
        details={'MM': None, 'note': 'MM value is NaN'}
    )

    tracker.add_exclusion(
        nucleus='8wTa181',
        reason='QM_REQUIRED',
        target='QM',
        a=181, z=73, n=108,
        details={'Q': None, 'note': 'Q required for QM target'}
    )

    tracker.add_exclusion(
        nucleus='9wW184',
        reason='OUTLIER_REMOVED',
        target='MM',
        a=184, z=74, n=110,
        details={'MM': 15.5, 'z_score': 4.2, 'threshold': 3.0}
    )

    # Bulk exclusions
    tracker.add_bulk_exclusions(
        nuclei_list=['10wRe187', '11wOs188', '12wIr191'],
        reason='MISSING_QM_FOR_FEATURES',
        target='Beta_2',
        details_dict={
            '10wRe187': {'Q': None, 'has_q_features': True},
            '11wOs188': {'Q': None, 'has_q_features': True},
            '12wIr191': {'Q': None, 'has_q_features': True}
        }
    )

    # Print summary
    tracker.print_summary()

    # Save reports
    tracker.save_to_excel('test_excluded_nuclei_report.xlsx')
    tracker.save_to_csv('test_excluded_nuclei_report.csv')
    tracker.save_to_json('test_excluded_nuclei_report.json')

    print("\n[OK] Tracker test completed!")


if __name__ == "__main__":
    test_excluded_nuclei_tracker()
