"""
aaa2.txt Detaylı Veri Kalite Kontrol Raporu
Comprehensive Data Quality Control for aaa2.txt

Bu modül aaa2.txt dosyasındaki tüm veriyi kontrol eder ve detaylı Excel raporu oluşturur.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.file_io_utils import read_nuclear_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class AAA2DataQualityChecker:
    """aaa2.txt için detaylı veri kalitesi kontrolü"""
    
    def __init__(self, output_dir='reports/data_quality'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.issues = {
            'missing_values': [],
            'zero_values': [],
            'text_in_numeric': [],
            'date_in_numeric': [],
            'parity_errors': [],
            'azn_mismatch': [],
            'mm_zero_odd_a': [],
            'qm_missing': [],
            'other_issues': []
        }
        
    def load_and_check(self, filepath='aaa2.txt'):
        """
        Veriyi yükle ve tüm kontrolleri yap
        
        Args:
            filepath: aaa2.txt dosya yolu
            
        Returns:
            DataFrame: Temizlenmiş veri
            dict: Tespit edilen sorunlar
        """
        logger.info("="*80)
        logger.info("AAA2.TXT DETAYLI VERİ KALİTE KONTROLÜ")
        logger.info("="*80)
        
        # 1. Dosyayı yükle
        logger.info("\n1. Dosya yükleniyor...")
        df = read_nuclear_data(filepath, encoding='utf-8')
        df.columns = df.columns.str.strip()
        
        logger.info(f"   [OK] Toplam satır: {len(df)}")
        logger.info(f"   [OK] Sütunlar: {list(df.columns)}")
        
        # 2. Eksik değer kontrolü
        logger.info("\n2. Eksik değer kontrolü...")
        self._check_missing_values(df)
        
        # 3. Sıfır değer kontrolü (MM için kritik)
        logger.info("\n3. Sıfır değer kontrolü...")
        self._check_zero_values(df)
        
        # 4. Metin/tarih kontrolleri
        logger.info("\n4. Metin ve tarih kontrolleri...")
        self._check_text_and_dates(df)
        
        # 5. PARITY kontrolü
        logger.info("\n5. PARITY kontrolü...")
        self._check_parity(df)
        
        # 6. A = Z + N kontrolü
        logger.info("\n6. A = Z + N kontrolü...")
        self._check_azn_consistency(df)
        
        # 7. MM=0 tek-A kontrolü
        logger.info("\n7. MM=0 tek-A kontrolü...")
        self._check_mm_zero_odd_a(df)
        
        # 8. QM (Q) eksiklik kontrolü
        logger.info("\n8. QM (Q) eksiklik kontrolü...")
        self._check_qm_missing(df)
        
        # 9. İstatistiksel özetler
        logger.info("\n9. İstatistiksel özetler...")
        stats = self._calculate_statistics(df)
        
        # 10. Excel raporu oluştur
        logger.info("\n10. Excel raporu oluşturuluyor...")
        self._create_excel_report(df, stats)
        
        logger.info("\n" + "="*80)
        logger.info("VERİ KALİTE KONTROLÜ TAMAMLANDI")
        logger.info("="*80)
        
        return df, self.issues, stats
    
    def _check_missing_values(self, df):
        """Eksik değerleri tespit et"""
        for col in df.columns:
            if col == 'NUCLEUS':
                continue
                
            missing = df[df[col].isna()]
            if len(missing) > 0:
                for idx, row in missing.iterrows():
                    self.issues['missing_values'].append({
                        'index': idx,
                        'nucleus': row['NUCLEUS'],
                        'column': col,
                        'issue': f'Missing value in {col}'
                    })
        
        logger.info(f"   -> Eksik değer sayısı: {len(self.issues['missing_values'])}")
    
    def _check_zero_values(self, df):
        """MM sütununda sıfır değerleri tespit et"""
        if 'MM' in df.columns or 'MAGNETIC MOMENT [µ]' in df.columns:
            mm_col = 'MM' if 'MM' in df.columns else 'MAGNETIC MOMENT [µ]'
            zero_mm = df[df[mm_col] == 0]
            
            for idx, row in zero_mm.iterrows():
                self.issues['zero_values'].append({
                    'index': idx,
                    'nucleus': row['NUCLEUS'],
                    'column': mm_col,
                    'A': row['A'],
                    'issue': f'MM=0 (A={row["A"]}, odd-A check required)'
                })
        
        logger.info(f"   -> MM=0 sayısı: {len(self.issues['zero_values'])}")
    
    def _check_text_and_dates(self, df):
        """Sayısal sütunlarda metin/tarih kontrolü"""
        numeric_cols = ['A', 'Z', 'N', 'SPIN', 'PARITY', 'P-factor', 'Beta_2', 
                       'MAGNETIC MOMENT [µ]', 'QUADRUPOLE MOMENT [Q]', 'Nn', 'Np']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            # String kontrolü
            for idx, val in df[col].items():
                if pd.isna(val):
                    continue
                    
                val_str = str(val).strip()
                
                # Metin kontrolü (nokta, virgül, eksi hariç)
                if any(c.isalpha() for c in val_str):
                    # Özel durumlar: 'Nis.' gibi değerler
                    if 'Nis' in val_str or 'âˆ' in val_str:
                        self.issues['text_in_numeric'].append({
                            'index': idx,
                            'nucleus': df.loc[idx, 'NUCLEUS'],
                            'column': col,
                            'value': val_str,
                            'issue': f'Text found in {col}: "{val_str}"'
                        })
        
        logger.info(f"   -> Metin içeren sayısal veri: {len(self.issues['text_in_numeric'])}")
    
    def _check_parity(self, df):
        """PARITY değerlerini kontrol et (sadece -1 ve +1 olmalı)"""
        if 'PARITY' not in df.columns:
            return
        
        invalid_parity = df[~df['PARITY'].isin([-1, 1])]
        
        for idx, row in invalid_parity.iterrows():
            self.issues['parity_errors'].append({
                'index': idx,
                'nucleus': row['NUCLEUS'],
                'column': 'PARITY',
                'value': row['PARITY'],
                'issue': f'Invalid PARITY: {row["PARITY"]} (must be -1 or +1)'
            })
        
        logger.info(f"   -> Geçersiz PARITY: {len(self.issues['parity_errors'])}")
    
    def _check_azn_consistency(self, df):
        """A = Z + N kontrolü"""
        if not all(col in df.columns for col in ['A', 'Z', 'N']):
            return
        
        df_temp = df.copy()
        df_temp['A_calc'] = df_temp['Z'] + df_temp['N']
        mismatches = df_temp[df_temp['A'] != df_temp['A_calc']]
        
        for idx, row in mismatches.iterrows():
            self.issues['azn_mismatch'].append({
                'index': idx,
                'nucleus': row['NUCLEUS'],
                'A': row['A'],
                'Z': row['Z'],
                'N': row['N'],
                'A_calculated': row['A_calc'],
                'issue': f'A≠Z+N (A={row["A"]}, Z+N={row["A_calc"]})'
            })
        
        logger.info(f"   -> A≠Z+N uyuşmazlığı: {len(self.issues['azn_mismatch'])}")
    
    def _check_mm_zero_odd_a(self, df):
        """Tek-A çekirdekler için MM=0 kontrolü"""
        if not all(col in df.columns for col in ['A', 'MM']):
            mm_col = 'MAGNETIC MOMENT [µ]' if 'MAGNETIC MOMENT [µ]' in df.columns else None
            if mm_col is None:
                return
        else:
            mm_col = 'MM'
        
        df_temp = df.copy()
        df_temp['is_odd_A'] = df_temp['A'] % 2 == 1
        
        invalid_mm = df_temp[(df_temp['is_odd_A']) & (df_temp[mm_col] == 0)]
        
        for idx, row in invalid_mm.iterrows():
            self.issues['mm_zero_odd_a'].append({
                'index': idx,
                'nucleus': row['NUCLEUS'],
                'A': row['A'],
                'MM': row[mm_col],
                'issue': f'MM=0 for odd-A nucleus (A={row["A"]}, physically inconsistent)'
            })
        
        logger.info(f"   -> MM=0 tek-A: {len(self.issues['mm_zero_odd_a'])}")
    
    def _check_qm_missing(self, df):
        """Q (Quadrupole Moment) eksik değerleri tespit et"""
        qm_col = 'Q' if 'Q' in df.columns else 'QUADRUPOLE MOMENT [Q]'
        
        if qm_col not in df.columns:
            return
        
        missing_qm = df[df[qm_col].isna()]
        
        for idx, row in missing_qm.iterrows():
            self.issues['qm_missing'].append({
                'index': idx,
                'nucleus': row['NUCLEUS'],
                'A': row['A'],
                'Z': row['Z'],
                'N': row['N'],
                'issue': 'QM (Q) value missing'
            })
        
        logger.info(f"   -> QM eksik: {len(self.issues['qm_missing'])}")
    
    def _calculate_statistics(self, df):
        """Veri seti istatistikleri"""
        stats = {
            'total_nuclei': len(df),
            'A_range': (df['A'].min(), df['A'].max()),
            'Z_range': (df['Z'].min(), df['Z'].max()),
            'N_range': (df['N'].min(), df['N'].max()),
            'mm_available': df['MM'].notna().sum() if 'MM' in df.columns else 0,
            'qm_available': df['Q'].notna().sum() if 'Q' in df.columns else 0,
            'beta2_available': df['Beta_2'].notna().sum() if 'Beta_2' in df.columns else 0,
            'parity_positive': (df['PARITY'] == 1).sum() if 'PARITY' in df.columns else 0,
            'parity_negative': (df['PARITY'] == -1).sum() if 'PARITY' in df.columns else 0,
        }
        
        logger.info(f"   [OK] Toplam çekirdek: {stats['total_nuclei']}")
        logger.info(f"   [OK] A aralığı: {stats['A_range']}")
        logger.info(f"   [OK] MM mevcut: {stats['mm_available']} ({stats['mm_available']/stats['total_nuclei']*100:.1f}%)")
        logger.info(f"   [OK] QM mevcut: {stats['qm_available']} ({stats['qm_available']/stats['total_nuclei']*100:.1f}%)")
        
        return stats
    
    def _create_excel_report(self, df, stats):
        """Detaylı Excel raporu oluştur"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'aaa2_quality_report_{timestamp}.xlsx'
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Sheet 1: Özet
            summary_data = {
                'Metrik': [
                    'Toplam Çekirdek',
                    'A Aralığı',
                    'Z Aralığı',
                    'N Aralığı',
                    'MM Mevcut',
                    'QM Mevcut',
                    'Beta_2 Mevcut',
                    'Parite (+1)',
                    'Parite (-1)',
                    'Toplam Sorun Sayısı'
                ],
                'Değer': [
                    stats['total_nuclei'],
                    f"{stats['A_range'][0]} - {stats['A_range'][1]}",
                    f"{stats['Z_range'][0]} - {stats['Z_range'][1]}",
                    f"{stats['N_range'][0]} - {stats['N_range'][1]}",
                    f"{stats['mm_available']} ({stats['mm_available']/stats['total_nuclei']*100:.1f}%)",
                    f"{stats['qm_available']} ({stats['qm_available']/stats['total_nuclei']*100:.1f}%)",
                    f"{stats['beta2_available']} ({stats['beta2_available']/stats['total_nuclei']*100:.1f}%)",
                    stats['parity_positive'],
                    stats['parity_negative'],
                    sum(len(v) for v in self.issues.values())
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2-10: Her sorun tipi için ayrı sheet
            for issue_type, issue_list in self.issues.items():
                if len(issue_list) > 0:
                    issue_df = pd.DataFrame(issue_list)
                    sheet_name = issue_type[:31]  # Excel limit
                    issue_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet 11: Tam veri
            df.to_excel(writer, sheet_name='Full_Data', index=False)
        
        logger.info(f"   [OK] Excel raporu kaydedildi: {report_path}")
        return report_path


def main():
    """Test fonksiyonu"""
    checker = AAA2DataQualityChecker()
    df, issues, stats = checker.load_and_check('aaa2.txt')
    
    print("\n" + "="*80)
    print("SORUN ÖZETİ")
    print("="*80)
    for issue_type, issue_list in issues.items():
        if len(issue_list) > 0:
            print(f"{issue_type}: {len(issue_list)} sorun")
    print("="*80)


if __name__ == "__main__":
    main()
