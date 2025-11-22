"""
Nükleer Veri Yükleme ve Temizleme Modülü
Nuclear Data Loading and Cleaning Module

YENİ ÖZELLİK:
- Removed Nuclei Log: Ayıklanan çekirdekleri detaylı kayıt
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.file_io_utils import read_nuclear_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NuclearDataLoader:
    """Nükleer veri yükleyici ve temizleyici"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.removed_nuclei = []  # Ayıklanan çekirdekler
    
    def load_data(self, filepath='aaa2.txt'):
        """
        Ham veriyi yükle ve temizle
        
        Returns:
            pd.DataFrame: Temizlenmiş veri
        """
        logger.info(f"Veri yükleniyor: {filepath}")

        # Veriyi oku (esnek format desteği)
        df = read_nuclear_data(filepath, encoding='utf-8')
        
        logger.info(f"[OK] Ham veri yüklendi: {len(df)} satır")
        
        # Temizlik işlemleri
        df_cleaned = self._clean_data(df)
        
        # Temizlenmiş veriyi kaydet
        output_file = self.output_dir / 'cleaned_nuclear_data.csv'
        df_cleaned.to_csv(output_file, index=False)
        logger.info(f"[OK] Temizlenmiş veri kaydedildi: {output_file}")
        
        # Ayıklanan çekirdekleri kaydet
        self._save_removed_nuclei()
        
        return df_cleaned
    
    def _clean_data(self, df):
        """
        Veriyi temizle ve geçersiz satırları kaydet
        
        Temizlik kuralları:
        1. NUMERIC COLUMNS: A, Z, N, SPIN, PARITY, MM, Q sayısal olmalı
        2. PARITY: {-1, +1} olmalı
        3. PHYSICAL CONSTRAINT: A = Z + N
        4. MM=0 RULE: Tek-A çekirdekler için MM=0 fiziksel olarak tutarsız
        5. EVEN-EVEN EXCLUSION: Çift-çift çekirdekler veri setinde yok
        """
        df_original = df.copy()
        initial_count = len(df)
        
        logger.info("\n" + "="*70)
        logger.info("VERİ TEMİZLEME BAŞLIYOR")
        logger.info("="*70)
        
        # 1. Sütun isimlerini düzenle (boşluk temizle)
        df.columns = df.columns.str.strip()
        
        # 2. Numeric conversion
        numeric_cols = ['A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR', 'Beta_2', 'MM', 'Q']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. PARITY kontrolü
        removed_parity = df[~df['PARITY'].isin([-1, 1])].copy()
        if len(removed_parity) > 0:
            for idx, row in removed_parity.iterrows():
                self._record_removal(row, f"Invalid PARITY ({row['PARITY']})")
            df = df[df['PARITY'].isin([-1, 1])]
            logger.info(f"  -> {len(removed_parity)} çekirdek kaldırıldı (Geçersiz PARITY)")
        
        # 4. A = Z + N kontrolü
        df['A_calculated'] = df['Z'] + df['N']
        removed_azn = df[df['A'] != df['A_calculated']].copy()
        if len(removed_azn) > 0:
            for idx, row in removed_azn.iterrows():
                self._record_removal(row, f"A≠Z+N (A={row['A']}, Z+N={row['A_calculated']})")
            df = df[df['A'] == df['A_calculated']]
            logger.info(f"  -> {len(removed_azn)} çekirdek kaldırıldı (A≠Z+N)")
        df.drop('A_calculated', axis=1, inplace=True)
        
        # 5. MM=0 kontrolü (Tek-A için)
        df['is_odd_A'] = df['A'] % 2 == 1
        removed_mm0 = df[(df['is_odd_A']) & (df['MM'] == 0)].copy()
        if len(removed_mm0) > 0:
            for idx, row in removed_mm0.iterrows():
                self._record_removal(row, f"MM=0 for odd-A nucleus (physically inconsistent)")
            df = df[~((df['is_odd_A']) & (df['MM'] == 0))]
            logger.info(f"  -> {len(removed_mm0)} çekirdek kaldırıldı (MM=0 tek-A için geçersiz)")
        df.drop('is_odd_A', axis=1, inplace=True)
        
        # 6. Çift-çift kontrol (opsiyonel log)
        even_even = df[(df['Z'] % 2 == 0) & (df['N'] % 2 == 0)].copy()
        if len(even_even) > 0:
            for idx, row in even_even.iterrows():
                self._record_removal(row, f"Even-even nucleus (Z={row['Z']}, N={row['N']})")
            df = df[~((df['Z'] % 2 == 0) & (df['N'] % 2 == 0))]
            logger.info(f"  -> {len(even_even)} çekirdek kaldırıldı (Çift-çift)")
        
        # 7. NaN kontrolü (kritik sütunlar)
        critical_cols = ['A', 'Z', 'N', 'SPIN', 'PARITY']
        removed_nan = df[df[critical_cols].isna().any(axis=1)].copy()
        if len(removed_nan) > 0:
            for idx, row in removed_nan.iterrows():
                nan_cols = row[critical_cols][row[critical_cols].isna()].index.tolist()
                self._record_removal(row, f"Missing critical data: {', '.join(nan_cols)}")
            df = df[~df[critical_cols].isna().any(axis=1)]
            logger.info(f"  -> {len(removed_nan)} çekirdek kaldırıldı (Kritik sütunlarda NaN)")
        
        # 8. Ondalık ayırıcı düzeltme (virgül -> nokta)
        for col in ['Beta_2', 'MM', 'Q']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        logger.info("="*70)
        logger.info(f"TEMİZLEME TAMAMLANDI")
        logger.info(f"  Başlangıç: {initial_count} çekirdek")
        logger.info(f"  Kalan: {final_count} çekirdek")
        logger.info(f"  Kaldırılan: {removed_count} çekirdek ({removed_count/initial_count*100:.1f}%)")
        logger.info("="*70 + "\n")
        
        return df
    
    def _record_removal(self, row, reason):
        """Ayıklanan çekirdeği kaydet"""
        record = {
            'NUCLEUS': row.get('NUCLEUS', 'Unknown'),
            'A': row.get('A', np.nan),
            'Z': row.get('Z', np.nan),
            'N': row.get('N', np.nan),
            'SPIN': row.get('SPIN', np.nan),
            'PARITY': row.get('PARITY', np.nan),
            'MM': row.get('MM', np.nan),
            'Q': row.get('Q', np.nan),
            'Beta_2': row.get('Beta_2', np.nan),
            'Reason': reason,
            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.removed_nuclei.append(record)
    
    def _save_removed_nuclei(self):
        """Ayıklanan çekirdekleri Excel'e kaydet"""
        if not self.removed_nuclei:
            logger.info("[OK] Ayıklanan çekirdek yok")
            return
        
        df_removed = pd.DataFrame(self.removed_nuclei)
        output_file = self.output_dir / 'removed_nuclei.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Tüm ayıklananlar
            df_removed.to_excel(writer, sheet_name='All_Removed', index=False)
            
            # Sheet 2: Nedene göre özet
            reason_summary = df_removed['Reason'].value_counts().reset_index()
            reason_summary.columns = ['Reason', 'Count']
            reason_summary.to_excel(writer, sheet_name='Reason_Summary', index=False)
            
            # Sheet 3: Z'ye göre dağılım
            if 'Z' in df_removed.columns:
                z_summary = df_removed.groupby('Z').size().reset_index(name='Count')
                z_summary.to_excel(writer, sheet_name='By_Proton_Number', index=False)
        
        logger.info(f"[OK] Ayıklanan çekirdekler kaydedildi: {output_file}")
        logger.info(f"  Toplam: {len(self.removed_nuclei)} çekirdek")
        
        # En yaygın nedenleri göster
        reason_counts = df_removed['Reason'].value_counts()
        logger.info("\n  En yaygın ayıklanma nedenleri:")
        for reason, count in reason_counts.head(5).items():
            logger.info(f"    • {reason}: {count}")
    
    def get_data_statistics(self, df):
        """Veri istatistikleri"""
        stats = {
            'total_nuclei': len(df),
            'proton_range': (int(df['Z'].min()), int(df['Z'].max())),
            'neutron_range': (int(df['N'].min()), int(df['N'].max())),
            'mass_range': (int(df['A'].min()), int(df['A'].max())),
            'spin_range': (df['SPIN'].min(), df['SPIN'].max()),
            'mm_available': df['MM'].notna().sum(),
            'q_available': df['Q'].notna().sum(),
            'beta2_available': df['Beta_2'].notna().sum(),
            'parity_positive': (df['PARITY'] == 1).sum(),
            'parity_negative': (df['PARITY'] == -1).sum()
        }
        
        logger.info("\n" + "="*70)
        logger.info("VERİ İSTATİSTİKLERİ")
        logger.info("="*70)
        logger.info(f"Toplam çekirdek: {stats['total_nuclei']}")
        logger.info(f"Proton sayısı (Z): {stats['proton_range'][0]} - {stats['proton_range'][1]}")
        logger.info(f"Nötron sayısı (N): {stats['neutron_range'][0]} - {stats['neutron_range'][1]}")
        logger.info(f"Kütle sayısı (A): {stats['mass_range'][0]} - {stats['mass_range'][1]}")
        logger.info(f"Spin aralığı: {stats['spin_range'][0]} - {stats['spin_range'][1]}")
        logger.info(f"\nMevcut veriler:")
        logger.info(f"  MM (Magnetik Moment): {stats['mm_available']} ({stats['mm_available']/stats['total_nuclei']*100:.1f}%)")
        logger.info(f"  Q (Kuadrupol Moment): {stats['q_available']} ({stats['q_available']/stats['total_nuclei']*100:.1f}%)")
        logger.info(f"  Beta_2 (Deformasyon): {stats['beta2_available']} ({stats['beta2_available']/stats['total_nuclei']*100:.1f}%)")
        logger.info(f"\nParite dağılımı:")
        logger.info(f"  Pozitif (+1): {stats['parity_positive']}")
        logger.info(f"  Negatif (-1): {stats['parity_negative']}")
        logger.info("="*70 + "\n")
        
        return stats


def main():
    """Test fonksiyonu"""
    loader = NuclearDataLoader('test_output')
    
    # Veriyi yükle ve temizle
    df = loader.load_data('aaa2.txt')
    
    # İstatistikleri göster
    stats = loader.get_data_statistics(df)
    
    print("\n[OK] Data loading test completed")
    print(f"  Cleaned data: {len(df)} nuclei")
    print(f"  Removed: {len(loader.removed_nuclei)} nuclei")


if __name__ == "__main__":
    main()