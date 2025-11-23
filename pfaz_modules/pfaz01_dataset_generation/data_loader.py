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

# Configure logging with UTF-8 encoding to handle Turkish characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Set UTF-8 encoding for stdout if possible (for Windows compatibility)
if getattr(sys.stdout, 'encoding', None) != 'utf-8':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    except (AttributeError, OSError):
        # If we can't set UTF-8, just continue - ASCII-safe messages will work
        pass

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
        initial_count = len(df)

        logger.info("\n" + "="*70)
        logger.info("VERİ TEMİZLEME BAŞLIYOR")
        logger.info("="*70)

        # 1. Sütun isimlerini normalize et (boşluk temizle)
        df.columns = df.columns.str.strip()
        logger.info("  -> Sütun isimleri normalize edildi")

        # 2. String temizleme - ÖNCE numeric conversion'dan ÖNCE yapılmalı!
        # Ondalık ayırıcı düzeltme (virgül -> nokta) ve unicode minus işareti düzeltme
        string_cols = ['Beta_2', 'MM', 'Q', 'SPIN', 'P_FACTOR']
        for col in string_cols:
            if col in df.columns and df[col].dtype == 'object':
                # Virgül -> nokta
                df[col] = df[col].astype(str).str.replace(',', '.')
                # Unicode minus (U+2212) -> ASCII minus (-)
                df[col] = df[col].str.replace('\u2212', '-')
                # 'nan' string değerlerini temizle
                df[col] = df[col].replace('nan', np.nan)

        logger.info("  -> String cleaning tamamlandi (virgul->nokta, unicode minus->ASCII)")

        # 3. Numeric conversion - string temizlemeden SONRA
        numeric_cols = ['A', 'Z', 'N', 'SPIN', 'PARITY', 'P_FACTOR', 'Beta_2', 'MM', 'Q']
        conversion_warnings = []

        for col in numeric_cols:
            if col in df.columns:
                original_values = df[col].copy()
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Non-numeric değerleri logla
                non_numeric_mask = pd.to_numeric(original_values, errors='coerce').isna() & original_values.notna()
                if non_numeric_mask.any():
                    non_numeric_count = non_numeric_mask.sum()
                    example_value = original_values[non_numeric_mask].iloc[0] if non_numeric_count > 0 else None
                    conversion_warnings.append((col, non_numeric_count, example_value))

        # Conversion uyarılarını göster
        for col, count, example in conversion_warnings:
            logger.warning(f"  -> '{col}' sütununda {count} non-numeric değer NaN'a dönüştürüldü")
            if example:
                logger.warning(f"     Örnek: '{example}'")

        # 4. PARITY kontrolü
        if 'PARITY' in df.columns:
            removed_parity = df[~df['PARITY'].isin([-1, 1])].copy()
            if len(removed_parity) > 0:
                for idx, row in removed_parity.iterrows():
                    self._record_removal(row, f"Invalid PARITY ({row['PARITY']})")
                df = df[df['PARITY'].isin([-1, 1])]
                logger.info(f"  -> {len(removed_parity)} çekirdek kaldırıldı (Geçersiz PARITY)")

        # 5. A = Z + N kontrolü
        if all(col in df.columns for col in ['A', 'Z', 'N']):
            df['A_calculated'] = df['Z'] + df['N']
            removed_azn = df[df['A'] != df['A_calculated']].copy()
            if len(removed_azn) > 0:
                for idx, row in removed_azn.iterrows():
                    self._record_removal(row, f"A≠Z+N (A={row['A']}, Z+N={row['A_calculated']})")
                df = df[df['A'] == df['A_calculated']]
                logger.info(f"  -> {len(removed_azn)} çekirdek kaldırıldı (A≠Z+N)")
            df.drop('A_calculated', axis=1, inplace=True)

        # 6. MM=0 kontrolü (Tek-A için) - sadece MM sütunu varsa
        if all(col in df.columns for col in ['A', 'MM']):
            df['is_odd_A'] = df['A'] % 2 == 1
            removed_mm0 = df[(df['is_odd_A']) & (df['MM'] == 0)].copy()
            if len(removed_mm0) > 0:
                for idx, row in removed_mm0.iterrows():
                    self._record_removal(row, f"MM=0 for odd-A nucleus (physically inconsistent)")
                df = df[~((df['is_odd_A']) & (df['MM'] == 0))]
                logger.info(f"  -> {len(removed_mm0)} çekirdek kaldırıldı (MM=0 tek-A için geçersiz)")
            df.drop('is_odd_A', axis=1, inplace=True)

        # 7. Çift-çift kontrol (opsiyonel log)
        if all(col in df.columns for col in ['Z', 'N']):
            even_even = df[(df['Z'] % 2 == 0) & (df['N'] % 2 == 0)].copy()
            if len(even_even) > 0:
                for idx, row in even_even.iterrows():
                    self._record_removal(row, f"Even-even nucleus (Z={row['Z']}, N={row['N']})")
                df = df[~((df['Z'] % 2 == 0) & (df['N'] % 2 == 0))]
                logger.info(f"  -> {len(even_even)} çekirdek kaldırıldı (Çift-çift)")

        # 8. NaN kontrolü (kritik sütunlar)
        critical_cols = ['A', 'Z', 'N', 'SPIN', 'PARITY']
        existing_critical_cols = [col for col in critical_cols if col in df.columns]

        if existing_critical_cols:
            removed_nan = df[df[existing_critical_cols].isna().any(axis=1)].copy()
            if len(removed_nan) > 0:
                for idx, row in removed_nan.iterrows():
                    nan_cols = row[existing_critical_cols][row[existing_critical_cols].isna()].index.tolist()
                    self._record_removal(row, f"Missing critical data: {', '.join(nan_cols)}")
                df = df[~df[existing_critical_cols].isna().any(axis=1)]
                logger.info(f"  -> {len(removed_nan)} çekirdek kaldırıldı (Kritik sütunlarda NaN)")

        final_count = len(df)
        removed_count = initial_count - final_count

        logger.info("="*70)
        logger.info(f"TEMİZLEME TAMAMLANDI")
        logger.info(f"  Baslangic: {initial_count} cekirdek")
        logger.info(f"  Kalan: {final_count} cekirdek")
        logger.info(f"  Kaldirilan: {removed_count} cekirdek ({removed_count/initial_count*100:.1f}%)")
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