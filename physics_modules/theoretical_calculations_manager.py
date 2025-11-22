"""
Teorik Hesaplama Yöneticisi
Theoretical Calculations Manager

Bu modül, tüm teorik fizik hesaplamalarını koordine eder:
- SEMF (Semi-Empirical Mass Formula)
- Shell Model (Kabuk Modeli)
- Woods-Saxon Potential
- Nilsson Model (Deformed nuclei)
- Collective Model (Kollektif Model)
- Schmidt Moments
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TheoreticalCalculationsManager:
    """Tüm teorik hesaplamaları koordine eden merkezi sistem"""
    
    def __init__(self, enable_all=True):
        """
        Args:
            enable_all: Tüm hesaplamaları etkinleştir
        """
        self.enable_all = enable_all
        self.calculations_done = {
            'semf': False,
            'shell_model': False,
            'woods_saxon': False,
            'nilsson': False,
            'collective': False,
            'schmidt': False,
            'deformation': False
        }
        
        logger.info("Teorik Hesaplama Yöneticisi başlatıldı")
    
    def calculate_all_theoretical_properties(self, df, save_report=True):
        """
        Tüm teorik özellikleri hesapla
        
        Args:
            df: DataFrame - Temizlenmiş veri
            save_report: bool - Hesaplama raporunu kaydet
            
        Returns:
            DataFrame: Teorik özelliklerle zenginleştirilmiş veri
        """
        logger.info("="*80)
        logger.info("TEORİK HESAPLAMALAR BAŞLIYOR")
        logger.info("="*80)
        
        df_enriched = df.copy()
        start_time = datetime.now()
        
        # 1. SEMF Hesaplamaları
        logger.info("\n1. SEMF (Semi-Empirical Mass Formula) Hesaplamaları...")
        df_enriched = self._calculate_semf(df_enriched)
        self.calculations_done['semf'] = True
        
        # 2. Shell Model Hesaplamaları
        logger.info("\n2. Shell Model (Kabuk Modeli) Hesaplamaları...")
        df_enriched = self._calculate_shell_model(df_enriched)
        self.calculations_done['shell_model'] = True
        
        # 3. Deformation Hesaplamaları
        logger.info("\n3. Deformation (Deformasyon) Hesaplamaları...")
        df_enriched = self._calculate_deformation(df_enriched)
        self.calculations_done['deformation'] = True
        
        # 4. Schmidt Moment Hesaplamaları
        logger.info("\n4. Schmidt Moment Hesaplamaları...")
        df_enriched = self._calculate_schmidt_moments(df_enriched)
        self.calculations_done['schmidt'] = True
        
        # 5. Collective Model Hesaplamaları
        logger.info("\n5. Collective Model (Kollektif Model) Hesaplamaları...")
        df_enriched = self._calculate_collective_model(df_enriched)
        self.calculations_done['collective'] = True
        
        # 6. Woods-Saxon (opsiyonel - hesaplama yoğun)
        if self.enable_all:
            logger.info("\n6. Woods-Saxon Potansiyel Hesaplamaları...")
            df_enriched = self._calculate_woods_saxon(df_enriched)
            self.calculations_done['woods_saxon'] = True
        
        # 7. Nilsson Model (opsiyonel - sadece deformed nuclei için)
        if self.enable_all:
            logger.info("\n7. Nilsson Model Hesaplamaları (Deformed Nuclei)...")
            df_enriched = self._calculate_nilsson(df_enriched)
            self.calculations_done['nilsson'] = True
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Özet
        logger.info("\n" + "="*80)
        logger.info("TEORİK HESAPLAMALAR TAMAMLANDI")
        logger.info("="*80)
        logger.info(f"Süre: {duration:.2f} saniye")
        logger.info(f"Başlangıç özellik sayısı: {len(df.columns)}")
        logger.info(f"Son özellik sayısı: {len(df_enriched.columns)}")
        logger.info(f"Eklenen özellik sayısı: {len(df_enriched.columns) - len(df.columns)}")
        
        # Rapor kaydet
        if save_report:
            self._save_calculation_report(df, df_enriched, duration)
        
        return df_enriched
    
    def _calculate_semf(self, df):
        """SEMF hesaplamaları"""
        from constants import SEMF_PARAMS, R0
        
        # Temel parametreler
        a_v = SEMF_PARAMS['a_v']
        a_s = SEMF_PARAMS['a_s']
        a_c = SEMF_PARAMS['a_c']
        a_a = SEMF_PARAMS['a_a']
        a_p = SEMF_PARAMS['a_p']
        
        # Binding Energy terimleri
        df['BE_volume'] = a_v * df['A']
        df['BE_surface'] = -a_s * (df['A'] ** (2/3))
        df['BE_coulomb'] = -a_c * (df['Z']**2) / (df['A'] ** (1/3))
        df['BE_asymmetry'] = -a_a * ((df['N'] - df['Z'])**2) / df['A']
        
        # Pairing term
        def pairing_delta(row):
            if row['Z'] % 2 == 0 and row['N'] % 2 == 0:
                return a_p / np.sqrt(row['A'])  # even-even
            elif row['Z'] % 2 == 1 and row['N'] % 2 == 1:
                return -a_p / np.sqrt(row['A'])  # odd-odd
            else:
                return 0.0  # even-odd or odd-even
        
        df['BE_pairing'] = df.apply(pairing_delta, axis=1)
        
        # Total binding energy
        df['BE_total'] = (df['BE_volume'] + df['BE_surface'] + 
                         df['BE_coulomb'] + df['BE_asymmetry'] + df['BE_pairing'])
        
        # Binding energy per nucleon
        df['BE_per_A'] = df['BE_total'] / df['A']
        
        # Nuclear radius
        df['nuclear_radius'] = R0 * (df['A'] ** (1/3))
        
        # Separation energies (simplified - requires mass table)
        df['S_n_approx'] = 8.0 + 0.5 * (df['N'] - df['Z']) / df['A']  # Approximate
        df['S_p_approx'] = 8.0 - 0.5 * (df['N'] - df['Z']) / df['A']  # Approximate
        
        logger.info(f"  [OK] SEMF: 11 özellik eklendi")
        return df
    
    def _calculate_shell_model(self, df):
        """Shell Model hesaplamaları"""
        from constants import MAGIC_NUMBERS_Z, MAGIC_NUMBERS_N, SHELL_GAPS
        
        # En yakın magic numbers
        df['Z_nearest_magic'] = df['Z'].apply(
            lambda z: min(MAGIC_NUMBERS_Z, key=lambda x: abs(x - z))
        )
        df['N_nearest_magic'] = df['N'].apply(
            lambda n: min(MAGIC_NUMBERS_N, key=lambda x: abs(x - n))
        )
        
        # Magic number'lara uzaklık
        df['Z_magic_dist'] = abs(df['Z'] - df['Z_nearest_magic'])
        df['N_magic_dist'] = abs(df['N'] - df['N_nearest_magic'])
        
        # Shell gaps
        df['Z_shell_gap'] = df['Z_nearest_magic'].apply(
            lambda z: SHELL_GAPS.get(z, 5.0)
        )
        df['N_shell_gap'] = df['N_nearest_magic'].apply(
            lambda n: SHELL_GAPS.get(n, 5.0)
        )
        
        # Valence nucleons
        df['Z_valence'] = df.apply(
            lambda row: min(abs(row['Z'] - row['Z_nearest_magic']),
                          abs(row['Z_nearest_magic'] - row['Z'])), axis=1
        )
        df['N_valence'] = df.apply(
            lambda row: min(abs(row['N'] - row['N_nearest_magic']),
                          abs(row['N_nearest_magic'] - row['N'])), axis=1
        )
        
        # Magic character (0 = very magic, 1 = far from magic)
        df['magic_character'] = 1.0 / (1.0 + df['Z_magic_dist'] + df['N_magic_dist'])
        
        # Double magic
        df['is_doubly_magic'] = ((df['Z_magic_dist'] == 0) & 
                                 (df['N_magic_dist'] == 0)).astype(int)
        
        # Single magic
        df['is_singly_magic'] = (((df['Z_magic_dist'] == 0) | 
                                  (df['N_magic_dist'] == 0)) & 
                                 (df['is_doubly_magic'] == 0)).astype(int)
        
        logger.info(f"  [OK] Shell Model: 11 özellik eklendi")
        return df
    
    def _calculate_deformation(self, df):
        """Deformation hesaplamaları"""
        from constants import R0, SPHERICAL_THRESHOLD
        
        # β₂ varsa kullan, yoksa tahmin et
        if 'Beta_2' not in df.columns or df['Beta_2'].isna().all():
            # Tahmin: Shell model'e göre
            df['Beta_2_estimated'] = df.apply(
                lambda row: self._estimate_beta2(row['Z'], row['N']), axis=1
            )
        else:
            df['Beta_2_estimated'] = df['Beta_2']
        
        # Deformation type
        def classify_deformation(beta2):
            if pd.isna(beta2):
                return 'unknown'
            elif abs(beta2) < SPHERICAL_THRESHOLD:
                return 'spherical'
            elif beta2 > 0.35:
                return 'strongly_prolate'
            elif beta2 > 0.15:
                return 'prolate'
            elif beta2 > 0.05:
                return 'weakly_prolate'
            elif beta2 < -0.35:
                return 'strongly_oblate'
            elif beta2 < -0.15:
                return 'oblate'
            else:
                return 'weakly_oblate'
        
        df['deformation_type'] = df['Beta_2_estimated'].apply(classify_deformation)
        
        # Intrinsic quadrupole moment (Q₀)
        df['Q0_intrinsic'] = (3.0 / np.sqrt(5 * np.pi)) * df['Z'] * \
                             (R0 * df['A']**(1/3))**2 * df['Beta_2_estimated']
        
        # Rotational parameter
        df['rotational_param'] = (HBAR_C**2) / (2 * 0.5 * 1.2**2 * df['A']**(5/3))
        
        logger.info(f"  [OK] Deformation: 4 özellik eklendi")
        return df
    
    def _calculate_schmidt_moments(self, df):
        """Schmidt moment hesaplamaları"""
        from constants import G_FACTORS
        
        g_l_p = G_FACTORS['proton']['g_l']
        g_s_p = G_FACTORS['proton']['g_s']
        g_l_n = G_FACTORS['neutron']['g_l']
        g_s_n = G_FACTORS['neutron']['g_s']
        
        # Odd-mass nuclei için Schmidt moments
        def schmidt_moment(row):
            I = row['SPIN']
            parity = row['PARITY']
            A = row['A']
            Z = row['Z']
            N = row['N']
            
            if I == 0:
                return 0.0  # Even-even
            
            # Odd proton
            if Z % 2 == 1 and N % 2 == 0:
                if parity == 1:  # j = l + 1/2
                    l = I - 0.5
                    mu = g_l_p * l + g_s_p * 0.5
                else:  # j = l - 1/2
                    l = I + 0.5
                    mu = (g_l_p * (I + 1) - g_s_p * 0.5) * I / (I + 1)
                return mu
            
            # Odd neutron
            elif Z % 2 == 0 and N % 2 == 1:
                if parity == 1:  # j = l + 1/2
                    l = I - 0.5
                    mu = g_l_n * l + g_s_n * 0.5
                else:  # j = l - 1/2
                    l = I + 0.5
                    mu = (g_l_n * (I + 1) - g_s_n * 0.5) * I / (I + 1)
                return mu
            
            return np.nan
        
        df['schmidt_moment'] = df.apply(schmidt_moment, axis=1)
        
        # Schmidt deviation (if experimental MM available)
        if 'MM' in df.columns:
            df['schmidt_deviation'] = abs(df['MM'] - df['schmidt_moment'])
            df['schmidt_quenching'] = df['MM'] / df['schmidt_moment'].replace(0, np.nan)
        
        logger.info(f"  [OK] Schmidt: 1-3 özellik eklendi")
        return df
    
    def _calculate_collective_model(self, df):
        """Collective Model (Kollektif Model) hesaplamaları"""
        
        # Moment of inertia (rigid rotor)
        df['moment_of_inertia'] = 0.5 * 1.2**2 * df['A']**(5/3) / (HBAR_C**2)
        
        # Rotational energy (E = ℏ²/(2I) * J(J+1))
        df['E_2plus'] = (HBAR_C**2) / (2 * df['moment_of_inertia']) * 2 * 3  # J=2
        
        # Vibrational frequency
        df['vib_frequency'] = np.sqrt(80.0 / df['A'])  # MeV
        
        # Vibrator vs Rotor classification
        def classify_nucleus_type(row):
            if row['deformation_type'] == 'spherical':
                return 'vibrator'
            elif 'prolate' in row['deformation_type'] or 'oblate' in row['deformation_type']:
                return 'rotor'
            else:
                return 'transitional'
        
        df['nucleus_collective_type'] = df.apply(classify_nucleus_type, axis=1)
        
        logger.info(f"  [OK] Collective Model: 4 özellik eklendi")
        return df
    
    def _calculate_woods_saxon(self, df):
        """Woods-Saxon potansiyel hesaplamaları (simplified)"""
        from constants import WOODS_SAXON_PARAMS
        
        V0 = WOODS_SAXON_PARAMS['V0']
        r0 = WOODS_SAXON_PARAMS['r0']
        
        # Surface diffuseness etkisi
        df['ws_surface_thick'] = WOODS_SAXON_PARAMS['a']
        
        # Fermi energy (approximate)
        df['fermi_energy'] = 33.0 * (df['A'] ** (2/3)) / (r0**2 * df['A'])
        
        logger.info(f"  [OK] Woods-Saxon: 2 özellik eklendi")
        return df
    
    def _calculate_nilsson(self, df):
        """Nilsson model hesaplamaları (sadece deformed nuclei için)"""
        
        # Sadece deformed çekirdekler için
        deformed_mask = df['Beta_2_estimated'].abs() > 0.15
        
        # Deformation parameter
        df.loc[deformed_mask, 'nilsson_epsilon'] = 0.95 * df.loc[deformed_mask, 'Beta_2_estimated']
        
        # Oscillator frequency
        df.loc[deformed_mask, 'nilsson_omega'] = 41.0 / (df.loc[deformed_mask, 'A'] ** (1/3))
        
        logger.info(f"  [OK] Nilsson: 2 özellik eklendi (sadece deformed nuclei)")
        return df
    
    def _estimate_beta2(self, Z, N):
        """β₂ tahmini (shell model bazlı)"""
        # Magic nuclei'den uzaklık
        magic_z_dist = min([abs(Z - m) for m in [2, 8, 20, 28, 50, 82]])
        magic_n_dist = min([abs(N - m) for m in [2, 8, 20, 28, 50, 82, 126]])
        
        # Uzaksa daha deformed
        if magic_z_dist < 2 and magic_n_dist < 2:
            return 0.0  # Spherical (magic)
        elif magic_z_dist > 10 or magic_n_dist > 10:
            return 0.3  # Highly deformed
        else:
            return 0.15  # Moderately deformed
    
    def _save_calculation_report(self, df_original, df_enriched, duration):
        """Hesaplama raporunu kaydet"""
        report_dir = Path('reports/theoretical_calculations')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'theoretical_calc_report_{timestamp}.json'
        
        report = {
            'timestamp': timestamp,
            'duration_seconds': duration,
            'original_features': len(df_original.columns),
            'enriched_features': len(df_enriched.columns),
            'added_features': len(df_enriched.columns) - len(df_original.columns),
            'new_features': list(set(df_enriched.columns) - set(df_original.columns)),
            'calculations_done': self.calculations_done,
            'sample_statistics': {
                'BE_per_A_mean': float(df_enriched['BE_per_A'].mean()),
                'magic_character_mean': float(df_enriched['magic_character'].mean()),
                'doubly_magic_count': int(df_enriched['is_doubly_magic'].sum())
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n[OK] Hesaplama raporu kaydedildi: {report_file}")


# Global constants (imported from constants.py)
HBAR_C = 197.3269804  # MeV·fm


def main():
    """Test fonksiyonu"""
    # Test data
    test_data = pd.DataFrame({
        'NUCLEUS': ['Pb-208', 'O-16', 'Fe-56'],
        'A': [208, 16, 56],
        'Z': [82, 8, 26],
        'N': [126, 8, 30],
        'SPIN': [0, 0, 0],
        'PARITY': [1, 1, 1],
        'MM': [0, 0, 0],
        'Q': [0, 0, 0.16],
        'Beta_2': [0, 0, 0.1]
    })
    
    # Hesaplama yöneticisi
    calc_manager = TheoreticalCalculationsManager(enable_all=True)
    
    # Tüm hesaplamaları yap
    enriched_data = calc_manager.calculate_all_theoretical_properties(test_data)
    
    print("\n" + "="*80)
    print("ÖRNEKDosyaSonuçlar")
    print("="*80)
    print(enriched_data[['NUCLEUS', 'BE_per_A', 'magic_character', 
                         'deformation_type', 'is_doubly_magic']].to_string())


if __name__ == "__main__":
    main()
