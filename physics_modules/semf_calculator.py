"""
Yarı-Ampirik Kütle Formülü (SEMF) ve Teorik Hesaplamalar
Semi-Empirical Mass Formula and Theoretical Calculations
"""

import numpy as np
import pandas as pd
import logging

from .constants import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SEMFCalculator:
    """Yarı-Ampirik Kütle Formülü hesaplayıcı"""
    
    def __init__(self, params=None):
        """
        Args:
            params: SEMF parametreleri (dict), None ise default kullanılır
        """
        self.params = params if params else SEMF_PARAMS
    
    def calculate_binding_energy(self, A, Z, N):
        """
        Toplam bağlanma enerjisini hesapla
        
        BE = a_v·A - a_s·A^(2/3) - a_c·Z²/A^(1/3) - a_a·(N-Z)²/A + δ(A,Z)
        
        Args:
            A: Kütle numarası
            Z: Proton sayısı
            N: Nötron sayısı
            
        Returns:
            dict: Tüm bileşenler ve toplam BE
        """
        # Hacim terimi (Volume)
        BE_volume = self.params['a_v'] * A
        
        # Yüzey terimi (Surface)
        BE_surface = -self.params['a_s'] * (A ** (2/3))
        
        # Coulomb terimi
        BE_coulomb = -self.params['a_c'] * (Z ** 2) / (A ** (1/3))
        
        # Asimetri terimi (Asymmetry)
        BE_asymmetry = -self.params['a_a'] * ((N - Z) ** 2) / A
        
        # Çiftleşme terimi (Pairing)
        BE_pairing = self._calculate_pairing_term(A, Z, N)
        
        # Toplam bağlanma enerjisi
        BE_total = BE_volume + BE_surface + BE_coulomb + BE_asymmetry + BE_pairing
        
        # Nükleon başına bağlanma enerjisi
        BE_per_A = BE_total / A
        
        return {
            'BE_volume': BE_volume,
            'BE_surface': BE_surface,
            'BE_coulomb': BE_coulomb,
            'BE_asymmetry': BE_asymmetry,
            'BE_pairing': BE_pairing,
            'BE_total': BE_total,
            'BE_per_A': BE_per_A
        }
    
    def _calculate_pairing_term(self, A, Z, N):
        """
        Çiftleşme terimi: δ(A,Z)
        
        δ = +a_p/A^(1/2)  (even-even)
        δ = 0             (odd-A)
        δ = -a_p/A^(1/2)  (odd-odd)
        """
        if Z % 2 == 0 and N % 2 == 0:  # even-even
            return self.params['a_p'] / np.sqrt(A)
        elif Z % 2 == 1 and N % 2 == 1:  # odd-odd
            return -self.params['a_p'] / np.sqrt(A)
        else:  # odd-A
            return 0.0
    
    def calculate_separation_energies(self, A, Z, N):
        """
        Ayırma enerjilerini hesapla
        
        S_n = BE(A,Z) - BE(A-1,Z)  (nötron ayırma enerjisi)
        S_p = BE(A,Z) - BE(A-1,Z-1)  (proton ayırma enerjisi)
        """
        BE_current = self.calculate_binding_energy(A, Z, N)['BE_total']
        
        # Nötron ayırma enerjisi
        BE_minus_neutron = self.calculate_binding_energy(A-1, Z, N-1)['BE_total']
        S_n = BE_current - BE_minus_neutron
        
        # Proton ayırma enerjisi
        BE_minus_proton = self.calculate_binding_energy(A-1, Z-1, N)['BE_total']
        S_p = BE_current - BE_minus_proton
        
        return {
            'S_n': S_n,
            'S_p': S_p
        }


class ShellModelCalculator:
    """Kabuk modeli hesaplamaları"""
    
    def calculate_shell_properties(self, Z, N):
        """
        Kabuk modeli özelliklerini hesapla
        
        Args:
            Z: Proton sayısı
            N: Nötron sayısı
            
        Returns:
            dict: Kabuk modeli özellikleri
        """
        # En yakın magik sayıları bul
        Z_nearest_magic = nearest_magic_number(Z, MAGIC_NUMBERS_Z)
        N_nearest_magic = nearest_magic_number(N, MAGIC_NUMBERS_N)
        
        # Magik sayılara uzaklık
        Z_magic_dist = distance_to_magic(Z, MAGIC_NUMBERS_Z)
        N_magic_dist = distance_to_magic(N, MAGIC_NUMBERS_N)
        
        # Kabuk açıklıkları (shell gap)
        Z_shell_gap = self._get_shell_gap(Z)
        N_shell_gap = self._get_shell_gap(N)
        
        # Valans nükleonlar
        Z_valence = self._calculate_valence(Z, MAGIC_NUMBERS_Z)
        N_valence = self._calculate_valence(N, MAGIC_NUMBERS_N)
        
        # Magik karakter (0 = tam magik, 1 = magikten uzak)
        magic_character = 1.0 / (1.0 + Z_magic_dist + N_magic_dist)
        
        # Çift magik mi?
        is_doubly_magic = (Z in MAGIC_NUMBERS_Z) and (N in MAGIC_NUMBERS_N)
        
        return {
            'Z_nearest_magic': Z_nearest_magic,
            'N_nearest_magic': N_nearest_magic,
            'Z_magic_dist': Z_magic_dist,
            'N_magic_dist': N_magic_dist,
            'Z_shell_gap': Z_shell_gap,
            'N_shell_gap': N_shell_gap,
            'Z_valence': Z_valence,
            'N_valence': N_valence,
            'magic_character': magic_character,
            'is_doubly_magic': is_doubly_magic
        }
    
    def _get_shell_gap(self, nucleon_number):
        """Kabuk açıklığını al"""
        for magic_num in sorted(SHELL_GAPS.keys(), reverse=True):
            if nucleon_number >= magic_num:
                return SHELL_GAPS.get(magic_num, 0.0)
        return 0.0
    
    def _calculate_valence(self, nucleon_number, magic_numbers):
        """Valans nükleon sayısı"""
        # Altındaki magik sayıyı bul
        below_magic = [m for m in magic_numbers if m <= nucleon_number]
        
        if not below_magic:
            return nucleon_number
        
        last_magic = max(below_magic)
        
        # Bir sonraki magik sayıyı bul
        above_magic = [m for m in magic_numbers if m > nucleon_number]
        
        if not above_magic:
            return nucleon_number - last_magic
        
        next_magic = min(above_magic)
        
        # Hangi magik sayıya daha yakınsa ona göre valans hesapla
        dist_to_last = nucleon_number - last_magic
        dist_to_next = next_magic - nucleon_number
        
        if dist_to_last <= dist_to_next:
            return dist_to_last  # Particle
        else:
            return -dist_to_next  # Hole (negative indicates holes)


class SchmidtMomentCalculator:
    """Schmidt moment hesaplayıcı"""
    
    def calculate_schmidt_moments(self, Z, N, spin, parity, mm_exp):
        """
        Schmidt momentlerini hesapla
        
        Args:
            Z, N: Proton ve nötron sayıları
            spin: Nükleus spini
            parity: Parite
            mm_exp: Deneysel manyetik moment (μN)
            
        Returns:
            dict: Schmidt moment özellikleri
        """
        # Tek nükleon tipini belirle
        if Z % 2 == 1:  # Odd-Z
            nucleon_type = 'proton'
        elif N % 2 == 1:  # Odd-N
            nucleon_type = 'neutron'
        else:
            return {
                'nucleon_type': 'even-even',
                'schmidt_plus': np.nan,
                'schmidt_minus': np.nan,
                'schmidt_nearest': np.nan,
                'schmidt_deviation': np.nan
            }
        
        # g-faktörleri
        g_l = G_FACTORS[nucleon_type]['g_l']
        g_s = G_FACTORS[nucleon_type]['g_s']
        
        j = spin
        
        # j = l + 1/2 durumu
        l_plus = j - 0.5
        schmidt_plus = g_l * l_plus + g_s * 0.5
        
        # j = l - 1/2 durumu
        l_minus = j + 0.5
        schmidt_minus = (g_l * l_minus - g_s * 0.5) * j / (j + 1)
        
        # En yakın Schmidt değeri
        if abs(mm_exp - schmidt_plus) < abs(mm_exp - schmidt_minus):
            schmidt_nearest = schmidt_plus
        else:
            schmidt_nearest = schmidt_minus
        
        # Sapma (μN cinsinden)
        schmidt_deviation = abs(mm_exp - schmidt_nearest)
        
        return {
            'nucleon_type': nucleon_type,
            'schmidt_plus': schmidt_plus,
            'schmidt_minus': schmidt_minus,
            'schmidt_nearest': schmidt_nearest,
            'schmidt_deviation': schmidt_deviation
        }


class DeformationCalculator:
    """Deformasyon parametreleri hesaplayıcı"""
    
    def calculate_beta2_from_Q(self, Z, A, Q):
        """
        Kuadrupol momentten β₂ hesapla
        
        β₂ = (√(5π)/3) × Q / (Z × R²)
        R = R₀ × A^(1/3)
        
        Args:
            Z: Proton sayısı
            A: Kütle numarası
            Q: Kuadrupol moment (barn)
            
        Returns:
            float: β₂ deformasyon parametresi
        """
        if pd.isna(Q) or Q == 0:
            return np.nan
        
        R = R0 * (A ** (1/3))  # fm
        
        beta_2 = (np.sqrt(5 * np.pi) / 3) * Q / (Z * R**2)
        
        return beta_2
    
    def calculate_B_E2(self, Z, A, beta_2):
        """
        E2 geçiş olasılığını hesapla (Weisskopf birimleri)
        
        B(E2) = (3ZeR₀²β₂A^(2/3) / 4π)²
        """
        if pd.isna(beta_2):
            return np.nan
        
        R0_fm = R0  # fm
        B_E2_fm4 = ((3 * Z * R0_fm**2 * beta_2 * A**(2/3)) / (4 * np.pi))**2
        
        # Weisskopf birimine normalize et
        W_unit = weisskopf_unit(A)
        B_E2_WU = B_E2_fm4 / W_unit
        
        return B_E2_WU
    
    def classify_deformation(self, beta_2):
        """Deformasyon tipini sınıflandır"""
        return get_deformation_type(beta_2)


class TheoreticalCalculator:
    """Tüm teorik hesaplamaları koordine eden ana sınıf"""
    
    def __init__(self):
        self.semf = SEMFCalculator()
        self.shell = ShellModelCalculator()
        self.schmidt = SchmidtMomentCalculator()
        self.deformation = DeformationCalculator()
    
    def calculate_all_properties(self, df):
        """
        Tüm teorik özellikleri hesapla
        
        Args:
            df: Temizlenmiş veri DataFrame'i
            
        Returns:
            DataFrame: Zenginleştirilmiş veri (60+ özellik)
        """
        logger.info("Teorik hesaplamalar başlıyor...")
        df = df.copy()
        
        # 1. SEMF hesaplamaları
        logger.info("  -> SEMF hesaplamaları...")
        semf_results = df.apply(
            lambda row: self.semf.calculate_binding_energy(
                row['A'], row['Z'], row['N']
            ), axis=1
        )
        for key in semf_results[0].keys():
            df[key] = semf_results.apply(lambda x: x[key])
        
        # 2. Ayırma enerjileri
        logger.info("  -> Ayırma enerjileri...")
        sep_results = df.apply(
            lambda row: self.semf.calculate_separation_energies(
                row['A'], row['Z'], row['N']
            ), axis=1
        )
        for key in sep_results[0].keys():
            df[key] = sep_results.apply(lambda x: x[key])
        
        # 3. Kabuk modeli özellikleri
        logger.info("  -> Kabuk modeli özellikleri...")
        shell_results = df.apply(
            lambda row: self.shell.calculate_shell_properties(
                row['Z'], row['N']
            ), axis=1
        )
        for key in shell_results[0].keys():
            df[key] = shell_results.apply(lambda x: x[key])
        
        # 4. Schmidt momentler
        logger.info("  -> Schmidt momentler...")
        schmidt_results = df.apply(
            lambda row: self.schmidt.calculate_schmidt_moments(
                row['Z'], row['N'], row['SPIN'], row['PARITY'], row['MM']
            ), axis=1
        )
        for key in schmidt_results[0].keys():
            df[key] = schmidt_results.apply(lambda x: x[key])
        
        # 5. Deformasyon parametreleri
        logger.info("  -> Deformasyon parametreleri...")
        
        # β₂ hesapla (Q'dan)
        df['beta_2_calc'] = df.apply(
            lambda row: self.deformation.calculate_beta2_from_Q(
                row['Z'], row['A'], row['Q']
            ), axis=1
        )
        
        # β₂ karşılaştırması
        df['Delta_Beta2'] = df['Beta_2'] - df['beta_2_calc']
        
        # B(E2) hesapla
        df['B_E2'] = df.apply(
            lambda row: self.deformation.calculate_B_E2(
                row['Z'], row['A'], row['Beta_2']
            ), axis=1
        )
        
        # Deformasyon tipi
        df['deformation_type'] = df['Beta_2'].apply(
            self.deformation.classify_deformation
        )
        
        # 6. Ek özellikler
        logger.info("  -> Ek özellikler...")
        
        # Nükleer yarıçap
        df['nuclear_radius'] = R0 * (df['A'] ** (1/3))
        
        # N/Z oranı
        df['N_Z_ratio'] = df['N'] / df['Z']
        
        # Asimetri parametresi
        df['asymmetry'] = (df['N'] - df['Z']) / df['A']
        
        # Çiftleşme açıklığı (pairing gap)
        df['pairing_gap'] = 12.0 / np.sqrt(df['A'])
        
        # p-faktör
        df['p_factor'] = (df['Z'] * df['N']) / (df['Z'] + df['N'])
        
        # Fermi enerjisi (yaklaşık)
        df['fermi_energy'] = 33.0 * (df['A'] ** (2/3)) / (R0**2 * df['A'])
        
        logger.info(f"[OK] Teorik hesaplamalar tamamlandı: {len(df.columns)} özellik")
        
        return df


def main():
    """Test fonksiyonu"""
    # Örnek veri
    test_data = pd.DataFrame({
        'A': [208, 16, 56],
        'Z': [82, 8, 26],
        'N': [126, 8, 30],
        'SPIN': [0, 0, 0],
        'PARITY': [1, 1, 1],
        'MM': [0, 0, 0],
        'Q': [0, 0, 0.16],
        'Beta_2': [0, 0, 0.1]
    })
    
    calculator = TheoreticalCalculator()
    enriched = calculator.calculate_all_properties(test_data)
    
    print("\n=== TEORİK HESAPLAMALAR ===")
    print(enriched[['A', 'Z', 'N', 'BE_per_A', 'magic_character', 'deformation_type']])


if __name__ == "__main__":
    main()