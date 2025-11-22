"""
Nilsson Model İmplementasyonu
Nilsson Model for Deformed Nuclei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

import sys
# sys.path.append('..') - REMOVED
from core_modules.constants import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NilssonModel:
    """
    Nilsson modeli - Deforme nükleuslar için tek-parçacık seviyeleri
    
    Hamiltonian:
    H = H_osc + H_l2 + H_ls
    
    where:
    - H_osc: Anizotropik harmonik osilatör
    - H_l2: l² düzeltme terimi
    - H_ls: Spin-orbit etkileşmesi
    """
    
    def __init__(self, omega0=41.0):
        """
        Args:
            omega0: Osilatör frekansı (MeV) - A^(-1/3) ile ölçeklenir
        """
        self.omega0 = omega0
        self.hbar = 1.0  # Natural units
    
    def calculate_single_particle_energy(self, N, nz, Lambda, Omega, delta, 
                                         kappa=0.05, mu=0.6):
        """
        Nilsson tek-parçacık enerjisi
        
        Args:
            N: Major shell quantum number
            nz: z-yönünde kuantum sayısı
            Lambda: z-ekseninde açısal momentum projeksiyon

u
            Omega: Total angular momentum projection
            delta: Deformasyon parametresi (ε ≈ 0.95 × β₂)
            kappa: l² düzeltme parametresi
            mu: Spin-orbit parametresi
            
        Returns:
            float: Enerji (ℏω birimi)
        """
        # Anizotropik osilatör
        E_osc = self._harmonic_oscillator(N, nz, delta)
        
        # l² düzeltmesi
        E_l2 = self._l_squared_correction(N, Lambda, kappa)
        
        # Spin-orbit
        E_ls = self._spin_orbit(Lambda, Omega, mu)
        
        return E_osc + E_l2 + E_ls
    
    def _harmonic_oscillator(self, N, nz, delta):
        """Deformed harmonik osilatör"""
        omega_z = 1.0 + (2.0/3.0) * delta
        omega_perp = 1.0 - (1.0/3.0) * delta
        
        E = (N + 1.5) - nz * delta
        return E
    
    def _l_squared_correction(self, N, Lambda, kappa):
        """l² düzeltme terimi"""
        l_squared = N * (N + 3) - Lambda**2
        return -kappa * l_squared
    
    def _spin_orbit(self, Lambda, Omega, mu):
        """Spin-orbit etkileşmesi"""
        # Omega = Lambda ± 1/2
        sigma = Omega - Lambda
        return 2 * mu * Lambda * sigma
    
    def generate_nilsson_diagram(self, N_shells, proton=True, 
                                output_dir='visualizations/nilsson'):
        """
        Nilsson diyagramı oluştur
        
        Args:
            N_shells: Major shell listesi (örn: [3, 4, 5])
            proton: True ise proton, False ise nötron
            output_dir: Çıktı dizini
        """
        logger.info(f"Nilsson diyagramı oluşturuluyor: {'Proton' if proton else 'Neutron'}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Deformasyon aralığı
        delta_range = np.linspace(-0.3, 0.3, 100)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Her major shell için seviyeleri hesapla
        for N in N_shells:
            levels = self._calculate_levels_for_shell(N, delta_range)
            
            # Her seviyeyi çiz
            for level in levels:
                label = level['label']
                energies = level['energies']
                
                ax.plot(delta_range, energies, linewidth=1.5, 
                       label=f"N={N}, {label}")
        
        # Magic numbers çizgileri
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Deformasyon Parametresi δ', fontsize=14)
        ax.set_ylabel('Enerji (ℏω)', fontsize=14)
        ax.set_title(f'Nilsson Diyagramı: {"Proton" if proton else "Nötron"}', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        particle_type = 'proton' if proton else 'neutron'
        filename = f'nilsson_diagram_{particle_type}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Nilsson diyagramı kaydedildi: {output_dir / filename}")
    
    def _calculate_levels_for_shell(self, N, delta_range):
        """Bir shell için tüm seviyeleri hesapla"""
        levels = []
        
        # N shell'i için olası kuantum sayıları
        for nz in range(N + 1):
            nl = N - nz
            
            # Lambda değerleri
            for Lambda in range(-nl, nl + 1):
                # Omega = Lambda ± 1/2
                for sigma in [-0.5, 0.5]:
                    Omega = Lambda + sigma
                    
                    energies = []
                    for delta in delta_range:
                        E = self.calculate_single_particle_energy(
                            N, nz, Lambda, Omega, delta
                        )
                        energies.append(E)
                    
                    # Seviye etiketi
                    label = self._format_level_label(N, nz, Lambda, Omega)
                    
                    levels.append({
                        'N': N,
                        'nz': nz,
                        'Lambda': Lambda,
                        'Omega': Omega,
                        'label': label,
                        'energies': energies
                    })
        
        return levels
    
    def _format_level_label(self, N, nz, Lambda, Omega):
        """Seviye etiketini formatla (spectroscopic notation)"""
        # Orbital isimler
        orbital_names = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k']
        l = N - nz
        
        if l < len(orbital_names):
            orbital = orbital_names[l]
        else:
            orbital = f'l={l}'
        
        # Omega notation
        omega_str = f"{abs(Omega):.1f}"
        omega_sign = '+' if Omega > 0 else '-'
        
        return f"{N+1}{orbital}[{N}{nz}{abs(Lambda)}]{omega_sign}"


class NilssonDiagramVisualizer:
    """Nilsson diyagramları için görselleştirme sınıfı"""
    
    def __init__(self, output_dir='visualizations/nilsson'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = NilssonModel()
    
    def plot_all_diagrams(self):
        """Tüm Nilsson diyagramlarını oluştur"""
        logger.info("Nilsson diyagramları oluşturuluyor...")
        
        # 1. Proton diyagramı (N = 3, 4, 5)
        self.model.generate_nilsson_diagram([3, 4, 5], proton=True, 
                                            output_dir=self.output_dir)
        
        # 2. Nötron diyagramı (N = 3, 4, 5)
        self.model.generate_nilsson_diagram([3, 4, 5], proton=False,
                                            output_dir=self.output_dir)
        
        # 3. Ağır bölge (N = 5, 6)
        self.model.generate_nilsson_diagram([5, 6], proton=True,
                                            output_dir=self.output_dir / 'heavy_region')
        
        # 4. Shape coexistence bölgeleri
        self._plot_shape_coexistence()
        
        logger.info("[OK] Tüm Nilsson diyagramları tamamlandı")
    
    def _plot_shape_coexistence(self):
        """Shape coexistence (şekil eşyaşamı) bölgelerini vurgula"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Kritik bölgeler
        regions = {
            'A~70 (Se, Kr)': {'Z': 34, 'N_range': (36, 42), 'color': 'red'},
            'A~100 (Zr, Mo)': {'Z': 40, 'N_range': (56, 62), 'color': 'blue'},
            'Pb region': {'Z': 82, 'N_range': (104, 110), 'color': 'green'}
        }
        
        # N-Z chart'ı çiz
        for region_name, info in regions.items():
            Z = info['Z']
            N_min, N_max = info['N_range']
            color = info['color']
            
            # Bölgeyi vurgula
            ax.fill_between([Z-2, Z+2], [N_min, N_min], [N_max, N_max],
                           alpha=0.3, color=color, label=region_name)
        
        ax.set_xlabel('Proton Sayısı (Z)', fontsize=12)
        ax.set_ylabel('Nötron Sayısı (N)', fontsize=12)
        ax.set_title('Shape Coexistence Bölgeleri', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'shape_coexistence_regions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_deformation_energy_surface(self, Z, N, output_name='def_energy_surface.png'):
        """
        Deformasyon enerji yüzeyini çiz (β₂ - γ uzayında)
        
        Args:
            Z: Proton sayısı
            N: Nötron sayısı
            output_name: Çıktı dosya adı
        """
        logger.info(f"Deformasyon enerji yüzeyi oluşturuluyor: Z={Z}, N={N}")
        
        # β₂ ve γ grid
        beta2_range = np.linspace(-0.4, 0.4, 50)
        gamma_range = np.linspace(0, 60, 50)
        
        Beta2, Gamma = np.meshgrid(beta2_range, gamma_range)
        
        # Her nokta için enerji hesapla (simplified)
        Energy = np.zeros_like(Beta2)
        
        for i in range(len(gamma_range)):
            for j in range(len(beta2_range)):
                beta2 = Beta2[i, j]
                gamma = Gamma[i, j]
                
                # Basitleştirilmiş enerji (gerçek HFB hesabı çok karmaşık)
                E = self._simplified_deformation_energy(Z, N, beta2, gamma)
                Energy[i, j] = E
        
        # Contour plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(Beta2, Gamma, Energy, levels=20, cmap='RdYlBu_r')
        ax.contour(Beta2, Gamma, Energy, levels=10, colors='black', 
                  linewidths=0.5, alpha=0.5)
        
        # Minimum işaretle
        min_idx = np.unravel_index(np.argmin(Energy), Energy.shape)
        beta2_min = Beta2[min_idx]
        gamma_min = Gamma[min_idx]
        
        ax.plot(beta2_min, gamma_min, 'r*', markersize=20, 
               label=f'Minimum: β₂={beta2_min:.3f}, γ={gamma_min:.1f}°')
        
        ax.set_xlabel('β₂', fontsize=14)
        ax.set_ylabel('γ (derece)', fontsize=14)
        ax.set_title(f'Deformasyon Enerji Yüzeyi: ¹⁸⁰Hg (Z={Z}, N={N})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Enerji (MeV)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _simplified_deformation_energy(self, Z, N, beta2, gamma):
        """
        Basitleştirilmiş deformasyon enerjisi
        Gerçek hesap için HFB+BCS gerekir
        """
        A = Z + N
        
        # Liquid drop model benzeri
        E_surf = 18.0 * A**(2/3) * (1 + (5/(4*np.pi)) * beta2**2)
        E_coul = 0.7 * Z**2 / A**(1/3) * (1 - beta2**2/5)
        
        # Shell correction (simplified)
        # Magic'e yakınsa daha kararlı (daha düşük enerji)
        Z_magic_dist = distance_to_magic(Z, MAGIC_NUMBERS_Z)
        N_magic_dist = distance_to_magic(N, MAGIC_NUMBERS_N)
        
        E_shell = -5.0 / (1 + Z_magic_dist + N_magic_dist) * np.exp(-beta2**2 / 0.05)
        
        # Gamma bağımlılığı (simplified)
        # γ=0: prolate, γ=60: oblate
        E_gamma = 0.5 * beta2**2 * np.sin(3 * np.deg2rad(gamma))**2
        
        E_total = E_surf + E_coul + E_shell + E_gamma
        
        # Normalize
        return E_total - E_total.min() if hasattr(E_total, 'min') else E_total


def plot_single_particle_levels(Z_or_N, particle_type='proton', 
                                output_dir='visualizations/nilsson'):
    """
    Belirli bir Z veya N için tek-parçacık seviyelerini çiz
    
    Args:
        Z_or_N: Proton veya nötron sayısı
        particle_type: 'proton' veya 'neutron'
        output_dir: Çıktı dizini
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = NilssonModel()
    
    # Major shell'i belirle
    N_major = int(np.sqrt(2 * Z_or_N)) + 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Deformasyon aralığı
    delta_range = np.linspace(-0.3, 0.3, 100)
    
    # Seviyeleri hesapla
    levels = model._calculate_levels_for_shell(N_major, delta_range)
    
    # Çiz
    for level in levels[:20]:  # İlk 20 seviye
        ax.plot(delta_range, level['energies'], linewidth=1.5, alpha=0.7)
    
    # Mevcut Z_or_N değerini işaretle
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Spherical')
    
    ax.set_xlabel('Deformasyon δ', fontsize=12)
    ax.set_ylabel('Enerji (ℏω)', fontsize=12)
    ax.set_title(f'{particle_type.capitalize()} Tek-Parçacık Seviyeleri: N={Z_or_N}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    filename = f'sp_levels_{particle_type}_{Z_or_N}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"[OK] Tek-parçacık seviyeleri kaydedildi: {output_dir / filename}")


def main():
    """Test fonksiyonu"""
    
    # 1. Nilsson diyagramları
    print("\n=== NILSSON DİYAGRAMLARI ===")
    visualizer = NilssonDiagramVisualizer('output/test_nilsson')
    visualizer.plot_all_diagrams()
    
    # 2. Deformasyon enerji yüzeyi
    print("\n=== DEFORMASYON ENERJİ YÜZEYİ ===")
    visualizer.plot_deformation_energy_surface(Z=80, N=100, 
                                               output_name='Hg180_energy_surface.png')
    
    # 3. Tek-parçacık seviyeleri
    print("\n=== TEK-PARÇACIK SEVİYELERİ ===")
    plot_single_particle_levels(Z_or_N=82, particle_type='proton',
                                output_dir='output/test_nilsson')
    
    print("\n[OK] Tüm Nilsson testleri tamamlandı")


if __name__ == "__main__":
    main()