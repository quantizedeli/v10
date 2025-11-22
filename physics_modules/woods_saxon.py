"""
Woods-Saxon Potansiyel Modeli
Woods-Saxon Potential for Nuclear Shell Model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import odeint, simpson
from pathlib import Path
import logging

import sys
# sys.path.append('..') - REMOVED
from core_modules.constants import WOODS_SAXON_PARAMS, HBAR_C, R0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WoodsSaxonPotential:
    """
    Woods-Saxon potansiyeli
    
    V(r) = -V₀ / (1 + exp((r - R)/a))
    
    R = r₀ × A^(1/3): Nükleer yarıçap
    a: Yüzey kalınlığı (diffuseness)
    """
    
    def __init__(self, params=None):
        """
        Args:
            params: Woods-Saxon parametreleri (dict)
        """
        self.params = params or WOODS_SAXON_PARAMS
    
    def central_potential(self, r, A):
        """
        Merkezi Woods-Saxon potansiyeli
        
        Args:
            r: Radyal mesafe (fm)
            A: Kütle numarası
            
        Returns:
            float: Potansiyel (MeV)
        """
        V0 = self.params['V0']
        r0 = self.params['r0']
        a = self.params['a']
        
        R = r0 * A**(1/3)
        
        V = -V0 / (1.0 + np.exp((r - R) / a))
        
        return V
    
    def spin_orbit_potential(self, r, A, l):
        """
        Spin-orbit potansiyeli
        
        V_so(r) = V_so × (1/r) × (dV/dr) × l·s
        
        Args:
            r: Radyal mesafe (fm)
            A: Kütle numarası
            l: Orbital açısal momentum
            
        Returns:
            float: Spin-orbit potansiyel (MeV)
        """
        V_so = self.params['V_so']
        r_so = self.params['r_so']
        a_so = self.params['a_so']
        
        R_so = r_so * A**(1/3)
        
        # dV/dr (gradient)
        exp_term = np.exp((r - R_so) / a_so)
        dV_dr = -V_so * exp_term / (a_so * (1 + exp_term)**2)
        
        # (1/r) × dV/dr
        if r > 0:
            V_ls = (1.0 / r) * dV_dr
        else:
            V_ls = 0.0
        
        return V_ls
    
    def coulomb_potential(self, r, Z, A):
        """
        Coulomb potansiyeli (protonlar için)
        
        Args:
            r: Radyal mesafe (fm)
            Z: Proton sayısı
            A: Kütle numarası
            
        Returns:
            float: Coulomb potansiyeli (MeV)
        """
        R = R0 * A**(1/3)
        
        # Fine structure constant (α ≈ 1/137)
        alpha = 1.0 / 137.036
        
        # e²/4πε₀ = α × ℏc ≈ 1.44 MeV·fm
        k = alpha * HBAR_C
        
        if r <= R:
            # İçeride: uniform yük dağılımı
            V_coul = (k * Z / (2 * R)) * (3 - (r/R)**2)
        else:
            # Dışarıda: nokta yük
            V_coul = k * Z / r
        
        return V_coul
    
    def total_potential(self, r, A, Z, l, j, is_proton=True):
        """
        Toplam potansiyel
        
        V_total = V_central + V_ls + V_coulomb (proton için)
        
        Args:
            r: Radyal mesafe (fm)
            A: Kütle numarası
            Z: Proton sayısı
            l: Orbital açısal momentum
            j: Toplam açısal momentum
            is_proton: Proton mu?
            
        Returns:
            float: Toplam potansiyel (MeV)
        """
        # Merkezi potansiyel
        V_central = self.central_potential(r, A)
        
        # Spin-orbit
        V_ls_radial = self.spin_orbit_potential(r, A, l)
        
        # <l·s> = (j(j+1) - l(l+1) - s(s+1))/2
        # s = 1/2 için: <l·s> = (j(j+1) - l(l+1) - 3/4)/2
        ls_factor = (j*(j+1) - l*(l+1) - 0.75) / 2.0
        V_ls = V_ls_radial * ls_factor
        
        # Toplam
        V_total = V_central + V_ls
        
        # Protonlar için Coulomb ekle
        if is_proton:
            V_coul = self.coulomb_potential(r, Z, A)
            V_total += V_coul
        
        return V_total
    
    def plot_potential(self, A, Z, l=0, j=0.5, output_path=None):
        """
        Potansiyeli görselleştir
        
        Args:
            A: Kütle numarası
            Z: Proton sayısı
            l: Orbital açısal momentum
            j: Toplam açısal momentum
            output_path: Çıktı dosya yolu
        """
        r_range = np.linspace(0.01, 15.0, 200)
        
        # Potansiyel bileşenleri
        V_central = [self.central_potential(r, A) for r in r_range]
        V_ls_radial = [self.spin_orbit_potential(r, A, l) for r in r_range]
        V_coul_proton = [self.coulomb_potential(r, Z, A) for r in r_range]
        V_total_proton = [self.total_potential(r, A, Z, l, j, is_proton=True) 
                         for r in r_range]
        V_total_neutron = [self.total_potential(r, A, Z, l, j, is_proton=False) 
                          for r in r_range]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sol: Bileşenler
        axes[0].plot(r_range, V_central, 'b-', linewidth=2, label='Central')
        axes[0].plot(r_range, V_coul_proton, 'r--', linewidth=2, label='Coulomb (proton)')
        axes[0].axhline(0, color='k', linestyle=':', alpha=0.5)
        axes[0].set_xlabel('r (fm)', fontsize=12)
        axes[0].set_ylabel('V(r) (MeV)', fontsize=12)
        axes[0].set_title(f'Woods-Saxon Potansiyel Bileşenleri (A={A}, Z={Z})',
                         fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-60, 20)
        
        # Sağ: Toplam potansiyeller
        axes[1].plot(r_range, V_total_proton, 'r-', linewidth=2, 
                    label=f'Proton (l={l}, j={j})')
        axes[1].plot(r_range, V_total_neutron, 'b-', linewidth=2, 
                    label=f'Neutron (l={l}, j={j})')
        axes[1].axhline(0, color='k', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('r (fm)', fontsize=12)
        axes[1].set_ylabel('V(r) (MeV)', fontsize=12)
        axes[1].set_title(f'Toplam Potansiyeller',
                         fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-60, 20)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"[OK] Potansiyel grafiği kaydedildi: {output_path}")
        else:
            plt.show()
        
        plt.close()


class ShellModelSolver:
    """Woods-Saxon potansiyelinde tek-parçacık seviyelerini çöz"""
    
    def __init__(self, potential):
        """
        Args:
            potential: WoodsSaxonPotential nesnesi
        """
        self.potential = potential
    
    def solve_radial_equation(self, A, Z, n, l, is_proton=True):
        """
        Radyal Schrödinger denklemini çöz (simplified)
        
        [-ℏ²/2m d²/dr² + l(l+1)ℏ²/2mr² + V(r)] u(r) = E u(r)
        
        Args:
            A: Kütle numarası
            Z: Proton sayısı
            n: Radyal kuantum sayısı
            l: Orbital açısal momentum
            is_proton: Proton mu?
            
        Returns:
            dict: {'energy': E, 'wavefunction': u(r), 'r': r_grid}
        """
        # j = l ± 1/2 (iki olasılık)
        j_options = [l + 0.5, l - 0.5] if l > 0 else [0.5]
        
        results = []
        
        for j in j_options:
            if j < 0:
                continue
            
            # Simplified: Enerji seviyesini tahmin et
            # Gerçek çözüm için numerik integrasyon gerekir
            E_estimate = self._estimate_energy_level(A, Z, n, l, j, is_proton)
            
            results.append({
                'n': n,
                'l': l,
                'j': j,
                'energy': E_estimate,
                'notation': self._spectroscopic_notation(n, l, j)
            })
        
        return results
    
    def _estimate_energy_level(self, A, Z, n, l, j, is_proton):
        """
        Enerji seviyesini tahmin et (simplified)
        Gerçek hesap için Numerov veya shooting method gerekir
        """
        # Harmonic oscillator benzeri tahmin
        N = 2*n + l  # Major shell quantum number
        
        # Base energy
        hbar_omega = 41.0 / A**(1/3)  # MeV
        E_base = hbar_omega * (N + 1.5)
        
        # Spin-orbit splitting
        ls_factor = (j*(j+1) - l*(l+1) - 0.75) / 2.0
        E_ls = -0.5 * self.potential.params['V_so'] * ls_factor / A**(1/3)
        
        # Coulomb shift (proton için)
        E_coulomb = 0.7 * Z**2 / A**(1/3) if is_proton else 0.0
        
        # Toplam
        E_total = -self.potential.params['V0'] + E_base + E_ls + E_coulomb
        
        return E_total
    
    def _spectroscopic_notation(self, n, l, j):
        """Spektroskopik notasyon: n l_j"""
        orbital_names = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k']
        
        if l < len(orbital_names):
            orbital = orbital_names[l]
        else:
            orbital = f'l={l}'
        
        return f"{n+1}{orbital}_{int(2*j)}/2"
    
    def calculate_shell_structure(self, A, Z, max_N=5):
        """
        Kabuk yapısını hesapla
        
        Args:
            A: Kütle numarası
            Z: Proton sayısı
            max_N: Maksimum major shell
            
        Returns:
            DataFrame: Enerji seviyeleri
        """
        levels = []
        
        for N in range(max_N + 1):
            for l in range(N + 1):
                n = (N - l) // 2
                if n < 0:
                    continue
                
                # Proton seviyeleri
                proton_levels = self.solve_radial_equation(A, Z, n, l, is_proton=True)
                for level in proton_levels:
                    level['particle'] = 'proton'
                    level['N'] = N
                    levels.append(level)
                
                # Nötron seviyeleri
                neutron_levels = self.solve_radial_equation(A, Z, n, l, is_proton=False)
                for level in neutron_levels:
                    level['particle'] = 'neutron'
                    level['N'] = N
                    levels.append(level)
        
        df = pd.DataFrame(levels)
        
        # Sırala
        df = df.sort_values(['particle', 'energy'], ascending=[True, True])
        df = df.reset_index(drop=True)
        
        return df
    
    def plot_energy_levels(self, A, Z, output_path=None):
        """
        Enerji seviyelerini görselleştir
        
        Args:
            A: Kütle numarası
            Z: Proton sayısı
            output_path: Çıktı dosya yolu
        """
        logger.info(f"Enerji seviyeleri hesaplanıyor: A={A}, Z={Z}")
        
        df = self.calculate_shell_structure(A, Z, max_N=5)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        
        # Proton seviyeleri
        proton_df = df[df['particle'] == 'proton']
        self._plot_single_particle_levels(axes[0], proton_df, 'Proton', A, Z)
        
        # Nötron seviyeleri
        neutron_df = df[df['particle'] == 'neutron']
        self._plot_single_particle_levels(axes[1], neutron_df, 'Neutron', A, Z)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"[OK] Enerji seviyeleri kaydedildi: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_single_particle_levels(self, ax, df, particle_name, A, Z):
        """Tek-parçacık seviyelerini çiz"""
        
        # Her seviyeyi çiz
        for idx, row in df.iterrows():
            energy = row['energy']
            notation = row['notation']
            N = row['N']
            
            # Çizgi uzunluğu (degeneracy göstergesi)
            line_length = 0.3
            
            # Çizgi rengi (major shell'e göre)
            colors = plt.cm.tab10(np.linspace(0, 1, 6))
            color = colors[N % 6]
            
            ax.hlines(energy, idx - line_length, idx + line_length,
                     colors=color, linewidth=3)
            
            # Etiket
            ax.text(idx, energy, f'  {notation}', 
                   verticalalignment='center', fontsize=8)
        
        ax.set_ylabel('Enerji (MeV)', fontsize=12)
        ax.set_title(f'{particle_name} Tek-Parçacık Seviyeleri\nA={A}, Z={Z}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-1, len(df))
        ax.set_xticks([])


def main():
    """Test fonksiyonu"""
    
    # 1. Woods-Saxon potansiyel
    print("\n=== WOODS-SAXON POTANSİYEL ===")
    ws = WoodsSaxonPotential()
    ws.plot_potential(A=208, Z=82, l=0, j=0.5,
                     output_path='output/test_woods_saxon/potential_Pb208.png')
    
    # 2. Shell structure
    print("\n=== KABUK YAPISI ===")
    solver = ShellModelSolver(ws)
    df = solver.calculate_shell_structure(A=208, Z=82, max_N=5)
    print(df.head(20))
    
    # 3. Enerji seviyeleri
    print("\n=== ENERJİ SEVİYELERİ ===")
    solver.plot_energy_levels(A=208, Z=82,
                              output_path='output/test_woods_saxon/levels_Pb208.png')
    
    print("\n[OK] Tüm Woods-Saxon testleri tamamlandı")


if __name__ == "__main__":
    main()