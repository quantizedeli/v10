"""
Nuclei Distribution Analyzer
Çekirdek Dağılım Analiz Modülü

Bu modül veri setlerindeki çekirdek dağılımlarını analiz eder ve detaylı raporlar oluşturur.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NucleiDistributionAnalyzer:
    """Çekirdek dağılım analizi ve raporlama"""

    def __init__(self, output_dir: str = 'distribution_reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Tek bir veri seti için detaylı dağılım analizi

        Args:
            df: DataFrame with NUCLEUS, A, Z, N columns
            dataset_name: Veri seti adı

        Returns:
            Dict: Analiz sonuçları
        """
        analysis = {
            'dataset_name': dataset_name,
            'total_nuclei': len(df),
            'z_distribution': self._analyze_z_distribution(df),
            'n_distribution': self._analyze_n_distribution(df),
            'mass_distribution': self._analyze_mass_distribution(df),
            'nucleus_type_distribution': self._analyze_nucleus_types(df),
            'magic_distribution': self._analyze_magic_proximity(df),
            'isotope_diversity': self._analyze_isotope_diversity(df),
            'isotone_diversity': self._analyze_isotone_diversity(df),
            'deformation_distribution': self._analyze_deformation(df) if 'Beta_2' in df.columns else None
        }

        return analysis

    def _analyze_z_distribution(self, df: pd.DataFrame) -> Dict:
        """Proton sayısı (Z) dağılımı"""
        z_counts = df['Z'].value_counts().sort_index()

        return {
            'min': int(df['Z'].min()),
            'max': int(df['Z'].max()),
            'mean': float(df['Z'].mean()),
            'median': float(df['Z'].median()),
            'unique_count': len(z_counts),
            'distribution': z_counts.to_dict(),
            'range': int(df['Z'].max() - df['Z'].min())
        }

    def _analyze_n_distribution(self, df: pd.DataFrame) -> Dict:
        """Nötron sayısı (N) dağılımı"""
        n_counts = df['N'].value_counts().sort_index()

        return {
            'min': int(df['N'].min()),
            'max': int(df['N'].max()),
            'mean': float(df['N'].mean()),
            'median': float(df['N'].median()),
            'unique_count': len(n_counts),
            'distribution': n_counts.to_dict(),
            'range': int(df['N'].max() - df['N'].min())
        }

    def _analyze_mass_distribution(self, df: pd.DataFrame) -> Dict:
        """Kütle sayısı (A) dağılımı"""
        a_counts = df['A'].value_counts().sort_index()

        # Kütle grupları
        mass_groups = {
            'light (A<40)': len(df[df['A'] < 40]),
            'medium_light (40≤A<90)': len(df[(df['A'] >= 40) & (df['A'] < 90)]),
            'medium (90≤A<140)': len(df[(df['A'] >= 90) & (df['A'] < 140)]),
            'medium_heavy (140≤A<200)': len(df[(df['A'] >= 140) & (df['A'] < 200)]),
            'heavy (A≥200)': len(df[df['A'] >= 200])
        }

        return {
            'min': int(df['A'].min()),
            'max': int(df['A'].max()),
            'mean': float(df['A'].mean()),
            'median': float(df['A'].median()),
            'unique_count': len(a_counts),
            'distribution': a_counts.to_dict(),
            'mass_groups': mass_groups,
            'range': int(df['A'].max() - df['A'].min())
        }

    def _analyze_nucleus_types(self, df: pd.DataFrame) -> Dict:
        """Çekirdek tipi dağılımı (even-even, odd-odd, vb.)"""
        if 'nucleus_type' in df.columns:
            type_counts = df['nucleus_type'].value_counts().to_dict()
        else:
            # Manuel hesapla
            type_counts = {
                'even-even': len(df[(df['Z'] % 2 == 0) & (df['N'] % 2 == 0)]),
                'even-odd': len(df[(df['Z'] % 2 == 0) & (df['N'] % 2 == 1)]),
                'odd-even': len(df[(df['Z'] % 2 == 1) & (df['N'] % 2 == 0)]),
                'odd-odd': len(df[(df['Z'] % 2 == 1) & (df['N'] % 2 == 1)])
            }

        total = sum(type_counts.values())
        percentages = {k: (v/total*100) for k, v in type_counts.items()}

        return {
            'counts': type_counts,
            'percentages': percentages
        }

    def _analyze_magic_proximity(self, df: pd.DataFrame) -> Dict:
        """Magik sayılara yakınlık analizi"""
        magic_z = [2, 8, 20, 28, 50, 82, 114, 126]
        magic_n = [2, 8, 20, 28, 50, 82, 126, 184]

        # Magik veya magike yakın (±2)
        magic_z_exact = df['Z'].isin(magic_z).sum()
        magic_n_exact = df['N'].isin(magic_n).sum()

        # Yakın magik
        def near_magic(values, magic_list, tolerance=2):
            count = 0
            for val in values:
                if any(abs(val - m) <= tolerance for m in magic_list):
                    count += 1
            return count

        near_magic_z = near_magic(df['Z'].values, magic_z)
        near_magic_n = near_magic(df['N'].values, magic_n)

        # Çift magik (Z ve N her ikisi de magik)
        double_magic = 0
        for _, row in df.iterrows():
            if row['Z'] in magic_z and row['N'] in magic_n:
                double_magic += 1

        return {
            'magic_Z_exact': int(magic_z_exact),
            'magic_N_exact': int(magic_n_exact),
            'near_magic_Z': int(near_magic_z),
            'near_magic_N': int(near_magic_n),
            'double_magic': int(double_magic)
        }

    def _analyze_isotope_diversity(self, df: pd.DataFrame) -> Dict:
        """İzotop çeşitliliği (aynı Z, farklı N)"""
        isotope_chains = df.groupby('Z')['N'].apply(list).to_dict()

        isotope_counts = {z: len(n_list) for z, n_list in isotope_chains.items()}

        return {
            'total_elements': len(isotope_chains),
            'isotopes_per_element': isotope_counts,
            'avg_isotopes_per_element': float(np.mean(list(isotope_counts.values()))),
            'max_isotopes': int(max(isotope_counts.values())),
            'elements_with_max': [z for z, count in isotope_counts.items() if count == max(isotope_counts.values())]
        }

    def _analyze_isotone_diversity(self, df: pd.DataFrame) -> Dict:
        """İzoton çeşitliliği (aynı N, farklı Z)"""
        isotone_chains = df.groupby('N')['Z'].apply(list).to_dict()

        isotone_counts = {n: len(z_list) for n, z_list in isotone_chains.items()}

        return {
            'total_isotone_chains': len(isotone_chains),
            'isotones_per_chain': isotone_counts,
            'avg_isotones_per_chain': float(np.mean(list(isotone_counts.values()))),
            'max_isotones': int(max(isotone_counts.values()))
        }

    def _analyze_deformation(self, df: pd.DataFrame) -> Dict:
        """Deformasyon (Beta_2) dağılımı"""
        if 'Beta_2' not in df.columns:
            return None

        beta2_data = df['Beta_2'].dropna()

        if len(beta2_data) == 0:
            return None

        # Bölge tanımları
        regions = {
            'spherical': len(beta2_data[beta2_data.abs() < 0.05]),
            'weakly_deformed': len(beta2_data[(beta2_data.abs() >= 0.05) & (beta2_data.abs() < 0.15)]),
            'moderately_deformed': len(beta2_data[(beta2_data.abs() >= 0.15) & (beta2_data.abs() < 0.35)]),
            'highly_deformed': len(beta2_data[beta2_data.abs() >= 0.35])
        }

        return {
            'min': float(beta2_data.min()),
            'max': float(beta2_data.max()),
            'mean': float(beta2_data.mean()),
            'std': float(beta2_data.std()),
            'regions': regions
        }

    def create_distribution_report(self, analysis: Dict, save_path: Path):
        """Dağılım raporunu Excel olarak kaydet"""

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Özet sayfa
            summary_data = {
                'Metric': ['Total Nuclei', 'Z Range', 'N Range', 'A Range',
                          'Unique Z', 'Unique N', 'Unique A',
                          'Double Magic Nuclei'],
                'Value': [
                    analysis['total_nuclei'],
                    f"{analysis['z_distribution']['min']}-{analysis['z_distribution']['max']}",
                    f"{analysis['n_distribution']['min']}-{analysis['n_distribution']['max']}",
                    f"{analysis['mass_distribution']['min']}-{analysis['mass_distribution']['max']}",
                    analysis['z_distribution']['unique_count'],
                    analysis['n_distribution']['unique_count'],
                    analysis['mass_distribution']['unique_count'],
                    analysis['magic_distribution']['double_magic']
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Z dağılımı
            z_dist_df = pd.DataFrame([
                {'Z': z, 'Count': count}
                for z, count in sorted(analysis['z_distribution']['distribution'].items())
            ])
            z_dist_df.to_excel(writer, sheet_name='Z_Distribution', index=False)

            # N dağılımı
            n_dist_df = pd.DataFrame([
                {'N': n, 'Count': count}
                for n, count in sorted(analysis['n_distribution']['distribution'].items())
            ])
            n_dist_df.to_excel(writer, sheet_name='N_Distribution', index=False)

            # A dağılımı
            a_dist_df = pd.DataFrame([
                {'A': a, 'Count': count}
                for a, count in sorted(analysis['mass_distribution']['distribution'].items())
            ])
            a_dist_df.to_excel(writer, sheet_name='A_Distribution', index=False)

            # Kütle grupları
            mass_groups_df = pd.DataFrame([
                {'Mass Group': group, 'Count': count}
                for group, count in analysis['mass_distribution']['mass_groups'].items()
            ])
            mass_groups_df.to_excel(writer, sheet_name='Mass_Groups', index=False)

            # Çekirdek tipleri
            nucleus_types_df = pd.DataFrame([
                {'Type': ntype, 'Count': count, 'Percentage': f"{analysis['nucleus_type_distribution']['percentages'][ntype]:.2f}%"}
                for ntype, count in analysis['nucleus_type_distribution']['counts'].items()
            ])
            nucleus_types_df.to_excel(writer, sheet_name='Nucleus_Types', index=False)

            # İzotop çeşitliliği
            isotope_df = pd.DataFrame([
                {'Z (Element)': z, 'Number of Isotopes': count}
                for z, count in sorted(analysis['isotope_diversity']['isotopes_per_element'].items())
            ])
            isotope_df.to_excel(writer, sheet_name='Isotope_Diversity', index=False)

            # Magik sayılar
            magic_df = pd.DataFrame([
                {'Metric': 'Exact Magic Z', 'Count': analysis['magic_distribution']['magic_Z_exact']},
                {'Metric': 'Exact Magic N', 'Count': analysis['magic_distribution']['magic_N_exact']},
                {'Metric': 'Near Magic Z (±2)', 'Count': analysis['magic_distribution']['near_magic_Z']},
                {'Metric': 'Near Magic N (±2)', 'Count': analysis['magic_distribution']['near_magic_N']},
                {'Metric': 'Double Magic', 'Count': analysis['magic_distribution']['double_magic']}
            ])
            magic_df.to_excel(writer, sheet_name='Magic_Numbers', index=False)

            # Deformasyon (varsa)
            if analysis.get('deformation_distribution'):
                deform_df = pd.DataFrame([
                    {'Metric': key, 'Value': value}
                    for key, value in analysis['deformation_distribution'].items()
                    if key != 'regions'
                ])
                deform_df.to_excel(writer, sheet_name='Deformation_Stats', index=False)

                deform_regions_df = pd.DataFrame([
                    {'Region': region, 'Count': count}
                    for region, count in analysis['deformation_distribution']['regions'].items()
                ])
                deform_regions_df.to_excel(writer, sheet_name='Deformation_Regions', index=False)

        logger.info(f"✓ Distribution report saved: {save_path}")

    def create_master_nuclei_catalog(self, df_full: pd.DataFrame, output_path: Path):
        """
        Tüm kullanılan çekirdeklerin master kataloğunu oluştur

        Args:
            df_full: Tüm çekirdekleri içeren DataFrame
            output_path: Kayıt yolu
        """
        # Çekirdek bilgileri
        nuclei_info = df_full[['NUCLEUS', 'A', 'Z', 'N']].copy()

        # Ekstra bilgiler ekle (varsa)
        if 'SPIN' in df_full.columns:
            nuclei_info['SPIN'] = df_full['SPIN']
        if 'PARITY' in df_full.columns:
            nuclei_info['PARITY'] = df_full['PARITY']
        if 'Beta_2' in df_full.columns:
            nuclei_info['Beta_2'] = df_full['Beta_2']
        if 'MM' in df_full.columns:
            nuclei_info['Magnetic_Moment'] = df_full['MM']
        if 'Q' in df_full.columns:
            nuclei_info['Quadrupole_Moment'] = df_full['Q']

        # Çekirdek tipi
        nuclei_info['Nucleus_Type'] = nuclei_info.apply(
            lambda row: self._get_nucleus_type(row['Z'], row['N']), axis=1
        )

        # Kütle grubu
        nuclei_info['Mass_Group'] = nuclei_info['A'].apply(self._get_mass_group)

        # Magik yakınlık
        magic_z = [2, 8, 20, 28, 50, 82, 114, 126]
        magic_n = [2, 8, 20, 28, 50, 82, 126, 184]

        nuclei_info['Is_Magic_Z'] = nuclei_info['Z'].isin(magic_z)
        nuclei_info['Is_Magic_N'] = nuclei_info['N'].isin(magic_n)
        nuclei_info['Is_Double_Magic'] = nuclei_info['Is_Magic_Z'] & nuclei_info['Is_Magic_N']

        # Sırala
        nuclei_info = nuclei_info.sort_values(['Z', 'N']).reset_index(drop=True)

        # Kaydet
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Ana liste
            nuclei_info.to_excel(writer, sheet_name='All_Nuclei', index=False)

            # Element bazında
            for z in sorted(nuclei_info['Z'].unique()):
                element_df = nuclei_info[nuclei_info['Z'] == z]
                sheet_name = f'Z{z}'
                if len(sheet_name) > 31:  # Excel limit
                    sheet_name = sheet_name[:31]
                element_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"✓ Master nuclei catalog saved: {output_path}")

    def _get_nucleus_type(self, Z: int, N: int) -> str:
        """Çekirdek tipini belirle"""
        if Z % 2 == 0 and N % 2 == 0:
            return 'even-even'
        elif Z % 2 == 0 and N % 2 == 1:
            return 'even-odd'
        elif Z % 2 == 1 and N % 2 == 0:
            return 'odd-even'
        else:
            return 'odd-odd'

    def _get_mass_group(self, A: int) -> str:
        """Kütle grubunu belirle"""
        if A < 40:
            return 'light'
        elif A < 90:
            return 'medium_light'
        elif A < 140:
            return 'medium'
        elif A < 200:
            return 'medium_heavy'
        else:
            return 'heavy'


def main():
    """Test fonksiyonu"""
    # Test data
    test_df = pd.DataFrame({
        'NUCLEUS': ['H2', 'He4', 'Li6', 'C12', 'O16'],
        'A': [2, 4, 6, 12, 16],
        'Z': [1, 2, 3, 6, 8],
        'N': [1, 2, 3, 6, 8],
        'SPIN': [1, 0, 1, 0, 0],
        'PARITY': [1, 1, 1, 1, 1],
        'Beta_2': [0.0, 0.0, 0.1, 0.0, 0.0]
    })

    analyzer = NucleiDistributionAnalyzer('test_output')
    analysis = analyzer.analyze_dataset(test_df, 'test_dataset')

    print("Analysis Results:")
    print(f"Total nuclei: {analysis['total_nuclei']}")
    print(f"Z range: {analysis['z_distribution']['min']}-{analysis['z_distribution']['max']}")
    print(f"Nucleus types: {analysis['nucleus_type_distribution']['counts']}")

    # Save report
    analyzer.create_distribution_report(analysis, Path('test_output/test_report.xlsx'))
    analyzer.create_master_nuclei_catalog(test_df, Path('test_output/master_catalog.xlsx'))

    print("\n✓ Test completed!")


if __name__ == "__main__":
    main()
