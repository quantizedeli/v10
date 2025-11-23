"""
Feature Set Builder - Dynamic Feature Combination Generator
Dinamik Özellik Kombinasyon Oluşturucu

Bu modül, binlerce farklı feature kombinasyonu oluşturmak için kullanılır.
AZN, AZNS, AZNP, AZNSP ve benzeri tüm olası kombinasyonları üretir.

Versiyon: 2.0.0
Tarih: 2025-11-23
"""

from typing import Dict, List, Set, Tuple
from itertools import combinations, chain


class FeatureSetBuilder:
    """
    Dinamik feature set oluşturucu

    Binlerce farklı feature kombinasyonu oluşturarak comprehensive bir
    dataset sistemi için gerekli feature set'leri üretir.
    """

    def __init__(self):
        """Feature gruplarını tanımla"""

        # ====================================================================
        # BASE FEATURES - Her zaman AZN temelinde
        # ====================================================================
        self.BASE_FEATURES = {
            'AZN': ['A', 'Z', 'N']
        }

        # ====================================================================
        # BASIC ADDONS - Spin ve Parity
        # ====================================================================
        self.BASIC_ADDONS = {
            'S': ['SPIN'],
            'P': ['PARITY'],
            'SP': ['SPIN', 'PARITY']
        }

        # ====================================================================
        # PHYSICS FEATURE GROUPS
        # ====================================================================

        # 1. Deformation Features
        self.DEFORMATION_GROUP = {
            'beta': ['beta_2'],
            'beta4': ['beta_2', 'Beta_4'],
            'def_full': ['beta_2', 'Beta_4', 'spherical_index']
        }

        # 2. Pairing Features
        self.PAIRING_GROUP = {
            'p': ['p_factor'],
            'pair_full': ['p_factor', 'BE_pairing']
        }

        # 3. Magic Number Features
        self.MAGIC_GROUP = {
            'magic': ['Z_magic_dist', 'N_magic_dist'],
            'magic_full': ['Z_magic_dist', 'N_magic_dist', 'magic_character'],
            'magic_adv': ['Z_magic_dist', 'N_magic_dist', 'magic_character',
                         'is_magic_Z', 'is_magic_N', 'is_doubly_magic']
        }

        # 4. Binding Energy Features
        self.BINDING_GROUP = {
            'BE': ['BE_per_A'],
            'BE_full': ['BE_per_A', 'BE_total'],
            'BE_adv': ['BE_per_A', 'BE_total', 'S2n', 'S2p', 'separation_energy']
        }

        # 5. SEMF Features
        self.SEMF_GROUP = {
            'semf_basic': ['BE_volume', 'BE_surface', 'BE_coulomb'],
            'semf_full': ['BE_volume', 'BE_surface', 'BE_coulomb',
                         'BE_asymmetry', 'BE_pairing'],
            'semf_all': ['BE_volume', 'BE_surface', 'BE_coulomb',
                        'BE_asymmetry', 'BE_pairing', 'shell_correction']
        }

        # 6. Shell Model Features
        self.SHELL_GROUP = {
            'shell_basic': ['Z_shell_gap', 'N_shell_gap'],
            'shell_full': ['Z_shell_gap', 'N_shell_gap', 'shell_closure_index'],
            'shell_all': ['Z_shell_gap', 'N_shell_gap', 'shell_closure_index',
                         'valence_nucleons']
        }

        # 7. Schmidt Model Features (FOR MM TARGET)
        self.SCHMIDT_GROUP = {
            'schmidt': ['schmidt_nearest'],
            'schmidt_full': ['schmidt_nearest', 'schmidt_deviation']
        }

        # 8. Collective Model Features
        self.COLLECTIVE_GROUP = {
            'collective': ['collective_parameter'],
            'collective_full': ['collective_parameter', 'rotational_constant',
                              'vibrational_frequency']
        }

        # 9. Asymmetry Features
        self.ASYMMETRY_GROUP = {
            'asym': ['asymmetry'],
            'asym_full': ['asymmetry', 'neutron_excess']
        }

        # 10. Woods-Saxon Features (Optional, computationally expensive)
        self.WOODS_SAXON_GROUP = {
            'ws': ['ws_surface_thick', 'fermi_energy']
        }

        # 11. Nilsson Model Features (For deformed nuclei)
        self.NILSSON_GROUP = {
            'nilsson': ['nilsson_epsilon', 'nilsson_omega']
        }

        # ====================================================================
        # COMBINED PHYSICS GROUPS
        # ====================================================================
        self.COMBINED_GROUPS = {
            # Beta + Pairing (very common combination)
            'beta_p': ['beta_2', 'p_factor'],

            # Magic + Binding Energy
            'magic_BE': ['Z_magic_dist', 'N_magic_dist', 'BE_per_A'],

            # Shell + Deformation
            'shell_beta': ['Z_shell_gap', 'N_shell_gap', 'beta_2'],

            # Complete basic physics
            'physics_basic': ['beta_2', 'p_factor', 'BE_per_A',
                             'Z_magic_dist', 'N_magic_dist'],

            # Advanced physics (as in original ADVANCED)
            'physics_adv': ['beta_2', 'p_factor', 'BE_per_A',
                           'Z_magic_dist', 'N_magic_dist', 'shell_correction'],

            # Ultra-advanced (all key features)
            'physics_ultra': ['beta_2', 'p_factor', 'BE_per_A',
                             'Z_magic_dist', 'N_magic_dist',
                             'shell_correction', 'valence_nucleons',
                             'asymmetry', 'collective_parameter']
        }

        # ====================================================================
        # TARGET-SPECIFIC RECOMMENDED GROUPS
        # ====================================================================
        self.TARGET_RECOMMENDED = {
            'MM': {
                'critical': ['SPIN', 'PARITY', 'schmidt_nearest'],
                'important': ['schmidt_deviation', 'magic_character',
                             'valence_nucleons'],
                'useful': ['beta_2', 'collective_parameter']
            },
            'QM': {
                'critical': ['beta_2'],
                'important': ['valence_nucleons', 'Z_magic_dist', 'N_magic_dist'],
                'useful': ['Beta_4', 'collective_parameter', 'nilsson_epsilon']
            },
            'Beta_2': {
                'critical': ['p_factor'],
                'important': ['Z_magic_dist', 'N_magic_dist', 'asymmetry'],
                'useful': ['valence_nucleons', 'BE_per_A']
            },
            'MM_QM': {
                'critical': ['SPIN', 'PARITY', 'beta_2', 'schmidt_nearest'],
                'important': ['valence_nucleons', 'Z_magic_dist', 'N_magic_dist'],
                'useful': ['schmidt_deviation', 'collective_parameter']
            }
        }

    def generate_base_combinations(self) -> Dict[str, List[str]]:
        """
        Temel kombinasyonları oluştur: AZN, AZNS, AZNP, AZNSP

        Returns:
            Dict[str, List[str]]: Feature set ismi -> feature listesi
        """
        base = self.BASE_FEATURES['AZN']
        feature_sets = {}

        # AZN (base)
        feature_sets['AZN'] = base.copy()

        # AZN + S
        feature_sets['AZNS'] = base + self.BASIC_ADDONS['S']

        # AZN + P
        feature_sets['AZNP'] = base + self.BASIC_ADDONS['P']

        # AZN + SP
        feature_sets['AZNSP'] = base + self.BASIC_ADDONS['SP']

        return feature_sets

    def generate_physics_combinations(self,
                                      include_groups: List[str] = None,
                                      max_features: int = None) -> Dict[str, List[str]]:
        """
        Fizik feature gruplarını kombine et

        Args:
            include_groups: Dahil edilecek grup isimleri (None = hepsi)
            max_features: Maksimum feature sayısı (None = sınırsız)

        Returns:
            Dict[str, List[str]]: Feature set ismi -> feature listesi
        """
        feature_sets = {}

        # Tüm physics gruplarını topla
        all_groups = {
            **self.DEFORMATION_GROUP,
            **self.PAIRING_GROUP,
            **self.MAGIC_GROUP,
            **self.BINDING_GROUP,
            **self.SEMF_GROUP,
            **self.SHELL_GROUP,
            **self.SCHMIDT_GROUP,
            **self.COLLECTIVE_GROUP,
            **self.ASYMMETRY_GROUP,
            **self.COMBINED_GROUPS
        }

        # Filtreleme
        if include_groups:
            all_groups = {k: v for k, v in all_groups.items() if k in include_groups}

        # Her grubu ekle
        for group_name, features in all_groups.items():
            if max_features is None or len(features) <= max_features:
                feature_sets[group_name] = features

        return feature_sets

    def generate_full_combinations(self,
                                   base_variants: List[str] = None,
                                   physics_groups: List[str] = None,
                                   max_combinations: int = None) -> Dict[str, List[str]]:
        """
        Tam kombinasyonları oluştur: Base + Physics

        Örnek: AZN_beta, AZNS_p, AZNSP_beta_p, vb.

        Args:
            base_variants: Kullanılacak base varyantlar ['AZN', 'AZNS', ...]
            physics_groups: Kullanılacak physics grupları
            max_combinations: Maksimum kombinasyon sayısı

        Returns:
            Dict[str, List[str]]: Feature set ismi -> feature listesi
        """
        feature_sets = {}

        # Base kombinasyonları al
        base_combos = self.generate_base_combinations()
        if base_variants:
            base_combos = {k: v for k, v in base_combos.items() if k in base_variants}

        # Physics kombinasyonları al
        physics_combos = self.generate_physics_combinations(include_groups=physics_groups)

        # Her base için her physics grubunu ekle
        count = 0
        for base_name, base_features in base_combos.items():
            for phys_name, phys_features in physics_combos.items():
                # Kombinasyon ismini oluştur
                combo_name = f"{base_name}_{phys_name}"

                # Feature'ları birleştir (unique)
                combined = list(dict.fromkeys(base_features + phys_features))

                feature_sets[combo_name] = combined

                count += 1
                if max_combinations and count >= max_combinations:
                    return feature_sets

        return feature_sets

    def generate_target_optimized_sets(self,
                                      target_name: str,
                                      complexity_levels: List[str] = None) -> Dict[str, List[str]]:
        """
        Target'a özel optimize edilmiş feature setleri oluştur

        Args:
            target_name: 'MM', 'QM', 'Beta_2', 'MM_QM'
            complexity_levels: ['minimal', 'basic', 'standard', 'advanced', 'ultra']

        Returns:
            Dict[str, List[str]]: Feature set ismi -> feature listesi
        """
        if complexity_levels is None:
            complexity_levels = ['minimal', 'basic', 'standard', 'advanced', 'ultra']

        feature_sets = {}
        base = self.BASE_FEATURES['AZN']

        if target_name not in self.TARGET_RECOMMENDED:
            raise ValueError(f"Unknown target: {target_name}")

        rec = self.TARGET_RECOMMENDED[target_name]

        # Minimal: Sadece base
        if 'minimal' in complexity_levels:
            feature_sets[f'{target_name}_minimal'] = base.copy()

        # Basic: Base + critical
        if 'basic' in complexity_levels:
            feature_sets[f'{target_name}_basic'] = base + rec['critical']

        # Standard: Base + critical + important
        if 'standard' in complexity_levels:
            feature_sets[f'{target_name}_standard'] = (
                base + rec['critical'] + rec['important']
            )

        # Advanced: Base + critical + important + useful
        if 'advanced' in complexity_levels:
            feature_sets[f'{target_name}_advanced'] = (
                base + rec['critical'] + rec['important'] + rec['useful']
            )

        # Ultra: Everything important for this target
        if 'ultra' in complexity_levels:
            ultra_features = base + rec['critical'] + rec['important'] + rec['useful']

            # Target'a özgü ek feature'lar
            if target_name == 'MM':
                ultra_features += ['BE_per_A', 'Z_shell_gap', 'N_shell_gap']
            elif target_name == 'QM':
                ultra_features += ['BE_per_A', 'asymmetry', 'rotational_constant']
            elif target_name == 'Beta_2':
                ultra_features += ['BE_total', 'shell_correction', 'S2n', 'S2p']
            elif target_name == 'MM_QM':
                ultra_features += ['BE_per_A', 'asymmetry', 'shell_correction']

            # Unique yap
            ultra_features = list(dict.fromkeys(ultra_features))
            feature_sets[f'{target_name}_ultra'] = ultra_features

        return feature_sets

    def generate_all_for_target(self,
                               target_name: str,
                               include_base_combos: bool = True,
                               include_physics_combos: bool = True,
                               include_optimized: bool = True,
                               max_total: int = None) -> Dict[str, List[str]]:
        """
        Bir target için TÜM olası kombinasyonları oluştur

        Bu fonksiyon binlerce feature set üretebilir!

        Args:
            target_name: Target ismi
            include_base_combos: Base kombinasyonları dahil et
            include_physics_combos: Physics kombinasyonları dahil et
            include_optimized: Optimize edilmiş setleri dahil et
            max_total: Maksimum toplam set sayısı

        Returns:
            Dict[str, List[str]]: Tüm feature setleri
        """
        all_feature_sets = {}

        # 1. Base kombinasyonları
        if include_base_combos:
            base_sets = self.generate_base_combinations()
            for name, features in base_sets.items():
                all_feature_sets[f"{target_name}_{name}"] = features

        # 2. Optimize edilmiş setler
        if include_optimized:
            opt_sets = self.generate_target_optimized_sets(target_name)
            all_feature_sets.update(opt_sets)

        # 3. Full kombinasyonlar (Base + Physics)
        if include_physics_combos:
            # Tüm base varyantlar için tüm physics grupları
            full_combos = self.generate_full_combinations(
                max_combinations=max_total
            )
            for name, features in full_combos.items():
                all_feature_sets[f"{target_name}_{name}"] = features

                if max_total and len(all_feature_sets) >= max_total:
                    break

        return all_feature_sets

    def generate_multi_group_combinations(self,
                                          base_variants: List[str] = None,
                                          physics_groups: List[str] = None,
                                          max_groups_per_combo: int = 3,
                                          max_combinations: int = None) -> Dict[str, List[str]]:
        """
        Çoklu physics gruplarını kombine et (THOUSANDS of combinations!)

        Örnek: AZN_beta_p_magic (3 physics grubu birleştirilmiş)

        Args:
            base_variants: Base varyantlar
            physics_groups: Kullanılacak physics grupları
            max_groups_per_combo: Maksimum kaç physics grubu birleştirilebilir
            max_combinations: Maksimum kombinasyon sayısı

        Returns:
            Dict[str, List[str]]: Feature sets
        """
        feature_sets = {}

        # Base kombinasyonları al
        base_combos = self.generate_base_combinations()
        if base_variants:
            base_combos = {k: v for k, v in base_combos.items() if k in base_variants}

        # Physics kombinasyonları al
        all_physics = self.generate_physics_combinations(include_groups=physics_groups)
        physics_names = list(all_physics.keys())

        count = 0

        # Her base için
        for base_name, base_features in base_combos.items():

            # 1-group, 2-group, ..., max_groups_per_combo
            for n_groups in range(1, max_groups_per_combo + 1):

                # Physics gruplarından n_groups tane seç
                for combo_tuple in combinations(physics_names, n_groups):

                    # Combo ismini oluştur
                    combo_name = f"{base_name}_{'_'.join(combo_tuple)}"

                    # Tüm grupların feature'larını birleştir
                    combined_features = base_features.copy()
                    for phys_group in combo_tuple:
                        combined_features.extend(all_physics[phys_group])

                    # Unique yap
                    combined_features = list(dict.fromkeys(combined_features))

                    feature_sets[combo_name] = combined_features

                    count += 1
                    if max_combinations and count >= max_combinations:
                        return feature_sets

        return feature_sets

    def count_possible_combinations(self,
                                    max_groups_per_combo: int = 3) -> Dict[str, int]:
        """
        Oluşturulabilecek toplam kombinasyon sayısını hesapla

        Args:
            max_groups_per_combo: Multi-group kombinasyonlarda max grup sayısı

        Returns:
            Dict[str, int]: Farklı kategorilerde kombinasyon sayıları
        """
        counts = {}

        # Base kombinasyonları
        base = self.generate_base_combinations()
        counts['base_combinations'] = len(base)

        # Physics grupları
        physics = self.generate_physics_combinations()
        counts['physics_groups'] = len(physics)
        n_physics = len(physics)

        # Single-group kombinasyonlar (base x physics)
        counts['base_x_physics_1group'] = counts['base_combinations'] * counts['physics_groups']

        # Multi-group kombinasyonlar
        # C(n, 2) + C(n, 3) + ... = sum of combinations
        multi_group_count = 0
        for k in range(2, max_groups_per_combo + 1):
            # Binomial coefficient: n! / (k! * (n-k)!)
            from math import comb
            multi_group_count += comb(n_physics, k)

        counts['base_x_physics_multigroup'] = counts['base_combinations'] * multi_group_count

        # Target-optimized (her target için 5 complexity level)
        counts['target_optimized'] = 4 * 5  # 4 targets × 5 levels

        # TOTAL olası kombinasyonlar
        counts['total_possible'] = (
            counts['base_combinations'] +
            counts['physics_groups'] +
            counts['base_x_physics_1group'] +
            counts['base_x_physics_multigroup'] +
            counts['target_optimized']
        )

        return counts

    def export_feature_sets_summary(self, feature_sets: Dict[str, List[str]]) -> str:
        """
        Feature setlerinin özetini oluştur

        Args:
            feature_sets: Feature set dictionary

        Returns:
            str: Özet rapor
        """
        summary = []
        summary.append("=" * 80)
        summary.append("FEATURE SETS SUMMARY")
        summary.append("=" * 80)
        summary.append(f"\nToplam Feature Set Sayısı: {len(feature_sets)}")
        summary.append("\n" + "-" * 80)

        # Feature sayısına göre grupla
        by_count = {}
        for name, features in feature_sets.items():
            count = len(features)
            if count not in by_count:
                by_count[count] = []
            by_count[count].append(name)

        summary.append("\nFeature Sayısına Göre Dağılım:")
        for count in sorted(by_count.keys()):
            summary.append(f"  {count} features: {len(by_count[count])} sets")

        summary.append("\n" + "-" * 80)
        summary.append("\nÖrnek Feature Sets:")
        for i, (name, features) in enumerate(list(feature_sets.items())[:10]):
            summary.append(f"\n{i+1}. {name} ({len(features)} features):")
            summary.append(f"   {features}")

        if len(feature_sets) > 10:
            summary.append(f"\n... ve {len(feature_sets) - 10} tane daha")

        summary.append("\n" + "=" * 80)

        return "\n".join(summary)


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def create_comprehensive_feature_sets(target_name: str = None,
                                      max_sets: int = 1000) -> Dict[str, List[str]]:
    """
    Comprehensive feature set koleksiyonu oluştur

    Args:
        target_name: Specific target için (None = genel)
        max_sets: Maksimum set sayısı

    Returns:
        Dict[str, List[str]]: Feature sets
    """
    builder = FeatureSetBuilder()

    if target_name:
        return builder.generate_all_for_target(
            target_name=target_name,
            max_total=max_sets
        )
    else:
        # Genel koleksiyon
        all_sets = {}

        # Base kombinasyonları
        all_sets.update(builder.generate_base_combinations())

        # Physics kombinasyonları
        all_sets.update(builder.generate_physics_combinations())

        # Full kombinasyonlar
        full = builder.generate_full_combinations(max_combinations=max_sets)
        all_sets.update(full)

        return all_sets


def get_feature_set_statistics() -> Dict:
    """Feature set istatistiklerini al"""
    builder = FeatureSetBuilder()
    return builder.count_possible_combinations()


# ============================================================================
# TEST/DEMO
# ============================================================================

if __name__ == "__main__":
    print("Feature Set Builder v2.0.0")
    print("=" * 80)

    builder = FeatureSetBuilder()

    # İstatistikleri göster
    print("\n📊 OLASI KOMBİNASYON SAYILARI:")
    stats = builder.count_possible_combinations(max_groups_per_combo=3)
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    print(f"\n  💡 TOPLAM: {stats['total_possible']:,} farklı feature set kombinasyonu!")

    # Örnek: MM target için feature setler
    print("\n\n🎯 MM TARGET İÇİN ÖRNEK FEATURE SETS:")
    mm_sets = builder.generate_target_optimized_sets('MM')
    for name, features in mm_sets.items():
        print(f"\n{name} ({len(features)} features):")
        print(f"  {features}")

    # Örnek: Base kombinasyonlar
    print("\n\n🔬 BASE KOMBİNASYONLAR:")
    base_sets = builder.generate_base_combinations()
    for name, features in base_sets.items():
        print(f"  {name}: {features}")

    # Örnek: Single-group kombinasyonlar (ilk 5)
    print("\n\n🚀 SINGLE-GROUP KOMBİNASYONLAR (ilk 5):")
    single_sets = builder.generate_full_combinations(
        base_variants=['AZN', 'AZNS'],
        physics_groups=['beta', 'p', 'magic'],
        max_combinations=5
    )
    for name, features in single_sets.items():
        print(f"  {name} ({len(features)}): {features}")

    # Örnek: Multi-group kombinasyonlar (ilk 5)
    print("\n\n🔥 MULTI-GROUP KOMBİNASYONLAR (ilk 5):")
    multi_sets = builder.generate_multi_group_combinations(
        base_variants=['AZN', 'AZNSP'],
        physics_groups=['beta', 'p', 'magic', 'BE'],
        max_groups_per_combo=2,
        max_combinations=5
    )
    for name, features in multi_sets.items():
        print(f"  {name} ({len(features)}): {features}")

    # Comprehensive örnek
    print("\n\n💪 COMPREHENSIVE GENERATION ÖRNEĞİ:")
    print("  Generating first 100 comprehensive combinations...")
    comprehensive = builder.generate_multi_group_combinations(
        max_groups_per_combo=3,
        max_combinations=100
    )
    print(f"  ✅ Generated {len(comprehensive)} feature sets")
    print(f"  📊 Feature count range: {min(len(f) for f in comprehensive.values())} - {max(len(f) for f in comprehensive.values())}")

    print("\n" + "=" * 80)
    print("✅ Feature Set Builder hazır!")
    print(f"💡 Bu sistem binlerce farklı feature kombinasyonu üretebilir!")
