"""
Stratejik Veri Seti Oluşturma Modülü
Strategic Dataset Generation Module
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import json
import scipy.io as sio
import logging

import sys
# sys.path.append('..') - REMOVED
from nuclear_physics_modules.constants import *
from .nuclei_distribution_analyzer import NucleiDistributionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StratifiedSampler:
    """Katmanlı örnekleme sınıfı"""
    
    def __init__(self):
        pass
    
    def create_stratification_key(self, df):
        """
        Katmanlaştırma anahtarı oluştur
        
        Anahtar: mass_group + spin_bin + parity + nucleus_type
        """
        df = df.copy()
        
        # Spin'i 0.5'lik gruplara böl
        df['spin_bin'] = (df['SPIN'] * 2).round() / 2
        
        # Katmanlaştırma anahtarı
        df['stratification_key'] = (
            df['mass_group'].astype(str) + '_' +
            df['spin_bin'].astype(str) + '_' +
            df['PARITY'].astype(str) + '_' +
            df['nucleus_type'].astype(str)
        )
        
        return df
    
    def stratified_sample(self, df, n_samples, random_state=42):
        """
        Katmanlı örnekleme yap
        
        Args:
            df: DataFrame
            n_samples: Hedef örnek sayısı
            random_state: Random seed
            
        Returns:
            DataFrame (örneklenmiş)
        """
        if n_samples >= len(df):
            return df
        
        # Katmanlaştırma anahtarı oluştur
        df = self.create_stratification_key(df)
        
        # Her katmandan orantılı örnek al
        strata = df['stratification_key'].unique()
        sampled_dfs = []
        
        for stratum in strata:
            stratum_df = df[df['stratification_key'] == stratum]
            
            # Bu katmandan alınacak örnek sayısı
            stratum_size = len(stratum_df)
            stratum_target = int(n_samples * (stratum_size / len(df)))
            
            # En az 1 örnek al
            stratum_target = max(1, stratum_target)
            
            # Eğer katmanda yeterli örnek yoksa hepsini al
            if stratum_target >= stratum_size:
                sampled_dfs.append(stratum_df)
            else:
                # İzotop çeşitliliğini koru
                sampled = self._sample_with_isotope_diversity(
                    stratum_df, stratum_target, random_state
                )
                sampled_dfs.append(sampled)
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        
        # Eğer hedef sayıya ulaşamadıysak, eksik olanı rastgele ekle
        if len(result) < n_samples:
            remaining = n_samples - len(result)
            remaining_indices = df.index.difference(result.index)
            additional = df.loc[remaining_indices].sample(
                n=min(remaining, len(remaining_indices)),
                random_state=random_state
            )
            result = pd.concat([result, additional], ignore_index=True)
        
        # Eğer hedef sayıyı aştıysak, rastgele azalt
        if len(result) > n_samples:
            result = result.sample(n=n_samples, random_state=random_state)
        
        return result.drop(columns=['stratification_key', 'spin_bin'])
    
    def _sample_with_isotope_diversity(self, df, n, random_state):
        """İzotop çeşitliliğini koruyarak örnekle"""
        
        # Her Z'den eşit sayıda al
        z_values = df['Z'].unique()
        samples_per_z = max(1, n // len(z_values))
        
        sampled_dfs = []
        for z in z_values:
            z_df = df[df['Z'] == z]
            n_from_z = min(samples_per_z, len(z_df))
            sampled_dfs.append(z_df.sample(n=n_from_z, random_state=random_state))
        
        result = pd.concat(sampled_dfs)
        
        # Hedef sayıya ulaşmadıysak rastgele ekle
        if len(result) < n:
            remaining = n - len(result)
            remaining_indices = df.index.difference(result.index)
            if len(remaining_indices) > 0:
                additional = df.loc[remaining_indices].sample(
                    n=min(remaining, len(remaining_indices)),
                    random_state=random_state
                )
                result = pd.concat([result, additional])
        
        return result


class DatasetGenerator:
    """Ana veri seti oluşturucu"""
    
    def __init__(self, base_path='ANFIS_Datasets'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.sampler = StratifiedSampler()
        self.generated_datasets = []
        self.distribution_analyzer = NucleiDistributionAnalyzer(
            output_dir=self.base_path / 'distribution_reports'
        )
    
    def generate_all_datasets(self, df, targets=None, nucleus_counts=None,
                            scenarios=None, anomaly_modes=None,
                            feature_sets=None, scaling_methods=None,
                            sampling_methods=None):
        """
        Tüm veri seti kombinasyonlarını oluştur
        
        Args:
            df: Zenginleştirilmiş ve anomali işaretli DataFrame
            targets: Hedef değişken setleri
            ... diğer parametreler
        """
        # Default parametreler
        targets = targets or TARGETS
        nucleus_counts = nucleus_counts or NUCLEUS_COUNTS
        scenarios = scenarios or SCENARIOS
        anomaly_modes = anomaly_modes or ANOMALY_MODES
        scaling_methods = scaling_methods or SCALING_METHODS
        sampling_methods = sampling_methods or SAMPLING_METHODS
        
        logger.info("Veri seti oluşturma başlıyor...")
        logger.info(f"  Toplam kombinasyon sayısı: ~{self._estimate_total_combinations()}")
        
        dataset_count = 0
        
        for target_name, target_cols in targets.items():
            logger.info(f"\n→ Hedef: {target_name}")
            
            # Hedef için özellik setlerini belirle
            if target_name == 'Beta_2':
                feature_sets_to_use = BETA2_FEATURE_SETS
            else:
                feature_sets_to_use = STANDARD_FEATURE_SETS
            
            # Target için veriyi filtrele
            target_df = self._filter_for_target(df, target_cols)
            
            if len(target_df) == 0:
                logger.warning(f"  ⚠ {target_name} için veri yok, atlanıyor")
                continue
            
            for nucleus_count in nucleus_counts:
                for scenario_name, (train_r, check_r, test_r) in scenarios.items():
                    for anomaly_mode in anomaly_modes:
                        for feature_set_name, features in feature_sets_to_use.items():
                            for scaling in scaling_methods:
                                for sampling in sampling_methods:
                                    
                                    try:
                                        self._generate_single_dataset(
                                            target_df, target_name, target_cols,
                                            nucleus_count, scenario_name,
                                            (train_r, check_r, test_r),
                                            anomaly_mode, feature_set_name,
                                            features, scaling, sampling
                                        )
                                        dataset_count += 1
                                        
                                        if dataset_count % 50 == 0:
                                            logger.info(f"  ✓ {dataset_count} veri seti oluşturuldu")
                                    
                                    except Exception as e:
                                        logger.error(f"  ✗ Hata: {e}")
                                        continue
        
        logger.info(f"\n✓ Toplam {dataset_count} veri seti oluşturuldu")
        self._save_catalog()

        # Master çekirdek kataloğu oluştur
        logger.info("\n→ Master çekirdek kataloğu oluşturuluyor...")
        try:
            self.distribution_analyzer.create_master_nuclei_catalog(
                df, self.base_path / 'Master_Nuclei_Catalog.xlsx'
            )
            logger.info("✓ Master çekirdek kataloğu oluşturuldu")
        except Exception as e:
            logger.error(f"✗ Master katalog hatası: {e}")
    
    def _generate_single_dataset(self, df, target_name, target_cols,
                                nucleus_count, scenario_name, split_ratios,
                                anomaly_mode, feature_set_name, features,
                                scaling, sampling):
        """Tek bir veri seti oluştur"""
        
        # Anomali filtreleme
        if anomaly_mode == 'anomalisiz':
            df_filtered = df[df['is_anomaly'] == False].copy()
        else:
            df_filtered = df.copy()
        
        # Nükleus sayısı seçimi
        if nucleus_count == 'ALL':
            df_selected = df_filtered
        else:
            if sampling == 'stratified':
                df_selected = self.sampler.stratified_sample(
                    df_filtered, nucleus_count, random_state=42
                )
            else:  # random
                df_selected = df_filtered.sample(
                    n=min(nucleus_count, len(df_filtered)),
                    random_state=42
                )
        
        # Özellikleri filtrele (NaN içermeyenler)
        available_features = [f for f in features if f in df_selected.columns]
        
        # Veri hazırlığı
        X = df_selected[available_features]
        y = df_selected[target_cols]
        
        # NaN kontrolü
        mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            logger.warning(f"  ⚠ Yetersiz veri ({len(X)} örnek), atlanıyor")
            return
        
        # Train/Check/Test split
        train_r, check_r, test_r = split_ratios
        
        # İlk split: train + check vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_r, random_state=42, shuffle=True
        )
        
        # İkinci split: train vs check
        check_ratio_adjusted = check_r / (train_r + check_r)
        X_train, X_check, y_train, y_check = train_test_split(
            X_temp, y_temp, test_size=check_ratio_adjusted, 
            random_state=42, shuffle=True
        )
        
        # Scaling
        scaler = None
        if scaling == 'standard':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_check_scaled = scaler.transform(X_check)
            X_test_scaled = scaler.transform(X_test)
        elif scaling == 'robust':
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_check_scaled = scaler.transform(X_check)
            X_test_scaled = scaler.transform(X_test)
        else:  # none
            X_train_scaled = X_train.values
            X_check_scaled = X_check.values
            X_test_scaled = X_test.values
        
        # DataFrame'lere geri çevir
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features)
        X_check_scaled = pd.DataFrame(X_check_scaled, columns=available_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features)
        
        # Dataset adı
        dataset_name = self._create_dataset_name(
            target_name, nucleus_count, scenario_name,
            anomaly_mode, feature_set_name, scaling, sampling
        )
        
        # Kaydetme yolu
        save_path = self.base_path / target_name / scenario_name / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Veri setlerini kaydet
        self._save_dataset_files(
            save_path, X_train_scaled, X_check_scaled, X_test_scaled,
            y_train, y_check, y_test, scaler, available_features,
            target_cols, df_selected
        )
        
        # Metadata
        metadata = {
            'dataset_name': dataset_name,
            'target': target_name,
            'target_columns': target_cols,
            'features': available_features,
            'n_features': len(available_features),
            'nucleus_count': nucleus_count if nucleus_count != 'ALL' else len(df_selected),
            'scenario': scenario_name,
            'split_ratios': {'train': train_r, 'check': check_r, 'test': test_r},
            'anomaly_mode': anomaly_mode,
            'feature_set': feature_set_name,
            'scaling': scaling,
            'sampling': sampling,
            'n_train': len(X_train_scaled),
            'n_check': len(X_check_scaled),
            'n_test': len(X_test_scaled),
            'total_samples': len(X_train_scaled) + len(X_check_scaled) + len(X_test_scaled)
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Kataloga ekle
        self.generated_datasets.append(metadata)
    
    def _filter_for_target(self, df, target_cols):
        """Hedef değişken için veriyi filtrele"""
        df_copy = df.copy()
        
        # Hedef sütunlarda NaN olmayanları seç
        for col in target_cols:
            if col in df_copy.columns:
                df_copy = df_copy[df_copy[col].notna()]
        
        return df_copy
    
    def _create_dataset_name(self, target, n_count, scenario, anomaly,
                           feature_set, scaling, sampling):
        """Veri seti adı oluştur"""
        n_str = str(n_count)
        return f"{target}_{n_str}_{scenario}_{anomaly}_{feature_set}_{scaling}_{sampling}"
    
    def _save_dataset_files(self, path, X_train, X_check, X_test,
                           y_train, y_check, y_test, scaler,
                           features, target_cols, df_full):
        """Veri seti dosyalarını kaydet"""
        
        # 1. CSV formatı
        pd.concat([X_train, y_train.reset_index(drop=True)], axis=1).to_csv(
            path / 'train.csv', index=False
        )
        pd.concat([X_check, y_check.reset_index(drop=True)], axis=1).to_csv(
            path / 'check.csv', index=False
        )
        pd.concat([X_test, y_test.reset_index(drop=True)], axis=1).to_csv(
            path / 'test.csv', index=False
        )
        
        # 2. Excel formatı
        with pd.ExcelWriter(path / 'train.xlsx', engine='openpyxl') as writer:
            pd.concat([X_train, y_train.reset_index(drop=True)], axis=1).to_excel(
                writer, index=False
            )
        
        with pd.ExcelWriter(path / 'check.xlsx', engine='openpyxl') as writer:
            pd.concat([X_check, y_check.reset_index(drop=True)], axis=1).to_excel(
                writer, index=False
            )
        
        with pd.ExcelWriter(path / 'test.xlsx', engine='openpyxl') as writer:
            pd.concat([X_test, y_test.reset_index(drop=True)], axis=1).to_excel(
                writer, index=False
            )
        
        # 3. MATLAB formatı (.mat)
        train_mat = {
            'train_input': X_train.values,
            'train_output': y_train.values,
            'feature_names': features,
            'target_names': target_cols
        }
        sio.savemat(path / 'train.mat', train_mat)
        
        check_mat = {
            'check_input': X_check.values,
            'check_output': y_check.values
        }
        sio.savemat(path / 'check.mat', check_mat)
        
        test_mat = {
            'test_input': X_test.values,
            'test_output': y_test.values
        }
        sio.savemat(path / 'test.mat', test_mat)
        
        # 4. Scaler kaydet
        if scaler is not None:
            joblib.dump(scaler, path / 'scaler.pkl')
        
        # 5. Nükleus seçimi
        nucleus_selection = df_full[['NUCLEUS', 'A', 'Z', 'N']].copy()

        # Ek bilgiler varsa ekle
        for col in ['SPIN', 'PARITY', 'Beta_2', 'MM', 'Q', 'p_factor']:
            if col in df_full.columns:
                nucleus_selection[col] = df_full[col]

        nucleus_selection.to_excel(path / 'nucleus_selection.xlsx', index=False)

        # 6. Dağılım analizi raporu oluştur
        try:
            analysis = self.distribution_analyzer.analyze_dataset(
                df_full, dataset_name
            )
            self.distribution_analyzer.create_distribution_report(
                analysis, path / 'nuclei_distribution_report.xlsx'
            )
        except Exception as e:
            logger.warning(f"  ⚠ Dağılım raporu oluşturulamadı: {e}")
    
    def _estimate_total_combinations(self):
        """Toplam kombinasyon sayısını tahmin et"""
        n_targets = len(TARGETS)
        n_nucleus_counts = len(NUCLEUS_COUNTS)
        n_scenarios = len(SCENARIOS)
        n_anomaly_modes = len(ANOMALY_MODES)
        n_feature_sets_avg = (len(STANDARD_FEATURE_SETS) + len(BETA2_FEATURE_SETS)) / 2
        n_scaling = len(SCALING_METHODS)
        n_sampling = len(SAMPLING_METHODS)
        
        total = (n_targets * n_nucleus_counts * n_scenarios * n_anomaly_modes *
                n_feature_sets_avg * n_scaling * n_sampling)
        
        return int(total)
    
    def _save_catalog(self):
        """Veri seti kataloğunu kaydet"""
        catalog_df = pd.DataFrame(self.generated_datasets)
        
        output_path = self.base_path / 'Dataset_Catalog.xlsx'
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Tam katalog
            catalog_df.to_excel(writer, sheet_name='All_Datasets', index=False)
            
            # Hedef değişkene göre
            for target in catalog_df['target'].unique():
                target_df = catalog_df[catalog_df['target'] == target]
                target_df.to_excel(writer, sheet_name=f'Target_{target}', index=False)
            
            # Özet istatistikler
            summary = {
                'Metric': [
                    'Total Datasets',
                    'Targets',
                    'Avg Datasets per Target',
                    'Min Samples',
                    'Max Samples',
                    'Avg Samples'
                ],
                'Value': [
                    len(catalog_df),
                    catalog_df['target'].nunique(),
                    len(catalog_df) / catalog_df['target'].nunique(),
                    catalog_df['total_samples'].min(),
                    catalog_df['total_samples'].max(),
                    catalog_df['total_samples'].mean()
                ]
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"✓ Veri seti kataloğu kaydedildi: {output_path}")


class ControlGroupGenerator:
    """Kontrol grubu oluşturucu (blind test için)"""
    
    def __init__(self):
        pass
    
    def generate_control_groups(self, df_full, df_used_in_training, targets):
        """
        Kontrol grupları oluştur
        
        Args:
            df_full: Tam veri seti
            df_used_in_training: Eğitimde kullanılan nükleuslar
            targets: Hedef değişkenler
        """
        logger.info("Kontrol grupları oluşturuluyor...")
        
        # Eğitimde kullanılmayan nükleusları bul
        used_nuclei = set(df_used_in_training['NUCLEUS'].unique())
        control_df = df_full[~df_full['NUCLEUS'].isin(used_nuclei)]
        
        logger.info(f"  Kontrol grubu adayı: {len(control_df)} nükleus")
        
        for target_name, target_cols in targets.items():
            # Hedef için filtrele
            target_control = control_df.copy()
            for col in target_cols:
                if col in target_control.columns:
                    target_control = target_control[target_control[col].notna()]
            
            if len(target_control) == 0:
                logger.warning(f"  ⚠ {target_name} için kontrol grubu yok")
                continue
            
            # Çeşitlilik için stratejik seçim
            selected = self._select_diverse_control_group(target_control)
            
            # Kaydet
            output_path = Path('ANFIS_Datasets') / f'Control_Group_{target_name}.xlsx'
            selected.to_excel(output_path, index=False)
            
            logger.info(f"  ✓ {target_name}: {len(selected)} nükleus")
    
    def _select_diverse_control_group(self, df, max_size=100):
        """Çeşitli kontrol grubu seç"""
        
        selected_dfs = []
        
        # 1. Magik sayı komşuları
        magic_neighbors = df[
            (df['Z_magic_dist'] <= 2) | (df['N_magic_dist'] <= 2)
        ]
        if len(magic_neighbors) > 0:
            selected_dfs.append(
                magic_neighbors.sample(n=min(20, len(magic_neighbors)), random_state=42)
            )
        
        # 2. Deformasyon bölgeleri
        deformed = df[df['Beta_2'].abs() > 0.2]
        if len(deformed) > 0:
            selected_dfs.append(
                deformed.sample(n=min(15, len(deformed)), random_state=42)
            )
        
        # 3. Her kütle grubundan
        for mass_group in df['mass_group'].unique():
            mg_df = df[df['mass_group'] == mass_group]
            selected_dfs.append(
                mg_df.sample(n=min(15, len(mg_df)), random_state=42)
            )
        
        # Birleştir ve unique yap
        result = pd.concat(selected_dfs).drop_duplicates(subset=['NUCLEUS'])
        
        # Max boyuta kırp
        if len(result) > max_size:
            result = result.sample(n=max_size, random_state=42)
        
        return result


def main():
    """Test fonksiyonu"""
    # Örnek veri (gerçekte zenginleştirilmiş veri olmalı)
    test_data = pd.DataFrame({
        'NUCLEUS': [f'Nucleus_{i}' for i in range(500)],
        'A': np.random.randint(20, 250, 500),
        'Z': np.random.randint(10, 100, 500),
        'N': np.random.randint(10, 150, 500),
        'SPIN': np.random.uniform(0, 10, 500),
        'PARITY': np.random.choice([-1, 1], 500),
        'MM': np.random.uniform(-3, 3, 500),
        'Q': np.random.uniform(-0.5, 0.5, 500),
        'Beta_2': np.random.uniform(-0.3, 0.3, 500),
        'p_factor': np.random.uniform(10, 50, 500),
        'BE_per_A': np.random.uniform(7, 9, 500),
        'Z_magic_dist': np.random.randint(0, 10, 500),
        'N_magic_dist': np.random.randint(0, 10, 500),
        'magic_character': np.random.uniform(0, 1, 500),
        'is_anomaly': np.random.choice([True, False], 500, p=[0.1, 0.9]),
        'nucleus_type': np.random.choice(['even-even', 'odd-odd', 'odd-even'], 500),
        'mass_group': np.random.choice(['light', 'medium', 'heavy'], 500)
    })
    
    # Veri seti oluşturucu
    generator = DatasetGenerator('output/test_datasets')
    
    # Küçük bir subset ile test
    test_targets = {'MM': ['MM']}
    test_nucleus_counts = [50, 75]
    test_scenarios = {'S70': (0.7, 0.15, 0.15)}
    
    generator.generate_all_datasets(
        test_data,
        targets=test_targets,
        nucleus_counts=test_nucleus_counts,
        scenarios=test_scenarios,
        anomaly_modes=['anomalisiz'],
        scaling_methods=['standard'],
        sampling_methods=['random']
    )
    
    print("\n✓ Test tamamlandı")


if __name__ == "__main__":
    main()