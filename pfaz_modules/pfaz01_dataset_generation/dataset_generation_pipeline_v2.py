"""
FAZ 3: Dataset Generation Pipeline
===================================

Kapsamlı dataset oluşturma sistemi:
- Teorik hesaplamalar (SEMF, Shell Model, Woods-Saxon, Nilsson, Schmidt)
- QM filtreleme (target-based)
- Çoklu çekirdek sayıları (50, 75, 100, 150, 175, 200, 250, 300, 350)
- Çoklu targetler (MM, QM, MM_QM, Beta_2)
- Feature kombinasyonları (Basic, Extended, Full, ANFIS variants)
- I/O Configurations (3In1Out, 4In1Out, 5InAdv, etc.)
- Scenario System (S70, S80)
- Enhanced naming convention (7-part format)
- Scaling options (NoScaling, Standard, Robust)
- Stratified sampling (Random, Stratified, StratifiedMagic, StratifiedHybrid)
- Kalite kontrolü ve validasyon
- Otomatik raporlama

Author: Nuclear Physics AI Project
Version: 3.0.0 (FAZ 3)
Date: 2025-11-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Project imports
from physics_modules.theoretical_calculations_manager import TheoreticalCalculationsManager
from .qm_filter_manager import QMFilterManager
from .data_quality_modules import OutlierHandler, DataValidator
from .excluded_nuclei_tracker import ExcludedNucleiTracker
from .feature_combination_manager import FeatureCombinationManager, get_default_feature_sets
from .io_config_manager import InputOutputConfigManager, ScenarioManager
from .scaling_manager import ScalingManager
from .sampling_manager import SamplingManager, get_sampling_statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_for_json(obj):
    """
    Recursively sanitize objects for JSON serialization.
    Converts tuple keys to strings and handles non-serializable types.

    Args:
        obj: Object to sanitize (dict, list, or primitive)

    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, dict):
        # Convert any tuple keys to string representations
        sanitized = {}
        for key, value in obj.items():
            # Convert tuple keys to string
            if isinstance(key, tuple):
                key = str(key)
            # Ensure key is JSON-serializable
            elif not isinstance(key, (str, int, float, bool, type(None))):
                key = str(key)
            # Recursively sanitize value
            sanitized[key] = sanitize_for_json(value)
        return sanitized
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # For any other non-serializable type, convert to string
        return str(obj)


class DatasetGenerationPipelineV2:
    """
    Ana Dataset Generation Pipeline V2 (FAZ 3)

    Workflow:
    1. Ham veriyi yükle
    2. Teorik hesaplamalar ekle
    3. Target-based QM filtreleme uygula
    4. Farklı çekirdek sayıları için örnekle (stratified or random)
    5. Kalite kontrolü
    6. Dataset'leri kaydet (I/O configs, scenarios, 7-part naming)
    7. Scaling uygula (NoScaling, Standard, Robust)
    8. Metadata ve raporlar oluştur
    """

    def __init__(self,
                 source_data_path: str = None,
                 output_base_dir: str = 'generated_datasets',
                 nucleus_counts: List[int] = None,
                 targets: List[str] = None,
                 feature_sets: List[str] = None,
                 scenario: str = None,
                 scaling: str = None,
                 sampling: str = None,
                 # Backward compatibility aliases
                 aaa2_txt_path: str = None,
                 output_dir: str = None):
        """
        Args:
            source_data_path: Ham veri dosyası yolu (or use aaa2_txt_path)
            output_base_dir: Çıktı ana dizini (or use output_dir)
            nucleus_counts: Oluşturulacak dataset boyutları
            targets: Target değişkenler
            feature_sets: Feature kombinasyonları (Basic, Extended, Full)
            scenario: Split scenario (S70, S80) - Default: S70
            scaling: Scaling method (NoScaling, Standard, Robust) - Default: NoScaling
            sampling: Sampling method (Random, Stratified) - Default: Random
            aaa2_txt_path: Alias for source_data_path (backward compatibility)
            output_dir: Alias for output_base_dir (backward compatibility)
        """
        # Handle aliases
        if aaa2_txt_path is not None:
            source_data_path = aaa2_txt_path
        if output_dir is not None:
            output_base_dir = output_dir

        self.source_data_path = Path(source_data_path)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)


        # Default parameters (updated per requirements)
        # [UPDATED]: 75, 100, 150, 200, ALL (all available nuclei)
        self.nucleus_counts = nucleus_counts or [75, 100, 150, 200, 'ALL']
        self.targets = targets or ['MM', 'QM', 'MM_QM', 'Beta_2']

        # [FAZ 1 NEW]: Feature combinations
        self.feature_sets = feature_sets or get_default_feature_sets()  # ['Basic', 'Extended', 'Full']

        # [FAZ 2 NEW]: Scenario, Scaling, Sampling
        self.scenario = scenario or 'S70'  # Default: 70/15/15 split
        self.scaling = scaling or 'NoScaling'  # Default: no scaling (FAZ 3 will add options)
        self.sampling = sampling or 'Random'  # Default: random (FAZ 3 will add stratified)

        # Target column name mapping (simplified name -> actual column name)
        self.target_column_map = {
            'MM': 'MAGNETIC MOMENT [µ]',
            'QM': 'QUADRUPOLE MOMENT [Q]',
            'Q': 'QUADRUPOLE MOMENT [Q]',
            'Beta_2': 'Beta_2'
        }
        
        # Initialize exclusion tracker
        self.exclusion_tracker = ExcludedNucleiTracker()

        # Initialize managers with tracker
        self.theoretical_calc_manager = TheoreticalCalculationsManager(enable_all=True)
        self.qm_filter_manager = QMFilterManager(tracker=self.exclusion_tracker)
        self.outlier_handler = OutlierHandler(
            output_dir=self.output_base_dir / 'quality_reports',
            tracker=self.exclusion_tracker
        )
        self.data_validator = DataValidator(output_dir=self.output_base_dir / 'quality_reports')

        # [FAZ 1 NEW]: Feature combination manager
        self.feature_manager = FeatureCombinationManager()

        # [FAZ 2 NEW]: I/O Config and Scenario managers
        self.io_config_manager = InputOutputConfigManager()
        self.scenario_manager = ScenarioManager()

        # Storage
        self.raw_data = None
        self.enriched_data = None
        self.filtered_data = {}  # {target: df}
        self.generated_datasets = []
        self.generation_report = {}
        
        logger.info("=" * 80)
        logger.info("DATASET GENERATION PIPELINE INITIALIZED (FAZ 3)")
        logger.info("=" * 80)
        logger.info(f"Source data: {self.source_data_path}")
        logger.info(f"Output directory: {self.output_base_dir}")
        logger.info(f"Nucleus counts: {self.nucleus_counts}")
        logger.info(f"Targets: {self.targets}")
        logger.info(f"Feature sets: {self.feature_sets}")  # [FAZ 1]
        logger.info(f"Scenario: {self.scenario}")  # [FAZ 2]
        logger.info(f"Scaling: {self.scaling}")  # [FAZ 2]
        logger.info(f"Sampling: {self.sampling}")  # [FAZ 2]
    
    def run_complete_pipeline(self) -> Dict:
        """
        Tam pipeline'ı çalıştır
        
        Returns:
            Generation report dictionary
        """
        start_time = datetime.now()
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE DATASET GENERATION PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load raw data
        logger.info("\n[STEP 1] LOADING RAW DATA")
        logger.info("-" * 80)
        self._load_raw_data()

        # Step 2: Add theoretical calculations
        logger.info("\n[STEP 2] ADDING THEORETICAL CALCULATIONS")
        logger.info("-" * 80)
        self._add_theoretical_calculations()

        # Step 3: Apply QM filtering per target
        logger.info("\n[STEP 3] APPLYING QM FILTERING")
        logger.info("-" * 80)
        self._apply_qm_filtering()

        # Step 4: Quality control
        logger.info("\n[STEP 4] QUALITY CONTROL")
        logger.info("-" * 80)
        self._perform_quality_control()

        # Step 5: Generate datasets
        logger.info("\n[STEP 5] GENERATING DATASETS")
        logger.info("-" * 80)
        self._generate_all_datasets()

        # Step 6: Create metadata and reports
        logger.info("\n[STEP 6] CREATING METADATA & REPORTS")
        logger.info("-" * 80)
        self._create_metadata_and_reports()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Total datasets generated: {len(self.generated_datasets)}")
        logger.info(f"Output directory: {self.output_base_dir}")
        
        self.generation_report['pipeline_completed'] = True
        self.generation_report['total_duration_seconds'] = duration
        self.generation_report['completion_timestamp'] = datetime.now().isoformat()
        
        return self.generation_report
    
    def _load_raw_data(self):
        """Ham veriyi yükle"""
        logger.info(f"Loading data from: {self.source_data_path}")
        
        if not self.source_data_path.exists():
            raise FileNotFoundError(f"Source data not found: {self.source_data_path}")
        
        # Detect file format and load with UTF-8 encoding
        if self.source_data_path.suffix == '.csv':
            self.raw_data = pd.read_csv(self.source_data_path, encoding='utf-8')
        elif self.source_data_path.suffix in ['.xlsx', '.xls']:
            self.raw_data = pd.read_excel(self.source_data_path)
        elif self.source_data_path.suffix == '.tsv':
            self.raw_data = pd.read_csv(self.source_data_path, sep='\t', encoding='utf-8')
        elif self.source_data_path.suffix == '.txt':
            # Try to detect delimiter automatically for .txt files
            # First try tab-delimited, then comma, then whitespace
            try:
                self.raw_data = pd.read_csv(self.source_data_path, sep='\t', encoding='utf-8')
            except:
                try:
                    self.raw_data = pd.read_csv(self.source_data_path, sep=',', encoding='utf-8')
                except:
                    # Last resort: whitespace-delimited
                    self.raw_data = pd.read_csv(self.source_data_path, delim_whitespace=True, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {self.source_data_path.suffix}")
        
        logger.info(f"[SUCCESS] Loaded: {len(self.raw_data)} nuclei")
        logger.info(f"   Columns: {list(self.raw_data.columns)}")

        # Basic validation
        required_cols = ['A', 'Z', 'N']
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean and convert numeric columns
        self._clean_and_convert_numeric_columns()

        self.generation_report['raw_data'] = {
            'n_nuclei': len(self.raw_data),
            'n_features': len(self.raw_data.columns),
            'columns': list(self.raw_data.columns)
        }

    def _clean_and_convert_numeric_columns(self):
        """
        Clean and convert numeric columns to proper numeric types.
        Handles comma decimal separators and ensures data integrity.
        """
        logger.info("Cleaning and converting numeric columns...")

        # Columns that should always be excluded from numeric conversion
        exclude_cols = ['NUCLEUS']

        # Get columns to process (all except excluded ones)
        cols_to_process = [col for col in self.raw_data.columns if col not in exclude_cols]

        conversions_made = 0
        errors_found = []

        for col in cols_to_process:
            try:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(self.raw_data[col]):
                    continue

                # Try to convert to numeric
                # First, handle comma decimal separators (e.g., "1,024" -> "1.024")
                if self.raw_data[col].dtype == object:
                    # Replace commas with dots for decimal conversion
                    cleaned_values = self.raw_data[col].astype(str).str.replace(',', '.', regex=False)
                    # Convert to numeric, coercing errors to NaN
                    converted = pd.to_numeric(cleaned_values, errors='coerce')

                    # Only apply conversion if it resulted in at least some numeric values
                    if converted.notna().any():
                        self.raw_data[col] = converted
                        conversions_made += 1

                        # Log if any values were converted to NaN
                        nan_count = converted.isna().sum() - self.raw_data[col].isna().sum()
                        if nan_count > 0:
                            logger.warning(f"  [WARNING] Column '{col}': {nan_count} values could not be converted to numeric")
                            errors_found.append(f"{col}: {nan_count} invalid values")

            except Exception as e:
                logger.warning(f"  [WARNING] Error converting column '{col}': {e}")
                errors_found.append(f"{col}: {str(e)}")

        logger.info(f"[SUCCESS] Converted {conversions_made} columns to numeric types")
        if errors_found:
            logger.info(f"   Conversion warnings: {len(errors_found)}")

    def _add_theoretical_calculations(self):
        """Teorik hesaplamaları ekle"""
        logger.info("Adding theoretical physics calculations...")
        
        self.enriched_data = self.theoretical_calc_manager.calculate_all_theoretical_properties(
            self.raw_data,
            save_report=True
        )
        
        n_added_features = len(self.enriched_data.columns) - len(self.raw_data.columns)

        logger.info(f"[SUCCESS] Added {n_added_features} theoretical features")
        logger.info(f"   Total features now: {len(self.enriched_data.columns)}")
        
        self.generation_report['theoretical_calculations'] = {
            'n_original_features': len(self.raw_data.columns),
            'n_enriched_features': len(self.enriched_data.columns),
            'n_added_features': n_added_features,
            'calculations_performed': self.theoretical_calc_manager.calculations_done
        }
    
    def _apply_qm_filtering(self):
        """Her target için QM filtreleme uygula"""
        logger.info("Applying QM filtering for each target...")

        qm_reports = []

        for target in self.targets:
            logger.info(f"\n-> Target: {target}")

            # Get actual target column names
            target_cols = self._get_actual_column_names(target, self.enriched_data)

            # Check if target exists
            missing_targets = [col for col in target_cols if col not in self.enriched_data.columns]
            if missing_targets:
                logger.warning(f"  [WARNING] Target columns not found: {missing_targets}, skipping {target}")
                continue

            # Apply filter
            filtered_df, filter_report = self.qm_filter_manager.filter_by_target(
                self.enriched_data,
                target_name=target,
                target_cols=target_cols,
                features=list(self.enriched_data.columns)
            )

            self.filtered_data[target] = filtered_df
            filter_report['target'] = target
            qm_reports.append(filter_report)

            logger.info(f"  [SUCCESS] Filtered: {len(filtered_df)} nuclei remain")
            logger.info(f"     Removed: {filter_report['removed']} nuclei")
        
        # Save QM filter report
        self.qm_filter_manager.create_filter_report(
            qm_reports,
            output_path=self.output_base_dir / 'quality_reports' / 'qm_filter_report.xlsx'
        )
        
        self.generation_report['qm_filtering'] = {
            'targets_processed': list(self.filtered_data.keys()),
            'filter_reports': qm_reports
        }
    
    def _perform_quality_control(self):
        """Kalite kontrolü uygula"""
        logger.info("Performing quality control...")
        
        quality_reports = {}
        
        for target, df in self.filtered_data.items():
            logger.info(f"\n-> Quality control for {target}")
            
            # Identify numeric columns (exclude NUCLEUS if exists)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'NUCLEUS' in numeric_cols:
                numeric_cols.remove('NUCLEUS')
            
            # Outlier detection
            outlier_mask = self.outlier_handler.detect_outliers_iqr(
                df, numeric_cols, threshold=3.0
            )
            
            n_outliers = outlier_mask.sum()
            logger.info(f"  Detected {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")
            
            # Data validation
            validation_rules = {
                'ranges': {
                    'A': (1, 300),
                    'Z': (1, 120),
                    'N': (1, 200)
                }
            }

            # Get actual target column names for validation
            target_cols = self._get_actual_column_names(target, df)
            for target_col in target_cols:
                if target_col in df.columns:
                    # Add target-specific validation based on target type
                    if target == 'MM' or 'MAGNETIC' in target_col.upper():
                        validation_rules['ranges'][target_col] = (-10, 10)
                    elif target == 'QM' or 'QUADRUPOLE' in target_col.upper():
                        validation_rules['ranges'][target_col] = (-5, 5)
                    elif target == 'Beta_2':
                        validation_rules['ranges'][target_col] = (-0.5, 0.5)
            
            issues = self.data_validator.validate_dataset(df, validation_rules)
            
            logger.info(f"  Validation issues: {len(issues)}")
            
            quality_reports[target] = {
                'n_samples': len(df),
                'n_outliers': int(n_outliers),
                'outlier_percentage': float(n_outliers / len(df) * 100),
                'validation_issues': len(issues)
            }
        
        self.generation_report['quality_control'] = quality_reports
    
    def _generate_all_datasets(self):
        """
        Tüm dataset kombinasyonlarını oluştur

        [FAZ 1 UPDATE]: target × nucleus_count × feature_set kombinasyonları
        """
        logger.info("Generating all dataset combinations...")

        # [FAZ 1 UPDATE]: Üçlü kombinasyon
        total_combinations = len(self.targets) * len(self.nucleus_counts) * len(self.feature_sets)
        logger.info(f"Total combinations: {len(self.targets)} targets × {len(self.nucleus_counts)} sizes × {len(self.feature_sets)} feature sets = {total_combinations}")

        generated_count = 0

        # [FAZ 1 UPDATE]: Üçlü döngü
        for target in self.targets:
            if target not in self.filtered_data:
                logger.warning(f"[WARNING] No filtered data for {target}, skipping")
                continue

            target_df = self.filtered_data[target]

            logger.info(f"\n-> Target: {target} (Available nuclei: {len(target_df)})")

            for n_nuclei in self.nucleus_counts:
                # Handle 'ALL' case
                if n_nuclei == 'ALL':
                    actual_n = len(target_df)
                    size_label = f"ALL_{actual_n}"
                elif n_nuclei > len(target_df):
                    logger.warning(f"  [WARNING] Requested {n_nuclei} nuclei but only {len(target_df)} available, skipping")
                    continue
                else:
                    size_label = str(n_nuclei)

                # [FAZ 1 NEW]: Feature set döngüsü
                for feature_set_name in self.feature_sets:
                    try:
                        # Sample dataset with specific feature set
                        dataset = self._create_single_dataset_with_features(
                            target_df, target, n_nuclei, feature_set_name
                        )

                        if dataset is not None:
                            self.generated_datasets.append(dataset)
                            generated_count += 1

                            logger.info(f"  [OK] {dataset['dataset_name']} | {dataset['n_features']} features | {len(dataset['data'])} nuclei")

                    except Exception as e:
                        logger.error(f"  ✗ Error generating {target}_{size_label}_{feature_set_name}: {e}")
                        continue

        logger.info(f"\n[SUCCESS] Total datasets generated: {generated_count}/{total_combinations}")

        self.generation_report['dataset_generation'] = {
            'total_requested': total_combinations,
            'total_generated': generated_count,
            'success_rate': generated_count / total_combinations if total_combinations > 0 else 0,
            'by_target': {},
            'by_feature_set': {}
        }

        # Collect stats
        for dataset in self.generated_datasets:
            target = dataset['target']
            feature_set = dataset['feature_set']

            if target not in self.generation_report['dataset_generation']['by_target']:
                self.generation_report['dataset_generation']['by_target'][target] = 0
            self.generation_report['dataset_generation']['by_target'][target] += 1

            if feature_set not in self.generation_report['dataset_generation']['by_feature_set']:
                self.generation_report['dataset_generation']['by_feature_set'][feature_set] = 0
            self.generation_report['dataset_generation']['by_feature_set'][feature_set] += 1
    
    def _get_actual_column_names(self, target: str, df: pd.DataFrame) -> List[str]:
        """
        Get actual column names for a target from the dataframe

        Args:
            target: Target name (simplified, e.g., 'MM', 'QM')
            df: DataFrame to check for actual column names

        Returns:
            List of actual column names
        """
        if target == 'MM_QM':
            # Special case: both MM and Q targets
            actual_cols = []
            for t in ['MM', 'Q']:
                col_name = self.target_column_map.get(t, t)
                if col_name in df.columns:
                    actual_cols.append(col_name)
                elif t in df.columns:
                    actual_cols.append(t)
            return actual_cols
        else:
            # Single target
            col_name = self.target_column_map.get(target, target)
            if col_name in df.columns:
                return [col_name]
            elif target in df.columns:
                return [target]
            else:
                # Try to find column by checking if it contains the target name
                for col in df.columns:
                    if target.upper() in col.upper():
                        return [col]
                return [target]  # Fallback to original name

    def _create_single_dataset_with_features(self,
                                             source_df: pd.DataFrame,
                                             target: str,
                                             n_nuclei: int,
                                             feature_set_name: str) -> Optional[Dict]:
        """
        [FAZ 3 UPDATE] Belirli bir feature set ile dataset oluştur

        Args:
            source_df: Kaynak DataFrame
            target: Target adı (MM, QM, etc.)
            n_nuclei: Çekirdek sayısı (veya 'ALL')
            feature_set_name: Feature set adı (Basic, Extended, Full)

        Returns:
            Dataset metadata dictionary
        """
        # [FAZ 3 NEW]: Sampling with SamplingManager
        if n_nuclei == 'ALL':
            sampled_df = source_df.copy()
            actual_n = len(sampled_df)
            size_label = f"ALL_{actual_n}"
        else:
            # Use SamplingManager for sampling
            seed = hash(f"{target}_{n_nuclei}_{feature_set_name}") % (2**32)
            sampling_manager = SamplingManager(method=self.sampling, random_seed=seed)
            sampled_df = sampling_manager.sample(source_df, n_nuclei, group_col='A')
            actual_n = n_nuclei
            size_label = str(n_nuclei)

        # [FAZ 1]: Feature set selection
        target_cols = self._get_actual_column_names(target, sampled_df)

        # Get features for this specific feature set
        try:
            feature_cols = self.feature_manager.get_feature_set(
                feature_set_name,
                sampled_df.columns.tolist(),
                target_cols
            )
        except ValueError as e:
            logger.error(f"Error getting feature set '{feature_set_name}': {e}")
            return None

        # [FAZ 2 NEW]: Detect I/O configuration
        n_features = len(feature_cols)
        io_config_name = self.io_config_manager.get_config_for_feature_set(
            feature_set_name,
            n_features,
            target
        )

        # [FAZ 2 NEW]: Enhanced naming convention (7-part format)
        # Format: {TARGET}_{SIZE}_{SCENARIO}_{IO_CONFIG}_{FEATURE_SET}_{SCALING}_{SAMPLING}
        # Example: MM_75_S70_3In1Out_Basic_NoScaling_Random
        dataset_name = f"{target}_{size_label}_{self.scenario}_{io_config_name}_{feature_set_name}_{self.scaling}_{self.sampling}"

        # FIXED: Each dataset gets its own directory for PFAZ2/3 discovery
        # Old (WRONG): output_base_dir / target / feature_set_name -> outputs/generated_datasets/MM/Basic/
        # New (CORRECT): output_base_dir / dataset_name -> outputs/generated_datasets/MM_75_S70_3In1Out_Basic_NoScaling_Random/
        dataset_dir = self.output_base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # [FAZ 2 UPDATE]: Split using scenario ratios instead of hardcoded values
        train_ratio, val_ratio, test_ratio = self.scenario_manager.get_split_ratios(self.scenario)
        n_total = len(sampled_df)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        # Shuffle with fixed seed for reproducibility
        split_seed = hash(f"split_{target}_{n_nuclei}_{feature_set_name}") % (2**32)
        shuffled_df = sampled_df.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)

        train_df = shuffled_df[:n_train]
        val_df = shuffled_df[n_train:n_train+n_val]
        test_df = shuffled_df[n_train+n_val:]

        # [FAZ 3 NEW]: Apply scaling
        scaler = ScalingManager(method=self.scaling)
        scaling_metadata = {}

        if self.scaling != 'NoScaling':
            # Fit scaler on train features only (not targets!)
            scaler.fit(train_df, feature_cols)

            # Transform train/val/test
            train_df = scaler.transform(train_df)
            val_df = scaler.transform(val_df)
            test_df = scaler.transform(test_df)

            # Get scaling metadata
            scaling_metadata = scaler.get_metadata()

            logger.info(f"  Scaling applied: {self.scaling} (scaled {len(scaler.features_to_scale)} features)")
        else:
            logger.info(f"  No scaling applied")

        # Save train/val/test splits
        split_files = {}

        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Select only the relevant columns (features + targets)
            cols_to_save = ['NUCLEUS'] + feature_cols + target_cols if 'NUCLEUS' in split_df.columns else feature_cols + target_cols
            split_data = split_df[cols_to_save]

            # CSV format - FIXED: Simplified naming (dataset name is already in directory name)
            # Old: MM_75_S70_3In1Out_Basic_NoScaling_Random/MM_75_S70_3In1Out_Basic_NoScaling_Random_train.csv
            # New: MM_75_S70_3In1Out_Basic_NoScaling_Random/train.csv
            csv_file = dataset_dir / f"{split_name}.csv"
            split_data.to_csv(csv_file, index=False, encoding='utf-8')

            # MAT format (for ANFIS/MATLAB) - with enhanced metadata
            mat_file = dataset_dir / f"{split_name}.mat"
            self._save_as_mat(
                split_data, mat_file, feature_cols, target_cols,
                scaling_metadata=scaling_metadata,
                dataset_name=dataset_name,
                target=target,
                split_name=split_name
            )

            split_files[split_name] = {
                'csv': csv_file,
                'mat': mat_file
            }

        # [FAZ 3 UPDATE]: Enhanced metadata with scaling and sampling
        metadata = {
            'dataset_name': dataset_name,
            'target': target,
            'feature_set': feature_set_name,
            'io_config': io_config_name,  # [FAZ 2]
            'scenario': self.scenario,  # [FAZ 2]
            'scaling': self.scaling,  # [FAZ 2/3]
            'sampling': self.sampling,  # [FAZ 2/3]
            'n_nuclei_requested': n_nuclei,
            'n_nuclei_total': actual_n,
            'n_nuclei_train': len(train_df),
            'n_nuclei_val': len(val_df),
            'n_nuclei_test': len(test_df),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'target_names': target_cols,
            'split_ratio': [train_ratio, val_ratio, test_ratio],  # [FAZ 2]
            'generation_timestamp': datetime.now().isoformat(),
            'folder_structure': str(dataset_dir.relative_to(self.output_base_dir)),
            # [FAZ 2]: I/O config details
            'io_config_details': self.io_config_manager.get_config_info(io_config_name),
            # [FAZ 3 NEW]: Scaling metadata
            'scaling_metadata': scaling_metadata if self.scaling != 'NoScaling' else {},
            # [FAZ 3 NEW]: Sampling statistics
            'sampling_info': {
                'method': self.sampling,
                'statistics': get_sampling_statistics(sampled_df)
            }
        }

        # Save metadata (sanitized for JSON)
        # FIXED: Simplified naming (dataset name is already in directory)
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(sanitize_for_json(metadata), f, indent=2)

        return {
            'dataset_name': dataset_name,
            'target': target,
            'feature_set': feature_set_name,
            'n_features': len(feature_cols),
            'dataset_dir': dataset_dir,
            'split_files': split_files,
            'data_file_csv': split_files['train']['csv'],  # Keep for backward compatibility
            'data_file_mat': split_files['train']['mat'],  # Keep for backward compatibility
            'metadata_file': metadata_file,
            'metadata': metadata,
            'data': shuffled_df  # Full shuffled data
        }

    def _create_single_dataset(self, source_df: pd.DataFrame, target: str, n_nuclei: int) -> Optional[Dict]:
        """Tek bir dataset oluştur ve train/val/test olarak böl"""

        # Handle 'ALL' case
        if n_nuclei == 'ALL':
            sampled_df = source_df.copy()
            actual_n = len(sampled_df)
            dataset_name = f"{target}_ALL_{actual_n}nuclei"
        else:
            # Random sampling with seed for reproducibility
            seed = hash(f"{target}_{n_nuclei}") % (2**32)
            sampled_df = source_df.sample(n=n_nuclei, random_state=seed)
            actual_n = n_nuclei
            dataset_name = f"{target}_{n_nuclei}nuclei"

        # Determine features and target columns using actual column names
        target_cols = self._get_actual_column_names(target, sampled_df)

        # All columns except target columns are features
        feature_cols = [col for col in sampled_df.columns if col not in target_cols and col != 'NUCLEUS']

        # Create dataset directory
        dataset_dir = self.output_base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # CRITICAL: Split into train/val/test (70/15/15) with fixed seed
        # This ensures ALL models use the SAME train/val/test split for fair comparison
        n_total = len(sampled_df)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        # n_test = remaining

        # Shuffle with fixed seed for reproducibility
        split_seed = hash(f"split_{target}_{n_nuclei}") % (2**32)
        shuffled_df = sampled_df.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)

        train_df = shuffled_df[:n_train]
        val_df = shuffled_df[n_train:n_train+n_val]
        test_df = shuffled_df[n_train+n_val:]

        logger.info(f"  Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Save train/val/test splits in MULTIPLE FORMATS: CSV, Excel (.xlsx), MAT
        # Excel requested for easier viewing/editing in Excel
        split_files = {}

        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # CSV format
            csv_file = dataset_dir / f"{split_name}.csv"
            split_df.to_csv(csv_file, index=False, encoding='utf-8')

            # Excel format (.xlsx)
            xlsx_file = dataset_dir / f"{split_name}.xlsx"
            split_df.to_excel(xlsx_file, index=False, engine='openpyxl')

            # MATLAB format (.mat)
            mat_file = dataset_dir / f"{split_name}.mat"
            self._save_as_mat(split_df, mat_file, feature_cols, target_cols)

            split_files[split_name] = {
                'csv': csv_file,
                'xlsx': xlsx_file,
                'mat': mat_file,
                'n_samples': len(split_df)
            }

        logger.info(f"  [SUCCESS] Saved train/val/test in CSV, Excel, and MAT formats")

        # Create metadata with split information
        metadata = {
            'dataset_name': dataset_name,
            'target': target,
            'target_columns': target_cols,
            'n_nuclei_total': actual_n,
            'n_features': len(feature_cols),
            'feature_columns': feature_cols,
            'creation_timestamp': datetime.now().isoformat(),
            'split_info': {
                'split_ratio': '70/15/15',
                'train': {
                    'n_samples': split_files['train']['n_samples'],
                    'csv': str(split_files['train']['csv']),
                    'xlsx': str(split_files['train']['xlsx']),
                    'mat': str(split_files['train']['mat'])
                },
                'val': {
                    'n_samples': split_files['val']['n_samples'],
                    'csv': str(split_files['val']['csv']),
                    'xlsx': str(split_files['val']['xlsx']),
                    'mat': str(split_files['val']['mat'])
                },
                'test': {
                    'n_samples': split_files['test']['n_samples'],
                    'csv': str(split_files['test']['csv']),
                    'xlsx': str(split_files['test']['xlsx']),
                    'mat': str(split_files['test']['mat'])
                }
            },
            'statistics': {
                'A_range': [int(shuffled_df['A'].min()), int(shuffled_df['A'].max())],
                'Z_range': [int(shuffled_df['Z'].min()), int(shuffled_df['Z'].max())],
                'N_range': [int(shuffled_df['N'].min()), int(shuffled_df['N'].max())]
            }
        }
        
        # Add target statistics
        for target_col in target_cols:
            if target_col in sampled_df.columns:
                # Ensure column is numeric before calculating statistics
                if not pd.api.types.is_numeric_dtype(sampled_df[target_col]):
                    logger.warning(f"  [WARNING] Target column '{target_col}' is not numeric, attempting conversion...")
                    sampled_df[target_col] = pd.to_numeric(sampled_df[target_col], errors='coerce')

                # Calculate statistics only on non-null values
                col_data = sampled_df[target_col].dropna()
                if len(col_data) > 0:
                    metadata['statistics'][f'{target_col}_mean'] = float(col_data.mean())
                    metadata['statistics'][f'{target_col}_std'] = float(col_data.std())
                    metadata['statistics'][f'{target_col}_range'] = [
                        float(col_data.min()),
                        float(col_data.max())
                    ]
                else:
                    logger.warning(f"  [WARNING] Target column '{target_col}' has no valid numeric values")
                    metadata['statistics'][f'{target_col}_mean'] = None
                    metadata['statistics'][f'{target_col}_std'] = None
                    metadata['statistics'][f'{target_col}_range'] = [None, None]
        
        # Save metadata (sanitized for JSON)
        # FIXED: Simplified naming (dataset name is already in directory)
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(sanitize_for_json(metadata), f, indent=2)

        return {
            'dataset_name': dataset_name,
            'dataset_dir': dataset_dir,
            'split_files': split_files,
            'data_file_csv': split_files['train']['csv'],  # Keep for backward compatibility
            'data_file_mat': split_files['train']['mat'],  # Keep for backward compatibility
            'metadata_file': metadata_file,
            'metadata': metadata,
            'data': shuffled_df  # Full shuffled data
        }
    
    def _save_as_mat(self,
                     df: pd.DataFrame,
                     filepath: Path,
                     feature_cols: List[str],
                     target_cols: List[str],
                     scaling_metadata: Dict = None,
                     dataset_name: str = '',
                     target: str = '',
                     split_name: str = ''):
        """
        [FAZ 3 UPDATE] Save dataset as MATLAB .mat file with enhanced metadata

        Args:
            df: DataFrame to save
            filepath: Output .mat file path
            feature_cols: Feature column names
            target_cols: Target column names
            scaling_metadata: Scaling metadata (mean, std, etc.)
            dataset_name: Dataset name
            target: Target variable name
            split_name: Split name (train/val/test)
        """
        try:
            from scipy.io import savemat

            # Prepare enhanced data dictionary for MATLAB/ANFIS
            mat_dict = {
                # Data arrays
                'features': df[feature_cols].values,
                'targets': df[target_cols].values,
                'nucleus_names': df['NUCLEUS'].values if 'NUCLEUS' in df.columns else [],

                # Feature information
                'feature_names': feature_cols,
                'target_names': target_cols,
                'n_features': len(feature_cols),
                'n_targets': len(target_cols),
                'n_samples': len(df),

                # Dataset metadata
                'dataset_name': dataset_name,
                'target': target,
                'split': split_name,

                # [FAZ 3 NEW]: Scaling metadata
                'scaling_applied': scaling_metadata is not None and len(scaling_metadata) > 0,
            }

            # Add scaling parameters if available
            if scaling_metadata and len(scaling_metadata) > 0:
                mat_dict['scaling_method'] = scaling_metadata.get('method', 'NoScaling')
                mat_dict['features_scaled'] = scaling_metadata.get('features_scaled', [])
                mat_dict['features_excluded'] = scaling_metadata.get('features_excluded', [])

                # Add scaler parameters (mean, std, median, IQR)
                scaler_params = scaling_metadata.get('scaler_params', {})
                if 'mean' in scaler_params:
                    mat_dict['scaler_mean'] = np.array(scaler_params['mean'])
                    mat_dict['scaler_std'] = np.array(scaler_params['std'])
                if 'median' in scaler_params:
                    mat_dict['scaler_median'] = np.array(scaler_params['median'])
                    mat_dict['scaler_iqr'] = np.array(scaler_params['iqr'])
            else:
                mat_dict['scaling_method'] = 'NoScaling'
                mat_dict['features_scaled'] = []

            # Save to .mat file
            savemat(filepath, mat_dict)
            # logger.info(f"  [SUCCESS] MAT file saved: {filepath.name}")

        except ImportError:
            logger.warning("  [WARNING] scipy not available, skipping MAT file export")
        except Exception as e:
            logger.error(f"  [ERROR] Error saving MAT file: {e}")
    
    def _create_metadata_and_reports(self):
        """Master metadata ve raporlar oluştur"""
        logger.info("Creating master metadata and reports...")

        # Master metadata
        master_metadata = {
            'pipeline_version': '1.0.0',
            'creation_timestamp': datetime.now().isoformat(),
            'source_data': str(self.source_data_path),
            'total_datasets': len(self.generated_datasets),
            'targets': self.targets,
            'nucleus_counts': self.nucleus_counts,
            'datasets': []
        }

        for dataset in self.generated_datasets:
            master_metadata['datasets'].append({
                'name': dataset['dataset_name'],
                'target': dataset['metadata']['target'],
                'n_nuclei_total': dataset['metadata']['n_nuclei_total'],
                'n_features': dataset['metadata']['n_features'],
                'data_file_csv': str(dataset['data_file_csv']),
                'data_file_mat': str(dataset['data_file_mat'])
            })

        # Save master metadata (sanitized for JSON)
        master_metadata_file = self.output_base_dir / 'master_metadata.json'
        with open(master_metadata_file, 'w') as f:
            json.dump(sanitize_for_json(master_metadata), f, indent=2)

        logger.info(f"[SUCCESS] Master metadata: {master_metadata_file}")

        # Generation report (sanitized for JSON)
        report_file = self.output_base_dir / 'generation_report.json'
        with open(report_file, 'w') as f:
            json.dump(sanitize_for_json(self.generation_report), f, indent=2)

        logger.info(f"[SUCCESS] Generation report: {report_file}")

        # Exclusion tracker reports
        logger.info("\nSaving exclusion tracker reports...")
        self.exclusion_tracker.print_summary()

        # Save to multiple formats
        metadata_dir = self.output_base_dir / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Excel report (main format)
        excel_path = metadata_dir / 'excluded_nuclei_report.xlsx'
        self.exclusion_tracker.save_to_excel(str(excel_path))

        # CSV report (for easy parsing)
        csv_path = metadata_dir / 'excluded_nuclei_report.csv'
        self.exclusion_tracker.save_to_csv(str(csv_path))

        # JSON report (for programmatic access)
        json_path = metadata_dir / 'excluded_nuclei_report.json'
        self.exclusion_tracker.save_to_json(str(json_path))

        logger.info(f"[SUCCESS] Exclusion reports saved to: {metadata_dir}")

        # QM filter reports
        if self.qm_filter_manager.filter_reports:
            qm_report_path = metadata_dir / 'qm_filter_report.xlsx'
            self.qm_filter_manager.save_filter_report_excel(str(qm_report_path))

        # [FAZ 1]: Feature combinations JSON
        feature_combos_path = metadata_dir / 'feature_combinations.json'
        self.feature_manager.save_feature_combinations_json(str(feature_combos_path))

        # [FAZ 2 NEW]: I/O configurations JSON
        io_configs_path = metadata_dir / 'io_configurations.json'
        self.io_config_manager.save_io_configs_json(str(io_configs_path))

        # Summary Excel report
        self._create_summary_excel()
    
    def _create_summary_excel(self):
        """[FAZ 2 UPDATE] Excel özet raporu oluştur"""
        summary_data = []

        for dataset in self.generated_datasets:
            meta = dataset['metadata']

            # Handle both FAZ 1 format (old) and FAZ 2 format (new)
            if 'split_info' in meta:
                # Old FAZ 1 format
                n_train = meta['split_info']['train']['n_samples']
                n_val = meta['split_info']['val']['n_samples']
                n_test = meta['split_info']['test']['n_samples']
                a_min = meta['statistics']['A_range'][0]
                a_max = meta['statistics']['A_range'][1]
                z_min = meta['statistics']['Z_range'][0]
                z_max = meta['statistics']['Z_range'][1]
            else:
                # New FAZ 2 format
                n_train = meta.get('n_nuclei_train', 0)
                n_val = meta.get('n_nuclei_val', 0)
                n_test = meta.get('n_nuclei_test', 0)
                a_min = meta.get('statistics', {}).get('A_range', [None, None])[0] if 'statistics' in meta else None
                a_max = meta.get('statistics', {}).get('A_range', [None, None])[1] if 'statistics' in meta else None
                z_min = meta.get('statistics', {}).get('Z_range', [None, None])[0] if 'statistics' in meta else None
                z_max = meta.get('statistics', {}).get('Z_range', [None, None])[1] if 'statistics' in meta else None

            summary_data.append({
                'Dataset_Name': meta['dataset_name'],
                'Target': meta['target'],
                'Feature_Set': meta.get('feature_set', 'N/A'),  # [FAZ 2]
                'IO_Config': meta.get('io_config', 'N/A'),  # [FAZ 2]
                'Scenario': meta.get('scenario', 'N/A'),  # [FAZ 2]
                'Scaling': meta.get('scaling', 'N/A'),  # [FAZ 2]
                'Sampling': meta.get('sampling', 'N/A'),  # [FAZ 2]
                'N_Nuclei_Total': meta['n_nuclei_total'],
                'N_Train': n_train,
                'N_Val': n_val,
                'N_Test': n_test,
                'N_Features': meta['n_features'],
                'A_Min': a_min,
                'A_Max': a_max,
                'Z_Min': z_min,
                'Z_Max': z_max
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to Excel
        excel_file = self.output_base_dir / 'datasets_summary.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Datasets_Summary', index=False)
            
            # Quality control summary
            if 'quality_control' in self.generation_report:
                qc_data = []
                for target, qc_info in self.generation_report['quality_control'].items():
                    qc_data.append({
                        'Target': target,
                        'N_Samples': qc_info['n_samples'],
                        'N_Outliers': qc_info['n_outliers'],
                        'Outlier_Percentage': qc_info['outlier_percentage'],
                        'Validation_Issues': qc_info['validation_issues']
                    })
                
                qc_df = pd.DataFrame(qc_data)
                qc_df.to_excel(writer, sheet_name='Quality_Control', index=False)

        logger.info(f"[SUCCESS] Summary Excel: {excel_file}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("FAZ 1: DATASET GENERATION PIPELINE")
    print("=" * 80)
    
    # Configuration
    SOURCE_DATA = "/mnt/user-data/uploads/your_source_data.csv"  # Kullanıcı source data yolunu girecek
    OUTPUT_DIR = "generated_datasets"
    NUCLEUS_COUNTS = [75, 100, 150, 200, 'ALL']  # [UPDATED] per requirements
    TARGETS = ['MM', 'QM', 'MM_QM', 'Beta_2']
    
    # Check if source data exists (demo mode if not)
    if not Path(SOURCE_DATA).exists():
        logger.warning(f"[WARNING] Source data not found: {SOURCE_DATA}")
        logger.info("Creating DEMO dataset for testing...")

        # Create demo data
        demo_data = _create_demo_data()
        demo_path = Path("demo_source_data.csv")
        demo_data.to_csv(demo_path, index=False)
        SOURCE_DATA = str(demo_path)
        logger.info(f"[SUCCESS] Demo data created: {demo_path}")
    
    # Initialize pipeline
    pipeline = DatasetGenerationPipelineV2(
        source_data_path=SOURCE_DATA,
        output_base_dir=OUTPUT_DIR,
        nucleus_counts=NUCLEUS_COUNTS,
        targets=TARGETS
    )
    
    # Run pipeline
    report = pipeline.run_complete_pipeline()
    
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Total datasets generated: {len(pipeline.generated_datasets)}")
    print(f"Output directory: {pipeline.output_base_dir}")
    print(f"Duration: {report.get('total_duration_seconds', 0):.2f} seconds")
    
    return pipeline, report


def _create_demo_data() -> pd.DataFrame:
    """Demo veri oluştur (test için)"""
    np.random.seed(42)
    n_samples = 500
    
    demo_data = pd.DataFrame({
        'NUCLEUS': [f"Nucleus_{i}" for i in range(n_samples)],
        'A': np.random.randint(20, 250, n_samples),
        'Z': np.random.randint(10, 100, n_samples),
        'N': np.random.randint(10, 150, n_samples),
        'SPIN': np.random.choice([0, 0.5, 1, 1.5, 2], n_samples),
        'PARITY': np.random.choice([1, -1], n_samples),
        'MM': np.random.randn(n_samples) * 2,
        'Q': np.random.randn(n_samples) * 0.5,
        'Beta_2': np.random.uniform(-0.3, 0.4, n_samples)
    })
    
    # Add some missing Q values (for QM filtering test)
    demo_data.loc[np.random.choice(n_samples, 50, replace=False), 'Q'] = np.nan
    
    return demo_data


if __name__ == "__main__":
    pipeline, report = main()
    print("\n[SUCCESS] FAZ 1: DATASET GENERATION PIPELINE - COMPLETE!")
