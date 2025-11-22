"""
FAZ 1: Dataset Generation Pipeline
===================================

Kapsamlı dataset oluşturma sistemi:
- Teorik hesaplamalar (SEMF, Shell Model, Woods-Saxon, Nilsson, Schmidt)
- QM filtreleme (target-based)
- Çoklu çekirdek sayıları (50, 75, 100, 150, 175, 200, 250, 300, 350)
- Çoklu targetler (MM, QM, MM_QM, Beta_2)
- Kalite kontrolü ve validasyon
- Otomatik raporlama

Author: Nuclear Physics AI Project
Version: 1.0.0
Date: 2025-10-15
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetGenerationPipelineV2:
    """
    Ana Dataset Generation Pipeline V2

    Workflow:
    1. Ham veriyi yükle
    2. Teorik hesaplamalar ekle
    3. Target-based QM filtreleme uygula
    4. Farklı çekirdek sayıları için örnekle
    5. Kalite kontrolü
    6. Dataset'leri kaydet
    7. Metadata ve raporlar oluştur
    """
    
    def __init__(self,
                 source_data_path: str = None,
                 output_base_dir: str = 'generated_datasets',
                 nucleus_counts: List[int] = None,
                 targets: List[str] = None,
                 # Backward compatibility aliases
                 aaa2_txt_path: str = None,
                 output_dir: str = None):
        """
        Args:
            source_data_path: Ham veri dosyası yolu (or use aaa2_txt_path)
            output_base_dir: Çıktı ana dizini (or use output_dir)
            nucleus_counts: Oluşturulacak dataset boyutları
            targets: Target değişkenler
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

        # Target column name mapping (simplified name -> actual column name)
        self.target_column_map = {
            'MM': 'MAGNETIC MOMENT [µ]',
            'QM': 'QUADRUPOLE MOMENT [Q]',
            'Q': 'QUADRUPOLE MOMENT [Q]',
            'Beta_2': 'Beta_2'
        }
        
        # Initialize managers
        self.theoretical_calc_manager = TheoreticalCalculationsManager(enable_all=True)
        self.qm_filter_manager = QMFilterManager()
        self.outlier_handler = OutlierHandler(output_dir=self.output_base_dir / 'quality_reports')
        self.data_validator = DataValidator(output_dir=self.output_base_dir / 'quality_reports')
        
        # Storage
        self.raw_data = None
        self.enriched_data = None
        self.filtered_data = {}  # {target: df}
        self.generated_datasets = []
        self.generation_report = {}
        
        logger.info("=" * 80)
        logger.info("DATASET GENERATION PIPELINE INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Source data: {self.source_data_path}")
        logger.info(f"Output directory: {self.output_base_dir}")
        logger.info(f"Nucleus counts: {self.nucleus_counts}")
        logger.info(f"Targets: {self.targets}")
    
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
        """Tüm dataset kombinasyonlarını oluştur"""
        logger.info("Generating all dataset combinations...")
        
        total_combinations = len(self.targets) * len(self.nucleus_counts)
        logger.info(f"Total combinations to generate: {total_combinations}")
        
        generated_count = 0
        
        for target in self.targets:
            if target not in self.filtered_data:
                logger.warning(f"[WARNING] No filtered data for {target}, skipping")
                continue

            target_df = self.filtered_data[target]

            logger.info(f"\n-> Generating datasets for target: {target}")
            logger.info(f"   Available nuclei: {len(target_df)}")

            for n_nuclei in self.nucleus_counts:
                # Handle 'ALL' case
                if n_nuclei == 'ALL':
                    actual_n = len(target_df)
                    logger.info(f"  -> 'ALL' option: using all {actual_n} available nuclei")
                elif n_nuclei > len(target_df):
                    logger.warning(f"  [WARNING] Requested {n_nuclei} nuclei but only {len(target_df)} available, skipping")
                    continue

                # Sample dataset
                dataset = self._create_single_dataset(target_df, target, n_nuclei)

                if dataset is not None:
                    self.generated_datasets.append(dataset)
                    generated_count += 1

                    logger.info(f"  [SUCCESS] Generated: {dataset['dataset_name']} ({len(dataset['data'])} nuclei)")
                    logger.info(f"     Files: CSV + MAT")

        logger.info(f"\n[SUCCESS] Total datasets generated: {generated_count}/{total_combinations}")
        
        self.generation_report['dataset_generation'] = {
            'total_requested': total_combinations,
            'total_generated': generated_count,
            'success_rate': generated_count / total_combinations if total_combinations > 0 else 0
        }
    
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
        
        # Save metadata
        metadata_file = dataset_dir / f"{dataset_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

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
    
    def _save_as_mat(self, df: pd.DataFrame, filepath: Path, feature_cols: List[str], target_cols: List[str]):
        """
        Save dataset as MATLAB .mat file
        
        Args:
            df: DataFrame to save
            filepath: Output .mat file path
            feature_cols: Feature column names
            target_cols: Target column names
        """
        try:
            from scipy.io import savemat
            
            # Prepare data dictionary for MATLAB
            mat_dict = {
                'features': df[feature_cols].values,
                'targets': df[target_cols].values,
                'feature_names': feature_cols,
                'target_names': target_cols,
                'nucleus_names': df['NUCLEUS'].values if 'NUCLEUS' in df.columns else []
            }
            
            # Save
            savemat(filepath, mat_dict)
            logger.info(f"  [SUCCESS] MAT file saved: {filepath.name}")

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
        
        # Save master metadata
        master_metadata_file = self.output_base_dir / 'master_metadata.json'
        with open(master_metadata_file, 'w') as f:
            json.dump(master_metadata, f, indent=2)

        logger.info(f"[SUCCESS] Master metadata: {master_metadata_file}")

        # Generation report
        report_file = self.output_base_dir / 'generation_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.generation_report, f, indent=2)

        logger.info(f"[SUCCESS] Generation report: {report_file}")
        
        # Summary Excel report
        self._create_summary_excel()
    
    def _create_summary_excel(self):
        """Excel özet raporu oluştur"""
        summary_data = []
        
        for dataset in self.generated_datasets:
            meta = dataset['metadata']
            summary_data.append({
                'Dataset_Name': meta['dataset_name'],
                'Target': meta['target'],
                'N_Nuclei_Total': meta['n_nuclei_total'],
                'N_Train': meta['split_info']['train']['n_samples'],
                'N_Val': meta['split_info']['val']['n_samples'],
                'N_Test': meta['split_info']['test']['n_samples'],
                'N_Features': meta['n_features'],
                'A_Min': meta['statistics']['A_range'][0],
                'A_Max': meta['statistics']['A_range'][1],
                'Z_Min': meta['statistics']['Z_range'][0],
                'Z_Max': meta['statistics']['Z_range'][1],
                'Train_CSV': str(dataset['split_files']['train']['csv'].name),
                'Train_Excel': str(dataset['split_files']['train']['xlsx'].name)
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
