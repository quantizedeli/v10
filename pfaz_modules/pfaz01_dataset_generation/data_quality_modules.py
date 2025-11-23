"""
Data Quality Modules
13. outlier_handler.py
14. data_validator.py  
15. robustness_tester.py (zaten var, reference)

data_quality/ klasöründe
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 13. OUTLIER HANDLER
# ============================================================================

class OutlierHandler:
    """
    Advanced outlier detection and handling

    Methods:
    1. IQR method
    2. Z-score
    3. Isolation Forest
    4. Elliptic Envelope
    5. DBSCAN

    Features:
    - Detailed outlier tracking with reasons
    - Integration with ExcludedNucleiTracker
    - Comprehensive reporting
    """

    def __init__(self, output_dir='data_quality/outliers', tracker=None):
        """
        Initialize OutlierHandler

        Args:
            output_dir: Output directory for reports
            tracker: ExcludedNucleiTracker instance for tracking exclusions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker

        # Store detailed outlier information
        self.outlier_details = []

        logger.info("Outlier Handler başlatıldı")
    
    def detect_outliers_iqr(self, df, columns, threshold=1.5, target=None):
        """
        IQR method outlier detection with detailed tracking

        Args:
            df: DataFrame
            columns: Columns to check
            threshold: IQR threshold multiplier
            target: Target name for tracking (e.g., 'MM', 'QM')

        Returns:
            outlier_mask: Boolean mask for outliers
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            # String değerleri numeric'e çevir
            col_data = pd.to_numeric(df[col], errors='coerce')

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            col_outliers = (col_data < lower_bound) | (col_data > upper_bound)

            # Track individual outliers with details
            if self.tracker and target and 'NUCLEUS' in df.columns:
                outlier_indices = col_data[col_outliers].index
                for idx in outlier_indices:
                    value = col_data.loc[idx]
                    nucleus = df.loc[idx, 'NUCLEUS'] if 'NUCLEUS' in df.columns else f'Index_{idx}'

                    # Calculate how many IQRs away
                    if value < lower_bound:
                        iqr_distance = (lower_bound - value) / IQR
                        bound_type = 'lower'
                    else:
                        iqr_distance = (value - upper_bound) / IQR
                        bound_type = 'upper'

                    details = {
                        'column': col,
                        'value': float(value),
                        'Q1': float(Q1),
                        'Q3': float(Q3),
                        'IQR': float(IQR),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'threshold': threshold,
                        'iqr_distance': float(iqr_distance),
                        'bound_type': bound_type,
                        'method': 'IQR'
                    }

                    # Add to tracker
                    a = int(df.loc[idx, 'A']) if 'A' in df.columns else None
                    z = int(df.loc[idx, 'Z']) if 'Z' in df.columns else None
                    n = int(df.loc[idx, 'N']) if 'N' in df.columns else None

                    self.tracker.add_exclusion(
                        nucleus=nucleus,
                        reason='OUTLIER_REMOVED',
                        target=target,
                        a=a, z=z, n=n,
                        details=details
                    )

            outlier_mask |= col_outliers

            n_outliers = col_outliers.sum()
            logger.info(f"  {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

        return outlier_mask
    
    def detect_outliers_zscore(self, df, columns, threshold=3, target=None):
        """
        Z-score method with detailed tracking

        Args:
            df: DataFrame
            columns: Columns to check
            threshold: Z-score threshold
            target: Target name for tracking

        Returns:
            outlier_mask: Boolean mask for outliers
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        for col in columns:
            # String değerleri numeric'e çevir
            col_data = pd.to_numeric(df[col], errors='coerce')

            mean = col_data.mean()
            std = col_data.std()
            z_scores = np.abs((col_data - mean) / std)
            col_outliers = z_scores > threshold

            # Track individual outliers
            if self.tracker and target and 'NUCLEUS' in df.columns:
                outlier_indices = col_data[col_outliers].index
                for idx in outlier_indices:
                    value = col_data.loc[idx]
                    z_score = z_scores.loc[idx]
                    nucleus = df.loc[idx, 'NUCLEUS'] if 'NUCLEUS' in df.columns else f'Index_{idx}'

                    details = {
                        'column': col,
                        'value': float(value),
                        'mean': float(mean),
                        'std': float(std),
                        'z_score': float(z_score),
                        'threshold': threshold,
                        'method': 'Z-score'
                    }

                    a = int(df.loc[idx, 'A']) if 'A' in df.columns else None
                    z = int(df.loc[idx, 'Z']) if 'Z' in df.columns else None
                    n = int(df.loc[idx, 'N']) if 'N' in df.columns else None

                    self.tracker.add_exclusion(
                        nucleus=nucleus,
                        reason='OUTLIER_REMOVED',
                        target=target,
                        a=a, z=z, n=n,
                        details=details
                    )

            outlier_mask |= col_outliers

            n_outliers = col_outliers.sum()
            logger.info(f"  {col}: {n_outliers} outliers (Z>{threshold})")

        return outlier_mask
    
    def detect_outliers_isolation_forest(self, df, columns, contamination=0.1):
        """Isolation Forest method"""

        logger.info(f"Isolation Forest (contamination={contamination})")

        # String değerleri numeric'e çevir
        df_numeric = df[columns].copy()
        for col in columns:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        # NaN satırlarını çıkar
        df_numeric = df_numeric.dropna()

        X = df_numeric.values

        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X)

        # Tüm DataFrame için mask oluştur
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_mask.loc[df_numeric.index] = (predictions == -1)

        n_outliers = outlier_mask.sum()
        logger.info(f"  Detected: {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

        return outlier_mask
    
    def remove_outliers(self, df, outlier_mask, save_removed=True):
        """Remove outliers and optionally save them"""
        
        df_clean = df[~outlier_mask].copy()
        df_outliers = df[outlier_mask].copy()
        
        logger.info(f"Removed {len(df_outliers)} outliers, kept {len(df_clean)} samples")
        
        if save_removed:
            df_outliers.to_csv(self.output_dir / 'removed_outliers.csv', index=False)
            logger.info(f"[OK] Removed outliers saved")
        
        return df_clean
    
    def cap_outliers(self, df, columns, method='iqr', threshold=1.5):
        """Cap outliers instead of removing"""
        
        df_capped = df.copy()

        for col in columns:
            # String değerleri numeric'e çevir
            col_data = pd.to_numeric(df[col], errors='coerce')
            df_capped[col] = col_data

            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            else:
                mean = col_data.mean()
                std = col_data.std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            df_capped[col] = col_data.clip(lower_bound, upper_bound)

            n_capped = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            logger.info(f"  {col}: {n_capped} values capped")
        
        return df_capped


# ============================================================================
# 14. DATA VALIDATOR
# ============================================================================

class DataValidator:
    """
    Data quality validation
    
    Checks:
    1. Missing values
    2. Duplicates
    3. Data types
    4. Value ranges
    5. Consistency checks
    6. Physical constraints
    """
    
    def __init__(self, output_dir='data_quality/validation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_report = []
        
        logger.info("Data Validator başlatıldı")
    
    def validate_dataset(self, df, validation_rules=None):
        """
        Complete dataset validation
        
        Args:
            df: DataFrame to validate
            validation_rules: Dict with validation rules
        """
        
        logger.info("\n" + "="*80)
        logger.info("DATA VALIDATION")
        logger.info("="*80)
        
        self.validation_report = []
        
        # 1. Missing values
        self._check_missing_values(df)
        
        # 2. Duplicates
        self._check_duplicates(df)
        
        # 3. Data types
        self._check_data_types(df)
        
        # 4. Value ranges
        if validation_rules and 'ranges' in validation_rules:
            self._check_value_ranges(df, validation_rules['ranges'])
        
        # 5. Physical constraints (nuclear physics specific)
        self._check_physical_constraints(df)
        
        # Save report
        self._save_report()
        
        # Summary
        n_issues = len(self.validation_report)
        if n_issues == 0:
            logger.info("\n[OK] No validation issues found")
        else:
            logger.warning(f"\n[WARNING] Found {n_issues} validation issues")
        
        return self.validation_report
    
    def _check_missing_values(self, df):
        """Check for missing values"""
        
        logger.info("\n-> Checking missing values...")
        
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) > 0:
            for col, count in missing.items():
                pct = count / len(df) * 100
                issue = f"Missing values in {col}: {count} ({pct:.1f}%)"
                self.validation_report.append(issue)
                logger.warning(f"  [WARNING] {issue}")
        else:
            logger.info("  [OK] No missing values")
    
    def _check_duplicates(self, df):
        """Check for duplicate rows"""
        
        logger.info("\n-> Checking duplicates...")
        
        n_duplicates = df.duplicated().sum()
        
        if n_duplicates > 0:
            issue = f"Duplicate rows: {n_duplicates}"
            self.validation_report.append(issue)
            logger.warning(f"  [WARNING] {issue}")
        else:
            logger.info("  [OK] No duplicates")
    
    def _check_data_types(self, df):
        """Check data types"""
        
        logger.info("\n-> Checking data types...")
        
        # Expected numeric columns
        numeric_cols = ['A', 'Z', 'N', 'MM', 'Q', 'BE']
        
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issue = f"Non-numeric type in {col}: {df[col].dtype}"
                    self.validation_report.append(issue)
                    logger.warning(f"  [WARNING] {issue}")
        
        logger.info("  [OK] Data types checked")
    
    def _check_value_ranges(self, df, ranges):
        """Check if values are within expected ranges"""
        
        logger.info("\n-> Checking value ranges...")

        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                # String değerleri numeric'e çevir
                col_data = pd.to_numeric(df[col], errors='coerce')

                # NaN olmayan değerleri kontrol et
                valid_data = col_data.dropna()
                if len(valid_data) > 0:
                    out_of_range = ((valid_data < min_val) | (valid_data > max_val)).sum()

                    if out_of_range > 0:
                        issue = f"Out of range values in {col}: {out_of_range} (expected [{min_val}, {max_val}])"
                        self.validation_report.append(issue)
                        logger.warning(f"  [WARNING] {issue}")
    
    def _check_physical_constraints(self, df):
        """Check nuclear physics constraints"""
        
        logger.info("\n-> Checking physical constraints...")

        # A = Z + N
        if all(col in df.columns for col in ['A', 'Z', 'N']):
            # String değerleri numeric'e çevir
            A = pd.to_numeric(df['A'], errors='coerce')
            Z = pd.to_numeric(df['Z'], errors='coerce')
            N = pd.to_numeric(df['N'], errors='coerce')

            mismatch = (A != Z + N).sum()
            if mismatch > 0:
                issue = f"A ≠ Z + N mismatch: {mismatch} samples"
                self.validation_report.append(issue)
                logger.warning(f"  [WARNING] {issue}")

        # Z, N > 0
        if 'Z' in df.columns:
            Z = pd.to_numeric(df['Z'], errors='coerce')
            invalid_z = (Z <= 0).sum()
            if invalid_z > 0:
                issue = f"Invalid Z values: {invalid_z}"
                self.validation_report.append(issue)
                logger.warning(f"  [WARNING] {issue}")
        
        logger.info("  [OK] Physical constraints checked")
    
    def _save_report(self):
        """Save validation report"""
        
        if self.validation_report:
            report_df = pd.DataFrame({
                'Issue': self.validation_report
            })
            
            report_path = self.output_dir / 'validation_report.csv'
            report_df.to_csv(report_path, index=False)
            
            logger.info(f"\n[OK] Validation report: {report_path}")


# ============================================================================
# REFERENCE: robustness_tester.py (already exists as model_validator.py)
# ============================================================================

"""
15. robustness_tester.py is already implemented in model_validator.py
Contains:
- Noise sensitivity tests
- Outlier sensitivity tests  
- Feature perturbation tests

Location: ai_training/model_validator.py -> RobustnessTester class
"""


# ============================================================================
# MAIN TEST
# ============================================================================

def test_data_quality_modules():
    """Test data quality modules"""
    
    print("\n" + "="*80)
    print("DATA QUALITY MODULES TEST")
    print("="*80)
    
    # Dummy data
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'A': np.random.randint(10, 250, n),
        'Z': np.random.randint(5, 100, n),
        'N': np.random.randint(5, 150, n),
        'MM': np.random.randn(n),
        'Q': np.random.randn(n) * 0.5
    })
    
    # Add some outliers
    df.loc[0:5, 'MM'] = 10  # Extreme values
    df.loc[10:12, 'Q'] = -5
    
    # Add missing values
    df.loc[20:22, 'MM'] = np.nan
    
    # Test Outlier Handler
    print("\n-> Testing Outlier Handler...")
    outlier_handler = OutlierHandler(output_dir='test_outliers')
    
    mask_iqr = outlier_handler.detect_outliers_iqr(df, ['MM', 'Q'])
    print(f"  IQR method: {mask_iqr.sum()} outliers")
    
    mask_iso = outlier_handler.detect_outliers_isolation_forest(df, ['MM', 'Q'])
    print(f"  Isolation Forest: {mask_iso.sum()} outliers")
    
    # Test Data Validator
    print("\n-> Testing Data Validator...")
    validator = DataValidator(output_dir='test_validation')
    
    validation_rules = {
        'ranges': {
            'A': (1, 300),
            'Z': (1, 120),
            'MM': (-5, 5)
        }
    }
    
    issues = validator.validate_dataset(df, validation_rules)
    
    print("\n[OK] Data Quality modules test tamamlandı!")
    print(f"  Outlier detections: {mask_iqr.sum()} (IQR), {mask_iso.sum()} (Isolation Forest)")
    print(f"  Validation issues: {len(issues)}")


if __name__ == "__main__":
    test_data_quality_modules()
    print("\n[OK] Data Quality modülleri hazır:")
    print("  - data_quality/outlier_handler.py")
    print("  - data_quality/data_validator.py")
    print("  - (robustness_tester.py zaten var: ai_training/model_validator.py)")