"""
String-numeric karşılaştırma hatalarını düzelten script

Bu script, DataFrame'lerde string değerlerin numeric değerlerle
karşılaştırılmasından kaynaklanan hataları düzeltir.

Hata: TypeError: '<' not supported between instances of 'str' and 'float'
"""

import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class NumericComparisonFixer:
    """DataFrame karşılaştırma hatalarını düzelten sınıf"""

    def __init__(self, project_root='/home/user/nucdatav1'):
        self.project_root = Path(project_root)
        self.fixes_applied = []

    def find_comparison_patterns(self, file_path):
        """Dosyada problemli karşılaştırma pattern'lerini bul"""

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        issues = []

        # Pattern 1: df[col] < value veya df[col] > value
        pattern1 = r'df\[[^\]]+\]\s*(<|>|<=|>=|==|!=)\s*[^\s]+'

        # Pattern 2: (df[col] < value) | (df[col] > value)
        pattern2 = r'\(df\[[^\]]+\]\s*(<|>|<=|>=)\s*[^\)]+\)\s*\|'

        # Pattern 3: df[col].mean(), df[col].std() gibi numeric metodlar
        pattern3 = r'df\[[^\]]+\]\.(mean|std|quantile|sum|min|max)\('

        for i, line in enumerate(lines, 1):
            if re.search(pattern1, line) or re.search(pattern2, line) or re.search(pattern3, line):
                # Yorum satırı değilse
                if not line.strip().startswith('#'):
                    issues.append({
                        'line_num': i,
                        'line': line,
                        'file': file_path
                    })

        return issues

    def fix_data_quality_modules(self):
        """data_quality_modules.py dosyasını düzelt"""

        file_path = self.project_root / 'pfaz_modules/pfaz01_dataset_generation/data_quality_modules.py'

        logger.info(f"\n{'='*80}")
        logger.info(f"Düzeltiliyor: {file_path.name}")
        logger.info(f"{'='*80}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix 1: detect_outliers_iqr - line 43-62
        old_iqr = '''    def detect_outliers_iqr(self, df, columns, threshold=1.5):
        """IQR method outlier detection"""

        outlier_mask = pd.Series([False] * len(df))

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask |= col_outliers

            n_outliers = col_outliers.sum()
            logger.info(f"  {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

        return outlier_mask'''

        new_iqr = '''    def detect_outliers_iqr(self, df, columns, threshold=1.5):
        """IQR method outlier detection"""

        outlier_mask = pd.Series([False] * len(df))

        for col in columns:
            # String değerleri numeric'e çevir
            col_data = pd.to_numeric(df[col], errors='coerce')

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            col_outliers = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_mask |= col_outliers

            n_outliers = col_outliers.sum()
            logger.info(f"  {col}: {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

        return outlier_mask'''

        if old_iqr in content:
            content = content.replace(old_iqr, new_iqr)
            self.fixes_applied.append("detect_outliers_iqr")
            logger.info("✓ detect_outliers_iqr düzeltildi")

        # Fix 2: detect_outliers_zscore - line 64-77
        old_zscore = '''    def detect_outliers_zscore(self, df, columns, threshold=3):
        """Z-score method"""

        outlier_mask = pd.Series([False] * len(df))

        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = z_scores > threshold
            outlier_mask |= col_outliers

            n_outliers = col_outliers.sum()
            logger.info(f"  {col}: {n_outliers} outliers (Z>{threshold})")

        return outlier_mask'''

        new_zscore = '''    def detect_outliers_zscore(self, df, columns, threshold=3):
        """Z-score method"""

        outlier_mask = pd.Series([False] * len(df))

        for col in columns:
            # String değerleri numeric'e çevir
            col_data = pd.to_numeric(df[col], errors='coerce')

            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            col_outliers = z_scores > threshold
            outlier_mask |= col_outliers

            n_outliers = col_outliers.sum()
            logger.info(f"  {col}: {n_outliers} outliers (Z>{threshold})")

        return outlier_mask'''

        if old_zscore in content:
            content = content.replace(old_zscore, new_zscore)
            self.fixes_applied.append("detect_outliers_zscore")
            logger.info("✓ detect_outliers_zscore düzeltildi")

        # Fix 3: cap_outliers - line 110-133
        old_cap = '''    def cap_outliers(self, df, columns, method='iqr', threshold=1.5):
        """Cap outliers instead of removing"""

        df_capped = df.copy()

        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            else:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            df_capped[col] = df_capped[col].clip(lower_bound, upper_bound)

            n_capped = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            logger.info(f"  {col}: {n_capped} values capped")

        return df_capped'''

        new_cap = '''    def cap_outliers(self, df, columns, method='iqr', threshold=1.5):
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

        return df_capped'''

        if old_cap in content:
            content = content.replace(old_cap, new_cap)
            self.fixes_applied.append("cap_outliers")
            logger.info("✓ cap_outliers düzeltildi")

        # Fix 4: _check_value_ranges - line 252-264
        old_ranges = '''    def _check_value_ranges(self, df, ranges):
        """Check if values are within expected ranges"""

        logger.info("\\n-> Checking value ranges...")

        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()

                if out_of_range > 0:
                    issue = f"Out of range values in {col}: {out_of_range} (expected [{min_val}, {max_val}])"
                    self.validation_report.append(issue)
                    logger.warning(f"  [WARNING] {issue}")'''

        new_ranges = '''    def _check_value_ranges(self, df, ranges):
        """Check if values are within expected ranges"""

        logger.info("\\n-> Checking value ranges...")

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
                        logger.warning(f"  [WARNING] {issue}")'''

        if old_ranges in content:
            content = content.replace(old_ranges, new_ranges)
            self.fixes_applied.append("_check_value_ranges")
            logger.info("✓ _check_value_ranges düzeltildi")

        # Fix 5: _check_physical_constraints - line 266-287
        old_constraints = '''    def _check_physical_constraints(self, df):
        """Check nuclear physics constraints"""

        logger.info("\\n-> Checking physical constraints...")

        # A = Z + N
        if all(col in df.columns for col in ['A', 'Z', 'N']):
            mismatch = (df['A'] != df['Z'] + df['N']).sum()
            if mismatch > 0:
                issue = f"A ≠ Z + N mismatch: {mismatch} samples"
                self.validation_report.append(issue)
                logger.warning(f"  [WARNING] {issue}")

        # Z, N > 0
        if 'Z' in df.columns:
            invalid_z = (df['Z'] <= 0).sum()
            if invalid_z > 0:
                issue = f"Invalid Z values: {invalid_z}"
                self.validation_report.append(issue)
                logger.warning(f"  [WARNING] {issue}")

        logger.info("  [OK] Physical constraints checked")'''

        new_constraints = '''    def _check_physical_constraints(self, df):
        """Check nuclear physics constraints"""

        logger.info("\\n-> Checking physical constraints...")

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

        logger.info("  [OK] Physical constraints checked")'''

        if old_constraints in content:
            content = content.replace(old_constraints, new_constraints)
            self.fixes_applied.append("_check_physical_constraints")
            logger.info("✓ _check_physical_constraints düzeltildi")

        # Dosyayı kaydet
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"\n[OK] {file_path.name} başarıyla düzeltildi!")
        logger.info(f"Toplam {len(self.fixes_applied)} düzeltme uygulandı")

        return True

    def scan_all_files(self):
        """Tüm Python dosyalarını tara ve problemli olanları listele"""

        logger.info("\n" + "="*80)
        logger.info("PROJE GENELİNDE TARAMA BAŞLIYOR")
        logger.info("="*80)

        py_files = list(self.project_root.rglob("*.py"))
        problematic_files = []

        for py_file in py_files:
            # Test ve cache dosyalarını atla
            if '__pycache__' in str(py_file) or 'test_' in py_file.name:
                continue

            try:
                issues = self.find_comparison_patterns(py_file)
                if issues:
                    problematic_files.append({
                        'file': py_file,
                        'issues': issues
                    })
            except Exception as e:
                logger.warning(f"Hata: {py_file.name} - {e}")

        # Sonuçları raporla
        logger.info(f"\nToplam {len(py_files)} Python dosyası tarandı")
        logger.info(f"Potansiyel sorunlu dosya sayısı: {len(problematic_files)}")

        if problematic_files:
            logger.info("\n" + "="*80)
            logger.info("POTANSİYEL SORUNLU DOSYALAR")
            logger.info("="*80)

            for item in problematic_files:
                logger.info(f"\n📁 {item['file'].relative_to(self.project_root)}")
                logger.info(f"   Toplam {len(item['issues'])} potansiyel sorun")

                # İlk 3 satırı göster
                for issue in item['issues'][:3]:
                    logger.info(f"   Line {issue['line_num']}: {issue['line'].strip()}")

                if len(item['issues']) > 3:
                    logger.info(f"   ... ve {len(item['issues']) - 3} satır daha")

        return problematic_files

    def run(self):
        """Ana düzeltme işlemini çalıştır"""

        logger.info("\n" + "="*80)
        logger.info("NUMERİK KARŞILAŞTIRMA HATALARINI DÜZELT")
        logger.info("="*80)
        logger.info("\nHata Tipi: TypeError: '<' not supported between instances of 'str' and 'float'")
        logger.info("Çözüm: DataFrame sütunlarını pd.to_numeric() ile dönüştür\n")

        # Ana dosyayı düzelt
        self.fix_data_quality_modules()

        # Tüm projeyi tara
        problematic_files = self.scan_all_files()

        # Özet rapor
        logger.info("\n" + "="*80)
        logger.info("ÖZET RAPOR")
        logger.info("="*80)
        logger.info(f"✓ data_quality_modules.py düzeltildi")
        logger.info(f"✓ {len(self.fixes_applied)} fonksiyon güncellendi:")
        for fix in self.fixes_applied:
            logger.info(f"  - {fix}")

        logger.info(f"\n⚠ Kontrol edilmesi gereken {len(problematic_files)} dosya bulundu")

        logger.info("\n[OK] Düzeltme işlemi tamamlandı!")
        logger.info("\nÖneriler:")
        logger.info("1. Değişiklikleri test edin")
        logger.info("2. Diğer dosyaları manuel kontrol edin")
        logger.info("3. Git commit yapın")


def main():
    """Ana fonksiyon"""

    fixer = NumericComparisonFixer()
    fixer.run()


if __name__ == "__main__":
    main()
