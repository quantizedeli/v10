"""
Nuclear Physics AI Project - System Health Check
=================================================

Kapsamlı sistem sağlık kontrolü:
1. Dosya format desteği kontrolü
2. Gerekli modüllerin varlığı
3. Veri dosyalarının erişilebilirliği
4. Anomoli tespit sistemi kontrolü
5. Excel rapor yetenekleri
6. PFAZ modüllerinin durumu

Author: Nuclear Physics AI Research Team
Version: 1.0.0
Date: 2025-11-22
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemHealthChecker:
    """Sistem sağlık kontrolü"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or PROJECT_ROOT
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }

    def run_all_checks(self) -> Dict:
        """Tüm kontrolleri çalıştır"""
        logger.info("=" * 80)
        logger.info("NUCLEAR PHYSICS AI PROJECT - SYSTEM HEALTH CHECK")
        logger.info("=" * 80)
        logger.info(f"Project Root: {self.project_root}")
        logger.info(f"Check Time: {self.results['timestamp']}")
        logger.info("")

        # 1. File I/O Utilities Check
        self._check_file_io_utils()

        # 2. Data Files Check
        self._check_data_files()

        # 3. PFAZ Modules Check
        self._check_pfaz_modules()

        # 4. Anomaly Detection Check
        self._check_anomaly_detection()

        # 5. Excel Export Capabilities
        self._check_excel_capabilities()

        # 6. Required Libraries
        self._check_required_libraries()

        # 7. File Format Support
        self._check_file_format_support()

        # Final Summary
        self._print_summary()

        # Save report
        self._save_report()

        return self.results

    def _check_file_io_utils(self):
        """File I/O utilities kontrolü"""
        check_name = "File I/O Utilities"
        logger.info(f"\n[CHECK] {check_name}")
        logger.info("-" * 60)

        try:
            from utils.file_io_utils import read_nuclear_data, save_nuclear_data

            # Test file creation
            test_df = pd.DataFrame({
                'A': [10, 20],
                'Z': [5, 10],
                'N': [5, 10]
            })

            test_file = self.project_root / 'test_health_check.csv'
            save_nuclear_data(test_df, test_file, index=False)

            # Test file reading
            loaded_df = read_nuclear_data(test_file)

            # Cleanup
            test_file.unlink()

            if test_df.equals(loaded_df):
                self._record_check(check_name, 'PASS',
                                  'File I/O utilities working correctly')
            else:
                self._record_check(check_name, 'FAIL',
                                  'Data mismatch in read/write test')
        except Exception as e:
            self._record_check(check_name, 'FAIL', str(e))

    def _check_data_files(self):
        """Veri dosyaları kontrolü"""
        check_name = "Data Files"
        logger.info(f"\n[CHECK] {check_name}")
        logger.info("-" * 60)

        required_files = ['aaa2.txt']
        optional_files = ['aaa2.csv', 'aaa2.xlsx']

        found_files = []
        missing_files = []

        for file in required_files:
            file_path = self.project_root / file
            if file_path.exists():
                found_files.append(file)
                logger.info(f"  [OK] Found: {file}")
            else:
                missing_files.append(file)
                logger.warning(f"  [MISSING] Not found: {file}")

        for file in optional_files:
            file_path = self.project_root / file
            if file_path.exists():
                logger.info(f"  [OPTIONAL] Found: {file}")

        if missing_files:
            self._record_check(check_name, 'WARN',
                              f'Missing files: {", ".join(missing_files)}')
        else:
            self._record_check(check_name, 'PASS',
                              f'All required data files found')

    def _check_pfaz_modules(self):
        """PFAZ modülleri kontrolü"""
        check_name = "PFAZ Modules"
        logger.info(f"\n[CHECK] {check_name}")
        logger.info("-" * 60)

        pfaz_modules = {
            'PFAZ 1': 'pfaz_modules.pfaz01_dataset_generation.dataset_generation_pipeline_v2',
            'PFAZ 2': 'pfaz_modules.pfaz02_ai_training.parallel_ai_trainer',
            'PFAZ 3': 'pfaz_modules.pfaz03_anfis_training.anfis_dataset_selector',
            'PFAZ 9': 'pfaz_modules.pfaz09_aaa2_monte_carlo.aaa2_quality_checker',
            'PFAZ 11': 'pfaz_modules.pfaz11_production.production_model_serving'
        }

        available = []
        missing = []

        for pfaz_name, module_path in pfaz_modules.items():
            try:
                __import__(module_path)
                available.append(pfaz_name)
                logger.info(f"  [OK] {pfaz_name}: Available")
            except ImportError as e:
                missing.append(pfaz_name)
                logger.warning(f"  [MISSING] {pfaz_name}: {str(e)}")

        if missing:
            self._record_check(check_name, 'WARN',
                              f'{len(available)}/{len(pfaz_modules)} modules available')
        else:
            self._record_check(check_name, 'PASS',
                              'All critical PFAZ modules available')

    def _check_anomaly_detection(self):
        """Anomali tespit sistemi kontrolü"""
        check_name = "Anomaly Detection System"
        logger.info(f"\n[CHECK] {check_name}")
        logger.info("-" * 60)

        try:
            from core_modules.anomaly_detector import AnomalyDetector

            # Test anomaly detection
            import numpy as np
            np.random.seed(42)

            test_df = pd.DataFrame({
                'NUCLEUS': [f'N{i}' for i in range(50)],
                'Z': np.random.randint(10, 50, 50),
                'N': np.random.randint(10, 60, 50),
                'A': np.random.randint(20, 100, 50),
                'SPIN': np.random.choice([0.5, 1.0, 1.5], 50),
                'PARITY': np.random.choice([-1, 1], 50),
                'MM': np.random.normal(0, 1, 50)
            })

            # Add some anomalies
            test_df.loc[48, 'MM'] = 10.0
            test_df.loc[49, 'MM'] = -10.0

            detector = AnomalyDetector(
                contamination=0.1,
                output_dir=self.project_root / 'test_anomaly_output'
            )
            result_df = detector.detect_anomalies(test_df, ['Z', 'N', 'MM', 'SPIN'])

            # Check if anomalies detected
            n_anomalies = result_df['is_anomaly'].sum()

            # Check if Excel report created
            report_path = self.project_root / 'test_anomaly_output' / 'Anomaly_Detection.xlsx'
            report_exists = report_path.exists()

            # Cleanup
            import shutil
            if (self.project_root / 'test_anomaly_output').exists():
                shutil.rmtree(self.project_root / 'test_anomaly_output')

            if n_anomalies > 0 and report_exists:
                self._record_check(check_name, 'PASS',
                                  f'Anomaly detection working (detected {n_anomalies} anomalies, Excel report created)')
            else:
                self._record_check(check_name, 'FAIL',
                                  'Anomaly detection issues detected')
        except Exception as e:
            self._record_check(check_name, 'FAIL', str(e))

    def _check_excel_capabilities(self):
        """Excel yetenekleri kontrolü"""
        check_name = "Excel Export Capabilities"
        logger.info(f"\n[CHECK] {check_name}")
        logger.info("-" * 60)

        try:
            import openpyxl
            import xlsxwriter

            # Test Excel creation
            test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            test_excel = self.project_root / 'test_excel.xlsx'

            with pd.ExcelWriter(test_excel, engine='openpyxl') as writer:
                test_df.to_excel(writer, sheet_name='Sheet1', index=False)
                test_df.to_excel(writer, sheet_name='Sheet2', index=False)

            # Verify
            with pd.ExcelFile(test_excel) as xls:
                sheets = xls.sheet_names

            # Cleanup
            test_excel.unlink()

            if len(sheets) >= 2:
                self._record_check(check_name, 'PASS',
                                  'Excel export working (openpyxl, xlsxwriter available)')
            else:
                self._record_check(check_name, 'FAIL',
                                  'Excel multi-sheet creation failed')
        except Exception as e:
            self._record_check(check_name, 'FAIL', str(e))

    def _check_required_libraries(self):
        """Gerekli kütüphaneler kontrolü"""
        check_name = "Required Libraries"
        logger.info(f"\n[CHECK] {check_name}")
        logger.info("-" * 60)

        required_libs = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'scikit-learn': 'sklearn',
            'scipy': 'scipy',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'openpyxl': 'openpyxl',
            'xlsxwriter': 'xlsxwriter'
        }

        missing = []
        available = []

        for lib_name, import_name in required_libs.items():
            try:
                __import__(import_name)
                available.append(lib_name)
                logger.info(f"  [OK] {lib_name}")
            except ImportError:
                missing.append(lib_name)
                logger.warning(f"  [MISSING] {lib_name}")

        if missing:
            self._record_check(check_name, 'FAIL',
                              f'Missing libraries: {", ".join(missing)}')
        else:
            self._record_check(check_name, 'PASS',
                              'All required libraries available')

    def _check_file_format_support(self):
        """Dosya format desteği kontrolü"""
        check_name = "File Format Support"
        logger.info(f"\n[CHECK] {check_name}")
        logger.info("-" * 60)

        try:
            from utils.file_io_utils import read_nuclear_data

            test_data = pd.DataFrame({
                'A': [10, 20, 30],
                'Z': [5, 10, 15],
                'N': [5, 10, 15]
            })

            formats_to_test = {
                '.csv': lambda path: test_data.to_csv(path, index=False),
                '.txt': lambda path: test_data.to_csv(path, sep='\t', index=False),
                '.tsv': lambda path: test_data.to_csv(path, sep='\t', index=False),
                '.xlsx': lambda path: test_data.to_excel(path, index=False)
            }

            supported = []
            failed = []

            for ext, save_func in formats_to_test.items():
                test_file = self.project_root / f'test_format{ext}'
                try:
                    save_func(test_file)
                    loaded = read_nuclear_data(test_file)
                    test_file.unlink()

                    if len(loaded) == len(test_data):
                        supported.append(ext)
                        logger.info(f"  [OK] Format {ext}: Supported")
                    else:
                        failed.append(ext)
                        logger.warning(f"  [FAIL] Format {ext}: Data mismatch")
                except Exception as e:
                    failed.append(ext)
                    logger.warning(f"  [FAIL] Format {ext}: {str(e)}")
                    if test_file.exists():
                        test_file.unlink()

            if not failed:
                self._record_check(check_name, 'PASS',
                                  f'All formats supported: {", ".join(supported)}')
            else:
                self._record_check(check_name, 'WARN',
                                  f'Some formats failed: {", ".join(failed)}')
        except Exception as e:
            self._record_check(check_name, 'FAIL', str(e))

    def _record_check(self, check_name: str, status: str, message: str):
        """Kontrol sonucunu kaydet"""
        self.results['checks'][check_name] = {
            'status': status,
            'message': message
        }

        if status == 'PASS':
            self.results['passed'] += 1
            logger.info(f"  ✓ PASS: {message}")
        elif status == 'FAIL':
            self.results['failed'] += 1
            logger.error(f"  ✗ FAIL: {message}")
        elif status == 'WARN':
            self.results['warnings'] += 1
            logger.warning(f"  ⚠ WARN: {message}")

    def _print_summary(self):
        """Özet yazdır"""
        logger.info("\n" + "=" * 80)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("=" * 80)

        total_checks = len(self.results['checks'])
        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"✓ Passed: {self.results['passed']}")
        logger.info(f"⚠ Warnings: {self.results['warnings']}")
        logger.info(f"✗ Failed: {self.results['failed']}")

        health_percentage = (self.results['passed'] / total_checks * 100) if total_checks > 0 else 0
        logger.info(f"\nSystem Health: {health_percentage:.1f}%")

        if self.results['failed'] == 0 and self.results['warnings'] == 0:
            logger.info("\n🎉 SYSTEM STATUS: EXCELLENT - All checks passed!")
        elif self.results['failed'] == 0:
            logger.info("\n✅ SYSTEM STATUS: GOOD - All critical checks passed")
        elif self.results['failed'] < 3:
            logger.info("\n⚠️  SYSTEM STATUS: FAIR - Some issues detected")
        else:
            logger.info("\n❌ SYSTEM STATUS: POOR - Multiple issues detected")

        logger.info("=" * 80)

    def _save_report(self):
        """Raporu kaydet"""
        report_dir = self.project_root / 'reports' / 'health_checks'
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON report
        json_path = report_dir / f'health_check_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n[REPORT] JSON saved: {json_path}")

        # Excel report
        try:
            excel_path = report_dir / f'health_check_{timestamp}.xlsx'

            # Create summary DataFrame
            summary_data = [{
                'Metric': 'Total Checks',
                'Value': len(self.results['checks'])
            }, {
                'Metric': 'Passed',
                'Value': self.results['passed']
            }, {
                'Metric': 'Warnings',
                'Value': self.results['warnings']
            }, {
                'Metric': 'Failed',
                'Value': self.results['failed']
            }, {
                'Metric': 'Health %',
                'Value': f"{(self.results['passed'] / len(self.results['checks']) * 100):.1f}%"
            }]

            summary_df = pd.DataFrame(summary_data)

            # Create detailed DataFrame
            details_data = []
            for check_name, check_result in self.results['checks'].items():
                details_data.append({
                    'Check': check_name,
                    'Status': check_result['status'],
                    'Message': check_result['message']
                })

            details_df = pd.DataFrame(details_data)

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                details_df.to_excel(writer, sheet_name='Details', index=False)

            logger.info(f"[REPORT] Excel saved: {excel_path}")
        except Exception as e:
            logger.warning(f"[REPORT] Could not save Excel: {e}")


def main():
    """Ana fonksiyon"""
    checker = SystemHealthChecker()
    results = checker.run_all_checks()

    # Exit code
    exit_code = 0 if results['failed'] == 0 else 1
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
