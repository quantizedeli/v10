"""
COMPREHENSIVE REPORTS MODULE
============================

Tez için detaylı ve kapsamlı raporlama sistemi

Özellikler:
- Multi-sheet Excel raporları (20+ sheet)
- JSON raporları (yapılandırılmış veri)
- CSV exportları (analiz için)
- HTML summary raporları
- Report metadata ve tracking
- Otomatik validation ve quality checks

Author: Nuclear Physics AI Project
Date: October 2025
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# COMPREHENSIVE REPORTS BUILDER
# =============================================================================

class ComprehensiveReportsBuilder:
    """Kapsamlı raporlar oluşturan ana sınıf"""
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report tracking
        self.reports_generated = []
        self.report_metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_reports': 0,
            'total_sheets': 0,
            'reports': []
        }
        
        logger.info(f"✓ Reports Builder initialized at {self.output_dir}")
    
    # =========================================================================
    # 1. EXCEL REPORTS
    # =========================================================================
    
    def create_master_excel_report(self,
                                   results_df: pd.DataFrame,
                                   model_metrics: Dict[str, Dict],
                                   training_history: Optional[Dict] = None,
                                   anomalies: Optional[pd.DataFrame] = None,
                                   config: Optional[Dict] = None,
                                   save_name: str = 'MASTER_REPORT') -> Path:
        """
        Master Excel raporu - 15+ sheet ile kapsamlı sonuçlar
        
        Sheets:
        1. Summary (Özet)
        2. Model_Performance (Model performansı)
        3. Results_All (Tüm sonuçlar)
        4. Results_by_Target (Target'e göre)
        5. Results_by_Model (Modele göre)
        6. Results_by_Config (Konfigürasyona göre)
        7. Training_History (Eğitim geçmişi)
        8. Metrics_Comparison (Metrik karşılaştırması)
        9. Statistical_Summary (İstatistiksel özet)
        10. Top_Performers (En iyi performans gösterenler)
        11. Anomalies (Anomali analizi)
        12. Errors_Analysis (Hata analizi)
        13. Feature_Statistics (Özellik istatistikleri)
        14. Dataset_Info (Veri seti bilgisi)
        15. Report_Metadata (Rapor metadata)
        """
        
        if not OPENPYXL_AVAILABLE:
            logger.warning("openpyxl not available, using pandas Excel writer")
            return self._create_pandas_excel(results_df, model_metrics, save_name)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Creating Master Excel Report: {save_name}")
        logger.info(f"{'='*70}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = self.output_dir / f'{save_name}_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            workbook = writer.book
            
            # Format tanımla
            formats = self._create_excel_formats(workbook)
            sheet_count = 0
            
            # Sheet 1: Summary
            logger.info("  1/15: Creating Summary sheet...")
            self._create_summary_sheet(writer, results_df, model_metrics, formats)
            sheet_count += 1
            
            # Sheet 2: Model Performance
            logger.info("  2/15: Creating Model Performance sheet...")
            self._create_model_performance_sheet(writer, model_metrics, formats)
            sheet_count += 1
            
            # Sheet 3: All Results
            logger.info("  3/15: Creating All Results sheet...")
            self._create_all_results_sheet(writer, results_df, formats)
            sheet_count += 1
            
            # Sheet 4: Results by Target
            if 'target' in results_df.columns:
                logger.info("  4/15: Creating Results by Target sheet...")
                self._create_results_by_target_sheet(writer, results_df, formats)
                sheet_count += 1
            
            # Sheet 5: Results by Model
            if 'model' in results_df.columns or 'model_name' in results_df.columns:
                logger.info("  5/15: Creating Results by Model sheet...")
                self._create_results_by_model_sheet(writer, results_df, formats)
                sheet_count += 1
            
            # Sheet 6: Results by Config
            if 'config' in results_df.columns or 'config_id' in results_df.columns:
                logger.info("  6/15: Creating Results by Config sheet...")
                self._create_results_by_config_sheet(writer, results_df, formats)
                sheet_count += 1
            
            # Sheet 7: Training History
            if training_history:
                logger.info("  7/15: Creating Training History sheet...")
                self._create_training_history_sheet(writer, training_history, formats)
                sheet_count += 1
            
            # Sheet 8: Metrics Comparison
            logger.info("  8/15: Creating Metrics Comparison sheet...")
            self._create_metrics_comparison_sheet(writer, model_metrics, formats)
            sheet_count += 1
            
            # Sheet 9: Statistical Summary
            logger.info("  9/15: Creating Statistical Summary sheet...")
            self._create_statistical_summary_sheet(writer, results_df, formats)
            sheet_count += 1
            
            # Sheet 10: Top Performers
            logger.info(" 10/15: Creating Top Performers sheet...")
            self._create_top_performers_sheet(writer, results_df, formats)
            sheet_count += 1
            
            # Sheet 11: Anomalies
            if anomalies is not None and len(anomalies) > 0:
                logger.info(" 11/15: Creating Anomalies sheet...")
                self._create_anomalies_sheet(writer, anomalies, formats)
                sheet_count += 1
            
            # Sheet 12: Errors Analysis
            logger.info(" 12/15: Creating Errors Analysis sheet...")
            self._create_errors_analysis_sheet(writer, results_df, formats)
            sheet_count += 1
            
            # Sheet 13: Feature Statistics
            if 'features' in results_df.columns or results_df.shape[1] > 5:
                logger.info(" 13/15: Creating Feature Statistics sheet...")
                self._create_feature_statistics_sheet(writer, results_df, formats)
                sheet_count += 1
            
            # Sheet 14: Dataset Info
            logger.info(" 14/15: Creating Dataset Info sheet...")
            self._create_dataset_info_sheet(writer, results_df, formats)
            sheet_count += 1
            
            # Sheet 15: Report Metadata
            logger.info(" 15/15: Creating Report Metadata sheet...")
            self._create_report_metadata_sheet(writer, formats, sheet_count, config)
            sheet_count += 1
        
        logger.info(f"✓ Excel report saved: {excel_file.name}")
        logger.info(f"  Total sheets: {sheet_count}")
        
        self.reports_generated.append({
            'type': 'Excel',
            'file': str(excel_file),
            'sheets': sheet_count
        })
        
        return excel_file
    
    def _create_excel_formats(self, workbook):
        """Excel format'ları tanımla"""
        return {
            'header': workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            }),
            'title': workbook.add_format({
                'bold': True,
                'font_size': 14,
                'bg_color': '#D9E1F2',
                'align': 'left',
                'border': 1
            }),
            'number': workbook.add_format({
                'num_format': '0.0000',
                'align': 'center'
            }),
            'percent': workbook.add_format({
                'num_format': '0.00%',
                'align': 'center'
            }),
            'good': workbook.add_format({
                'bg_color': '#C6EFCE',
                'font_color': '#006100'
            }),
            'bad': workbook.add_format({
                'bg_color': '#FFC7CE',
                'font_color': '#9C0006'
            }),
            'warning': workbook.add_format({
                'bg_color': '#FFEB9C',
                'font_color': '#9C6500'
            })
        }
    
    def _create_summary_sheet(self, writer, df, metrics, formats):
        """Sheet 1: Özet"""
        summary_data = {
            'Metrik': [
                'Toplam Sonuç Sayısı',
                'Model Sayısı',
                'Target Sayısı',
                'Ortalama R² Skoru',
                'Maksimum R² Skoru',
                'Minimum R² Skoru',
                'Ortalama MAE',
                'Ortalama RMSE',
                'İyi Sonuçlar (R²>0.90)',
                'Kabul Edilebilir Sonuçlar (R²>0.70)',
                'Kötü Sonuçlar (R²<0.70)'
            ],
            'Değer': [
                len(df),
                len(metrics),
                df['target'].nunique() if 'target' in df.columns else 1,
                df['r2'].mean() if 'r2' in df.columns else 0,
                df['r2'].max() if 'r2' in df.columns else 0,
                df['r2'].min() if 'r2' in df.columns else 0,
                df['mae'].mean() if 'mae' in df.columns else 0,
                df['rmse'].mean() if 'rmse' in df.columns else 0,
                (df['r2'] > 0.90).sum() if 'r2' in df.columns else 0,
                (df['r2'] > 0.70).sum() if 'r2' in df.columns else 0,
                (df['r2'] < 0.70).sum() if 'r2' in df.columns else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='1_Özet', index=False)
        
        logger.info("    ✓ Summary sheet created")
    
    def _create_model_performance_sheet(self, writer, metrics, formats):
        """Sheet 2: Model Performance"""
        perf_data = []
        
        for model, meta in metrics.items():
            perf_data.append({
                'Model': model,
                'R² Score': meta.get('r2', np.nan),
                'MAE': meta.get('mae', np.nan),
                'RMSE': meta.get('rmse', np.nan),
                'Training Time (s)': meta.get('training_time', np.nan),
                'Prediction Variance': meta.get('prediction_variance', np.nan),
                'Status': 'Good' if meta.get('r2', 0) > 0.90 else 'Fair' if meta.get('r2', 0) > 0.70 else 'Poor'
            })
        
        perf_df = pd.DataFrame(perf_data).sort_values('R² Score', ascending=False)
        perf_df.to_excel(writer, sheet_name='2_Model_Performance', index=False)
        
        logger.info("    ✓ Model Performance sheet created")
    
    def _create_all_results_sheet(self, writer, df, formats):
        """Sheet 3: All Results"""
        df_sorted = df.sort_values('r2', ascending=False) if 'r2' in df.columns else df
        df_sorted.to_excel(writer, sheet_name='3_All_Results', index=False)
        
        logger.info("    ✓ All Results sheet created")
    
    def _create_results_by_target_sheet(self, writer, df, formats):
        """Sheet 4: Results by Target"""
        targets = df['target'].unique()
        
        for target in targets:
            target_df = df[df['target'] == target].sort_values('r2', ascending=False) if 'r2' in df.columns else df[df['target'] == target]
            sheet_name = f'Target_{str(target)[:10]}'[:31]
            target_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"    ✓ Results by Target sheet created ({len(targets)} targets)")
    
    def _create_results_by_model_sheet(self, writer, df, formats):
        """Sheet 5: Results by Model"""
        model_col = 'model' if 'model' in df.columns else 'model_name' if 'model_name' in df.columns else None
        
        if model_col:
            models = df[model_col].unique()
            
            for model in models:
                model_df = df[df[model_col] == model].sort_values('r2', ascending=False) if 'r2' in df.columns else df[df[model_col] == model]
                sheet_name = f'Model_{str(model)[:10]}'[:31]
                model_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"    ✓ Results by Model sheet created ({len(models)} models)")
    
    def _create_results_by_config_sheet(self, writer, df, formats):
        """Sheet 6: Results by Config"""
        config_col = 'config' if 'config' in df.columns else 'config_id' if 'config_id' in df.columns else None
        
        if config_col:
            configs = df[config_col].unique()
            
            for config in configs:
                config_df = df[df[config_col] == config].sort_values('r2', ascending=False) if 'r2' in df.columns else df[df[config_col] == config]
                sheet_name = f'Cfg_{str(config)[:10]}'[:31]
                config_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"    ✓ Results by Config sheet created ({len(configs)} configs)")
    
    def _create_training_history_sheet(self, writer, history, formats):
        """Sheet 7: Training History"""
        history_df = pd.DataFrame(history)
        history_df.to_excel(writer, sheet_name='7_Training_History', index=False)
        
        logger.info("    ✓ Training History sheet created")
    
    def _create_metrics_comparison_sheet(self, writer, metrics, formats):
        """Sheet 8: Metrics Comparison"""
        comparison_data = []
        
        for model, meta in metrics.items():
            comparison_data.append({
                'Model': model,
                'R²': meta.get('r2', np.nan),
                'MAE': meta.get('mae', np.nan),
                'RMSE': meta.get('rmse', np.nan),
                'MAPE': meta.get('mape', np.nan),
                'Correlation': meta.get('correlation', np.nan)
            })
        
        comp_df = pd.DataFrame(comparison_data)
        comp_df.to_excel(writer, sheet_name='8_Metrics_Comparison', index=False)
        
        logger.info("    ✓ Metrics Comparison sheet created")
    
    def _create_statistical_summary_sheet(self, writer, df, formats):
        """Sheet 9: Statistical Summary"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats_data = []
        for col in numeric_cols:
            stats_data.append({
                'Column': col,
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Median': df[col].median(),
                'Q25': df[col].quantile(0.25),
                'Q75': df[col].quantile(0.75)
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='9_Statistical_Summary', index=False)
        
        logger.info("    ✓ Statistical Summary sheet created")
    
    def _create_top_performers_sheet(self, writer, df, formats):
        """Sheet 10: Top Performers"""
        if 'r2' in df.columns:
            top_n = min(20, len(df))
            top_df = df.nlargest(top_n, 'r2')
        else:
            top_df = df.head(20)
        
        top_df.to_excel(writer, sheet_name='10_Top_Performers', index=False)
        
        logger.info(f"    ✓ Top Performers sheet created ({len(top_df)} records)")
    
    def _create_anomalies_sheet(self, writer, anomalies, formats):
        """Sheet 11: Anomalies"""
        anomalies.to_excel(writer, sheet_name='11_Anomalies', index=False)
        
        logger.info(f"    ✓ Anomalies sheet created ({len(anomalies)} anomalies)")
    
    def _create_errors_analysis_sheet(self, writer, df, formats):
        """Sheet 12: Errors Analysis"""
        errors_data = []
        
        if 'r2' in df.columns:
            errors_data.append({'Metric': 'Total Records', 'Value': len(df)})
            errors_data.append({'Metric': 'Low R² (<0.5)', 'Value': (df['r2'] < 0.5).sum()})
            errors_data.append({'Metric': 'Medium R² (0.5-0.7)', 'Value': ((df['r2'] >= 0.5) & (df['r2'] < 0.7)).sum()})
            errors_data.append({'Metric': 'High R² (0.7-0.9)', 'Value': ((df['r2'] >= 0.7) & (df['r2'] < 0.9)).sum()})
            errors_data.append({'Metric': 'Very High R² (>0.9)', 'Value': (df['r2'] >= 0.9).sum()})
        
        if 'mae' in df.columns:
            errors_data.append({'Metric': 'Max MAE', 'Value': df['mae'].max()})
            errors_data.append({'Metric': 'Mean MAE', 'Value': df['mae'].mean()})
            errors_data.append({'Metric': 'Std MAE', 'Value': df['mae'].std()})
        
        errors_df = pd.DataFrame(errors_data)
        errors_df.to_excel(writer, sheet_name='12_Errors_Analysis', index=False)
        
        logger.info("    ✓ Errors Analysis sheet created")
    
    def _create_feature_statistics_sheet(self, writer, df, formats):
        """Sheet 13: Feature Statistics"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 0:
            corr_matrix = numeric_df.corr()
            corr_matrix.to_excel(writer, sheet_name='13_Feature_Statistics')
        
        logger.info("    ✓ Feature Statistics sheet created")
    
    def _create_dataset_info_sheet(self, writer, df, formats):
        """Sheet 14: Dataset Info"""
        info_data = {
            'Property': [
                'Total Records',
                'Total Columns',
                'Numeric Columns',
                'String Columns',
                'Missing Values',
                'Duplicate Records',
                'Memory Usage (MB)'
            ],
            'Value': [
                len(df),
                len(df.columns),
                len(df.select_dtypes(include=[np.number]).columns),
                len(df.select_dtypes(include=['object']).columns),
                df.isnull().sum().sum(),
                df.duplicated().sum(),
                df.memory_usage(deep=True).sum() / 1024**2
            ]
        }
        
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='14_Dataset_Info', index=False)
        
        logger.info("    ✓ Dataset Info sheet created")
    
    def _create_report_metadata_sheet(self, writer, formats, sheet_count, config):
        """Sheet 15: Report Metadata"""
        metadata = {
            'Property': [
                'Report Timestamp',
                'Total Sheets',
                'Python Version',
                'Pandas Version',
                'Report Generated By',
                'Report Purpose',
                'Data Preparation',
                'Analysis Type'
            ],
            'Value': [
                datetime.now().isoformat(),
                sheet_count,
                '3.8+',
                pd.__version__,
                'Nuclear Physics AI Project',
                'Comprehensive Results Analysis',
                'Automated Pipeline',
                'Multi-Model Comparison'
            ]
        }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name='15_Metadata', index=False)
        
        logger.info("    ✓ Report Metadata sheet created")
    
    def _create_pandas_excel(self, results_df, model_metrics, save_name):
        """Pandas ile Excel oluştur (fallback)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = self.output_dir / f'{save_name}_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            metrics_df = pd.DataFrame(model_metrics).T
            metrics_df.to_excel(writer, sheet_name='Model_Metrics')
        
        logger.info(f"✓ Excel report (pandas) saved: {excel_file.name}")
        return excel_file
    
    # =========================================================================
    # 2. JSON REPORTS
    # =========================================================================
    
    def create_json_report(self,
                          results_df: pd.DataFrame,
                          model_metrics: Dict,
                          metadata: Optional[Dict] = None,
                          save_name: str = 'REPORT') -> Path:
        """
        Kapsamlı JSON raporu oluştur
        
        Yapı:
        {
            "metadata": {...},
            "summary": {...},
            "models": {...},
            "results": [...],
            "statistics": {...},
            "quality_metrics": {...}
        }
        """
        logger.info(f"\nCreating JSON Report: {save_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f'{save_name}_{timestamp}.json'
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'report_name': save_name,
                'total_records': len(results_df),
                'total_models': len(model_metrics),
                'generated_by': 'Nuclear Physics AI Project'
            },
            'summary': {
                'total_results': len(results_df),
                'unique_targets': results_df['target'].nunique() if 'target' in results_df.columns else 1,
                'average_r2': float(results_df['r2'].mean()) if 'r2' in results_df.columns else None,
                'best_r2': float(results_df['r2'].max()) if 'r2' in results_df.columns else None,
                'worst_r2': float(results_df['r2'].min()) if 'r2' in results_df.columns else None
            },
            'models': model_metrics,
            'results': results_df.to_dict('records'),
            'statistics': {
                'numeric_summary': {}
            },
            'quality_metrics': {
                'good_results': int((results_df['r2'] > 0.90).sum()) if 'r2' in results_df.columns else 0,
                'acceptable_results': int((results_df['r2'] > 0.70).sum()) if 'r2' in results_df.columns else 0,
                'poor_results': int((results_df['r2'] < 0.70).sum()) if 'r2' in results_df.columns else 0
            }
        }
        
        # Add numeric statistics
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report['statistics']['numeric_summary'][col] = {
                'mean': float(results_df[col].mean()),
                'std': float(results_df[col].std()),
                'min': float(results_df[col].min()),
                'max': float(results_df[col].max())
            }
        
        # Save JSON
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ JSON report saved: {json_file.name}")
        
        return json_file
    
    # =========================================================================
    # 3. CSV EXPORTS
    # =========================================================================
    
    def create_csv_exports(self,
                          results_df: pd.DataFrame,
                          model_metrics: Dict,
                          export_dir: Optional[str] = None) -> List[Path]:
        """
        CSV export'ları oluştur (analiz ve paylaşım için)
        
        Exports:
        - results_all.csv (tüm sonuçlar)
        - model_metrics.csv (model metrikleri)
        - results_by_target.csv (target'e göre)
        - top_performers.csv (en iyi performans gösterenler)
        - errors_summary.csv (hata özeti)
        """
        logger.info("\nCreating CSV Exports")
        
        if export_dir:
            csv_dir = Path(export_dir)
        else:
            csv_dir = self.output_dir / 'csv_exports'
        
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        files = []
        
        # 1. All results
        csv_file = csv_dir / 'results_all.csv'
        results_df.to_csv(csv_file, index=False)
        files.append(csv_file)
        logger.info(f"  ✓ {csv_file.name}")
        
        # 2. Model metrics
        metrics_df = pd.DataFrame(model_metrics).T
        csv_file = csv_dir / 'model_metrics.csv'
        metrics_df.to_csv(csv_file)
        files.append(csv_file)
        logger.info(f"  ✓ {csv_file.name}")
        
        # 3. Results by target
        if 'target' in results_df.columns:
            csv_file = csv_dir / 'results_by_target.csv'
            target_summary = results_df.groupby('target').agg({
                'r2': ['count', 'mean', 'std', 'min', 'max'],
                'mae': ['mean', 'std'],
                'rmse': ['mean', 'std']
            }).round(4)
            target_summary.to_csv(csv_file)
            files.append(csv_file)
            logger.info(f"  ✓ {csv_file.name}")
        
        # 4. Top performers
        if 'r2' in results_df.columns:
            csv_file = csv_dir / 'top_performers.csv'
            top_df = results_df.nlargest(20, 'r2')
            top_df.to_csv(csv_file, index=False)
            files.append(csv_file)
            logger.info(f"  ✓ {csv_file.name}")
        
        # 5. Errors summary
        csv_file = csv_dir / 'errors_summary.csv'
        if 'r2' in results_df.columns:
            errors = pd.DataFrame({
                'Category': ['Low R² (<0.5)', 'Medium R² (0.5-0.7)', 'High R² (0.7-0.9)', 'Very High R² (>0.9)'],
                'Count': [
                    (results_df['r2'] < 0.5).sum(),
                    ((results_df['r2'] >= 0.5) & (results_df['r2'] < 0.7)).sum(),
                    ((results_df['r2'] >= 0.7) & (results_df['r2'] < 0.9)).sum(),
                    (results_df['r2'] >= 0.9).sum()
                ]
            })
            errors.to_csv(csv_file, index=False)
            files.append(csv_file)
            logger.info(f"  ✓ {csv_file.name}")
        
        logger.info(f"✓ CSV exports completed ({len(files)} files)")
        
        return files
    
    # =========================================================================
    # 4. HTML SUMMARY REPORTS
    # =========================================================================
    
    def create_html_summary_report(self,
                                   results_df: pd.DataFrame,
                                   model_metrics: Dict,
                                   save_name: str = 'REPORT') -> Path:
        """
        HTML özet raporu (sunumlar için uygun)
        """
        logger.info(f"\nCreating HTML Summary Report: {save_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = self.output_dir / f'{save_name}_{timestamp}.html'
        
        # HTML content oluştur
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{save_name} - Comprehensive Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            line-height: 1.6;
            color: #333;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .stat-card h3 {{ color: #667eea; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .section {{ 
            background: white; 
            padding: 30px; 
            margin-bottom: 20px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{ 
            color: #667eea; 
            margin-bottom: 20px; 
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th {{ 
            background-color: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-weight: 600;
            color: #667eea;
        }}
        td {{ 
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{ background-color: #f9f9f9; }}
        .good {{ background-color: #d4edda; color: #155724; }}
        .warning {{ background-color: #fff3cd; color: #856404; }}
        .bad {{ background-color: #f8d7da; color: #721c24; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{save_name}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Results</h3>
                <div class="value">{len(results_df)}</div>
            </div>
            <div class="stat-card">
                <h3>Models</h3>
                <div class="value">{len(model_metrics)}</div>
            </div>
            <div class="stat-card">
                <h3>Average R²</h3>
                <div class="value">{results_df['r2'].mean():.3f}</div>
            </div>
            <div class="stat-card">
                <h3>Best R²</h3>
                <div class="value">{results_df['r2'].max():.3f}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Model Performance</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>R² Score</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>Status</th>
                </tr>
"""
        
        for model, meta in model_metrics.items():
            r2 = meta.get('r2', 0)
            status_class = 'good' if r2 > 0.90 else 'warning' if r2 > 0.70 else 'bad'
            status = 'Good' if r2 > 0.90 else 'Fair' if r2 > 0.70 else 'Poor'
            
            html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{r2:.4f}</td>
                    <td>{meta.get('mae', 0):.4f}</td>
                    <td>{meta.get('rmse', 0):.4f}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
"""
        
        html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Results Summary</h2>
            <p>Total records analyzed: <strong>""" + str(len(results_df)) + """</strong></p>
            <p>Good results (R²>0.90): <strong>""" + str((results_df['r2'] > 0.90).sum()) + """</strong></p>
            <p>Acceptable results (R²>0.70): <strong>""" + str((results_df['r2'] > 0.70).sum()) + """</strong></p>
            <p>Poor results (R²<0.70): <strong>""" + str((results_df['r2'] < 0.70).sum()) + """</strong></p>
        </div>
        
        <div class="footer">
            <p>Report generated by Nuclear Physics AI Project</p>
            <p>© 2025 - Comprehensive Analysis System</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✓ HTML report saved: {html_file.name}")
        
        return html_file
    
    # =========================================================================
    # 5. GENERATE ALL REPORTS
    # =========================================================================
    
    def generate_all_reports(self,
                            results_df: pd.DataFrame,
                            model_metrics: Dict,
                            training_history: Optional[Dict] = None,
                            anomalies: Optional[pd.DataFrame] = None,
                            config: Optional[Dict] = None) -> Dict[str, Path]:
        """Tüm raporları oluştur"""
        
        logger.info("\n" + "="*70)
        logger.info("GENERATING ALL REPORTS")
        logger.info("="*70)
        
        reports = {}
        
        # Excel report
        reports['excel'] = self.create_master_excel_report(
            results_df, model_metrics, training_history, anomalies, config
        )
        
        # JSON report
        reports['json'] = self.create_json_report(results_df, model_metrics, config)
        
        # CSV exports
        reports['csv'] = self.create_csv_exports(results_df, model_metrics)
        
        # HTML report
        reports['html'] = self.create_html_summary_report(results_df, model_metrics)
        
        logger.info("\n" + "="*70)
        logger.info("✓ ALL REPORTS GENERATED")
        logger.info("="*70)
        logger.info(f"\n  Excel: {reports['excel'].name}")
        logger.info(f"  JSON:  {reports['json'].name}")
        logger.info(f"  CSV:   {len(reports['csv'])} files")
        logger.info(f"  HTML:  {reports['html'].name}")
        
        return reports


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test ve demo"""
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE REPORTS MODULE - TEST")
    logger.info("="*70)
    
    # Test data oluştur
    n_samples = 50
    test_results = pd.DataFrame({
        'model': np.random.choice(['RF', 'DNN', 'ANFIS'], n_samples),
        'target': np.random.choice(['MM', 'QM', 'Beta_2'], n_samples),
        'r2': np.random.uniform(0.6, 0.98, n_samples),
        'mae': np.random.uniform(0.01, 0.1, n_samples),
        'rmse': np.random.uniform(0.02, 0.15, n_samples),
        'training_time': np.random.uniform(10, 300, n_samples)
    })
    
    test_metrics = {
        'RandomForest': {'r2': 0.95, 'mae': 0.02, 'rmse': 0.03},
        'DNN': {'r2': 0.93, 'mae': 0.025, 'rmse': 0.035},
        'ANFIS': {'r2': 0.91, 'mae': 0.03, 'rmse': 0.04}
    }
    
    # Reports builder
    builder = ComprehensiveReportsBuilder('output/test_reports')
    
    # Tüm raporları oluştur
    reports = builder.generate_all_reports(
        test_results, test_metrics
    )
    
    logger.info("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()
