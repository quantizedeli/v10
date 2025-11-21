"""
Comprehensive Excel Reporter
Based on ooo.py's detailed multi-sheet Excel reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveExcelReporter:
    """
    Creates detailed Excel reports with multiple sheets
    Matches ooo.py's comprehensive reporting style
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(self,
                            results_df: pd.DataFrame,
                            replacements: List[Dict] = None,
                            config: Dict = None) -> Path:
        """
        Generate comprehensive Excel report with 10+ sheets
        
        Sheets:
        1. Özet (Summary)
        2. Tüm_Sonuçlar (All Results)
        3-5. Target-specific sheets (MM_Sonuçlar, QM_Sonuçlar, etc.)
        6-13. Config-specific sheets (CFG001_Detay, etc.)
        14. Hedef_İstatistikleri (Target Statistics)
        15. Config_İstatistikleri (Config Statistics)
        16. En_İyi_20 (Top 20)
        17. Çekirdek_Değişimleri (Nucleus Replacements)
        18. Kısaltmalar (Abbreviations)
        """
        logger.info("\n" + "="*60)
        logger.info("GENERATING COMPREHENSIVE EXCEL REPORT")
        logger.info("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = self.output_dir / f'ANFIS_Comprehensive_Report_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            formats = self._create_formats(workbook)
            
            # 1. ÖZET (Summary)
            logger.info("1/18: Creating Özet sheet...")
            self._create_summary_sheet(writer, results_df, replacements, formats)
            
            # 2. TÜM SONUÇLAR (All Results)
            logger.info("2/18: Creating Tüm_Sonuçlar sheet...")
            self._create_all_results_sheet(writer, results_df, formats)
            
            # 3-5. TARGET-SPECIFIC SHEETS
            logger.info("3-5/18: Creating target-specific sheets...")
            targets = results_df['target'].unique() if 'target' in results_df.columns else []
            for target in targets:
                self._create_target_sheet(writer, results_df, target, formats)
            
            # 6-13. CONFIG-SPECIFIC SHEETS
            logger.info("6-13/18: Creating config-specific sheets...")
            configs = results_df['config_id'].unique() if 'config_id' in results_df.columns else []
            for config_id in configs:
                self._create_config_sheet(writer, results_df, config_id, formats)
            
            # 14. HEDEF İSTATİSTİKLERİ (Target Statistics)
            logger.info("14/18: Creating Hedef_İstatistikleri sheet...")
            self._create_target_statistics(writer, results_df, formats)
            
            # 15. CONFIG İSTATİSTİKLERİ (Config Statistics)
            logger.info("15/18: Creating Config_İstatistikleri sheet...")
            self._create_config_statistics(writer, results_df, formats)
            
            # 16. EN İYİ 20 (Top 20)
            logger.info("16/18: Creating En_İyi_20 sheet...")
            self._create_top_20(writer, results_df, formats)
            
            # 17. ÇEKİRDEK DEĞİŞİMLERİ (Nucleus Replacements)
            logger.info("17/18: Creating Çekirdek_Değişimleri sheet...")
            if replacements:
                self._create_replacements_sheet(writer, replacements, formats)
            
            # 18. KISALTMALAR (Abbreviations)
            logger.info("18/18: Creating Kısaltmalar sheet...")
            self._create_abbreviations_sheet(writer, formats)
        
        logger.info(f"\n✅ Excel report saved: {excel_file.name}")
        logger.info("="*60)
        
        return excel_file
    
    def _create_formats(self, workbook):
        """Create Excel formats"""
        return {
            'header': workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'text_wrap': True
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
            'integer': workbook.add_format({
                'num_format': '0',
                'align': 'center'
            }),
            'good': workbook.add_format({
                'bg_color': '#C6EFCE',
                'font_color': '#006100'
            }),
            'bad': workbook.add_format({
                'bg_color': '#FFC7CE',
                'font_color': '#9C0006'
            })
        }
    
    def _create_summary_sheet(self, writer, df, replacements, formats):
        """Sheet 1: Özet"""
        summary_data = {
            'Metrik': [
                'Toplam Test Sayısı',
                'Başarılı Test (R²>0.5)',
                'Başarılı Test (R²>0.7)',
                'Başarılı Test (R²>0.85)',
                'Ortalama R²',
                'Maksimum R²',
                'Minimum R²',
                'Std R²',
                'Ortalama İyileştirme',
                'Toplam Eğitim Süresi (dakika)',
                'Ortalama Eğitim Süresi (saniye)',
                'Ortalama Kural Sayısı',
                'Çekirdek Değişimi Sayısı',
                'Farklı Dataset Sayısı',
                'Farklı Config Sayısı',
                'Farklı Target Sayısı'
            ],
            'Değer': [
                len(df),
                (df['final_r2'] > 0.5).sum() if 'final_r2' in df.columns else 0,
                (df['final_r2'] > 0.7).sum() if 'final_r2' in df.columns else 0,
                (df['final_r2'] > 0.85).sum() if 'final_r2' in df.columns else 0,
                df['final_r2'].mean() if 'final_r2' in df.columns else 0,
                df['final_r2'].max() if 'final_r2' in df.columns else 0,
                df['final_r2'].min() if 'final_r2' in df.columns else 0,
                df['final_r2'].std() if 'final_r2' in df.columns else 0,
                df['improvement'].mean() if 'improvement' in df.columns else 0,
                df['training_time'].sum() / 60 if 'training_time' in df.columns else 0,
                df['training_time'].mean() if 'training_time' in df.columns else 0,
                df['num_rules'].mean() if 'num_rules' in df.columns else 0,
                len(replacements) if replacements else 0,
                df['dataset_name'].nunique() if 'dataset_name' in df.columns else 0,
                df['config_id'].nunique() if 'config_id' in df.columns else 0,
                df['target'].nunique() if 'target' in df.columns else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Özet', index=False)
        
        # Format
        worksheet = writer.sheets['Özet']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 20, formats['number'])
    
    def _create_all_results_sheet(self, writer, df, formats):
        """Sheet 2: Tüm_Sonuçlar"""
        df.to_excel(writer, sheet_name='Tüm_Sonuçlar', index=False)
        
        worksheet = writer.sheets['Tüm_Sonuçlar']
        
        # Auto-fit columns
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.set_column(idx, idx, min(max_len, 50))
    
    def _create_target_sheet(self, writer, df, target, formats):
        """Sheets 3-5: Target-specific"""
        target_df = df[df['target'] == target] if 'target' in df.columns else pd.DataFrame()
        
        if not target_df.empty:
            sheet_name = f'{target}_Sonuçlar'
            target_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(target_df.columns):
                max_len = max(target_df[col].astype(str).apply(len).max(), len(col)) + 2
                worksheet.set_column(idx, idx, min(max_len, 50))
    
    def _create_config_sheet(self, writer, df, config_id, formats):
        """Sheets 6-13: Config-specific"""
        config_df = df[df['config_id'] == config_id] if 'config_id' in df.columns else pd.DataFrame()
        
        if not config_df.empty:
            sheet_name = f'{config_id}_Detay'
            config_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(config_df.columns):
                max_len = max(config_df[col].astype(str).apply(len).max(), len(col)) + 2
                worksheet.set_column(idx, idx, min(max_len, 50))
    
    def _create_target_statistics(self, writer, df, formats):
        """Sheet 14: Hedef_İstatistikleri"""
        if 'target' not in df.columns:
            return
        
        stats = df.groupby('target').agg({
            'final_r2': ['mean', 'std', 'max', 'min'],
            'final_rmse': ['mean', 'std'] if 'final_rmse' in df.columns else lambda x: 0,
            'improvement': ['mean', 'std'],
            'training_time': ['mean', 'sum'],
            'num_rules': ['mean', 'std'] if 'num_rules' in df.columns else lambda x: 0,
            'iterations': 'mean' if 'iterations' in df.columns else lambda x: 0
        }).round(4)
        
        stats.to_excel(writer, sheet_name='Hedef_İstatistikleri')
        
        worksheet = writer.sheets['Hedef_İstatistikleri']
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:Z', 12, formats['number'])
    
    def _create_config_statistics(self, writer, df, formats):
        """Sheet 15: Config_İstatistikleri"""
        if 'config_name' not in df.columns:
            return
        
        stats = df.groupby('config_name').agg({
            'final_r2': ['mean', 'std', 'max', 'min'],
            'training_time': 'mean',
            'num_rules': 'mean' if 'num_rules' in df.columns else lambda x: 0
        }).round(4)
        
        stats.to_excel(writer, sheet_name='Config_İstatistikleri')
        
        worksheet = writer.sheets['Config_İstatistikleri']
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:Z', 12, formats['number'])
    
    def _create_top_20(self, writer, df, formats):
        """Sheet 16: En_İyi_20"""
        if 'final_r2' in df.columns:
            top_20 = df.nlargest(20, 'final_r2')
            top_20.to_excel(writer, sheet_name='En_İyi_20', index=False)
            
            worksheet = writer.sheets['En_İyi_20']
            for idx, col in enumerate(top_20.columns):
                max_len = max(top_20[col].astype(str).apply(len).max(), len(col)) + 2
                worksheet.set_column(idx, idx, min(max_len, 50))
    
    def _create_replacements_sheet(self, writer, replacements, formats):
        """Sheet 17: Çekirdek_Değişimleri"""
        if replacements:
            repl_df = pd.DataFrame(replacements)
            repl_df.to_excel(writer, sheet_name='Çekirdek_Değişimleri', index=False)
            
            worksheet = writer.sheets['Çekirdek_Değişimleri']
            worksheet.set_column('A:E', 20)
    
    def _create_abbreviations_sheet(self, writer, formats):
        """Sheet 18: Kısaltmalar"""
        abbreviations = {
            'Kısaltma': [
                'MM', 'QM', 'ANFIS', 'BNN', 'PINN', 'FIS', 'MF', 
                'R²', 'RMSE', 'MAE', 'MAPE',
                'CFG', 'A', 'Z', 'N', 'S', 'P',
                'Grid', 'SubClust', 'Gauss', 'Bell', 'Tri', 'Trap'
            ],
            'Açıklama': [
                'Mass Excess (Kütle Fazlası) - MeV',
                'Q-value (Q Değeri) - MeV',
                'Adaptive Neuro-Fuzzy Inference System',
                'Bayesian Neural Network',
                'Physics-Informed Neural Network',
                'Fuzzy Inference System',
                'Membership Function (Üyelik Fonksiyonu)',
                'Coefficient of Determination (Belirleme Katsayısı)',
                'Root Mean Square Error (Kök Ortalama Kare Hata)',
                'Mean Absolute Error (Ortalama Mutlak Hata)',
                'Mean Absolute Percentage Error (Ortalama Mutlak Yüzde Hata)',
                'Configuration (Konfigürasyon)',
                'Mass Number (Kütle Sayısı)',
                'Atomic Number (Proton Sayısı)',
                'Neutron Number (Nötron Sayısı)',
                'Spin Pairing (Spin Eşleşmesi)',
                'Pairing Energy (Eşleşme Enerjisi)',
                'Grid Partition (Izgara Bölümleme)',
                'Subtractive Clustering (Çıkarmalı Kümeleme)',
                'Gaussian Membership Function',
                'Generalized Bell Membership Function',
                'Triangular Membership Function',
                'Trapezoidal Membership Function'
            ],
            'Kategori': [
                'Target', 'Target', 'Model', 'Model', 'Model', 'ANFIS', 'ANFIS',
                'Metric', 'Metric', 'Metric', 'Metric',
                'System', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature',
                'ANFIS Method', 'ANFIS Method', 'MF Type', 'MF Type', 'MF Type', 'MF Type'
            ]
        }
        
        abbr_df = pd.DataFrame(abbreviations)
        abbr_df.to_excel(writer, sheet_name='Kısaltmalar', index=False)
        
        worksheet = writer.sheets['Kısaltmalar']
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:B', 60)
        worksheet.set_column('C:C', 20)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    n_samples = 100
    test_data = {
        'dataset_name': [f'Dataset_{i}' for i in range(n_samples)],
        'dataset_id': [f'DS{i:04d}' for i in range(n_samples)],
        'target': np.random.choice(['MM', 'QM', 'MM-QM'], n_samples),
        'config_id': np.random.choice(['CFG001', 'CFG002', 'CFG003'], n_samples),
        'config_name': np.random.choice(['Grid_2MF_Gauss', 'Grid_2MF_Bell', 'SubClust_R05'], n_samples),
        'ai_r2': np.random.uniform(0.7, 0.95, n_samples),
        'final_r2': np.random.uniform(0.75, 0.98, n_samples),
        'final_rmse': np.random.uniform(0.01, 0.1, n_samples),
        'improvement': np.random.uniform(-0.05, 0.15, n_samples),
        'training_time': np.random.uniform(30, 300, n_samples),
        'num_rules': np.random.randint(10, 100, n_samples),
        'iterations': np.random.randint(1, 5, n_samples)
    }
    
    df = pd.DataFrame(test_data)
    
    # Test replacements
    replacements = [
        {'dataset': 'DS0001', 'config': 'CFG001', 'iteration': 1,
         'original': 'Fe56', 'replacement': 'Fe58'},
        {'dataset': 'DS0002', 'config': 'CFG001', 'iteration': 2,
         'original': 'O16', 'replacement': 'O18'}
    ]
    
    # Create report
    reporter = ComprehensiveExcelReporter('test_output')
    excel_file = reporter.generate_full_report(df, replacements)
    
    print(f"\n✅ Test Excel report created: {excel_file}")
