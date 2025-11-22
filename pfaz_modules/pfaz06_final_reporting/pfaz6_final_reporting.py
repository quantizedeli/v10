"""
PFAZ 6: Final Reporting & Thesis Integration
Tüm sonuçların toplanması ve thesis-ready raporlar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FinalReportingPipeline:
    """Thesis için final raporlama sistemi"""
    
    def __init__(self, reports_dir='reports', output_dir='final_reports'):
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
        
    def collect_all_results(self):
        """Tüm fazlardan sonuçları topla"""
        logger.info("\n" + "="*80)
        logger.info("TÜM SONUÇLAR TOPLANIYOR")
        logger.info("="*80)
        
        # PFAZ 2: AI models
        logger.info("\n1. AI Model Sonuçları...")
        self._collect_ai_results()
        
        # PFAZ 3: ANFIS
        logger.info("2. ANFIS Sonuçları...")
        self._collect_anfis_results()
        
        # PFAZ 4: Individual analysis
        logger.info("3. Individual Analysis...")
        self._collect_individual_analysis()
        
        # PFAZ 5: Cross-model
        logger.info("4. Cross-Model Analiz...")
        self._collect_crossmodel_results()
        
        logger.info(f"\n✓ Toplanan sonuç sayısı: {self._count_total_results()}")
        return self.all_results
    
    def _count_total_results(self):
        """Toplam sonuç sayısı"""
        count = 0
        count += len(self.all_results.get('ai_models', {}))
        count += len(self.all_results.get('anfis_models', {}))
        return count
    
    def _collect_ai_results(self):
        """AI model sonuçlarını topla"""
        ai_dir = Path('trained_models/AI')
        self.all_results['ai_models'] = {}
        
        if ai_dir.exists():
            for model_dir in ai_dir.iterdir():
                if model_dir.is_dir():
                    # Her target için sonuçları topla
                    model_name = model_dir.name
                    self.all_results['ai_models'][model_name] = {}
                    
                    for target in ['MM', 'QM', 'MM_QM', 'Beta_2']:
                        metrics_file = model_dir / f'{target}_metrics.json'
                        if metrics_file.exists():
                            with open(metrics_file) as f:
                                self.all_results['ai_models'][model_name][target] = json.load(f)
        
        logger.info(f"  ✓ {len(self.all_results['ai_models'])} AI model")
    
    def _collect_anfis_results(self):
        """ANFIS sonuçlarını topla"""
        anfis_dir = Path('trained_models/ANFIS')
        self.all_results['anfis_models'] = {}
        
        if anfis_dir.exists():
            for config_dir in anfis_dir.iterdir():
                if config_dir.is_dir():
                    config_name = config_dir.name
                    self.all_results['anfis_models'][config_name] = {}
                    
                    for target in ['MM', 'QM', 'MM_QM', 'Beta_2']:
                        metrics_file = config_dir / f'{target}_metrics.json'
                        if metrics_file.exists():
                            with open(metrics_file) as f:
                                self.all_results['anfis_models'][config_name][target] = json.load(f)
        
        logger.info(f"  ✓ {len(self.all_results['anfis_models'])} ANFIS config")
    
    def _collect_individual_analysis(self):
        """Individual analysis sonuçları"""
        ind_file = self.reports_dir / 'individual_analysis' / 'analysis_summary.json'
        if ind_file.exists():
            with open(ind_file) as f:
                self.all_results['individual_analysis'] = json.load(f)
            logger.info(f"  ✓ Individual analysis yüklendi")
    
    def _collect_crossmodel_results(self):
        """Cross-model analiz sonuçları"""
        json_file = self.reports_dir / 'cross_model_analysis' / 'cross_model_analysis_summary.json'
        if json_file.exists():
            with open(json_file) as f:
                self.all_results['cross_model'] = json.load(f)
            logger.info(f"  ✓ Cross-model analiz yüklendi")
    
    def generate_thesis_tables(self):
        """Tez için KAPSAMLI Excel tabloları"""
        logger.info("\n" + "="*80)
        logger.info("TEZ TABLOLARI OLUŞTURULUYOR")
        logger.info("="*80)
        
        excel_file = self.output_dir / 'THESIS_COMPLETE_RESULTS.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Tablo 1: TÜM AI modellerin TÜM targetlerde performansı
            self._write_all_ai_models(writer)
            
            # Tablo 2: TÜM ANFIS konfigürasyonların TÜM targetlerde performansı
            self._write_all_anfis_models(writer)
            
            # Tablo 3: AI vs ANFIS karşılaştırması (target bazında)
            self._write_ai_anfis_comparison(writer)
            
            # Tablo 4: Target bazında TÜM modeller
            for target in ['MM', 'QM', 'MM_QM', 'Beta_2']:
                self._write_target_specific(writer, target)
            
            # Tablo 5: Cross-model sonuçları
            self._write_crossmodel_details(writer)
            
            # Tablo 6: Genel istatistikler
            self._write_overall_statistics(writer)
        
        logger.info(f"\n✓ Excel: {excel_file}")
        logger.info(f"  Toplam sheet: {6 + 4}")  # 6 genel + 4 target-specific
        return excel_file
    
    def _write_all_ai_models(self, writer):
        """TÜM AI modellerin TÜM targetlerde performansı"""
        data = []
        
        for model_name, targets in self.all_results.get('ai_models', {}).items():
            for target, metrics in targets.items():
                data.append({
                    'Model': model_name,
                    'Target': target,
                    'R²': metrics.get('R2', metrics.get('r2', np.nan)),
                    'RMSE': metrics.get('RMSE', metrics.get('rmse', np.nan)),
                    'MAE': metrics.get('MAE', metrics.get('mae', np.nan)),
                    'Training_Time': metrics.get('training_time', np.nan)
                })
        
        if data:
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name='All_AI_Models', index=False)
            logger.info(f"  ✓ All_AI_Models ({len(data)} sonuç)")
    
    def _write_all_anfis_models(self, writer):
        """TÜM ANFIS konfigürasyonların TÜM targetlerde performansı"""
        data = []
        
        for config_name, targets in self.all_results.get('anfis_models', {}).items():
            for target, metrics in targets.items():
                data.append({
                    'Config': config_name,
                    'Target': target,
                    'R²': metrics.get('R2', metrics.get('r2', np.nan)),
                    'RMSE': metrics.get('RMSE', metrics.get('rmse', np.nan)),
                    'MAE': metrics.get('MAE', metrics.get('mae', np.nan)),
                    'Training_Time': metrics.get('training_time', np.nan)
                })
        
        if data:
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name='All_ANFIS_Models', index=False)
            logger.info(f"  ✓ All_ANFIS_Models ({len(data)} sonuç)")
    
    def _write_ai_anfis_comparison(self, writer):
        """AI vs ANFIS karşılaştırması (target bazında)"""
        data = []
        
        for target in ['MM', 'QM', 'MM_QM', 'Beta_2']:
            # AI models için ortalama
            ai_r2_list = []
            ai_rmse_list = []
            for model_name, targets in self.all_results.get('ai_models', {}).items():
                if target in targets:
                    ai_r2_list.append(targets[target].get('R2', targets[target].get('r2', np.nan)))
                    ai_rmse_list.append(targets[target].get('RMSE', targets[target].get('rmse', np.nan)))
            
            # ANFIS models için ortalama
            anfis_r2_list = []
            anfis_rmse_list = []
            for config_name, targets in self.all_results.get('anfis_models', {}).items():
                if target in targets:
                    anfis_r2_list.append(targets[target].get('R2', targets[target].get('r2', np.nan)))
                    anfis_rmse_list.append(targets[target].get('RMSE', targets[target].get('rmse', np.nan)))
            
            data.append({
                'Target': target,
                'AI_Count': len(ai_r2_list),
                'AI_Mean_R²': np.nanmean(ai_r2_list) if ai_r2_list else np.nan,
                'AI_Mean_RMSE': np.nanmean(ai_rmse_list) if ai_rmse_list else np.nan,
                'ANFIS_Count': len(anfis_r2_list),
                'ANFIS_Mean_R²': np.nanmean(anfis_r2_list) if anfis_r2_list else np.nan,
                'ANFIS_Mean_RMSE': np.nanmean(anfis_rmse_list) if anfis_rmse_list else np.nan
            })
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='AI_vs_ANFIS_Comparison', index=False)
        logger.info(f"  ✓ AI_vs_ANFIS_Comparison")
    
    def _write_target_specific(self, writer, target):
        """Belirli bir target için TÜM modeller"""
        data = []
        
        # AI models
        for model_name, targets in self.all_results.get('ai_models', {}).items():
            if target in targets:
                metrics = targets[target]
                data.append({
                    'Model': model_name,
                    'Type': 'AI',
                    'R²': metrics.get('R2', metrics.get('r2', np.nan)),
                    'RMSE': metrics.get('RMSE', metrics.get('rmse', np.nan)),
                    'MAE': metrics.get('MAE', metrics.get('mae', np.nan))
                })
        
        # ANFIS models
        for config_name, targets in self.all_results.get('anfis_models', {}).items():
            if target in targets:
                metrics = targets[target]
                data.append({
                    'Model': config_name,
                    'Type': 'ANFIS',
                    'R²': metrics.get('R2', metrics.get('r2', np.nan)),
                    'RMSE': metrics.get('RMSE', metrics.get('rmse', np.nan)),
                    'MAE': metrics.get('MAE', metrics.get('mae', np.nan))
                })
        
        if data:
            df = pd.DataFrame(data)
            df = df.sort_values('R²', ascending=False)
            df.to_excel(writer, sheet_name=f'{target}_All_Models', index=False)
            logger.info(f"  ✓ {target}_All_Models ({len(data)} model)")
    
    def _write_crossmodel_details(self, writer):
        """Cross-model detaylı sonuçlar"""
        cm_data = self.all_results.get('cross_model', {}).get('results_summary', {})
        
        data = []
        for target, info in cm_data.items():
            data.append({
                'Target': target,
                'N_Models': info.get('n_models', 0),
                'Good_Nuclei': info.get('good_count', 0),
                'Medium_Nuclei': info.get('medium_count', 0),
                'Poor_Nuclei': info.get('poor_count', 0),
                'Overall_Agreement': info.get('overall_agreement', 0)
            })
        
        if data:
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name='CrossModel_Summary', index=False)
            logger.info(f"  ✓ CrossModel_Summary")
    
    def _write_overall_statistics(self, writer):
        """Genel istatistikler"""
        data = [{
            'Metric': 'Total_AI_Models',
            'Value': len(self.all_results.get('ai_models', {}))
        }, {
            'Metric': 'Total_ANFIS_Configs',
            'Value': len(self.all_results.get('anfis_models', {}))
        }, {
            'Metric': 'Total_Targets',
            'Value': 4
        }, {
            'Metric': 'Total_Results',
            'Value': self._count_total_results() * 4
        }]
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name='Overall_Statistics', index=False)
        logger.info(f"  ✓ Overall_Statistics")
    
    def generate_summary_json(self):
        """JSON özet dosyası"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_ai_models': len(self.all_results.get('ai_models', {})),
            'total_anfis_configs': len(self.all_results.get('anfis_models', {})),
            'targets': ['MM', 'QM', 'MM_QM', 'Beta_2'],
            'phases_completed': ['PFAZ1', 'PFAZ2', 'PFAZ3', 'PFAZ4', 'PFAZ5', 'PFAZ6']
        }
        
        json_file = self.output_dir / 'final_summary.json'
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ JSON: {json_file}")
        return json_file
    
    def run_complete_pipeline(self):
        """Tam pipeline çalıştır"""
        start = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info("PFAZ 6: FINAL REPORTING & THESIS")
        logger.info("="*80)
        
        # 1. Sonuçları topla
        self.collect_all_results()
        
        # 2. Excel tabloları
        excel_file = self.generate_thesis_tables()
        
        # 3. JSON özet
        json_file = self.generate_summary_json()
        
        # 4. Summary
        duration = (datetime.now() - start).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("[SUCCESS] PFAZ 6 TAMAMLANDI")
        logger.info("="*80)
        logger.info(f"Süre: {duration:.1f} saniye")
        logger.info(f"Excel: {excel_file}")
        logger.info(f"JSON: {json_file}")
        logger.info(f"\nNot: Tüm modellerin tüm sonuçları Excel'de mevcut.")
        logger.info(f"     En iyi model seçimi TEZ yazarı tarafından yapılmalı.")
        
        return {
            'results': self.all_results,
            'excel_file': excel_file,
            'json_file': json_file,
            'duration': duration
        }


def main():
    pipeline = FinalReportingPipeline()
    results = pipeline.run_complete_pipeline()
    return results


if __name__ == "__main__":
    main()
