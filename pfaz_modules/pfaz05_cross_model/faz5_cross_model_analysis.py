"""
FAZ 5: Cross-Model Analysis Pipeline
Tüm Modellerin Ortak Performans Analizi

Bu script:
1. Tüm eğitilmiş modellerin tahminlerini toplar (AI + ANFIS)
2. Her target (MM, QM, MM_QM, Beta_2) için cross-model analizi yapar
3. Kapsamlı Excel raporu oluşturur (TEZ İÇİN HAZIR)
4. Görselleştirmeler üretir

Yazar: Nükleer Fizik AI Projesi
Tarih: 15 Ekim 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple
import sys

# Proje modüllerini import et
try:
    from .cross_model_evaluator import CrossModelEvaluator
except ImportError:
    print("[ERROR] cross_model_evaluator.py bulunamadı!")
    print("Lütfen cross_model_evaluator.py'yi aynı klasöre kopyalayın.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossModelAnalysisPipeline:
    """
    Cross-Model Analysis Pipeline - FAZ 5
    
    Tüm modellerin tahminlerini toplar ve analiz eder
    """
    
    def __init__(self, 
                 trained_models_dir='trained_models',
                 output_dir='reports/cross_model_analysis'):
        """
        Args:
            trained_models_dir: Eğitilmiş modellerin dizini
            output_dir: Çıktı dizini
        """
        self.trained_models_dir = Path(trained_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model kategorileri
        self.ai_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'DNN', 'BNN', 'PINN']
        self.anfis_configs = ['GAU2MF', 'GEN2MF', 'TRI2MF', 'TRA2MF', 'GAU3MF', 
                             'SUBR03', 'SUBR05', 'SUBR07']
        
        # Targetler
        self.targets = ['MM', 'QM', 'MM_QM', 'Beta_2']
        
        # Predictions storage
        self.all_predictions = {}  # {target: {model_name: df}}
        
        logger.info("Cross-Model Analysis Pipeline initialized")
    
    def run_complete_analysis(self):
        """
        Tam cross-model analizi çalıştır
        
        Returns:
            dict: Tüm sonuçlar
        """
        logger.info("\n" + "="*80)
        logger.info("FAZ 5: CROSS-MODEL ANALYSIS BAŞLIYOR")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Tüm model tahminlerini topla
        logger.info("\n1. Model tahminleri toplanıyor...")
        self._collect_all_predictions()
        
        # 2. Her target için analiz yap
        logger.info("\n2. Target-bazlı analizler yapılıyor...")
        results = {}
        
        for target in self.targets:
            logger.info(f"\n{'='*80}")
            logger.info(f"TARGET: {target}")
            logger.info(f"{'='*80}")
            
            if target not in self.all_predictions or len(self.all_predictions[target]) == 0:
                logger.warning(f"[WARNING] {target} için tahmin bulunamadı, atlanıyor...")
                continue
            
            # Cross-model evaluator oluştur
            evaluator = CrossModelEvaluator(self.output_dir / target)
            
            # Model tahminlerini ekle
            for model_name, df in self.all_predictions[target].items():
                evaluator.add_predictions(
                    model_name, df,
                    target_col='experimental',
                    prediction_col='predicted',
                    nucleus_col='nucleus'
                )
            
            # Analiz yap
            target_results = evaluator.evaluate_common_performance(
                target_name=target,
                top_n=50  # 50 çekirdek
            )
            
            # Görselleştir
            evaluator.visualize_results(target)
            
            # Kaydet
            evaluator.save_cross_model_report(f'{target}_cross_model_report.xlsx')
            
            results[target] = target_results
        
        # 3. Birleştirilmiş master rapor oluştur
        logger.info("\n3. Master rapor oluşturuluyor...")
        self._create_master_report(results)
        
        # 4. Özet rapor
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("FAZ 5: CROSS-MODEL ANALYSIS TAMAMLANDI")
        logger.info("="*80)
        logger.info(f"Süre: {duration:.2f} saniye ({duration/60:.1f} dakika)")
        logger.info(f"Analiz edilen target sayısı: {len(results)}")
        logger.info(f"Toplam model sayısı: {sum(len(self.all_predictions.get(t, {})) for t in self.targets)}")
        
        # JSON özet kaydet
        self._save_summary_json(results, duration)
        
        return results
    
    def _collect_all_predictions(self):
        """
        Tüm model tahminlerini topla
        
        Beklenen dosya yapısı:
        trained_models/
          AI/
            RandomForest/
              MM_predictions.csv
              QM_predictions.csv
              ...
          ANFIS/
            GAU2MF/
              MM_predictions.csv
              ...
        """
        # AI modelleri
        ai_dir = self.trained_models_dir / 'AI'
        if ai_dir.exists():
            for model_name in self.ai_models:
                model_dir = ai_dir / model_name
                if not model_dir.exists():
                    logger.debug(f"  {model_name} dizini bulunamadı, atlanıyor...")
                    continue
                
                self._load_model_predictions(model_name, model_dir, 'AI')
        else:
            logger.warning(f"[WARNING] AI modelleri dizini bulunamadı: {ai_dir}")
        
        # ANFIS modelleri
        anfis_dir = self.trained_models_dir / 'ANFIS'
        if anfis_dir.exists():
            for config_name in self.anfis_configs:
                config_dir = anfis_dir / config_name
                if not config_dir.exists():
                    logger.debug(f"  ANFIS {config_name} dizini bulunamadı, atlanıyor...")
                    continue
                
                self._load_model_predictions(f'ANFIS_{config_name}', config_dir, 'ANFIS')
        else:
            logger.warning(f"[WARNING] ANFIS modelleri dizini bulunamadı: {anfis_dir}")
        
        # Özet
        logger.info(f"\n  [OK] Tahmin toplama özeti:")
        for target in self.targets:
            n_models = len(self.all_predictions.get(target, {}))
            logger.info(f"    {target}: {n_models} model")
    
    def _load_model_predictions(self, model_name: str, model_dir: Path, model_type: str):
        """
        Bir modelin tüm target tahminlerini yükle
        
        Args:
            model_name: Model ismi
            model_dir: Model dizini
            model_type: 'AI' veya 'ANFIS'
        """
        for target in self.targets:
            # Olası dosya isimleri
            possible_files = [
                model_dir / f'{target}_predictions.csv',
                model_dir / f'{target.lower()}_predictions.csv',
                model_dir / f'predictions_{target}.csv',
                model_dir / f'predictions_{target.lower()}.csv',
                model_dir / 'predictions.csv',  # Genel dosya
            ]
            
            prediction_file = None
            for file in possible_files:
                if file.exists():
                    prediction_file = file
                    break
            
            if prediction_file is None:
                logger.debug(f"    {model_name} - {target}: Tahmin dosyası bulunamadı")
                continue
            
            try:
                # Tahminleri yükle
                df = pd.read_csv(prediction_file)
                
                # Gerekli kolonları kontrol et
                required_cols = ['nucleus', 'experimental', 'predicted']
                if not all(col in df.columns for col in required_cols):
                    # Alternatif kolon isimleri dene
                    col_mapping = {
                        'Nucleus': 'nucleus',
                        'NUCLEUS': 'nucleus',
                        'Experimental': 'experimental',
                        'Experimental_Value': 'experimental',
                        'target': 'experimental',
                        'True': 'experimental',
                        'Predicted': 'predicted',
                        'Prediction': 'predicted',
                        'pred': 'predicted'
                    }
                    
                    df = df.rename(columns=col_mapping)
                    
                    # Hala eksikse atla
                    if not all(col in df.columns for col in required_cols):
                        logger.warning(f"    {model_name} - {target}: Gerekli kolonlar bulunamadı")
                        continue
                
                # Target için storage oluştur
                if target not in self.all_predictions:
                    self.all_predictions[target] = {}
                
                # Kaydet
                self.all_predictions[target][model_name] = df[required_cols].copy()
                
                logger.info(f"    [OK] {model_name} - {target}: {len(df)} tahmin yüklendi")
            
            except Exception as e:
                logger.error(f"    [FAIL] {model_name} - {target}: Yükleme hatası: {e}")
    
    def _create_master_report(self, results: Dict):
        """
        Tüm targetleri içeren master Excel raporu oluştur
        
        Excel Yapısı:
        - Sheet 1: Overall_Summary (Tüm targetler özet)
        - Sheet 2: MM_Good (50 çekirdek)
        - Sheet 3: MM_Medium (50 çekirdek)
        - Sheet 4: MM_Poor (50 çekirdek)
        - Sheet 5: QM_Good
        - Sheet 6: QM_Medium
        - Sheet 7: QM_Poor
        - Sheet 8: MM_QM_Good
        - Sheet 9: MM_QM_Medium
        - Sheet 10: MM_QM_Poor
        - Sheet 11: Beta_2_Good
        - Sheet 12: Beta_2_Medium
        - Sheet 13: Beta_2_Poor
        - Sheet 14: Model_Agreement_All (Tüm targetler)
        - Sheet 15: Detailed_Statistics_All
        """
        master_file = self.output_dir / 'MASTER_CROSS_MODEL_REPORT.xlsx'
        
        logger.info(f"  Master rapor oluşturuluyor: {master_file}")
        
        with pd.ExcelWriter(master_file, engine='openpyxl') as writer:
            # Sheet 1: Overall Summary
            self._write_overall_summary(writer, results)
            
            # Her target için detaylı sheet'ler
            for target, target_results in results.items():
                self._write_target_sheets(writer, target, target_results)
            
            # Aggregated sheets
            self._write_model_agreement_all(writer, results)
            self._write_detailed_statistics_all(writer)
        
        logger.info(f"  [OK] Master rapor kaydedildi: {master_file}")
    
    def _write_overall_summary(self, writer, results):
        """Overall summary sheet - Tüm targetler"""
        summary_data = []
        
        for target, res in results.items():
            if not res:
                continue
            
            summary_data.append({
                'Target': target,
                'N_Models': len(self.all_predictions.get(target, {})),
                'Good_Count': len(res.get('good_nuclei', [])),
                'Medium_Count': len(res.get('medium_nuclei', [])),
                'Poor_Count': len(res.get('poor_nuclei', [])),
                'Good_Mean_Error': res.get('good_stats', {}).get('mean_error', np.nan),
                'Good_Mean_R2': res.get('good_stats', {}).get('mean_r2', np.nan),
                'Medium_Mean_Error': res.get('medium_stats', {}).get('mean_error', np.nan),
                'Medium_Mean_R2': res.get('medium_stats', {}).get('mean_r2', np.nan),
                'Poor_Mean_Error': res.get('poor_stats', {}).get('mean_error', np.nan),
                'Poor_Mean_R2': res.get('poor_stats', {}).get('mean_r2', np.nan),
                'Overall_Agreement': res.get('model_agreement', {}).get('overall_agreement', np.nan)
            })
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Overall_Summary', index=False)
        
        logger.info(f"    [OK] Overall_Summary sheet yazıldı")
    
    def _write_target_sheets(self, writer, target, results):
        """Her target için Good/Medium/Poor sheet'ler"""
        categories = ['good', 'medium', 'poor']
        
        for category in categories:
            nuclei_list = results.get(f'{category}_nuclei', [])
            
            if len(nuclei_list) == 0:
                continue
            
            # Detaylı data hazırla
            detailed_data = []
            
            for nucleus in nuclei_list:
                nucleus_row = {'Nucleus': nucleus}
                
                # Her modelden tahmin ve hata ekle
                for model_name, df in self.all_predictions[target].items():
                    nucleus_data = df[df['nucleus'] == nucleus]
                    
                    if len(nucleus_data) > 0:
                        exp_val = nucleus_data['experimental'].iloc[0]
                        pred_val = nucleus_data['predicted'].iloc[0]
                        error = abs(exp_val - pred_val)
                        delta = pred_val - exp_val
                        
                        nucleus_row[f'{model_name}_Pred'] = pred_val
                        nucleus_row[f'{model_name}_Error'] = error
                        nucleus_row[f'{model_name}_Delta'] = delta
                
                detailed_data.append(nucleus_row)
            
            df_detailed = pd.DataFrame(detailed_data)
            
            # İstatistik satırı ekle
            stats_row = {'Nucleus': 'MEAN'}
            for col in df_detailed.columns:
                if col != 'Nucleus' and df_detailed[col].dtype in [np.float64, np.int64]:
                    stats_row[col] = df_detailed[col].mean()
            
            df_detailed = pd.concat([df_detailed, pd.DataFrame([stats_row])], ignore_index=True)
            
            # Sheet'e yaz
            sheet_name = f'{target}_{category.capitalize()}'[:31]
            df_detailed.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"    [OK] {sheet_name} sheet yazıldı ({len(nuclei_list)} çekirdek)")
    
    def _write_model_agreement_all(self, writer, results):
        """Tüm targetler için model agreement"""
        agreement_data = []
        
        for target, res in results.items():
            agreement = res.get('model_agreement', {})
            
            agreement_data.append({
                'Target': target,
                'Overall_Agreement': agreement.get('overall_agreement', np.nan),
                'Good_Nuclei_Agreement': agreement.get('good_nuclei_agreement', np.nan),
                'Medium_Nuclei_Agreement': agreement.get('medium_nuclei_agreement', np.nan),
                'Poor_Nuclei_Agreement': agreement.get('poor_nuclei_agreement', np.nan)
            })
        
        df = pd.DataFrame(agreement_data)
        df.to_excel(writer, sheet_name='Model_Agreement_All', index=False)
        
        logger.info(f"    [OK] Model_Agreement_All sheet yazıldı")
    
    def _write_detailed_statistics_all(self, writer):
        """Tüm modeller için detaylı istatistikler"""
        stats_data = []
        
        for target in self.targets:
            if target not in self.all_predictions:
                continue
            
            for model_name, df in self.all_predictions[target].items():
                error = abs(df['experimental'] - df['predicted'])
                
                stats_data.append({
                    'Target': target,
                    'Model': model_name,
                    'N_Predictions': len(df),
                    'Mean_Error': error.mean(),
                    'Std_Error': error.std(),
                    'Min_Error': error.min(),
                    'Max_Error': error.max(),
                    'Median_Error': error.median(),
                    'Q25_Error': error.quantile(0.25),
                    'Q75_Error': error.quantile(0.75)
                })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Detailed_Statistics_All', index=False)
        
        logger.info(f"    [OK] Detailed_Statistics_All sheet yazıldı")
    
    def _save_summary_json(self, results, duration):
        """Özet JSON kaydet"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'targets_analyzed': list(results.keys()),
            'total_models': sum(len(self.all_predictions.get(t, {})) for t in self.targets),
            'results_summary': {}
        }
        
        for target, res in results.items():
            summary['results_summary'][target] = {
                'n_models': len(self.all_predictions.get(target, {})),
                'good_count': len(res.get('good_nuclei', [])),
                'medium_count': len(res.get('medium_nuclei', [])),
                'poor_count': len(res.get('poor_nuclei', [])),
                'overall_agreement': res.get('model_agreement', {}).get('overall_agreement', 0)
            }
        
        json_file = self.output_dir / 'cross_model_analysis_summary.json'
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"  [OK] Özet JSON kaydedildi: {json_file}")


def main():
    """
    FAZ 5 - Cross-Model Analysis Pipeline
    
    Usage:
        python faz5_cross_model_analysis.py
    """
    logger.info("="*80)
    logger.info("FAZ 5: CROSS-MODEL ANALYSIS PIPELINE")
    logger.info("="*80)
    
    # Pipeline oluştur
    pipeline = CrossModelAnalysisPipeline(
        trained_models_dir='trained_models',
        output_dir='reports/cross_model_analysis'
    )
    
    # Tam analiz çalıştır
    results = pipeline.run_complete_analysis()
    
    # Başarı mesajı
    logger.info("\n" + "="*80)
    logger.info("[SUCCESS] FAZ 5 BAŞARIYLA TAMAMLANDI!")
    logger.info("="*80)
    logger.info("\nOluşturulan raporlar:")
    logger.info("  1. MASTER_CROSS_MODEL_REPORT.xlsx (Tüm targetler)")
    logger.info("  2. MM/MM_cross_model_report.xlsx")
    logger.info("  3. QM/QM_cross_model_report.xlsx")
    logger.info("  4. MM_QM/MM_QM_cross_model_report.xlsx")
    logger.info("  5. Beta_2/Beta_2_cross_model_report.xlsx")
    logger.info("  6. cross_model_analysis_summary.json")
    logger.info("\nGörselleştirmeler:")
    logger.info("  - MM/cross_model_visualization_MM.png")
    logger.info("  - QM/cross_model_visualization_QM.png")
    logger.info("  - MM_QM/cross_model_visualization_MM_QM.png")
    logger.info("  - Beta_2/cross_model_visualization_Beta_2.png")
    
    return results


if __name__ == "__main__":
    main()
