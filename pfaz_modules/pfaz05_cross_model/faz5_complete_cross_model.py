"""
FAZ 5: Complete Cross-Model Analysis - %100 TAMAMLANMIŞ
=========================================================

Tüm modellerin (AI + ANFIS) ortak performans analizi
Her target için: Good/Medium/Poor çekirdek sınıflandırması

Author: Nuclear Physics AI Project
Date: 2025-10-17
Version: 1.0 COMPLETE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CrossModelEvaluator:
    """Çok-model ortak performans değerlendirici"""
    
    def __init__(self, output_dir='reports/cross_model'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions = {}  # {model_name: df}
        self.results = {}
        
        # Sınıflandırma eşikleri
        self.thresholds = {
            'good': {'error': 0.1, 'r2': 0.90},
            'medium': {'error': 0.5, 'r2': 0.70},
            'poor': {'error': 0.5, 'r2': 0.70}
        }
        
    def add_predictions(self, model_name: str, df: pd.DataFrame,
                       target_col='target', prediction_col='prediction',
                       nucleus_col='nucleus'):
        """Model tahminlerini ekle"""
        required = [nucleus_col, target_col, prediction_col]
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must have columns: {required}")
        
        # Standart sütun isimleri
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={
            nucleus_col: 'nucleus',
            target_col: 'target',
            prediction_col: 'prediction'
        })
        
        # Hata hesapla
        df_copy['error'] = abs(df_copy['target'] - df_copy['prediction'])
        
        # R² hesapla (nucleus bazında)
        df_copy['r2'] = 1 - (df_copy['error']**2) / (df_copy['target'].std()**2 + 1e-10)
        
        self.predictions[model_name] = df_copy
        logger.info(f"[OK] {model_name}: {len(df_copy)} çekirdek eklendi")
    
    def evaluate_common_performance(self, target_name='MM', top_n=50):
        """Ortak performans analizi"""
        if len(self.predictions) == 0:
            raise ValueError("No predictions added. Use add_predictions() first.")
        
        # Ortak çekirdekler
        common_nuclei = self._get_common_nuclei()
        logger.info(f"Ortak çekirdek sayısı: {len(common_nuclei)}")
        
        # Aggregate performans
        nucleus_perf = self._calculate_aggregate_performance(common_nuclei, target_name)
        
        # Sınıflandırma
        classification = self._classify_nuclei(nucleus_perf)
        
        # Top N seç
        results = self._select_top_nuclei(classification, top_n)
        
        # Model uyumu
        results['model_agreement'] = self._analyze_model_agreement(results, nucleus_perf)
        
        self.results[target_name] = results
        
        # Özet
        logger.info(f"\n{'='*80}")
        logger.info(f"SONUÇLAR - {target_name}")
        logger.info(f"{'='*80}")
        logger.info(f"Good Nuclei: {len(results['good_nuclei'])}")
        logger.info(f"Medium Nuclei: {len(results['medium_nuclei'])}")
        logger.info(f"Poor Nuclei: {len(results['poor_nuclei'])}")
        logger.info(f"Model Agreement: {results['model_agreement']['overall_agreement']:.3f}")
        
        return results
    
    def _get_common_nuclei(self) -> List[str]:
        """Tüm modellerde ortak olan çekirdekler"""
        nucleus_sets = [set(df['nucleus'].unique()) for df in self.predictions.values()]
        common = set.intersection(*nucleus_sets)
        return sorted(list(common))
    
    def _calculate_aggregate_performance(self, common_nuclei, target_name):
        """Her çekirdek için toplam performans"""
        perf_data = []
        
        for nucleus in common_nuclei:
            errors = []
            r2_scores = []
            predictions = []
            targets = []
            
            for model_name, df in self.predictions.items():
                nuc_data = df[df['nucleus'] == nucleus]
                if len(nuc_data) > 0:
                    errors.append(nuc_data['error'].iloc[0])
                    r2_scores.append(nuc_data['r2'].iloc[0])
                    predictions.append(nuc_data['prediction'].iloc[0])
                    targets.append(nuc_data['target'].iloc[0])
            
            perf_data.append({
                'nucleus': nucleus,
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'mean_prediction': np.mean(predictions),
                'target': np.mean(targets),
                'n_models': len(errors)
            })
        
        return pd.DataFrame(perf_data)
    
    def _classify_nuclei(self, nucleus_perf):
        """Çekirdekleri sınıflandır"""
        good_mask = (nucleus_perf['mean_error'] < self.thresholds['good']['error']) & \
                    (nucleus_perf['mean_r2'] > self.thresholds['good']['r2'])
        
        poor_mask = (nucleus_perf['mean_error'] >= self.thresholds['poor']['error']) | \
                    (nucleus_perf['mean_r2'] < self.thresholds['poor']['r2'])
        
        medium_mask = ~(good_mask | poor_mask)
        
        return {
            'good': nucleus_perf[good_mask],
            'medium': nucleus_perf[medium_mask],
            'poor': nucleus_perf[poor_mask]
        }
    
    def _select_top_nuclei(self, classification, top_n):
        """Her kategoriden top N seç"""
        results = {}
        
        for category in ['good', 'medium', 'poor']:
            df = classification[category].copy()
            
            if category == 'good':
                # En düşük hata
                df = df.nsmallest(min(top_n, len(df)), 'mean_error')
            elif category == 'medium':
                # Orta hatalara yakın
                median_error = df['mean_error'].median()
                df['distance_to_median'] = abs(df['mean_error'] - median_error)
                df = df.nsmallest(min(top_n, len(df)), 'distance_to_median')
            else:  # poor
                # En yüksek hata
                df = df.nlargest(min(top_n, len(df)), 'mean_error')
            
            results[f'{category}_nuclei'] = df['nucleus'].tolist()
            results[f'{category}_stats'] = {
                'mean_error': df['mean_error'].mean(),
                'std_error': df['mean_error'].std(),
                'mean_r2': df['mean_r2'].mean(),
                'count': len(df),
                'mean_agreement': 1 - df['std_error'].mean() / (df['mean_error'].mean() + 1e-10)
            }
        
        return results
    
    def _analyze_model_agreement(self, results, nucleus_perf):
        """Model uyumu analizi"""
        agreement_data = {}
        
        for category in ['good', 'medium', 'poor']:
            nuclei = results[f'{category}_nuclei']
            cat_perf = nucleus_perf[nucleus_perf['nucleus'].isin(nuclei)]
            
            # Coefficient of variation
            cv = cat_perf['std_error'] / (cat_perf['mean_error'] + 1e-10)
            agreement_data[f'{category}_nuclei_agreement'] = 1 - cv.mean()
        
        # Genel uyum
        agreement_data['overall_agreement'] = np.mean([
            agreement_data['good_nuclei_agreement'],
            agreement_data['medium_nuclei_agreement'],
            agreement_data['poor_nuclei_agreement']
        ])
        
        return agreement_data
    
    def save_cross_model_report(self, filename='cross_model_report.xlsx'):
        """Excel rapor kaydet"""
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 1. Summary
            self._write_summary(writer)
            
            # 2. Her kategori için detaylı sheet
            for target_name, results in self.results.items():
                for category in ['good', 'medium', 'poor']:
                    self._write_category_details(writer, target_name, category, results)
            
            # 3. Model agreement matrix
            self._write_agreement_matrix(writer)
            
            # 4. Detailed statistics
            self._write_detailed_stats(writer)
        
        logger.info(f"[OK] Excel rapor kaydedildi: {filepath}")
    
    def _write_summary(self, writer):
        """Özet sheet"""
        summary_data = []
        
        for target_name, results in self.results.items():
            for category in ['good', 'medium', 'poor']:
                stats = results[f'{category}_stats']
                summary_data.append({
                    'Target': target_name,
                    'Category': category.capitalize(),
                    'Count': stats['count'],
                    'Mean_Error': stats['mean_error'],
                    'Std_Error': stats['std_error'],
                    'Mean_R2': stats['mean_r2'],
                    'Agreement': stats['mean_agreement']
                })
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Summary', index=False)
    
    def _write_category_details(self, writer, target_name, category, results):
        """Kategori detayları"""
        nuclei_list = results[f'{category}_nuclei']
        
        detailed_data = []
        for nucleus in nuclei_list:
            row = {'Nucleus': nucleus}
            
            # Her modelden bilgi
            for model_name, df in self.predictions.items():
                nuc_data = df[df['nucleus'] == nucleus]
                if len(nuc_data) > 0:
                    row[f'{model_name}_Pred'] = nuc_data['prediction'].iloc[0]
                    row[f'{model_name}_Error'] = nuc_data['error'].iloc[0]
                    row[f'{model_name}_R2'] = nuc_data['r2'].iloc[0]
            
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        
        # İstatistik satırı
        stats_row = {'Nucleus': 'MEAN'}
        for col in df_detailed.columns:
            if col != 'Nucleus' and df_detailed[col].dtype in [np.float64, np.int64]:
                stats_row[col] = df_detailed[col].mean()
        
        df_detailed = pd.concat([df_detailed, pd.DataFrame([stats_row])], ignore_index=True)
        
        sheet_name = f'{target_name}_{category.capitalize()}'[:31]
        df_detailed.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _write_agreement_matrix(self, writer):
        """Model uyum matrisi"""
        model_names = list(self.predictions.keys())
        n_models = len(model_names)
        
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    df1 = self.predictions[model1]
                    df2 = self.predictions[model2]
                    
                    merged = df1.merge(df2, on='nucleus', suffixes=('_1', '_2'))
                    
                    if len(merged) > 0:
                        corr = merged['error_1'].corr(merged['error_2'])
                        agreement_matrix[i, j] = corr
        
        df_matrix = pd.DataFrame(agreement_matrix, index=model_names, columns=model_names)
        df_matrix.to_excel(writer, sheet_name='Agreement_Matrix')
    
    def _write_detailed_stats(self, writer):
        """Detaylı istatistikler"""
        stats_data = []
        
        for model_name, df in self.predictions.items():
            stats_data.append({
                'Model': model_name,
                'N_Predictions': len(df),
                'Mean_Error': df['error'].mean(),
                'Std_Error': df['error'].std(),
                'Min_Error': df['error'].min(),
                'Max_Error': df['error'].max(),
                'Median_Error': df['error'].median(),
                'Mean_R2': df['r2'].mean()
            })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='Model_Statistics', index=False)
    
    def visualize_results(self, target_name='MM'):
        """Görselleştirmeler oluştur"""
        if target_name not in self.results:
            logger.warning(f"No results for {target_name}")
            return
        
        results = self.results[target_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Kategori dağılımı
        ax = axes[0, 0]
        categories = ['Good', 'Medium', 'Poor']
        counts = [
            len(results['good_nuclei']),
            len(results['medium_nuclei']),
            len(results['poor_nuclei'])
        ]
        colors = ['green', 'orange', 'red']
        ax.bar(categories, counts, color=colors, alpha=0.7)
        ax.set_title(f'Nucleus Classification - {target_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Ortalama hata
        ax = axes[0, 1]
        mean_errors = [
            results['good_stats']['mean_error'],
            results['medium_stats']['mean_error'],
            results['poor_stats']['mean_error']
        ]
        ax.bar(categories, mean_errors, color=colors, alpha=0.7)
        ax.set_title(f'Mean Error by Category - {target_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Agreement skorları
        ax = axes[1, 0]
        agreements = [
            results['good_stats']['mean_agreement'],
            results['medium_stats']['mean_agreement'],
            results['poor_stats']['mean_agreement']
        ]
        ax.bar(categories, agreements, color=colors, alpha=0.7)
        ax.set_title(f'Model Agreement by Category - {target_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Agreement Score')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Genel istatistikler
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
CROSS-MODEL ANALYSIS - {target_name}
{'='*50}

Toplam Model Sayısı: {len(self.predictions)}

Good Nuclei:
  Sayı: {len(results['good_nuclei'])}
  Ortalama Hata: {results['good_stats']['mean_error']:.4f}
  Ortalama R²: {results['good_stats']['mean_r2']:.4f}
  Model Uyumu: {results['good_stats']['mean_agreement']:.4f}

Medium Nuclei:
  Sayı: {len(results['medium_nuclei'])}
  Ortalama Hata: {results['medium_stats']['mean_error']:.4f}
  Ortalama R²: {results['medium_stats']['mean_r2']:.4f}
  Model Uyumu: {results['medium_stats']['mean_agreement']:.4f}

Poor Nuclei:
  Sayı: {len(results['poor_nuclei'])}
  Ortalama Hata: {results['poor_stats']['mean_error']:.4f}
  Ortalama R²: {results['poor_stats']['mean_r2']:.4f}
  Model Uyumu: {results['poor_stats']['mean_agreement']:.4f}

Genel Model Uyumu: {results['model_agreement']['overall_agreement']:.3f}
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
               fontfamily='monospace')
        
        plt.tight_layout()
        
        viz_path = self.output_dir / f'cross_model_visualization_{target_name}.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Görselleştirme kaydedildi: {viz_path}")


class CrossModelAnalysisPipeline:
    """Ana cross-model analiz pipeline'ı"""
    
    def __init__(self, trained_models_dir='trained_models',
                 output_dir='reports/cross_model_analysis'):
        self.trained_models_dir = Path(trained_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ai_models = ['RandomForest', 'GradientBoosting', 'XGBoost', 'DNN', 'BNN', 'PINN']
        self.anfis_configs = ['GAU2MF', 'GEN2MF', 'TRI2MF', 'TRA2MF', 'GAU3MF', 
                             'SUBR03', 'SUBR05', 'SUBR07']
        self.targets = ['MM', 'QM', 'MM_QM', 'Beta_2']
        
        self.all_predictions = {}
        
        logger.info("Cross-Model Analysis Pipeline initialized")
    
    def run_complete_analysis(self):
        """Tam analiz çalıştır"""
        logger.info("\n" + "="*80)
        logger.info("FAZ 5: CROSS-MODEL ANALYSIS BAŞLIYOR")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Tahminleri topla
        logger.info("\n1. Model tahminleri toplanıyor...")
        self._collect_all_predictions()
        
        # 2. Her target için analiz
        logger.info("\n2. Target-bazlı analizler yapılıyor...")
        results = {}
        
        for target in self.targets:
            logger.info(f"\n{'='*80}")
            logger.info(f"TARGET: {target}")
            logger.info(f"{'='*80}")
            
            if target not in self.all_predictions or len(self.all_predictions[target]) == 0:
                logger.warning(f"[WARNING] {target} için tahmin bulunamadı, atlanıyor...")
                continue
            
            evaluator = CrossModelEvaluator(self.output_dir / target)
            
            # Model tahminlerini ekle
            for model_name, df in self.all_predictions[target].items():
                evaluator.add_predictions(
                    model_name, df,
                    target_col='experimental',
                    prediction_col='predicted',
                    nucleus_col='nucleus'
                )
            
            # Analiz
            target_results = evaluator.evaluate_common_performance(
                target_name=target,
                top_n=50
            )
            
            # Görselleştir
            evaluator.visualize_results(target)
            
            # Kaydet
            evaluator.save_cross_model_report(f'{target}_cross_model_report.xlsx')
            
            results[target] = target_results
        
        # 3. Master rapor
        logger.info("\n3. Master rapor oluşturuluyor...")
        self._create_master_report(results)
        
        # 4. Özet
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("FAZ 5: CROSS-MODEL ANALYSIS TAMAMLANDI")
        logger.info("="*80)
        logger.info(f"Süre: {duration:.2f} saniye ({duration/60:.1f} dakika)")
        logger.info(f"Analiz edilen target sayısı: {len(results)}")
        logger.info(f"Toplam model sayısı: {sum(len(self.all_predictions.get(t, {})) for t in self.targets)}")
        
        self._save_summary_json(results, duration)
        
        return results
    
    def _collect_all_predictions(self):
        """Tüm model tahminlerini topla"""
        # AI modelleri
        ai_dir = self.trained_models_dir / 'AI'
        if ai_dir.exists():
            for model_name in self.ai_models:
                model_dir = ai_dir / model_name
                if model_dir.exists():
                    self._load_model_predictions(model_name, model_dir, 'AI')
        
        # ANFIS modelleri
        anfis_dir = self.trained_models_dir / 'ANFIS'
        if anfis_dir.exists():
            for config_name in self.anfis_configs:
                config_dir = anfis_dir / config_name
                if config_dir.exists():
                    self._load_model_predictions(f'ANFIS_{config_name}', config_dir, 'ANFIS')
        
        # Özet
        logger.info(f"\n  [OK] Tahmin toplama özeti:")
        for target in self.targets:
            n_models = len(self.all_predictions.get(target, {}))
            logger.info(f"    {target}: {n_models} model")
    
    def _load_model_predictions(self, model_name, model_dir, model_type):
        """Bir modelin tahminlerini yükle"""
        for target in self.targets:
            pred_file = model_dir / f'{target}_predictions.csv'
            
            if pred_file.exists():
                try:
                    df = pd.read_csv(pred_file)
                    
                    # Sütun kontrolü
                    required_cols = ['nucleus', 'experimental', 'predicted']
                    if not all(col in df.columns for col in required_cols):
                        logger.warning(f"[WARNING] {pred_file} gerekli sütunlara sahip değil")
                        continue
                    
                    # Target'a ekle
                    if target not in self.all_predictions:
                        self.all_predictions[target] = {}
                    
                    self.all_predictions[target][model_name] = df
                    logger.debug(f"  [OK] {model_name} - {target}: {len(df)} çekirdek")
                    
                except Exception as e:
                    logger.error(f"  [FAIL] {model_name} - {target} yükleme hatası: {e}")
    
    def _create_master_report(self, results):
        """Master Excel rapor"""
        master_file = self.output_dir / 'MASTER_CROSS_MODEL_REPORT.xlsx'
        
        with pd.ExcelWriter(master_file, engine='openpyxl') as writer:
            # 1. Genel özet
            self._write_overall_summary(writer, results)
            
            # 2. Her target için özet
            for target, res in results.items():
                self._write_target_summary(writer, target, res)
            
            # 3. Kategori karşılaştırmaları
            self._write_category_comparison(writer, results)
            
            # 4. Model istatistikleri
            self._write_model_statistics_all(writer)
        
        logger.info(f"  [OK] Master rapor kaydedildi: {master_file}")
    
    def _write_overall_summary(self, writer, results):
        """Genel özet sheet"""
        summary_data = []
        
        for target, res in results.items():
            summary_data.append({
                'Target': target,
                'N_Models': len(self.all_predictions.get(target, {})),
                'Good_Count': len(res.get('good_nuclei', [])),
                'Medium_Count': len(res.get('medium_nuclei', [])),
                'Poor_Count': len(res.get('poor_nuclei', [])),
                'Overall_Agreement': res.get('model_agreement', {}).get('overall_agreement', 0),
                'Good_Mean_Error': res.get('good_stats', {}).get('mean_error', 0),
                'Medium_Mean_Error': res.get('medium_stats', {}).get('mean_error', 0),
                'Poor_Mean_Error': res.get('poor_stats', {}).get('mean_error', 0)
            })
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Overall_Summary', index=False)
    
    def _write_target_summary(self, writer, target, results):
        """Target özeti"""
        summary_data = []
        
        for category in ['good', 'medium', 'poor']:
            stats = results[f'{category}_stats']
            summary_data.append({
                'Category': category.capitalize(),
                'Count': stats['count'],
                'Mean_Error': stats['mean_error'],
                'Std_Error': stats['std_error'],
                'Mean_R2': stats['mean_r2'],
                'Agreement': stats['mean_agreement']
            })
        
        df = pd.DataFrame(summary_data)
        sheet_name = f'{target}_Summary'[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _write_category_comparison(self, writer, results):
        """Kategori karşılaştırma"""
        comparison_data = []
        
        for target, res in results.items():
            for category in ['good', 'medium', 'poor']:
                stats = res[f'{category}_stats']
                nuclei = res[f'{category}_nuclei']
                
                comparison_data.append({
                    'Target': target,
                    'Category': category.capitalize(),
                    'Count': len(nuclei),
                    'Mean_Error': stats['mean_error'],
                    'Mean_R2': stats['mean_r2'],
                    'Agreement': stats['mean_agreement']
                })
        
        df = pd.DataFrame(comparison_data)
        df.to_excel(writer, sheet_name='Category_Comparison', index=False)
    
    def _write_model_statistics_all(self, writer):
        """Tüm model istatistikleri"""
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
                    'Median_Error': error.median()
                })
        
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_excel(writer, sheet_name='All_Model_Statistics', index=False)
    
    def _save_summary_json(self, results, duration):
        """JSON özet kaydet"""
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
        
        logger.info(f"  [OK] JSON özet kaydedildi: {json_file}")


def main():
    """Ana fonksiyon"""
    logger.info("="*80)
    logger.info("FAZ 5: CROSS-MODEL ANALYSIS PIPELINE")
    logger.info("="*80)
    
    # Pipeline oluştur
    pipeline = CrossModelAnalysisPipeline(
        trained_models_dir='trained_models',
        output_dir='reports/cross_model_analysis'
    )
    
    # Analiz çalıştır
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
