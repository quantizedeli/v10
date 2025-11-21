"""
Kontrol Grubu Değerlendirme Modülü
Control Group Evaluation Module

Tüm modellerin kontrol grubundaki tahminlerini toplar, karşılaştırır ve raporlar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControlGroupEvaluator:
    """Kontrol grubu değerlendirici - Tüm modellerin blind test sonuçları"""
    
    def __init__(self, output_dir='reports/control_group'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = []
        
    def evaluate_all_models(self, control_group_path, trained_models_dir, target='MM'):
        """
        Tüm modellerin kontrol grubundaki tahminlerini değerlendir
        
        Args:
            control_group_path: Kontrol grubu dosya yolu
            trained_models_dir: Eğitilmiş modellerin klasörü
            target: Hedef değişken (MM, QM, MM_QM, Beta_2)
        
        Returns:
            DataFrame: Tüm tahminleri içeren detaylı rapor
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"KONTROL GRUBU DEĞERLENDİRMESİ - {target}")
        logger.info(f"{'='*80}")
        
        # Kontrol grubunu yükle
        control_df = pd.read_excel(control_group_path)
        logger.info(f"✓ Kontrol grubu yüklendi: {len(control_df)} nükleus")
        
        # Gerçek değerleri al
        if target == 'MM':
            y_true = control_df['MM'].values
            target_cols = ['MM']
        elif target == 'QM':
            y_true = control_df['Q'].values
            target_cols = ['Q']
        elif target == 'MM_QM':
            y_true = control_df[['MM', 'Q']].values
            target_cols = ['MM', 'Q']
        elif target == 'Beta_2':
            y_true = control_df['Beta_2'].values
            target_cols = ['Beta_2']
        
        # Tüm modelleri bul
        trained_models_dir = Path(trained_models_dir)
        all_predictions = []
        
        model_types = ['RF', 'GBM', 'XGBoost', 'DNN', 'BNN', 'PINN', 'ANFIS']
        
        for model_type in model_types:
            model_dir = trained_models_dir / model_type
            
            if not model_dir.exists():
                logger.warning(f"⚠ {model_type} klasörü bulunamadı, atlanıyor")
                continue
            
            logger.info(f"\n→ {model_type} modelleri test ediliyor...")
            
            # Bu model tipinin tüm varyantlarını test et
            for model_path in model_dir.glob('**/'):
                if not (model_path / 'model.pkl').exists() and not (model_path / 'model.keras').exists():
                    continue
                
                try:
                    pred_result = self._predict_control_group(
                        model_path, control_df, target_cols, model_type
                    )
                    
                    if pred_result is not None:
                        pred_result['y_true'] = y_true if len(target_cols) == 1 else y_true[:, 0]
                        pred_result['nucleus'] = control_df['NUCLEUS'].values
                        pred_result['A'] = control_df['A'].values
                        pred_result['Z'] = control_df['Z'].values
                        pred_result['N'] = control_df['N'].values
                        all_predictions.append(pred_result)
                        
                except Exception as e:
                    logger.warning(f"  ✗ Hata ({model_path.name}): {e}")
                    continue
        
        # Sonuçları birleştir
        if not all_predictions:
            logger.error("✗ Hiç tahmin yapılamadı!")
            return None
        
        logger.info(f"\n✓ {len(all_predictions)} model başarıyla test edildi")
        
        # Excel raporu oluştur
        report_path = self._create_excel_report(all_predictions, control_df, target)
        
        # Görselleştirmeler oluştur
        self._create_visualizations(all_predictions, target)
        
        # Özet istatistikler
        self._print_summary_statistics(all_predictions)
        
        return all_predictions
    
    def _predict_control_group(self, model_path, control_df, target_cols, model_type):
        """Bir model ile kontrol grubunda tahmin yap"""
        
        try:
            # Metadata'yı oku
            metadata_path = model_path / 'metadata.json'
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Feature'ları hazırla
            features = metadata['features']
            X_control = control_df[features].values
            
            # Scaler varsa uygula
            scaler_path = model_path / 'scaler.pkl'
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X_control = scaler.transform(X_control)
            
            # Modeli yükle ve tahmin yap
            if model_type in ['RF', 'GBM', 'XGBoost']:
                model = joblib.load(model_path / 'model.pkl')
                y_pred = model.predict(X_control)
                
            elif model_type in ['DNN', 'BNN', 'PINN']:
                from tensorflow import keras
                model = keras.models.load_model(model_path / 'model.keras')
                y_pred = model.predict(X_control, verbose=0).flatten()
            
            elif model_type == 'ANFIS':
                # ANFIS için MATLAB gerekli (şimdilik atlayalım)
                return None
            
            # Metrikleri hesapla
            y_true = control_df[target_cols[0]].values  # İlk target
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            error = y_pred - y_true
            
            result = {
                'model_type': model_type,
                'model_name': model_path.name,
                'dataset_config': metadata.get('dataset_name', 'Unknown'),
                'features': features,
                'n_features': len(features),
                'y_pred': y_pred,
                'error': error,
                'abs_error': np.abs(error),
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'max_error': np.max(np.abs(error)),
                'median_error': np.median(np.abs(error))
            }
            
            logger.info(f"  ✓ {model_path.name}: R²={r2:.4f}, RMSE={rmse:.4f}")
            
            return result
            
        except Exception as e:
            logger.debug(f"  Tahmin hatası: {e}")
            return None
    
    def _create_excel_report(self, predictions, control_df, target):
        """Detaylı Excel raporu oluştur"""
        
        output_file = self.output_dir / f'Control_Group_Predictions_{target}.xlsx'
        
        logger.info(f"\n→ Excel raporu oluşturuluyor: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Format tanımlamaları
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'align': 'center',
                'border': 1
            })
            
            # Sheet 1: Tüm Tahminler (Her nükleus için her modelin tahmini)
            all_preds = []
            for pred in predictions:
                for i, nucleus in enumerate(pred['nucleus']):
                    all_preds.append({
                        'NUCLEUS': nucleus,
                        'A': pred['A'][i],
                        'Z': pred['Z'][i],
                        'N': pred['N'][i],
                        'True_Value': pred['y_true'][i],
                        'Model_Type': pred['model_type'],
                        'Model_Name': pred['model_name'],
                        'Prediction': pred['y_pred'][i],
                        'Error': pred['error'][i],
                        'Abs_Error': pred['abs_error'][i],
                        'N_Features': pred['n_features']
                    })
            
            df_all = pd.DataFrame(all_preds)
            df_all.to_excel(writer, sheet_name='All_Predictions', index=False)
            
            # Sheet 2: Model Özeti
            model_summary = []
            for pred in predictions:
                model_summary.append({
                    'Model_Type': pred['model_type'],
                    'Model_Name': pred['model_name'],
                    'Dataset_Config': pred['dataset_config'],
                    'N_Features': pred['n_features'],
                    'R²': pred['r2'],
                    'RMSE': pred['rmse'],
                    'MAE': pred['mae'],
                    'Max_Error': pred['max_error'],
                    'Median_Error': pred['median_error']
                })
            
            df_summary = pd.DataFrame(model_summary)
            df_summary = df_summary.sort_values('R²', ascending=False)
            df_summary.to_excel(writer, sheet_name='Model_Summary', index=False)
            
            # Sheet 3: En İyi Modeller (Her model tipinden en iyisi)
            best_models = df_summary.groupby('Model_Type').apply(
                lambda x: x.nlargest(1, 'R²')
            ).reset_index(drop=True)
            best_models.to_excel(writer, sheet_name='Best_Models', index=False)
            
            # Sheet 4: Nükleus Bazında Karşılaştırma
            nucleus_comparison = []
            for nucleus in control_df['NUCLEUS'].unique():
                nucleus_data = df_all[df_all['NUCLEUS'] == nucleus]
                
                if len(nucleus_data) > 0:
                    nucleus_comparison.append({
                        'NUCLEUS': nucleus,
                        'A': nucleus_data.iloc[0]['A'],
                        'Z': nucleus_data.iloc[0]['Z'],
                        'N': nucleus_data.iloc[0]['N'],
                        'True_Value': nucleus_data.iloc[0]['True_Value'],
                        'N_Models': len(nucleus_data),
                        'Best_Prediction': nucleus_data.loc[nucleus_data['Abs_Error'].idxmin(), 'Prediction'],
                        'Worst_Prediction': nucleus_data.loc[nucleus_data['Abs_Error'].idxmax(), 'Prediction'],
                        'Mean_Prediction': nucleus_data['Prediction'].mean(),
                        'Std_Prediction': nucleus_data['Prediction'].std(),
                        'Min_Error': nucleus_data['Abs_Error'].min(),
                        'Max_Error': nucleus_data['Abs_Error'].max(),
                        'Mean_Error': nucleus_data['Abs_Error'].mean()
                    })
            
            df_nucleus = pd.DataFrame(nucleus_comparison)
            df_nucleus.to_excel(writer, sheet_name='Nucleus_Comparison', index=False)
            
            # Sheet 5: Model Tipi Bazında İstatistikler
            type_stats = df_summary.groupby('Model_Type').agg({
                'R²': ['mean', 'std', 'max', 'min'],
                'RMSE': ['mean', 'std', 'min', 'max'],
                'MAE': ['mean', 'std', 'min', 'max']
            }).round(4)
            type_stats.to_excel(writer, sheet_name='Model_Type_Stats')
        
        logger.info(f"✓ Excel raporu kaydedildi: {output_file}")
        return output_file
    
    def _create_visualizations(self, predictions, target):
        """Görselleştirmeler oluştur"""
        
        logger.info("\n→ Görselleştirmeler oluşturuluyor...")
        
        # 1. Model Karşılaştırma (R² skorları)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_r2 = [(p['model_name'], p['r2'], p['model_type']) for p in predictions]
        model_r2.sort(key=lambda x: x[1], reverse=True)
        
        names = [m[0][:30] for m in model_r2[:20]]  # Top 20
        r2s = [m[1] for m in model_r2[:20]]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        ax.barh(names, r2s, color=colors)
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_title(f'Top 20 Models - Control Group Performance ({target})', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'top20_models_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Tipi Karşılaştırma
        fig, ax = plt.subplots(figsize=(10, 6))
        
        type_data = {}
        for p in predictions:
            if p['model_type'] not in type_data:
                type_data[p['model_type']] = []
            type_data[p['model_type']].append(p['r2'])
        
        ax.boxplot(type_data.values(), labels=type_data.keys())
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title(f'Model Type Comparison - Control Group ({target})', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'model_type_comparison_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Görselleştirmeler kaydedildi: {self.output_dir}")
    
    def _print_summary_statistics(self, predictions):
        """Özet istatistikleri yazdır"""
        
        logger.info(f"\n{'='*80}")
        logger.info("ÖZET İSTATİSTİKLER")
        logger.info(f"{'='*80}")
        
        r2_scores = [p['r2'] for p in predictions]
        rmse_scores = [p['rmse'] for p in predictions]
        
        logger.info(f"Toplam model sayısı: {len(predictions)}")
        logger.info(f"\nR² İstatistikleri:")
        logger.info(f"  Ortalama: {np.mean(r2_scores):.4f}")
        logger.info(f"  Std: {np.std(r2_scores):.4f}")
        logger.info(f"  Min: {np.min(r2_scores):.4f}")
        logger.info(f"  Max: {np.max(r2_scores):.4f}")
        logger.info(f"  Median: {np.median(r2_scores):.4f}")
        
        logger.info(f"\nRMSE İstatistikleri:")
        logger.info(f"  Ortalama: {np.mean(rmse_scores):.4f}")
        logger.info(f"  Min: {np.min(rmse_scores):.4f}")
        logger.info(f"  Max: {np.max(rmse_scores):.4f}")
        
        # En iyi modeller
        logger.info(f"\n{'='*80}")
        logger.info("EN İYİ 5 MODEL (R² skoruna göre)")
        logger.info(f"{'='*80}")
        
        sorted_preds = sorted(predictions, key=lambda x: x['r2'], reverse=True)[:5]
        for i, pred in enumerate(sorted_preds, 1):
            logger.info(f"{i}. {pred['model_type']} - {pred['model_name']}")
            logger.info(f"   R²={pred['r2']:.4f}, RMSE={pred['rmse']:.4f}, MAE={pred['mae']:.4f}")
        
        logger.info(f"{'='*80}\n")


def main():
    """Test fonksiyonu"""
    evaluator = ControlGroupEvaluator('test_control_group_reports')
    
    # Örnek kullanım
    print("""
    Kullanım Örneği:
    ================
    
    evaluator = ControlGroupEvaluator('reports/control_group')
    
    # MM hedefi için değerlendirme
    results = evaluator.evaluate_all_models(
        control_group_path='ANFIS_Datasets/Control_Group_MM.xlsx',
        trained_models_dir='output/trained_models',
        target='MM'
    )
    
    # QM hedefi için
    results = evaluator.evaluate_all_models(
        control_group_path='ANFIS_Datasets/Control_Group_QM.xlsx',
        trained_models_dir='output/trained_models',
        target='QM'
    )
    """)


if __name__ == "__main__":
    main()