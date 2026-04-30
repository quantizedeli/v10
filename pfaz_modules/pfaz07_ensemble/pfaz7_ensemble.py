
"""
PFAZ 7: Ensemble & Meta-Model
Voting ve Stacking ensemble yöntemleri
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class EnsemblePipeline:
    """Ensemble modelleri oluştur ve değerlendir"""
    
    def __init__(self, trained_models_dir='trained_models', output_dir='ensemble_results'):
        self.trained_models_dir = Path(trained_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_predictions = {}
        self.ensemble_results = {}
        
    def load_predictions(self, target='MM'):
        """Base model tahminlerini yükle"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TAHMİNLER YÜKLENİYOR - {target}")
        logger.info(f"{'='*80}")
        
        predictions = {}
        
        # AI models
        ai_dir = self.trained_models_dir / 'AI'
        if ai_dir.exists():
            for model_dir in ai_dir.iterdir():
                if model_dir.is_dir():
                    pred_file = model_dir / f'{target}_predictions.csv'
                    if pred_file.exists():
                        df = pd.read_csv(pred_file)
                        predictions[model_dir.name] = df
                        logger.info(f"  [OK] {model_dir.name}: {len(df)} tahmin")
        
        # ANFIS models
        anfis_dir = self.trained_models_dir / 'ANFIS'
        if anfis_dir.exists():
            for config_dir in anfis_dir.iterdir():
                if config_dir.is_dir():
                    pred_file = config_dir / f'{target}_predictions.csv'
                    if pred_file.exists():
                        df = pd.read_csv(pred_file)
                        predictions[f'ANFIS_{config_dir.name}'] = df
                        logger.info(f"  [OK] ANFIS_{config_dir.name}: {len(df)} tahmin")
        
        self.base_predictions[target] = predictions
        logger.info(f"\n[OK] Toplam {len(predictions)} model yüklendi")
        return predictions
    
    def create_simple_voting(self, target='MM'):
        """Simple voting ensemble"""
        logger.info(f"\n{'='*80}")
        logger.info("SIMPLE VOTING ENSEMBLE")
        logger.info(f"{'='*80}")
        
        predictions = self.base_predictions.get(target, {})
        if len(predictions) == 0:
            logger.warning("! Tahmin bulunamadı")
            return None
        
        # Ortak çekirdekler
        common_nuclei = set.intersection(*[set(df['nucleus']) for df in predictions.values()])
        logger.info(f"  Ortak çekirdek: {len(common_nuclei)}")
        
        # Ensemble tahminleri
        ensemble_data = []
        for nucleus in common_nuclei:
            preds = []
            exp_val = None
            
            for model_name, df in predictions.items():
                row = df[df['nucleus'] == nucleus]
                if len(row) > 0:
                    preds.append(row['predicted'].values[0])
                    if exp_val is None:
                        exp_val = row['experimental'].values[0]
            
            if len(preds) > 0:
                ensemble_data.append({
                    'nucleus': nucleus,
                    'experimental': exp_val,
                    'predicted': np.mean(preds),
                    'n_models': len(preds)
                })
        
        df_ensemble = pd.DataFrame(ensemble_data)
        
        # Metrikler
        r2 = r2_score(df_ensemble['experimental'], df_ensemble['predicted'])
        rmse = np.sqrt(mean_squared_error(df_ensemble['experimental'], df_ensemble['predicted']))
        mae = mean_absolute_error(df_ensemble['experimental'], df_ensemble['predicted'])
        
        result = {
            'method': 'simple_voting',
            'target': target,
            'n_models': len(predictions),
            'n_predictions': len(df_ensemble),
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'predictions': df_ensemble
        }
        
        logger.info(f"\n  R^2 = {r2:.4f}")
        logger.info(f"  RMSE = {rmse:.4f}")
        logger.info(f"  MAE = {mae:.4f}")
        
        return result
    
    def create_weighted_voting(self, target='MM', weights=None):
        """Weighted voting ensemble"""
        logger.info(f"\n{'='*80}")
        logger.info("WEIGHTED VOTING ENSEMBLE")
        logger.info(f"{'='*80}")
        
        predictions = self.base_predictions.get(target, {})
        if len(predictions) == 0:
            return None
        
        # Ağırlıklar (eşit ağırlık)
        n_models = len(predictions)
        if weights is None:
            weights = np.ones(n_models) / n_models
        
        logger.info(f"  Ağırlıklar: {weights}")
        
        # Ortak çekirdekler
        common_nuclei = set.intersection(*[set(df['nucleus']) for df in predictions.values()])
        
        # Ensemble tahminleri
        ensemble_data = []
        for nucleus in common_nuclei:
            preds = []
            exp_val = None
            
            for model_name, df in predictions.items():
                row = df[df['nucleus'] == nucleus]
                if len(row) > 0:
                    preds.append(row['predicted'].values[0])
                    if exp_val is None:
                        exp_val = row['experimental'].values[0]
            
            if len(preds) == n_models:
                weighted_pred = np.dot(preds, weights)
                ensemble_data.append({
                    'nucleus': nucleus,
                    'experimental': exp_val,
                    'predicted': weighted_pred,
                    'n_models': n_models
                })
        
        df_ensemble = pd.DataFrame(ensemble_data)
        
        # Metrikler
        r2 = r2_score(df_ensemble['experimental'], df_ensemble['predicted'])
        rmse = np.sqrt(mean_squared_error(df_ensemble['experimental'], df_ensemble['predicted']))
        mae = mean_absolute_error(df_ensemble['experimental'], df_ensemble['predicted'])
        
        result = {
            'method': 'weighted_voting',
            'target': target,
            'n_models': n_models,
            'weights': weights.tolist(),
            'n_predictions': len(df_ensemble),
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'predictions': df_ensemble
        }
        
        logger.info(f"\n  R^2 = {r2:.4f}")
        logger.info(f"  RMSE = {rmse:.4f}")
        logger.info(f"  MAE = {mae:.4f}")
        
        return result
    
    def generate_report(self):
        """Ensemble karşılaştırma raporu"""
        logger.info(f"\n{'='*80}")
        logger.info("ENSEMBLE KARŞILAŞTIRMA RAPORU")
        logger.info(f"{'='*80}")
        
        report_file = self.output_dir / 'ensemble_comparison.xlsx'
        
        with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
            for target, methods in self.ensemble_results.items():
                data = []
                for method_name, result in methods.items():
                    data.append({
                        'Method': method_name,
                        'N_Models': result['n_models'],
                        'N_Predictions': result['n_predictions'],
                        'R²': result['R2'],
                        'RMSE': result['RMSE'],
                        'MAE': result['MAE']
                    })
                
                df = pd.DataFrame(data)
                df = df.sort_values('R²', ascending=False)
                df.to_excel(writer, sheet_name=f'{target}_Comparison', index=False)
                logger.info(f"  [OK] {target}_Comparison sheet")
        
        logger.info(f"\n[OK] Excel: {report_file}")
        return report_file
    
    def run_complete_pipeline(self, targets=['MM', 'QM']):
        """Tam ensemble pipeline"""
        start = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info("PFAZ 7: ENSEMBLE & META-MODEL")
        logger.info("="*80)
        
        for target in targets:
            logger.info(f"\n{'='*80}")
            logger.info(f"TARGET: {target}")
            logger.info(f"{'='*80}")
            
            # Tahminleri yükle
            predictions = self.load_predictions(target)
            
            if len(predictions) == 0:
                logger.warning(f"! {target} için tahmin yok, atlanıyor...")
                continue
            
            # Ensemble yöntemleri
            results = {}
            
            # Simple voting
            result_simple = self.create_simple_voting(target)
            if result_simple:
                results['Simple_Voting'] = result_simple
            
            # Weighted voting
            result_weighted = self.create_weighted_voting(target)
            if result_weighted:
                results['Weighted_Voting'] = result_weighted
            
            self.ensemble_results[target] = results
        
        # Rapor
        self.generate_report()
        
        # Summary
        duration = (datetime.now() - start).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("[SUCCESS] PFAZ 7 TAMAMLANDI")
        logger.info("="*80)
        logger.info(f"Süre: {duration:.1f} saniye")
        logger.info(f"Analiz edilen target: {len(self.ensemble_results)}")
        
        return self.ensemble_results


def main():
    pipeline = EnsemblePipeline()
    results = pipeline.run_complete_pipeline()
    return results


if __name__ == "__main__":
    main()
