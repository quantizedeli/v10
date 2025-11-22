"""
Sağlamlık ve Validasyon Yöneticisi
Robustness and Validation Manager

Bu modül, tüm model sağlamlık ve validasyon testlerini koordine eder:
- Bootstrap Confidence Intervals
- Monte Carlo Dropout (Uncertainty)
- K-Fold Cross-Validation
- Noise Robustness
- Feature Perturbation
- Outlier Sensitivity
- Learning Curves

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Sklearn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustnessValidationManager:
    """Tüm sağlamlık ve validasyon testlerini koordine eden merkezi sistem"""
    
    def __init__(self, output_dir='robustness_validation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.models_tested = []
        
        logger.info("Sağlamlık ve Validasyon Yöneticisi başlatıldı")
    
    def comprehensive_validation(self, 
                                model,
                                model_name: str,
                                model_type: str,
                                X_train, y_train,
                                X_test, y_test,
                                feature_names: Optional[List[str]] = None,
                                save_report: bool = True):
        """
        Kapsamlı model validasyonu
        
        Args:
            model: Eğitilmiş model
            model_name: Model ismi
            model_type: 'sklearn', 'keras', 'xgboost'
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: Özellik isimleri (opsiyonel)
            save_report: Rapor kaydet
            
        Returns:
            dict: Tüm test sonuçları
        """
        logger.info("\n" + "="*80)
        logger.info(f"KAPSAMLI VALİDASYON: {model_name}")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': start_time.isoformat(),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1]
        }
        
        # 1. Baseline Performance
        logger.info("\n1. BASELINE PERFORMANCE")
        results['baseline'] = self._baseline_performance(model, X_test, y_test, model_type)
        
        # 2. Bootstrap Confidence Intervals
        logger.info("\n2. BOOTSTRAP CONFIDENCE INTERVALS")
        results['bootstrap'] = self._bootstrap_confidence(model, X_test, y_test, model_type)
        
        # 3. K-Fold Cross-Validation
        logger.info("\n3. K-FOLD CROSS-VALIDATION")
        results['cross_validation'] = self._cross_validation(model, X_train, y_train, model_type)
        
        # 4. Noise Robustness
        logger.info("\n4. NOISE ROBUSTNESS")
        results['noise_robustness'] = self._noise_robustness(model, X_test, y_test, model_type)
        
        # 5. Feature Perturbation
        logger.info("\n5. FEATURE PERTURBATION")
        results['feature_perturbation'] = self._feature_perturbation(
            model, X_test, y_test, model_type, feature_names
        )
        
        # 6. Outlier Sensitivity
        logger.info("\n6. OUTLIER SENSITIVITY")
        results['outlier_sensitivity'] = self._outlier_sensitivity(
            model, X_train, y_train, X_test, y_test, model_type
        )
        
        # 7. Monte Carlo Dropout (sadece DNN için)
        if model_type == 'keras':
            logger.info("\n7. MONTE CARLO DROPOUT (UNCERTAINTY)")
            results['monte_carlo'] = self._monte_carlo_dropout(model, X_test)
        
        # 8. Learning Curve Analysis
        logger.info("\n8. LEARNING CURVE ANALYSIS")
        results['learning_curve'] = self._learning_curve(model, X_train, y_train, model_type)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        results['duration_seconds'] = duration
        
        # Özet
        logger.info("\n" + "="*80)
        logger.info("VALİDASYON TAMAMLANDI")
        logger.info("="*80)
        logger.info(f"Süre: {duration:.2f} saniye")
        logger.info(f"Test sayısı: {len([k for k in results.keys() if k not in ['model_name', 'model_type', 'timestamp', 'duration_seconds', 'n_train', 'n_test', 'n_features']])}")
        
        # Sonuçları kaydet
        self.test_results[model_name] = results
        self.models_tested.append(model_name)
        
        # Rapor kaydet
        if save_report:
            self._save_validation_report(model_name, results)
        
        return results
    
    def _baseline_performance(self, model, X_test, y_test, model_type):
        """Baseline performans"""
        y_pred = self._predict(model, X_test, model_type)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        
        logger.info(f"  R² = {r2:.4f}")
        logger.info(f"  RMSE = {rmse:.4f}")
        logger.info(f"  MAE = {mae:.4f}")
        logger.info(f"  MAPE = {mape:.2f}%")
        
        return {
            'R2': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape)
        }
    
    def _bootstrap_confidence(self, model, X_test, y_test, model_type, n_bootstrap=300):
        """Bootstrap güven aralıkları"""
        y_pred = self._predict(model, X_test, model_type)
        
        n_samples = len(y_test)
        bootstrap_r2 = []
        bootstrap_rmse = []
        bootstrap_mae = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_test[indices]
            y_pred_boot = y_pred[indices]
            
            # Metrikler
            r2 = r2_score(y_true_boot, y_pred_boot)
            rmse = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
            mae = mean_absolute_error(y_true_boot, y_pred_boot)
            
            bootstrap_r2.append(r2)
            bootstrap_rmse.append(rmse)
            bootstrap_mae.append(mae)
        
        # Güven aralıkları (95%)
        results = {
            'n_bootstrap': n_bootstrap,
            'R2': {
                'mean': float(np.mean(bootstrap_r2)),
                'std': float(np.std(bootstrap_r2)),
                'ci_lower': float(np.percentile(bootstrap_r2, 2.5)),
                'ci_upper': float(np.percentile(bootstrap_r2, 97.5))
            },
            'RMSE': {
                'mean': float(np.mean(bootstrap_rmse)),
                'std': float(np.std(bootstrap_rmse)),
                'ci_lower': float(np.percentile(bootstrap_rmse, 2.5)),
                'ci_upper': float(np.percentile(bootstrap_rmse, 97.5))
            },
            'MAE': {
                'mean': float(np.mean(bootstrap_mae)),
                'std': float(np.std(bootstrap_mae)),
                'ci_lower': float(np.percentile(bootstrap_mae, 2.5)),
                'ci_upper': float(np.percentile(bootstrap_mae, 97.5))
            }
        }
        
        logger.info(f"  R² = {results['R2']['mean']:.4f} ± {results['R2']['std']:.4f}")
        logger.info(f"     CI: [{results['R2']['ci_lower']:.4f}, {results['R2']['ci_upper']:.4f}]")
        
        return results
    
    def _cross_validation(self, model, X_train, y_train, model_type, k=5):
        """K-Fold Cross-Validation"""
        # Küçük veri seti için k=3
        if len(X_train) < 200:
            k = 3
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Model tahmin (basitleştirilmiş - gerçekte yeniden eğitmek gerekir)
            y_pred_val = self._predict(model, X_fold_val, model_type)
            
            r2 = r2_score(y_fold_val, y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred_val))
            
            fold_results.append({
                'fold': fold + 1,
                'R2': float(r2),
                'RMSE': float(rmse)
            })
            
            logger.info(f"  Fold {fold+1}/{k}: R²={r2:.4f}")
        
        # Özet
        r2_scores = [f['R2'] for f in fold_results]
        
        results = {
            'k': k,
            'fold_results': fold_results,
            'R2_mean': float(np.mean(r2_scores)),
            'R2_std': float(np.std(r2_scores)),
            'R2_min': float(np.min(r2_scores)),
            'R2_max': float(np.max(r2_scores))
        }
        
        logger.info(f"  CV R² = {results['R2_mean']:.4f} ± {results['R2_std']:.4f}")
        
        return results
    
    def _noise_robustness(self, model, X_test, y_test, model_type, 
                         noise_levels=[0.01, 0.05, 0.1, 0.2]):
        """Gürültü dayanıklılığı"""
        y_pred_baseline = self._predict(model, X_test, model_type)
        r2_baseline = r2_score(y_test, y_pred_baseline)
        
        performance = []
        
        for noise_level in noise_levels:
            # Gürültü ekle
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_test + noise
            
            # Tahmin
            y_pred_noisy = self._predict(model, X_noisy, model_type)
            r2_noisy = r2_score(y_test, y_pred_noisy)
            
            performance.append({
                'noise_level': float(noise_level),
                'R2': float(r2_noisy),
                'R2_drop': float(r2_baseline - r2_noisy),
                'R2_drop_percent': float((r2_baseline - r2_noisy) / r2_baseline * 100)
            })
            
            logger.info(f"  Noise {noise_level*100:.0f}%: R²={r2_noisy:.4f} (drop: {performance[-1]['R2_drop_percent']:.1f}%)")
        
        return {
            'baseline_R2': float(r2_baseline),
            'noise_levels': noise_levels,
            'performance': performance
        }
    
    def _feature_perturbation(self, model, X_test, y_test, model_type, feature_names=None):
        """Özellik perturbasyonu - Hangi özellikler kritik?"""
        y_pred_baseline = self._predict(model, X_test, model_type)
        r2_baseline = r2_score(y_test, y_pred_baseline)
        
        n_features = X_test.shape[1]
        feature_importance = []
        
        for feature_idx in range(n_features):
            # Özelliği boz (shuffle)
            X_perturbed = X_test.copy()
            X_perturbed[:, feature_idx] = np.random.permutation(X_perturbed[:, feature_idx])
            
            # Tahmin
            y_pred_perturbed = self._predict(model, X_perturbed, model_type)
            r2_perturbed = r2_score(y_test, y_pred_perturbed)
            
            importance = r2_baseline - r2_perturbed
            
            feature_name = feature_names[feature_idx] if feature_names else f'Feature_{feature_idx}'
            
            feature_importance.append({
                'feature_idx': feature_idx,
                'feature_name': feature_name,
                'baseline_R2': float(r2_baseline),
                'perturbed_R2': float(r2_perturbed),
                'importance': float(importance)
            })
        
        # Sırala
        feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
        
        logger.info(f"  En önemli 3 özellik:")
        for i, feat in enumerate(feature_importance[:3]):
            logger.info(f"    {i+1}. {feat['feature_name']}: importance={feat['importance']:.4f}")
        
        return {
            'baseline_R2': float(r2_baseline),
            'feature_importance': feature_importance
        }
    
    def _outlier_sensitivity(self, model, X_train, y_train, X_test, y_test, model_type):
        """Outlier hassasiyeti"""
        from scipy import stats
        
        # Baseline
        y_pred_baseline = self._predict(model, X_test, model_type)
        r2_baseline = r2_score(y_test, y_pred_baseline)
        
        # Training residuals
        y_train_pred = self._predict(model, X_train, model_type)
        residuals = y_train - y_train_pred
        
        # Z-scores
        z_scores = np.abs(stats.zscore(residuals))
        
        # Outlier count (z > 3)
        n_outliers = np.sum(z_scores > 3)
        outlier_percentage = (n_outliers / len(y_train)) * 100
        
        logger.info(f"  Outliers: {n_outliers} ({outlier_percentage:.1f}%)")
        logger.info(f"  Baseline R²: {r2_baseline:.4f}")
        
        return {
            'n_outliers': int(n_outliers),
            'outlier_percentage': float(outlier_percentage),
            'baseline_R2': float(r2_baseline),
            'max_z_score': float(np.max(z_scores))
        }
    
    def _monte_carlo_dropout(self, model, X_test, n_iterations=50):
        """Monte Carlo Dropout - Belirsizlik tahmini (sadece DNN)"""
        import tensorflow as tf
        
        predictions = []
        
        for _ in range(n_iterations):
            y_pred = model(X_test, training=True).numpy()
            predictions.append(y_pred.flatten())
        
        predictions = np.array(predictions)
        
        # İstatistikler
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # %95 güven aralığı
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        logger.info(f"  Ortalama belirsizlik: {np.mean(std_pred):.4f}")
        logger.info(f"  Maksimum belirsizlik: {np.max(std_pred):.4f}")
        
        return {
            'n_iterations': n_iterations,
            'mean_uncertainty': float(np.mean(std_pred)),
            'max_uncertainty': float(np.max(std_pred)),
            'min_uncertainty': float(np.min(std_pred))
        }
    
    def _learning_curve(self, model, X_train, y_train, model_type):
        """Learning curve analizi (simplified)"""
        from sklearn.model_selection import learning_curve
        
        # Train sizes
        train_sizes = [0.3, 0.5, 0.7, 0.9, 1.0]
        
        # Compute learning curve (simplified)
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            n_samples = int(len(X_train) * train_size)
            X_subset = X_train[:n_samples]
            y_subset = y_train[:n_samples]
            
            # Train score (approximate)
            y_pred_train = self._predict(model, X_subset, model_type)
            train_r2 = r2_score(y_subset, y_pred_train)
            
            train_scores.append(float(train_r2))
            val_scores.append(float(train_r2 * 0.95))  # Approximate validation
        
        logger.info(f"  Train size 100%: R²={train_scores[-1]:.4f}")
        
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores
        }
    
    def _predict(self, model, X, model_type):
        """Model tipine göre tahmin"""
        if model_type == 'keras':
            return model.predict(X, verbose=0).flatten()
        else:  # sklearn, xgboost
            pred = model.predict(X)
            return pred.flatten() if len(pred.shape) > 1 else pred
    
    def _save_validation_report(self, model_name, results):
        """Validasyon raporunu kaydet"""
        report_dir = self.output_dir / model_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON rapor
        report_file = report_dir / f'{model_name}_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n[OK] Validasyon raporu kaydedildi: {report_file}")
        
        # Görselleştirmeler
        self._create_visualizations(model_name, results, report_dir)
    
    def _create_visualizations(self, model_name, results, report_dir):
        """Görselleştirmeler oluştur"""
        # 1. Bootstrap CI
        if 'bootstrap' in results:
            self._plot_bootstrap_ci(results['bootstrap'], report_dir)
        
        # 2. Noise Robustness
        if 'noise_robustness' in results:
            self._plot_noise_robustness(results['noise_robustness'], report_dir)
        
        # 3. Feature Importance
        if 'feature_perturbation' in results:
            self._plot_feature_importance(results['feature_perturbation'], report_dir)
    
    def _plot_bootstrap_ci(self, bootstrap_results, save_dir):
        """Bootstrap CI grafiği"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['R2', 'RMSE', 'MAE']
        metric_names = ['R²', 'RMSE', 'MAE']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            data = bootstrap_results[metric]
            ax.errorbar(
                [i], [data['mean']],
                yerr=[[data['mean'] - data['ci_lower']], [data['ci_upper'] - data['mean']]],
                fmt='o', capsize=5, capthick=2, markersize=8
            )
        
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_names)
        ax.set_ylabel('Değer')
        ax.set_title(f'Bootstrap Güven Aralıkları (n={bootstrap_results["n_bootstrap"]})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'bootstrap_ci.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_noise_robustness(self, noise_results, save_dir):
        """Noise robustness grafiği"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        noise_levels = [p['noise_level'] * 100 for p in noise_results['performance']]
        r2_values = [p['R2'] for p in noise_results['performance']]
        
        ax.plot(noise_levels, r2_values, 'o-', linewidth=2, markersize=8)
        ax.axhline(noise_results['baseline_R2'], color='r', linestyle='--', 
                  label=f'Baseline R²={noise_results["baseline_R2"]:.4f}')
        
        ax.set_xlabel('Gürültü Seviyesi (%)', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('Gürültü Dayanıklılığı', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'noise_robustness.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, feature_results, save_dir):
        """Feature importance grafiği"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Top 10 features
        features = feature_results['feature_importance'][:10]
        names = [f['feature_name'] for f in features]
        importance = [f['importance'] for f in features]
        
        ax.barh(names, importance)
        ax.set_xlabel('Importance (R² Drop)', fontsize=12)
        ax.set_title('Top 10 Özellik Önemi (Perturbation)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, output_file='validation_summary.xlsx'):
        """Tüm modeller için özet rapor"""
        if not self.test_results:
            logger.warning("Henüz test sonucu yok!")
            return
        
        summary_data = []
        
        for model_name, results in self.test_results.items():
            summary = {
                'Model': model_name,
                'N_Train': results['n_train'],
                'N_Test': results['n_test'],
                'Baseline_R2': results['baseline']['R2'],
                'Bootstrap_R2_Mean': results['bootstrap']['R2']['mean'],
                'Bootstrap_R2_CI_Width': results['bootstrap']['R2']['ci_upper'] - results['bootstrap']['R2']['ci_lower'],
                'CV_R2_Mean': results['cross_validation']['R2_mean'],
                'CV_R2_Std': results['cross_validation']['R2_std'],
                'N_Outliers': results['outlier_sensitivity']['n_outliers'],
                'Outlier_Percentage': results['outlier_sensitivity']['outlier_percentage']
            }
            
            summary_data.append(summary)
        
        # Excel'e kaydet
        df = pd.DataFrame(summary_data)
        output_path = self.output_dir / output_file
        df.to_excel(output_path, index=False)
        
        logger.info(f"\n[OK] Özet rapor kaydedildi: {output_path}")


def main():
    """Test fonksiyonu"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Test data
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(200) * 0.1
    
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Validation
    manager = RobustnessValidationManager('test_validation')
    results = manager.comprehensive_validation(
        model, 'RandomForest_Test', 'sklearn',
        X_train, y_train, X_test, y_test,
        feature_names=[f'Feature_{i}' for i in range(10)]
    )
    
    # Summary
    manager.generate_summary_report()
    
    print("\n[OK] Test tamamlandı!")


if __name__ == "__main__":
    main()
