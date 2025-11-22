"""
SHAP Analizi ve Model Yorumlanabilirlik
SHAP Analysis and Model Interpretability

Dokümanda belirtilen SHAP grafikleri:
- SHAP summary plot
- SHAP beeswarm plot
- SHAP dependence plots
- SHAP waterfall plots
- Feature importance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """SHAP tabanlı model yorumlanabilirlik analizi"""
    
    def __init__(self, output_dir='visualizations/shap'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not SHAP_AVAILABLE:
            logger.error("SHAP kurulu değil! pip install shap")
    
    def analyze_model(self, model, X_train, X_test, feature_names, model_name='Model'):
        """
        Modelin SHAP analizi
        
        Args:
            model: Eğitilmiş model
            X_train: Training data
            X_test: Test data (SHAP için)
            feature_names: Özellik isimleri
            model_name: Model adı
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP analizi atlanıyor")
            return
        
        logger.info(f"SHAP analizi başlıyor: {model_name}")
        
        save_dir = self.output_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # SHAP explainer oluştur
        explainer = self._create_explainer(model, X_train)
        
        # SHAP values hesapla
        shap_values = explainer(X_test)
        
        # 1. Summary Plot (bar)
        self._plot_summary_bar(shap_values, feature_names, save_dir)
        
        # 2. Beeswarm Plot
        self._plot_beeswarm(shap_values, feature_names, save_dir)
        
        # 3. Dependence Plots (en önemli 3 özellik)
        self._plot_dependence(shap_values, X_test, feature_names, save_dir)
        
        # 4. Waterfall Plot (örnek tahminler)
        self._plot_waterfall(shap_values, feature_names, save_dir)
        
        # 5. Force Plot (interaktif HTML)
        self._plot_force(explainer, shap_values, X_test, feature_names, save_dir)
        
        logger.info(f"[OK] SHAP analizi tamamlandı: {save_dir}")
    
    def _create_explainer(self, model, X_train):
        """Model tipine göre uygun explainer oluştur"""
        
        # Tree-based modeller için TreeExplainer
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        
        # Sklearn modelleri için
        elif hasattr(model, 'predict'):
            # KernelExplainer (yavaş ama evrensel)
            explainer = shap.KernelExplainer(
                model.predict,
                shap.sample(X_train, 100)  # Background data
            )
        
        # Deep learning modelleri için
        else:
            explainer = shap.DeepExplainer(model, X_train[:100])
        
        return explainer
    
    def _plot_summary_bar(self, shap_values, feature_names, save_dir):
        """SHAP summary plot (bar chart)"""
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            plot_type='bar',
            show=False
        )
        
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # HTML versiyonu
        try:
            shap.plots.bar(shap_values, show=False)
            plt.savefig(save_dir / 'shap_summary_bar.html', dpi=300, bbox_inches='tight')
            plt.close()
        except:
            pass
    
    def _plot_beeswarm(self, shap_values, feature_names, save_dir):
        """SHAP beeswarm plot"""
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            show=False
        )
        
        plt.title('SHAP Beeswarm Plot', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # HTML interaktif
        try:
            shap.plots.beeswarm(shap_values, show=False)
            plt.savefig(save_dir / 'shap_beeswarm.html', dpi=300, bbox_inches='tight')
            plt.close()
        except:
            pass
    
    def _plot_dependence(self, shap_values, X_test, feature_names, save_dir, top_n=3):
        """SHAP dependence plots - en önemli özelliklere odaklan"""
        
        # En önemli özellikleri bul
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        fig, axes = plt.subplots(1, top_n, figsize=(15, 5))
        if top_n == 1:
            axes = [axes]
        
        for i, idx in enumerate(top_indices):
            feature_name = feature_names[idx]
            
            # SHAP dependence plot
            shap.dependence_plot(
                idx,
                shap_values.values,
                X_test,
                feature_names=feature_names,
                ax=axes[i],
                show=False
            )
            
            axes[i].set_title(f'{feature_name}', fontsize=12, fontweight='bold')
        
        plt.suptitle('SHAP Dependence Plots (Top 3 Features)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_waterfall(self, shap_values, feature_names, save_dir, n_samples=3):
        """SHAP waterfall plots - örnek tahminler"""
        
        for i in range(min(n_samples, len(shap_values))):
            plt.figure(figsize=(10, 6))
            
            shap.plots.waterfall(shap_values[i], show=False)
            
            plt.title(f'SHAP Waterfall Plot - Sample {i+1}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_dir / f'shap_waterfall_sample_{i+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_force(self, explainer, shap_values, X_test, feature_names, save_dir):
        """SHAP force plot (interaktif HTML)"""
        
        try:
            # İlk örnek için force plot
            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values[0].values,
                X_test.iloc[0] if hasattr(X_test, 'iloc') else X_test[0],
                feature_names=feature_names,
                show=False
            )
            
            # HTML olarak kaydet
            shap.save_html(str(save_dir / 'shap_force_plot.html'), force_plot)
            
        except Exception as e:
            logger.warning(f"Force plot oluşturulamadı: {e}")
    
    def compare_feature_importance(self, models_dict, X_test, feature_names, save_path):
        """
        Farklı modellerin feature importance karşılaştırması
        
        Args:
            models_dict: {model_name: model}
            X_test: Test data
            feature_names: Feature names
            save_path: Kayıt yolu
        """
        if not SHAP_AVAILABLE:
            return
        
        logger.info("Model feature importance karşılaştırması...")
        
        importance_data = []
        
        for model_name, model in models_dict.items():
            try:
                # SHAP values hesapla
                explainer = self._create_explainer(model, X_test[:100])
                shap_values = explainer(X_test[:100])
                
                # Ortalama mutlak SHAP değerleri
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                
                for feature, importance in zip(feature_names, mean_abs_shap):
                    importance_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': importance
                    })
            
            except Exception as e:
                logger.warning(f"{model_name} için importance hesaplanamadı: {e}")
        
        if not importance_data:
            return
        
        # DataFrame oluştur
        df = pd.DataFrame(importance_data)
        
        # Görselleştir
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Pivot table
        pivot_df = df.pivot(index='Feature', columns='Model', values='Importance')
        
        # Heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'SHAP Importance'})
        
        ax.set_title('Feature Importance Comparison Across Models', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Feature importance karşılaştırması kaydedildi: {save_path}")


class PermutationImportanceAnalyzer:
    """Permutation importance analizi (SHAP alternatifi/tamamlayıcı)"""
    
    def __init__(self, output_dir='visualizations/permutation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_model(self, model, X_test, y_test, feature_names, model_name='Model'):
        """Permutation importance analizi"""
        from sklearn.inspection import permutation_importance
        
        logger.info(f"Permutation importance analizi: {model_name}")
        
        # Permutation importance hesapla
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Sırala
        sorted_idx = result.importances_mean.argsort()[::-1]
        
        # Görselleştir
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(sorted_idx)), 
               result.importances_mean[sorted_idx],
               xerr=result.importances_std[sorted_idx])
        
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Permutation Importance', fontsize=12)
        ax.set_title(f'Permutation Feature Importance - {model_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{model_name}_permutation_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Permutation importance kaydedildi: {save_path}")
        
        return result


def main():
    """Test fonksiyonu"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Örnek veri
    np.random.seed(42)
    X_train = np.random.randn(200, 5)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] - X_train[:, 2] * 0.5
    
    X_test = np.random.randn(50, 5)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] - X_test[:, 2] * 0.5
    
    feature_names = ['Feature_A', 'Feature_Z', 'Feature_N', 'Feature_Spin', 'Feature_Beta2']
    
    # Model eğit
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # SHAP analizi
    if SHAP_AVAILABLE:
        print("\n=== SHAP Analizi ===")
        shap_analyzer = SHAPAnalyzer('test_output/shap')
        shap_analyzer.analyze_model(
            model, 
            X_train, 
            pd.DataFrame(X_test, columns=feature_names),
            feature_names,
            'RandomForest_Test'
        )
    
    # Permutation importance
    print("\n=== Permutation Importance ===")
    perm_analyzer = PermutationImportanceAnalyzer('test_output/permutation')
    perm_analyzer.analyze_model(
        model,
        X_test,
        y_test,
        feature_names,
        'RandomForest_Test'
    )
    
    print("\n[OK] Test tamamlandı")


if __name__ == "__main__":
    main()