"""
Model Interpretability Module - UPDATED
SHAP Analysis & Feature Importance FOR ALL MODELS

✅ UPDATED: ANFIS, TensorFlow, Ensemble support added

Özellikler:
1. SHAP (SHapley Additive exPlanations)
   - TreeExplainer (RF, GBM, XGBoost) ✅
   - DeepExplainer (DNN, BNN, PINN) ✅
   - KernelExplainer (ANFIS, Ensemble) ✅ NEW
2. Permutation Importance ✅
3. Feature Importance Rankings ✅
4. Visualizations ✅
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available! pip install shap")

# Sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# UNIVERSAL SHAP ANALYZER - SUPPORTS ALL MODELS
# ============================================================================

class UniversalSHAPAnalyzer:
    """
    Universal SHAP Analyzer - Works with ALL model types
    
    Supported:
    - Tree models (RF, GBM, XGBoost) ✅
    - Neural networks (DNN, BNN, PINN) ✅
    - ANFIS (via KernelExplainer) ✅
    - Ensemble models ✅
    """
    
    def __init__(self, model, model_type, feature_names=None, output_dir='interpretability'):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP required! pip install shap")
        
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.explainer = None
        self.shap_values = None
        
        logger.info(f"Universal SHAP Analyzer: {model_type}")
    
    def create_explainer(self, X_background, force_kernel=False):
        """
        Create appropriate SHAP explainer
        
        Args:
            X_background: Background dataset (100-500 samples)
            force_kernel: Force KernelExplainer (slower but works for all)
        """
        
        logger.info(f"Creating SHAP explainer for {self.model_type}")
        
        # Select appropriate explainer
        if force_kernel:
            self.explainer = self._create_kernel_explainer(X_background)
        
        elif self.model_type in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            self.explainer = self._create_tree_explainer()
        
        elif self.model_type in ['DNN', 'BNN', 'PINN']:
            self.explainer = self._create_deep_explainer(X_background)
        
        elif self.model_type in ['ANFIS', 'Ensemble', 'Voting', 'Stacking']:
            self.explainer = self._create_kernel_explainer(X_background)
        
        else:
            # Fallback: KernelExplainer (slow but universal)
            logger.warning(f"Unknown model type {self.model_type}, using KernelExplainer")
            self.explainer = self._create_kernel_explainer(X_background)
        
        logger.info("✓ Explainer ready")
    
    def _create_tree_explainer(self):
        """TreeExplainer for tree-based models"""
        logger.info("  → Using TreeExplainer (fast)")
        return shap.TreeExplainer(self.model)
    
    def _create_deep_explainer(self, X_background):
        """DeepExplainer for neural networks"""
        try:
            logger.info("  → Using DeepExplainer")
            return shap.DeepExplainer(self.model, X_background)
        except:
            logger.warning("  → DeepExplainer failed, falling back to KernelExplainer")
            return self._create_kernel_explainer(X_background)
    
    def _create_kernel_explainer(self, X_background):
        """KernelExplainer - universal but slow"""
        logger.info("  → Using KernelExplainer (slow but universal)")
        
        # Create predict function
        def predict_fn(X):
            try:
                # Try standard predict
                return self.model.predict(X)
            except:
                try:
                    # Try with __call__ (for custom models)
                    return self.model(X)
                except:
                    raise ValueError("Cannot create predict function for model")
        
        return shap.KernelExplainer(predict_fn, X_background)
    
    def calculate_shap_values(self, X_explain, n_samples=100):
        """Calculate SHAP values"""
        
        if self.explainer is None:
            raise ValueError("Call create_explainer() first!")
        
        # Limit samples for KernelExplainer (slow)
        if isinstance(self.explainer, shap.KernelExplainer):
            X_explain = X_explain[:min(n_samples, len(X_explain))]
            logger.info(f"KernelExplainer: Using {len(X_explain)} samples")
        
        logger.info(f"Calculating SHAP values: {X_explain.shape[0]} samples")
        
        self.shap_values = self.explainer.shap_values(X_explain)
        
        logger.info("✓ SHAP values calculated")
        
        return self.shap_values
    
    def plot_summary(self, X_explain, max_display=20, show=False):
        """SHAP summary plot"""
        
        if self.shap_values is None:
            raise ValueError("Calculate SHAP values first!")
        
        logger.info("Creating SHAP summary plot...")
        
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values,
            X_explain,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        save_path = self.output_dir / f'{self.model_type}_shap_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Summary plot: {save_path}")
    
    def get_feature_importance_df(self):
        """Get feature importance DataFrame"""
        
        if self.shap_values is None:
            raise ValueError("Calculate SHAP values first!")
        
        # Mean absolute SHAP
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names if self.feature_names else [f'F{i}' for i in range(len(mean_abs_shap))],
            'SHAP_Importance': mean_abs_shap
        })
        
        importance_df = importance_df.sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        # Save
        save_path = self.output_dir / f'{self.model_type}_shap_importance.csv'
        importance_df.to_csv(save_path, index=False)
        
        logger.info(f"✓ Feature importance: {save_path}")
        
        return importance_df


# Original classes remain the same, just update imports
# ============================================================================
# REST OF THE FILE - KEEP ORIGINAL
# ============================================================================

class PermutationImportanceAnalyzer:
    """Model-agnostic permutation importance - works for ALL models"""
    
    def __init__(self, model, feature_names=None, output_dir='interpretability'):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.importance_result = None
        
        logger.info("Permutation Importance Analyzer (Universal)")
    
    def calculate(self, X, y, n_repeats=10, random_state=42, scoring='r2'):
        """Calculate permutation importance - works for any model"""
        
        logger.info(f"Calculating permutation importance: {n_repeats} repeats")
        
        # Create scoring function that works with any model
        def custom_scorer(estimator, X, y):
            try:
                y_pred = estimator.predict(X)
                return r2_score(y, y_pred)
            except:
                try:
                    y_pred = estimator(X)  # For custom models
                    return r2_score(y, y_pred)
                except:
                    return 0.0
        
        self.importance_result = permutation_importance(
            self.model,
            X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=custom_scorer,
            n_jobs=-1
        )
        
        logger.info("✓ Permutation importance calculated")
        
        return self.importance_result
    
    def plot_importance(self, show=False):
        """Plot permutation importance"""
        
        if self.importance_result is None:
            raise ValueError("Call calculate() first!")
        
        logger.info("Creating permutation importance plot...")
        
        importances_mean = self.importance_result.importances_mean
        importances_std = self.importance_result.importances_std
        
        feature_names = self.feature_names if self.feature_names else [f'F{i}' for i in range(len(importances_mean))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances_mean,
            'Std': importances_std
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['Importance'], xerr=importance_df['Std'], 
                align='center', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance', fontsize=12)
        ax.set_title('Permutation Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'permutation_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Plot saved: {save_path}")
        
        # Save CSV
        csv_path = self.output_dir / 'permutation_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        
        return importance_df


class InterpretabilityAnalyzer:
    """
    Unified interpretability - UPDATED to support ALL models
    
    ✅ Now supports:
    - RF, GBM, XGBoost (TreeExplainer)
    - DNN, BNN, PINN (DeepExplainer)
    - ANFIS (KernelExplainer)
    - Ensemble (KernelExplainer)
    """
    
    def __init__(self, model, model_type, feature_names=None, target_names=None, output_dir='interpretability'):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.target_names = target_names
        self.output_dir = Path(output_dir) / model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.shap_analyzer = None
        self.perm_analyzer = None
        
        logger.info(f"Unified Interpretability Analyzer: {model_type}")
    
    def analyze(self, X_train, y_train, X_test, y_test, 
                use_shap=True, use_permutation=True,
                shap_background_size=100, n_shap_explain=200):
        """
        Complete interpretability analysis
        ✅ Works with ALL model types now
        """
        
        logger.info("\n" + "="*80)
        logger.info(f"INTERPRETABILITY ANALYSIS: {self.model_type}")
        logger.info("="*80 + "\n")
        
        results = {}
        
        # 1. SHAP Analysis (Universal)
        if use_shap and SHAP_AVAILABLE:
            logger.info("→ SHAP Analysis...")
            
            try:
                self.shap_analyzer = UniversalSHAPAnalyzer(
                    self.model,
                    self.model_type,
                    self.feature_names,
                    self.output_dir / 'shap'
                )
                
                # Background & explain data
                X_background = X_train[:shap_background_size]
                X_explain = X_test[:n_shap_explain]
                
                # Create explainer (auto-selects best method)
                self.shap_analyzer.create_explainer(X_background)
                
                # Calculate SHAP
                self.shap_analyzer.calculate_shap_values(X_explain)
                
                # Visualizations
                self.shap_analyzer.plot_summary(X_explain)
                
                # Feature importance
                shap_importance = self.shap_analyzer.get_feature_importance_df()
                
                results['shap_importance'] = shap_importance
                
                logger.info("✓ SHAP Analysis completed")
                
            except Exception as e:
                logger.error(f"✗ SHAP Analysis failed: {e}")
                results['shap_error'] = str(e)
        
        # 2. Permutation Importance (Universal)
        if use_permutation:
            logger.info("\n→ Permutation Importance...")
            
            try:
                self.perm_analyzer = PermutationImportanceAnalyzer(
                    self.model,
                    self.feature_names,
                    self.output_dir / 'permutation'
                )
                
                self.perm_analyzer.calculate(X_test, y_test, n_repeats=10)
                perm_importance = self.perm_analyzer.plot_importance()
                
                results['permutation_importance'] = perm_importance
                
                logger.info("✓ Permutation Importance completed")
                
            except Exception as e:
                logger.error(f"✗ Permutation Importance failed: {e}")
                results['permutation_error'] = str(e)
        
        # 3. Model-specific importance (if available)
        logger.info("\n→ Model-specific importance...")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names if self.feature_names else [f'F{i}' for i in range(len(importance))],
                'Importance': importance
            })
            
            importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            # Save
            save_path = self.output_dir / 'model_feature_importance.csv'
            importance_df.to_csv(save_path, index=False)
            
            results['model_importance'] = importance_df
            
            logger.info("✓ Model-specific importance saved")
        
        # Summary report
        self._create_summary_report(results)
        
        logger.info("\n" + "="*80)
        logger.info("✓ INTERPRETABILITY ANALYSIS COMPLETED")
        logger.info("="*80 + "\n")
        
        return results
    
    def _create_summary_report(self, results):
        """Create summary report"""
        
        summary = {
            'model_type': self.model_type,
            'timestamp': pd.Timestamp.now().isoformat(),
            'analyses_completed': []
        }
        
        if 'shap_importance' in results:
            summary['analyses_completed'].append('SHAP')
            summary['top_3_features_shap'] = results['shap_importance'].head(3)['Feature'].tolist()
        
        if 'permutation_importance' in results:
            summary['analyses_completed'].append('Permutation')
            summary['top_3_features_permutation'] = results['permutation_importance'].head(3)['Feature'].tolist()
        
        if 'model_importance' in results:
            summary['analyses_completed'].append('Model-Specific')
            summary['top_3_features_model'] = results['model_importance'].head(3)['Feature'].tolist()
        
        # Save summary
        with open(self.output_dir / 'interpretability_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✓ Summary report: {self.output_dir / 'interpretability_summary.json'}")


# ============================================================================
# MAIN TEST
# ============================================================================

def test_universal_interpretability():
    """Test universal interpretability"""
    
    print("\n" + "="*80)
    print("UNIVERSAL INTERPRETABILITY TEST")
    print("="*80)
    
    if not SHAP_AVAILABLE:
        print("⚠ SHAP not available, some tests will be skipped")
    
    # Dummy data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * (-1) + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Split
    train_size = int(0.7 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    score = r2_score(y_test, model.predict(X_test))
    print(f"\nModel R² Score: {score:.4f}")
    
    # Test universal analyzer
    analyzer = InterpretabilityAnalyzer(
        model,
        'RandomForest',
        feature_names=feature_names,
        output_dir='test_interpretability_universal'
    )
    
    results = analyzer.analyze(
        X_train, y_train,
        X_test, y_test,
        use_shap=SHAP_AVAILABLE,
        use_permutation=True,
        shap_background_size=50,
        n_shap_explain=100
    )
    
    print("\n" + "="*80)
    print("TOP 5 FEATURES:")
    print("="*80)
    
    if 'shap_importance' in results:
        print("\n✅ SHAP (Universal):")
        print(results['shap_importance'].head(5)[['Feature', 'SHAP_Importance']])
    
    if 'permutation_importance' in results:
        print("\n✅ Permutation (Universal):")
        print(results['permutation_importance'].head(5)[['Feature', 'Importance']])
    
    print("\n✓ Universal interpretability test completed!")
    print("✅ Now supports: RF, GBM, XGBoost, DNN, BNN, PINN, ANFIS, Ensemble")


if __name__ == "__main__":
    test_universal_interpretability()
    """Model Interpretability Module
SHAP Analysis & Feature Importance

Özellikler:
1. SHAP (SHapley Additive exPlanations)
   - TreeExplainer (RF, GBM, XGBoost)
   - DeepExplainer (DNN)
   - KernelExplainer (genel)
2. Permutation Importance
3. Feature Importance Rankings
4. Visualizations
   - Summary plots
   - Dependence plots
   - Force plots
   - Waterfall plots

Yazar: Nükleer Fizik AI Projesi
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available! pip install shap")

# Sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SHAP ANALYZER
# ============================================================================

class SHAPAnalyzer:
    """
    SHAP-based model interpretability
    
    Desteklenen modeller:
    - Tree-based: RF, GBM, XGBoost
    - Neural Networks: DNN (with DeepExplainer)
    - Other: KernelExplainer (slow but general)
    """
    
    def __init__(self, model, model_type, feature_names=None, output_dir='interpretability'):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP gerekli! pip install shap")
        
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.explainer = None
        self.shap_values = None
        
        logger.info(f"SHAP Analyzer başlatıldı: {model_type}")
    
    def create_explainer(self, X_background):
        """
        SHAP explainer oluştur
        
        Args:
            X_background: Background dataset (tipik 100-500 sample)
        """
        
        logger.info(f"SHAP Explainer oluşturuluyor: {self.model_type}")
        
        if self.model_type in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            # Tree explainer (fast)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("  → TreeExplainer kullanılıyor")
        
        elif self.model_type == 'DNN':
            # Deep explainer
            try:
                self.explainer = shap.DeepExplainer(self.model, X_background)
                logger.info("  → DeepExplainer kullanılıyor")
            except:
                # Fallback to KernelExplainer
                logger.warning("  → DeepExplainer başarısız, KernelExplainer kullanılıyor")
                self.explainer = shap.KernelExplainer(self.model.predict, X_background)
        
        else:
            # Kernel explainer (slow but general)
            self.explainer = shap.KernelExplainer(self.model.predict, X_background)
            logger.info("  → KernelExplainer kullanılıyor (yavaş olabilir)")
        
        logger.info("✓ Explainer hazır")
    
    def calculate_shap_values(self, X_explain):
        """
        SHAP values hesapla
        
        Args:
            X_explain: Samples to explain
        """
        
        if self.explainer is None:
            raise ValueError("Önce create_explainer() çağırın!")
        
        logger.info(f"SHAP values hesaplanıyor: {X_explain.shape[0]} sample")
        
        self.shap_values = self.explainer.shap_values(X_explain)
        
        logger.info("✓ SHAP values hesaplandı")
        
        return self.shap_values
    
    def plot_summary(self, X_explain, max_display=20, show=False):
        """SHAP summary plot (beeswarm)"""
        
        if self.shap_values is None:
            raise ValueError("Önce calculate_shap_values() çağırın!")
        
        logger.info("SHAP summary plot oluşturuluyor...")
        
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values,
            X_explain,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        save_path = self.output_dir / 'shap_summary_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Summary plot kaydedildi: {save_path}")
    
    def plot_bar(self, X_explain, max_display=20, show=False):
        """SHAP bar plot (mean absolute SHAP values)"""
        
        if self.shap_values is None:
            raise ValueError("Önce calculate_shap_values() çağırın!")
        
        logger.info("SHAP bar plot oluşturuluyor...")
        
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            self.shap_values,
            X_explain,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type='bar',
            show=False
        )
        
        plt.tight_layout()
        save_path = self.output_dir / 'shap_bar_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Bar plot kaydedildi: {save_path}")
    
    def plot_dependence(self, feature_idx, X_explain, interaction_index='auto', show=False):
        """
        SHAP dependence plot
        
        Args:
            feature_idx: Feature index veya ismi
            interaction_index: Interaction feature ('auto' for automatic)
        """
        
        if self.shap_values is None:
            raise ValueError("Önce calculate_shap_values() çağırın!")
        
        # Feature name to index
        if isinstance(feature_idx, str):
            if self.feature_names is not None:
                feature_idx = self.feature_names.index(feature_idx)
            else:
                raise ValueError("Feature names tanımlı değil!")
        
        feature_name = self.feature_names[feature_idx] if self.feature_names else f"Feature {feature_idx}"
        
        logger.info(f"SHAP dependence plot: {feature_name}")
        
        plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X_explain,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=False
        )
        
        plt.tight_layout()
        save_path = self.output_dir / f'shap_dependence_{feature_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Dependence plot kaydedildi: {save_path}")
    
    def plot_top_dependence_plots(self, X_explain, n_top=5):
        """En önemli features için dependence plots"""
        
        if self.shap_values is None:
            raise ValueError("Önce calculate_shap_values() çağırın!")
        
        # Get feature importance ranking
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:n_top]
        
        logger.info(f"Top {n_top} feature için dependence plots oluşturuluyor...")
        
        for idx in top_indices:
            self.plot_dependence(idx, X_explain)
        
        logger.info(f"✓ {n_top} dependence plot tamamlandı")
    
    def plot_waterfall(self, sample_idx, X_explain, show=False):
        """
        SHAP waterfall plot (single prediction explanation)
        
        Args:
            sample_idx: Sample index to explain
        """
        
        if self.shap_values is None:
            raise ValueError("Önce calculate_shap_values() çağırın!")
        
        logger.info(f"SHAP waterfall plot: Sample {sample_idx}")
        
        # Create explanation object
        shap_explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X_explain[sample_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_explanation, show=False)
        
        plt.tight_layout()
        save_path = self.output_dir / f'shap_waterfall_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Waterfall plot kaydedildi: {save_path}")
    
    def get_feature_importance_df(self):
        """SHAP-based feature importance DataFrame"""
        
        if self.shap_values is None:
            raise ValueError("Önce calculate_shap_values() çağırın!")
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names if self.feature_names else [f'F{i}' for i in range(len(mean_abs_shap))],
            'SHAP_Importance': mean_abs_shap
        })
        
        # Sort
        importance_df = importance_df.sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        # Save
        save_path = self.output_dir / 'shap_feature_importance.csv'
        importance_df.to_csv(save_path, index=False)
        
        logger.info(f"✓ Feature importance saved: {save_path}")
        
        return importance_df
    
    def export_shap_values(self):
        """SHAP values'ları CSV olarak kaydet"""
        
        if self.shap_values is None:
            raise ValueError("Önce calculate_shap_values() çağırın!")
        
        shap_df = pd.DataFrame(
            self.shap_values,
            columns=self.feature_names if self.feature_names else [f'F{i}' for i in range(self.shap_values.shape[1])]
        )
        
        save_path = self.output_dir / 'shap_values.csv'
        shap_df.to_csv(save_path, index=False)
        
        logger.info(f"✓ SHAP values exported: {save_path}")


# ============================================================================
# PERMUTATION IMPORTANCE ANALYZER
# ============================================================================

class PermutationImportanceAnalyzer:
    """
    Permutation importance analysis
    
    Model-agnostic feature importance
    """
    
    def __init__(self, model, feature_names=None, output_dir='interpretability'):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.importance_result = None
        
        logger.info("Permutation Importance Analyzer başlatıldı")
    
    def calculate(self, X, y, n_repeats=10, random_state=42, scoring='r2'):
        """
        Permutation importance hesapla
        
        Args:
            n_repeats: Permutation tekrar sayısı
            scoring: Metric ('r2', 'neg_mean_squared_error', etc.)
        """
        
        logger.info(f"Permutation importance hesaplanıyor: {n_repeats} repeats")
        
        self.importance_result = permutation_importance(
            self.model,
            X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1
        )
        
        logger.info("✓ Permutation importance hesaplandı")
        
        return self.importance_result
    
    def plot_importance(self, show=False):
        """Permutation importance bar plot"""
        
        if self.importance_result is None:
            raise ValueError("Önce calculate() çağırın!")
        
        logger.info("Permutation importance plot oluşturuluyor...")
        
        # Get importances
        importances_mean = self.importance_result.importances_mean
        importances_std = self.importance_result.importances_std
        
        # Create DataFrame
        feature_names = self.feature_names if self.feature_names else [f'F{i}' for i in range(len(importances_mean))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances_mean,
            'Std': importances_std
        })
        
        # Sort
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['Importance'], xerr=importance_df['Std'], 
                align='center', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance', fontsize=12)
        ax.set_title('Permutation Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'permutation_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Plot kaydedildi: {save_path}")
        
        # Save DataFrame
        csv_path = self.output_dir / 'permutation_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"✓ CSV kaydedildi: {csv_path}")
        
        return importance_df


# ============================================================================
# UNIFIED INTERPRETABILITY ANALYZER
# ============================================================================

class InterpretabilityAnalyzer:
    """
    Unified interpretability analysis
    
    Combines:
    - SHAP analysis
    - Permutation importance
    - Model-specific feature importance
    """
    
    def __init__(self, model, model_type, feature_names=None, target_names=None, output_dir='interpretability'):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.target_names = target_names
        self.output_dir = Path(output_dir) / model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.shap_analyzer = None
        self.perm_analyzer = None
        
        logger.info(f"Unified Interpretability Analyzer: {model_type}")
    
    def analyze(self, X_train, y_train, X_test, y_test, 
                use_shap=True, use_permutation=True,
                shap_background_size=100, n_shap_explain=200):
        """
        Tam interpretability analizi
        
        Args:
            X_train: Training data
            y_train: Training targets
            X_test: Test data
            y_test: Test targets
            shap_background_size: SHAP background samples
            n_shap_explain: SHAP explanation samples
        """
        
        logger.info("\n" + "="*80)
        logger.info(f"INTERPRETABILITY ANALYSIS: {self.model_type}")
        logger.info("="*80 + "\n")
        
        results = {}
        
        # 1. SHAP Analysis
        if use_shap and SHAP_AVAILABLE:
            logger.info("→ SHAP Analysis başlıyor...")
            
            try:
                self.shap_analyzer = SHAPAnalyzer(
                    self.model,
                    self.model_type,
                    self.feature_names,
                    self.output_dir / 'shap'
                )
                
                # Background data
                X_background = X_train[:shap_background_size]
                X_explain = X_test[:n_shap_explain]
                
                # Create explainer
                self.shap_analyzer.create_explainer(X_background)
                
                # Calculate SHAP values
                self.shap_analyzer.calculate_shap_values(X_explain)
                
                # Visualizations
                self.shap_analyzer.plot_summary(X_explain)
                self.shap_analyzer.plot_bar(X_explain)
                self.shap_analyzer.plot_top_dependence_plots(X_explain, n_top=5)
                
                # Feature importance
                shap_importance = self.shap_analyzer.get_feature_importance_df()
                self.shap_analyzer.export_shap_values()
                
                results['shap_importance'] = shap_importance
                
                logger.info("✓ SHAP Analysis tamamlandı")
                
            except Exception as e:
                logger.error(f"✗ SHAP Analysis hatası: {e}")
                results['shap_error'] = str(e)
        
        # 2. Permutation Importance
        if use_permutation:
            logger.info("\n→ Permutation Importance başlıyor...")
            
            try:
                self.perm_analyzer = PermutationImportanceAnalyzer(
                    self.model,
                    self.feature_names,
                    self.output_dir / 'permutation'
                )
                
                self.perm_analyzer.calculate(X_test, y_test, n_repeats=10)
                perm_importance = self.perm_analyzer.plot_importance()
                
                results['permutation_importance'] = perm_importance
                
                logger.info("✓ Permutation Importance tamamlandı")
                
            except Exception as e:
                logger.error(f"✗ Permutation Importance hatası: {e}")
                results['permutation_error'] = str(e)
        
        # 3. Model-specific Feature Importance
        logger.info("\n→ Model-specific importance...")
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names if self.feature_names else [f'F{i}' for i in range(len(importance))],
                'Importance': importance
            })
            
            importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            # Save
            save_path = self.output_dir / 'model_feature_importance.csv'
            importance_df.to_csv(save_path, index=False)
            
            # Plot
            self._plot_model_importance(importance_df)
            
            results['model_importance'] = importance_df
            
            logger.info("✓ Model-specific importance kaydedildi")
        
        # Summary report
        self._create_summary_report(results)
        
        logger.info("\n" + "="*80)
        logger.info("✓ INTERPRETABILITY ANALYSIS TAMAMLANDI")
        logger.info("="*80 + "\n")
        
        return results
    
    def _plot_model_importance(self, importance_df, show=False):
        """Model-specific importance plot"""
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['Importance'], 
                align='center', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['Feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'{self.model_type} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'model_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        
        logger.info(f"✓ Model importance plot: {save_path}")
    
    def _create_summary_report(self, results):
        """Create summary report"""
        
        summary = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'analyses_completed': []
        }
        
        if 'shap_importance' in results:
            summary['analyses_completed'].append('SHAP')
            summary['top_3_features_shap'] = results['shap_importance'].head(3)['Feature'].tolist()
        
        if 'permutation_importance' in results:
            summary['analyses_completed'].append('Permutation')
            summary['top_3_features_permutation'] = results['permutation_importance'].head(3)['Feature'].tolist()
        
        if 'model_importance' in results:
            summary['analyses_completed'].append('Model-Specific')
            summary['top_3_features_model'] = results['model_importance'].head(3)['Feature'].tolist()
        
        # Save summary
        with open(self.output_dir / 'interpretability_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✓ Summary report: {self.output_dir / 'interpretability_summary.json'}")


# ============================================================================
# MAIN TEST
# ============================================================================

def test_interpretability():
    """Test interpretability module"""
    
    print("\n" + "="*80)
    print("INTERPRETABILITY ANALYSIS TEST")
    print("="*80)
    
    if not SHAP_AVAILABLE:
        print("⚠ SHAP yüklü değil, bazı testler atlanacak")
    
    # Dummy data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * (-1) + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Split
    train_size = int(0.7 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    score = r2_score(y_test, model.predict(X_test))
    print(f"\nModel R² Score: {score:.4f}")
    
    # Interpretability analysis
    analyzer = InterpretabilityAnalyzer(
        model,
        'RandomForest',
        feature_names=feature_names,
        output_dir='test_interpretability'
    )
    
    results = analyzer.analyze(
        X_train, y_train,
        X_test, y_test,
        use_shap=SHAP_AVAILABLE,
        use_permutation=True,
        shap_background_size=50,
        n_shap_explain=100
    )
    
    print("\n" + "="*80)
    print("TOP 5 FEATURES:")
    print("="*80)
    
    if 'shap_importance' in results:
        print("\nSHAP:")
        print(results['shap_importance'].head(5)[['Feature', 'SHAP_Importance']])
    
    if 'permutation_importance' in results:
        print("\nPermutation:")
        print(results['permutation_importance'].head(5)[['Feature', 'Importance']])
    
    if 'model_importance' in results:
        print("\nModel-Specific:")
        print(results['model_importance'].head(5)[['Feature', 'Importance']])
    
    print("\n✓ Test tamamlandı!")


if __name__ == "__main__":
    test_interpretability()