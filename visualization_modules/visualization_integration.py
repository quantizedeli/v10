"""
VISUALIZATION INTEGRATION MODULE
==================================

Tüm görselleştirme modüllerinin entegrasyon ve kullanımı

Bu modül:
1. Tüm visualizer'ları bir araya getirir
2. Veri yükleme ve hazırlama
3. Hepsi bir arada çalışması
4. Otomatik HTML raporları
5. Tez hazırlık çıktıları

Author: Nuclear Physics AI Project
Date: October 2025
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import all visualization modules
import sys
sys.path.insert(0, '/mnt/user-data/outputs')

try:
    from visualization_master_system import (
        MasterVisualizationSystem,
        RobustnessVisualizer,
        SHAPVisualizer,
        AnomalyKernelVisualizer,
        MasterReportVisualizer,
        PredictionVisualizer,
        ModelComparisonVisualizer,
        TrainingMetricsVisualizer,
        OptimizationMetricsVisualizer,
        FeatureImportanceVisualizer
    )
    from visualization_advanced_modules import (
        DataCatalogVisualizer,
        ReportsVisualizer,
        LogAnalyticsVisualizer,
        Advanced3DVisualizer,
        EnsembleVisualizationExtended,
        ProductionReadinessDashboard
    )
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import visualization modules: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationIntegrationManager:
    """Tüm görselleştirme sistemlerini yöneten entegrasyon sınıfı"""
    
    def __init__(self, 
                 output_base_dir: str = 'output/thesis_visualizations',
                 config_file: Optional[str] = None):
        """
        Args:
            output_base_dir: Temel çıktı dizini
            config_file: Konfigürasyon dosyası yolu
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config(config_file) if config_file else self._default_config()
        
        # Initialize all visualizers
        self._initialize_visualizers()
        
        # Project data storage
        self.project_data = {}
        self.generated_visualizations = []
        self.report_metadata = {
            'timestamp': datetime.now().isoformat(),
            'visualizations': [],
            'statistics': {}
        }
        
        logger.info("[OK] Visualization Integration Manager initialized")
    
    def _default_config(self) -> Dict:
        """Varsayılan konfigürasyon"""
        return {
            'enable_plotly': True,
            'enable_3d': True,
            'dpi': 300,
            'save_formats': ['png', 'pdf'],
            'generate_html_report': True,
            'generate_summary_statistics': True,
            'verbose': True,
            'modules': {
                'robustness': True,
                'shap': True,
                'anomaly': True,
                'master_report': True,
                'predictions': True,
                'model_comparison': True,
                'training_metrics': True,
                'optimization': True,
                'features': True,
                'data_catalog': True,
                'reports': True,
                'log_analytics': True,
                'advanced_3d': True,
                'ensemble': True,
                'production_readiness': True
            }
        }
    
    def _load_config(self, config_file: str) -> Dict:
        """Konfigürasyon dosyasından yükle"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
            return self._default_config()
    
    def _initialize_visualizers(self):
        """Tüm visualizer'ları başlat"""
        base_dir = self.output_base_dir / 'visualizations'
        
        self.visualizers = {}
        
        if self.config['modules']['robustness']:
            self.visualizers['robustness'] = RobustnessVisualizer(base_dir / 'robustness')
        
        if self.config['modules']['shap']:
            self.visualizers['shap'] = SHAPVisualizer(base_dir / 'shap')
        
        if self.config['modules']['anomaly']:
            self.visualizers['anomaly'] = AnomalyKernelVisualizer(base_dir / 'anomaly')
        
        if self.config['modules']['master_report']:
            self.visualizers['master_report'] = MasterReportVisualizer(base_dir / 'master_report')
        
        if self.config['modules']['predictions']:
            self.visualizers['predictions'] = PredictionVisualizer(base_dir / 'predictions')
        
        if self.config['modules']['model_comparison']:
            self.visualizers['model_comparison'] = ModelComparisonVisualizer(base_dir / 'model_comparison')
        
        if self.config['modules']['training_metrics']:
            self.visualizers['training_metrics'] = TrainingMetricsVisualizer(base_dir / 'training_metrics')
        
        if self.config['modules']['optimization']:
            self.visualizers['optimization'] = OptimizationMetricsVisualizer(base_dir / 'optimization')
        
        if self.config['modules']['features']:
            self.visualizers['features'] = FeatureImportanceVisualizer(base_dir / 'features')
        
        if self.config['modules']['data_catalog']:
            self.visualizers['data_catalog'] = DataCatalogVisualizer(base_dir / 'data_catalog')
        
        if self.config['modules']['reports']:
            self.visualizers['reports'] = ReportsVisualizer(base_dir / 'reports')
        
        if self.config['modules']['log_analytics']:
            self.visualizers['log_analytics'] = LogAnalyticsVisualizer(base_dir / 'log_analytics')
        
        if self.config['modules']['advanced_3d']:
            self.visualizers['advanced_3d'] = Advanced3DVisualizer(base_dir / 'advanced_3d')
        
        if self.config['modules']['ensemble']:
            self.visualizers['ensemble'] = EnsembleVisualizationExtended(base_dir / 'ensemble')
        
        if self.config['modules']['production_readiness']:
            self.visualizers['production_readiness'] = ProductionReadinessDashboard(base_dir / 'production_readiness')
        
        logger.info(f"[OK] Initialized {len(self.visualizers)} visualizer modules")
    
    def add_robustness_data(self, 
                           predictions: pd.DataFrame,
                           perturbation_results: Dict,
                           noisy_predictions: Dict,
                           cv_results: Dict):
        """Sağlamlık test verilerini ekle"""
        self.project_data['robustness'] = {
            'predictions': predictions,
            'perturbation_results': perturbation_results,
            'noisy_predictions': noisy_predictions,
            'cv_results': cv_results
        }
        logger.info("[OK] Added robustness data")
    
    def add_shap_data(self,
                     feature_names: List[str],
                     shap_values: np.ndarray,
                     X: np.ndarray):
        """SHAP açıklayıcılık verilerini ekle"""
        self.project_data['shap'] = {
            'features': feature_names,
            'values': shap_values,
            'X': X
        }
        logger.info("[OK] Added SHAP data")
    
    def add_anomaly_data(self,
                        normal_data: pd.DataFrame,
                        anomaly_data: pd.DataFrame,
                        feature_cols: List[str],
                        anomaly_mask: np.ndarray = None):
        """Anomali analizi verilerini ekle"""
        self.project_data['anomaly'] = {
            'normal': normal_data,
            'anomalous': anomaly_data,
            'features': feature_cols,
            'mask': anomaly_mask
        }
        logger.info("[OK] Added anomaly data")
    
    def add_prediction_data(self,
                           experimental: np.ndarray,
                           predictions_dict: Dict[str, np.ndarray],
                           target_name: str,
                           X: np.ndarray = None):
        """Tahmin verilerini ekle"""
        self.project_data['predictions'] = {
            'experimental': experimental,
            'models': predictions_dict,
            'target_name': target_name,
            'X': X
        }
        logger.info("[OK] Added prediction data")
    
    def add_model_metrics(self, models_metrics: Dict[str, Dict]):
        """Model metriklerini ekle"""
        self.project_data['model_metrics'] = models_metrics
        logger.info("[OK] Added model metrics")
    
    def add_training_history(self, history_dict: Dict):
        """Eğitim geçmişini ekle"""
        self.project_data['training_history'] = history_dict
        logger.info("[OK] Added training history")
    
    def add_datasets_info(self, datasets_info: Dict, datasets_features: Dict):
        """Veri setleri bilgisini ekle"""
        self.project_data['datasets_info'] = datasets_info
        self.project_data['datasets_features'] = datasets_features
        logger.info("[OK] Added datasets info")
    
    def add_log_data(self, log_df: pd.DataFrame):
        """Log verilerini ekle"""
        self.project_data['logs'] = log_df
        logger.info("[OK] Added log data")
    
    def add_production_readiness_data(self,
                                     model_metrics: Dict,
                                     robustness_scores: Dict,
                                     resource_requirements: Dict):
        """Production hazırlık verilerini ekle"""
        self.project_data['production'] = {
            'model_metrics': model_metrics,
            'robustness_scores': robustness_scores,
            'resource_requirements': resource_requirements
        }
        logger.info("[OK] Added production readiness data")
    
    def generate_all_visualizations(self):
        """Tüm görselleştirmeleri oluştur"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING ALL VISUALIZATIONS")
        logger.info("="*80)
        
        # 1. Robustness
        if 'robustness' in self.project_data and 'robustness' in self.visualizers:
            logger.info("\n-> Robustness Visualizations")
            try:
                data = self.project_data['robustness']
                self.visualizers['robustness'].plot_perturbation_sensitivity(
                    data['predictions'], data['perturbation_results']
                )
                self.visualizers['robustness'].plot_noise_robustness(
                    data['predictions'].values, data['noisy_predictions']
                )
                self.visualizers['robustness'].plot_cross_validation_stability(data['cv_results'])
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 2. SHAP
        if 'shap' in self.project_data and 'shap' in self.visualizers:
            logger.info("\n-> SHAP Visualizations")
            try:
                data = self.project_data['shap']
                self.visualizers['shap'].plot_shap_summary(
                    data['features'], data['values'], data['X']
                )
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 3. Anomaly
        if 'anomaly' in self.project_data and 'anomaly' in self.visualizers:
            logger.info("\n-> Anomaly Analysis")
            try:
                data = self.project_data['anomaly']
                self.visualizers['anomaly'].plot_anomaly_characteristics(
                    data['normal'], data['anomalous'], data['features']
                )
                if data.get('mask') is not None:
                    self.visualizers['anomaly'].plot_anomaly_clustering(
                        pd.concat([data['normal'], data['anomalous']]),
                        data['mask'], data['features']
                    )
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 4. Predictions
        if 'predictions' in self.project_data and 'predictions' in self.visualizers:
            logger.info("\n-> Prediction Visualizations")
            try:
                data = self.project_data['predictions']
                self.visualizers['predictions'].plot_prediction_comparison(
                    data['experimental'], data['models'], data['target_name']
                )
                self.visualizers['predictions'].plot_residual_analysis(
                    data['experimental'], data['models'], data['target_name']
                )
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 5. Model Comparison
        if 'model_metrics' in self.project_data and 'model_comparison' in self.visualizers:
            logger.info("\n-> Model Comparison")
            try:
                self.visualizers['model_comparison'].plot_model_ranking(self.project_data['model_metrics'])
                self.visualizers['model_comparison'].plot_parallel_coordinates(self.project_data['model_metrics'])
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 6. Training Metrics
        if 'training_history' in self.project_data and 'training_metrics' in self.visualizers:
            logger.info("\n-> Training Metrics")
            try:
                self.visualizers['training_metrics'].plot_training_curves(self.project_data['training_history'])
                self.visualizers['training_metrics'].plot_training_convergence(self.project_data['training_history'])
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 7. Data Catalog
        if 'datasets_info' in self.project_data and 'data_catalog' in self.visualizers:
            logger.info("\n-> Data Catalog")
            try:
                self.visualizers['data_catalog'].plot_dataset_overview(self.project_data['datasets_info'])
                if 'datasets_features' in self.project_data:
                    self.visualizers['data_catalog'].plot_feature_statistics_catalog(
                        self.project_data['datasets_features']
                    )
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 8. Log Analytics
        if 'logs' in self.project_data and 'log_analytics' in self.visualizers:
            logger.info("\n-> Log Analytics")
            try:
                self.visualizers['log_analytics'].plot_log_statistics(self.project_data['logs'])
                error_logs = self.project_data['logs'][self.project_data['logs']['level'] == 'ERROR']
                if len(error_logs) > 0:
                    self.visualizers['log_analytics'].plot_error_analysis(error_logs)
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        # 9. Production Readiness
        if 'production' in self.project_data and 'production_readiness' in self.visualizers:
            logger.info("\n-> Production Readiness")
            try:
                data = self.project_data['production']
                self.visualizers['production_readiness'].create_production_readiness_report(
                    data['model_metrics'], data['robustness_scores'], data['resource_requirements']
                )
            except Exception as e:
                logger.error(f"  [FAIL] Error: {e}")
        
        logger.info("\n" + "="*80)
        logger.info("[OK] VISUALIZATION GENERATION COMPLETED")
        logger.info("="*80)
    
    def generate_html_report(self, report_title: str = "Thesis Visualization Report"):
        """Kapsamlı HTML raporu oluştur"""
        if not self.config['generate_html_report']:
            return
        
        logger.info("\nGenerating HTML report...")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .subsection {{ background-color: #ecf0f1; margin: 10px 0; padding: 15px; border-left: 4px solid #3498db; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background-color: #ecf0f1; border-radius: 3px; }}
        .footer {{ text-align: center; color: #7f8c8d; margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #ecf0f1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        # Add sections
        for module_name in sorted(self.visualizers.keys()):
            module_dir = self.output_base_dir / 'visualizations' / module_name
            if module_dir.exists():
                images = list(module_dir.glob('*.png'))
                if images:
                    html_content += f"""
    <div class="section">
        <h2>{module_name.replace('_', ' ').title()}</h2>
        <p>Total visualizations: {len(images)}</p>
        <div class="subsection">
"""
                    for img in sorted(images)[:5]:  # Show first 5 images
                        rel_path = img.relative_to(self.output_base_dir)
                        html_content += f'            <img src="{rel_path}" alt="{img.stem}">\n'
                    
                    if len(images) > 5:
                        html_content += f'            <p><em>... and {len(images) - 5} more visualizations</em></p>\n'
                    
                    html_content += """
        </div>
    </div>
"""
        
        html_content += f"""
    <div class="footer">
        <p>Total visualizations generated: {len(list(self.output_base_dir.rglob('*.png')))}</p>
        <p>Report generated by Visualization Integration System</p>
    </div>
</body>
</html>
"""
        
        # Save HTML
        html_file = self.output_base_dir / 'visualization_report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"[OK] HTML report saved: {html_file}")
    
    def generate_summary_statistics(self):
        """Özet istatistikler oluştur"""
        if not self.config['generate_summary_statistics']:
            return
        
        logger.info("\nGenerating summary statistics...")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'modules_enabled': self.config['modules'],
            'visualization_count': len(list(self.output_base_dir.rglob('*.png'))),
            'data_summary': {}
        }
        
        # Add data summaries
        if 'model_metrics' in self.project_data:
            models = list(self.project_data['model_metrics'].keys())
            summary['data_summary']['models'] = {
                'count': len(models),
                'names': models
            }
        
        if 'predictions' in self.project_data:
            summary['data_summary']['predictions'] = {
                'target': self.project_data['predictions'].get('target_name', 'unknown'),
                'n_predictions': len(self.project_data['predictions']['experimental'])
            }
        
        # Save summary
        summary_file = self.output_base_dir / 'summary_statistics.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"[OK] Summary statistics saved: {summary_file}")
    
    def save_config(self):
        """Konfigürasyonu kaydet"""
        config_file = self.output_base_dir / 'visualization_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"[OK] Config saved: {config_file}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Kullanım örneği"""
    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION INTEGRATION - USAGE EXAMPLE")
    logger.info("="*80)
    
    # Initialize manager
    manager = VisualizationIntegrationManager('output/thesis_viz_example')
    
    logger.info("\n1. Create sample data...")
    
    # Sample prediction data
    n_samples = 100
    experimental = np.random.randn(n_samples) * 0.1 + 0.5
    predictions = {
        'Model_1': experimental + np.random.randn(n_samples) * 0.05,
        'Model_2': experimental + np.random.randn(n_samples) * 0.03,
    }
    
    # Add data
    manager.add_prediction_data(experimental, predictions, 'Target_MM')
    
    # Sample model metrics
    models_metrics = {
        'Model_1': {'r2': 0.92, 'mae': 0.04, 'rmse': 0.05},
        'Model_2': {'r2': 0.95, 'mae': 0.02, 'rmse': 0.03}
    }
    manager.add_model_metrics(models_metrics)
    
    logger.info("\n2. Generate visualizations...")
    manager.generate_all_visualizations()
    
    logger.info("\n3. Generate reports...")
    manager.generate_html_report("Nuclear Physics AI - Thesis Visualizations")
    manager.generate_summary_statistics()
    manager.save_config()
    
    logger.info(f"\n[OK] All outputs saved to: {manager.output_base_dir}")


if __name__ == "__main__":
    example_usage()
