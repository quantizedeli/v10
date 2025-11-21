# 🗂️ GITHUB REPOSITORY ORGANIZATION GUIDE
## Nuclear Physics AI Project - Klasör Yapısı ve Dosya Organizasyonu

**Tarih:** 21 Kasım 2025  
**Versiyon:** 1.0.0

---

## 📁 ÖNERİLEN KLASÖR YAPISI

```
nuclear-physics-ai-project/
│
├── 📄 README.md                          # Ana proje açıklaması (detaylı)
├── 📄 LICENSE                            # MIT veya GPLv3
├── 📄 .gitignore                         # Git ignore kuralları
├── 📄 requirements.txt                   # Python bağımlılıkları (60+ paket)
├── 📄 config.json                        # Ana konfigürasyon dosyası
├── 📄 main.py                            # Ana çalıştırma scripti
├── 📄 setup.py                           # Package setup (pip install -e .)
│
├── 📁 pfaz_modules/                      # === TÜM PFAZ FAZLARI ===
│   │
│   ├── 📁 pfaz01_dataset_generation/     # PFAZ 1: Dataset Generation
│   │   ├── __init__.py
│   │   ├── data_loader.py               # Veri yükleme
│   │   ├── dataset_generator.py         # Dataset oluşturma
│   │   ├── data_quality_modules.py      # Kalite kontrol
│   │   ├── qm_filter_manager.py         # QM filtreleme
│   │   ├── control_group_generator.py   # Control group
│   │   ├── data_enricher.py             # Veri zenginleştirme
│   │   └── README.md                    # PFAZ 1 dokümantasyon
│   │
│   ├── 📁 pfaz02_ai_training/           # PFAZ 2: AI Model Training
│   │   ├── __init__.py
│   │   ├── model_trainer.py             # Ana eğitim modülü
│   │   ├── hyperparameter_tuner.py      # HP optimizasyonu
│   │   ├── model_validator.py           # Model validasyon
│   │   ├── parallel_ai_trainer.py       # Paralel eğitim
│   │   ├── advanced_models.py           # DNN, BNN, PINN
│   │   ├── overfitting_detector.py      # Overfitting tespiti
│   │   └── README.md
│   │
│   ├── 📁 pfaz03_anfis_training/        # PFAZ 3: ANFIS Training
│   │   ├── __init__.py
│   │   ├── matlab_anfis_trainer.py      # MATLAB entegrasyonu
│   │   ├── anfis_parallel_trainer_v2.py # Paralel ANFIS
│   │   ├── anfis_config_manager.py      # Config yönetimi
│   │   ├── anfis_adaptive_strategy.py   # Adaptif strateji
│   │   ├── anfis_dataset_selector.py    # Dataset seçimi
│   │   ├── anfis_visualizer.py          # ANFIS görselleştirme
│   │   ├── anfis_performance_analyzer.py # Performans analizi
│   │   ├── anfis_robustness_tester.py   # Robustluk testi
│   │   ├── anfis_all_nuclei_predictor.py # Tüm tahminler
│   │   ├── anfis_model_saver.py         # Model kaydetme
│   │   └── README.md
│   │
│   ├── 📁 pfaz04_unknown_predictions/   # PFAZ 4: Unknown Nuclei
│   │   ├── __init__.py
│   │   ├── unknown_nuclei_predictor.py  # Tahmin modülü
│   │   ├── unknown_nuclei_splitter.py   # Dataset ayırma
│   │   ├── all_nuclei_predictor.py      # Tüm çekirdek tahminleri
│   │   ├── generalization_analyzer.py   # Genelleme analizi
│   │   └── README.md
│   │
│   ├── 📁 pfaz05_cross_model/           # PFAZ 5: Cross-Model Analysis
│   │   ├── __init__.py
│   │   ├── cross_model_evaluator.py     # Ana değerlendirme
│   │   ├── faz5_complete_cross_model.py # Tam pipeline
│   │   ├── best_model_selector.py       # En iyi model seçimi
│   │   └── README.md
│   │
│   ├── 📁 pfaz06_final_reporting/       # PFAZ 6: Final Reporting
│   │   ├── __init__.py
│   │   ├── pfaz6_final_reporting.py     # Ana raporlama
│   │   ├── comprehensive_excel_reporter.py # Excel raporları
│   │   ├── excel_formatter.py           # Excel formatı
│   │   ├── excel_charts.py              # Grafik üretimi
│   │   ├── latex_generator.py           # LaTeX üretici
│   │   ├── advanced_analysis_reporting_manager.py
│   │   ├── reports_comprehensive_module.py
│   │   └── README.md
│   │
│   ├── 📁 pfaz07_ensemble/              # PFAZ 7: Ensemble Methods
│   │   ├── __init__.py
│   │   ├── ensemble_model_builder.py    # Ensemble oluşturucu
│   │   ├── stacking_meta_learner.py     # Stacking
│   │   ├── ensemble_evaluator.py        # Ensemble değerlendirme
│   │   ├── faz7_ensemble_pipeline.py    # Tam pipeline
│   │   ├── pfaz7_complete_ensemble_pipeline.py
│   │   └── README.md
│   │
│   ├── 📁 pfaz08_visualization/         # PFAZ 8: Visualization
│   │   ├── __init__.py
│   │   ├── visualization_system.py      # Ana görselleştirme
│   │   ├── visualization_advanced_modules.py
│   │   ├── visualization_master_system.py
│   │   ├── ai_visualizer.py             # AI görselleştirme
│   │   ├── interactive_html_visualizer.py
│   │   ├── model_comparison_dashboard.py
│   │   ├── shap_analysis.py             # SHAP analizi
│   │   ├── robustness_visualizations_complete.py
│   │   ├── anomaly_visualizations_complete.py
│   │   ├── master_report_visualizations_complete.py
│   │   ├── log_analytics_visualizations_complete.py
│   │   └── README.md
│   │
│   ├── 📁 pfaz09_aaa2_monte_carlo/      # PFAZ 9: AAA2 & Monte Carlo
│   │   ├── __init__.py
│   │   ├── aaa2_control_group_complete_v4.py
│   │   ├── aaa2_control_group_comprehensive.py
│   │   ├── aaa2_quality_checker.py
│   │   ├── monte_carlo_simulation_system.py
│   │   ├── advanced_analytics_comprehensive.py
│   │   └── README.md
│   │
│   ├── 📁 pfaz10_thesis_compilation/    # PFAZ 10: Thesis Compilation
│   │   ├── __init__.py
│   │   ├── pfaz10_complete_package.py   # Ana paket
│   │   ├── pfaz10_master_integration.py # Master orchestrator
│   │   ├── pfaz10_content_generator.py  # İçerik üretimi
│   │   ├── pfaz10_latex_integration.py  # LaTeX entegrasyonu
│   │   ├── pfaz10_visualization_qa.py   # Görsel & QA
│   │   ├── pfaz10_thesis_orchestrator.py
│   │   ├── pfaz10_chapter_generator.py
│   │   ├── pfaz10_discussion_conclusion.py
│   │   └── README.md
│   │
│   ├── 📁 pfaz11_production/            # PFAZ 11: Production Deployment
│   │   ├── __init__.py
│   │   ├── production_model_serving.py  # Model serving
│   │   ├── production_monitoring_system.py
│   │   ├── production_web_interface.py  # Web arayüzü
│   │   ├── production_cicd_pipeline.py  # CI/CD
│   │   ├── pfaz7_production_complete.py
│   │   └── README.md
│   │
│   ├── 📁 pfaz12_advanced_analytics/    # PFAZ 12: Advanced Analytics
│   │   ├── __init__.py
│   │   ├── advanced_analytics_comprehensive.py
│   │   ├── statistical_testing_suite.py
│   │   ├── bootstrap_confidence_intervals.py
│   │   ├── advanced_sensitivity_analysis.py
│   │   └── README.md
│   │
│   └── 📁 pfaz13_automl/                # PFAZ 13: AutoML
│       ├── __init__.py
│       ├── automl_anfis_optimizer.py
│       ├── automl_hyperparameter_optimizer.py
│       ├── automl_feature_engineer.py
│       ├── automl_visualizer.py
│       ├── automl_logging_reporting_system.py
│       └── README.md
│
├── 📁 core_modules/                     # === ÇEKIRDEK MODÜLLER ===
│   ├── __init__.py
│   ├── constants_v1_1_0.py              # Sabitler ve konfigürasyon
│   ├── progress_tracker.py              # İlerleme takibi
│   ├── anomaly_detector.py              # Anomali tespiti
│   ├── control_group_evaluator.py       # Kontrol grubu
│   └── README.md
│
├── 📁 physics_modules/                  # === FİZİK HESAPLAMALARI ===
│   ├── __init__.py
│   ├── semf_calculator.py               # SEMF hesaplamaları
│   ├── woods_saxon.py                   # Woods-Saxon potansiyeli
│   ├── nilsson_model.py                 # Nilsson modeli
│   ├── theoretical_calculations_manager.py
│   └── README.md
│
├── 📁 analysis_modules/                 # === ANALİZ MODÜLLERI ===
│   ├── __init__.py
│   ├── shap_analysis.py                 # SHAP explainability
│   ├── model_interpretability.py        # Model yorumlanabilirlik
│   ├── robustness_validation_manager.py # Robustluk validasyonu
│   ├── real_data_integration_manager.py # Gerçek veri entegrasyonu
│   └── README.md
│
├── 📁 visualization_modules/            # === GÖRSELLEŞTİRME ===
│   ├── __init__.py
│   ├── visualization_system.py
│   ├── visualization_integration.py
│   └── README.md
│
├── 📁 data/                             # === VERİ DOSYALARI ===
│   ├── 📄 aaa2.txt                      # Ana dataset (267 çekirdek)
│   ├── 📄 README_DATA.md                # Veri açıklaması
│   └── 📄 .gitignore                    # Büyük dosyalar için
│
├── 📁 outputs/                          # === ÇIKTILAR (FAZ BAZINDA) ===
│   ├── 📁 pfaz01/                       # Dataset çıktıları
│   │   ├── datasets/                    # Üretilen datasetler
│   │   ├── reports/                     # Excel raporları
│   │   └── logs/
│   ├── 📁 pfaz02/                       # AI model çıktıları
│   │   ├── models/                      # Trained models (.pkl, .h5)
│   │   ├── reports/
│   │   ├── checkpoints/                 # Training checkpoints
│   │   └── logs/
│   ├── 📁 pfaz03/                       # ANFIS çıktıları
│   │   ├── models/                      # ANFIS models (.mat, .pkl)
│   │   ├── reports/
│   │   └── logs/
│   ├── 📁 pfaz04/                       # Unknown prediction çıktıları
│   ├── 📁 pfaz05/                       # Cross-model analiz
│   ├── 📁 pfaz06/                       # Final reports
│   ├── 📁 pfaz07/                       # Ensemble models
│   ├── 📁 pfaz08/                       # Visualizations
│   │   ├── dataset/                     # Dataset plots
│   │   ├── training/                    # Training plots
│   │   ├── performance/                 # Performance plots
│   │   ├── feature_importance/          # SHAP plots
│   │   ├── cross_model/                 # Cross-model plots
│   │   ├── ensemble/                    # Ensemble plots
│   │   ├── unknown/                     # Unknown nuclei plots
│   │   └── robustness/                  # Robustness plots
│   ├── 📁 pfaz09/                       # AAA2 & Monte Carlo
│   ├── 📁 pfaz10/                       # Thesis files
│   │   └── thesis/
│   │       ├── thesis_main.tex
│   │       ├── thesis_main.pdf
│   │       ├── chapters/
│   │       ├── figures/
│   │       └── references.bib
│   ├── 📁 pfaz11/                       # Production deployment
│   ├── 📁 pfaz12/                       # Advanced analytics
│   └── 📁 pfaz13/                       # AutoML outputs
│
├── 📁 models/                           # === EĞİTİLMİŞ MODELLER ===
│   ├── 📁 ai_models/                    # AI models (RF, XGBoost, DNN, etc.)
│   │   ├── random_forest/
│   │   ├── xgboost/
│   │   ├── dnn/
│   │   ├── bnn/
│   │   └── pinn/
│   ├── 📁 anfis_models/                 # ANFIS models
│   │   ├── gridpartition_trimf/
│   │   ├── subclust_gaussmf/
│   │   └── ...
│   ├── 📁 ensemble_models/              # Ensemble & stacking
│   └── 📁 best_models/                  # En iyi performans gösteren modeller
│       └── README.md                    # Model seçim kriterleri
│
├── 📁 logs/                             # === LOG DOSYALARI ===
│   ├── 📄 main.log                      # Ana log
│   ├── 📄 error.log                     # Hata logları
│   ├── 📄 pfaz01.log
│   ├── 📄 pfaz02.log
│   ├── ...
│   ├── 📄 pfaz13.log
│   └── 📄 .gitignore                    # Log dosyalarını ignore et
│
├── 📁 docs/                             # === DOKÜMANTASYON ===
│   ├── 📄 INSTALLATION.md               # Kurulum talimatları
│   ├── 📄 USAGE.md                      # Kullanım kılavuzu
│   ├── 📄 METHODOLOGY.md                # Metodoloji açıklaması
│   ├── 📄 API.md                        # API dokümantasyonu
│   ├── 📄 CONTRIBUTING.md               # Katkıda bulunma rehberi
│   ├── 📄 CHANGELOG.md                  # Versiyon geçmişi
│   ├── 📄 FAQ.md                        # Sık sorulan sorular
│   └── 📁 images/                       # Dokümantasyon görselleri
│       ├── architecture.png
│       └── workflow.png
│
├── 📁 tests/                            # === TEST DOSYALARI ===
│   ├── __init__.py
│   ├── 📄 conftest.py                   # Pytest configuration
│   ├── 📄 test_dataset.py               # Dataset testleri
│   ├── 📄 test_models.py                # Model testleri
│   ├── 📄 test_physics.py               # Fizik hesaplama testleri
│   ├── 📄 test_visualization.py         # Görselleştirme testleri
│   ├── 📄 test_integration.py           # Entegrasyon testleri
│   └── 📁 fixtures/                     # Test fixtures
│
├── 📁 scripts/                          # === YARDIMCI SCRİPTLER ===
│   ├── 📄 setup.sh                      # Otomatik kurulum (Linux/Mac)
│   ├── 📄 setup.bat                     # Otomatik kurulum (Windows)
│   ├── 📄 run_all.sh                    # Tüm fazları çalıştır
│   ├── 📄 cleanup.sh                    # Temizlik scripti
│   ├── 📄 check_modules.py              # Modül varlık kontrolü
│   ├── 📄 generate_report.py            # Hızlı rapor üretimi
│   └── 📄 backup.sh                     # Yedekleme scripti
│
├── 📁 .github/                          # === GITHUB WORKFLOWS ===
│   ├── 📁 workflows/
│   │   ├── ci.yml                       # Continuous Integration
│   │   ├── tests.yml                    # Otomatik testler
│   │   └── deploy.yml                   # Deployment
│   ├── ISSUE_TEMPLATE.md                # Issue şablonu
│   └── PULL_REQUEST_TEMPLATE.md         # PR şablonu
│
├── 📁 docker/                           # === DOCKER FILES ===
│   ├── Dockerfile                       # Ana Docker image
│   ├── docker-compose.yml               # Multi-container setup
│   └── README_DOCKER.md
│
└── 📁 notebooks/                        # === JUPYTER NOTEBOOKS ===
    ├── 01_data_exploration.ipynb
    ├── 02_model_training_demo.ipynb
    ├── 03_visualization_demo.ipynb
    └── README.md
```

---

## 📋 DOSYA TAŞıMA PLANI

### Şu Anki Durumdan Hedef Yapıya

#### 1. PFAZ Modüllerini Taşıma

```python
# Mevcut: Tüm .py dosyaları root'ta
# Hedef: pfaz_modules/ altında klasörlenmiş

import shutil
from pathlib import Path

# PFAZ 1 dosyaları
pfaz1_files = [
    'data_loader.py',
    'dataset_generator.py',
    'data_quality_modules.py',
    'qm_filter_manager.py',
    'control_group_generator.py',
    'data_enricher.py'
]

Path('pfaz_modules/pfaz01_dataset_generation').mkdir(parents=True, exist_ok=True)
for file in pfaz1_files:
    if Path(file).exists():
        shutil.move(file, f'pfaz_modules/pfaz01_dataset_generation/{file}')

# PFAZ 2 dosyaları
pfaz2_files = [
    'model_trainer.py',
    'hyperparameter_tuner.py',
    'model_validator.py',
    'parallel_ai_trainer.py',
    'advanced_models.py',
    'overfitting_detector.py'
]

Path('pfaz_modules/pfaz02_ai_training').mkdir(parents=True, exist_ok=True)
for file in pfaz2_files:
    if Path(file).exists():
        shutil.move(file, f'pfaz_modules/pfaz02_ai_training/{file}')

# ... Tüm PFAZ fazları için tekrarla
```

#### 2. Core Modülleri Taşıma

```python
core_files = [
    'constants_v1_1_0.py',
    'progress_tracker.py',
    'anomaly_detector.py',
    'control_group_evaluator.py'
]

Path('core_modules').mkdir(exist_ok=True)
for file in core_files:
    if Path(file).exists():
        shutil.move(file, f'core_modules/{file}')
```

#### 3. Physics Modülleri Taşıma

```python
physics_files = [
    'semf_calculator.py',
    'woods_saxon.py',
    'nilsson_model.py',
    'theoretical_calculations_manager.py'
]

Path('physics_modules').mkdir(exist_ok=True)
for file in physics_files:
    if Path(file).exists():
        shutil.move(file, f'physics_modules/{file}')
```

#### 4. Çıktıları Organize Etme

```python
# Mevcut outputs/ içindeki dosyaları faz bazında organize et
import os

output_base = Path('outputs')

# PFAZ bazında klasörler oluştur
for i in range(1, 14):
    Path(f'outputs/pfaz{i:02d}').mkdir(parents=True, exist_ok=True)
    Path(f'outputs/pfaz{i:02d}/reports').mkdir(exist_ok=True)
    Path(f'outputs/pfaz{i:02d}/logs').mkdir(exist_ok=True)

# Excel raporlarını taşı
excel_files = list(Path('.').glob('*.xlsx'))
for excel in excel_files:
    if 'PFAZ1' in excel.name or 'Dataset' in excel.name:
        shutil.move(excel, 'outputs/pfaz01/reports/')
    elif 'PFAZ2' in excel.name or 'AI_Training' in excel.name:
        shutil.move(excel, 'outputs/pfaz02/reports/')
    # ... vs
```

---

## 🔧 OTOMATIK TAŞıMA SCRİPTİ

```python
#!/usr/bin/env python3
"""
Otomatik Dosya Organizasyon Scripti
Nuclear Physics AI Project için dosyaları organize eder
"""

import shutil
from pathlib import Path
import json

# Dosya mapping (mevcut -> hedef)
FILE_MAPPING = {
    # PFAZ 1
    'data_loader.py': 'pfaz_modules/pfaz01_dataset_generation/',
    'dataset_generator.py': 'pfaz_modules/pfaz01_dataset_generation/',
    'data_quality_modules.py': 'pfaz_modules/pfaz01_dataset_generation/',
    'qm_filter_manager.py': 'pfaz_modules/pfaz01_dataset_generation/',
    'control_group_generator.py': 'pfaz_modules/pfaz01_dataset_generation/',
    'data_enricher.py': 'pfaz_modules/pfaz01_dataset_generation/',
    
    # PFAZ 2
    'model_trainer.py': 'pfaz_modules/pfaz02_ai_training/',
    'hyperparameter_tuner.py': 'pfaz_modules/pfaz02_ai_training/',
    'model_validator.py': 'pfaz_modules/pfaz02_ai_training/',
    'parallel_ai_trainer.py': 'pfaz_modules/pfaz02_ai_training/',
    'advanced_models.py': 'pfaz_modules/pfaz02_ai_training/',
    'overfitting_detector.py': 'pfaz_modules/pfaz02_ai_training/',
    
    # PFAZ 3
    'matlab_anfis_trainer.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_parallel_trainer_v2.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_config_manager.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_adaptive_strategy.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_dataset_selector.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_visualizer.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_performance_analyzer.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_robustness_tester.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_all_nuclei_predictor.py': 'pfaz_modules/pfaz03_anfis_training/',
    'anfis_model_saver.py': 'pfaz_modules/pfaz03_anfis_training/',
    
    # PFAZ 4
    'unknown_nuclei_predictor.py': 'pfaz_modules/pfaz04_unknown_predictions/',
    'unknown_nuclei_splitter.py': 'pfaz_modules/pfaz04_unknown_predictions/',
    'all_nuclei_predictor.py': 'pfaz_modules/pfaz04_unknown_predictions/',
    'generalization_analyzer.py': 'pfaz_modules/pfaz04_unknown_predictions/',
    
    # PFAZ 5
    'cross_model_evaluator.py': 'pfaz_modules/pfaz05_cross_model/',
    'faz5_complete_cross_model.py': 'pfaz_modules/pfaz05_cross_model/',
    'best_model_selector.py': 'pfaz_modules/pfaz05_cross_model/',
    
    # PFAZ 6
    'pfaz6_final_reporting.py': 'pfaz_modules/pfaz06_final_reporting/',
    'comprehensive_excel_reporter.py': 'pfaz_modules/pfaz06_final_reporting/',
    'excel_formatter.py': 'pfaz_modules/pfaz06_final_reporting/',
    'excel_charts.py': 'pfaz_modules/pfaz06_final_reporting/',
    'latex_generator.py': 'pfaz_modules/pfaz06_final_reporting/',
    'advanced_analysis_reporting_manager.py': 'pfaz_modules/pfaz06_final_reporting/',
    'reports_comprehensive_module.py': 'pfaz_modules/pfaz06_final_reporting/',
    
    # PFAZ 7
    'ensemble_model_builder.py': 'pfaz_modules/pfaz07_ensemble/',
    'stacking_meta_learner.py': 'pfaz_modules/pfaz07_ensemble/',
    'ensemble_evaluator.py': 'pfaz_modules/pfaz07_ensemble/',
    'faz7_ensemble_pipeline.py': 'pfaz_modules/pfaz07_ensemble/',
    'pfaz7_complete_ensemble_pipeline.py': 'pfaz_modules/pfaz07_ensemble/',
    
    # PFAZ 8
    'visualization_system.py': 'pfaz_modules/pfaz08_visualization/',
    'visualization_advanced_modules.py': 'pfaz_modules/pfaz08_visualization/',
    'visualization_master_system.py': 'pfaz_modules/pfaz08_visualization/',
    'ai_visualizer.py': 'pfaz_modules/pfaz08_visualization/',
    'interactive_html_visualizer.py': 'pfaz_modules/pfaz08_visualization/',
    'model_comparison_dashboard.py': 'pfaz_modules/pfaz08_visualization/',
    'shap_analysis.py': 'pfaz_modules/pfaz08_visualization/',
    'robustness_visualizations_complete.py': 'pfaz_modules/pfaz08_visualization/',
    'anomaly_visualizations_complete.py': 'pfaz_modules/pfaz08_visualization/',
    'master_report_visualizations_complete.py': 'pfaz_modules/pfaz08_visualization/',
    'log_analytics_visualizations_complete.py': 'pfaz_modules/pfaz08_visualization/',
    
    # PFAZ 9
    'aaa2_control_group_complete_v4.py': 'pfaz_modules/pfaz09_aaa2_monte_carlo/',
    'aaa2_control_group_comprehensive.py': 'pfaz_modules/pfaz09_aaa2_monte_carlo/',
    'aaa2_quality_checker.py': 'pfaz_modules/pfaz09_aaa2_monte_carlo/',
    'monte_carlo_simulation_system.py': 'pfaz_modules/pfaz09_aaa2_monte_carlo/',
    'advanced_analytics_comprehensive.py': 'pfaz_modules/pfaz09_aaa2_monte_carlo/',
    
    # PFAZ 10
    'pfaz10_complete_package.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    'pfaz10_master_integration.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    'pfaz10_content_generator.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    'pfaz10_latex_integration.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    'pfaz10_visualization_qa.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    'pfaz10_thesis_orchestrator.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    'pfaz10_chapter_generator.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    'pfaz10_discussion_conclusion.py': 'pfaz_modules/pfaz10_thesis_compilation/',
    
    # PFAZ 11
    'production_model_serving.py': 'pfaz_modules/pfaz11_production/',
    'production_monitoring_system.py': 'pfaz_modules/pfaz11_production/',
    'production_web_interface.py': 'pfaz_modules/pfaz11_production/',
    'production_cicd_pipeline.py': 'pfaz_modules/pfaz11_production/',
    'pfaz7_production_complete.py': 'pfaz_modules/pfaz11_production/',
    
    # PFAZ 12
    'statistical_testing_suite.py': 'pfaz_modules/pfaz12_advanced_analytics/',
    'bootstrap_confidence_intervals.py': 'pfaz_modules/pfaz12_advanced_analytics/',
    'advanced_sensitivity_analysis.py': 'pfaz_modules/pfaz12_advanced_analytics/',
    
    # PFAZ 13
    'automl_anfis_optimizer.py': 'pfaz_modules/pfaz13_automl/',
    'automl_hyperparameter_optimizer.py': 'pfaz_modules/pfaz13_automl/',
    'automl_feature_engineer.py': 'pfaz_modules/pfaz13_automl/',
    'automl_visualizer.py': 'pfaz_modules/pfaz13_automl/',
    'automl_logging_reporting_system.py': 'pfaz_modules/pfaz13_automl/',
    
    # Core modules
    'constants_v1_1_0.py': 'core_modules/',
    'progress_tracker.py': 'core_modules/',
    'anomaly_detector.py': 'core_modules/',
    'control_group_evaluator.py': 'core_modules/',
    
    # Physics modules
    'semf_calculator.py': 'physics_modules/',
    'woods_saxon.py': 'physics_modules/',
    'nilsson_model.py': 'physics_modules/',
    'theoretical_calculations_manager.py': 'physics_modules/',
    
    # Analysis modules
    'model_interpretability.py': 'analysis_modules/',
    'robustness_validation_manager.py': 'analysis_modules/',
    'real_data_integration_manager.py': 'analysis_modules/',
    
    # Visualization modules (overlap removed from pfaz08)
    'visualization_integration.py': 'visualization_modules/',
}


def create_directory_structure():
    """Klasör yapısını oluştur"""
    print("📁 Klasör yapısı oluşturuluyor...")
    
    directories = [
        'pfaz_modules/pfaz01_dataset_generation',
        'pfaz_modules/pfaz02_ai_training',
        'pfaz_modules/pfaz03_anfis_training',
        'pfaz_modules/pfaz04_unknown_predictions',
        'pfaz_modules/pfaz05_cross_model',
        'pfaz_modules/pfaz06_final_reporting',
        'pfaz_modules/pfaz07_ensemble',
        'pfaz_modules/pfaz08_visualization',
        'pfaz_modules/pfaz09_aaa2_monte_carlo',
        'pfaz_modules/pfaz10_thesis_compilation',
        'pfaz_modules/pfaz11_production',
        'pfaz_modules/pfaz12_advanced_analytics',
        'pfaz_modules/pfaz13_automl',
        'core_modules',
        'physics_modules',
        'analysis_modules',
        'visualization_modules',
        'data',
        'outputs',
        'models/ai_models',
        'models/anfis_models',
        'models/ensemble_models',
        'models/best_models',
        'logs',
        'docs',
        'tests',
        'scripts',
        'docker',
        'notebooks',
        '.github/workflows'
    ]
    
    for dir in directories:
        Path(dir).mkdir(parents=True, exist_ok=True)
        # __init__.py ekle (Python packages için)
        if 'pfaz_modules' in dir or 'modules' in dir:
            (Path(dir) / '__init__.py').touch()
    
    print("✓ Klasör yapısı oluşturuldu")


def move_files():
    """Dosyaları hedef konumlara taşı"""
    print("\n📦 Dosyalar taşınıyor...")
    
    moved_count = 0
    missing_count = 0
    
    for source, target_dir in FILE_MAPPING.items():
        source_path = Path(source)
        
        if source_path.exists():
            target_path = Path(target_dir) / source
            
            try:
                shutil.move(str(source_path), str(target_path))
                print(f"  ✓ {source} -> {target_dir}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ HATA: {source} taşınamadı - {e}")
        else:
            print(f"  ⚠ EKSIK: {source} bulunamadı")
            missing_count += 1
    
    print(f"\n📊 Özet:")
    print(f"  Taşınan: {moved_count}")
    print(f"  Eksik: {missing_count}")


def create_readme_files():
    """Her modül için README.md oluştur"""
    print("\n📝 README dosyaları oluşturuluyor...")
    
    pfaz_descriptions = {
        'pfaz01_dataset_generation': 'Dataset Generation - 267 nuclei data processing and feature engineering',
        'pfaz02_ai_training': 'AI Model Training - Random Forest, XGBoost, DNN, BNN, PINN',
        'pfaz03_anfis_training': 'ANFIS Training - 8 configurations with fuzzy logic',
        'pfaz04_unknown_predictions': 'Unknown Nuclei Predictions - Extrapolation and uncertainty',
        'pfaz05_cross_model': 'Cross-Model Analysis - Model agreement and comparison',
        'pfaz06_final_reporting': 'Final Reporting - Excel and LaTeX report generation',
        'pfaz07_ensemble': 'Ensemble Methods - Stacking, voting, and meta-learning',
        'pfaz08_visualization': 'Visualization System - 80+ plots and dashboards',
        'pfaz09_aaa2_monte_carlo': 'AAA2 & Monte Carlo - Control group and uncertainty quantification',
        'pfaz10_thesis_compilation': 'Thesis Compilation - Automatic LaTeX thesis generation',
        'pfaz11_production': 'Production Deployment - Model serving and CI/CD',
        'pfaz12_advanced_analytics': 'Advanced Analytics - Statistical tests and sensitivity analysis',
        'pfaz13_automl': 'AutoML Integration - Automated hyperparameter optimization'
    }
    
    for pfaz_name, description in pfaz_descriptions.items():
        readme_path = Path(f'pfaz_modules/{pfaz_name}/README.md')
        content = f"# {pfaz_name.upper()}\n\n## Description\n\n{description}\n\n## Modules\n\n- See Python files in this directory\n"
        readme_path.write_text(content)
        print(f"  ✓ {pfaz_name}/README.md")


def create_gitignore():
    """Ana .gitignore dosyası oluştur"""
    print("\n🚫 .gitignore oluşturuluyor...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# Model files (too large for git)
*.h5
*.pkl
*.pth
*.pt
*.mat

# Outputs (generated files)
outputs/
models/
logs/
*.log

# Data files (if large)
data/*.csv
data/*.txt

# Temporary files
*.tmp
*.bak
*.swp
*~

# OS
.DS_Store
Thumbs.db

# LaTeX
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.synctex.gz
*.toc
*.lof
*.lot
"""
    
    Path('.gitignore').write_text(gitignore_content)
    print("  ✓ .gitignore oluşturuldu")


def generate_summary():
    """Organizasyon özeti oluştur"""
    print("\n" + "="*60)
    print("📊 ORGANIZASYON TAMAMLANDI!")
    print("="*60)
    
    print("\n✅ Oluşturulan Yapı:")
    print("  • 13 PFAZ modül klasörü")
    print("  • Core, Physics, Analysis, Visualization modülleri")
    print("  • Outputs, Models, Logs, Docs, Tests klasörleri")
    print("  • README dosyaları")
    print("  • .gitignore")
    
    print("\n📋 Sonraki Adımlar:")
    print("  1. main.py'i güncelle (import pathlerini düzelt)")
    print("  2. Git repository başlat: git init")
    print("  3. İlk commit: git add . && git commit -m 'Initial commit'")
    print("  4. GitHub'a push: git remote add origin <repo-url> && git push -u origin main")
    print("  5. VS Code'da aç ve test et")


if __name__ == "__main__":
    print("="*60)
    print("🚀 NUCLEAR PHYSICS AI PROJECT - DOSYA ORGANIZASYONU")
    print("="*60)
    
    create_directory_structure()
    move_files()
    create_readme_files()
    create_gitignore()
    generate_summary()
    
    print("\n✨ İşlem tamamlandı! ✨\n")
```

---

## 📝 .gitignore İÇERİĞİ

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.idea/
.vscode/
*.swp
*.swo
*~

# Model files (large files)
*.h5
*.pkl
*.pth
*.pt
*.mat
*.onnx

# Generated outputs
outputs/**/*.xlsx
outputs/**/*.png
outputs/**/*.html
outputs/**/*.pdf
models/

# Logs
logs/
*.log

# Large data files
data/*.csv
data/*.txt
!data/README_DATA.md

# Temporary files
*.tmp
*.bak
.DS_Store
Thumbs.db

# LaTeX
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.synctex.gz
*.toc
*.lof
*.lot

# Checkpoints
checkpoints/
*.ckpt

# Virtual environments
venv/
ENV/
env/

# Docker
docker/*.log
```

---

## 🚀 VS CODE'DA AÇMA ADIMARI

### 1. GitHub'dan Clone
```bash
# Terminal'de
cd ~/projects/
git clone https://github.com/username/nuclear-physics-ai-project.git
cd nuclear-physics-ai-project
```

### 2. VS Code'da Aç
```bash
code .
```

### 3. Virtual Environment Oluştur
```bash
# VS Code terminal'inde
python -m venv venv

# Aktifleştir (Linux/Mac)
source venv/bin/activate

# Aktifleştir (Windows)
venv\Scripts\activate
```

### 4. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 5. İlk Test
```bash
# Modül kontrolü
python -c "from pfaz_modules.pfaz01_dataset_generation import data_loader; print('OK')"

# Ana script
python main.py --check-deps
```

---

## ✅ SON KONTROL LİSTESİ

- [ ] Tüm klasörler oluşturuldu
- [ ] Dosyalar doğru yerlere taşındı
- [ ] `__init__.py` dosyaları eklendi
- [ ] README.md dosyaları oluşturuldu
- [ ] .gitignore yapılandırıldı
- [ ] requirements.txt güncel
- [ ] main.py import pathları düzeltildi
- [ ] Git repository başlatıldı
- [ ] İlk commit yapıldı
- [ ] GitHub'a push edildi
- [ ] VS Code'da başarıyla açıldı
- [ ] Virtual environment çalışıyor
- [ ] İlk test başarılı

---

**Hazırlayan:** Claude (Anthropic)  
**Tarih:** 21 Kasım 2025  
**Versiyon:** 1.0.0

🎉 **Başarılar!**
