# PFAZ 9 — AAA2 Control Group & Monte Carlo

Top-50 modeli tüm 267 çekirdeğe uygular; Monte Carlo ile belirsizlik ölçer; 15 sayfalık Excel ve 13 grafik üretir.

## Ana Sınıf

**`AAA2ControlGroupAnalyzerComplete`** (`aaa2_control_group_complete_v4.py`)

```python
from pfaz_modules.pfaz09_aaa2_monte_carlo import AAA2ControlGroupAnalyzerComplete

analyzer = AAA2ControlGroupAnalyzerComplete(
    pfaz01_output_path="outputs/generated_datasets",
    aaa2_txt_path="data/aaa2.txt",
    trained_models_dir="outputs/trained_models",
    output_dir="outputs/aaa2_pfaz9_complete_results",
)
analyzer.run_complete_pfaz9_pipeline(targets=["MM", "QM", "Beta_2"])
```

## Çalışma Akışı

```
PHASE 1: load_and_enrich_aaa2_data()
  -> aaa2.txt yükle
  -> TheoreticalFeaturesCalculator ile 14 teorik özellik hesapla
     - Woods-Saxon (V0=-51 MeV, r0=1.25 fm, a=0.65 fm)
     - Nilsson model (Beta_2, epsilon, frekans)
     - Kabuk modeli (magic_dist, magic_flag, double_magic)
  -> aaa2_enriched_with_theory.csv kaydet

PHASE 2: select_top50_models(target)
  -> trained_models/ ve anfis_models/ tara
  -> Her model için metrics JSON'dan test_r2 oku
  -> En iyi 50 modeli seç

PHASE 3: predict_with_top50(target)
  -> Her model kendi feature_names ile veri hazırlar
  -> 267 çekirdek için tahmin üret
  -> predictions_matrix: (50 model x 267 çekirdek)

PHASE 4: quantify_prediction_uncertainty(target)
  -> MonteCarloUncertaintyQuantifier çalıştır
  -> mean, std, 95% CI, entropy, CV hesapla
  -> high_uncertainty (std>0.3) ve low_uncertainty (std<0.05) çekirdekler

PHASE 5: generate_comprehensive_excel(target)
  -> 5 ana sayfa + ExcelPivotTableCreator ile 8 pivot tablo
```

## Excel Çıktısı: `AAA2_Complete_{TARGET}.xlsx`

| Sayfa | İçerik |
|-------|--------|
| Predictions | 267 çekirdek × model tahminleri + deneysel değer |
| Uncertainty | mean, std, CI_lower, CI_upper, entropy, CV |
| PerModel_Top25 | Top-25 model bazlı sonuçlar |
| Model_Ranking | R² sıralaması |
| Pivot_Model_Perf | Model performansı pivot |
| Pivot_Nucleus_Cat | Çekirdek kategorisi analizi |
| Pivot_Mass_Region | Kütle bölgesi performansı |
| Pivot_Magic | Magic sayı etkisi |
| Pivot_Shell_Closure | Kabuk kapanması analizi |
| Pivot_QM_Empty | QM boş çekirdekler |
| Pivot_Best_Model | Çekirdek başına en iyi model |
| Pivot_Worst | En kötü performanslı çekirdekler |

## Monte Carlo Sistemi (`monte_carlo_simulation_system.py`)

**`MonteCarloSimulationSystem`** — 5 MC yöntemi:

| Yöntem | Sınıf | Açıklama |
|--------|-------|---------|
| MC Dropout | MCDropoutSimulator | DNN için 100 ileri geçiş (training=True) |
| Bootstrap | BootstrapSimulator | Stratified resample, 100 örneklem |
| Noise Sensitivity | NoiseSimulator | 5 gürültü seviyesi (0.01–0.2σ) |
| Feature Dropout | FeatureDropoutSimulator | %10/%20/%30 özellik silinmesi |
| Ensemble Uncertainty | EnsembleUncertaintyAnalyzer | Modeller arası std ve korelasyon |

### MC Konfigürasyonu (varsayılan)
```python
DEFAULT_MC_CONFIG = {
    "mc_dropout": {"n_samples": 100},
    "bootstrap": {"n_samples": 100, "stratified": True},
    "noise_sensitivity": {"levels": [0.01, 0.05, 0.1, 0.15, 0.2], "n_samples": 100},
    "feature_dropout": {"rates": [0.1, 0.2, 0.3], "n_samples": 500},
    "thresholds": {"high_uncertainty": 0.3, "low_uncertainty": 0.05},
}
```

### MC Grafikleri (13 adet)
- Chart 11: 3D belirsizlik haritası (A × Z × σ)
- Chart 12: 3D model anlaşması
- Chart 13: 3D gürültü robustluk
- Chart 1: Belirsizlik histogramı
- Chart 2: Belirsizlik vs kütle numarası

## Teorik Özellik Hesaplamaları

**`TheoreticalFeaturesCalculator`** — 14 özellik:

| Model | Özellikler |
|-------|-----------|
| Woods-Saxon | Yarıçap, yüzey kalınlığı, Fermi enerjisi, potansiyel derinliği |
| Nilsson | Beta_2, epsilon, osilator frekansı, seviye yoğunluğu |
| Kabuk modeli | Z/N magic mesafesi, magic_flag, double_magic_flag, kabuk kapanma etkisi |

## I/O Yapısı

**Giriş:**
- `data/aaa2.txt` (267 çekirdek ham veri)
- `outputs/trained_models/` (AI PKL + metrics JSON)
- `outputs/anfis_models/` (ANFIS PKL + metrics JSON)
- `outputs/generated_datasets/*/metadata.json` (feature_names)

**Çıkış:**
```
outputs/aaa2_pfaz9_complete_results/
├── aaa2_enriched_with_theory.csv
├── AAA2_Complete_MM.xlsx       <- 15 sayfa
├── AAA2_Complete_QM.xlsx
├── AAA2_Complete_Beta_2.xlsx
├── data_quality/               <- AAA2QualityChecker
└── monte_carlo_analysis/
    ├── model_selection/
    │   └── top10_models_{TARGET}.json
    ├── visualizations/{TARGET}/
    │   └── chart_01.png ... chart_13.png
    ├── excel_reports/
    │   └── MC_Analysis_{TARGET}_{timestamp}.xlsx
    └── summaries/
        └── mc_summary_{TARGET}.json
```

## Modüller

| Dosya | Sınıf | Görev |
|-------|-------|-------|
| `aaa2_control_group_complete_v4.py` | AAA2ControlGroupAnalyzerComplete, TheoreticalFeaturesCalculator, MonteCarloUncertaintyQuantifier, ExcelPivotTableCreator | Ana pipeline + teorik özellikler + MC + Excel pivot |
| `monte_carlo_simulation_system.py` | MonteCarloSimulationSystem, MCDropoutSimulator, BootstrapSimulator, NoiseSimulator, FeatureDropoutSimulator, EnsembleUncertaintyAnalyzer | 5 MC yöntemi + 13 grafik |
| `aaa2_quality_checker.py` | AAA2QualityChecker | Veri kalitesi kontrolü |
| `advanced_analytics_comprehensive.py` | AdvancedAnalyticsComprehensive | Kapsamlı ek analitik |
