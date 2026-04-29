# PFAZ 12 — Advanced Analytics

Model karşılaştırması için parametrik/parametrik-olmayan istatistiksel testler, duyarlılık analizi ve nükleer örüntü analizi yapar.

## Modüller ve Sınıflar

### 1. İstatistiksel Test Süiti (`statistical_testing_suite.py`)

**`StatisticalTestingSuite`**

```python
from pfaz_modules.pfaz12_advanced_analytics import StatisticalTestingSuite

suite = StatisticalTestingSuite(alpha=0.05)
results = suite.compare_models_comprehensive(scores_dict)
suite.export_to_excel(results, "outputs/advanced_analytics/pfaz12_statistical_tests.xlsx")
```

**Desteklenen Testler:**

| Test | Metod | Kullanım | Etki Büyüklüğü |
|------|-------|---------|----------------|
| Eşleştirilmiş t-testi | `paired_t_test()` | 2 model karşılaştırma | Cohen's d |
| Wilcoxon İşaret Sıra | `wilcoxon_test()` | Parametrik olmayan 2 model | Cliff's delta |
| Tek yönlü ANOVA | `one_way_anova()` | 3+ model karşılaştırma | Eta-squared |
| Friedman Testi | `friedman_test()` | Parametrik olmayan 3+ model | — |
| Tukey HSD | `tukey_hsd_posthoc()` | ANOVA sonrası pairwise | — |
| Pairwise Wilcoxon | `pairwise_wilcoxon()` | Bonferroni/Holm düzeltme | — |

**Etki Büyüklüğü Yorumları:**

| Ölçüt | Küçük | Orta | Büyük |
|-------|-------|------|-------|
| Cohen's d | <0.5 | <0.8 | ≥0.8 |
| Eta-squared | <0.06 | <0.14 | ≥0.14 |
| Cliff's delta | <0.33 | <0.474 | ≥0.474 |

---

### 2. Duyarlılık Analizi (`advanced_sensitivity_analysis.py`)

**`AdvancedSensitivityAnalysis`**

| Metod | Açıklama | Çıktı |
|-------|---------|-------|
| `sobol_analysis()` | Varyans tabanlı — S1, ST, S2 endeksleri | SALib formatı |
| `morris_analysis()` | Ekranlama — μ, μ*, σ | Etki sıralaması |
| `tornado_analysis()` | ±%10 tek-seferde değişim | Tornado diyagramı |
| `plot_sobol_indices()` | S1/ST bar chart | PNG |
| `plot_tornado_diagram()` | Tornado diyagramı | PNG |
| `export_to_excel()` | Tüm analizler multi-sheet Excel | .xlsx |

Problem tanımı:
```python
problem = {
    "num_vars": 3,
    "names": ["A", "Z", "SPIN"],
    "bounds": [[2, 209], [1, 83], [0, 7]],
}
```

---

### 3. Nükleer Örüntü Analizi (`nuclear_pattern_analyzer.py`)

**`NuclearPatternAnalyzer`**

```python
from pfaz_modules.pfaz12_advanced_analytics import NuclearPatternAnalyzer

analyzer = NuclearPatternAnalyzer(
    data_path="data/aaa2.txt",
    output_dir="outputs/advanced_analytics",
    jump_sigma=2.0,      # sıçrama eşiği: 2σ
    min_chain_len=3,     # minimum zincir uzunluğu
)
analyzer.run_all()
```

**Analizler (her hedef için: MM, QM, Beta_2):**

| Analiz | Metod | Açıklama |
|--------|-------|---------|
| Küme analizi | `_mean_cluster_analysis()` | ±1σ, ±2σ bantlarına göre gruplama |
| İzotop zinciri | `_chain_analysis()` | Z sabit, N değişir; sıçrama tespiti |
| İzotone zinciri | `_chain_analysis()` | N sabit, Z değişir |
| İzobar zinciri | `_chain_analysis()` | A sabit, Z değişir |
| Magic sayı analizi | `_magic_number_analysis()` | KS + Mann-Whitney testi |
| Sıçrama özellik analizi | `_jump_feature_analysis()` | T-testi (sıçrama vs normal) |

Sıçrama kriteri: `|ΔTarget| > jump_sigma × σ_zincir`

**Excel Sayfaları (Turkish başlıklar):**
- Genel_Özet, Target_Küme, Target_İzotop_Sıçrama, Target_İzotone_Sıçrama, Target_İzobar_Sıçrama, Target_İzotop_ZincirStat, Target_Magic_Analiz, Target_Sıçrama_Özellik, Target_Sıçrama_Çekirdek

---

### 4. Bootstrap Güven Aralıkları (`bootstrap_confidence_intervals.py`)

**`BootstrapConfidenceIntervals`** — metrik güven aralıkları (R², RMSE, MAE) için n=5000 bootstrap örneklem.

---

### 5. Bayesian Model Karşılaştırması (`bayesian_model_comparison.py`)

**`BayesianModelComparison`** — ROPE (Region of Practical Equivalence) testi ile Bayesian model karşılaştırması.

## I/O Yapısı

**Giriş:** Tüm model metrikleri, tahmin dizileri, aaa2.txt

**Çıkış:**
```
outputs/advanced_analytics/
├── pfaz12_statistical_tests_{timestamp}.xlsx
├── sensitivity_analysis/
│   ├── sobol_indices.xlsx
│   ├── tornado_diagram.png
│   └── sobol_bar_chart.png
└── nuclear_patterns/
    ├── nuclear_patterns_{timestamp}.xlsx
    └── plots/
        ├── isotope_chains.png
        ├── magic_number_violin.png
        └── ...
```
