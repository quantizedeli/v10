# PFAZ 13 — AutoML Retraining Stratejisi

**Modül:** `pfaz_modules/pfaz13_automl/automl_retraining_loop.py`  
**Ana Sınıf:** `AutoMLRetrainingLoop`

---

## Genel Akış

```
PFAZ2 metrics_*.json
    → Kategorik seçim (Poor / Medium / Good)
        → AI optimizasyonu (Optuna TPE)
        → ANFIS optimizasyonu (AutoMLANFISOptimizer)
        → Feature Engineering (en iyi iyileşen için)
    → automl_improvement_report.xlsx + automl_retraining_log.json
    → AutoMLVisualizer grafikleri
```

---

## Adım 1 — Aday Seçimi (Kategorik)

PFAZ2 çıktı dizinindeki tüm `metrics_*.json` dosyaları taranır.

| Kategori | R2 Aralığı | Seçilen Aday |
|----------|------------|--------------|
| Poor     | R2 < 0.70  | En düşük R2'li 25 dataset |
| Medium   | 0.70 ≤ R2 < 0.90 | En düşük R2'li 25 dataset |
| Good     | 0.90 ≤ R2 < 0.95 | En düşük R2'li 25 dataset |
| Excellent | R2 ≥ 0.95 | **Seçilmez** — optimize etmeye gerek yok |

Her kategoriden **en düşük R2'li 25** aday alınır (toplam max 75 kombinasyon).  
`n_per_category` parametresiyle değiştirilebilir (varsayılan: 25).

---

## Adım 2 — AI Optimizasyonu (Optuna TPE)

Her `(target, dataset, model_type)` kombinasyonu için:

1. Train/val/test split yüklenir
2. `AutoMLOptimizer` — Optuna TPE sampler ile `n_trials=30` hiperparametre araması
3. Birden fazla model tipi denenebilir (`model_types` parametresi, varsayılan: `['rf', 'xgb', 'lgb']`)
4. En iyi val_R2 > önceki val_R2 ise iyileşme kaydedilir
5. En iyi parametrelerle test seti tahmini üretilir

**Pruner:** Optuna MedianPruner (kötü trial'lar erken kesilir).

---

## Adım 3 — ANFIS Optimizasyonu

Aynı kategorik adaylar için paralel olarak çalışır:

- `AutoMLANFISOptimizer` — Optuna ile `n_trials=min(20, n_trials)` araması, `timeout=300s`
- MF tipi ve n_mfs hiperparametre olarak aranır
- Test seti üzerinde TakagiSugenoANFIS ile tahmin üretilir
- Her kategori için ayrı çıktı dizini: `anfis_optimization/{poor|medium|good}/`

---

## Adım 4 — Feature Engineering (Bonus)

Tüm AI optimizasyonları bittikten sonra, **en çok iyileşen tek dataset** için:

- `AutoMLFeatureEngineer` çalışır
- Polinom kombinasyonlar (derece 2) + dönüşümler + etkileşim özellikleri üretilir
- Seçim yöntemleri: RFE + Lasso + Mutual Info
- Hedef: `min(40, max(10, n_features × 3))` özellik
- Çıktı: `feature_engineering/` dizini

---

## Çıktılar

### `automl_improvement_report.xlsx` — 6 Sayfa

| Sayfa | İçerik |
|-------|--------|
| AI_Improvements | Tüm AI kombinasyonları: önce/sonra val_R2, iyileşme delta |
| ANFIS_Improvements | ANFIS kombinasyonları: önce/sonra val_R2, best_config |
| Predictions | Test seti tahminleri — çekirdek bazlı: deneysel vs. tahmin |
| Category_Summary | Her kategori için iyileşen/iyileşmeyen sayısı |
| Best_Params | En iyi hiperparametre setleri |
| Feature_Engineering | Özellik mühendisliği sonuçları |

### `automl_retraining_log.json`

Tüm `_improvement_records` + `_anfis_records` + `_prediction_records` ham verisi.

### `automl_visualizations/`

`AutoMLVisualizer` tarafından üretilen grafikler (improvement_report.xlsx'ten).

---

## Önemli Parametreler

| Parametre | Varsayılan | Açıklama |
|-----------|------------|---------|
| `n_trials` | 30 | Her kombinasyon için Optuna trial sayısı |
| `n_per_category` | 25 | Her kategoriden max aday sayısı |
| `model_types` | `['rf', 'xgb', 'lgb']` | Denenen AI model tipleri |
| `POOR_MAX` | 0.70 | Poor kategorisi üst sınırı |
| `MEDIUM_MAX` | 0.90 | Medium kategorisi üst sınırı |
| `GOOD_MAX` | 0.95 | Good kategorisi üst sınırı |

---

## Notlar

- PFAZ2'nin `training_results_summary.xlsx` DEĞIL, `metrics_*.json` dosyaları taranır — PFAZ2 çalışmamış olsa bile kısmi sonuçlar işlenir.
- Iyileşme eşiği: `delta > 0.001` (gürültüyü ele almak için).
- ANFIS optimizasyonu `AutoMLANFISOptimizer` modülüne bağlı — import başarısız olursa sessizce atlanır.
- `aaa2_txt_path` verilirse deneysel veri karşılaştırması da Predictions sayfasına eklenir.
